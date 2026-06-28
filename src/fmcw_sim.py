import os
import numpy as np
import scipy.linalg as la
from scipy.constants import c, k
from scipy.signal import butter, filtfilt
from scipy.signal.windows import chebwin
from scipy.optimize import linear_sum_assignment # GNN Data Association
from dataclasses import dataclass


##### RADAR AND ADC CONFIGURATION #####
@dataclass
class RadarConfig:
    """Hardware and 12-Bit ADC"""
    f0: float = 77e9            # carrier freq
    B: float = 250e6            # sweeping bandwidth
    T_chirp: float = 40e-6      # chirp duration
    fs: float = 40e6            # sampling freq
    N_chirps: int = 64          # No. chirps
    N_tx: int = 2               # No. transmitters
    N_rx: int = 4               # No. receivers

    P_tx_dBm: float = 13.0                  # transmitter power (dBm)
    G_ant_dB: float = 16.0                  # antenna gain (dB)
    tx_to_rx_leakage_dB: float = -35.0      # leakage RX-TX
    bumper_reflection_dB: float = -40.0
    phase_noise_floor: float = -110.0       # phase noise floor (dBc/Hz)
    adc_bits: int = 12                      # adc bits
    V_adc_max: float = 1.0                  # adc max voltage

    def __post_init__(self):
        self.slope = self.B / self.T_chirp              # sweep slope
        self.lambda_val = c / self.f0                   # wavelength
        self.Ts = 1 / self.fs                           # sampling time
        self.N_samples = int(self.T_chirp * self.fs)    # No. samples per chirp
        self.N_virtual = self.N_tx * self.N_rx          # No. virtual antennas
        self.d_rx = self.lambda_val / 2                 # physical RX spacing (λ/2)
        self.d_tx = self.N_rx * self.d_rx               # TX spacing = N_rx * d_rx (virtual aperture)
        self.P_tx = 10**((self.P_tx_dBm - 30) / 10)
        self.G_ant = 10**(self.G_ant_dB / 10)


##### ENVIRONMENT (TARGETS, CLUTTER, INTERFERENCE) #####
class Environment:
    def __init__(self, config: RadarConfig):
        self.cfg = config
        self.real_targets = [
            {"r": 8.0,  "v": 1.5,   "az": -25.0, "rcs": 1.0},       # pedestrian
            {"r": 20.0, "v": -7.5,  "az": -15.0, "rcs": 5.0},       # cyclist
            {"r": 35.0, "v": 8.0,   "az": 12.0,  "rcs": 30.0},      # truck
            {"r": 20.0, "v": 0.0,   "az": 0.0,   "rcs": 25.0},      # parked car 1
            {"r": 20.0, "v": 0.0,   "az": -10.0,   "rcs": 25.0},    # parked car 2
        ]
        self.clutter_points = [
            {"r": 15.0, "v": 0.0, "az": 30.0,  "rcs": 0.5},     # concrete trash can
            {"r": 28.0, "v": 0.0, "az": -30.0, "rcs": 2.0}      # metal utility pole
        ]

        # Ghost target: multipath reflection of the cyclist via the parked car 1
        self.barrier_offset = 8.0
        self.ghost_target = {
            "r": self.real_targets[1]["r"] + self.barrier_offset,
            "v": self.real_targets[1]["v"],
            "az": 0.0,      
            "rcs": 0.2
        }

        # Interference
        self.slope_interf_asynch = self.cfg.slope * -0.85
        self.f0_interf_asynch = self.cfg.f0 + 3e6
        self.amp_interf_asynch = 0.001
        self.tau_interf_synch = 1.4e-6
        self.amp_interf_synch = 0.000002

    def update(self, dt: float):
        """Moves all dynamic targets forward by dt seconds."""
        for target in self.real_targets:
            target["r"] += target["v"] * dt
        
        # update ghost 
        self.ghost_target["r"] = self.real_targets[1]["r"] + self.barrier_offset
        self.ghost_target["v"] = self.real_targets[1]["v"]


##### SIMULATOR MODULE #####
class RadarSimulator:
    def __init__(self, config: RadarConfig, env: Environment):
        self.cfg = config
        self.env = env
        self.t = np.linspace(0, self.cfg.T_chirp, self.cfg.N_samples, endpoint=False)

        # Filters
        self.b_hp, self.a_hp = butter(2, 150e3 / (self.cfg.fs / 2), btype='high')
        self.b_lp, self.a_lp = butter(4, 15e6 / (self.cfg.fs / 2), btype='low')

    def generate_mimo_cube(self):
        mimo_data_cube = np.zeros(
            (self.cfg.N_virtual, self.cfg.N_chirps, self.cfg.N_samples), dtype=complex
        )

        for chirp_idx in range(self.cfg.N_chirps):
            for tx_idx in range(self.cfg.N_tx):
                t_frame = (chirp_idx * self.cfg.N_tx + tx_idx) * self.cfg.T_chirp

                # Generate transmitter phase noise ONCE per transmitted chirp
                p_noise = np.cumsum(
                    np.random.normal(0, 10**(self.cfg.phase_noise_floor / 20), self.cfg.N_samples)
                )
                phase_tx = (
                    2 * np.pi * (self.cfg.f0 * self.t + 0.5 * self.cfg.slope * self.t**2)
                    + p_noise
                )
                tx_sig = np.sqrt(self.cfg.P_tx) * np.exp(1j * phase_tx)

                # RX channels receive the same transmitted pulse simultaneously
                for rx_idx in range(self.cfg.N_rx):
                    v_idx = tx_idx * self.cfg.N_rx + rx_idx
                    virtual_element_pos = (tx_idx * self.cfg.d_tx) + (rx_idx * self.cfg.d_rx)
                    rx_total = np.zeros(self.cfg.N_samples, dtype=complex)

                    # Direct paths (real targets + clutter)
                    for obj in (self.env.real_targets + self.env.clutter_points):
                        rx_total += self._simulate_target_return(
                            obj, t_frame, virtual_element_pos, p_noise
                        )

                    # Ghost target
                    rx_total += self._simulate_target_return(
                        self.env.ghost_target, t_frame, virtual_element_pos,
                        p_noise, rcs_penalty=0.1
                    )

                    # Leakage + bumper reflection
                    rx_total += self._simulate_artifacts(phase_tx, p_noise)

                    # Demodulate
                    beat_signal = tx_sig * np.conj(rx_total)

                    # Thermal noise (independent per RX channel)
                    beat_signal += (
                        np.sqrt(k * 290 * (self.cfg.fs / 2) * 10**(4 / 10))
                        * (np.random.normal(0, 1, self.cfg.N_samples)
                           + 1j * np.random.normal(0, 1, self.cfg.N_samples))
                    )

                    # Filtering & quantization
                    beat_signal = self._apply_filters(beat_signal)
                    mimo_data_cube[v_idx, chirp_idx, :] = self._quantize_adc(beat_signal)

        return mimo_data_cube

    def _simulate_target_return(self, obj, t_frame, v_pos, p_noise, rcs_penalty=1.0):
        r_t = obj["r"] + obj["v"] * t_frame
        tau = 2 * r_t / c

        spatial_phase = (
            -2 * np.pi * v_pos * np.sin(np.radians(obj["az"])) / self.cfg.lambda_val
        )

        power = (
            self.cfg.P_tx * (self.cfg.G_ant**2) * (self.cfg.lambda_val**2)
            * obj["rcs"] * rcs_penalty
            / ((4 * np.pi)**3 * (r_t**4))
        )

        tau_samples = int(round(tau * self.cfg.fs))
        p_noise_delayed = np.roll(p_noise, tau_samples)

        phase_rx = (
            2 * np.pi * (self.cfg.f0 * (self.t - tau) + 0.5 * self.cfg.slope * (self.t - tau)**2)
            + p_noise_delayed
        )

        return np.sqrt(power) * np.exp(1j * phase_rx + 1j * spatial_phase)

    def _simulate_artifacts(self, phase_tx, p_noise):
        rx_art = np.zeros(self.cfg.N_samples, dtype=complex)

        P_leak = 10**((self.cfg.P_tx_dBm + self.cfg.tx_to_rx_leakage_dB - 30) / 10)
        rx_art += np.sqrt(P_leak) * np.exp(1j * phase_tx)

        tau_bumper = 2 * 0.05 / c
        tau_bumper_samples = int(round(tau_bumper * self.cfg.fs))
        P_bumper = 10**((self.cfg.P_tx_dBm + self.cfg.bumper_reflection_dB - 30) / 10)
        p_noise_bumper = np.roll(p_noise, tau_bumper_samples)
        phase_bumper = (
            2 * np.pi * (self.cfg.f0 * (self.t - tau_bumper)
                         + 0.5 * self.cfg.slope * (self.t - tau_bumper)**2)
            + p_noise_bumper
        )
        rx_art += np.sqrt(P_bumper) * np.exp(1j * phase_bumper)

        return rx_art

    def _apply_filters(self, signal):
        sig = (
            filtfilt(self.b_hp, self.a_hp, signal.real)
            + 1j * filtfilt(self.b_hp, self.a_hp, signal.imag)
        )
        return (
            filtfilt(self.b_lp, self.a_lp, sig.real)
            + 1j * filtfilt(self.b_lp, self.a_lp, sig.imag)
        )

    def _quantize_adc(self, signal):
        scaled = signal * 5000
        clipped = np.clip(scaled, -self.cfg.V_adc_max, self.cfg.V_adc_max)
        step_size = (2 * self.cfg.V_adc_max) / (2**self.cfg.adc_bits)
        return np.round(clipped / step_size) * step_size
    

class SignalProcessor:
    def __init__(self, config: RadarConfig):
        self.cfg = config
        self.N_angle_fft = 64 

        freq_range = np.fft.fftfreq(self.cfg.N_samples, d=1 / self.cfg.fs)
        raw_ranges = (freq_range * c) / (2 * self.cfg.slope)

        self.pos_r_idx = raw_ranges >= 0
        self.ranges = raw_ranges[self.pos_r_idx]

        freq_doppler = np.fft.fftshift(
            np.fft.fftfreq(self.cfg.N_chirps, d=self.cfg.T_chirp * self.cfg.N_tx)
        )
        self.velocities = (freq_doppler * c) / (2 * self.cfg.f0)

        spatial_freq = np.fft.fftshift(np.fft.fftfreq(self.N_angle_fft, d=0.5))
        self.angles = np.degrees(np.arcsin(np.clip(spatial_freq, -0.999, 0.999)))

    def process(self, mimo_cube):
        mimo_cube_dynamic = mimo_cube.copy()
        dc_idx_local = self.cfg.N_chirps // 2 

        win_range_doppler = np.outer(
            chebwin(self.cfg.N_chirps, at=85),
            np.blackman(self.cfg.N_samples)
        )
        windowed_cube = mimo_cube_dynamic * win_range_doppler[np.newaxis, :, :]

        fft_range   = np.fft.fft(windowed_cube, axis=2)
        fft_doppler = np.fft.fftshift(np.fft.fft(fft_range, axis=1), axes=1)

        tdm_comp = np.ones_like(fft_doppler)
        for v_idx in range(self.cfg.N_virtual):
            tx_idx = v_idx // self.cfg.N_rx
            if tx_idx == 0:
                continue 
            phi = (4 * np.pi * self.cfg.f0 * self.velocities
                   * tx_idx * self.cfg.T_chirp / c)
            tdm_comp[v_idx, :, :] = np.exp(-1j * phi)[:, np.newaxis]
        
        fft_doppler_comp = fft_doppler * tdm_comp

        spatial_win = np.hamming(self.cfg.N_virtual)[:, np.newaxis, np.newaxis]
        fft_angle_full = np.fft.fftshift(
            np.fft.fft(fft_doppler_comp * spatial_win, n=self.N_angle_fft, axis=0),
            axes=0
        )

        fft_angle = fft_angle_full[:, :, self.pos_r_idx]

        rd_master_map = np.mean(
            np.abs(fft_doppler[:, :, self.pos_r_idx])**2, axis=0
        )
        
        # Clutter Suppression (Zero-Doppler Nulling)
        rd_master_map[dc_idx_local, :] = 1e-12
        
        rd_master_db  = 10 * np.log10(rd_master_map + 1e-12)
        rd_master_db -= np.max(rd_master_db)

        static_coherent   = np.sum(np.sum(mimo_cube, axis=1), axis=0, keepdims=True)
        static_range_fft  = np.fft.fft(static_coherent * np.blackman(self.cfg.N_samples), axis=1)
        static_range_power = np.mean(np.abs(static_range_fft[:, self.pos_r_idx])**2, axis=0)

        return fft_angle, rd_master_map, rd_master_db, static_range_power, dc_idx_local, fft_doppler_comp

    def execute_esprit_doa(self, spatial_vector, num_sources=2, sub_array_size=6):
        N_v = len(spatial_vector)
        M = N_v - sub_array_size + 1
        
        R_f = np.zeros((sub_array_size, sub_array_size), dtype=complex)
        for i in range(M):
            sub_vec = spatial_vector[i : i+sub_array_size].reshape(-1, 1)
            R_f += sub_vec @ sub_vec.conj().T
        R_f /= M
        
        J = np.fliplr(np.eye(sub_array_size))
        R_fb = 0.5 * (R_f + J @ R_f.conj() @ J)
        
        eigvals, eigvecs = la.eigh(R_fb)
        
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        Es = eigvecs[:, :num_sources]
        
        E1 = Es[:-1, :]
        E2 = Es[1:, :]
        
        Psi = la.pinv(E1) @ E2
        phi_eigvals = la.eigvals(Psi)
        
        phases = np.angle(phi_eigvals)
        angles_rad = np.arcsin(np.clip(phases / np.pi, -0.999, 0.999))
        angles_deg = np.degrees(angles_rad)
        
        return sorted(angles_deg)

    def execute_ca_cfar_1d(self, range_profile, guard=4, train=10, pfa=1e-4, r_min_idx=0):
        n = len(range_profile)
        hits = []
        n_train = train * 2
        alpha = n_train * (pfa ** (-1.0 / n_train) - 1)
        for r in range(r_min_idx + guard + train, n - guard - train):
            cells = np.concatenate([
                range_profile[r - guard - train : r - guard],
                range_profile[r + guard + 1    : r + guard + train + 1]
            ])
            if range_profile[r] > alpha * np.mean(cells):
                hits.append(r)
        return hits

    def execute_ca_cfar_2d(self, power_matrix, guard_r=2, guard_d=1,
                            train_r=6, train_d=3, target_pfa=1e-5):
        n_rows, n_cols = power_matrix.shape
        detections = np.zeros_like(power_matrix)
        n_train = (
            (2*train_r + 2*guard_r + 1) * (2*train_d + 2*guard_d + 1)
            - (2*guard_r + 1) * (2*guard_d + 1)
        )
        alpha = n_train * (target_pfa**(-1 / n_train) - 1)
        pad_d, pad_r = train_d + guard_d, train_r + guard_r
        
        # constant mode padding to prevent boundary false alarms
        padded = np.pad(power_matrix, ((pad_d, pad_d), (pad_r, pad_r)), mode='constant', constant_values=np.inf)
        
        for d in range(n_rows):
            for r in range(n_cols):
                cut_val   = padded[d + pad_d, r + pad_r]
                sub_win   = padded[d : d + 2*pad_d + 1, r : r + 2*pad_r + 1]
                guard_win = padded[
                    d + pad_d - guard_d : d + pad_d + guard_d + 1,
                    r + pad_r - guard_r : r + pad_r + guard_r + 1
                ]
                noise_est = (np.sum(sub_win) - np.sum(guard_win)) / n_train
                if cut_val > noise_est * alpha:
                    detections[d, r] = 1
        return np.where(detections == 1)

#### KALMAN FILTER BLOCK ####
class KalmanTracker:
    def __init__(self, dt=0.04):
        self.dt = dt
        self.F = np.array([
            [1, self.dt, 0], 
            [0, 1,       0],
            [0, 0,       1]
        ])
        self.H = np.eye(3)
        self.P = np.eye(3) * 1000
        
        self.Q = np.diag([0.1, 0.1, 0.5]) 
        self.R = np.diag([0.5, 0.5, 2.0]) 
        self.x = None 

    def predict(self):
        if self.x is not None:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            
    def mahalanobis_distance(self, z):
        if self.x is None:
            return np.inf
            
        z_vec = np.array([[z[0]], [z[1]], [z[2]]])
        y = z_vec - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        
        dist_sq = y.T @ la.inv(S) @ y
        return dist_sq[0, 0]
            
    def update(self, z):
        if self.x is None: 
            self.x = np.array([[z[0]], [z[1]], [z[2]]])
            return self.x.flatten()
        
        z_vec = np.array([[z[0]], [z[1]], [z[2]]])
        y = z_vec - (self.H @ self.x) 
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ la.inv(S) 
        
        self.x = self.x + (K @ y)
        
        # Joseph form for numerical stability
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.flatten()

#### ESPRIT DOA ####
class ESPRITSolver:
    @staticmethod
    def solve(spatial_vec, num_sources=1):
        N = len(spatial_vec)
        M = N // 2 
        X = np.array([spatial_vec[i:i+M] for i in range(N-M+1)]).T
        R = (X @ X.conj().T) / (N-M+1)
        eigvals, eigvecs = la.eigh(R)
        Es = eigvecs[:, -num_sources:]
        Phi = la.pinv(Es[:-1, :]) @ Es[1:, :]
        angles = np.degrees(np.arcsin(np.clip(np.angle(la.eigvals(Phi)) / np.pi, -0.999, 0.999)))
        return angles
    
def cluster_detections(d_hits, r_hits, rd_map, proc, r_thresh=2, d_thresh=2):
    hits = list(zip(d_hits, r_hits))
    if not hits: return []

    clusters = []
    processed = set()

    for i in range(len(hits)):
        if i in processed:
            continue

        group = [hits[i]]
        processed.add(i)
        queue = [i]

        while queue:
            ci = queue.pop(0)
            cd, cr = hits[ci]
            for j, (d2, r2) in enumerate(hits):
                if j not in processed and abs(cd - d2) <= d_thresh and abs(cr - r2) <= r_thresh:
                    group.append(hits[j])
                    processed.add(j)
                    queue.append(j)

        weights = np.array([rd_map[g[0], g[1]] for g in group])
        weights = weights / weights.sum()
        mean_d = int(round(np.sum([g[0] * w for g, w in zip(group, weights)])))
        mean_r = int(round(np.sum([g[1] * w for g, w in zip(group, weights)])))

        clusters.append({'d': mean_d, 'r': mean_r})
    return clusters

##### Execution Pipeline ##### 
if __name__ == "__main__":
    config = RadarConfig()
    env = Environment(config)
    sim = RadarSimulator(config, env)
    proc = SignalProcessor(config)
    
    active_tracks = {} 
    next_id = 0
    
    frame_time = 0.04 # 40ms per frame matches Kalman dt=0.04

    for frame in range(20): 
        mimo_cube = sim.generate_mimo_cube()
        fft_angle, rd_map, _, _, _, fft_doppler_comp = proc.process(mimo_cube)
        
        d_hits, r_hits = proc.execute_ca_cfar_2d(rd_map)
        clusters = cluster_detections(d_hits, r_hits, rd_map, proc)

        frame_detections = []
        for cl in clusters:
            d, r = cl['d'], cl['r']
            spatial_vec = fft_doppler_comp[:, d, r]

            _sub = 6
            _M   = len(spatial_vec) - _sub + 1
            _Rf  = np.zeros((_sub, _sub), dtype=complex)
            for _i in range(_M):
                _s = spatial_vec[_i:_i+_sub].reshape(-1, 1)
                _Rf += _s @ _s.conj().T
            _Rf /= _M
            _J   = np.fliplr(np.eye(_sub))
            _Rfb = 0.5 * (_Rf + _J @ _Rf.conj() @ _J)
            _eigs = np.sort(np.real(la.eigvalsh(_Rfb)))[::-1]

            max_eig = max(_eigs[0], 1e-12)
            num_src = np.sum(_eigs > max_eig * 0.02)   
            num_src = min(max(int(num_src), 1), 2)     

            if num_src == 2:
                angles = proc.execute_esprit_doa(spatial_vec, num_sources=2)
                for az in angles:
                    frame_detections.append({'r': proc.ranges[r], 'v': proc.velocities[d], 'az': float(az)})
            else:
                angle = ESPRITSolver.solve(spatial_vec, num_sources=1)
                frame_detections.append({'r': proc.ranges[r], 'v': proc.velocities[d], 'az': float(angle[0])})

        T_cpi = proc.cfg.N_chirps * proc.cfg.N_tx * proc.cfg.T_chirp
        v_res = proc.cfg.lambda_val / (2 * T_cpi)
        ghost_v_thresh = 2 * v_res   

        filtered_detections = []
        frame_detections.sort(key=lambda x: x['r'])

        for det in frame_detections:
            is_ghost = False
            for prev_det in filtered_detections:
                v_diff = abs(det['v'] - prev_det['v'])
                r_diff = det['r'] - prev_det['r']
                if v_diff < ghost_v_thresh and r_diff > 3.0:
                    is_ghost = True
                    break
            if not is_ghost:
                filtered_detections.append(det)

        frame_detections = filtered_detections

        # Data Association via Hungarian Algorithm
        for tid, track in active_tracks.items():
            track['filter'].predict()
            track['misses'] += 1 
            
        num_tracks = len(active_tracks)
        num_dets = len(frame_detections)
        track_ids = list(active_tracks.keys())
        
        cost_matrix = np.full((num_tracks, num_dets), np.inf)
        
        for i, tid in enumerate(track_ids):
            track = active_tracks[tid]
            for j, det in enumerate(frame_detections):
                meas = np.array([det['r'], det['v'], det['az']])
                dist_sq = track['filter'].mahalanobis_distance(meas)
                if dist_sq < 11.34: 
                    cost_matrix[i, j] = dist_sq
                    
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_track_ids = set()
        matched_det_indices = set()

        for r_idx, c_idx in zip(row_ind, col_ind):
            if cost_matrix[r_idx, c_idx] != np.inf:
                tid = track_ids[r_idx]
                det = frame_detections[c_idx]
                meas = np.array([det['r'], det['v'], det['az']])
                
                active_tracks[tid]['filter'].update(meas)
                active_tracks[tid]['misses'] = 0
                active_tracks[tid]['hits'] += 1
                
                matched_track_ids.add(tid)
                matched_det_indices.add(c_idx)

        # Create tracks for unassigned detections
        for j, det in enumerate(frame_detections):
            if j not in matched_det_indices:
                meas = np.array([det['r'], det['v'], det['az']])
                new_kf = KalmanTracker()
                new_kf.update(meas)
                active_tracks[next_id] = {
                    'filter': new_kf, 'hits': 1, 'misses': 0
                }
                next_id += 1

        active_tracks = {tid: t for tid, t in active_tracks.items() if t['misses'] < 3}

        min_hits = min(3, frame + 1)
        confirmed_tracks = [t for t in active_tracks.values() if t['hits'] >= min_hits]
        print(f"\nFrame {frame} | {len(confirmed_tracks)} confirmed objects.")

        for tid, track in active_tracks.items():
            if track['hits'] >= min_hits:
                r_est = track['filter'].x[0, 0]
                v_est = track['filter'].x[1, 0]
                az_est = track['filter'].x[2, 0]
                print(f"  ID {tid}: Range={r_est:.2f}m, Velocity={v_est:.2f}m/s, Angle={az_est:.2f}°")
                
        # Move the targets forward in time for the next frame
        env.update(frame_time)