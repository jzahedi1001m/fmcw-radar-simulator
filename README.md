# FMCW MIMO Radar Simulation & Tracking Pipeline
This repository contains a high-fidelity FMCW (Frequency-Modulated Continuous Wave) MIMO radar simulator written in Python. It models the complete end-to-end signal processing chain—from physical RF hardware behaviors to multi-target tracking.

The framework captures real-world physical constraints, including phase noise, thermal noise, multipath ghosting, and quantization behavior, while providing robust state estimation for dynamic targets.

# System Architecture & Mathematical Core
The pipeline employs a modular architecture, transforming raw electromagnetic returns into stabilized target tracks.

```text
+-----------------------+     +------------------------+     +-------------------+
|   Raw ADC Data Cube   |     |   DSP Processing Chain |     |   GNN Tracking    |
| (Noise + Interference)|     | (FFT, CFAR, ESPRIT)    |     | (Kalman Filters)  |
+-----------+-----------+     +-----------+------------+     +---------+---------+
            |                             |                          |
            v                             v                          v
+-----------------------+     +------------------------+     +-------------------+
|  Environment & Plant  +---->+   Range-Doppler Map    +---->+  Confirmed Track  |
|  (Targets & Clutter)  |     | (Clutter Suppression)  |     |  State Estimates  |
+-----------------------+     +------------------------+     +-------------------+

```

### 1. Radar Frontend & Physical PlantSignal Generation: 
Models per-chirp phase noise, $kTB$ thermal noise floors, and hardware-level ADC quantization (12-bit).MIMO Synthesis: Implements TDM (Time Division Multiplexing) MIMO phase compensation to synthesize a virtual aperture, enabling high-resolution spatial estimation.Interference Engine: Generates environmental complexities including direct leakage, bumper reflections, and asynchronous radar interference.

### 2. DSP & Detection PipelineWindowing: 
Utilizes a Chebyshev/Blackman windowing combination to optimize range sidelobe suppression.2D-CFAR: Implements a Constant False Alarm Rate detector with constant-value padding to eliminate boundary ghost hits.ESPRIT DOA: Employs high-resolution ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques) with forward-backward spatial smoothing for sub-degree angle estimation.

### 3. State Estimation & FilteringNumerical Stability: 
Utilizes the Joseph form of the Kalman covariance update equation to guarantee positive semi-definiteness, ensuring long-term filter stability.Mahalanobis Gating: Applies a 3.5-Sigma Mahalanobis Innovation Validation Gate to filter transient measurement outliers.

### 4. Data AssociationGlobal Nearest Neighbor (GNN): 
Replaces greedy heuristics with the Hungarian Algorithm (scipy.optimize.linear_sum_assignment) to perform optimal global assignment, preventing track swapping during target crossings or proximity events.

---

## Getting Started

### Prerequisites
* Python 3.8 or higher
* scipy
* numpy

### Installation
1. Clone this repository to your machine:
   git clone https://github.com/yourusername/fmcw-radar-sim.git

### Running the Simulation
To execute the runtime hardware simulator loop and view the real-time control, run:
```bash
python src/fmcw_sim.py
```

### Simulation Analysis
When executed, the system tracks multiple targets through successive frames. Based on the default environment specifications, the output for first 2 frames as follows:

```text
Frame 0 | 5 confirmed objects.
  ID 0: Range=7.79m, Velocity=1.52m/s, Angle=-25.00°
  ID 1: Range=19.79m, Velocity=-7.60m/s, Angle=-14.95°
  ID 2: Range=19.79m, Velocity=0.00m/s, Angle=-10.01°
  ID 3: Range=19.79m, Velocity=0.00m/s, Angle=-0.02°
  ID 4: Range=34.78m, Velocity=7.98m/s, Angle=11.99°

Frame 1 | 5 confirmed objects.
  ID 0: Range=7.79m, Velocity=1.52m/s, Angle=-25.02°
  ID 1: Range=19.79m, Velocity=-7.60m/s, Angle=-14.96°
  ID 2: Range=19.79m, Velocity=0.00m/s, Angle=-9.98°
  ID 3: Range=19.79m, Velocity=0.00m/s, Angle=-0.00°
  ID 4: Range=35.38m, Velocity=7.98m/s, Angle=11.98°

Frame 2 | 5 confirmed objects.
  ID 0: Range=8.15m, Velocity=1.53m/s, Angle=-25.01°
  ID 1: Range=19.32m, Velocity=-7.61m/s, Angle=-14.96°
  ID 2: Range=19.79m, Velocity=0.00m/s, Angle=-9.98°
  ID 3: Range=19.79m, Velocity=0.00m/s, Angle=0.01°
  ID 4: Range=35.52m, Velocity=7.98m/s, Angle=12.00°

Frame 3 | 5 confirmed objects.
  ID 0: Range=8.29m, Velocity=1.53m/s, Angle=-25.00°
  ID 1: Range=19.09m, Velocity=-7.60m/s, Angle=-14.97°
  ID 2: Range=19.79m, Velocity=0.00m/s, Angle=-9.99°
  ID 3: Range=19.79m, Velocity=0.00m/s, Angle=0.01°
  ID 4: Range=35.90m, Velocity=7.98m/s, Angle=12.01°

Frame 4 | 5 confirmed objects.
  ID 0: Range=8.37m, Velocity=1.52m/s, Angle=-25.01°
  ID 1: Range=18.71m, Velocity=-7.61m/s, Angle=-14.96°
  ID 2: Range=19.79m, Velocity=0.00m/s, Angle=-10.00°
  ID 3: Range=19.79m, Velocity=0.00m/s, Angle=0.01°
  ID 4: Range=36.36m, Velocity=7.99m/s, Angle=12.01°
  ```
