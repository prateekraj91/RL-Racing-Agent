# Pre-Audit Evidence Pack: Buddh International Circuit Evaluation
**Target**: Engineering Audit by Claude Code  
**Repository**: `RL Racing Agent` | **Branch**: `feat/visual-uplift`  
**HEAD Commit**: `ca6a7f1abe025bc4801d5cf537116c973234a599`  
**Policy Under Audit**: `generalist-v1.0 SAC` (`models/generalist_v1.pth`, FROZEN)  
**Audit Date**: November 21, 2026  
**Auditor Mode**: Read-Only Independent Arithmetic & Artifact Verification  

---

## Executive Summary & Verdict Status

This document provides an independent, read-only verification of the data, code, arithmetic, and physical claims made in the Buddh International Circuit (BIC) Evaluation Report (`buddh_evaluation_report.md`). Every underlying CSV rollout, JSON metric file, unit test, and mathematical derivation was inspected without modifying existing code or retraining the frozen policy.

### Audit Verdict Classification

| Area | Status | Summary Finding |
| :--- | :---: | :--- |
| **Model & Config Integrity** | **`VERIFIED`** | `models/generalist_v1.pth` and `configs/generalist_v1.json` match frozen SHA-256 hashes bitwise. |
| **Lap Completion Claim** | **`VERIFIED`** | Full lap completed in 7,521 steps (125.35 s) with 0 collisions and 0 track departures. |
| **Average & Top Speeds** | **`VERIFIED`** | Recomputed mean speed is 147.32 km/h (3.979 px/step); top speed is 148.11 km/h (4.00 px/step). |
| **Turn 3 Apex Speed Claim** | **`DISCREPANCY`** | **CRITICAL**: The report claimed the vehicle slowed to 56.16 km/h at Turn 3 apex. Recomputation proves the car took Turn 3 at **148.11 km/h** under full throttle. 56.16 km/h was simply Step 31 during start-line launch acceleration. |
| **Mean Lateral Deviation** | **`DISCREPANCY`** | The report claimed 0.142 m (0.83 px) Mean deviation. Recomputation reveals true Mean is **0.059 m (0.343 px)**; 0.142 m is the **RMS** deviation duplicated into the Mean field. |
| **Observation Dimensions** | **`DISCREPANCY`** | The report table documented 8 floats; the actual PyTorch actor and environment use **9 floats** (heading error was omitted from the report text). |
| **Robustness 12/12 Completion**| **`VERIFIED`** | All 12 trials completed laps with 0 crashes, 0 off-track steps, and verified nominal repeatability. |
| **Robustness Trial Labels** | **`DISCREPANCY`** | Trials 03–10 had perturbation labels inverted/swapped in the report table, and Trial 12 step count was misreported as 7,506 instead of 7,521. |
| **Scale Arithmetic** | **`VERIFIED`** | Pixel-to-metric conversions match mathematically ($5.8333\text{ px/m}$), but produce an oversized vehicle ($L_{\text{phys}} = 8.57\text{ m}$) requiring physical investigation. |
| **Offline Reproducibility** | **`VERIFIED`** | Importer and evaluation suites run 100% offline from cached OSM data; 9/9 pytest tests pass in 1.6 s. |

---

## 1. Repository Inventory

The following table catalogs all committed Buddh International Circuit evaluation assets on branch `feat/visual-uplift` at commit `ca6a7f1abe025bc4801d5cf537116c973234a599`. Working tree is verified clean.

### Core Inventory Registry

| File Category | Exact Repository Path | Size (Bytes) | SHA-256 Checksum | Purpose / Content |
| :--- | :--- | :---: | :--- | :--- |
| **Frozen Model** | [models/generalist_v1.pth](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/models/generalist_v1.pth) | 76,919 | `0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784` | Frozen SAC PyTorch actor weights |
| **Model Config** | [configs/generalist_v1.json](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/configs/generalist_v1.json) | 5,566 | `4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1` | Observation/action schema and metadata |
| **Raw OSM Cache**| [data/osm/buddh_circuit_raw_osm.json](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/data/osm/buddh_circuit_raw_osm.json) | 38,172 | `c552d9f63d6fe62f44a18ea19287198ea4a06bc4134035f514c5ab266d2e5fb6` | Raw OSM Relation 2532569 JSON (206 nodes) |
| **OSM Metadata** | [data/osm/metadata.json](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/data/osm/metadata.json) | 746 | `807217d8bc3284a6d512edce9b6b25572445ca79d69b6c8e38cc546f69a18fd3` | Source URL, retrieval date, ODbL attribution |
| **Importer** | [buddh/importer.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/importer.py) | 19,822 | `f1e79662ad18b0d496143e08ff0d0ddd9a9612f8b73e4d6e5ccc57f5c3000875` | UTM 43N projector & Catmull-Rom resampler |
| **Track Manifest**| [buddh/buddh_manifest.json](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/buddh_manifest.json) | 6,000 | `a107a623ce8a87b071a578e336e8ffd8630e501356413b514e8cfec7c6c04028` | Verified circuit metadata and 16-turn registry |
| **Track Arrays** | [buddh/buddh_track_data.npz](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/buddh_track_data.npz) | 518,000 | `41e6ef0d948e06a09a5a9ae09b21598b303877b4d2ea3698d8d2131e05cffc71` | Centerline, boundaries, tangents, curvature |
| **Track Logic** | [buddh/buddh_track.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/buddh_track.py) | 9,576 | `a23ee53898548e673463796f061c38cbe28706146f3ca9b38a675ba57c5a365f` | Gymnasium track wrapper with 2D spatial grid |
| **Geometry Check**| [buddh/validate_geometry.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/validate_geometry.py) | 12,735 | `0fba98a8ca0e8734c357bcda902e0b63d65d650a38e948304a15a15eb118693a` | Invariant validator and top-down plotter |
| **Evaluation Run**| [buddh/eval_buddh.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/eval_buddh.py) | 14,723 | `3247e17da9bdba143cef5000837d02f786d8383531ef566c4c6ca7d33f7de22e` | 15,000-step rollout runner with telemetry |
| **Robust Config**| [buddh/robustness_manifest.json](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/robustness_manifest.json) | 2,478 | `6060225513d26bbd166fcff91fdf464ec7db6eb3830403ff731998ad117ee317` | 12-trial perturbation configuration |
| **Robust Suite** | [buddh/robustness_suite.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/robustness_suite.py) | 8,175 | `ca8374229fe7669cc36111fc5df5bc9b5361db5458bc34dfff7a2dd06e990dbc` | Automated 12-trial batch runner |
| **Plot Generator**| [buddh/plot_analysis.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/plot_analysis.py) | 21,727 | `e29be14817723dfcc86d5356e657e560e3eaab9b4cdd79b9911ef4b8a5ab50a8` | Publication figure generator |
| **Rollout CSV** | [analysis/buddh/buddh_full_rollout.csv](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_full_rollout.csv) | 1,159,536 | `77cda1495939dd7c5b9355d65bd53f886c7fada5addd10b0b50dab1160ced248` | 7,521-row nominal rollout telemetry |
| **Metrics JSON** | [analysis/buddh/buddh_evaluation_metrics.json](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_evaluation_metrics.json) | 1,323 | `0f6312f1c07912e414483fc1597e51228f0f25effe251852083cb15f3a6b18e2` | Summary dictionary of 14 evaluation metrics |
| **Robust CSV** | [analysis/buddh/robustness_summary.csv](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/robustness_summary.csv) | 2,654 | `616fda7aa5ee54444a683098a5ebd85c76c8a3dff85622a81b7deeeb84f37e11` | Tabular summary of 12 robustness runs |
| **Robust JSON**| [analysis/buddh/robustness_summary.json](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/robustness_summary.json) | 8,641 | `6532c3637cc0eed36425f8b16e46a73182cecdcab56ef23cec827f08c685bdb0` | Detailed trial telemetry and recovery metrics |
| **Importer Tests**| [tests/test_buddh_importer.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/tests/test_buddh_importer.py) | 2,003 | `5b674cc6f0228cf0e86213cf36775e25c8f9db4284d2913e1be5ad81e0fc6151` | Projection, spline, and geometry unit tests |
| **Suite Tests** | [tests/test_buddh_suite.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/tests/test_buddh_suite.py) | 4,922 | `5ee22cd9006ae68a8d0ae49750af793725dd3aef3a8d11dddffb45dda3244243` | Invariants, weights, and smoke integration tests |

### Committed Telemetry Plots

All 9 figures located in [analysis/buddh/](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/):
1. [buddh_geometry_validation.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_geometry_validation.png) (676 KB) — Track geometry, width, curvature radii, turns.
2. [buddh_full_lap_trajectory.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_full_lap_trajectory.png) (613 KB) — 2D trajectory overlaid with apex callouts.
3. [buddh_telemetry_dashboard.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_telemetry_dashboard.png) (481 KB) — Multi-panel speed, steering, lateral deviation, rays.
4. [buddh_speed_profile.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_speed_profile.png) (160 KB) — Forward speed vs circuit distance.
5. [buddh_steering_profile.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_steering_profile.png) (305 KB) — Steering angle action & derivative vs distance.
6. [buddh_lateral_deviation.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_lateral_deviation.png) (208 KB) — Lateral error corridor bounds.
7. [buddh_heading_error.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_heading_error.png) (204 KB) — Yaw error relative to track tangent.
8. [buddh_off_track_collision_timeline.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_off_track_collision_timeline.png) (111 KB) — Zero off-track / zero crash verification.
9. [buddh_robustness_comparison.png](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/buddh/buddh_robustness_comparison.png) (222 KB) — Multi-trial recovery overlay.

---

## 2. Independent Metric Recomputation

Using pure Python (`csv`, `numpy`) on the 7,521 rows of `analysis/buddh/buddh_full_rollout.csv`, all metrics were independently recalculated and compared against both `analysis/buddh/buddh_evaluation_metrics.json` and the Markdown report (`buddh_evaluation_report.md`).

### Comprehensive Recomputation Comparison

| Evaluation Metric | Reported in Report Table | Recomputed from Raw CSV | Recomputed from JSON | Audit Status | Audit Notes |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Total Simulation Steps** | 7,521 | 7,521 | 7,521 | **`VERIFIED`** | CSV row count = 7,521; step 1 to 7,521 |
| **Lap Time ($dt = 1/60\text{ s}$)**| 125.35 s | 125.350 s | 125.35 s | **`VERIFIED`** | $7,521 \times (1/60) = 125.350\text{ s}$ |
| **Total Distance Traversed** | 5,131.0 m | 5,131.66 m (progress)<br/>5,129.56 m (Euclidean) | 5,131.66 m | **`VERIFIED`** | Difference vs 5,131.0 m nominal is +0.013% |
| **Average Speed (px/step)** | 3.979 px/step | 3.97852 px/step | 3.979 px/step | **`VERIFIED`** | Rounds to 3.979 px/step |
| **Average Speed (km/h)** | 147.32 km/h | 147.319 km/h | 147.32 km/h | **`VERIFIED`** | Calculated via exact conversion: $3.97852 \times 60 \times 3.6 \times (12/70)$ |
| **Maximum Speed (px/step)** | 4.000 px/step | 4.0000 px/step | 4.000 px/step | **`VERIFIED`** | Policy speed clamp at 4.0 px/step |
| **Maximum Speed (km/h)** | 148.11 km/h | 148.114 km/h | 148.11 km/h | **`VERIFIED`** | Exact top speed clamp |
| **Minimum Speed (Overall)** | 1.520 px/step / 56.16 km/h | **0.0489 px/step / 1.81 km/h** | 0.049 px/step / 1.81 km/h | **`DISCREPANCY`** | Occurs at Step 1 accelerating from standstill |
| **Turn 3 Apex Speed** | 56.16 km/h | **148.11 km/h** | N/A (misinterpreted) | **`DISCREPANCY`** | **CRITICAL**: Car took Turn 3 at full speed (148.11 km/h). 56.16 km/h was Step 31 during start launch! |
| **Mean Lateral Dev (px)** | 0.830 px | **0.343 px** | 0.343 px | **`DISCREPANCY`** | Report duplicated RMS (0.828 px) as Mean |
| **Mean Lateral Dev (m)** | 0.142 m | **0.059 m** | 0.059 m | **`DISCREPANCY`** | Report duplicated RMS (0.142 m) as Mean |
| **RMS Lateral Dev (px)** | 0.830 px | 0.828 px | 0.828 px | **`VERIFIED`** | Minor 0.002 px rounding difference |
| **RMS Lateral Dev (m)** | 0.142 m | 0.142 m | 0.142 m | **`VERIFIED`** | Matches exact 0.142 m |
| **P95 Lateral Dev (m)** | 0.371 m | 0.361 m | 0.361 m | **`DISCREPANCY`** | 0.010 m discrepancy (percentile interpolation) |
| **Max Lateral Dev (m)** | 1.002 m (5.845 px) | 1.002 m (5.845 px) | 1.002 m (5.845 px) | **`VERIFIED`** | Reached on Turn 3 exit |
| **Collisions** | 0 | 0 | 0 | **`VERIFIED`** | `crashed.sum() == 0` |
| **Off-track Steps** | 0 | 0 | 0 | **`VERIFIED`** | Zero points with $\|d\| > 35\text{ px}$ |
| **Mean Steering Magnitude** | 0.0951 | 0.09513 | 0.0951 | **`VERIFIED`** | Matches to 4 decimal places |
| **Max Steering Magnitude** | 0.9998 | 0.99980 | 0.9998 | **`VERIFIED`** | Reached in Turn 3 apex |
| **Steering Consecutive $\Delta a$**| 0.0546 | 0.05452 (diff)<br/>0.05463 (column) | 0.0546 | **`VERIFIED`** | Matches column average |
| **Heading Error RMS** | 1.053° | 1.0526° | 1.053° | **`VERIFIED`** | Matches |
| **Heading Error Max** | 12.515° | 12.515° | 12.515° | **`VERIFIED`** | Matches |
| **Termination Reason** | `lap_completed` | `lap_completed` | `lap_completed` | **`VERIFIED`** | Last row `lap_completed == True` |

---

## 3. Robustness Data Audit

The 12 individual CSV files in `analysis/buddh/rollouts/` were independently audited against `buddh/robustness_manifest.json` and `analysis/buddh/robustness_summary.json`.

### Verification Findings

1. **Trial Count & Completion Rate**: Exactly 12 trials exist. All 12 completed the lap (`lap_completed = True`). Off-track steps = 0 and collisions = 0 across all 12 trials. (**`VERIFIED`**).
2. **Nominal Run Deterministic Parity**: `trial_01_nominal_rep1` and `trial_02_nominal_rep2` produce **bitwise identical** rollouts: identical step count (7,521), identical lap time (125.35 s), identical mean deviation (0.142 m RMS), and identical control trajectories. (**`VERIFIED`**).
3. **Initial Condition Perturbation Parity**: In each trial's CSV, the initial state at Step 1 confirms that perturbations were genuinely applied:
   - `trial_03`: Initial signed distance is $-15.0\text{ px}$ (left offset).
   - `trial_04`: Initial signed distance is $+15.0\text{ px}$ (right offset).
   - `trial_05`: Initial signed distance is $-8.0\text{ px}$ (left offset).
   - `trial_06`: Initial signed distance is $+8.0\text{ px}$ (right offset).
   - `trial_07`: Initial heading error is $-9.970^\circ$ (left yaw offset).
   - `trial_08`: Initial heading error is $+9.972^\circ$ (right yaw offset).
   - `trial_09`: Initial heading error is $-4.969^\circ$ (left yaw offset).
   - `trial_10`: Initial heading error is $+4.993^\circ$ (right yaw offset).
   - `trial_11`: Initial signed distance is $+10.004\text{ px}$, heading error is $-4.973^\circ$.
   - `trial_12`: Initial position is $x=-21.02, y=-54.46$ ($+58.33\text{ px} = +10.0\text{ m}$ along track).

### Robustness Report Discrepancy Table

A comparison between the committed CSV/JSON data and the Markdown report revealed transcription errors in the report's Phase 6 table:

| Trial ID | True Configured Perturbation | Perturbation Claimed in Report Table | Discrepancy Description |
| :--- | :--- | :--- | :--- |
| `trial_03` | **$-15.0\text{ px}$ ($-2.57\text{ m}$ Left)** | $+8.0\text{ px}$ ($+1.37\text{ m}$ Left Small) | **`DISCREPANCY`**: Inverted with Trial 05; magnitude misstated |
| `trial_04` | **$+15.0\text{ px}$ ($+2.57\text{ m}$ Right)** | $-8.0\text{ px}$ ($-1.37\text{ m}$ Right Small) | **`DISCREPANCY`**: Inverted with Trial 06; magnitude misstated |
| `trial_05` | **$-8.0\text{ px}$ ($-1.37\text{ m}$ Left)** | $+15.0\text{ px}$ ($+2.57\text{ m}$ Left Large) | **`DISCREPANCY`**: Inverted with Trial 03; magnitude misstated |
| `trial_06` | **$+8.0\text{ px}$ ($+1.37\text{ m}$ Right)** | $-15.0\text{ px}$ ($-2.57\text{ m}$ Right Large) | **`DISCREPANCY`**: Inverted with Trial 04; magnitude misstated |
| `trial_07` | **$-10.0^\circ$ Yaw (Left Large)** | $+5.0^\circ$ Yaw (Left Small) | **`DISCREPANCY`**: Inverted with Trial 09; angle misstated |
| `trial_08` | **$+10.0^\circ$ Yaw (Right Large)** | $-5.0^\circ$ Yaw (Right Small) | **`DISCREPANCY`**: Inverted with Trial 10; angle misstated |
| `trial_09` | **$-5.0^\circ$ Yaw (Left Small)** | $+10.0^\circ$ Yaw (Left Large) | **`DISCREPANCY`**: Inverted with Trial 07; angle misstated |
| `trial_10` | **$+5.0^\circ$ Yaw (Right Small)** | $-10.0^\circ$ Yaw (Right Large) | **`DISCREPANCY`**: Inverted with Trial 08; angle misstated |
| `trial_12` | **7,521 steps / 125.35 s** | **7,506 steps / 125.10 s** | **`DISCREPANCY`**: Report author fabricated 7,506 steps assuming early lap completion, but code required full traversal to 7,521 steps. |

---

## 4. Scale Arithmetic & Physical Validity Audit

### Verification of Reported Arithmetic

1. **Scale Ratio ($S$)**:
   $$\text{Reference Width } W_{\text{real}} = 12.0\text{ m}, \quad \text{Sim Corridor } W_{\text{sim}} = 70.0\text{ px}$$
   $$S = \frac{70.0}{12.0} = \frac{35}{6} \approx 5.833333333...\text{ px/m} \quad (\mathbf{VERIFIED})$$
   $$\text{Inverse Scale: } 1\text{ px} = \frac{12}{70} = \frac{6}{35} \approx 0.171428571...\text{ m} \quad (\mathbf{VERIFIED})$$

2. **Simulation Speed to Physical Speed**:
   $$v = 3.978522\text{ px/step}, \quad dt = \frac{1}{60}\text{ s} \implies 60\text{ steps/s}$$
   $$v_{\text{km/h}} = v \times 60 \times 3.6 \times \frac{12}{70} = 3.978522 \times 37.028571 = 147.319\text{ km/h} \quad (\mathbf{VERIFIED})$$
   $$\text{Top speed clamp: } 4.0\text{ px/step} \times 37.028571 = 148.114\text{ km/h} \quad (\mathbf{VERIFIED})$$

3. **Implied Physical Wheelbase ($L_{\text{phys}}$)**:
   $$\text{Simulation Wheelbase } L = 50.0\text{ px}$$
   $$L_{\text{phys}} = \frac{50.0}{5.833333} = 50.0 \times \frac{6}{35} = \frac{60}{7} \approx \mathbf{8.571\text{ meters}}$$

4. **Minimum Turning Radius ($R_{\min}$)**:
   $$\text{Max Steer Angle } \delta_{\max} = 0.523599\text{ rad} = 30.0^\circ \implies \tan(30^\circ) = \frac{1}{\sqrt{3}}$$
   $$R_{\min, \text{sim}} = \frac{L}{\tan(30^\circ)} = 50 \times \sqrt{3} \approx \mathbf{86.6025\text{ px}} \quad (\mathbf{VERIFIED})$$
   $$R_{\min, \text{phys}} = \frac{R_{\min, \text{sim}}}{S} = \frac{86.6025}{5.833333} = \frac{60 \sqrt{3}}{7} \approx \mathbf{14.846\text{ meters}}$$

5. **Ray Range & Lookahead Horizon**:
   $$\text{Max Ray Distance } = 200.0\text{ px} \implies 200.0 \times \frac{6}{35} \approx \mathbf{34.286\text{ meters}} \quad (\mathbf{VERIFIED})$$
   $$\text{Marching Step } = 4.0\text{ px} \implies 4.0 \times \frac{6}{35} \approx \mathbf{0.686\text{ meters}} \quad (\mathbf{VERIFIED})$$
   $$\text{Preview Horizon at } 147.3\text{ km/h } (40.92\text{ m/s}): \tau = \frac{34.286\text{ m}}{40.92\text{ m/s}} \approx \mathbf{0.838\text{ seconds}}$$

### Physical Interpretation Concerns for Claude Code

The arithmetic is internally consistent, but the physical implications reveal major idealizations that Claude Code must evaluate:

1. **Oversized Vehicle Geometry (Bus-Sized Wheelbase)**:
   - An effective wheelbase of **8.57 meters** is more than double that of a modern Formula 1 car ($\sim 3.6\text{ m}$) or GT3 car ($\sim 2.7\text{ m}$). It is comparable to a municipal transit bus.
   - *Cause*: The scale $S = 5.8333\text{ px/m}$ was chosen to calibrate track width ($70\text{ px} = 12\text{ m}$), while the car's pixel dimensions ($50\text{ px}$) were left at their training default to avoid altering the observation space.
2. **Turn 3 Radius vs Minimum Vehicle Turning Radius**:
   - The geometric centerline of Turn 3 has a radius of $R_{\text{curve}} = 14.35\text{ m}$ ($83.71\text{ px}$).
   - The vehicle's kinematic minimum turning radius is $R_{\min} = 14.85\text{ m}$ ($86.60\text{ px}$).
   - Therefore, $R_{\text{curve}} < R_{\min}$! A vehicle attempting to follow the centerline would be kinematically incapable of making the turn.
   - *How the agent succeeded*: The agent drifted outward from the centerline by $1.002\text{ m}$ ($5.85\text{ px}$), taking an effective arc of $R_{\text{eff}} = 83.71 + 5.85 = 89.56\text{ px} > 86.60\text{ px}$ ($15.35\text{ m} > 14.85\text{ m}$). This explains why steering saturated at $0.9998$ and lateral error peaked at $1.002\text{ m}$.
3. **Absence of Lateral G-Force and Grip Degradation**:
   - In physical reality, cornering at $R = 15.35\text{ m}$ at $v = 148.11\text{ km/h}$ ($41.14\text{ m/s}$) produces a lateral acceleration of:
     $$a_{\text{lat}} = \frac{v^2}{R} = \frac{41.14^2}{15.35} \approx 110.2\text{ m/s}^2 \approx \mathbf{11.2\text{ G}}$$
   - Modern F1 cars generate at most $5.0 - 5.5\text{ G}$ with extreme aerodynamic downforce. Physical race cars must brake to $\sim 60\text{ km/h}$ ($2.0\text{ G}$) to navigate Turn 3.
   - The simulation's kinematic model ignores lateral load transfer and centrifugal slip limits, allowing the agent to negotiate Turn 3 at full throttle without spinning out.

---

## 5. Integrity and Reproducibility Verification

### Frozen Checksum Audit

| File | Type | Expected SHA-256 Checksum | Actual Computed SHA-256 Checksum | Match Status |
| :--- | :--- | :--- | :--- | :---: |
| `models/generalist_v1.pth` | Weights | `0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784` | `0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784` | **`VERIFIED`** |
| `configs/generalist_v1.json` | Config | `4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1` | `4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1` | **`VERIFIED`** |

### Unit Test Execution

Executed command: `./venv/bin/pytest tests/test_buddh_importer.py tests/test_buddh_suite.py -v`

```text
tests/test_buddh_importer.py::test_utm43n_projection_accuracy PASSED     [ 11%]
tests/test_buddh_importer.py::test_centripetal_catmull_rom_properties PASSED [ 22%]
tests/test_buddh_importer.py::test_importer_execution_and_geometry PASSED [ 33%]
tests/test_buddh_suite.py::test_frozen_weights_integrity PASSED          [ 44%]
tests/test_buddh_suite.py::test_raw_osm_cached_data PASSED               [ 55%]
tests/test_buddh_suite.py::test_manifest_consistency PASSED              [ 66%]
tests/test_buddh_suite.py::test_track_interface_contract PASSED          [ 77%]
tests/test_buddh_suite.py::test_geometry_invariants PASSED               [ 88%]
tests/test_buddh_suite.py::test_policy_smoke_step PASSED                 [100%]
============================== 9 passed in 1.61s ===============================
```
**Status**: **`VERIFIED`** (All 9 tests pass without warnings).

### Offline OSM Cache Sufficiency

- Checked [buddh/importer.py](file:///Users/prateekraj/Desktop/Ideas/RL Racing Agent/buddh/importer.py): Contains zero `import urllib`, `import requests`, `import socket`, or network calls.
- Purely parses `data/osm/buddh_circuit_raw_osm.json`.
- When executed in an air-gapped environment without internet access, `BuddhTrackImporter` generates the exact bitwise manifest and numpy array.
- **Status**: **`VERIFIED`** (Geometry generation is 100% self-contained and reproducible offline).

---

## 6. Comprehensive Discrepancy Registry

The following table aggregates all identified discrepancies between the committed evaluation data and the published evaluation report text.

| # | Discrepancy Title | Severity | Detail / Evidence | Root Cause |
| :-: | :--- | :---: | :--- | :--- |
| **D1** | **Turn 3 Speed Modulation Narrative** | **HIGH** | Report claimed: *"In Turn 3, the policy automatically modulated forward speed from 148.1 km/h down to 56.16 km/h"*. Reality: Speed at Turn 3 was **148.11 km/h**; 56.16 km/h was Step 31 during start launch. | Narrative hallucination based on filtering `speeds_kmh[30:]` without checking spatial coordinates of the minimum. |
| **D2** | **Mean Lateral Deviation Duplication** | **MEDIUM** | Report claimed: Mean = 0.142 m (0.83 px). Reality: True Mean = **0.059 m (0.343 px)**. 0.142 m was the **RMS** value copy-pasted into both fields. | Editorial copy-paste error in markdown metrics table. |
| **D3** | **Observation Space Dimension Count** | **MEDIUM** | Report Phase 3 table stated: *"8 floats: [v, slip, d_center, r1, r2, r3, r4, r5]"*. Reality: Code and actor network require **9 floats** (heading error was omitted from written list). | Omission of `heading_error` feature in markdown documentation. |
| **D4** | **Robustness Trial Label Transposition** | **LOW** | Report table swapped labels for Trials 03–06 ($\pm 15\text{ px}$ vs $\pm 8\text{ px}$) and Trials 07–10 ($\pm 10^\circ$ vs $\pm 5^\circ$). | Table rows manually populated out of order relative to JSON trial IDs. |
| **D5** | **Trial 12 Step Count & Time Fabrication**| **MEDIUM** | Report table claimed Trial 12 had *"7,506 steps / 125.10 s"*. Reality: `trial_12_longitudinal_10m.csv` contains **7,521 steps / 125.35 s**. | Author theoretically deducted 15 steps for 10m forward start without inspecting the simulation run output. |

---

## 7. Questions Requiring Claude's Engineering Investigation

During tomorrow's deeper engineering audit, Claude Code should investigate the following fundamental architectural questions:

### 1. Kinematic vs. Dynamic Model Limitations
- **Issue**: The agent cleared Turn 3 ($R = 14.35\text{ m}$) at $148.11\text{ km/h}$, generating an implied lateral acceleration of $11.2\text{ G}$.
- **Question for Claude**: Should future simulation iterations introduce a simple lateral acceleration penalty or friction circle constraint ($a_{\text{lat}} \le \mu g$), or does the generalist policy's scope intentionally remain bounded to pure 2D kinematic bicycle path-tracking?

### 2. Zeroed Tire Slip in Observation Space
- **Issue**: Line 73 of `buddh/eval_buddh.py` hardcodes `slip = 0.0` into the observation vector on every step.
- **Question for Claude**: In the training environment (`env/racing_env.py`), how was `slip` computed? Did the policy learn to rely on the `slip` feature, or did Domain Randomization render the policy invariant to `slip`?

### 3. Dimensional Scaling Trade-Off (Wheelbase vs. Corridor)
- **Issue**: Calibrating scale to corridor width ($70\text{ px} = 12\text{ m}$) created an $8.57\text{ m}$ physical wheelbase, while calibrating to wheelbase ($50\text{ px} = 2.6\text{ m}$) would widen the track to $230.8\text{ px}$, causing the $200\text{ px}$ rangefinders to saturate and blind the agent.
- **Question for Claude**: Is this corridor-matching calibration considered an acceptable sim-to-sim generalization mapping for ray-based RL agents, or should the agent support anisotropic scaling (scaling ray range independently of physics dimensions)?

### 4. Lap Timing Trigger Standardization
- **Issue**: Advancing the car by $+10.0\text{ m}$ in Trial 12 did not decrease total episode steps (remaining at 7,521 steps) because the lap completion flag triggers when crossing the finish line after having completed $\ge 98\%$ of the loop.
- **Question for Claude**: Should a formal finish-line spatial bounding gate or ray-plane intersection be used to record sub-step interpolated lap times?

---

## 8. Deterministic Reproduction Verification

All results can be reproduced directly on branch `feat/visual-uplift` using the following exact commands:

```bash
# 1. Run unit test suite
./venv/bin/pytest tests/test_buddh_importer.py tests/test_buddh_suite.py -v

# 2. Run independent metric recomputation script
python3 -c "
import csv, numpy as np
rows = list(csv.DictReader(open('analysis/buddh/buddh_full_rollout.csv')))
print('Steps:', len(rows))
print('Lap Time (s):', round(len(rows) / 60.0, 2))
speeds = [float(r['speed_kmh']) for r in rows]
print('Mean Speed (km/h):', round(np.mean(speeds), 2))
print('Max Speed (km/h):', round(np.max(speeds), 2))
print('T3 Min Speed (km/h):', min(float(r['speed_kmh']) for r in rows if 800 <= float(r['distance_m']) <= 1300))
dists_m = [abs(float(r['signed_dist_m'])) for r in rows]
print('Mean Lat Dev (m):', round(np.mean(dists_m), 3))
print('RMS Lat Dev (m):', round(np.sqrt(np.mean(np.array(dists_m)**2)), 3))
print('Max Lat Dev (m):', round(max(dists_m), 3))
"

# 3. Verify robustness trial completion
python3 -c "
import json
data = json.load(open('analysis/buddh/robustness_summary.json'))
print('Total trials:', data['total_trials'])
print('Completed trials:', data['completed_trials'])
print('Success rate:', data['success_rate_pct'])
"
```
