# Generalist Model Card: `generalist_v1`

## Model Details

- **Model Identifier**: `generalist_v1`
- **Release Version**: `1.0.0`
- **Frozen Weights Path**: [`models/generalist_v1.pth`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/models/generalist_v1.pth)
- **Frozen Weights SHA-256**: `0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784`
- **Configuration Path**: [`configs/generalist_v1.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/configs/generalist_v1.json)
- **Configuration SHA-256**: `4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1`
- **Training Provenance Checkpoint**: [`runs/ablation_A_DR_look200_steerOn/best_actor.pth`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/runs/ablation_A_DR_look200_steerOn/best_actor.pth) (identical bit-for-bit to [`runs/dr_multitrack_s42_calibrated_25tracks/best_actor.pth`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/runs/dr_multitrack_s42_calibrated_25tracks/best_actor.pth))
- **Base Algorithm**: Continuous Soft Actor-Critic (SAC) with automatic entropy temperature adjustment ($\alpha$)
- **Framework**: PyTorch (`torch.nn`)
- **Evaluation Dataset**: [`analysis/results/generalist_reproducibility.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/generalist_reproducibility.csv)
- **Summary Telemetry**: [`analysis/results/generalist_reproducibility_summary.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/generalist_reproducibility_summary.json)

---

## 1. Executive Summary & Selection Justification

The goal of this milestone was to lock the top-performing generalist racing agent from all candidate checkpoints developed across the research program, freeze its weights and environment parameters, and verify deterministic reproducibility across held-out track topologies and geometric stress tests.

### Why `generalist_v1` Was Selected:
1. **Factorial Synergy**: As proven by the 2×2×2 factorial ablation matrix (`docs/ABLATION_MATRIX.md`), `generalist_v1` (Run A) uniquely combines all three critical design factors:
   - **Calibrated Domain Randomization (DR On)**: Prevents corridor narrowness collapse and eliminates the width generalization gap.
   - **200 px Lookahead Depth (Short)**: Avoids the early apex cutting and 100% boundary collisions caused by 300 px lookahead rays penetrating through tight hairpins.
   - **Steering Smoothness Penalty ($\beta_{\text{steer}} = 0.01$)**: Suppresses lateral chatter and wandering loops that induced 23 timeouts in Run B.
2. **Narrow Corridor Robustness**: Achieves **100.0% completion (70/70 laps)** across all restricted corridor widths ($W \in [70, 40, 36, 32, 28, 25, 20]\text{ px}$) with zero crashes, maintaining an invariant racing pace ($394.60\text{ steps}$ at $W=20\text{ px}$). By comparison, the Single-Track baseline slows down by $+105.4\text{ steps}$ and crashes on 40% of laps at $W=20\text{ px}$, while Run E (no DR) crawls at $435.60\text{ steps}$.
3. **Acute Curvature Navigation**: Solves 50.0% of $f=0.50$ pinched hairpins (5/10 laps) and 20.0% of $f=0.35$ hairpins (2/10 laps), outperforming all other DR-trained models in the repository.
4. **Start-State Invariance**: Completes **100.0% (60/60 trials across 3 repetitions)** from randomized start poses, heading deviations ($\pm 15^\circ$), and initial speeds ($0.0 - 4.0\text{ px/step}$) across held-out circuits.
5. **Bit-for-Bit Reproducibility**: Evaluated 3 separate times across 110 deterministic benchmark rollouts (330 total deterministic episodes), yielding **0 outcome or metric discrepancies** across repetitions and a 100% bit-for-bit match with historical ablation records.

---

## 2. Comprehensive Candidate Audit & Comparison Table

An exhaustive audit of all 156 candidate checkpoints in `models/` and `runs/` was conducted. Candidates were evaluated across standardized benchmark regimes on the 10 held-out test circuits (`TEST_SEEDS` 1001–1010):

| Candidate Identifier | Architecture & Obs Space | Training Regimen | Standard Tracks ($W=70$) | Narrow Corridors ($W \in [20, 70]$) | Hairpin Benchmark ($f \in [0.20, 0.50]$) | Start-State Robustness | Verdict & Reason |
|---|---|---|---|---|---|---|---|
| **Single-Track Baseline v2** (`models/best_single_track_v2.pth`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | Single track (seed 101), fixed $W=70$, no DR, 45k steps | 100.0% comp, 398.90 steps, 0% crash | Collapses at $W=20$ (60% comp, 504.33 steps, 2 crashes, 2 touts) | 0.0% comp across all pinch factors (21 crashes, 9 touts) | 100.0% on medium track | **Rejected as Generalist**: Overfit to single nominal track; severe corridor slowdown; 0% hairpin completion. |
| **Pool 5 (Calibrated DR)** (`runs/dr_multitrack_s42_calibrated_5tracks/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | 5 tracks (seeds 1..5), calibrated DR, 200k steps | 100.0% comp, 395.10 steps, 0% crash | 100.0% comp (70/70), 395.80 steps at $W=20$ | 0.0% comp across all pinch factors (30 crashes) | Not tested | **Rejected**: Fails to solve any pinched hairpins (0/30). |
| **Pool 10 (Calibrated DR)** (`runs/dr_multitrack_s42_calibrated_10tracks/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | 10 tracks (seeds 1..10), calibrated DR, 200k steps | 100.0% comp, 399.50 steps, 0% crash | 97.1% comp (68/70; 2 crashes at $W=36$) | 6.7% comp (2/30; 2/10 at 0.50, 0/10 at 0.35/0.20) | Not tested | **Rejected**: Narrow corridor anomaly (2 crashes at $W=36$); low hairpin completion. |
| **Pool 20 (Calibrated DR)** (`runs/dr_multitrack_s42_calibrated_20tracks/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | 20 tracks (seeds 1..20), calibrated DR, 200k steps | 100.0% comp, 398.10 steps, 0% crash | 100.0% comp (70/70), but slow pace (420.10 steps at $W=20$) | 0.0% comp across all pinch factors (30 crashes) | Not tested | **Rejected**: Slower conservative speed profile; 0% hairpin completion. |
| **Pool 25 / Run A (WINNER)** (`runs/ablation_A_DR_look200_steerOn/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | 25 tracks (seeds 1..25), calibrated DR, 200k steps | **100.0% comp, 398.20 steps, 0% crash** | **100.0% comp (70/70), 394.60 steps at $W=20$** | **23.3% comp (7/30; 5/10 at 0.50, 2/10 at 0.35)** | **100.0% (20/20 held-out)** | **SELECTED WINNER**: Dominates on narrow corridors, highest hairpin completion among DR models, invariant pace. |
| **Run B (Ablation)** (`runs/ablation_B_DR_look200_steerOff/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | 25 tracks, calibrated DR, $\beta_{\text{steer}}=0.0$, 200k steps | 100.0% comp, 397.40 steps, 0% crash | 100.0% comp (70/70), 396.10 steps at $W=20$ | 0.0% comp (0/30; 7 crashes, 23 wandering timeouts) | Not tested | **Rejected**: Severe steering chatter; fails all hairpins due to wandering loop timeouts. |
| **Run C (Ablation)** (`runs/ablation_C_DR_look300_steerOn/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 300$px) | 25 tracks, calibrated DR, lookahead 300px, 200k steps | 100.0% comp, 397.80 steps, 0% crash | 100.0% comp (70/70), 395.10 steps at $W=20$ | 0.0% comp (0/30; 30/30 crashes) | Not tested | **Rejected**: 300 px lookahead cuts corners prematurely, causing 100% boundary crashes on hairpins. |
| **Run D (Ablation)** (`runs/ablation_D_DR_look300_steerOff/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 300$px) | 25 tracks, calibrated DR, look 300px, $\beta=0.0$, 200k | 100.0% comp, 398.50 steps, 0% crash | 100.0% comp (70/70), 397.20 steps at $W=20$ | 0.0% comp (0/30; 27 crashes, 3 timeouts) | Not tested | **Rejected**: Combined lookahead cut and steering chatter failure. |
| **Run E (Ablation)** (`runs/ablation_E_noDR_look200_steerOn/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | 25 tracks, no DR, look 200px, $\beta=0.01$, 200k steps | 100.0% comp, 395.00 steps, 0% crash | 100.0% comp (70/70), but severely slowed (435.60 at $W=20$) | 30.0% comp (9/30; 7/10 at 0.50, 2/10 at 0.35) | 100.0% (20/20 held-out) | **Rejected**: Lacks DR; suffers severe narrow corridor pace degradation (+59.2 steps at $W=25$). |
| **Run F (Ablation)** (`runs/ablation_F_noDR_look200_steerOff/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | 25 tracks, no DR, $\beta=0.0$, 200k steps | 100.0% comp, 395.20 steps, 0% crash | 97.1% comp (68/70; 2 timeouts at $W=25, 20$) | 10.0% comp (3/30; 27 crashes) | Not tested | **Rejected**: Lacks DR and steering penalty; incomplete laps. |
| **Run G (Ablation)** (`runs/ablation_G_noDR_look300_steerOn/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 300$px) | 25 tracks, no DR, look 300px, $\beta=0.01$, 200k steps | 100.0% comp, 393.50 steps, 0% crash | 100.0% comp (70/70), 419.00 steps at $W=20$ | 23.3% comp (7/30; 20 crashes, 3 timeouts) | Not tested | **Rejected**: Lacks DR; corridor slowdown; high crash rate. |
| **Run H (Ablation)** (`runs/ablation_H_noDR_look300_steerOff/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 300$px) | 25 tracks, no DR, look 300px, $\beta=0.0$, 200k steps | 100.0% comp, 401.20 steps, 0% crash | Catastrophic collapse (30% comp at $W=20$, 4 crashes, 3 touts) | 23.3% comp (7/30; 11 crashes, 12 timeouts) | Not tested | **Rejected**: Catastrophic collapse in narrow corridors (30% completion). |
| **DR Full Multi-Track** (`runs/dr_multitrack_s42_full/`) | SAC MLP [128, 128], obs_dim=9 (5 rays $\times 200$px) | 20 tracks, full uncalibrated DR (`min_r` down to 50), 200k | 100.0% comp, 396.60 steps, 0% crash | 100.0% comp (70/70), 396.10 steps at $W=20$ | 3.3% comp (1/30; 1/10 at 0.50, 0/10 at 0.35/0.20) | Not tested | **Rejected**: Uncalibrated geometric bounds degrade acute cornering. |
| **9-Ray Deep Lookahead** (`runs/exp3_demos_s42_9ray_d300/`) | SAC MLP [128, 128], obs_dim=13 (9 rays $\times 300$px) | Single track (seed 101), fixed width, no DR, 200k steps | 100.0% comp on standard seeds | Incompatible obs space (13 dims), not trained with DR | 0.0% comp (0/20 on factors 0.50, 0.35, 0.20) | Not tested | **Rejected**: Obs space incompatible; no DR; 0% hairpin completion. |

---

## 3. Specifications & Configuration Contract

### 3.1 Network Architecture
- **Model Class**: `SACActor`
- **Input Dimension**: `obs_dim = 9`
- **Hidden Layers**: Two fully-connected linear layers with 128 units each:
  - Layer 1: `nn.Linear(9, 128)` followed by `nn.ReLU()`
  - Layer 2: `nn.Linear(128, 128)` followed by `nn.ReLU()`
- **Output Heads**:
  - `mean`: `nn.Linear(128, 2)`
  - `log_std`: `nn.Linear(128, 2)`, clamped to $[-20, 2]$
- **Action Squashing**: `torch.tanh(mean)` for continuous bounded control in $[-1.0, 1.0]^2$
- **Total Trainable Parameters**: 18,310 parameters

### 3.2 Observation Space Contract ($\mathbf{s} \in \mathbb{R}^9$)
1. `s[0]` — Forward Velocity: $v \in [-2.0, 4.0]\text{ px/step}$
2. `s[1]` — Heading Error: $\theta_{\text{err}} \in [-180.0^\circ, 180.0^\circ]$ (relative to nearest track tangent)
3. `s[2]` — Signed Centerline Distance: $d_{\text{center}} \in (-\infty, \infty)\text{ px}$
4. `s[3]` — Slip channel: kinematic yaw-rate echo, computed each step (equals −Δheading), not tyre slip. *(Corrected after the independent audit, `docs/evaluation/BIC_INDEPENDENT_AUDIT.md` Q6; previously described as 0.0.)*
5. `s[4]` — Left Lateral Ray: $-90.0^\circ$ (range $[0.0, 1.0]$, max distance $200.0\text{ px}$)
6. `s[5]` — Left Diagonal Ray: $-45.0^\circ$ (range $[0.0, 1.0]$, max distance $200.0\text{ px}$)
7. `s[6]` — Center Forward Ray: $0.0^\circ$ (range $[0.0, 1.0]$, max distance $200.0\text{ px}$)
8. `s[7]` — Right Diagonal Ray: $+45.0^\circ$ (range $[0.0, 1.0]$, max distance $200.0\text{ px}$)
9. `s[8]` — Right Lateral Ray: $+90.0^\circ$ (range $[0.0, 1.0]$, max distance $200.0\text{ px}$)

### 3.3 Action Space Contract ($\mathbf{a} \in [-1.0, 1.0]^2$)
1. `a[0]` — Steering Command: Mapped linearly to vehicle front wheel steering angle $\delta \in [-30.0^\circ, 30.0^\circ]$
2. `a[1]` — Throttle / Brake Command:
   - $u_{\text{throttle}} > 0$: Acceleration ($\Delta v = 0.08 \cdot u_{\text{throttle}} - 0.03$)
   - $u_{\text{throttle}} < 0$: Braking ($\Delta v = 0.08 \cdot u_{\text{throttle}} - 0.03 = -0.11\text{ px/step}$ at max reverse)

### 3.4 Domain Randomization Configuration (`calibrated`)
```json
{
  "enabled": true,
  "ranges": {
    "width": [25.0, 70.0],
    "min_radius": [65.0, 95.0],
    "base_r": [200.0, 260.0],
    "n_ctrl": [7, 11],
    "friction": [0.02, 0.045],
    "acceleration": [0.075, 0.095]
  }
}
```

---

## 4. Empirical Performance & Reproducibility Verification

Reproducibility was verified by executing 3 independent repetitions of identical evaluation protocols using deterministic policy inference ($\tanh(\mu(s))$).

### 4.1 Evaluation Telemetry Across Repetitions ($N=390$ Total Rollouts)

| Benchmark Regime | Target Geometry | Repetition 1 | Repetition 2 | Repetition 3 | 3-Run Mean Lap Time | Cross-Run Discrepancies |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Standard Held-Out** | Width 70 px (`TEST_SEEDS` 1001–1010) | 10/10 (398.20 steps) | 10/10 (398.20 steps) | 10/10 (398.20 steps) | **398.20 steps** | **0 mismatches** (100% identical) |
| **Narrow Corridor** | Width 70 px | 10/10 (398.20 steps) | 10/10 (398.20 steps) | 10/10 (398.20 steps) | 398.20 steps | 0 mismatches |
| | Width 40 px | 10/10 (396.90 steps) | 10/10 (396.90 steps) | 10/10 (396.90 steps) | 396.90 steps | 0 mismatches |
| | Width 36 px | 10/10 (396.30 steps) | 10/10 (396.30 steps) | 10/10 (396.30 steps) | 396.30 steps | 0 mismatches |
| | Width 32 px | 10/10 (395.70 steps) | 10/10 (395.70 steps) | 10/10 (395.70 steps) | 395.70 steps | 0 mismatches |
| | Width 28 px | 10/10 (394.90 steps) | 10/10 (394.90 steps) | 10/10 (394.90 steps) | 394.90 steps | 0 mismatches |
| | Width 25 px | 10/10 (395.00 steps) | 10/10 (395.00 steps) | 10/10 (395.00 steps) | 395.00 steps | 0 mismatches |
| | Width 20 px | 10/10 (394.60 steps) | 10/10 (394.60 steps) | 10/10 (394.60 steps) | **394.60 steps** | 0 mismatches |
| **Hairpin Pinch** | Pinch $f=0.50$ (Mild) | 5/10 (404.00 steps) | 5/10 (404.00 steps) | 5/10 (404.00 steps) | 404.00 steps | 0 mismatches |
| | Pinch $f=0.35$ (Severe) | 2/10 (429.50 steps) | 2/10 (429.50 steps) | 2/10 (429.50 steps) | 429.50 steps | 0 mismatches |
| | Pinch $f=0.20$ (Extreme) | 0/10 (N/A) | 0/10 (N/A) | 0/10 (N/A) | N/A | 0 mismatches |
| **Perturbed Starts** | Randomized pose/speed/heading | 20/20 (372.65 steps) | 20/20 (374.30 steps) | 20/20 (373.15 steps) | **373.37 steps** | 100% completion across all runs |

### 4.2 Cross-Repetition and Historical Consistency Findings
- **Deterministic Reproducibility**: Across all 110 deterministic benchmark episodes ($10 \text{ std} + 70 \text{ narrow} + 30 \text{ hairpin}$), Repetition 1, 2, and 3 produced **zero outcome mismatches, zero step count differences, and zero progress deviations**.
- **Historical Consistency**: Comparison against the historical ablation matrix (`analysis/results/ablation_episodes.csv`, Condition A) confirmed a **100% bit-for-bit match across all 110 episodes**.

---

## 5. Strengths & Capabilities

1. **Corridor Width Invariance**: The model maintains high speed and lane centering as road corridors shrink from 70 px to 20 px (car width is ~20 px).
2. **Smooth Racing Lines**: Steering smoothness penalty ($\beta_{\text{steer}} = 0.01$) eliminates high-frequency oscillation, preventing boundary wall strikes.
3. **Pinch Resistance**: Solves acute hairpins down to $f=0.35$ on held-out seeds 1007 and 1008, where all single-track baselines crash.
4. **Generalization Coverage**: Successfully traverses arbitrary procedural circuits generated with 7 to 11 control vertices.

---

## 6. Known Limitations & Failure Modes

1. **Extreme Non-Convex Curvature ($f=0.20$)**: At $f=0.20$, the local radius of curvature drops below $40\text{ px}$. Because the vehicle's physical minimum turning radius is $R_{\min} = 86.60\text{ px}$, traversing these turns requires heavy pre-braking and drifting maneuvers that a 5-ray sensor representation cannot anticipate.
2. **Coarse Sensor Angular Resolution**: The 5-ray sensor array has a $45^\circ$ angular gap between the $0^\circ$ and $\pm 45^\circ$ rays. On sudden switchbacks that do not cross the vehicle centerline, the model cannot detect the apex wall until within 35 px.
3. **Single Training Seed Provenance**: The checkpoint was trained with a single random seed (`seed=42`). While evaluated across diverse unseen tracks, inter-run training variance remains an area for future multi-seed scaling.

---

## 7. How to Reproduce Evaluation

To independently evaluate and verify the frozen generalist model:

```bash
# 1. Run the reproducibility evaluation script
PYTHONPATH=. ./venv/bin/python analysis/verify_generalist_reproducibility.py

# 2. Run the full pytest verification test suite
PYTHONPATH=. ./venv/bin/pytest tests/test_generalist_v1.py tests/
```
