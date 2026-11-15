# Empirical Evaluation: Generalization Performance vs. Training Track Pool Diversity

## Executive Summary

This study measures how the diversity of procedural training tracks ($N \in \{5, 10, 20, 25\}$) affects an agent's ability to generalize to held-out procedural circuits and out-of-distribution geometric stresses, compared against a fixed Single-Track baseline reference.

> [!NOTE]
> **Metric Unit Clarification**: Throughout this report and evaluation suite, "lap time" is measured in **discrete simulation steps** (environment steps taken to complete a lap, where fewer steps indicates higher speed / shorter path), not wall-clock seconds.

### Core Scientific Findings:
1. **Standard Procedural Tracks ($W=70$ px, seeds 1001–1010): Early Saturation.**
   All evaluated models (including the Single-Track baseline trained on a single fixed circuit) achieve **100.0% completion** on the 10 held-out test tracks, with tightly matched lap times ($395.10\text{ steps}$ to $399.50\text{ steps}$). Diversity is not required to complete standard medium-curvature tracks once basic vehicle control is learned.
2. **Narrow Corridor Generalization ($W \in [20, 70]$ px): Robustness via Domain Randomization.**
   The Single-Track baseline (trained with fixed $W=70$ px and no domain randomization) suffers severe speed degradation as width decreases ($398.90\text{ steps} \to 504.33\text{ steps}$) and drops to **60.0% completion** at $W=20$ px. In contrast, all multi-track models trained with calibrated domain randomization ($W \in [25, 70]$) maintain rapid lap times and achieve **100.0% completion** at extreme narrow width $W=20$ px (with a minor boundary oscillation anomaly at $W=36$ px for Pool 10: 80.0% completion).
3. **Acute Curvature & Pinched Hairpins ($f_{\text{pinch}} \in [0.20, 0.50]$): Diversity and Stochasticity.**
   - At **$f_{\text{pinch}} = 0.50$**, the Single-Track baseline (0.0%) and Pool 5 (0.0%) fail to complete any laps. **Pool 10 achieves 20.0% completion (2/10 laps, $49.54\%$ mean progress)**, solving test seeds 1007 and 1008. Pool 20 achieves 0.0% completion ($44.32\%$ mean progress). **Pool 25 achieves 50.0% completion (5/10 laps, $76.22\%$ mean progress)**, solving seeds 1002, 1003, 1006, 1007, and 1008.
   - At **$f_{\text{pinch}} = 0.35$**, only **Pool 25** completes any laps (**20.0% completion (2/10 laps)**, mean lap time $429.50\text{ steps}$, $47.84\%$ mean progress).
   - At **$f_{\text{pinch}} = 0.20$**, all models achieve **0.0% completion** ($34.05\% - 40.42\%$ mean progress), bounded by the car's physical minimum turning circle ($R_{\text{turn}} = 86.6$ px).

---

## 1. Experimental Setup & Controls

To rigorously measure track diversity, models are divided into:
1. **Single-Track Baseline Reference**: A model trained on a single fixed track (seed 101) without Domain Randomization for 45,000 steps.
2. **Calibrated DR Multi-Track Series**: Four models trained across pools of 5, 10, 20, and 25 distinct procedural circuits, holding algorithm, network architecture, observation space, random seed, and training budget strictly constant.

| Parameter | Single-Track Baseline Reference | Multi-Track Models (Pools 5, 10, 20, 25) |
| :--- | :--- | :--- |
| **Model Checkpoint** | `models/best_single_track_v2.pth` | `runs/dr_multitrack_s42_calibrated_<N>tracks/best_actor.pth` |
| **Training Budget** | 45,000 steps (early-stopped when solved) | 200,000 environment steps (matched budget) |
| **Training Seed** | Fixed seed 101 | $N$ procedural seeds: `TRAIN_SEEDS[:N]` from `[1..25]` |
| **Random Seed** | 42 | 42 |
| **Demonstrations** | 40 demonstrations | 40 demonstrations |
| **RL Algorithm** | SAC (entropy tuning, $\gamma=0.99$, $\tau=0.005$) | SAC (entropy tuning, $\gamma=0.99$, $\tau=0.005$) |
| **Network Architecture** | MLP: $[128, 128]$, ReLU, Gaussian policy | MLP: $[128, 128]$, ReLU, Gaussian policy |
| **Observation Dim** | 9 dims (speed, heading error, signed dist, slip, 5 lidar rays $\times 200$ px) | 9 dims (identical) |
| **Domain Randomization** | None (fixed track seed 101, friction 1.0, width 70) | Calibrated DR (`configs/dr_calibrated.json`): width $\in [25, 70]$, friction $\in [0.7, 1.3]$ |

### Evaluated Model Inventory & Paths
- **Single-Track Baseline (No DR)**: `models/best_single_track_v2.pth`
- **Pool 5 (Calibrated DR)**: `runs/dr_multitrack_s42_calibrated_5tracks/best_actor.pth`
- **Pool 10 (Calibrated DR)**: `runs/dr_multitrack_s42_calibrated_10tracks/best_actor.pth`
- **Pool 20 (Calibrated DR)**: `runs/dr_multitrack_s42_calibrated_20tracks/best_actor.pth`
- **Pool 25 (Calibrated DR)**: `runs/dr_multitrack_s42_calibrated_25tracks/best_actor.pth`

All held-out evaluations were conducted on strictly disjoint seeds:
$$\text{TEST\_SEEDS} = \{1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010\}$$

---

## 2. Empirical Evaluation Results

### 2.1 Standard Geometry Held-Out Tracks (`TEST_SEEDS 1001..1010`, $W=70$)

![Held-Out Completion Rate](/Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/results/diversity_heldout_completion.png)

![Held-Out Lap Time Distribution](/Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/results/diversity_heldout_laptime.png)

| Model | Total Laps | Completed | Completion Rate | Mean Lap Time (steps) | Std Lap Time (steps) | Mean Progress |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Track Baseline (No DR)** | 10 | 10 | **100.0%** | $398.90$ | $\pm 7.33$ | $100.19\%$ |
| **Pool 5 (Calibrated DR)** | 10 | 10 | **100.0%** | $395.10$ | $\pm 7.08$ | $100.10\%$ |
| **Pool 10 (Calibrated DR)** | 10 | 10 | **100.0%** | $399.50$ | $\pm 6.95$ | $100.12\%$ |
| **Pool 20 (Calibrated DR)** | 10 | 10 | **100.0%** | $398.10$ | $\pm 6.59$ | $100.14\%$ |
| **Pool 25 (Calibrated DR)** | 10 | 10 | **100.0%** | $398.20$ | $\pm 6.93$ | $100.14\%$ |

**Takeaway:** On standard medium procedural tracks ($W=70$ px), generalization saturates immediately. A single-track policy drives just as well on held-out smooth tracks as an agent trained on 25 tracks.

---

### 2.2 Narrow Corridor Stress Test ($W \in [70, 40, 36, 32, 28, 25, 20]$ px)

![Narrow Track Generalization Completion](/Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/results/diversity_narrow_completion.png)

#### Completion Rates Across Corridor Widths:
| Model | $W=70$ | $W=40$ | $W=36$ | $W=32$ | $W=28$ | $W=25$ | $W=20$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Track Baseline (No DR)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **60.0%** |
| **Pool 5 (Calibrated DR)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Pool 10 (Calibrated DR)** | 100.0% | 100.0% | **80.0%** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Pool 20 (Calibrated DR)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Pool 25 (Calibrated DR)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |

#### Mean Lap Time for Completed Laps (simulation steps):
| Model | $W=70$ | $W=40$ | $W=36$ | $W=32$ | $W=28$ | $W=25$ | $W=20$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Track Baseline (No DR)** | $398.90$ | $415.00$ | $425.90$ | $432.30$ | $448.40$ | $464.40$ | $504.33$ |
| **Pool 5 (Calibrated DR)** | $395.10$ | $396.90$ | $397.10$ | $396.90$ | $396.30$ | $396.80$ | $395.80$ |
| **Pool 10 (Calibrated DR)** | $399.50$ | $397.80$ | $396.62$ | $395.40$ | $395.10$ | $395.30$ | $394.30$ |
| **Pool 20 (Calibrated DR)** | $398.10$ | $413.20$ | $414.50$ | $413.90$ | $417.90$ | $413.60$ | $420.10$ |
| **Pool 25 (Calibrated DR)** | $398.20$ | $396.90$ | $396.30$ | $395.70$ | $394.90$ | $395.00$ | $394.60$ |

**Key Observations:**
1. **The DR Benefit:** While the Single-Track baseline slows down monotonically as width decreases (from $398.90\text{ steps}$ up to $504.33\text{ steps}$ due to cautious wall-proximity reactions) and crashes on 40% of laps at $W=20$ px, all DR-trained models complete 100% of laps at $W=20$ px.
2. **Speed Profiles Under Narrow Widths:** Pool 5, Pool 10, and Pool 25 maintain high speed (~$394.5 - 397\text{ steps}$), while Pool 20 adopts a slightly more conservative speed profile (~$413 - 420\text{ steps}$), achieving 100% completion across all widths.
3. **Width 36 Anomaly:** Pool 10 experienced two crashes at $W=36$ (seeds 1001 and 1002), resulting in an 80% completion rate at that width before recovering to 100% on narrower tracks.

---

### 2.3 Acute Curvature & Pinched Hairpins

![Hairpin Benchmarks Progress](/Users/prateekraj/Desktop/Ideas/RL Racing Agent/analysis/results/diversity_hairpin_progress.png)

#### Detailed Hairpin Performance Metrics (Recomputed from Episode CSV):
| Model | Pinch $0.50$ (Comp % / Prog) | Pinch $0.35$ (Comp % / Prog) | Pinch $0.20$ (Comp % / Prog) |
| :--- | :---: | :---: | :---: |
| **Single-Track Baseline (No DR)** | 0.0% / $40.25\%$ (Med: $41.85\%$) | 0.0% / $38.52\%$ (Med: $40.66\%$) | 0.0% / $36.75\%$ (Med: $39.07\%$) |
| **Pool 5 (Calibrated DR)** | 0.0% / $43.69\%$ (Med: $41.07\%$) | 0.0% / $42.05\%$ (Med: $40.20\%$) | 0.0% / $40.42\%$ (Med: $39.62\%$) |
| **Pool 10 (Calibrated DR)** | **20.0%** / **$49.54\%$** (Med: $49.60\%$) | 0.0% / $37.08\%$ (Med: $35.63\%$) | 0.0% / $34.05\%$ (Med: $33.14\%$) |
| **Pool 20 (Calibrated DR)** | 0.0% / $44.32\%$ (Med: $41.94\%$) | 0.0% / $42.13\%$ (Med: $41.13\%$) | 0.0% / $40.33\%$ (Med: $40.15\%$) |
| **Pool 25 (Calibrated DR)** | **50.0%** / **$76.22\%$** (Med: $96.23\%$) | **20.0%** / **$47.84\%$** (Med: $48.95\%$) | 0.0% / $36.40\%$ (Med: $34.82\%$) |

#### Per-Seed Hairpin Anatomy at Pinch 0.50:
- **Seed 1007**: Both **Pool 10** (413 steps, 100.2% progress) and **Pool 25** (409 steps, 100.2% progress) completed cleanly! Single-Track Baseline crashed at 80.5% progress (step 350), Pool 5 crashed at 79.2% progress (step 335), and Pool 20 crashed at 80.4% progress (step 345).
- **Seed 1008**: Both **Pool 10** (403 steps, 100.1% progress) and **Pool 25** (394 steps, 100.1% progress) completed cleanly! Single-Track Baseline crashed at 30.5% progress (step 155), Pool 5 crashed at 28.6% progress (step 145), and Pool 20 crashed at 29.1% progress (step 151).
- **Seeds 1002, 1003, 1006**: Also completed cleanly by **Pool 25** (398 steps, 399 steps, 420 steps).
- **Seed 1005**: All models failed (Pool 10 and Pool 25 timed out near the start line; others crashed at steps 47–62).

---

## 3. Scientific Analysis & Audit of Discrepancies

### 1. Cause of Previous Discrepancies & Metric Unit Verification
- **Lap Time Unit Verification**: In earlier draft reports, lap times were casually labeled with the unit "s" (seconds). Verification against `env/environment.py`, `env/car.py`, and `analysis/eval_diversity_scaling.py` confirms that the metric recorded as `lap_time` is the **integer count of discrete simulation steps** taken from reset until `lap_completed` is signaled (`steps = step + 1` in the action loop). At maximum speed ($v = 4.0\text{ px/step}$), traversing an $\approx 1500\text{ px}$ circuit takes $\approx 395\text{ steps}$. At 60 Hz simulation time ($dt = 1/60\text{ s}$ per step), 395 steps equates to $\approx 6.58$ seconds of simulated driving time. All labels across the report and figures have been corrected to `steps` / `simulation steps` without altering numeric values.
- **Omission of Pool 10 in Discussion**: Previous draft summaries stated that models below $N=25$ collapsed to 0% completion. This omitted the fact that Pool 10 completed 2 out of 10 tracks at pinch 0.50, solving the exact same tracks (seeds 1007 and 1008) that Pool 25 completed.
- **Conflation of Single-Track Baseline with "Pool 1"**: Previous text referred to Single-Track v2 as "Pool 1", implying it was trained under the same calibrated DR protocol and 200k steps budget as the other models. It has now been cleanly separated as an external baseline reference (fixed seed 101, fixed width 70, no DR, 45k steps).
- **Improper Categorization of Pool 20**: Grouping Pool 20 (trained on 20 distinct tracks) as a "low-diversity model" alongside Pool 1 and Pool 5 was an artificial classification. Pool 20 represents 80% of the maximum 25-track pool.
- **Narrow Lap Time Reporting**: Numbers in the previous narrow lap time table were inadvertently copied from Pool 25 rather than computed from the CSV. Every number in Table 2.2 is now directly recomputed from `analysis/results/diversity_scaling_episodes.csv` and matches `diversity_scaling_summary.json` with 0 mismatches.
- **Plot 3 Width 36 Omission**: Previous code for Plot 3 omitted $W=36$ px (`widths = [70, 40, 32, 28, 25, 20]`), obscuring the 80% completion point for Pool 10. All 7 evaluated widths are now rendered.

### 2. Does Generalization Scale Monotonically with Pool Size?
**No, generalization does not scale monotonically across all difficulties:**
- On standard procedural tracks ($W=70$), generalization saturates at $N=1$ (100% completion across all models).
- On narrow corridors ($W \in [20, 70]$), domain randomization provides width-invariance, yielding 100% completion for all DR pool sizes down to $W=20$ px.
- On extreme hairpins ($f_{\text{pinch}} = 0.50$), performance fluctuates: Pool 5 achieves 0%, Pool 10 achieves 20%, Pool 20 achieves 0%, and Pool 25 achieves 50%.

### 3. Key Experimental Limitation: Single-Seed Training
A fundamental limitation of this evaluation is that **each pool size was trained using only one random seed (`seed=42`)**:
- In deep reinforcement learning, a single seed produces run-specific exploration trajectories and replay buffer sampling.
- The difference between Pool 10 (20%), Pool 20 (0%), and Pool 25 (50%) cannot be attributed solely to pool size; training-run stochasticity plays a substantial role when evaluating on $M=10$ test tracks.
- While Pool 25 demonstrates that high diversity enables learning acute cornering strategies that lower pools struggled to find, multi-seed training (e.g., 3–5 seeds per pool size) would be required to establish statistical significance.

---

## 4. Reproduction & Verification

### Run Verification Suite
```bash
./venv/bin/pytest test_diversity_evaluation_consistency.py
```

### Regenerate Plots from Data
```bash
./venv/bin/python analysis/plot_diversity_scaling.py
```
