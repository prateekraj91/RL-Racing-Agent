# Statistical Ablation Matrix: Factorial Analysis of Domain Randomization, Lookahead Distance, and Steering Smoothness

## Executive Summary

This document compiles the rigorous statistical ablation matrix for the **RL Racing Agent**, evaluating the 2×2×2 full factorial experiment across eight systematically trained configurations (**Runs A through H**). 

The analysis evaluates the independent main effects and interactive contributions of three design factors:
1. **Domain Randomization (DR)**: Calibrated procedural physics and track variation vs. Fixed nominal physics.
2. **Lidar Lookahead Distance**: Long (300 px) vs. Short (200 px).
3. **Steering Smoothness Reward Penalty**: Enabled ($\beta_{\text{steer}} = 0.01$) vs. Disabled ($\beta_{\text{steer}} = 0.00$).

The evaluation is conducted on the full dataset of **880 deterministic rollouts** across 10 matched held-out track seeds (`TEST_SEEDS` 1001–1010) spanning three distinct benchmark regimes:
- **Standard Tracks** (Nominal width $W = 70\text{ px}$).
- **Narrow Corridors** (Stress sweep across widths $W \in [70, 40, 36, 32, 28, 25, 20]\text{ px}$).
- **Hairpins** (Curvature pinch sweep across factors $f \in [0.50, 0.35, 0.20]$).

All statistics and factorial contrasts are computed directly from per-episode rollout telemetry without artificial imputation.

```
+-----------------------------------------------------------------------------------------------------------------+
|                                          KEY FACTORIAL FINDINGS AT A GLANCE                                     |
+--------------------------+--------------------------------------+-----------------------------------------------+
| Factor                   | Lap Time (Simulation Steps)          | Off-Track Failure Rate (%-pts)                |
+--------------------------+--------------------------------------+-----------------------------------------------+
| Domain Randomization     | Nominal: +1.75 steps (p = 0.00018)   | Narrow (W=20): -10.0 pp (p = 0.0368)          |
| (DR On vs Off)           | Narrow W=25: -38.31 steps (p < 1e-5) | Hairpin (f=0.50): +27.5 pp (p = 0.0243)       |
|                          | Massive speedup in tight corridors   | Hairpin (f=0.20): -15.0 pp (p = 0.0051)       |
+--------------------------+--------------------------------------+-----------------------------------------------+
| Lidar Lookahead          | Nominal: +1.30 steps (p = 0.0078)    | Narrow (W=20): +10.0 pp (p = 0.0368)          |
| (300 px vs 200 px)       | Narrow W=25: -4.03 steps (p = 0.004) | Hairpin (f=0.50): +12.5 pp (p = 0.0522)       |
|                          | Marginal impact across corridor widths| Consistently increases off-track crashes      |
+--------------------------+--------------------------------------+-----------------------------------------------+
| Steering Smoothness      | Nominal: -1.95 steps (p = 0.00054)   | Narrow (W=20): -10.0 pp (p = 0.0368)          |
| Penalty (0.01 vs 0.00)   | Narrow W=28: -7.40 steps (p = 0.001) | Hairpin (f=0.50): -2.5 pp (Inconclusive)      |
|                          | Consistently faster & smoother lines | Critical for preventing high-slip spinouts    |
+--------------------------+--------------------------------------+-----------------------------------------------+
```

---

## 1. Experimental Design Matrix & Conditions

The ablation experiment follows a completely balanced, orthogonal $2^3$ full factorial design. All conditions share identical algorithmic configurations:
- **Algorithm**: Soft Actor-Critic (SAC) with automatic entropy temperature tuning.
- **Base Architecture**: MLP $(64 \times 64)$ policy with Gaussian head, learning rate $3 \times 10^{-4}$, batch size 64.
- **Training Budget**: 200,000 environment steps per configuration on procedural training pool (`TRAIN_SEEDS` 1–25).
- **Demonstration Replay**: 40 clean and perturbed expert demonstration episodes (17,014 transitions) protected from eviction.
- **Training Seed**: Fixed at seed 42 across all eight models.

| Condition | DR Setting | Lidar Lookahead | Steer Penalty ($\beta_{\text{steer}}$) | Run Directory | Evaluated Checkpoint |
|---|---|---|---|---|---|
| **Run A** | **On** (Calibrated) | **200 px** (Short) | **0.01** (On) | `runs/ablation_A_DR_look200_steerOn/` | `best_actor.pth` (200k steps) |
| **Run B** | **On** (Calibrated) | **200 px** (Short) | **0.00** (Off) | `runs/ablation_B_DR_look200_steerOff/` | `best_actor.pth` (200k steps) |
| **Run C** | **On** (Calibrated) | **300 px** (Long) | **0.01** (On) | `runs/ablation_C_DR_look300_steerOn/` | `best_actor.pth` (200k steps) |
| **Run D** | **On** (Calibrated) | **300 px** (Long) | **0.00** (Off) | `runs/ablation_D_DR_look300_steerOff/` | `best_actor.pth` (200k steps) |
| **Run E** | **Off** (Fixed) | **200 px** (Short) | **0.01** (On) | `runs/ablation_E_noDR_look200_steerOn/` | `best_actor.pth` (200k steps) |
| **Run F** | **Off** (Fixed) | **200 px** (Short) | **0.00** (Off) | `runs/ablation_F_noDR_look200_steerOff/` | `best_actor.pth` (200k steps) |
| **Run G** | **Off** (Fixed) | **300 px** (Long) | **0.01** (On) | `runs/ablation_G_noDR_look300_steerOn/` | `best_actor.pth` (200k steps) |
| **Run H** | **Off** (Fixed) | **300 px** (Long) | **0.00** (Off) | `runs/ablation_H_noDR_look300_steerOff/` | `best_actor.pth` (200k steps) |

---

## 2. Metric Definitions & Failure Taxonomy

### 2.1 Lap Time
- **Simulation Steps**: Measured in discrete environment simulation steps (where 1 step = $1/60$ second physics update).
- **Completed Episodes Only**: Lap time is computed strictly on episodes where the agent completed 100% of the circuit (`completed == True`).
- **No Artificial Imputation**: Incomplete episodes (crashes, timeouts, stalls) receive **no artificial penalty lap time**. Assigning arbitrary penalty times (e.g., assigning 700 steps or $\infty$ to a crash) distorts statistical variance, confounds velocity with survival, and violates standard statistical practice.
- **Reporting Metrics**: For completed laps, we report the sample mean, median, standard deviation ($s$, with $N-1$ denominator), minimum, maximum, and sample size ($N_{\text{completed}}$).
- **Exposure of Survivor Bias**: Because lap time can only be evaluated for surviving vehicles, completion rates are reported alongside lap times. When completion rates drop, lap time reflects a heavily censored subset of easier circuits.

### 2.2 Off-Track Failure Rate & Failure Taxonomy
Inspection of the simulation codebase (`env/environment.py`, lines 218–221, and `env/track.py`, lines 123–130) establishes the exact termination logic:
```python
crashed = not self.track.is_on_track(self.car.x, self.car.y)
```
Where `is_on_track` evaluates whether the car's Euclidean distance from the circuit spline centerline exceeds half the track width ($d_{\text{min}} > W / 2$).

Therefore, the simulator defines three mutually exclusive and collectively exhaustive terminal outcomes:
1. **Completed Lap** (`completed == True`): `lap_progress >= 1.0` within `max_steps = 700`.
2. **Off-Track Crash** (`crashed == True`): The car exited the track boundary / collided with the wall boundary.
3. **Timeout / Incomplete** (`timeout == True`): The car remained within track boundaries for 700 steps without completing the lap.
   - **Stationary Stall Subcategory**: Timeouts where the vehicle failed to move or stalled at the start line (`max_speed < 0.05 px/step` or $|\text{progress}| < 0.005$). In our dataset, 9 episodes match this stationary criterion (all occurring on seed 1005 under extreme hairpin pinch).
   - **Progress / Wandering Timeout**: Timeouts where the vehicle continued to drive or oscillate within the road boundaries but failed to cross the finish line before step 700.

$$\text{Off-Track Failure Rate} = \frac{\text{Off-Track Crashes}}{\text{Total Evaluation Episodes}}$$

$$\text{Total Failure Rate} = \frac{\text{Crashes} + \text{Timeouts}}{\text{Total Evaluation Episodes}} = 1.0 - \text{Completion Rate}$$

> [!WARNING]
> **Critical Survivor / Censoring Phenomenon**:
> An agent that spins in place or crawls safely inside the track boundaries avoids crossing the boundary (`crashed == False`), but still fails the benchmark via timeout. For example, at Hairpin $f=0.20$, **Condition B achieves a 0.0% off-track crash rate**, but its **completion rate is 0.0%** (all 10 episodes timed out). Treating off-track crash rate in isolation without tracking completion and timeout rates would misinterpret complete vehicle paralysis as optimal driving.

---

## 3. Paired Statistical Methodology & Factorial Contrasts

### 3.1 Contrast Direction Conventions
To maintain strict interpretability across all metrics:
- **Lap Time Contrast**: **Negative change means faster** (fewer simulation steps required to complete the lap).
- **Off-Track Failure Rate Contrast**: **Negative change means fewer failures** (reduction in percentage points of wall-boundary crashes).

### 3.2 Factorial Model Formulation
For each benchmark slice with $N$ matched circuit seeds $s \in \{1001, \dots, 1010\}$, the $2^3$ design provides an 8-dimensional observation vector $\mathbf{y}(s) = [y_A(s), y_B(s), \dots, y_H(s)]^T$.

Using standard orthogonal Yates contrast vectors:
$$\mathbf{x}_{\text{DR}} = [+1, +1, +1, +1, -1, -1, -1, -1]^T$$
$$\mathbf{x}_{\text{Look}} = [-1, -1, +1, +1, -1, -1, +1, +1]^T$$
$$\mathbf{x}_{\text{Steer}} = [+1, -1, +1, -1, +1, -1, +1, -1]^T$$

The per-circuit factorial contrasts are:
$$\Delta_{\text{DR}}(s) = \frac{1}{4} \mathbf{x}_{\text{DR}}^T \mathbf{y}(s) = \frac{(y_A + y_B + y_C + y_D) - (y_E + y_F + y_G + y_H)}{4}$$
$$\Delta_{\text{Look}}(s) = \frac{1}{4} \mathbf{x}_{\text{Look}}^T \mathbf{y}(s) = \frac{(y_C + y_D + y_G + y_H) - (y_A + y_B + y_E + y_F)}{4}$$
$$\Delta_{\text{Steer}}(s) = \frac{1}{4} \mathbf{x}_{\text{Steer}}^T \mathbf{y}(s) = \frac{(y_A + y_C + y_E + y_G) - (y_B + y_D + y_F + y_H)}{4}$$

Two-way interaction vectors:
$$\mathbf{x}_{\text{DR} \times \text{Look}} = \mathbf{x}_{\text{DR}} \odot \mathbf{x}_{\text{Look}}$$
$$\mathbf{x}_{\text{DR} \times \text{Steer}} = \mathbf{x}_{\text{DR}} \odot \mathbf{x}_{\text{Steer}}$$
$$\mathbf{x}_{\text{Look} \times \text{Steer}} = \mathbf{x}_{\text{Look}} \odot \mathbf{x}_{\text{Steer}}$$

Three-way interaction vector:
$$\mathbf{x}_{\text{3-way}} = \mathbf{x}_{\text{DR}} \odot \mathbf{x}_{\text{Look}} \odot \mathbf{x}_{\text{Steer}}$$

### 3.3 Paired Bootstrap & Hypothesis Testing
For each contrast vector across circuits:
1. **Paired Bootstrap Confidence Interval**: $B = 10,000$ paired bootstrap resamples of circuit indices with replacement (fixed seed 42). 95% CIs are computed via the percentile method: $[\hat{\theta}^*_{0.025}, \hat{\theta}^*_{0.975}]$.
2. **Paired Student's t-test**: Tests null hypothesis $H_0: \mu_{\Delta} = 0$, evaluating $t = \bar{\Delta} / \text{SE}(\Delta)$ with $df = N_{\text{circuits}} - 1$. Two-sided $p$-values are computed via exact Gauss-Legendre quadrature of the Student's t CDF.
3. **Exact Wilcoxon Signed-Rank Permutation Test**: Non-parametric test of symmetry about zero. Evaluated via exact enumeration of all $2^N$ sign assignments for $N \le 10$, handling rank ties via standard midpoint averaging.

> [!IMPORTANT]
> **Single Training Seed Limitation**:
> Each experimental condition was trained from a **single random seed (42)**. Consequently, all reported confidence intervals, standard errors, and hypothesis tests measure **variation across held-out track geometries**, conditioned on the specific converged weights of that single training run. They **do not measure variance across independent RL training initializations**. True causal assertions regarding training stability require multi-seed replication.

---

## 4. Comprehensive Benchmark Results

### 4.1 Benchmark 1: Standard Held-Out Circuits ($W = 70\text{ px}$)

Evaluated across 10 held-out procedural tracks with nominal racing width ($cx=400, cy=300, base\_r=250, n\_ctrl=10, min\_radius=80, width=70$).

#### Condition-Level Performance (Standard $W=70\text{ px}$)
| Condition | DR | Lookahead | Steer Penalty | Completed / Total | Completion Rate | Crashed | Timeout | Mean Lap Time (steps) | Median LT | Std LT | Min LT | Max LT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Run A** | On | 200 px | 0.01 | 10 / 10 | 100.0% | 0 (0%) | 0 (0%) | 398.20 | 397.5 | 7.30 | 387 | 408 |
| **Run B** | On | 200 px | 0.00 | 10 / 10 | 100.0% | 0 (0%) | 0 (0%) | 397.40 | 396.5 | 7.29 | 386 | 407 |
| **Run C** | On | 300 px | 0.01 | 10 / 10 | 100.0% | 0 (0%) | 0 (0%) | 397.80 | 397.5 | 7.32 | 386 | 407 |
| **Run D** | On | 300 px | 0.00 | 10 / 10 | 100.0% | 0 (0%) | 0 (0%) | 398.50 | 399.0 | 7.34 | 386 | 408 |
| **Run E** | Off | 200 px | 0.01 | 10 / 10 | 100.0% | 0 (0%) | 0 (0%) | 395.00 | 393.5 | 7.41 | 384 | 406 |
| **Run F** | Off | 200 px | 0.00 | 10 / 10 | 100.0% | 0 (0%) | 0 (0%) | 395.20 | 394.5 | 7.08 | 384 | 405 |
| **Run G** | Off | 300 px | 0.01 | 10 / 10 | 100.0% | 0 (0%) | 0 (0%) | **393.50** | **392.5** | 7.21 | 382 | 403 |
| **Run H** | Off | 300 px | 0.00 | 10 / 10 | 100.0% | 0 (0%) | 0 (0%) | 401.20 | 401.0 | 6.73 | 387 | 410 |

#### Factorial Effects on Nominal Lap Time ($N = 10$ Paired Circuits)
| Factor / Interaction | Mean Effect (steps) | 95% Bootstrap CI | Std Error | Paired $t$-stat | $p$-value ($t$-test) | Wilcoxon Stat ($W$) | $p$-value (Wilcoxon) | Interpretation |
|---|---|---|---|---|---|---|---|---|
| **Main: Domain Randomization** | **+1.75** | [+1.20, +2.23] | 0.29 | +6.11 | **$p = 0.00018$** | 1.0 | **$p = 0.0039$** | Statistically significant slowdown (+0.44% lap time) |
| **Main: Lookahead (300 vs 200)** | **+1.30** | [+0.60, +2.02] | 0.38 | +3.39 | **$p = 0.0078$** | 4.0 | **$p = 0.0117$** | Significant marginal slowdown |
| **Main: Steering Penalty** | **-1.95** | [-2.67, -1.30] | 0.37 | -5.23 | **$p = 0.00054$** | 0.0 | **$p = 0.0020$** | **Consistently faster by 1.95 steps** |
| **Interaction: DR × Lookahead** | -0.95 | [-1.65, -0.35] | 0.35 | -2.71 | $p = 0.0236$ | 6.0 | $p = 0.0234$ | Sub-additive speedup |
| **Interaction: DR × SteerPenalty** | +2.00 | [+1.55, +2.60] | 0.30 | +6.71 | $p < 0.0001$ | 0.0 | $p = 0.0020$ | Antagonistic interaction |
| **Interaction: Look × Steer** | **-2.25** | [-2.77, -1.82] | 0.28 | -8.18 | **$p < 0.0001$** | 0.0 | **$p = 0.0020$** | Strong synergistic smoothing |
| **3-Way: DR × Look × Steer** | +1.50 | [+1.07, +2.00] | 0.26 | +5.83 | $p = 0.00026$ | 0.0 | $p = 0.0020$ | Higher-order modulation |

*Note on Off-Track Failure Rate on Standard Tracks*: Every condition achieved 0.0% crashes ($10/10$ completions). All failure contrasts are identically $0.0000$ ($p = 1.0$).

---

### 4.2 Benchmark 2: Narrow Corridor Stress Benchmark ($W \in [70, 40, 36, 32, 28, 25, 20]\text{ px}$)

In this suite, the track width is progressively restricted while maintaining the centerline trajectory and control vertices.

#### Condition-Level Performance Across Corridor Widths
```
Corridor Width Sweep: Lap Time (Mean ± Std, steps) & Completion Rate (%)
Condition      W=70 px        W=40 px        W=36 px        W=32 px        W=28 px        W=25 px        W=20 px
-------------------------------------------------------------------------------------------------------------------
Run A (DR)     398.2 (100%)   396.9 (100%)   396.3 (100%)   395.7 (100%)   394.9 (100%)   395.0 (100%)   394.6 (100%)
Run B (DR)     397.4 (100%)   397.0 (100%)   396.8 (100%)   396.5 (100%)   396.1 (100%)   396.2 (100%)   396.1 (100%)
Run C (DR)     397.8 (100%)   397.0 (100%)   396.5 (100%)   396.9 (100%)   395.9 (100%)   396.6 (100%)   395.1 (100%)
Run D (DR)     398.5 (100%)   397.6 (100%)   397.7 (100%)   397.2 (100%)   396.8 (100%)   396.9 (100%)   397.2 (100%)
Run E (noDR)   395.0 (100%)   406.0 (100%)   414.4 (100%)   416.0 (100%)   424.6 (100%)   454.2 (100%)   435.6 (100%)
Run F (noDR)   395.2 (100%)   408.0 (100%)   408.5 (100%)   410.3 (100%)   420.9 (100%)   424.3 (90%)    428.4 (90%)
Run G (noDR)   393.5 (100%)   402.4 (100%)   403.4 (100%)   407.7 (100%)   410.2 (100%)   411.7 (100%)   419.0 (100%)
Run H (noDR)   401.2 (100%)   410.5 (100%)   427.6 (100%)   418.4 (100%)   441.4 (100%)   449.1 (100%)   489.7 (30%)
```

#### Detailed Breakdown at Critical Thresholds ($W=25\text{ px}$ and $W=20\text{ px}$)
- **At $W = 25\text{ px}$**:
  - Runs A, B, C, D (DR-trained): 100% completion, mean lap time 395.0 to 396.9 steps.
  - Run E: 100% completion, but lap time degrades to 454.2 steps (+59.2 steps slower than Run A).
  - Run F: 90% completion (1 timeout on seed 1002; 0 crashes), lap time 424.3 steps.
  - Run G: 100% completion, lap time 411.7 steps.
  - Run H: 100% completion, lap time 449.1 steps.
- **At $W = 20\text{ px}$**:
  - Runs A, B, C, D (DR-trained): **100% completion, 0 crashes, 0 timeouts**, maintaining razor-sharp lap times (394.6 to 397.2 steps).
  - Run E: 100% completion, lap time 435.6 steps.
  - Run F: 90% completion (1 timeout on seed 1002; 0 crashes), lap time 428.4 steps.
  - Run G: 100% completion, lap time 419.0 steps.
  - **Run H (no DR, long lookahead, no steer penalty)**: **Suffers catastrophic collapse to 30.0% completion** (4 off-track crashes, 3 timeouts, only 3 completed laps). Completed laps crawl at 489.7 steps.

#### Factorial Contrasts on Lap Time Across Corridor Widths
| Width Level | $N_{\text{paired}}$ | DR Main Effect (steps) [95% CI] | Lookahead Effect [95% CI] | Steer Penalty Effect [95% CI] |
|---|---|---|---|---|
| $W = 70\text{ px}$ | 10 | **+1.75** [+1.20, +2.23] | +1.30 [+0.60, +2.02] | **-1.95** [-2.67, -1.30] |
| $W = 40\text{ px}$ | 10 | **-9.60** [-10.75, -8.43] | -0.10 [-0.55, +0.33] | **-2.70** [-3.23, -2.12] |
| $W = 36\text{ px}$ | 10 | **-16.65** [-18.90, -14.68] | +2.30 [+0.47, +3.85] | **-5.00** [-6.65, -3.07] |
| $W = 32\text{ px}$ | 10 | **-16.52** [-19.52, -14.12] | +0.42 [-0.98, +1.88] | -1.52 [-2.80, -0.05] |
| $W = 28\text{ px}$ | 10 | **-28.35** [-31.60, -25.20] | +1.95 [+0.22, +3.80] | **-7.40** [-10.15, -4.55] |
| $W = 25\text{ px}$ | 9 | **-38.31** [-42.42, -34.22] | **-4.03** [-5.75, -2.19] | -2.08 [-4.25, +0.19] |
| $W = 20\text{ px}$ | 3* | **-46.92** [-51.58, -42.42] | +12.75 [+10.83, +15.58] | **-16.25** [-19.50, -12.50] |

*\*Note on $W=20\text{ px}$ Lap Time Paired Sample*: Only 3 circuit seeds (1001, 1003, 1007) were completed by all eight configurations due to failures in Runs F and H. This estimate illustrates the survivor speed differential but carries small-sample uncertainty.

#### Factorial Contrasts on Off-Track Failure Rate at $W=20\text{ px}$ ($N=10$ Circuits)
| Factor / Interaction | Percentage Points | 95% Bootstrap CI | Paired $t$-stat | $p$-value ($t$-test) | Wilcoxon $p$-value | Conclusion |
|---|---|---|---|---|---|---|
| **DR (On vs Off)** | **-10.0 pp** | [-17.5 pp, -2.5 pp] | -2.45 | **$p = 0.0368$** | $p = 0.1250$ | **Reduces off-track failures by 10.0 pp** |
| **Lookahead (300 vs 200)** | **+10.0 pp** | [+2.5 pp, +17.5 pp] | +2.45 | **$p = 0.0368$** | $p = 0.1250$ | **Increases off-track failures by 10.0 pp** |
| **Steer Penalty (On vs Off)** | **-10.0 pp** | [-17.5 pp, -2.5 pp] | -2.45 | **$p = 0.0368$** | $p = 0.1250$ | **Reduces off-track failures by 10.0 pp** |
| **DR × Lookahead** | -10.0 pp | [-17.5 pp, -2.5 pp] | -2.45 | $p = 0.0368$ | $p = 0.1250$ | DR prevents Lookahead-induced crash |
| **DR × SteerPenalty** | +10.0 pp | [+2.5 pp, +17.5 pp] | +2.45 | $p = 0.0368$ | $p = 0.1250$ | Antagonistic buffering |
| **Lookahead × SteerPenalty** | -10.0 pp | [-17.5 pp, -2.5 pp] | -2.45 | $p = 0.0368$ | $p = 0.1250$ | Synergistic stabilization |
| **3-Way Interaction** | +10.0 pp | [+2.5 pp, +17.5 pp] | +2.45 | $p = 0.0368$ | $p = 0.1250$ | Joint buffering |

---

### 4.3 Benchmark 3: Hairpin Pinch Generalization ($f \in [0.50, 0.35, 0.20]$)

Hairpin pinch tracks introduce synthetic non-convex radius compressions ($r_{\text{local}} = r_{\text{nominal}} \times f$) along critical track vertices.

#### Condition-Level Performance Across Hairpin Severities
```
Hairpin Benchmark: Completed / Off-Track Crashes / Timeouts across 10 Seeds
Condition     f = 0.50 (Mild)              f = 0.35 (Severe)            f = 0.20 (Extreme)
---------------------------------------------------------------------------------------------
Run A (DR)    5 Comp / 5 Crash / 0 Tout    2 Comp / 6 Crash / 2 Tout    0 Comp / 9 Crash / 1 Tout
Run B (DR)    0 Comp / 5 Crash / 5 Tout    0 Comp / 2 Crash / 8 Tout    0 Comp / 0 Crash / 10 Tout
Run C (DR)    0 Comp / 10 Crash / 0 Tout   0 Comp / 10 Crash / 0 Tout   0 Comp / 10 Crash / 0 Tout
Run D (DR)    0 Comp / 9 Crash / 1 Tout    0 Comp / 9 Crash / 1 Tout    0 Comp / 9 Crash / 1 Tout
Run E (noDR)  7 Comp / 3 Crash / 0 Tout    2 Comp / 8 Crash / 0 Tout    0 Comp / 10 Crash / 0 Tout
Run F (noDR)  2 Comp / 8 Crash / 0 Tout    1 Comp / 9 Crash / 0 Tout    0 Comp / 10 Crash / 0 Tout
Run G (noDR)  5 Comp / 5 Crash / 0 Tout    2 Comp / 7 Crash / 1 Tout    0 Comp / 8 Crash / 2 Tout
Run H (noDR)  5 Comp / 2 Crash / 3 Tout    2 Comp / 3 Crash / 5 Tout    0 Comp / 6 Crash / 4 Tout
```

#### Detailed Failure Mechanics & Censoring in Hairpins
1. **Zero Imputation for Lap Time**: At $f = 0.50$, Runs B, C, and D completed **zero episodes**. At $f = 0.20$, **every single configuration achieved zero completions**. In accordance with strict statistical standards, **factorial contrasts on lap time cannot be computed for hairpin circuits**.
2. **The Run B Paradox (Timeout Censoring)**:
   - At $f = 0.20$, Run B has **0 off-track crashes** ($0.0\%$ failure rate), but **0 completed laps** ($10.0\%$ stationary stall, $90.0\%$ progress timeout).
   - In sharp contrast, Run A (identical to B except steering penalty is ON) has **9 off-track crashes** ($90.0\%$) and 1 timeout.
   - Without the steering penalty, Run B oscillates laterally inside the corner, enters a slow spin/loop within the track boundaries, and exhausts the 700-step horizon without ever striking the outer boundary.
3. **Long Lookahead Induces Catastrophic Boundary Collisions**:
   - Runs C and D (Lookahead 300 px with DR) suffer **90% to 100% crash rates across all three hairpin levels**.
   - Because 300 px rays penetrate through the apex to the opposing straight, the policy accelerates prematurely before corner exit, driving directly into the inside wall apex.

#### Factorial Contrasts on Off-Track Failure Rate in Hairpins ($N = 10$ Circuits)
| Benchmark Slice | Factor | Percentage Points | 95% Bootstrap CI | Paired $t$-stat | $p$-value ($t$-test) | Wilcoxon $p$-value | Status |
|---|---|---|---|---|---|---|---|
| **Hairpin $f=0.50$** | **DR** | **+27.5 pp** | [+7.5 pp, +45.0 pp] | +2.70 | **$p = 0.0243$** | **$p = 0.0469$** | Worsens crash rate (more aggressive speed profile) |
| | **Lookahead** | **+12.5 pp** | [+2.5 pp, +25.0 pp] | +2.24 | $p = 0.0522$ | $p = 0.1250$ | Worsens crash rate (borderline significance) |
| | **SteerPenalty** | -2.5 pp | [-25.0 pp, +20.0 pp] | -0.20 | $p = 0.8473$ | $p = 0.9219$ | Inconclusive on crashes (competing timeout mode) |
| **Hairpin $f=0.35$** | **DR** | +0.0 pp | [-15.0 pp, +20.0 pp] | 0.00 | $p = 1.0000$ | $p = 1.0000$ | Inconclusive |
| | **Lookahead** | +10.0 pp | [0.0 pp, +20.0 pp] | +1.81 | $p = 0.1039$ | $p = 0.2188$ | Weak trend toward higher crash rate |
| | **SteerPenalty** | **+20.0 pp** | [+2.5 pp, +35.0 pp] | +2.23 | $p = 0.0528$ | $p = 0.0859$ | Artifact of timeout censoring (see Run B) |
| **Hairpin $f=0.20$** | **DR** | **-15.0 pp** | [-22.5 pp, -7.5 pp] | -3.67 | **$p = 0.0051$** | **$p = 0.0312$** | Statistically significant crash reduction |
| | **Lookahead** | +10.0 pp | [0.0 pp, +20.0 pp] | +1.81 | $p = 0.1039$ | $p = 0.2188$ | Trend toward higher crashes |
| | **SteerPenalty** | **+30.0 pp** | [+20.0 pp, +40.0 pp] | +5.74 | **$p = 0.0002$** | **$p = 0.0039$** | Apparent crash increase due to Run B timeout |

---

## 5. Visualizations & Analytical Figures

The following publication-grade figures were generated from the empirical episode telemetry:

### Figure 1: Factorial Main Effects on Lap Time
![Main Effects on Lap Time](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/matrix_main_effects_laptime.png)
*Figure 1: Main effects of Domain Randomization (blue), Lookahead Distance (orange), and Steering Smoothness Penalty (green) on simulation lap time across corridor widths ($W=70$ to $W=20\text{ px}$). Negative values indicate faster lap times.*

### Figure 2: Factorial Main Effects on Off-Track Failure Rate
![Main Effects on Off-Track Failure Rate](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/matrix_main_effects_offtrack.png)
*Figure 2: Main effects on off-track wall collision rate across critical stress regimes. Negative values indicate fewer failures.*

### Figure 3: Corridor Width Progression & Robustness Collapse
![Corridor Width Progression](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/matrix_corridor_widths.png)
*Figure 3: (Left) Mean lap times for all eight configurations as corridor width restricts from 70 to 20 px. Models trained with DR maintain constant ~395-step lap times, while nominal models slow down dramatically. (Right) Off-track crash and timeout rates, highlighting the catastrophic failure of Run H at $W=20\text{ px}$.*

### Figure 4: Hairpin Outcome Breakdown (Exposing Survivor Bias)
![Hairpin Outcome Distribution](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/matrix_hairpin_difficulties.png)
*Figure 4: Outcome breakdown (completed, crashed, timeout/stalled) across hairpin pinch factors $f=0.50$, $0.35$, and $0.20$. Demonstrates why Run B achieves 0% crashes at $f=0.20$ solely through 100% timeouts.*

### Figure 5: 95% Bootstrap Confidence Intervals (Forest Plot)
![Confidence Intervals](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/matrix_effects_ci.png)
*Figure 5: Paired bootstrap 95% confidence intervals for main effects on lap time (left) and off-track failure rate (right).*

---

## 6. Compact Final Decision Matrix

The following table synthesizes the empirical evidence, categorizing each component's impact as **Improving**, **Worsening**, or **Inconclusive** across performance metrics and operational benchmarks.

| Component / Factor | Benchmark Domain | Metric: Lap Time | Metric: Off-Track Failure Rate | Metric: Overall Completion | Final Recommendation |
|---|---|---|---|---|---|
| **Domain Randomization (DR)** | **Standard Tracks ($W=70$)** | **Worsening** (+1.75 steps, $p=0.0002$) | **Neutral** (0.0 pp, all survive) | **Neutral** (100% completion) | Minor nominal speed tradeoff for massive generalization |
| | **Narrow Corridors ($W \le 40$)** | **Improving** (-9.6 to -46.9 steps, $p < 1e-4$) | **Improving** (-10.0 pp at $W=20$, $p=0.037$) | **Improving** (+20.0 pp completion at $W=20$) | **CRITICAL**: Eliminates corridor narrowing brittleness |
| | **Hairpin Pinch Curves** | **Undefined** (censored by non-completion) | **Mixed** (+27.5 pp at 0.50; -15.0 pp at 0.20) | **Inconclusive** (-35.0 pp at 0.50; neutral at 0.20) | Requires targeted curvature DR, not generic road scaling |
| **Short Lookahead (200 px vs 300 px)** | **Standard Tracks ($W=70$)** | **Improving** (-1.30 steps, $p=0.0078$) | **Neutral** (0.0 pp) | **Neutral** (100% completion) | 200 px is slightly faster on nominal tracks |
| | **Narrow Corridors ($W \le 40$)** | **Inconclusive** (oscillates $\pm 2$ steps) | **Improving** (-10.0 pp crashes at $W=20$) | **Improving** (+15.0 pp completion at $W=20$) | 200 px prevents wall skimming in tight gaps |
| | **Hairpin Pinch Curves** | **Undefined** (censored) | **Improving** (-12.5 pp crashes at $f=0.50$) | **Improving** (+10.0 pp completion at $f=0.50$) | **RECOMMENDED**: 300 px cuts corners prematurely |
| **Steering Penalty ($\beta_{\text{steer}} = 0.01$)** | **Standard Tracks ($W=70$)** | **Improving** (-1.95 steps, $p=0.0005$) | **Neutral** (0.0 pp) | **Neutral** (100% completion) | Produces smoother racing lines and higher exit speed |
| | **Narrow Corridors ($W \le 40$)** | **Improving** (-2.7 to -16.2 steps, $p < 0.001$) | **Improving** (-10.0 pp crashes at $W=20$) | **Improving** (+20.0 pp completion at $W=20$) | **CRITICAL**: Dampens destructive lateral oscillation |
| | **Hairpin Pinch Curves** | **Undefined** (censored) | **Inconclusive** (censored by timeout shift) | **Improving** (+25.0 pp completion at $f=0.50$) | **ESSENTIAL**: Without penalty, policy enters dead loops |

---

## 7. Statistical Uncertainty & Recommendations for Future Replications

### 7.1 Sources of Uncertainty
1. **Geometric Held-Out Variation vs. Policy Training Variance**:
   - The reported 95% bootstrap confidence intervals accurately measure the stability of a given policy across unseen held-out track topologies.
   - However, because each condition was trained from only **one random training seed (42)**, the measured differences may reflect seed-dependent stochasticity in SAC gradient optimization, replay buffer sampling, or policy entropy temperature trajectories.
2. **Small-Sample Paired Lap Time Sets in Extreme Corridors**:
   - At $W=20\text{ px}$, only 3 of the 10 circuit seeds were successfully completed by Run H. While the paired speedup of DR is massive ($-46.92$ steps), $N=3$ limits formal parametric power.
3. **Competing Risk Distortion in Curvature Benchmarks**:
   - In hairpins, the off-track crash rate metric suffers from competing-risk censoring due to vehicle stalls and wandering timeouts. 

### 7.2 Justification for Training-Seed Replications
Additional training-seed replications (e.g., evaluating 5 independent training seeds $s_{\text{train}} \in \{42, 101, 202, 303, 404\}$ per configuration) are strongly justified for:
1. **Validating the Hairpin Lookahead × Steer Penalty Interaction**: Confirming whether the $+25\text{ pp}$ hairpin completion advantage of Run E over Run A is an intrinsic property of nominal vs. DR training or a seed-dependent convergence artifact.
2. **Narrow Corridor Margin Testing**: Determining whether the catastrophic collapse of Run H (no DR, long lookahead, no steering penalty) at $W=20\text{ px}$ reproduces across all seeds or represents an unfortunate local optimum.
3. **Separating Algorithmic Variance from Environmental Generalization**: Providing true two-way nested ANOVA (training seed $\times$ circuit seed) to yield unassailable causal bounds for safety-critical deployment.
