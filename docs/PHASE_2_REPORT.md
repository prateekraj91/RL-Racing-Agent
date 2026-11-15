# Phase 2 Consolidated Research Report: Reward Evolution, Baselines, and Generalization Analysis

**Project:** Autonomous Reinforcement Learning Racing Agent
**Phase:** Phase 2 Consolidation (Reward Evolution, Single-Track Baselines, Generalization Gaps, and Domain Randomization Foundations)
**Status:** Complete & Consolidated
**Canonical Baseline Model:** `models/best_single_track_v2.pth` (`models/best_single_track_v2_config.json`)
**Benchmark Evidence:** `baselines/eval_v1_v2_heuristic.json`

---

## Executive Summary

Phase 2 focused on transforming the initial continuous Soft Actor-Critic (SAC) prototype into a reliable, high-performance racing agent capable of consistently lapping procedural circuits faster than classical control heuristics, understanding the limits of single-track reinforcement learning, and identifying the structural requirements for cross-difficulty generalization.

Key achievements consolidated in this report:
1. **Reward Formulation History**: Mapped the exact evolutionary trajectory from naive survival rewards to Reward v1 and Reward v2, documenting the mathematical and behavioral rationale for each term.
2. **Exploration Trap Resolution**: Diagnosed why Reward v2 suffered from null-policy collapse (agent standing still at $v=0$) and demonstrated how demonstration seeding (40 pure-pursuit laps) solved the medium track in 5,000 steps without changing reward weights.
3. **Locked Single-Track Baseline**: Frozen `models/best_single_track_v2.pth` achieves **10/10 lap completions** across medium and default tracks, outperforming classical pure pursuit by **1.65× on lap time** with zero crashes.
4. **Generalization Boundaries**: Proved that while egocentric track-relative observations generalize seamlessly across tracks of identical difficulty (30/30 laps), severe generalization gaps emerge across difficulty axes (narrow track widths $\le 28\text{ px}$ and acute hairpins $\le 0.50$).
5. **Physical Emergence of Corner Braking**: Demonstrated via the grip experiment that anticipatory corner braking emerges from physical dynamics ($a_{\text{lat}} \le 0.09$) rather than artificial reward shaping.

---

## 1. Reward Design Evolution

Reconstructed directly from Git history, commit diffs, and codebase audits, the reward function evolved through four verified stages:

```
[v0 Naive Gym Reward] (8a8c18a)
  └─> [v1 Initial] (7f27bb9): Progress + 0.01*v - 2.0*crash
        └─> [v1 Continuous Rescaled] (c6f554d, 9a48374): 100 * (Progress + 0.01*v - 0.1*crash)
              └─> [v2 Multi-Objective] (4c53e74): 100 * (20*Progress - CurvatureTax - Smoothness - 0.2*crash)
```

### 1.1 Version 0: Naive On-Track & Velocity Reward (Commit `8a8c18a`, Jul 22)

```python
reward = 0.0
if self.track.is_on_track(self.car.x, self.car.y):
    reward += 0.1
reward += self.car.velocity * 0.1
```

- **Intent**: Provide basic dense survival and forward-motion incentives.
- **Observed Effect / Failure Mode**: The agent had no concept of circuit progression. It learned to drive in tight circular donuts or oscillate back and forth within the road band near the start line, harvesting $+0.1$ survival reward indefinitely without making net progress along the track.

---

### 1.2 Version 1: Arc-Length Progress & Speed Reward (Commits `7f27bb9`, `c6f554d`, `9a48374`, Aug 10–17)

```python
speed_reward = self.car.velocity * 0.01
reward = progress + speed_reward
if crashed:
    reward -= 0.1
reward *= 100.0
```

- **Mathematical Form per Step**:
  $$R_{v1} = 100.0 \cdot \Delta\text{progress} + 1.0 \cdot v - 10.0 \cdot \mathbb{I}_{\text{crashed}}$$
  where $\Delta\text{progress}$ is normalized centerline arc length ($\approx +100.0$ for a complete lap).
- **Verified Behavioral Defects (`docs/DIAGNOSIS.md`)**:
  1. **FM-1 (Uni-directional steering bias)**: On single training tracks, the policy learned one-sided steering actions ($[-0.664, -0.244]$), completely losing the ability to turn in the opposite direction.
  2. **FM-2 (No anticipatory braking)**: In the kinematic bicycle model:
     $$R_{\text{turn}} = \frac{L}{\tan\delta}$$
     Turning radius is independent of velocity $v$. Because speed $1.0 \cdot v$ was rewarded unconditionally at every step and slowing down provided zero turning advantage under kinematic physics, the policy pinned throttle flat-out ($u_{\text{thr}} = +1.0$) into every bend.
  3. **High crash rate on tighter tracks**: While v1 completed 4/5 medium tracks, it failed 3/5 default tracks (40% completion rate) because washing wide into barriers was insufficiently penalized compared to accumulated speed reward.

---

### 1.3 Version 2: Multi-Objective Curvature & Smoothness Reward (Commit `4c53e74`, Sep 8)

```python
reward = progress * 20.0

# Curvature-aware speed penalty:
rays = self.car.cast_rays(self.track)
min_ahead = min(rays)
CORNER_THRESH = 120.0
speed_frac = max(self.car.velocity, 0.0) / 4.0
if min_ahead < CORNER_THRESH:
    tightness = (CORNER_THRESH - min_ahead) / CORNER_THRESH
    reward -= 0.02 * tightness * speed_frac
else:
    reward += 0.01 * speed_frac

# Steering smoothness penalty:
steer_change = abs(steering_action - self.prev_steering)
reward -= 0.01 * steer_change
self.prev_steering = steering_action

if crashed:
    reward -= 0.2

reward *= 100.0
```

- **Mathematical Form per Step**:
  $$R_{v2} = 2000.0 \cdot \Delta\text{progress} - 2.0 \cdot \text{tightness} \cdot \left(\frac{v}{4}\right) \cdot \mathbb{I}_{d_{\min} < 120} + 1.0 \cdot \left(\frac{v}{4}\right) \cdot \mathbb{I}_{d_{\min} \ge 120} - 1.0 \cdot |\Delta u_{\text{steer}}| - 20.0 \cdot \mathbb{I}_{\text{crashed}}$$

- **Design Intent**:
  - Scale progress reward by $20\times$ so that completing a lap pays $\approx +2000.0$.
  - Punish high speed when ray returns detect an obstacle/corner closer than $120\text{ px}$.
  - Penalize high-frequency steering chatter ($|\Delta u_{\text{steer}}|$).
  - Double the crash penalty to $-20.0$ to disincentivize barrier collisions.

- **Observed Effect & The Exploration Trap (`docs/REWARD_V2_SOLVED.md`)**:
  - When trained from scratch with standard random SAC warmup, the policy **collapsed to a zero-velocity null attractor**:
    $$\Delta v = \max\left(0,\, 0.08 \cdot u_{\text{thr}} - 0.03\right) \implies u_{\text{thr}} \le 0.375 \implies v = 0$$
    The agent remained frozen at the start line for 105,000 steps (`ctl_none_s42`: 0/21 evals, $v=0$).
  - **Reason**: Moving forward with untrained steering quickly triggered a $-20.0$ crash penalty, whereas standing still produced an exact return of $0.0$. Standing still was the safe local optimum.
  - **Tuning Failure**: Extensive attempts to retune reward weights alone failed to resolve the freeze.

- **The Demonstration-Seeding Solution (`docs/REWARD_V2_SOLVED.md`)**:
  - Pre-seeding the replay buffer with 40 pure-pursuit laps (17,014 transitions, 12 clean, 28 noisy) solved the medium track at **step 5,000** across 4 seeds (seeds 42, 0, 1, 7).
  - Reward v2 did not require retuning; it was an exploration problem, not an objective problem.
  - **Curvature Penalty Flaw**: Because `min(rays)` included the $\pm 90^\circ$ lateral beams which hit the track edge at $\sim 35\text{ px}$ on standard width 70 tracks, $d_{\min}$ was almost always $< 120\text{ px}$. Thus, the curvature term operated as a flat speed tax rather than an anticipatory corner brake.

- **Physics-Driven Braking Emergence (`docs/GRIP_EXPERIMENT.md`)**:
  - In commit `020cd8f`, physical tire grip limits were added to vehicle kinematics:
    $$a_{\text{lat}} = \frac{v^2}{R_{\text{turn}}} \le a_{\text{max\_grip}} = 0.09 \implies R_{\text{grip}} = \frac{v^2}{0.09}$$
  - On tight tracks where flat-out driving understeered off-track, the agent developed genuine threshold braking ($u_{\text{thr}} = -0.735$) **with zero change to the reward function**. This confirmed the core thesis: driving behavior emerges from physical constraints, not artificial reward shaping.

---

## 2. Single-Track Lap Benchmarks

### 2.1 The Locked Baseline Agent

The canonical Phase 2 baseline is **`models/best_single_track_v2.pth`** (MD5: `190026208ac4969f...`), generated from `runs/exp2_demos_s42/solved_actor.pth`.
- **Training Configuration**: Medium track (`width=70, base_r=250, n_ctrl=10, min_radius=80, track_seed=101`), 40 demo laps, seed 42, SAC batch size 64, learning rate $3 \times 10^{-4}$, 45,000 steps.
- **Verification**: Evaluated with deterministic policy rollouts ($\tanh(\mu)$) up to `max_steps=2000`.

### 2.2 Official Comparative Benchmark Results

Data source: `baselines/eval_v1_v2_heuristic.json` (generated 2026-09-10).

#### Medium Track Benchmark (5 Held-Out Seeds: 101, 202, 303, 404, 505)

| Controller | Laps Completed | Completion Rate | Crashes | Avg Lap Time (steps) | Avg Progress | Off-Track Rate | Avg Return |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v2 (Locked Baseline)** | **5 / 5** | **100.0%** | **0** | **401.8** | **1.0018** | **0.0000** | 1402.55 |
| v1 Baseline | 4 / 5 | 80.0% | 1 (seed 303) | 402.2 | 0.9031 | 0.0009 | 1335.08 |
| Pure-Pursuit Heuristic | 5 / 5 | 100.0% | 0 | 659.4 | 1.0008 | 0.0000 | 1475.95 |

*Performance Delta*: v2 is **1.64× faster** (39.1% lower lap time) than the classical pure-pursuit baseline on medium circuits.

#### Default Track Benchmark (Narrower/Tighter Geometry, 5 Seeds: 101, 202, 303, 404, 505)

| Controller | Laps Completed | Completion Rate | Crashes | Avg Lap Time (steps) | Avg Progress | Off-Track Rate | Avg Return |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v2 (Locked Baseline)** | **5 / 5** | **100.0%** | **0** | **346.4** | **1.0016** | **0.0000** | 1502.66 |
| v1 Baseline | 2 / 5 | 40.0% | 3 (101, 202, 303) | 346.0 | 0.5659 | 0.0054 | 865.36 |
| Pure-Pursuit Heuristic | 5 / 5 | 100.0% | 0 | 575.0 | 1.0008 | 0.0000 | 1555.51 |

*Performance Delta*: v2 is **1.66× faster** (39.8% lower lap time) than pure pursuit, while maintaining 100% reliability compared to v1's 60% failure rate.

### 2.3 Per-Seed Outcome Breakdown

| Track Config | Controller | Seed 101 | Seed 202 | Seed 303 | Seed 404 | Seed 505 | Total Laps |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Medium** | v1 | LAP (403) | LAP (400) | **CRASH (226)** | LAP (396) | LAP (410) | 4 / 5 |
| | **v2** | **LAP (402)** | **LAP (400)** | **LAP (408)** | **LAP (397)** | **LAP (402)** | **5 / 5** |
| | Pure-Pursuit | LAP (656) | LAP (664) | LAP (661) | LAP (656) | LAP (660) | 5 / 5 |
| **Default** | v1 | **CRASH (216)** | **CRASH (90)** | **CRASH (89)** | LAP (344) | LAP (348) | 2 / 5 |
| | **v2** | **LAP (346)** | **LAP (346)** | **LAP (346)** | **LAP (344)** | **LAP (350)** | **5 / 5** |
| | Pure-Pursuit | LAP (573) | LAP (578) | LAP (577) | LAP (569) | LAP (578) | 5 / 5 |

### 2.4 Start-State Robustness and Same-Difficulty Generalization

1. **Start-State Jitter (`docs/START_ROBUSTNESS.md`)**: Evaluated across 20 varied start positions (lateral displacements $\pm 21\text{ px}$, heading perturbations $\pm 12^\circ$). The v2 policy completed **20/20 laps (100%)** without crashing.
2. **Disjoint Track Pool Generalization (`docs/GENERALIZATION_TESTBED.md`)**: Evaluated across 20 training tracks and 10 disjoint test tracks of identical medium difficulty. The single-track trained v2 baseline completed **30/30 laps (100%)** zero-shot.
   - *Theoretical Takeaway*: Because the egocentric state representation $[v, \theta_{\text{err}}, d_{\text{signed}}, \text{slip}, \text{rays}]$ provides purely local track-relative coordinates, the policy transfers zero-shot across arbitrary track shapes of matching difficulty.

---

## 3. Generalization Limits and Failure Analyses

While same-difficulty generalization is solved, evaluating the single-track baseline across foreign difficulty regimes revealed severe boundaries:

### 3.1 Cross-Difficulty Width Degradation (`docs/CROSS_DIFFICULTY_GAP.md`)

When track width was narrowed below the training value of $70\text{ px}$ across 10 test seeds per width:

| Track Width (px) | v2 Completed Laps | Average Return | Failure Mode & Diagnosis |
|:---:|:---:|:---:|:---|
| **70 (Trained)** | 10 / 10 | 1399.8 | Baseline training regime |
| **40** | 10 / 10 | 1335.1 | Fully solid |
| **36** | 10 / 10 | 1334.9 | Fully solid |
| **32** | 10 / 10 | 1322.1 | Fully solid |
| **28** | 9 / 10 | 1265.0 | First crack: seed 9 times out |
| **25** | 8 / 10 | 1160.6 | Seeds 3 and 9 time out |
| **20** | 4 / 10 | — | 60% failure rate |

- **Failure Diagnosis**: Visual analysis revealed that failures at width 25 were **not crashes**, but **velocity-zero wedging/stalling**. In narrow corridors, the policy decelerated to avoid walls until $v=0$, entering an unvisited state with zero throttle recovery.

### 3.2 Sharp Hairpin Zero-Shot Collapse (`docs/CROSS_TRACK_FAILURE_ANALYSIS.md`)

When evaluated zero-shot on sharp hairpin tracks (factors 0.50, 0.35, 0.20):
- **Completion Rate**: **0 / 60 completed laps**.
- **Failure Mode**: The agent held flat-out throttle ($u_{\text{thr}} = +1.0$) into tight switchbacks. Understeer pushed the vehicle across the outer boundary.
- **Root Cause**: The 5-ray sensor layout ($[-90, -45, 0, 45, 90]^\circ$) had a $45^\circ$ forward blind cone, while single-track training on gentle curves never penalized flat-out cornering.

### 3.3 Sensor Lookahead Depth Resolution (`docs/SENSOR_REPRESENTATION_EXPERIMENT.md`)

Expanding the sensor configuration to a 9-ray uniform layout with $300\text{ px}$ lookahead depth (`9ray_d300`):
- Increased average progress on hairpin tracks by **+145% to +298%**.
- Enabled active threshold braking ($u_{\text{thr}} = -1.0$) prior to corner entry.
- Completely eliminated start-line freezes.
- **Key Insight**: While expanded sensors provide the necessary information horizon for braking, lap completion on extreme switchbacks requires training across diverse curvatures.

---

## 4. Reproducibility & Provenance Catalog

| Artifact Description | Filepath / Checkpoint | Provenance / Source Script |
|:---|:---|:---|
| Locked Single-Track v2 Weights | `models/best_single_track_v2.pth` | `runs/exp2_demos_s42/solved_actor.pth` |
| Baseline Configuration Manifest | `models/best_single_track_v2_config.json` | `runs/exp2_demos_s42/run_config.json` |
| Official Benchmark Records | `baselines/eval_v1_v2_heuristic.json` | `baselines/benchmark_agents.py` |
| Legacy v1 Baseline Weights | `best_actor.pth` | `sac/train.py --config medium` |
| 9-Ray 300px Finalist Weights | `runs/exp3_demos_s42_9ray_d300/best_actor.pth` | `sac/train_curriculum.py` |
| Hairpin Benchmark Pipeline | `analysis/eval_hairpins.py` | Standalone evaluation harness |
| Unit Test Suite | `test_env_hardening.py`, `test_eval_hairpins.py`, `test_domain_randomizer.py` | `pytest` validation suite |

### Exact Evaluation Commands

1. **Reproduce the Official Single-Track Benchmark**:
   ```bash
   PYTHONPATH=. python baselines/benchmark_agents.py
   ```
2. **Reproduce Disjoint Pool Generalization**:
   ```bash
   python -m sac.evaluate_pool --checkpoint models/best_single_track_v2.pth --pool test
   ```
3. **Reproduce Hairpin Evaluation**:
   ```bash
   python -m analysis.eval_hairpins --candidate-checkpoint models/best_single_track_v2.pth --factors 0.5 0.35 0.2
   ```

---

## 5. Transition to Domain Randomization

The empirical findings of Phase 2 established that:
1. Fixed-track reinforcement learning overfits to the trained track width ($70\text{ px}$) and curvature distribution, producing stall/wedge failures below width 28 and understeer crashes on acute hairpins.
2. Increasing sensor lookahead depth ($300\text{ px}$) provides the necessary perception, but the policy requires exposure to diverse environmental dynamics during training.
3. Therefore, **Domain Randomization (DR)**—sampling track width $U(20, 70)$, min curvature radius $U(50, 90)$, and tire friction $U(0.015, 0.05)$ per episode—is the principled mechanism to close the cross-difficulty generalization gap.
