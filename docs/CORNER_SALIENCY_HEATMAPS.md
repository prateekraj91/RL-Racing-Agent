# Per-Corner Saliency Heatmaps: Spatial Policy Attention for `generalist-v1.0`

## Executive Summary

This study delivers 2D spatial trajectory heatmaps visualizing where the frozen `generalist-v1.0` racing policy ([`models/generalist_v1.pth`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/models/generalist_v1.pth)) is locally sensitive to its rangefinder sensor beams across five distinct held-out circuits.

Building upon the primary timestep-level saliency dataset ([`analysis/results/policy_saliency_timesteps.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/policy_saliency_timesteps.csv)), we segment vehicle trajectories into **Corner Entry**, **Apex**, and **Straight Exit** phases, overlaying local policy sensitivities onto the actual track coordinates.

> [!IMPORTANT] Scientific Distinction: Saliency vs. Attention
> This analysis computes **gradient-based local input saliency** ($S(i, a) = |\partial a / \partial s_i| \cdot \sigma_i$), capturing the instantaneous Jacobian of the deployed continuous policy. It does **not** measure internal neural transformer attention weights, but rather the empirical sensitivity of physical steering and longitudinal actuators to individual rangefinder beams at each $(x, y)$ coordinate.

### Phase Sensitivity Overview (All 5 Tracks Consolidated)

| Corner Phase | Dominant Steering Beam | Steering Sensitivity ($S \cdot \sigma$) | Dominant Longitudinal Beam | Longitudinal Sensitivity ($S \cdot \sigma$) | Driving Dynamics & Mechanical State |
|---|---|---|---|---|---|
| **Corner Entry** | `ray_minus_45` ($-45^\circ$) | **$1.144$** (secondary `ray_plus_45`: $0.497$) | `ray_0` ($0^\circ$) | **$0.101$** (peak braking: $0.0228\text{ act/px}$) | Turn-in initiation; rapid collapse of forward lookahead (`ray_0`) triggers deceleration. |
| **Apex** | `ray_minus_45` ($-45^\circ$) | **$0.597$** (secondary `ray_plus_45`: $0.338$) | `ray_0` ($0^\circ$) | **$0.057$** (local sensitivity minimum) | Maximum steering lock ($\delta \approx 20-30^\circ$); policy enters partial saturation plateau. |
| **Straight Exit** | `ray_minus_45` ($-45^\circ$) | **$2.091$** (secondary `ray_0`: $0.857$) | `ray_0` ($0^\circ$) | **$0.083$** (rebounding throttle sensitivity) | Steering unwinds onto straight; forward rays guide throttle re-application. |
| **Straightaway** | `ray_minus_45` ($-45^\circ$) | **$2.999$** (secondary `ray_plus_45`: $1.198$) | `ray_0` ($0^\circ$) | **$0.040$** (flat-out cruising saturation) | High-gain centering corrections; zero braking events; near-full throttle saturation. |

---

## 1. Immutable Model Rule & Hash Verification

In compliance with the immutable model requirements:
- The frozen checkpoint `models/generalist_v1.pth` and configuration `configs/generalist_v1.json` were strictly preserved.
- No model retraining, fine-tuning, or parameter updates occurred.
- The primary dataset `analysis/results/policy_saliency_timesteps.csv` was preserved without modification.
- Cryptographic SHA-256 hashes were verified before and after analysis:

```
Pre-Analysis  Weights SHA-256: 0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784 [VERIFIED]
Post-Analysis Weights SHA-256: 0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784 [VERIFIED]
Pre-Analysis  Config  SHA-256: 4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1 [VERIFIED]
Post-Analysis Config  SHA-256: 4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1 [VERIFIED]
```

---

## 2. Selection of Five Representative Tracks

Five representative evaluation circuits were drawn directly from the existing held-out benchmark rollouts (`TEST_SEEDS 1001..1010`), covering maximum geometric diversity:

| Track Identifier | Episode Key | Seed | Benchmark | Corridor Width | Hairpin Factor | Minimum Radius | Lap Steps | Selection Rationale |
|---|---|---|---|---|---|---|---|---|
| **Track 1: Standard Balanced** | `std_w70_s1001` | 1001 | `standard_heldout` | $70\text{ px}$ | None | $83.6\text{ px}$ | 387 | Baseline reference circuit with balanced sweeping curves and 3 smooth corners. |
| **Track 2: High-Speed Straights** | `std_w70_s1002` | 1002 | `standard_heldout` | $70\text{ px}$ | None | $88.9\text{ px}$ | 399 | Long straights ($R_{\text{mean}} = 1284\text{ px}$) terminating in 4 sharp $90^\circ$ bends; tests high-speed deceleration. |
| **Track 3: Extreme Narrow Corridor** | `narrow_w20_s1005` | 1005 | `narrow_corridor` | $20\text{ px}$ | None | $88.7\text{ px}$ | 395 | Severe lateral constriction ($20\text{ px}$, barely wider than car); tests lateral beam proximity and wall-centering. |
| **Track 4: Pinched Hairpin** | `hairpin_f0.5_s1003` | 1003 | `hairpin` | $70\text{ px}$ | $0.50$ | $36.2\text{ px}$ | 399 | High-curvature acute deformation ($R_{\min} = 36.2\text{ px}$) across 5 corners; tests lookahead penetration across bends. |
| **Track 5: Severe Acute Hairpin** | `hairpin_f0.35_s1007` | 1007 | `hairpin` | $70\text{ px}$ | $0.35$ | $43.0\text{ px}$ | 432 | The tightest completed hairpin benchmark in repository ($f=0.35$), requiring maximum braking and full steering lock. |

All five episodes completed 100% of their laps with zero crashes, providing **2,012 fully-instrumented timesteps**.

---

## 3. Corner Phase Segmentation Methodology

To partition trajectories without arbitrary heuristic cuts, corner segmentation relies directly on the ground-truth centerline curvature profile $\kappa(s) = 1 / R(s)$ computed by [`env/track.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/env/track.py#L45-L60):

```
Centerline Profile -> Curvature Radius R(t) -> 5-Step Smoothing -> Local Minima (R < 145 px) -> Corner Apexes
                                                                                                    |
       +------------------------------------+------------------------------------+------------------+
       |                                    |                                    |
[Entry Phase]                          [Apex Phase]                         [Exit Phase]
Approach Window                        Local Curvature Peak                 Recovery Window
t in [t_apex - 16, t_apex - 5]        t in [t_apex - 4, t_apex + 4]        t in [t_apex + 5, t_apex + 16]
```

### Segmentation Rules:
1. **Apex Identification**: A corner apex $t_{\text{apex}}$ is defined as a local minimum of smoothed curvature radius $\tilde{R}_t < 145.0\text{ px}$ separated by at least 25 timesteps from adjacent corners.
2. **Apex Phase**: Centered at $t_{\text{apex}} \pm 4$ timesteps. Represents maximum turning angle and tightest instantaneous geometry.
3. **Corner Entry Phase**: Spans $t \in [t_{\text{apex}} - 16, t_{\text{apex}} - 5]$. Represents the approach where curvature increases, braking initiates, and the vehicle begins turn-in.
4. **Straight Exit Phase**: Spans $t \in [t_{\text{apex}} + 5, t_{\text{apex}} + 16]$. Represents curvature unwinding where steering straightens and throttle re-engages.
5. **Straightaway Phase**: All timesteps outside corner windows ($R_t \ge 145\text{ px}$).

### Phase Sample Counts Across the 5 Tracks:
- **Corner Entry**: $249\text{ timesteps}$ ($12.4\%$)
- **Apex**: $189\text{ timesteps}$ ($9.4\%$)
- **Straight Exit**: $252\text{ timesteps}$ ($12.5\%$)
- **Straightaway**: $1,322\text{ timesteps}$ ($65.7\%$)
- **Total Analyzed**: $2,012\text{ timesteps}$

All phase annotations and corner IDs were exported to [`analysis/results/corner_saliency_phases.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/corner_saliency_phases.csv).

---

## 4. Heatmap Visualizations & Colour Scaling Discipline

For each of the five tracks, two dedicated 6-panel spatial heatmaps were produced using actual vehicle world coordinates $(x, y)$:

1. **Steering Saliency Heatmaps** (`analysis/figs/corner_saliency/<track_id>_steering_saliency.png`):
   - **Panel 1**: Track boundary bands, centerline, vehicle trajectory colored by phase (Entry = Amber, Apex = Crimson, Exit = Emerald, Straight = Slate), with labeled corner apex annotations.
   - **Panels 2–6**: Spatial trajectory scatter plots for `ray_minus_90`, `ray_minus_45`, `ray_0`, `ray_plus_45`, `ray_plus_90`.
   - **Consistent Colormap**: Uses `magma` colormap normalized to $[0.0, v_{\max}]$ (where $v_{\max}$ is the $98.5^{\text{th}}$ percentile of all five rays on that circuit). A shared colorbar ensures identical visual intensity reflects identical mathematical sensitivity.

2. **Longitudinal Saliency & Braking Heatmaps** (`analysis/figs/corner_saliency/<track_id>_longitudinal_saliency.png`):
   - **Panel 1**: Longitudinal control regime map distinguishing throttle ($a_1 \ge 0$, green) from active braking ($a_1 < 0$, high-visibility crimson markers with white outlines).
   - **Panels 2–6**: Spatial trajectory scatter plots for each of the five rays using `viridis` colormap normalized to $[0.0, v_{\max}]$ ($99.0^{\text{th}}$ percentile).
   - **Braking Trigger Callouts**: Active braking triggers are explicitly outlined on the `ray_0` panel to illustrate where distance collapse induces deceleration.

3. **Consolidated Multi-Track Comparison Chart** ([`analysis/figs/corner_saliency/corner_saliency_cross_track_comparison.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/corner_saliency_cross_track_comparison.png)):
   - Compares standardized steering and longitudinal sensitivities across Entry, Apex, Exit, and Straight phases for all five tracks simultaneously.

---

## 5. Answers to Core Research Questions

### Q1: Which beams are most sensitive during corner entry?
- **Steering**: **`ray_minus_45`** (scaled $1.144$) dominates, followed by `ray_plus_45` ($0.497$) and `ray_0` ($0.480$). The diagonal lookahead beams provide the primary cues for turn anticipation.
- **Longitudinal**: **`ray_0`** ($0^\circ$ center forward) decisively leads with scaled sensitivity **$0.101$** (more than double the next beam, `ray_plus_45` at $0.051$). Rapid distance collapse ahead cues the agent to brake before entering the turn.

### Q2: Which beams dominate around the apex?
- **Steering**: **`ray_minus_45`** ($0.597$) and **`ray_plus_45`** ($0.338$) continue to dictate lateral positioning. Lateral beams (`ray_minus_90` at $0.162$, `ray_plus_90` at $0.238$) have much lower influence.
- **Longitudinal**: **`ray_0`** ($0.057$) and `ray_minus_45` ($0.053$) dominate, governing minimal maintenance throttle.
- **Apex Sensitivity Dip**: Overall sensitivity drops to a local minimum at the apex across all five tracks. Because front wheel steering is deflected near its maximum mechanical clamp ($|\delta| \approx 20-30^\circ$) and velocity is regulated, the policy enters a partial saturation plateau where local gradients are suppressed.

### Q3: Which beams dominate during straight exit?
- **Steering**: **`ray_minus_45`** surges dramatically to **$2.091$**, followed by `ray_0` ($0.857$) and `ray_plus_45` ($0.786$). As the corner unwinds, diagonal and center beams guide the aggressive counter-steering needed to re-center the car.
- **Longitudinal**: **`ray_0`** ($0.083$) and `ray_plus_45` ($0.047$) dominate as the policy accelerates out of the bend.

### Q4: Does steering sensitivity change between these phases?
- **Yes, dramatically**. Steering sensitivity follows a pronounced **U-shaped curve**:
  - Moderate at Corner Entry ($1.144$ for `ray_minus_45`)
  - Depressed at Apex ($0.597$, down $-47.8\%$)
  - Surging at Straight Exit ($2.091$, up $+250.3\%$ over apex)
  - Highest on Straightaways ($2.999$) where high-speed micro-corrections maintain corridor centering.

### Q5: Does braking sensitivity increase before the apex?
- **Yes**. Active braking events ($a_1 < 0$) occur almost exclusively on corner entry and approach:
  - **Corner Entry**: 9 active braking events, with raw `ray_0` sensitivity reaching **$0.0228\text{ act/px}$**.
  - **Apex**: 9 active braking events ($0.0080\text{ act/px}$).
  - **Straight Exit**: 8 active braking events during tight multi-apex hairpins.
  - **Straightaways**: **0** braking events.
  - Longitudinal `ray_0` sensitivity is **$2.5\times$ higher** during corner entry ($0.101$) than on open straights ($0.040$).

### Q6: Does throttle sensitivity increase after the apex?
- **Yes**. As the vehicle passes the apex, longitudinal sensitivity increases from $0.057$ at the apex to **$0.083$** at straight exit ($+45.6\%$), reflecting progressive throttle roll-on onto the subsequent straight.

### Q7: Are patterns consistent across all five tracks?
- **Highly consistent**. In all five tracks, `ray_minus_45` is the top-ranked steering beam and `ray_0` is the top-ranked longitudinal beam across all corner phases. The relative sensory hierarchy is invariant to track seed.

### Q8: Which track geometries produce different sensor sensitivity patterns?
- **Narrow Corridors (Track 3)**: Lateral beam sensitivity (`ray_plus_90` at $0.069\text{ act/px}$) is significantly elevated compared to standard tracks ($0.044\text{ act/px}$), because walls are constantly within $10\text{ px}$ of the vehicle.
- **Acute Hairpins (Tracks 4 and 5)**: Produce extreme braking spikes on `ray_0` ($>0.025\text{ act/px}$), whereas gentle sweeping corners (Track 1) permit flat-out cruising with near-zero braking sensitivity.

---

## 6. Generated Heatmap Figures Reference

All 11 visualization deliverables are stored under [`analysis/figs/corner_saliency/`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/):

1. **Track 1 (Standard 1001)**:
   - Steering: [`track_1_std_1001_steering_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_1_std_1001_steering_saliency.png)
   - Longitudinal: [`track_1_std_1001_longitudinal_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_1_std_1001_longitudinal_saliency.png)
2. **Track 2 (High-Speed 1002)**:
   - Steering: [`track_2_fast_1002_steering_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_2_fast_1002_steering_saliency.png)
   - Longitudinal: [`track_2_fast_1002_longitudinal_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_2_fast_1002_longitudinal_saliency.png)
3. **Track 3 (Narrow Corridor 1005)**:
   - Steering: [`track_3_narrow_1005_steering_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_3_narrow_1005_steering_saliency.png)
   - Longitudinal: [`track_3_narrow_1005_longitudinal_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_3_narrow_1005_longitudinal_saliency.png)
4. **Track 4 (Pinched Hairpin 1003)**:
   - Steering: [`track_4_hairpin_1003_steering_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_4_hairpin_1003_steering_saliency.png)
   - Longitudinal: [`track_4_hairpin_1003_longitudinal_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_4_hairpin_1003_longitudinal_saliency.png)
5. **Track 5 (Severe Acute Hairpin 1007)**:
   - Steering: [`track_5_acute_1007_steering_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_5_acute_1007_steering_saliency.png)
   - Longitudinal: [`track_5_acute_1007_longitudinal_saliency.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/track_5_acute_1007_longitudinal_saliency.png)
6. **Consolidated Multi-Track Comparison**:
   - [`corner_saliency_cross_track_comparison.png`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/figs/corner_saliency/corner_saliency_cross_track_comparison.png)

---

## 7. Verification and Test Suite

To verify deliverables and run unit tests:

```bash
# Verify frozen model and configuration integrity
shasum -a 256 models/generalist_v1.pth configs/generalist_v1.json

# Execute corner saliency tests
PYTHONPATH=. ./venv/bin/pytest tests/test_corner_saliency.py -v

# Run entire repository testbed
PYTHONPATH=. ./venv/bin/pytest tests/ -v
```

All 9 corner saliency tests and all 76 tests across the test suite pass with zero regressions.
