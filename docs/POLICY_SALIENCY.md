# Policy Saliency Analysis: Timestep-Level Input Sensitivity for `generalist-v1.0`

## Executive Summary

This study implements gradient-based input saliency analysis for the frozen `generalist-v1.0` continuous racing agent checkpoint ([`models/generalist_v1.pth`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/models/generalist_v1.pth)), resolving the core research question:

> **"Which ray-cast beams dictate throttle/brake and steering across varying track geometries?"**

By capturing the local Jacobian of the deployed policy actions with respect to every input feature at each simulation timestep, we quantify the empirical influence of each vehicle-state attribute and rangefinder beam.

### Key Findings Summary

| Action Domain | Highest Sensitivity Ray (Raw Gradient) | Highest Sensitivity Ray (Scaled $S \cdot \sigma$) | Sensitivity Value | Secondary Sensory Cue | Physical Driving Mechanism |
|---|---|---|---|---|---|
| **Steering ($a_0$)** | `ray_plus_90` ($+90^\circ$ right lateral) | `ray_minus_45` ($-45^\circ$ left diagonal) | Raw: $0.0692\text{ act/px}$<br>Scaled: $2.3827$ | `ray_plus_45` ($+45^\circ$, scaled $1.1512$) | Diagonal lookahead beams drive turn anticipation; lateral beams enforce corridor centering. |
| **Braking ($a_1 < 0$)** | `ray_0` ($0^\circ$ center forward) | `ray_0` ($0^\circ$ center forward) | Raw: $0.0181\text{ act/px}$<br>Scaled: $0.6110$ | `ray_plus_45` ($+45^\circ$, scaled $0.2726$) | **The center forward beam overwhelmingly dictates braking**. Rapid distance collapse triggers emergency decel into corners. |
| **Throttle ($a_1 \ge 0$)** | `ray_plus_45` ($+45^\circ$ right diagonal) | `ray_0` ($0^\circ$ center forward) | Raw: $0.0015\text{ act/px}$<br>Scaled: $0.0391$ | `ray_minus_45` ($-45^\circ$, scaled $0.0332$) | Forward lookahead maintains top speed; low gradient due to tanh saturation during flat-out cruising. |

---

## 1. Model Freeze Rule & Cryptographic Verification

The `generalist-v1.0` release is strictly immutable. In accordance with the model freeze rule:
- `models/generalist_v1.pth` and `configs/generalist_v1.json` were loaded without modification, retraining, or fine-tuning.
- The actor network was placed in evaluation mode (`actor.eval()`).
- All 18,310 network parameters were frozen using `param.requires_grad_(False)`.
- Input gradients were calculated exclusively on the observation tensor using `torch.autograd.grad`. No optimizer was instantiated, and no parameter gradients were accumulated.
- The cryptographic SHA-256 hashes were recorded before and after analysis:

```
Pre-Analysis  Weights SHA-256: 0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784 [VERIFIED]
Post-Analysis Weights SHA-256: 0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784 [VERIFIED]
Pre-Analysis  Config  SHA-256: 4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1 [VERIFIED]
Post-Analysis Config  SHA-256: 4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1 [VERIFIED]
```

Bit-for-bit invariance is verified across all evaluations.

---

## 2. Model Architecture & System Contracts

### 2.1 SACActor Neural Architecture

Derived from `sac/actor.py`:
- **Input Dimension**: $\mathbf{s} \in \mathbb{R}^9$
- **Feature Extraction**:
  - $h_1 = \text{ReLU}(W_1 \mathbf{s} + b_1)$ with $W_1 \in \mathbb{R}^{128 \times 9}, b_1 \in \mathbb{R}^{128}$
  - $h_2 = \text{ReLU}(W_2 h_1 + b_2)$ with $W_2 \in \mathbb{R}^{128 \times 128}, b_2 \in \mathbb{R}^{128}$
- **Heads**:
  - Mean head: $\boldsymbol{\mu}(\mathbf{s}) = W_\mu h_2 + b_\mu \in \mathbb{R}^2$
  - Log-standard deviation head: $\log \boldsymbol{\sigma}(\mathbf{s}) = \text{clamp}(W_\sigma h_2 + b_\sigma, -20, 2) \in \mathbb{R}^2$
- **Action Squashing**: Deployed deterministic action is $\mathbf{a} = \tanh(\boldsymbol{\mu}(\mathbf{s})) \in [-1.0, 1.0]^2$.

### 2.2 Observation Space Contract ($\mathbf{s} \in \mathbb{R}^9$)

Derived directly from `env/environment.py` (`reset()` and `step()`):

| Index | Feature Key | Physical Meaning | Canonical Units | Environment Range in Code | Dataset Statistics ($\mu \pm \sigma$) |
|---|---|---|---|---|---|
| `s[0]` | `velocity` | Vehicle longitudinal speed | $\text{px/step}$ | $[-2.0, 4.0]$ | $3.56 \pm 0.98\text{ px/step}$ |
| `s[1]` | `heading_error` | Angle from car heading to track tangent | degrees | $[-180.0, 180.0]$ | $-0.36 \pm 3.58^\circ$ |
| `s[2]` | `signed_distance` | Perpendicular distance to centerline | pixels | $(-\infty, \infty)$ | $-0.77 \pm 2.79\text{ px}$ |
| `s[3]` | `slip` | Tire velocity slip angle | degrees | $[-180.0, 180.0]$ | $0.90 \pm 1.02^\circ$ |
| `s[4]` | `ray_minus_90` | Left lateral rangefinder ($-90^\circ$) | pixels | $[0.0, 200.0]$ | $30.15 \pm 12.76\text{ px}$ |
| `s[5]` | `ray_minus_45` | Left diagonal rangefinder ($-45^\circ$) | pixels | $[0.0, 200.0]$ | $51.91 \pm 35.86\text{ px}$ |
| `s[6]` | `ray_0` | Center forward rangefinder ($0^\circ$) | pixels | $[0.0, 200.0]$ | $115.54 \pm 33.68\text{ px}$ |
| `s[7]` | `ray_plus_45` | Right diagonal rangefinder ($+45^\circ$) | pixels | $[0.0, 200.0]$ | $38.99 \pm 19.57\text{ px}$ |
| `s[8]` | `ray_plus_90` | Right lateral rangefinder ($+90^\circ$) | pixels | $[0.0, 200.0]$ | $28.17 \pm 10.03\text{ px}$ |

> [!NOTE] Observation Representation Finding
> In `configs/generalist_v1.json`, ray features were colloquially labeled as `normalized [0, 1]`. However, inspection of `env/car.py` (`cast_rays()`) and `env/environment.py` reveals that the simulation emits raw Euclidean pixel distances ($0.0$ to $200.0\text{ px}$) computed with ray marching step $4.0\text{ px}$. The network receives these unnormalized float32 values directly.

### 2.3 Action Space Mapping ($\mathbf{a} \in [-1.0, 1.0]^2$)

Derived directly from `env/environment.py`:
- `a[0]` (Steering): $u_{\text{steer}} \in [-1.0, 1.0]$. Directly controls vehicle front wheel steering angle $\delta = u_{\text{steer}} \cdot 30.0^\circ$. Negative commands steer left; positive commands steer right.
- `a[1]` (Longitudinal Control): $u_{\text{long}} \in [-1.0, 1.0]$. Modulates velocity by $\Delta v = u_{\text{long}} \cdot 0.08$:
  - When $u_{\text{long}} \ge 0$: **Throttle / Acceleration Regime**. Drives the car forward toward terminal velocity $4.0\text{ px/step}$.
  - When $u_{\text{long}} < 0$: **Braking / Deceleration Regime**. Actively reduces forward speed down to $-2.0\text{ px/step}$ (reverse).

---

## 3. Mathematical Formulation of Gradient Saliency

### 3.1 Deployed Action Sensitivity

For each observation $\mathbf{s} \in \mathbb{R}^9$ encountered at simulation timestep $t$, the deterministic deployed action is:

$$a_k(\mathbf{s}) = \tanh(\mu_k(\mathbf{s})), \quad k \in \{0 (\text{steer}), 1 (\text{long})\}$$

The gradient vector of deployed action $a_k$ with respect to observation feature $s_i$ is calculated using PyTorch automatic differentiation:

$$g_{k, i} = \frac{\partial a_k}{\partial s_i} = (1 - a_k^2) \frac{\partial \mu_k}{\partial s_i}$$

The primary local input saliency magnitude is defined as:

$$S(i, a_k) = \left| \frac{\partial a_k}{\partial s_i} \right|$$

### 3.2 Tanh Saturation and Pre-Squashing Latent Sensitivity

Because the generalist policy operates near maximum throttle ($a_1 \approx +0.98$) on high-speed straights, the derivative $(1 - \tanh^2(\mu_1))$ approaches zero:

$$1 - a_1^2 = 1 - (0.984)^2 \approx 0.0317$$

This derivative attenuation is physical: when the throttle is already fully open, small changes in sensor readings cannot push the physical throttle beyond $1.0$.

To avoid misinterpreting saturation attenuation as policy indifference, our analysis pipeline simultaneously calculates and archives the pre-squashed latent sensitivity:

$$S_\mu(i, a_k) = \left| \frac{\partial \mu_k}{\partial s_i} \right|$$

### 3.3 Feature Scaling & Standardized Saliency

Because observation features span differing dimensions (e.g., velocity $\sigma \approx 0.98\text{ px/step}$, rays $\sigma \approx 10 - 36\text{ px}$, heading error $\sigma \approx 3.58^\circ$), raw gradients carry units of $[\text{action} / \text{unit}]$.

To compare feature importances on a dimensionless basis, we report both:
1. **Raw Gradient Magnitude**: $S_{\text{raw}}(i) = |\partial a / \partial s_i|$
2. **Standardized Saliency**: $S_{\text{scaled}}(i) = |\partial a / \partial s_i| \cdot \sigma(s_i)$, measuring the expected action deviation produced by a one-standard-deviation perturbation in feature $s_i$.

---

## 4. Controlled Empirical Dataset Collection

The analysis was performed across 30 full benchmark episodes on held-out `TEST_SEEDS` 1001–1010, capturing **11,825 simulation timesteps**:

1. **Standard Geometry ($W=70\text{ px}$)**: All 10 held-out seeds (1001–1010). All 10 completed laps successfully (mean pace $399.4\text{ steps}$).
2. **Narrow Corridor Geometry ($W=20\text{ px}$ and $W=40\text{ px}$)**: Evaluated across representative held-out seeds (1001, 1002, 1003, 1005, 1007, 1008). All 12 episodes completed 100% of laps (mean pace $393.3\text{ steps}$).
3. **Acute Hairpin Geometry ($f \in [0.50, 0.35, 0.20]$)**:
   - $f=0.50$: Seeds 1002, 1003, 1006, 1007, 1008 all completed laps (mean pace $406.0\text{ steps}$).
   - $f=0.35$: Seeds 1007, 1008 completed laps (mean pace $429.5\text{ steps}$).
   - $f=0.20$: Seed 1001 executed genuine hairpin negotiation before an unrecoverable boundary crash at step 230. This episode was preserved in full without fabrication to represent failure dynamics.

All timestep records were stored in [`analysis/results/policy_saliency_timesteps.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/policy_saliency_timesteps.csv) and aggregated in [`analysis/results/policy_saliency_summary.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/policy_saliency_summary.json).

---

## 5. Detailed Saliency Findings

### 5.1 Which Ray Dictates Steering?

Examining all 11,825 timesteps, steering is governed by an interplay between lateral corridor centering and diagonal corner anticipation:

```
Steering Ray Saliency Rankings:
-----------------------------------------------------------------------------------------
Sensor Ray              Raw Mean (|da/ds|)       Empirical Std (px)    Scaled Mean (S * std)
-----------------------------------------------------------------------------------------
ray_minus_45 (-45 deg)  0.066439 act/px          35.86 px              2.382743  [RANK 1 SCALED]
ray_plus_45  (+45 deg)  0.058813 act/px          19.57 px              1.151198  [RANK 2 SCALED]
ray_0        (  0 deg)  0.029128 act/px          33.68 px              0.980945  [RANK 3 SCALED]
ray_plus_90  (+90 deg)  0.069235 act/px          10.03 px              0.694476  [RANK 1 RAW]
ray_minus_90 (-90 deg)  0.044275 act/px          12.76 px              0.565109  [RANK 5 SCALED]
-----------------------------------------------------------------------------------------
```

**Interpretation**:
- In raw terms, `ray_plus_90` has the highest instantaneous sensitivity ($0.0692\text{ act/px}$). When the vehicle draws near the right boundary, the agent applies immediate corrective countersteer.
- However, when accounting for sensor range variance, `ray_minus_45` dominates by more than $2\times$ ($S_{\text{scaled}} = 2.3827$). The diagonal lookahead beams provide the primary sensory gradient that cues the policy to begin steering before reaching the corner apex.

### 5.2 Which Ray Dictates Throttle vs. Braking?

A critical insight arises from partitioning timesteps into **Throttle Regime** ($a_1 \ge 0$, 11,743 steps, 99.31%) and **Braking Regime** ($a_1 < 0$, 82 steps, 0.69%):

```
Longitudinal Control Saliency Breakdown:
-----------------------------------------------------------------------------------------
Sensor Ray              Throttle Regime (|da/ds|)    Braking Regime (|da/ds|)   Brake/Throttle Ratio
-----------------------------------------------------------------------------------------
ray_0        (  0 deg)  0.001162 act/px              0.018142 act/px            15.6x higher
ray_plus_45  (+45 deg)  0.001516 act/px              0.013925 act/px             9.2x higher
ray_plus_90  (+90 deg)  0.001393 act/px              0.011047 act/px             7.9x higher
ray_minus_90 (-90 deg)  0.000956 act/px              0.005882 act/px             6.1x higher
ray_minus_45 (-45 deg)  0.000927 act/px              0.002450 act/px             2.6x higher
-----------------------------------------------------------------------------------------
```

```
Braking Regime Scaled Saliency Ranking:
  1. ray_0        (  0 deg): 0.610976  [DOMINANT SENSOR]
  2. ray_plus_45  (+45 deg): 0.272561
  3. ray_plus_90  (+90 deg): 0.110809
  4. ray_minus_45 (-45 deg): 0.087864
  5. ray_minus_90 (-90 deg): 0.075070
```

**Key Takeaways**:
1. **`ray_0` (center forward beam) overwhelmingly dictates braking**. When approaching an acute hairpin or constriction, `ray_0` drops rapidly from $\sim 150\text{ px}$ to $< 40\text{ px}$. This triggers maximum braking with sensitivity 15.6 times higher than during steady throttle.
2. During throttle application ($a_1 \ge 0$), all longitudinal gradients are small ($< 0.0016\text{ act/px}$) due to output tanh saturation. The forward diagonal rays `ray_plus_45` and `ray_0` provide the guidance that keeps the throttle pinned on open stretches.

### 5.3 Straight vs. Turn Saliency Comparison

Using ground-truth road curvature from `env.track.corner_radius(x, y)`:
- **Straight Sections** ($R \ge 150\text{ px}$): 1,552 timesteps
- **Turn Sections** ($R < 150\text{ px}$ or $|\delta| > 5^\circ$): 10,273 timesteps

```
Steering Saliency: Straights vs Turns
-----------------------------------------------------------------------------------------
Sensor Ray              Straight Sections (|da/ds|)    Turn Sections (|da/ds|)    Difference
-----------------------------------------------------------------------------------------
ray_plus_90  (+90 deg)  0.090259 act/px                0.066058 act/px            -26.8% (less on turns)
ray_plus_45  (+45 deg)  0.079050 act/px                0.055756 act/px            -29.5%
ray_minus_45 (-45 deg)  0.077915 act/px                0.064705 act/px            -16.9%
ray_minus_90 (-90 deg)  0.048883 act/px                0.043578 act/px            -10.9%
ray_0        (  0 deg)  0.037262 act/px                0.027900 act/px            -25.1%
-----------------------------------------------------------------------------------------
```

**Observation**: Overall steering gradient magnitudes are higher on straight sections than inside established turns. On straightaways, the policy operates in a high-gain centering mode where minor lateral drifts produce immediate steering corrections. Inside steady-state turns, the front wheels are already deflected ($|\delta| \approx 20 - 30^\circ$), entering partial saturation where local sensitivity drops slightly.

### 5.4 Temporal Evolution Across Lap Progress

Partitioning laps into quartiles of completion ($Q_1: 0-25\%, Q_2: 25-50\%, Q_3: 50-75\%, Q_4: 75-100\%$):

```
Scaled Steering Saliency Across Quartiles:
-----------------------------------------------------------------------------------------
Quartile       Top Ray (Scaled)       Scaled Value      Secondary Ray       Scaled Value
-----------------------------------------------------------------------------------------
Q1 (0-25%)     ray_minus_45           2.506250          ray_plus_45         1.169096
Q2 (25-50%)    ray_minus_45           2.145478          ray_plus_45         1.076653
Q3 (50-75%)    ray_minus_45           2.531594          ray_plus_45         1.214444
Q4 (75-100%)   ray_minus_45           2.294104          ray_plus_45         1.138559
-----------------------------------------------------------------------------------------
```

Sensitivity rankings remain highly consistent throughout the lap, demonstrating that sensory processing is stationary with respect to lap progress and driven by local geometric features rather than elapsed time.

---

## 6. Numerical Validation & Gradient Verification

### 6.1 Central Finite-Difference Perturbations

To verify autograd calculations against independent ground truth, central finite differences were evaluated across a representative sample of 50 timesteps across all geometric benchmarks:

$$g_{\text{fd}, i} = \frac{a(\mathbf{s} + \epsilon \mathbf{e}_i) - a(\mathbf{s} - \epsilon \mathbf{e}_i)}{2\epsilon}, \quad \epsilon = 1.0 \times 10^{-3}$$

**Validation Metrics**:
- **Test Sample Size**: 50 full 9-feature vectors (100 evaluated actions)
- **Validation Pass Rate**: **100.0% (50/50)** with tolerance $\tau = 5.0 \times 10^{-3}$
- **Maximum Observed Discrepancy**: $9.004 \times 10^{-4}$
- **Mean Observed Discrepancy**: $3.495 \times 10^{-5}$

The autograd input gradients match the numerical finite-difference perturbations to four decimal places.

### 6.2 Gradient Health Checks

Audit of all 11,825 collected timesteps:
- $\text{NaN}$ values: **0** (confirmed absent across all features)
- $\pm \infty$ values: **0** (confirmed absent)
- All-zero gradients: **0** occurrences across evaluated episodes
- Steering saliency range: $[0.0, 2.667]$
- Longitudinal saliency range: $[1.87 \times 10^{-9}, 0.1505]$

---

## 7. Limitations & Scope

1. **Local Sensitivity vs. Causal Interventions**: Saliency measures the local partial derivative $\partial a / \partial s_i$ at a specific point in state space. A large gradient indicates that minor perturbations in that feature alter the instantaneous control command; it does not imply that masking that sensor would cause identical failure modes across the entire circuit.
2. **Action Saturation**: As noted, deployed longitudinal gradients become small during flat-out cruising due to $\tanh$ squashing ($1 - a_1^2 \to 0$). Both deployed action saliency and pre-squashed latent saliency must be analyzed together.
3. **Subsequent Per-Corner Saliency Task**: This milestone delivers verified, timestep-level saliency records ([`policy_saliency_timesteps.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/policy_saliency_timesteps.csv)) indexed by $(x, y)$, heading, curvature, and track seed. This structured dataset forms the direct input substrate for the next scheduled research task: **Per-Corner Saliency Heatmaps**.

---

## 8. Verification Commands

To inspect deliverables and run the test suite:

```bash
# Verify frozen model and config hashes
shasum -a 256 models/generalist_v1.pth configs/generalist_v1.json

# Execute saliency unit and integration tests
PYTHONPATH=. ./venv/bin/pytest tests/test_policy_saliency.py -v

# Run full repository test suite
PYTHONPATH=. ./venv/bin/pytest tests/ -v
```
