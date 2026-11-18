# RL Racing Agent: Grounded Commentary Integration Audit

**Audit Status:** COMPLETE
**Final Audit Verdict:** **FAIL (Integration Grounding Audit) / PASS WITH SEVERE CAVEATS (Software Pipeline Scaffolding)**
**Target Release:** `generalist-v1.0` (Immutable Frozen Checkpoint)
**Evaluated Commit:** `9a3db88` (`feat: add Gemini telemetry commentary pipeline`)
**Audit Date:** November 2026
**Auditor:** Independent Research Verification Agent

---

## 1. Executive Summary

This independent audit evaluates the integrity, physical correctness, and semantic grounding of the **Gemini Telemetry Commentary Pipeline** committed in `9a3db88`.

The pipeline was designed to generate expert racing driver commentary grounded in empirical simulation telemetry and gradient-based policy saliencies for the frozen `generalist-v1.0` Soft Actor-Critic (SAC) model. The documentation in [`docs/GEMINI_TELEMETRY_COMMENTARY.md`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/docs/GEMINI_TELEMETRY_COMMENTARY.md) reported a **100% grounding validation pass rate** across 25 milestone events, accompanied by sub-12ms latency figures and three detailed narrative examples.

### Key Audit Findings

1. **Physical Velocity Impossibility (Severe Failure):**
   The commentary report ([`docs/GEMINI_TELEMETRY_COMMENTARY.md`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/docs/GEMINI_TELEMETRY_COMMENTARY.md) §7) prominently cites a car decelerating from **8.82 to 8.12 px/step** (mean 8.45 px/step). In the underlying simulation physics engine ([`env/car.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/env/car.py)), the vehicle has an absolute hard velocity clamp of $\mathbf{v_{\max} = 4.0\text{ px/step}}$. A forward velocity of 8.82 px/step is **physically impossible** in this environment, exceeding maximum speed by **$2.2\times$**.

2. **Fabricated Report Examples (Documentation Discrepancy):**
   All three "Concrete Grounded Commentary Examples" presented in §7 of [`docs/GEMINI_TELEMETRY_COMMENTARY.md`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/docs/GEMINI_TELEMETRY_COMMENTARY.md) (`ev_std_w70_s1001_s045_sustained_braking`, `ev_hairpin_f0.35_s1007_s112_corner_apex`, and `ev_narrow_w20_s1005_s088_lateral_wall_proximity`) **do not exist** in the actual output deliverable ([`analysis/results/telemetry_commentary_examples.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/telemetry_commentary_examples.json)). They were hand-authored narratives containing fictitious event IDs, unmeasured speeds, and unsupported telemetry claims.

3. **Critical Vulnerabilities in `GroundingValidator` (Validation Bypass):**
   The existing `GroundingValidator` has severe architectural blind spots that allowed 100% pass rates despite gross physical and semantic contradictions:
   - **Dead-Code Numerical Validation:** The validator extracts numerical values from text and constructs `valid_ranges`, but **never executes any comparison between them**. Completely fictitious numbers (e.g., speed of 888.8 px/step) pass as `CORRECT_GROUNDED`.
   - **Tautological Evidence Leak:** The sensor attribution check checks `top_l_ray in str(ev_data) or top_l_ray in text`. Because `ev_data` contains the ground-truth metadata, `str(ev_data)` always contains the sensor name, automatically passing commentary that completely omits or contradicts the sensor.
   - **Mislabeled Corner Entries:** Due to a template fallback in `_generate_mock_response`, all 5 `corner_approach` turn-in events (with steering lock up to $-29.4^\circ$) were described as *"Flat-out straightaway cruising with steering centered"*. The validator approved all five as `CORRECT_GROUNDED`.

4. **Unsupported Semantic Intentionality:**
   Generated commentary asserts that the agent *"executes straight-line braking to maximize tire grip"*, but tire grip limits are completely disabled in the simulation ([`env/car.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/env/car.py#L29): `self.grip_limit = False`). Commentary also asserts that the policy *"shifts visual gaze"*, despite the policy being a stateless feedforward MLP operating on a fixed 5-ray scalar vector without a camera or gaze mechanism.

5. **Dataset Provenance & Latency Illusions:**
   - The commentary architecture diagram cites **4,756 timesteps**, which corresponds to no dataset in the repository (actual full dataset: 11,825 rows; 5-lap subset: 2,012 rows).
   - Reported latency (12.0 ms per request, 60.0 ms total) reflects purely local Python mock string formatting, not Google Gemini 3.8 Flash API roundtrip latency (which was measured at $>15\text{ seconds}$ to HTTP 503 high-demand timeouts).

---

## 2. Immutable Model & Configuration Verification

The model freeze rule was strictly maintained throughout this audit. Checksums were verified prior to and following all audit operations.

| Resource | Path | Expected SHA-256 | Verified SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model Weights** | [`models/generalist_v1.pth`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/models/generalist_v1.pth) | `0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784` | `0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784` | **UNALTERED** |
| **Model Config** | [`configs/generalist_v1.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/configs/generalist_v1.json) | `4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1` | `4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1` | **UNALTERED** |

Git release tag `generalist-v1.0` remains pinned to commit `8f3177239ba0547ab191682a9c21f5aca5308994`.

---

## 3. Systematic Evidence Table & Claim Validation

We performed an item-by-item audit of all 25 output items in [`analysis/results/telemetry_commentary_examples.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/telemetry_commentary_examples.json) and all claims in [`docs/GEMINI_TELEMETRY_COMMENTARY.md`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/docs/GEMINI_TELEMETRY_COMMENTARY.md), extracting **109 discrete empirical claims**.

The complete machine-readable audit dataset is delivered in:
[`analysis/results/grounded_commentary_claim_validation.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/grounded_commentary_claim_validation.csv).

### Aggregate Audit Claim Breakdown

```
+---------------------------------------------------------------------------------------------------+
| Total Claims Audited:                                                                        109  |
+------------------------------------+---------------+----------------------------------------------+
| Verdict Classification             | Count         | Percentage                                   |
+------------------------------------+---------------+----------------------------------------------+
| SUPPORTED                          | 80            | 73.4%                                        |
| CONTRADICTED                       | 22            | 20.2%                                        |
| UNSUPPORTED_SEMANTICS              | 4             |  3.7%                                        |
| PHYSICALLY_IMPOSSIBLE              | 3             |  2.8%                                        |
+------------------------------------+---------------+----------------------------------------------+
| Total Unsupported Claims           | 29            | 26.6% (Failure Rate)                         |
+------------------------------------+---------------+----------------------------------------------+
```

### Representative Evidence Table Samples

| Claim ID | Source | Lap / Event | Claim Text | Measured Empirical Value | Verdict | Root Cause / Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CLM_DOC01_02` | `DOC_§7` | `lap_1` / `s045` | "decelerating from 8.82 to 8.12 px/step" | Actual velocity = 2.158 px/step; Hard limit = 4.0 | **PHYSICALLY_IMPOSSIBLE** | Exceeds environment `max_speed = 4.0 px/step` by $2.2\times$. |
| `CLM_DOC01_05` | `DOC_§7` | `lap_1` / `s045` | "straight-line braking to maximize tire grip" | `car.grip_limit = False` | **PHYSICALLY_IMPOSSIBLE** | Tire grip physics disabled in environment. |
| `CLM_DOC03_04` | `DOC_§7` | `lap_3` / `s088` | "policy shifts visual gaze to diagonal rays" | Static 5-ray input vector | **UNSUPPORTED_SEMANTICS** | Stateless feedforward MLP has no camera or gaze mechanism. |
| `CLM_EX00_01` | `JSON` | `lap_1` / `s063` | "Flat-out straightaway cruising" | $R=218.9\text{ px}$, phase=`entry` | **CONTRADICTED** | Event is corner approach turn-in, not a straightaway. |
| `CLM_EX00_03` | `JSON` | `lap_1` / `s063` | "steering centered" | $\delta = -0.457$ ($-13.7^\circ$) | **CONTRADICTED** | Steering lock is $-13.7^\circ$, not centered. |
| `CLM_EX20_03` | `JSON` | `lap_5` / `s000` | "steering centered" | $\delta = -0.981$ ($-29.4^\circ$) | **CONTRADICTED** | Steering is at $-29.4^\circ$ maximum lock into acute hairpin. |
| `CLM_EX19_02` | `JSON` | `lap_4` / `s210` | "centering sensitivity on lateral beams" | Dominant ray is `ray_0` ($S=0.010$) | **CONTRADICTED** | Sensory misattribution; center ray was dominant, not lateral beams. |
| `CLM_EX01_02` | `JSON` | `lap_1` / `s075` | "Steering angle -21.8 degrees" | $\delta = -0.728 \times 30^\circ = -21.8^\circ$ | **SUPPORTED** | Exactly matches empirical steering telemetry. |
| `CLM_EX01_03` | `JSON` | `lap_1` / `s075` | "guided by ray_minus_45" | Dominant steer ray = `ray_minus_45` | **SUPPORTED** | Peak input gradient correctly attributed. |

---

## 4. Audit Task C: Numerical Consistency & Velocity Analysis

### 4.1 The Anomaly: Velocity of 8.82 px/step
Section 7 of [`docs/GEMINI_TELEMETRY_COMMENTARY.md`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/docs/GEMINI_TELEMETRY_COMMENTARY.md) states:
> *"Approaching corner entry C1, the driver initiates threshold braking, decelerating from 8.82 to 8.12 px/step while maintaining a neutral steering angle of 0.013."*

### 4.2 Code Tracing of Vehicle Kinematics
We traced the exact source variable and transformation through [`env/car.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/env/car.py) and [`env/environment.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/env/environment.py):

```python
# env/car.py:18-20
self.acceleration = 0.08
self.max_speed = 4
self.friction = 0.03

# env/car.py:51
self.velocity = max(-self.max_speed / 2, min(self.velocity, self.max_speed))
```

In [`configs/generalist_v1.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/configs/generalist_v1.json#L7-L14), the observation contract explicitly documents:
- **Feature 0 (`velocity`):** Forward velocity of the vehicle in `pixels/step`.
- **Physical Range:** `[-2.0, 4.0]`.

In [`env/environment.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/env/environment.py#L241):
```python
speed_frac = max(self.car.velocity, 0.0) / 4.0  # 0..1, fraction of max speed
```

### 4.3 Conclusion on Velocity Discrepancy
1. In the actual simulation dataset ([`analysis/results/policy_saliency_timesteps.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/policy_saliency_timesteps.csv)), all 11,825 recorded velocities lie strictly within $[0.0, 4.0]\text{ px/step}$.
2. In the generated commentary JSON deliverable ([`analysis/results/telemetry_commentary_examples.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/telemetry_commentary_examples.json)), all 25 items have `mean_velocity` strictly within $[0.195, 4.0]\text{ px/step}$.
3. The reported numbers (8.82, 8.12, 8.45 px/step) in [`docs/GEMINI_TELEMETRY_COMMENTARY.md`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/docs/GEMINI_TELEMETRY_COMMENTARY.md) were **completely fabricated**. They do not originate from any unit conversion (e.g. m/s or km/h) or state transformation; they are ungrounded hallucinations introduced during documentation writing.

---

## 5. Audit Task D: Semantic Grounding & Intentionality Audit

We audited generated text against the actual neural network architecture and physics parameters:

### 5.1 "Maximizing Tire Grip"
- **Claim:** *"The agent executes straight-line braking prior to turn-in to maximize tire grip for directional transition."*
- **Audit:** In [`env/car.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/env/car.py#L29-L31):
  ```python
  # grip-limited cornering (experimental, off by default)
  self.grip_limit = False  # True enables realistic grip physics
  self.max_grip = 0.09     # max lateral accel before understeer
  ```
- **Finding:** Grip limits are explicitly disabled (`False`). The simulation operates on kinematic bicycle motion without dynamic tire slip friction or normal load transfer. The agent cannot maximize tire grip because tire grip does not exist in the simulation.
- **Verdict:** **PHYSICALLY_IMPOSSIBLE**.

### 5.2 "Shifting Visual Gaze"
- **Claim:** *"The policy shifts visual gaze from far downfield (ray_0) to immediate diagonal boundary rays to ensure corridor survival."*
- **Audit:** The policy is `SACActor`, a 2-layer MLP (`fc1: 9 -> 256`, `fc2: 256 -> 256`, `out: 256 -> 4`). Observations consist of a flat, fixed vector of 9 floats passed in parallel on every forward pass.
- **Finding:** The agent has no camera, no visual attention mechanism, no saccadic eye movements, and no sequential gaze control. Saliency variations reflect changing mathematical partial derivatives $\left|\frac{\partial \text{action}}{\partial \text{obs}_i}\right|$, not an intentional shift in gaze.
- **Verdict:** **UNSUPPORTED_SEMANTICS**.

### 5.3 "Steering Chatter"
- **Claim:** *"High centering sensitivity on lateral beams prevents wall clipping without induced steering chatter."*
- **Audit:** Telemetry records only scalar steering actions $\delta \in [-1, 1]$. No spectral frequency analysis, derivative of steering action ($\dot{\delta}$), or oscillation metric was computed or stored in `telemetry_summary`.
- **Verdict:** **UNSUPPORTED_SEMANTICS**.

---

## 6. Audit Task E: Corner Label Verification

### 6.1 Do Labels Like `C1` Correspond to Circuit Blueprints?
- **Finding:** No. The circuits in this research environment are procedurally generated cubic Catmull-Rom or Bezier splines produced from integer random seeds (e.g., `seed=1001`, `seed=1007`). There are no predefined turn numbers, sector markers, or circuit blueprints.
- **Tracing:** In [`analysis/plot_corner_saliency.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/plot_corner_saliency.py#L210-L245), a peak-detection algorithm identifies local minima in `corner_radius < 150 px` separated by at least 25 steps, labeling them sequentially along the trajectory as `corner_1`, `corner_2`, etc.
- **Conclusion:** Labels like `C1` or `corner_1` represent **detected empirical curvature events along the vehicle's path**, not canonical circuit map indices.

---

## 7. Audit Task F: Grounding Validator Vulnerability Analysis

The `GroundingValidator` class in [`analysis/telemetry_commentary.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/telemetry_commentary.py#L95-L195) was audited for bypass vulnerabilities. All vulnerabilities were confirmed and reproduced via unit tests in [`tests/test_grounded_commentary_audit.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/tests/test_grounded_commentary_audit.py).

### Vulnerability 1: Dead-Code Numerical Range Checking
In [`analysis/telemetry_commentary.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/telemetry_commentary.py#L173-L179):
```python
numbers = [float(x) for x in re.findall(r"\b\d+\.?\d*\b", text)]
valid_ranges = [
    telem["velocity_range"],
    [sensors["min_forward_ray_0"] - 5.0, sensors["mean_forward_ray_0"] + 5.0],
    [actions["steering_range"][0] * 30.0 - 5.0, actions["steering_range"][1] * 30.0 + 5.0],
    [telem["min_corner_radius"] - 10.0, telem["mean_corner_radius"] + 10.0],
]
grounding_score = max(0.0, 1.0 - (len(flagged_issues) * 0.25))
```
**Flaw:** The list `numbers` and `valid_ranges` are instantiated, but **never compared**. There is no loop checking whether extracted numbers fall inside `valid_ranges`. Consequently, any number—even `888.8 px/step`—passes without flagging.

### Vulnerability 2: Tautological Evidence Lookup
In [`analysis/telemetry_commentary.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/telemetry_commentary.py#L165-L170):
```python
if top_l_ray and (top_l_ray in str(ev_data) or top_l_ray in text):
    verified_facts.append(f"Correctly referenced dominant longitudinal sensor: {top_l_ray}")
```
**Flaw:** `ev_data` is the structured dictionary produced by the event generator, which *always* contains `top_longitudinal_ray`. Checking `top_l_ray in str(ev_data)` is tautologically `True`. The validator asserts that the commentary correctly referenced the sensor even if the text completely omitted it or named the wrong ray.

### Vulnerability 3: Naive Substring Control-Regime Matching
In [`analysis/telemetry_commentary.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/telemetry_commentary.py#L158-L160):
```python
elif act_mode == "throttle":
    if any(w in text.lower() for w in ["throttle", "accel", "speed", "cruis"]):
        verified_facts.append("Correctly identified throttle / cruising control regime")
```
**Flaw:** The word `"speed"` triggers a verified throttle regime match. In Item 20, the vehicle is crawling from standstill at 0.195 px/step with full steering lock ($-29.4^\circ$). Because the commentary contained `"speed"`, the validator registered a verified control regime match.

---

## 8. Audit Task G: Dataset Provenance & Record Count Discrepancy

Section 1 of [`docs/GEMINI_TELEMETRY_COMMENTARY.md`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/docs/GEMINI_TELEMETRY_COMMENTARY.md) states:
> `[Frozen generalist-v1.0 Policy] -> [Empirical Telemetry & Saliency Dataset] (4,756 Timesteps)`

### Empirical Record Inventory

| Dataset File | Track Scope | Actual Row Count | Notes |
| :--- | :--- | :--- | :--- |
| [`policy_saliency_timesteps.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/policy_saliency_timesteps.csv) | 30 evaluation tracks | **11,825 rows** | Full benchmark evaluation |
| [`corner_saliency_phases.csv`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/corner_saliency_phases.csv) | 5 representative tracks | **2,012 rows** | Source dataset used by `telemetry_events.py` |
| Claimed in Documentation Diagram | N/A | **4,756 rows** | **Unsupported / Discrepant** |

### Origin of the 4,756 Figure
No single file or combination of evaluation files in the repository equals 4,756 rows. The closest sub-total is the `narrow_corridor` partition of `policy_saliency_timesteps.csv` (4,734 rows). The figure 4,756 is an ungrounded transcription artifact.

---

## 9. Audit Task H: Latency and Cost Reporting Audit

### 9.1 Synthetic Mock Latency vs Live API Latency
The summary metrics in [`analysis/results/telemetry_commentary_summary.json`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/results/telemetry_commentary_summary.json) report:
- `total_latency_ms`: `60.0`
- `mean_latency_per_request_ms`: `12.0`

In [`analysis/telemetry_commentary.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/analysis/telemetry_commentary.py#L375):
```python
self.total_latency_ms += 12.0
```
**Audit Finding:** The 12.0 ms latency is a **hardcoded constant** in `_generate_mock_response`. It measures local Python string template formatting. When live network requests were initiated to Gemini 3.8 Flash, actual API roundtrip latency exceeded $15\text{ to }45\text{ seconds}$ before terminating in upstream HTTP 503 high-demand timeouts. Presenting 12.0 ms in research documentation without clearly noting it as synthetic mock execution creates a false impression of live model throughput.

---

## 10. Audit Verification Suite

We authored [`tests/test_grounded_commentary_audit.py`](file:///Users/prateekraj/Desktop/Ideas/RL%20Racing%20Agent/tests/test_grounded_commentary_audit.py) to programmatically test all vulnerabilities and claims:
1. `test_car_environment_physical_velocity_limits`: Confirms `max_speed == 4.0` and `grip_limit == False`.
2. `test_validator_fails_to_catch_impossible_velocity`: Proves validator passes 888.8 px/step due to dead-code range checks.
3. `test_validator_tautological_evidence_leak`: Proves sensor citation passes even when omitted from commentary.
4. `test_validator_passes_centered_steering_claim_during_sharp_turn`: Proves sharp turn entry labeled "steering centered" passes.
5. `test_validator_fails_to_flag_unsupported_grip_and_gaze`: Proves ungrounded grip and visual gaze claims pass.
6. `test_claim_validation_csv_exists_and_populated`: Verifies claim validation CSV integrity across all 109 claims.

### Test Execution Results
```bash
PYTHONPATH=. ./venv/bin/pytest tests/test_grounded_commentary_audit.py -v
============================== 6 passed in 0.18s ===============================
```
All repository unit tests (96 total) pass cleanly.

---

## 11. Final Audit Verdict

### **Verdict: FAIL (Integration Grounding Audit)**
*(With PASS on Pipeline Software Scaffolding & Frozen Checkpoint Preservation)*

### Justification:
1. **Physical Impossibility in Research Documentation:** Citing vehicle speeds of 8.82 px/step when the simulation engine has an immutable ceiling of 4.0 px/step is a disqualifying factual error.
2. **Fabricated Examples:** Presenting three narrative examples in research documentation that do not exist in the output JSON deliverable violates scientific reproducibility.
3. **Validator Bypass Vulnerabilities:** The grounding validator's dead-code numerical checks and tautological evidence lookup mean the 100% pass rate reported in `9a3db88` is not a reliable guarantee of empirical grounding.
4. **Regime Inversions:** 5 out of 25 generated output items mischaracterize high-steering corner entries (up to $-29.4^\circ$) as straightaway cruising with centered steering.

---

## 12. Actionable Recommendations & Remediation Plan

1. **Activate Numerical Range Checking in Validator:**
   Replace dead code with strict value verification:
   ```python
   for num in numbers:
       if not any(r[0] <= num <= r[1] for r in valid_ranges):
           flagged_issues.append(f"Numerical claim {num} falls outside all valid telemetry ranges.")
   ```

2. **Close the Sensor Evidence Loophole:**
   Require the sensor name to appear directly in the natural language text:
   ```python
   if top_s_ray not in text:
       flagged_issues.append(f"Commentary text omitted dominant steering sensor: {top_s_ray}")
   ```

3. **Fix Template Branching for Corner Approaches:**
   Add an explicit branch in `_generate_mock_response` for `corner_approach` rather than falling into `else`:
   ```python
   elif ev["event_type"] in ["corner_approach", "corner_entry"]:
       comment = (
           f"Initiating turn-in for corner {ev['corner_id']} (radius {telem['mean_corner_radius']} px). "
           f"Steering angle reaches {act['mean_steering'] * 30.0:.1f} deg guided by {sal['top_steering_ray']}."
       )
   ```

4. **Expand Forbidden Vocabulary to Match Environment Reality:**
   Add environment-specific non-existent concepts to `FORBIDDEN_HALLUCINATIONS`:
   ```python
   FORBIDDEN_HALLUCINATIONS += [
       "tire grip", "grip limit", "downforce", "aerodynamics",
       "visual gaze", "camera", "lookahead eye", "competitor", "opponent"
   ]
   ```

5. **Clearly Disclose Mock vs Live Latency:**
   Update summary schemas to clearly differentiate between synthetic local mock generation latency and live network API latency.
