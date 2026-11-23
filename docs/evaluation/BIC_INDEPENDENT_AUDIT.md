# BIC Independent Engineering Audit

**Target:** `generalist-v1.0` SAC policy on the Buddh International Circuit (BIC)
**Mode:** Read-only. Code inspected and metrics independently recomputed — not a re-run of the pre-audit.
**Branch:** `feat/visual-uplift` · **Date:** 2026-11-23

## Frozen-model integrity

| File | Expected SHA-256 | Start | End |
| :--- | :--- | :---: | :---: |
| `models/generalist_v1.pth` | `0d40dc47…b76784` | ✅ match | ✅ match |
| `configs/generalist_v1.json` | `4d76575b…3d63bdb1` | ✅ match | ✅ match |

Both frozen artifacts were unchanged before and after the audit. No code, CSV, JSON, plot, or report was modified.

> **Scope note:** The original report (`buddh_evaluation_report.md`) is **not present** in the working tree or git history. I verified the ground-truth code and committed data (CSV/JSON/manifest) directly; where a question concerns what "the report said," I compare against the pre-audit's quoted claims, since the report text itself is unavailable.

---

## Independently verified facts (recomputed from `buddh_full_rollout.csv`, 7,521 rows)

| Quantity | Value | Formula / source |
| :--- | :--- | :--- |
| Steps / termination | 7,521 · `lap_completed` | last `lap_progress` = 1.0001, `crashed` never true |
| Mean speed | 3.979 px/step | `mean(speed_px_step)` |
| Max speed | 4.000 px/step (clamp) | `Car.max_speed = 4` |
| Mean lateral deviation | **0.0588 m** (0.343 px) | `mean(|signed_dist|)` |
| RMS lateral deviation | 0.142 m (0.828 px) | `sqrt(mean(d²))` |
| Max lateral deviation | 1.002 m (5.845 px) | — |
| Braking steps | **0 of 7,521 (0.0%)** | `throttle_action` min = **+0.2745**, mean 0.978 |
| Mean steering jitter | 1.64 °/step | `mean(steer_change)·30°` |
| Slip channel | ±2.645° max, `corr(slip, −Δheading)=1.000` | kinematic yaw-rate echo, not tire slip |

---

## The 12 questions

**1. Turn 3 speed — 56.16 or 148.11 km/h?** Neither is a corner speed. **56.16 km/h occurs at step 31 (dist 4.2 m), during launch spin-up** from standstill (accel 0.08 − friction 0.03 = 0.05 px/step/step ⇒ ~1.52 px/step at step 31). The car reaches the 4.0 px/step clamp and holds it; it passes the importer's detected Turn 3 (d=454 m, r=96.4 m) and *every* corner at ~148.1 km/h. The report's "slowed to 56.16 at Turn 3" is false. 56.16 = `steady_minimum_km_per_h` = `min(speed_kmh[30:])`, mislabeled as a corner. **The pre-audit's conclusion is right; its specific "Turn 3 = 148.11 via 800–1300 m" window is arbitrary (no detected turn lies there).**

**2. Lateral deviation.** Recomputed **mean = 0.059 m, RMS = 0.142 m**. The report's "mean 0.142 m" is the RMS copied into the mean field. Pre-audit correct.

**3. Observation size.** Contract is **9-dim**, confirmed in three places: `RacingEnv` (`shape=(4+len(ray_angles))` = 9), `eval_buddh`/`SACActor(obs_dim=9)`, and `inference/schemas.py` (`len(v)!=9` guard). Layout: `[velocity, heading_err, signed_dist, slip, ray_-90, ray_-45, ray_0, ray_45, ray_90]`. "8 features" is wrong — heading error was dropped from the report's written list.

**4. Robustness labels 03–10.** The committed **CSV/manifest data is correctly labeled, not swapped**: trial_03 init −15 px, 04 +15, 05 −8, 06 +8, 07 −9.97°, 08 +9.97°, 09 −4.97°, 10 +4.99° — each matches its filename and `robustness_manifest.json`. Any inversion was in the absent report table; the underlying data is sound.

**5. Trial 12.** `trial_12_longitudinal_10m.csv` has **7,521 steps / 125.35 s** (not 7,506). The longitudinal offset moved the car along-track (init lateral ≈ 0), so step count is unchanged — see Q10.

**6. Slip.** **Not hardcoded to 0 during the rollout.** `slip = 0.0` appears only in `reset()` (eval_buddh.py:73 and environment.py:145). During `step()`, slip is *computed* (environment.py:290–301). But because `vx,vy` are derived straight from heading and `car.angle` is updated *after* `vx,vy`, the computed slip equals exactly `−Δheading` per step (correlation 1.000, bounded ±2.6°) — a **yaw-rate echo, not tire sideslip**. Training and eval use the identical `RacingEnv.step`, so there is **no train/eval mismatch** and no conclusion is invalidated. The real limitation is that the slip channel carries no tire-dynamics information (the kinematic model has none). *This corrects the pre-audit, which stated slip is "hardcoded 0.0 on every step."*

**7. Scale.** `world_units_per_meter = 70/12 = 5.8333 px/m` (width-calibrated). With `wheelbase = 50 px`, implied physical wheelbase = 50/5.8333 = **8.57 m** — bus-sized (F1 ≈ 3.6 m). Internally consistent but physically meaningless: width and vehicle size are not scaled by the same intent (width matched to 12 m; car left at training default). Pre-audit correct.

**8. Turn radius vs speed.** No friction/grip limit is active (`grip_limit=False` by default, and `BuddhRacingEnv` does not enable it), so lateral acceleration is unbounded. At the full-throttle 4.0 px/step (41.1 m/s), `a_lat = v²/r`:

| Corner | r (m) | a_lat |
| :--- | :---: | :---: |
| T1 | 18.9 | 9.1 G |
| T6 | 15.2 | 11.4 G |
| T8 | 14.4 | 12.0 G |
| tightest (14.35) | 14.35 | **12.0 G** |

F1 peaks ~5–5.5 G. These corners are **physically impossible** at the simulated speed; the kinematic bicycle model simply has no centripetal limit.

**9. Track geometry.** Exact WGS84→UTM 43N transverse-Mercator projection; closed centripetal Catmull-Rom spline (α=0.5); uniform 1 m arc-length resample (5,131 pts). Width is a **constant** 12 m → 70 px (no per-section width). **Pit lane is excluded by construction** — only 6 mainline OSM way IDs are assembled. Lengths: raw chord 5,128.33 m, spline 5,130.63 m, resampled 5,131.0 m vs published 5,138.0 m (0.14%). *Minor defect:* the manifest's `discrepancy_explanation` string hardcodes "7.4 m / 5,130.6 m" while the computed `length_discrepancy_m` is **7.0 m** (resampled rounds to 5,131.0). Cosmetic, not a geometry error.

**10. Lap completion gate.** There is **no 98% gate and no finish-line crossing**. `environment.py:211` fires `lap_completed` when **accumulated** `lap_progress ≥ 1.0` (one full loop length traveled). Because completion is distance-accumulated, a +10 m longitudinal head-start (Trial 12) still requires a full loop of travel ⇒ same ~7,521 steps. "Completed" is honest (the car does traverse the whole loop) but lap *time* = steps × dt is insensitive to start offset and is not measured at a fixed line.

**11. Time base — the critical one.** **No genuine seconds-per-step exists in the physics.** `Car.dt = 1/60` is declared, but every use is `* self.dt * 60`, so **dt cancels** (dt×60 = 1): position advances `velocity` px per step and velocity is in **px/step**. `eval_buddh` separately assumes `dt = 1/60 s` to convert to km/h via `v·60/scale·3.6`. So every km/h/m/s figure rests on an **assumed 60 steps/s** that the integrator does not enforce. Native, assumption-free units: **mean 3.979 px/step, max 4.0 px/step**; `≈0.171 m/px` gives ~0.682 m/step. Every km/h figure (147.3 mean, 148.1 max) must be stated as "assuming dt = 1/60 s."

**12. Policy behaviour on BIC.** From the BIC CSV: **braking = 0.0%** (throttle never negative) — identical to the standard-track pattern. **Steering saturates to full lock** (max |steer| = 0.9998) and jumps up to **47°/step** at corners (bang-bang character present), **but mean jitter is 1.64°/step**, far below the ~10.7°/step cited for the standard track — BIC is mostly long straights, so the *average* is calmer even though the no-brake + saturating-steer signature is the same.

---

## Critical errors
- **C1 (report):** Turn 3 "56.16 km/h" is a launch-transient value (step 31), not a corner speed. The speed-modulation narrative is fabricated — the car never brakes (Q1/Q12).
- **C2 (report):** Mean lateral deviation reported as 0.142 m; true mean 0.059 m (RMS duplicated) (Q2).
- **C3 (report):** 8-feature observation; actual contract is 9 (Q3).
- **C4 (pre-audit):** Claims slip is "hardcoded 0.0 every step"; it is reset-only and computed (as a yaw-rate echo) during stepping (Q6).

## Methodological limitations
- **No time base** — all SI speeds depend on an unenforced 60 steps/s assumption (Q11).
- **Kinematic bicycle model, no friction circle** — unbounded lateral g (up to 12 G); default `grip_limit=False` (Q8).
- **Slip channel is information-free** (kinematic yaw-rate echo) (Q6).
- **Rangefinders saturate on a 70 px corridor** — side rays pin `min_ahead ≈ 35 px`, so the reward's "corner-aware" speed penalty is effectively constant and the policy gets no braking signal; consistent with 0% braking.
- **Lap completion is distance-accumulated, not a timed line** (Q10).

## Reproducibility status
**Strong.** Importer and eval are fully offline/deterministic; eval = `tanh(mean)`, no sampling. `inference/` reuses the same `SACActor`, `obs_dim=9`, `tanh(mean)`, strict weight load, no normalization — **SACActor↔FastAPI parity holds**. All numeric claims above reproduce from committed CSVs. The absent `buddh_evaluation_report.md` is the one reproducibility gap (its exact table text cannot be re-verified).

## Recommended corrections (report text only — do not touch frozen artifacts or committed data)
1. Replace Turn 3 "56.16 km/h" with "148.1 km/h, no braking; 56.16 = launch-transient steady-min."
2. Fix mean lateral deviation to 0.059 m; keep RMS 0.142 m.
3. State observation as 9-dim with the explicit layout.
4. Report speeds primarily in px/step; annotate every km/h with "assuming dt = 1/60 s."
5. Correct the manifest `discrepancy_explanation` string (7.0 m, 5,131.0 m) in a future unfreeze.

---

## Verdict

**(a) What the sim genuinely demonstrates:** a frozen SAC policy that completes a geometrically faithful BIC centerline (real OSM geometry, 0.14% length error) with zero departures and zero collisions, deterministically and reproducibly — i.e., robust ray-based path-tracking on a 12 m-wide corridor under lateral/heading/longitudinal start perturbations.

**(b) Reproducible engineering results:** 7,521 steps; 3.979 px/step mean / 4.0 px/step max; mean lateral 0.059 m, RMS 0.142 m, max 1.002 m; 0 crashes/off-track; 12/12 robustness trials complete; bitwise-deterministic; correctly-labeled trial data; SACActor↔FastAPI parity. These stand.

**(c) Physically unrealistic:** cornering at 9–12 G with no grip model; an 8.57 m wheelbase; 0% braking at full throttle through 14–19 m-radius corners; a slip channel that encodes yaw rate, not sideslip; speeds that depend on an unenforced frame-rate assumption.

**(d) What CANNOT be claimed about a real car:** nothing about real lap times, real speed (no verified time base), tire/grip behaviour, braking skill, or race-car dynamics. The agent has **not** demonstrated physically valid high-speed driving — only kinematic path-following on a scaled 2D corridor. No sim-to-real transfer is supported by this evaluation.

*Frozen hashes verified identical at start and end of this audit.*
