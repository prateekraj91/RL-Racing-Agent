# Grip-Limited Evaluation of the Frozen Policy (experiment)

**Branch:** `exp/grip-limit` (never merged) · **Policy:** frozen `generalist_v1` (no retraining; hashes verified at start/end) · **Script:** `scripts/grip_eval.py` · **Outputs:** `runs/grip_eval/` (original BIC CSVs untouched).

## 1. Existing grip implementation (`env/car.py`)
Opt-in `Car.grip_limit` (default False) with `Car.max_grip` (default 0.09 px/step²). Each step: `a_lat = v²/r` for the radius implied by steering (`r = wheelbase / tan(steer)`). If `a_lat > max_grip`, the radius is **widened** to `v²/max_grip` (pure understeer). **No speed loss, no slide state, no termination** — the car just turns less than commanded and leaves the track if it goes wide. Counters `grip_events` / `grip_yaw_lost` exist. Complete and functional; the only test coverage is the domain-randomizer parameter range (no test of the clamp itself), so the experiment verifies the clamp empirically (achieved a_lat equals the limit in every grip run). Not broken → proceeded.

## 2. Physical meaning (assumption stated explicitly)
**Assumptions: 12 m = 70 px (1 px = 0.1714 m) and 60 steps/s** — neither is enforced by the physics (audit Q11). Then 1 px/step² = 0.1714 × 3600 = 617 m/s², so **1 G = 0.01589 px/step²**. The default `max_grip = 0.09` is therefore **5.66 G** (not road-car grip). Test levels: 0.8 G = 0.01271, 1.0 G = 0.01589, 1.2 G = 0.01907 px/step².
**v²/r check (T1, r = 18.9 m = 110 px):** policy speed 4.0 px/step = 41.1 m/s (148 km/h) ⇒ a = 41.1²/18.9 = **89 m/s² = 9.1 G**. Max speed T1 allows: √(a·r) = 12.2 / 13.6 / 14.9 m/s (44 / 49 / 54 km/h) at 0.8 / 1.0 / 1.2 G — about **⅓ of the speed the policy carries**.

## 3. Method
Deterministic (`tanh(mean)`), same `BuddhRacingEnv`/loop as the BIC eval; grip set on the car after `reset()` inside the script only. Conditions: OFF + 0.8 / 1.0 / 1.2 G. Tracks: BIC (single deterministic run) and `standard_medium` seeds 1001–1005 (CSV saved for 1001). Episode ends at first off-track (existing env behaviour). "Entry speed" = speed 50 m before the turn's listed distance. Lat accel = speed × |Δheading| per step, converted with the assumptions above. **Sanity: grip-OFF BIC CSV is byte-identical to `analysis/buddh/buddh_full_rollout.csv`.**

## 4. Results
| Track | Grip | Laps | Steps | Off-track at | T1 entry (px/step) | Max lat G achieved | Braking % |
|---|---|---|---|---|---|---|---|
| BIC | OFF | 1 | 7,521 | — | 4.00 (T6 4.00, T8 4.00) | 11.6 | 0.0 |
| BIC | 0.8 G | 0 | 484 | 301.0 m, **T1** (apex at 304 m), 4.00 px/step | 4.00 | 0.8 | 0.0 |
| BIC | 1.0 G | 0 | 484 | 301.1 m, **T1**, 4.00 px/step | 4.00 | 1.0 | 0.0 |
| BIC | 1.2 G | 0 | 485 | 301.9 m, **T1**, 4.00 px/step | 4.00 | 1.2 | 0.0 |
| Standard (5 seeds) | OFF | 5/5 | 387–406 | — | n/a | ~11.4 | 0.0 |
| Standard (5 seeds) | 0.8 G | 0/5 | 77–93 | first ~130–230 px of a ~1375 px lap | n/a | 0.8 | 0.0 |
| Standard (5 seeds) | 1.0 G | 0/5 | 79–97 | same region | n/a | 1.0 | 0.0 |
| Standard (5 seeds) | 1.2 G | 0/5 | 81–101 | same region | n/a | 1.2 | 0.0 |

Full per-run numbers: `runs/grip_eval/results.json`. Chart: `docs/evaluation/grip_limit/speed_vs_distance.png`. Playback frame at the failure: `docs/evaluation/grip_limit/playback_failure_bic_1p0G.jpg`.

## 5. Where and why it fails
Every grip-limited run, on both tracks, fails at the **first real corner**, still at **4.00 px/step**. On BIC the car exits at T1 (~301 m, steering at full lock, heading error ~36°, 6 m lateral); T6/T8 are never reached. The steering demands ~11.6 G, grip allows 0.8–1.2 G, the clamp bleeds off the yaw rate (understeer), and the car runs wide. The three grip levels are indistinguishable (±1 m), because all are far below what the corner needs at that speed.

## 6. What this says about the policy (data only)
- Braking is 0.0% in all 20+ runs, grip on or off: throttle never goes negative and speed never drops below the 4.0 clamp after launch. Under grip limits the policy does not adapt its speed even when it is visibly failing.
- Its tracking success without grip is consistent with having been trained where lateral acceleration was unbounded (≈9–12 G demanded). We do **not** claim this about its internal representation — only that, as evaluated, it behaves as if cornering speed is unconstrained.
- Caveat: the existing clamp is understeer-only with no speed scrub, and the G figures rest on the unenforced 12 m = 70 px / 60 steps/s assumptions; results are about this simulator, not real cars. Only the first off-track is observed (episode terminates), so later corners under grip are untested.

## 7. Motivated follow-up
Grip-aware retraining (grip enabled in training, e.g. via the existing flag/randomization) with an explicit braking incentive, because the current reward gives no braking signal (audit: side rays pin `min_ahead`, so the corner-speed penalty is effectively constant). Not run here; whether it would succeed is untested.
