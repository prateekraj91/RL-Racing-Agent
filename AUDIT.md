# AUDIT — DEEP track, July 7 → July 16

**Audit date:** 2026-09-02 · **Branch:** `main` @ `9a48374` · **Mode:** read-only verification
**Updated:** 2026-09-02 after the continuous-API port — see *Remediation* below · **Now at** `1fef1be`
**Runtime:** Python 3.14.7 (`venv/`), gymnasium 1.3.0, torch 2.13.0, numpy 2.5.1, pygame 2.6.1

Every claim below was executed, not read. Scratch scripts were run under `/tmp` and deleted.
`AUDIT.md` is the only addition, and `git status` is clean apart from it.
`baselines/results.json` was **not** regenerated. One transient exception, disclosed for honesty:
running `main.py` to test it rewrote `track_1..20.png` (its loop at lines 17-28 saves them before
the crash at line 44). They were immediately restored with `git checkout -- track_*.png` and are
byte-identical to the originals; the image analysis reported below was performed *before* that run.

---

## Env API shape

`RacingEnv` is a standard Gymnasium env and passes the official checker:

```
$ venv/bin/python -c "... from gymnasium.utils.env_checker import check_env; check_env(RacingEnv(max_steps=200), skip_render_check=True)"
gymnasium check_env: PASSED
```

| | |
|---|---|
| `reset(seed=None, options=None)` | → `(obs, info)` — `env/environment.py:48` |
| `step(action)` | → `(obs, reward, terminated, truncated, info)` — `env/environment.py:115,267` |
| Observation space | `Box(-1000.0, 1000.0, (9,), float32)` — `environment.py:41` |
| **Action space** | **`Box([-1,-1], [1,1], (2,), float32)` — CONTINUOUS** — `environment.py:35` |

Action semantics: `action[0]` → steering (scaled by `car.max_steering`), `action[1]` → throttle
(`environment.py:123-128`).

### ACTION-SPACE FLAG — read this

**The action space is genuinely continuous, and the SAC wire-in is real, not faked.** There is no
discrete→continuous adapter because none is needed. July 14 is sound.

The problem is the **mirror image** of the one worth worrying about. Git shows the env was flipped
from `Discrete(5)` to `Box(2,)` on **2026-08-17** in `c6f554d` ("Complete SAC tiny-track overfit
experiment"):

```
$ git log -S'gym.spaces.Box(' --oneline -- env/environment.py
c6f554d 2026-08-17 Complete SAC tiny-track overfit experiment
8a8c18a 2026-07-22 Build initial Gymnasium racing environment
```

**That flip was never propagated backwards.** Five files still called `env.step(<int>)` against the
old discrete API and raised `TypeError: 'int' object is not subscriptable`. **All are now ported**
(see *Remediation*):

| File | Was broken at | Consequence | Status |
|---|---|---|---|
| `test_env_hardening.py` | 74, 142, 167, 192, 218, 243, 245, 249 | 7 of 10 hardening tests error out | **PORTED** `825cc22` |
| `baselines/pure_pursuit.py` | returns int from `choose_action` | pure-pursuit baseline dead | **PORTED** `fd33c38` |
| `baselines/benchmark.py` | inherits from `pure_pursuit` | benchmark dies before pure-pursuit | **PORTED** `fd33c38` |
| `main.py` | 44 | cannot regenerate `turning_circle.png` or `track_*.png` | **PORTED** `1fef1be` |
| `play.py` | 46-58 | manual keyboard mode dead | **PORTED** `825cc22` |

So SAC (Jul 14-16) was always in good shape; **July 7-12 is what had regressed — and is now fixed.**

### Action contract (pinned before porting, reused in every ported file)

Verified empirically against `env/environment.py:123-128` and `env/car.py:39-54` — not assumed:

```
action = np.array([steering, throttle], dtype=np.float32)     Box(-1.0, 1.0, (2,))

action[0]  STEERING  ->  car.steering = action[0] * car.max_steering (30 deg)
             +1.0 = full LEFT   (heading angle increases, CCW on screen)
             -1.0 = full RIGHT  (heading angle decreases, CW on screen)
action[1]  THROTTLE  ->  car.velocity += action[1] * car.acceleration (0.08)
             +1.0 = accelerate    -1.0 = brake / reverse
             velocity clamped to [-2.0, 4.0]; friction 0.03/step decays toward 0

heading_err = wrap180(car.angle - track.track_heading(x, y))
              > 0 -> car points LEFT of road    -> correct with NEGATIVE steering
signed_dist = track.signed_distance(x, y)
              > 0 -> car is RIGHT of centerline -> correct with POSITIVE steering

Car min turning radius at full steer = wheelbase / tan(30 deg) = 86.6 px
```

---

## Day-by-day results

| Day | Deliverable | Status | Evidence |
|---|---|---|---|
| **Jul 7** | Kinematic bicycle model integrates | **PASS** | Constant steer 20° + full throttle, 1200 steps: all finite, curved, no NaN. Least-squares circle fit on steady state → **R = 137.379**, theory `wheelbase/tan(20°)` = **137.374**; radius spread std = **0.0000**. `env/car.py:28-54` |
| | Turning-circle sanity plot as an image | **PARTIAL** *(non-blocking)* | `turning_circle.png` exists (17.8 KB, Aug 13) but renders only a **partial arc, not a closed circle** — weak as a sanity artifact. Its generator `main.py` crashed at line 44; **`main.py` is now ported** (`1fef1be`) and the plot *can* be regenerated, but was deliberately not re-run, because `main.py` rewrites the 20 committed `track_*.png` artifacts before reaching the plot. Cosmetic only — the underlying physics is verified exact to 3 decimals. |
| | wandb project `rl-racing-agent` wired | **PASS** *(fixed `1fef1be`)* | Was MISSING — zero hits in any commit in history. Now wired into `sac/train.py` behind `--wandb` (**default OFF**, so offline runs are byte-identical). Verified: `WANDB_MODE=offline python -m sac.train --wandb --steps 1200 --out-dir /tmp/sac_smoke` ran clean, created `wandb/offline-run-20260902_193114-rc0pv4v9/`, no errors in `debug.log`, project recorded as `rl-racing-agent`. Metric groups confirmed present in the run record: `episode/reward` `episode/length` `episode/lap_time` `episode/off_track_rate` `episode/mean_speed` `episode/mean_abs_steering` `episode/mean_abs_heading_err` `eval/*` `train/actor_loss` `train/alpha`. |
| **Jul 8** | Procedural varied drivable tracks | **PASS** | Generated 8 fresh tracks live: 320 pts each, perimeters 1179.7–1224.7 (**std 14.4 → varied**), **all `min_curv_R` ≥ 70** (73.9–83.6) → drivable, closure gap ≈ 3 px. Catmull-Rom closed spline + width, rejection-sampled on curvature: `env/track.py:31-64`. seed1 ≠ seed2 confirmed. |
| | ~20 saved track images, closed drivable loops | **PASS** | 20 files `track_1..20.png`, 800×600, **all 20 md5-unique, no duplicates**, road pixel counts 70,015–76,540. Visually inspected `track_3.png`: clean closed loop, road band + centerline. |
| **Jul 9** | obs = velocity, heading err, signed dist, slip, ray beams | **PASS** | Real vector printed below. All 9 components present and **verified responsive**, not placeholders. |
| | Beams computed and visualised | **PASS** | `cast_rays` `env/car.py:56-72`; `draw_rays` `env/car.py:74-94`. Rendered headless (`SDL_VIDEODRIVER=dummy`) → 5 red beams terminating in yellow dots exactly on the track edge; visually confirmed. Beams shift correctly under heading sweep (0→180°) and go asymmetric under lateral offset (−25→+25 px). |
| **Jul 10** | Gymnasium 5-tuple API | **PASS** | `check_env` PASSED (above). |
| | Reward v1 non-trivial under random agent | **PASS** | Full 2000-step random episode: `min=-0.17773 max=0.14964 mean=-0.00142 std=0.04025`, **1302 distinct reward values**, 1301/2000 non-zero. Not constant, not always 0. Components wired: progress + speed − crash (`environment.py:172-180`); crash penalty verified firing → `crash at step 53, reward=-7.2544`, `terminated=True`. |
| **Jul 11** | `test_env_hardening.py` passes | **PASS — 10/10** *(fixed `825cc22`)* | Was 3 passed / 7 errored on the stale discrete API. Ported to continuous action vectors; env behaviour itself was never changed. `venv/bin/python -m pytest test_env_hardening.py -q` → **`.......... 10 passed in 3.89s`**. Two tests are now stronger than before: determinism compares **full observation + reward trajectories** under a fixed action stream (not just post-reset state), and truncation asserts against the env default **`max_steps=2000`**. |
| | float32 obs, truncation, determinism | **PASS (verified manually)** | Bypassing the broken suite with continuous actions: reset & step both `float32`, shape `(9,)`, `observation_space.contains` → True. `max_steps=40` coast → `step_count=40, terminated=False, truncated=True`, mutually exclusive. **Determinism:** seed 42 twice with a fixed action sequence → trajectories `(301,9)` and rewards **bit-identical**; seed 43 differs (max abs diff 16.0). |
| **Jul 12** | Random-policy baseline | **PASS** | `python -m baselines.random_policy` runs on all 5 fixed seeds (101/202/303/404/505), 2000 steps each, 0 laps — correct for a random policy. |
| | Pure-pursuit / centerline heuristic w/ speed cap + logged lap times on 5 tracks | **PASS** *(fixed `fd33c38`)* | Was dead on arrival (TypeError on step 1) with stale pre-flip numbers. Rewritten for the continuous API: steering from lookahead heading error + signed cross-track term, throttle from proportional control against a **curvature-aware speed cap** that brakes when over cap. `baselines/results.json` **regenerated from scratch**. **Live results — `pure_pursuit` 5/5 laps on medium (avg lap 659.4 steps) and 5/5 on default (avg 575.0), zero crashes, 0.00% off-track; `random` 0/5 on both.** Both controllers × 5 seeds × 2 configs, all with real lap times. |
| **Jul 14** | Twin critics + stochastic actor + entropy temperature, real SAC | **PASS** | Twin critics `critic1`/`critic2` + deep-copied targets, `torch.min` on both target and actor paths (`sac/agent.py:19-27,41-42,111-114,159`). Stochastic tanh-squashed Gaussian actor with `rsample` and the `log(1-a²)` Jacobian correction (`sac/actor.py:37-53`). Learned temperature: `log_alpha`, `alpha_optimizer`, `target_entropy=-action_dim`, `update_alpha` (`agent.py:29-39,173-184`). **Not TD3** (TD3 has no entropy term and a deterministic actor); not a stub. |
| | Overfit one track — does reward climb? | **PASS (reproduced live)** | No reward-curve artifact was ever saved to disk, so I trained from scratch, 10k steps, eval every 500. **Medium track:** first-3-evals mean **412.00** → last-3 **1557.77** (**+1145.77**, Pearson r vs step **+0.765**), reaching lap completion at step 6000. **Default track:** **−83.90** → **1463.39** (**+1547.29**, r **+0.828**), lap at step 8000. Reward climbs decisively on both. Full curves below. |
| **Jul 15** | Replay buffer, warmup, eval-every-N, checkpoint-best | **PASS** | `ReplayBuffer` cap 100k, circular (`sac/replay_buffer.py`); `WARMUP_STEPS=1000` with random-action prefill (`sac/train.py:53,70-71`); `EVAL_EVERY=1000` (`train.py:54,161`); best-checkpoint save gated on eval progress (`train.py:209-215`). All four exercised and observed working in my live run. |
| | Logs lap time + off-track rate | **PASS** | `lap_time` `train.py:194`, `off_track_rate` `train.py:195`, both printed `train.py:199-207`. Observed live, e.g. `step 6000 … lap_time=412 off_rate=0.0%`. (Console only — no wandb, see Jul 7.) |
| **Jul 16** | Saved checkpoint exists | **PASS** | `best_actor.pth` (76,919 B) and `actor.pth` (76,465 B), distinct md5s. Both load cleanly into `SACActor` — all 8 tensors match architecture (`network.0/2`, `mean`, `log_std`). |
| | **Loading it completes a real lap on a medium track** | **PASS — GATE MET** | Deterministic eval (`tanh(mean)`, no sampling), medium track seed 101: **lap_completed=True, lap_time=403 steps, max_progress=1.0026, off_track_steps=0, reward=1539.00**. Identical for both checkpoints and at `max_steps` 500 and 3000. **Visually confirmed**: rendered the trajectory — a genuine closed loop fully inside the road band, hugging the inside line, start and end points coincident. Not a filename claim. |

---

## Raw evidence

**Jul 9 — real observation vector**

```
obs shape: (9,) dtype: float32     space: Box(-1000.0, 1000.0, (9,), float32)

--- reset(seed=101) ---            --- after 60 steps, steer=0.6 throttle=1.0 ---
  velocity         0.0000            velocity         3.0000
  heading_err      0.0000            heading_err     60.4400
  signed_dist     -0.0000            signed_dist    -45.0573
  slip             0.0000            slip            -1.1170
  ray-90          36.0000            ray-90           0.0000
  ray-45          60.0000            ray-45           0.0000
  ray0           112.0000            ray0             0.0000
  ray+45          48.0000            ray+45           0.0000
  ray+90          36.0000            ray+90           0.0000

RAW: array([3., 60.439972, -45.0573, -1.1169916, 0., 0., 0., 0., 0.], dtype=float32)
```

Sensor responsiveness (parked on centerline, rays react correctly to pose):

```
heading sweep:  angle=  0 rays=[108, 48, 36, 48, 132]
                angle= 90 rays=[36, 48, 132, 60, 36]
lateral offset: offset=-25 signed_dist=-25.00 rays=[64, 200, 64, 16, 12]
                offset=  0 signed_dist= -0.00 rays=[36, 60, 112, 48, 36]
                offset=+25 signed_dist=+24.99 rays=[12, 16, 136, 76, 60]
max |slip| over 300 steps of full steer+throttle: 1.720146   (non-zero → genuinely computed)
```

`slip` is `0.0` at `reset()` (`environment.py:96`) by design and is computed for real in `step()`
(`environment.py:210-219`) — intentional placeholder, not a bug.

**Jul 11 — `venv/bin/python test_env_hardening.py`**

```
Observation contract:
  ✓ obs type from reset()
  ✗ obs type from step() — UNEXPECTED ERROR: TypeError: 'int' object is not subscriptable
Deterministic seeding:
  ✓ same seed → identical state
  ✓ different seeds → different tracks
Termination logic:
  ✗ off-track → terminated=True — TypeError: 'int' object is not subscriptable
  ✗ max steps → truncated=True — TypeError: 'int' object is not subscriptable
  ✗ terminated & truncated mutually exclusive — TypeError: 'int' object is not subscriptable
Edge-case stress tests:
  ✗ sustained reversing (500 steps) — TypeError: 'int' object is not subscriptable
  ✗ spin in place (500 steps) — TypeError: 'int' object is not subscriptable
  ✗ extreme steering oscillation (500 steps) — TypeError: 'int' object is not subscriptable
=============================================
  3 passed, 7 failed
=============================================
```

**Jul 14 — fresh overfit run, medium track seed 101** (10k steps, eval every 500)

```
  step  eval_reward  progress    lap  lap_time  off_rate
   500      -128.06   -0.0773  False         0     1.3%
  1500      1492.13    0.8984  False         0     0.0%
  2000         0.00    0.0000  False         0     0.0%
  3000       129.99    0.0819  False         0     1.0%
  5000       141.01    0.0890  False         0     1.3%
  6000      1581.85    1.0008   True       412     0.0%
  7000      1586.53    1.0025   True       413     0.0%
  8000      1560.65    1.0004   True       406     0.0%
 10000      1556.42    1.0008   True       405     0.0%

first 3 evals: 412.00 → last 3: 1557.77    TREND: CLIMBING (+1145.77)   corr(step,reward)=+0.765
```

Reproduced laps of 405–413 steps independently match the shipped checkpoint's 403 — the saved
artifact is consistent with what training actually produces.

**Jul 16 — checkpoint gate**

```
best_actor.pth  max_steps=500  steps=403  lap_completed=True  lap_time=403  max_progress=1.0026  off_track=0  reward=1539.00
best_actor.pth  max_steps=3000 steps=403  lap_completed=True  lap_time=403  max_progress=1.0026  off_track=0  reward=1539.00
actor.pth       max_steps=500  steps=403  lap_completed=True  lap_time=403  max_progress=1.0026  off_track=0  reward=1539.00
```

---

## Bonus: pre-scouting for July 17

Not part of the audit gate, but directly useful for the failure-mode diagnosis. The policy
generalises unevenly — real, reproducible failure modes exist to study:

```
MEDIUM config (trained on seed 101):        DEFAULT config (base_r=210, n_ctrl=8, min_radius=70):
  seed 101  lap=True   progress=1.0026        seed 101  lap=False  progress=0.5401  CRASH @216
  seed 202  lap=True   progress=1.0017        seed 202  lap=False  progress=0.1417  CRASH @ 90
  seed 303  lap=False  progress=0.5078 CRASH  seed 303  lap=False  progress=0.1440  CRASH @ 89
  seed 404  lap=True   progress=1.0018
  seed 505  lap=True   progress=1.0018      → 5/7 laps on medium, 0/3 on tighter default tracks
  seed  42  lap=False  progress=0.1124 CRASH
```

Two clean starting threads: (a) mid-lap crash at ~50% progress on medium seed 303, (b) total
failure on the tighter default geometry, where a ~137 px minimum turning radius meets a 70 px
minimum track radius — the car may be **physically unable** to make some default-config corners.

**Code note (not a blocker):** `SACAgent.update` draws a batch at `agent.py:188-190`, then
`update_critics` draws an **independent second batch** at `agent.py:75`. The actor is therefore
updated on a different batch than the critics, and each `update()` costs two samples. Harmless to
correctness but worth knowing before you attribute any diagnosed instability to the algorithm.

---

## Blockers — ALL RESOLVED

Original findings, with resolution. Ordered as they were prioritised.

1. **`baselines/pure_pursuit.py` + `benchmark.py` are dead; the July 12 lap-time numbers are stale.**
   *Hardest blocker for July 17.* Diagnosing a policy's failure modes means comparing it against a
   reference controller — "is this corner hard, or is the agent bad at it?" is unanswerable without
   one. Right now the only heuristic that runs crashes at 11% progress. `results.json` looks
   authoritative and is not: it predates the continuous flip and the reward rescale.
   → **RESOLVED** (`fd33c38`). Ported; `results.json` regenerated. pure_pursuit now completes
   **5/5 laps on medium (avg 659.4 steps) and 5/5 on default (avg 575.0)**, zero crashes. You now
   have a working reference controller for the July 17 comparison.

2. **`test_env_hardening.py` — 7 of 10 tests silently broken.**
   You are about to change behaviour while diagnosing, with essentially no regression net. The env
   properties themselves are fine (I verified every one by hand), so this is a mechanical port of
   `step(0..4)` → continuous action vectors. Cheap and high value.
   → **RESOLVED** (`825cc22`). **10/10 passing** under pytest, with determinism and truncation
   coverage strengthened. `play.py` ported in the same commit.

3. **No wandb wiring at all** (project `rl-racing-agent` never existed).
   July 17 produces exactly the kind of output — per-track failure rates, crash locations, reward
   decompositions — that you want tracked rather than lost in terminal scrollback. Worth doing
   before the diagnosis, not after.
   → **RESOLVED** (`1fef1be`). `--wandb` flag (default OFF) logging episodic reward, lap time,
   off-track rate, episode length and speed/steering/heading summaries. Verified offline.

4. **`main.py` crashes at line 44**, so `turning_circle.png` and `track_*.png` cannot be
   regenerated, and the turning-circle plot shows only a partial arc rather than a closed circle.
   Lowest urgency — the underlying physics is verified exact and the images on disk are valid — but
   the sanity artifact is not currently reproducible.
   → **PARTIALLY RESOLVED** (`1fef1be`). `main.py` is ported and will now run, so the plot *is*
   reproducible. It was deliberately **not executed**: `main.py` rewrites the 20 committed
   `track_*.png` artifacts (lines 17-28) before it ever reaches the plot. Regenerating a full
   turning circle is a one-line change to its loop duration and is left as a cosmetic follow-up.

5. **Housekeeping** (trivial, flagged for completeness): `requirements.txt`, `env/physics.py`, and
   root `train.py` are all **0 bytes**; `configs/` is empty. Nothing imports `env/physics.py`, so
   nothing breaks — but an empty `requirements.txt` means this repo has no reproducible install.
   `play.py` is also on the dead discrete API.
   → **PARTIALLY RESOLVED.** `play.py` ported (`825cc22`). The empty files and `requirements.txt`
   are **still outstanding** — deliberately out of scope for this port-only pass. Not a July 17
   blocker, but the missing `requirements.txt` remains a real reproducibility gap.

---

## Remediation — continuous-API port (2026-09-02)

Three commits, one root cause. Env dynamics, reward, and the trained checkpoint were **not touched**.

| Commit | Concern | Result |
|---|---|---|
| `fd33c38` | Day 12 — baseline port + fresh `results.json` | pure_pursuit **5/5 medium**, **5/5 default**, 0 crashes |
| `825cc22` | Day 11 — hardening test port (+ `play.py`) | **10/10 pytest** |
| `1fef1be` | Day 7 — wandb wiring (+ `main.py`) | offline run logs all metric groups, no errors |

**Regression guards, all verified after the port:**

- Trained checkpoint still completes the same lap: `lap_completed=True, lap_time=403,
  max_progress=1.0026` — **bit-for-bit the pre-port audit numbers**, proving env dynamics and reward
  are unchanged.
- `md5 actor.pth best_actor.pth` unchanged (`d8d95a5a7273…`, `8a355bd5c044…`). The wandb smoke run
  wrote to `/tmp/sac_smoke` via the new `--out-dir`, never to the committed checkpoints.
- The 20 committed `track_*.png` and `turning_circle.png` are untouched (`git diff` empty);
  `main.py` was ported but deliberately never executed.
- `--wandb` defaults to OFF, so existing offline training is unaffected.
- No `env.step(<int>)` callers remain anywhere in the repo.

**Baseline vs trained agent — the July 17 comparison is now possible:**

| Config | random | pure_pursuit | SAC checkpoint |
|---|---|---|---|
| medium | 0/5 laps | **5/5**, avg 659.4 steps | **5/7 laps**, 403 steps on seed 101 |
| default | 0/5 laps | **5/5**, avg 575.0 steps | **0/3 laps** |

The trained agent is **~39% faster than pure-pursuit** where it succeeds, but pure-pursuit is far
more robust. Note especially the bottom-right cell: **pure-pursuit clears the tighter default
tracks 5/5 while the SAC policy fails all of them.** That settles the open question from the
pre-scouting section — those corners are *not* physically impossible, so the default-config failure
is a genuine policy defect, not a limit of the car. That is a well-posed July 17 investigation.

---

## VERDICT

# GO

**for starting July 17.**

All three blocking items are closed with evidence: **Jul 7 wandb → PASS**, **Jul 11 tests →
PASS (10/10)**, **Jul 12 baseline → PASS (5/5 laps, fresh `results.json`)**. The July 14/15/16 gate
was already met and is **re-verified unchanged** after the port — the checkpoint still turns in the
identical 403-step lap.

You now have the two things July 17 actually needs and previously lacked: a **working reference
controller** to compare the policy against, and a **place to log the diagnosis**. Plus a **10/10
regression net** so you can change things while diagnosing without flying blind.

### Residuals — none blocking

- `turning_circle.png` still shows a partial arc. `main.py` is ported so it can be regenerated, but
  running it rewrites the 20 committed track PNGs, so it was left alone. Cosmetic; the physics
  behind it is verified exact to 3 decimals.
- `requirements.txt`, `env/physics.py`, and root `train.py` are still 0 bytes; `configs/` is still
  empty. Out of scope for a port-only pass. The empty `requirements.txt` is a real reproducibility
  gap worth closing, but it does not gate the diagnosis.
- `SACAgent.update` still double-samples the replay buffer (`agent.py:75` vs `188-190`), so the
  actor trains on a different batch than the critics. Untouched by design — it is a training-loop
  change, not an API port. Worth knowing before attributing any diagnosed instability to SAC itself.

**Scope discipline:** this pass ported stale API callers and wired logging. No env dynamics, no
reward changes, no reward v2, no new features, no checkpoint retraining.
