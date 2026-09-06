# Policy Failure-Mode Diagnosis — `best_actor.pth`

**Task:** Day 17 (Master Plan v3) — study the trained policy, characterise its failure
modes, and use them to define reward v2.

**One-line finding:** the trained SAC policy has learned exactly one behaviour —
*hold near-max throttle and hold a single fixed steering angle in one direction* — which
completes the medium track but washes off the outside of the first corner tighter than its
fixed turn radius. Every failure mode traces directly to reward v1 paying an unconditional
per-step bonus for speed.

**Crucially, this is a policy defect, not a physics limit.** A hand-written pure-pursuit
controller drives the default track 5/5 (see baselines), so the geometry is drivable. The
policy fails where a trivial controller succeeds.

---

## Method

Two deterministic rollouts of `best_actor.pth` (greedy action = `tanh(mean)`, no sampling),
captured to `.npz` and analysed offline:

- `medium_seed101` — the config the policy passes (`base_r=250, min_radius=80`).
- `default_seed101` — a config it fails (`base_r=210, min_radius=70`; tighter, smaller).

Per step we logged steering, throttle, velocity, position, reward and lap progress.
Scripts: `analysis/collect_run.py`, `analysis/plot_actions.py`, `analysis/plot_trajectory.py`.
Figures: `analysis/figs/`.

Reference points:

| Run | Result | Steps | Final progress |
|---|---|---|---|
| medium_seed101 | lap completed | 403 | 1.0026 |
| default_seed101 | **crashed** | 216 | **0.5401** |
| pure_pursuit (baseline) | 5/5 medium, 5/5 default | — | — |
| random (baseline) | 0/5 both | — | — |

---

## Failure modes

### FM-1 — Uni-directional steering (turn-handedness overfit)

**Evidence (medium):** steering ∈ **[−0.664, −0.244]**, mean −0.410.
**100 % of steps steer one direction; 0 % the other; it never even approaches straight
(zero).** Figure: `figs/steer_hist.png`.

> Correction to the Step-2 axis label: per the verified action contract
> (`train.py` header; `action[0] = +1 LEFT, −1 RIGHT`), negative steering is **RIGHT**.
> So the policy steers *right* the entire lap and drives the loop clockwise. The histogram
> axis label ("<0 = left") is inverted — cosmetic only; the finding (one-directional,
> never straight) stands.

**Mechanism:** the policy memorised this track's single turn direction. It has no
representation of turning the other way or of driving straight. On any track containing an
opposite-handed corner it is physically unable to follow the road.

**Fix path:** **NOT a reward fix.** This is a training-distribution problem — the agent only
ever saw one track's handedness. Deferred to the multi-track + domain-randomisation work
(Master Plan Jul 28+). Reward v2 must stay direction-symmetric (progress already is) so it
does not bake in a preferred hand.

### FM-2 — No braking / no longitudinal control

**Evidence (medium):** throttle ∈ **[0.931, 0.957]**, mean 0.946.
**0 % of steps brake; the entire throttle output is a 0.026-wide band pinned near max.**
Figure: `figs/throttle_hist.png`.

**Mechanism:** the policy does not control speed at all — it holds the gas near the ceiling
and relies on the env's `velocity` cap (4.0) to avoid flying off. It never lifts, never brakes.

**Fix path:** **reward v2** — this is the primary target (see Root Cause).

### FM-3 — Single operating speed (flat through corners)

**Evidence (medium):** velocity ramps 0 → 4.0 within the first ~13 % of the lap, then sits
**flat at the 4.0 cap for the remaining ~87 %** — no dip anywhere, corners included.
Figure: `figs/speed_profile.png`.

**Mechanism:** with throttle constant (FM-2), speed is constant too. The car carries the
same maximum velocity into a hairpin as down a straight. It has no slow-in/fast-out, no
speed-for-corner trade — because nothing ever rewarded going slower. (Note: the flat top is
the *env cap* clamping it, not a learned choice; the policy would go faster if allowed.)

**Fix path:** **reward v2** — same root cause as FM-2.

### FM-4 — Fixed turn radius → washes wide on any tighter corner (the crash)

**Evidence (default):** crashes at **54 % progress, step 216, at velocity ≈ 4.0**.
Figure: `figs/trajectory_default_seed101.png` — the driven path (bright yellow = full speed)
holds the centreline along the bottom, then drifts steadily outward through the sustained
left-side corner and the crash marker sits **on the outer boundary**. It runs **wide off the
outside**, not cutting the inside.

**Mechanism:** fixed steering angle (FM-1) + fixed max speed (FM-2/3) ⇒ the car has exactly
**one turning radius**. Any corner tighter than that radius, it cannot hold — it understeers
off the outside at full speed. Default's smaller `min_radius` (70 vs 80) is the first corner
tight enough to expose this. It does not fail at a sharp mid-lap hairpin; it fails on the
first *sustained* corner below its fixed radius.

**Fix path:** **reward v2** (remove the incentive to carry full speed into corners) **+ FM-1
fix** (so it can also tighten the line by steering more). The crash at full speed is the
visual proof that braking-before-corners is the missing behaviour.

---

## Root cause — reward v1

The entire failure set is the rational, optimal solution to the current reward. From
`env/environment.py::step()`:

```python
speed_reward = self.car.velocity * 0.01
reward = progress + speed_reward
if crashed:
    reward -= 0.1
reward *= 100
```

Read as the agent reads it:

- `progress` — reward for advancing along the track. **Correct core signal.**
- `speed_reward = velocity * 0.01` — **an unconditional per-step bonus for going fast**,
  anywhere on the track, regardless of what's ahead. This is the villain.
- `if crashed: reward -= 0.1` — the *only* thing discouraging recklessness, and it is
  far too weak and too late: it fires once, at the instant the car is already off-track,
  with no warning.
- `reward *= 100` — global scale only; no behavioural effect.

**The decisive number.** Per step over a ~403-step lap, `progress ≈ 1/403 ≈ 0.0025`, while
`speed_reward` at v=4 is `0.04`. **The speed bonus is ~16× the progress signal every step.**
So v1 is overwhelmingly "go fast"; progress is almost noise beside it. Under that objective,
the globally optimal policy is: floor the throttle always, and turn just enough to keep
banking speed points. Braking scores strictly *worse*. The agent did exactly this. It is not
broken — it is a correct optimiser of a reward that describes a reckless driver.

This is why more training will not help: it only sharpens the wrong objective.

---

## Reward v2 spec

Goal: make *"as fast as you can go while still making the corner"* the highest-scoring
behaviour, so the agent discovers braking-for-corners on its own.

### Change 1 — remove the unconditional speed bonus (targets FM-2, FM-3)

Progress already requires moving forward, so speed stays instrumental: fast only pays if it
yields progress, and progress stops when you crash. Drop the standalone term.

### Change 2 — curvature-aware speed penalty (targets FM-4)

Penalise carrying high speed when the road ahead is tight. The reward can't read true
curvature cheaply, but the **forward ray beams** (`obs[4:9]`) already sense it: a close wall
ahead ⇒ small min-ray ⇒ a corner. Penalty scales with *both* how tight the corner is *and*
how fast you're going, so braking-before-a-corner becomes the higher-scoring choice.

### Change 3 — steering-smoothness penalty (targets FM-4 jitter)

Small cost on the step-to-step change in steering, to stop the jerky actuation.

### Change 4 — stronger crash penalty

−0.1 is trivial against hundreds of accumulated speed points. Make a crash clearly negative
so recklessness stops paying.

### Concrete diff (`env/environment.py`)

**In `reset()`**, initialise the previous-steering memory (needed for Change 3):

```python
self.prev_steering = 0.0
```

**In `step()`**, replace the reward block:

```python
# ---- reward v1 (remove) ----
speed_reward = self.car.velocity * 0.01
reward = progress + speed_reward
if crashed:
    reward -= 0.1
reward *= 100
```

with reward v2:

```python
# ---- reward v2 ----
# 1. Core signal: progress along the track (speed is now instrumental, not paid directly)
reward = progress

# 2. Curvature-aware speed penalty — punish carrying speed into a tight corner.
#    rays = forward beam distances (same as obs[4:9]); small min = wall/corner close.
rays = self.car.cast_rays(self.track)
min_ahead = min(rays)
CORNER_THRESH = 120.0                       # beam distance below which a corner is "close"  [TUNE]
if min_ahead < CORNER_THRESH:
    tightness  = (CORNER_THRESH - min_ahead) / CORNER_THRESH   # 0..1
    speed_frac = max(self.car.velocity, 0.0) / 4.0             # 0..1 (v_max = 4)
    reward -= 0.02 * tightness * speed_frac                    # weight  [TUNE]

# 3. Steering-smoothness penalty — kill the jitter (FM-4)
steer_change = abs(steering_action - self.prev_steering)
reward -= 0.01 * steer_change                                  # weight  [TUNE]
self.prev_steering = steering_action

# 4. Crash penalty — large enough that recklessness is clearly net-negative
if crashed:
    reward -= 2.0                                              # weight  [TUNE]

reward *= 100
```

Notes on this diff:

- `rays` is currently computed *after* the reward block in `step()`. For v2 it's computed
  inside the reward block (shown above); leave the later `rays = self.car.cast_rays(...)`
  used for the observation as-is, or reuse this one — both are fine, it's a cheap call.
- All four weights (`CORNER_THRESH`, `0.02`, `0.01`, `2.0`) are **starting points, not final
  values.** Dial them in on the Jul 22 hyperparameter sweep. Keep the balance such that the
  progress gained by cornering cleanly still exceeds the penalties — otherwise the agent
  learns to crawl or stop.
- If learning bootstraps too slowly with progress as the only positive term, re-introduce a
  *small* speed term **gated on a clear road ahead** (e.g. only when `min_ahead > CORNER_THRESH`)
  — never the unconditional v1 form.

### Optional refinement — potential-based shaping (Ng 1999)

To reward hugging the racing line without opening a reward hack, add potential-based shaping
on `signed_dist` (already in `obs[2]`): `F = γ·Φ(s') − Φ(s)` with `Φ` decreasing in distance
from centre. Potential-based shaping is provably policy-invariant, so it can't create a new
loophole. Left as a refinement after the four core changes land.

---

## Summary table

| # | Failure mode | Hard evidence | Figure | Fix |
|---|---|---|---|---|
| FM-1 | Only steers one way, never straight | steer 100 % one-signed, [−0.66, −0.24] | steer_hist | multi-track / DR (Jul 28) |
| FM-2 | Never brakes | throttle 0 % <0, [0.93, 0.96] | throttle_hist | **reward v2 (C1)** |
| FM-3 | No speed control (flat 4.0 in corners) | speed flat after ~13 % progress | speed_profile | **reward v2 (C1)** |
| FM-4 | Fixed radius → washes wide off tight corners | crash at 54 %, outer edge, v≈4.0 | trajectory_default_seed101 | **reward v2 (C2–4) + FM-1 fix** |

**Root cause:** reward v1's unconditional speed bonus (~16× the per-step progress signal) plus
a trivial crash penalty make "floor it everywhere" optimal. Reward v2 removes the speed bonus,
penalises speed-into-corners and steering jitter, and hardens the crash penalty, so
brake-then-accelerate becomes the higher-scoring behaviour and the agent can learn a real line.

**Next (separate task, Jul 19):** implement reward v2, retrain, and re-run this exact
diagnosis on the new checkpoint to confirm the throttle histogram gains a braking tail and the
speed profile dips through corners.
