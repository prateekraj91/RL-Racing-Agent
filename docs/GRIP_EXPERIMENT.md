# Day 25 — The grip experiment: does behaviour emerge to fit the physics?

**Verdict up front: CONFIRMED.** Adding grip-limited cornering to the physics and retraining on a
track tight enough for it to bite produced **corner-specific braking** — a behaviour the reward-v2
agent does not have and, under the base physics, had no reason to acquire. Three caveats are
recorded at the bottom and they matter; the headline is real but narrower than "it learned to
race."

**Figure:** `analysis/figs/grip_speed_profile.png`
**Capture + analysis script:** `analysis/grip_speed_profile.py`
**Traces:** `analysis/runs/grip_agent_lap.npz`, `analysis/runs/v2_agent_lap_baseline.npz`

---

## Hypothesis

An agent develops a behaviour only when the physics reward it. It does not learn to drive "like a
racing driver" because racing is elegant; it learns whatever the dynamics make profitable.

The specific prediction: **reward-v2 drives flat out and never brakes because under the base
physics braking never pays.** The turn radius in `Car.update` is

```python
turning_radius = self.wheelbase / math.tan(math.radians(self.steering))   # env/car.py:62
```

which is **speed-independent**. The car turns exactly as tightly at v=4.0 as at v=1.0. Slowing
down therefore buys no extra cornering ability at all — it only costs progress reward. A
throttle-pinned policy is not a failure of exploration under those dynamics; it is the optimum.
`docs/TIMIDITY_CHECK.md` measured exactly that: post-launch mean speed 3.992, median 4.000, never
below 3.536, and it still beat the pure-pursuit expert by 29 steps.

If the hypothesis is right, then making turn radius depend on speed should make braking pay, and
the braking should appear on its own — with no change to the reward function.

## Method

### 1. The physics change

Grip-limited cornering was added to `Car.update` (`env/car.py:64-72`), gated behind
`car.grip_limit` so every pre-existing run keeps its meaning:

```python
if self.grip_limit:
    lateral_accel = (self.velocity ** 2) / abs(turning_radius)
    if lateral_accel > self.max_grip:                       # max_grip = 0.09, env/car.py:30
        grip_radius = (self.velocity ** 2) / self.max_grip  # understeer: wash wide
        turning_radius = math.copysign(grip_radius, turning_radius)
```

Demand more lateral acceleration than the tyres have and the car understeers — the radius widens
to the tightest arc the grip actually allows. The consequence is a speed-dependent corner limit:
the fastest a car can hold a corner of radius `R` is `sqrt(max_grip * R)`. At the 4.0 speed cap
that means no corner tighter than `4.0² / 0.09 = 177.8px` can be taken flat out.

The reward function was **not** touched. Whatever behaviour appears has to come from the dynamics.

### 2. First attempt — grip was on, and inert

`runs/grip_agent_s42` trained with grip on, on the *medium* track (seed 101, `min_radius 80`).
Result: solved, 406 steps, mean speed 3.558 — indistinguishable from non-grip v2. The clamp fired
on ~38% of steps yet cost so little yaw that flat-out still lapped clean. Medium's realized
tightest corner is R=120.7px, only 21px inside the grip floor at v=3.57 (141.6px), and a 70px band
gives ±35px of lateral room to absorb that much understeer.

**Grip was on and effectively inert.** A physics knob that changes no behaviour tests no hypothesis.

### 3. A track where grip bites

The fix was tighter geometry, not a lower `max_grip` — keeping `max_grip=0.09` constant leaves
every run in the repo comparable. 2100 `(min_radius, seed)` pairs were generated and scored on
realized minimum radius, lap length, and self-clearance. **`seed=24` at `min_radius=60`** was
selected (`sac/curricula.py:123-135`):

| property | value |
|---|---|
| realized tightest corner | **64.5px** |
| lap length | 1422.4px (within 1% of medium's 1435.2, so lap times stay comparable) |
| self-clearance | 99.2px (> width 70, so the band never overlaps itself) |
| centerline below the v=4.0 grip floor | 38.8% |

Verified to bite before any training was run — same controller, same track, grip the only difference:

| controller | grip | outcome |
|---|---|---|
| flat out (throttle pinned +1) | **ON** | **crashes at 53% progress** |
| flat out (throttle pinned +1) | OFF | clean lap, 398 steps, v=3.60 |
| grip-aware expert (`v ≤ sqrt(max_grip·R)`) | ON | clean lap, 596 steps |

"Just floor it down the centerline" is now a crash, and slowing for the corner is what fixes it.

### 4. Training

```
python -m sac.train_curriculum --curriculum tight_width3 --seed 42 \
    --name grip_bites_curric_s42 --demos 40 --target tight --grip
```

Demo-seeded (40 grip-aware expert episodes), 3-tier width ladder (110 → 90 → 70) on the same
seed-24 centerline, 200k steps, `max_steps=700`. Solved at step 95k; 526 episodes, 115 crashes,
270.8 min wall. Checkpoint: `runs/grip_bites_curric_s42/solved_actor.pth`.

### 5. Measurement

One deterministic (`tanh(mean)`, no sampling) lap in the exact world it trained in — tight track
seed 24, `grip_limit=True`, `max_steps=700` — logging per step position, speed, throttle,
lap progress, and the **ground-truth road curvature** at the car (`Track.corner_radius`,
`env/track.py:128`). The ray sensor cannot serve this role: at width 70 the ±90° rays are pinned
at ~36px everywhere, so `min(rays)` is constant and says nothing about the corner ahead.

The lap reproduces the training log exactly — **432 steps, `lap_completed=True`, `crashed=False`,
mean speed 3.360** vs the logged `solved_eval` of 432 / 3.3597.

The non-grip v2 agent (`runs/exp2_demos_s42/solved_actor.pth`) was captured the same way on *its*
target — medium, seed 101, grip OFF — as the baseline contrast.

A **corner** is defined physically and identically for both tracks: road radius < 177.8px, i.e.
`v_cap² / max_grip`, the tightest arc holdable flat out under grip.

---

## Result

### The control signal is the decisive evidence

Speed here is set *only* by throttle — `Car.update` applies a constant 0.03 friction and steering
does not scrub speed (`env/car.py:16-18`), so holding speed needs `throttle = 0.03/0.08 = 0.375`
and anything below 0 is a real brake. Speed is the *integral* of throttle and therefore lags;
throttle shows the decision without the lag.

| | corr(curvature, throttle) | mean throttle, corner | mean throttle, straight | steps braking (throttle < 0) |
|---|---|---|---|---|
| **grip agent** | **−0.735** | **+0.252** | +0.843 | **29% of corner steps, 0% of straight steps** |
| **v2 baseline** | −0.505 | +0.910 | +0.933 | **0% — anywhere, ever** |

v2's −0.505 looks superficially similar and is **cosmetic**: it lifts from 0.93 to 0.91, still far
above the 0.375 that holds speed. It produces no speed change whatsoever. The grip agent goes to
**−0.93** — full brake. v2's throttle never even drops below 0.783 across the whole lap.

### corr(track curvature, speed)

| | whole lap | excl. standing start |
|---|---|---|
| **grip agent** | **−0.586** (Spearman −0.432) | **−0.340** (Spearman −0.305) |
| **v2 baseline** | −0.018 (Spearman −0.074) | +0.040 (Spearman +0.037) |

Both agents start from rest, so the opening acceleration ramp (88 steps for the grip agent, 77 for
v2) is a speed change unrelated to corners; it is reported excluded as well as included. The grip
agent is clearly more negative on every measure. **v2's is essentially zero, and post-launch it is
faintly positive.**

### Corner vs straight speed

| | corner | straight | gap |
|---|---|---|---|
| grip, whole lap | 2.94 | 3.65 | **+0.71** |
| grip, excl. standing start | 3.64 | 3.84 | +0.20 |
| grip, tightest third vs most-open third (excl. start) | 3.62 | 3.90 | +0.28 |
| **v2, excl. standing start** | **4.000** | **3.993** | **−0.007** |

### Where the slowdown actually lives

The grip agent brakes at exactly two of the five corners — the two tight enough that grip bites:

| corner | road radius | speed through it |
|---|---|---|
| p 0.18–0.23 | 96px | 4.00 → 3.90 (flat out) |
| **p 0.40–0.55** | **86.6px** | **4.00 → 2.20 (hard brake, throttle to −0.74)** |
| p 0.65–0.69 | 115px | 4.00 (flat out) |
| p 0.82–0.85 | 148px | 4.00 (flat out) |
| **p 0.96–1.00** | **64.5px** (tightest on track) | **4.00 → 2.55 (hard brake, throttle to −0.93)** |

**This is not a cautious crawl.** 69.8% of the post-launch lap is pinned at the 4.0 speed cap.
Outside the standing start and those two braking zones the grip agent averages **3.961** —
statistically the same as v2's **3.994**. The entire 3.36-vs-3.57 mean-speed deficit is localized
to two corners; there is no meaningful uniform-slowdown component.

## Verdict

**CONFIRMED.** Grip physics produced **corner-specific speed management** that the non-grip agent
lacks entirely. The reward function was unchanged between the two, so the behaviour came from the
dynamics: once turn radius depended on speed, braking bought cornering ability, and the agent
found it.

The hypothesis's negative half also holds up under direct measurement. v2 does not brake *once* in
402 steps, and its corner and straight speeds differ by 0.007 px/step. Under speed-independent
turn radius, flat-out is not timidity's opposite — it is simply correct.

## Honest caveats

**1. Selective, not general.** The agent brakes for 2 of 5 corners, not for corners as a category.
It stays flat out through radii of 96, 115 and 148px and brakes only at 86.6 and 64.5px — the two
that approach or breach the grip floor. That is arguably the *right* policy, but the accurate
claim is "it learned to brake for the corners that bite," not "it learned to brake for corners."

**2. Reactive, not anticipatory.** Throttle correlates most strongly with curvature about 5 steps
ahead (−0.784 at lag +5 vs −0.735 at lag 0, decaying to −0.21 by lag +20) — roughly 20px of
look-ahead. The agent lifts at corner *entry*; it has not learned the long braking zone a human
driver uses, and it does not trade entry speed for exit speed. There is no evidence here of a
racing line, only of a speed limit being respected late.

**3. Track confound — the comparison is not fully controlled.** The grip agent was trained and
measured on the tight track (tightest radius 64.5px); v2 was trained and measured on medium
(tightest 121.3px). So part of the contrast is "harder corners," not purely "grip physics."
What survives the confound: v2's own track has sub-threshold corners on 17.9% of its steps and v2
brakes on **none** of them, while the grip agent's braking lands precisely on its two tightest.
That is suggestive, not conclusive.

A clean version of this experiment holds the track fixed and varies only grip — train two agents
on the *same* tight seed-24 track, one with `--grip` and one without, and compare. That was not
run. The blocker is that the tight track is what makes grip bite, and the earlier medium-track
attempt (`runs/grip_agent_s42`, §2 above) showed the reverse problem: on a track where grip is
inert, grip-on and grip-off agents are indistinguishable, so holding the track fixed *there*
tests nothing either. The honest statement is that a track where grip bites and a controlled
grip-on/grip-off pair on that same track is the missing run.
