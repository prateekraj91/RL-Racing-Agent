# Day 28 — The generalization testbed, and what the single-track agent already generalises

Everything before today measured the agent on the tracks it was trained on. That cannot
distinguish an agent that **learned to drive** from one that **memorised a track**. This day adds
the infrastructure that can tell the two apart — two disjoint track pools and a pool evaluator —
and runs the locked single-track agent through it as the baseline.

The headline: **v2, trained on exactly one track, completes 20/20 training-pool tracks and 10/10
held-out tracks with no measurable generalization gap.**

## Purpose

A "track" here is `(seed, config)`. The testbed holds the **config fixed** and varies only the
**seed**, so every track in both pools has the same difficulty and differs only in *shape*.

`sac/track_pools.py`:

| | |
|---|---|
| **Shared config** | `width 70, base_r 250, n_ctrl 10, min_radius 80` (cx 400, cy 300), `max_steps 700` |
| **TRAIN pool** | seeds **1–20** (20 tracks) — the agent may train on these |
| **TEST pool (held-out)** | seeds **1001–1010** (10 tracks) — never trained on |

The two pools are disjoint by construction, and the module enforces it at import time:

```python
assert set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS), "TRAIN and TEST seeds overlap!"
```

That assert is the point. If the pools ever drift into overlap, the import crashes and nothing
downstream runs — a dishonest generalization test cannot be executed by accident.

## Instrument

`sac/evaluate_pool.py` is the measuring device. Given a checkpoint and a list of seeds it runs one
**deterministic** lap per track (`tanh(mean)`, no exploration noise) and reports, per track:
completion, lap time (steps, `DNF` if the lap never closes), and final lap progress — then the pool
aggregate: tracks completed, average lap time over completed laps, and average progress.

Run over `TRAIN_SEEDS` and `TEST_SEEDS`, the difference between the two pool summaries **is** the
generalization gap.

```
python -m sac.evaluate_pool --checkpoint models/best_single_track_v2.pth
```

## Baseline: the single-track agent (v2)

Checkpoint `models/best_single_track_v2.pth` — demo-seeded SAC trained on **`track_seed=101`
alone** (`--curriculum none --seed 42 --demos 40`), the agent locked on Day 26. It has seen no
track in either pool.

Measured, fresh run of `python -m sac.evaluate_pool` (default checkpoint):

| Pool | Completed | Avg lap (completed) | Lap range | SD | Avg progress |
|---|---|---|---|---|---|
| TRAIN (seeds 1–20) | **20/20** | **400.6** steps | 384–426 | 10.0 | 1.001 |
| TEST, held-out (seeds 1001–1010) | **10/10** | **398.9** steps | 386–409 | 7.7 | 1.002 |

No DNFs and no crashes in either pool; every one of the 30 tracks reached progress ≈ 1.00.

**Generalization gap: 20/20 vs 10/10 on completion, and −1.7 steps on lap time** — the held-out
pool was marginally *faster*, which is inside the per-track spread (SD ≈ 8–10 steps), so the
honest reading is a gap of zero. For scale, the same agent's locked medium-track benchmark is
~401.8 steps, so both pools reproduce its home-track pace on tracks it has never driven.

## Key insight: the state representation, not the track count

v2 trained on **one** track and generalises near-perfectly across same-difficulty shapes. The
reason is in the observation vector, not the training distribution. Per `env/environment.py`, the
agent sees:

```
[ velocity, heading_error, signed_distance_to_centerline, slip, ray_0 … ray_4 ]
```

with the five rays cast at −90°/−45°/0°/+45°/+90° **relative to the car's heading**
(`env/car.py:cast_rays`). Every component is **track-relative**: heading error against the local
track heading, signed offset from the centerline, and beam distances to the track edge from the
car's own frame. There is no absolute position, no waypoint index, no track identity anywhere in
the input.

So a left-hand corner of a given radius produces essentially the **same observation signature** on
seed 7 and on seed 1007, and the policy that was learned for that signature on seed 101 transfers
unchanged. The agent never had a track to memorise — the representation gave it nothing but the
local geometry it is actually reacting to.

**State representation is what produced this generalization, not the number of training tracks.**

## Implication for what comes next

Same-difficulty, different-shape generalization is essentially solved, and the testbed's job now is
mainly to keep it honest as the agent changes. It is not a frontier to push on:

- **Multi-track training on same-difficulty tracks should be expected to show little or no
  improvement over this baseline.** The ceiling here is already 30/30. Running it is still worth it
  as a control — but a flat result is the prediction, not a failure.
- **The real frontier is cross-difficulty generalization**: tracks whose *kind* the agent has never
  seen — different widths, different base/min radii, different friction — where the observation
  signature of a corner genuinely changes (narrower corridors compress the ray returns, tighter
  radii demand a different speed/steer response, lower grip breaks the learned control mapping).
  That is the domain-randomization phase, and that is where a real gap should be expected to appear.

To extend the testbed for that phase, the thing to vary is `POOL_CONFIG["track_kwargs"]` — a pool
per difficulty class, held disjoint the same way.

## Files

| | |
|---|---|
| Pool definitions | `sac/track_pools.py` |
| Pool evaluator | `sac/evaluate_pool.py` |
| Baseline weights | `models/best_single_track_v2.pth` (`models/best_single_track_v2_config.json`) |
| Baseline provenance | `docs/BEST_SINGLE_TRACK.md` |
