# Cross-Difficulty Generalization — v2 Zero-Shot (Jul 30)

## Setup
v2 flagship (models/best_single_track_v2.pth, trained only on medium,
width=70, min_radius=80, seed 101) evaluated zero-shot, no retraining,
across two independent difficulty axes.

## Axis 1: min_radius (corner tightness) — NO GAP
Tested TIGHT_TARGET_CONFIG (min_radius 60 vs medium's 80, width unchanged)
across 10 seeds. Result: 10/10 lap_completed, zero crashes, lap times
(387-414 steps) inside medium's own run-to-run variance (403-450 steps).
Corner tightness alone produces no measurable degradation.

Mechanism: min_radius only changes what curvature the track GENERATOR
can produce; the agent's obs is purely local (ray-beams + heading-error
+ signed distance), so a tight corner is not a new class of observation
- it's a point already covered by the range of corners present on medium
tracks. Same root cause as the Jul 29 same-difficulty null result:
track-relative obs kills distribution shift for anything the local
geometry already spans.

## Axis 2: width (track narrowness) — REAL GAP, FOUND AND BRACKETED
Fixed min_radius=80 (medium's value), varied width only. 10 seeds per width.

  width=70 (trained on): baseline, avg_reward ~1350-1400
  width=40: 10/10 completed, avg_reward=1335.0  (mild decay, still solid)
  width=36: 10/10 completed, avg_reward=1328.3  (last fully solid width)
  width=32: 8/10 completed, avg_reward=1303.6   (cracks begin)
  width=28: 8/10 completed, avg_reward=1037.0   (avg drops hard - failures
                                                   are severe, not marginal)
  width=25: 6/10 completed, avg_reward=1032.8

Threshold: width 36 -> 32 is where zero-shot generalization first breaks.
Not a hard seed-independent cliff - seeds 2 and 8 fail starting at
width=32 and stay failed at every narrower width tested, while seeds
1 and 5 hold until width=25. Some track geometries are harder than
others under narrowing; the agent's margin is seed-dependent.

Failure mode: on most TIMEOUT cases reward collapses to 30-80 (vs
~1300 on completions), meaning the agent loses control and stalls
early rather than nearly finishing and running out of steps. This is
loss of control, not imprecision near the end.

## Interpretation
The two axes tested behave completely differently:
- min_radius: no shift, because the agent never conditioned on global
  track class, only local curvature it already spans.
- width: real shift, because narrowing directly shrinks the ray-beam
  readings and the safe heading-error margin the policy relies on -
  values it has never seen this small during training on width=70.

This is the first genuine, nonzero generalization gap found in the
project, and it is exactly where domain randomization should pay off:
training on a range of widths should push this threshold narrower.

## Status
Axis found and bracketed (width 36->32 threshold, seed-dependent).
Next: build a width-randomized trainer (sample width per reset from a
range spanning this threshold), retrain, then re-run this same sweep
on the new agent to measure how far the threshold moves.
