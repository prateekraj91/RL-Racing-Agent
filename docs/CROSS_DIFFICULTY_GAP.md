# Cross-Difficulty Generalization — v2 Zero-Shot (Jul 30, corrected)

## Correction notice
The first version of this doc (commit 91cb1ad..63ef4fb) used env.reset(seed=N),
which does NOT set the track directly - it seeds Gymnasium's RNG, which then
DERIVES an unrelated track_seed. Every seed label in that version was wrong.
This version uses env.reset(options={"track_seed": N}), which sets the track
seed directly (confirmed by reading RacingEnv.reset() and cross-checking
against sac/visualize.py, which already used the correct call). All numbers
below are re-measured with correct seeding.

## Setup
v2 flagship (models/best_single_track_v2.pth, trained only on medium,
width=70, min_radius=80, seed 101) evaluated zero-shot, no retraining,
across two independent difficulty axes.

## Axis 1: min_radius (corner tightness) - NO GAP
Tested TIGHT_TARGET_CONFIG (min_radius 60 vs medium's 80, width unchanged)
across 10 seeds. 2 of 10 (seeds 101, 6) produced a degenerate track/spawn
where the car is stuck off-track from step 0 (reward=-1.0 for the full
episode, confirmed visually via sac.visualize) - a track-generation edge
case specific to pairing this config with those seeds, not a driving
failure. TIGHT_TARGET_CONFIG's own comment notes it is tuned for seed 24
specifically.

Excluding those 2: 8/8 completed, zero crashes, rewards in line with
medium's own variance. Corner tightness alone produces no measurable
driving degradation.

Mechanism: min_radius only changes what curvature the track GENERATOR can
produce; the agent's obs is purely local (ray-beams + heading-error +
signed distance), so a tight corner is not a new class of observation -
it is a point already covered by the range of corners present on medium
tracks. Same root cause as the Jul 29 same-difficulty null result.

## Axis 2: width (track narrowness) - REAL GAP, GRADUAL, NOT A HARD CLIFF
Fixed min_radius=80 (medium's value), varied width only. 10 seeds per width,
seeds re-verified with correct track_seed passing.

  width=70 (trained on): baseline, avg_reward ~1330-1400
  width=40: 10/10 completed, avg_reward=1335.1
  width=36: 10/10 completed, avg_reward=1334.9
  width=32: 10/10 completed, avg_reward=1322.1
  width=28: 9/10  completed, avg_reward=1265.0  (first crack, seed 9 times out)
  width=25: 8/10  completed, avg_reward=1160.6  (seeds 3, 9 time out; seed 9
                                                   reward collapses to 31.0)

Corrected threshold: fully solid through width=32 (10/10). Degradation is
gradual, not a cliff - first failures appear at width=28, worsen at 25.
The earlier reported "hard cliff at 36->32" and "seeds 2/8 consistently
fail" pattern were artifacts of the seeding bug and are retracted - with
correct seeding, no stable seed-specific difficulty pattern is evident
(different seeds fail at width=28 vs width=25), though 10 seeds is too
few to rule this out definitively.

## Interpretation
min_radius and width behave completely differently under zero-shot
generalization:
- min_radius: no shift - the agent never conditioned on global track
  class, only local curvature it already spans during training.
- width: real, gradual shift - narrowing directly shrinks the ray-beam
  readings and the safe heading-error margin, and degradation appears
  once width drops meaningfully below the trained value of 70 (first
  measurable cracks around width=28, roughly 60% of training width).

This is the first genuine, nonzero generalization gap in the project.
It is gradual rather than catastrophic, which changes how domain
randomization should be framed: not "fix a broken regime" but "extend
the solid range further before gradual decay sets in."

## Status
Axis found, corrected, and quantified. min_radius: solved (no gap once
degenerate seeds excluded). width: real gradual gap starting ~width=28.
Next: build a width-randomized trainer, retrain, re-run this exact sweep
on the new agent to measure how far the solid range extends.


## Addendum: failure mode confirmed visually (width=25, seed=9)

Watched via sac.visualize (narrow25 config, seed 9 - the reward=31.0 case).
The car does not crash and does not oscillate. It decelerates smoothly
through a tight section, velocity reaches exactly 0, and then STAYS at 0
for the remainder of the episode - action output frozen (same steering/
throttle repeated every step). This is a stall/wedge, not a collision.

Interpretation: v2 never stalls on width=70 (its trained regime), so it
never learned any recovery behavior for near-zero-velocity states. When
a narrow corridor forces it to slow enough to hit zero, there is nothing
in its policy to un-stick it - it is an unvisited state, not a misjudged
one.

Implication for the randomized trainer: training on narrower widths alone
may not be sufficient if the agent still eventually finds a width narrow
enough to stall at. The trainer should also expose the agent to low/zero-
velocity recovery during training (e.g. via demo-seeding laps that include
tight, near-stall maneuvering, or simply enough steps at narrow widths for
stalls to occur and be penalized/recovered from during exploration) rather
than assuming narrower-width exposure alone teaches recovery.
