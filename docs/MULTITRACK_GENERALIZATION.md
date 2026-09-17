# Multi-Track Training — Generalization Gap (Jul 29)

## Experiment
Trained SAC on 5 tracks (seeds 1–5, medium config), evaluated zero-shot
on 3 unseen tracks (seeds 1001–1003). Agent: runs/multitrack5_s42/best_actor.pth
(solved at step 5000).

## Results
TRAIN (seeds 1–5):     5/5 complete | avg lap 398.6 | range 389–407
TEST  (seeds 1001–3):  3/3 complete | avg lap 391.3 | range 385–398

Generalization gap = 398.6 − 391.3 = −7.3 steps (test FASTER by 1.8%).
Well inside within-pool variance (18-step train spread). => gap ≈ 0.

## Interpretation
Gap is statistically indistinguishable from zero, sign is meaningless
(test tracks slightly shorter, not better learned).

KEY POINT — this is a NULL RESULT and a CONTROL, not an improvement:
v2 (single-track, Jul 28) already generalized 30/30 zero-shot with gap≈0.
So 5-track training added NO measurable same-difficulty generalization
over 1-track training. Reason: track-relative observation removes
distribution shift between same-difficulty tracks, so there is nothing
to generalize to — it's interpolation within one distribution. Training
diversity is redundant when the encoding already kills the shift.

Implication: the payoff of multi-track / diverse training will only
appear when there IS distribution shift — i.e. CROSS-difficulty tracks
(varying widths/radii). That is domain randomization = next task.

## Metric note
Both pools 100% complete, so lap-time is the only discriminator and is
dominated by track geometry, not skill. For cross-difficulty runs use
completion% as primary metric; normalize speed by progress/reward-per-step.

## Status
Same-difficulty generalization: SOLVED and confirmed under multi-track.
Frontier: cross-difficulty (domain randomization).
