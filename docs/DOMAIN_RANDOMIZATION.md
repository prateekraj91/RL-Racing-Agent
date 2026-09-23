# Domain Randomization — Closing the Width Generalization Gap (Jul 30/31)

## Setup
Retrained with domain randomization: width ~ U(20,70), min_radius ~ U(50,90),
friction ~ U(0.015,0.05), sampled fresh every episode reset. Same recipe as
v2 otherwise (demo-seeded, 40 demo laps, seed 42, sac.train_curriculum).
Command: python -m sac.train_curriculum --curriculum none --seed 42
--name dr_demos_s42 --demos 40 --domain_randomize

A first DR attempt (dr_medium_s42, via sac.train, no demo-seeding) collapsed
to a near-stationary policy (reward ~50 vs ~1400) - demo-seeding was
required for DR to converge at all, consistent with the project's
established finding that plain reward-tuning freezes the agent.

With demo-seeding, DR converged immediately (solved at step 5000, same as
v2) and stayed solved through 160,000 steps with stable reward (~1340-1400),
no collapse.

## Result: real, rigorous, 10 seeds per width
  width | V2 (no DR) | DR
  ------|-----------|----
  40    | 10/10     | 10/10
  36    | 10/10     | 10/10
  32    | 10/10     | 10/10
  28    |  9/10     | 10/10
  25    |  8/10     | 10/10
  22    |  5/10     | 10/10
  20    |  4/10     | 10/10

V2 numbers match the earlier corrected sweep in docs/CROSS_DIFFICULTY_GAP.md
exactly (cross-check passed). DR is 10/10 at every width tested, including
width=20 where V2 fails on 60% of seeds. The width generalization gap
found in docs/CROSS_DIFFICULTY_GAP.md is fully closed by domain
randomization on this axis.

## Caveat
This tests width only (the axis with the proven gap). min_radius and
friction were also randomized during training but not independently
re-tested here - unclear how much each contributed vs width alone.
A single-seed-per-width sweep (width_sweep.py) was run first and looked
similarly clean, but was discarded as insufficient evidence (one seed
per width can't distinguish "robust" from "got lucky"); the 10-seed
sweep above is the trustworthy version.

## Status
Domain randomization hypothesis confirmed on the width axis: training
across a randomized distribution, not a fixed environment, produces a
policy that generalizes well past where the fixed-environment agent
degrades and fails. Next: verify min_radius/friction contribution
independently, and test whether the DR agent still fails at some width
below 20 (untested floor) or has genuinely learned a general narrow-
corridor driving strategy.