# Day 25 — Start-state robustness: is the agent brittle away from its trained start?

**Verdict up front: ROBUST.** The solved reward-v2 agent completed **20/20 laps (100%)** from
randomized start positions, speeds and headings. The task's "if brittle, add start-state
randomization" branch is therefore **not taken** — no randomization was added, because the
condition is false.

**Script:** `analysis/test_start_robustness.py`
**Live version:** `python -m sac.visualize --checkpoint runs/exp2_demos_s42/solved_actor.pth --random-start`
(`--random-start` respawns the car at a random position/heading/speed each episode, so the same
check can be watched rather than only measured.)

---

## Task

Test the agent from varied start positions and speeds, and add start-state randomization to
training if it turns out to be brittle (i.e. overfit to the single start state it trains from).

## Method

`analysis/test_start_robustness.py` loads the solved v2 actor (`runs/exp2_demos_s42/solved_actor.pth`)
and drops it into the medium track (`track_seed=101`) 20 times. Each trial:

- **position** — a uniformly random point on the track centerline (indices spanned 11–365 of ~400)
- **speed** — uniform in `[0, 4]` (observed 0.2 to 3.9)
- **heading** — track heading at that point plus uniform noise in `±15°`

Lap bookkeeping (`previous_progress`, `lap_progress`, `lap_completed`, `step_count`) is reset so
progress is measured from the new start, not from the track's nominal origin. Each trial then runs
the deterministic policy (`tanh(mean)`, no exploration noise) for up to 2000 steps and records
whether a lap completed, crashed, or timed out.

Run with `PYTHONPATH=. python analysis/test_start_robustness.py`.

## Result

```
completed 20/20 varied-start laps (100%)
ROBUST
```

Every trial returned `LAP`; no crashes, no timeouts. The starts were genuinely varied — positions
spread around the whole loop, starting speeds from near-stationary (0.2) to near-top (3.9).

## Verdict

The agent is robust to start state. **No start-state randomization is needed.**

## Why

Two reasons, and neither is luck:

1. **The track is a loop.** During normal driving the agent visits every position on the circuit
   anyway, so "start here facing roughly this way" is a state it has already seen thousands of
   times mid-lap. There is no privileged start state to overfit to — only a privileged *place to
   begin counting*, which is bookkeeping, not physics.
2. **Demo-seeding already injected start jitter.** `sac/demo_seed.py:_jitter_start` perturbs each
   noisy demo episode's start pose — lateral offset up to `0.6 * half_width` perpendicular to the
   road, heading `±12°`, speed uniform in `[0, 2]` — so the training buffer contained varied start
   states from the outset. Note this jitter varies *offset, heading and speed*, not position
   around the loop; the loop argument above is what covers position, and the jitter's speed range
   (`0-2`) is narrower than the range tested here (`0-4`), so the top half of the tested speeds is
   covered by mid-lap experience rather than by seeded starts.

The robustness comes from **state coverage**, not from any explicit robustness mechanism.

## Caveat — a logging bug found and fixed

The first version of the script printed `start_speed` by reading `env.car.velocity` *after* the
2000-step driving loop, so the reported starting speed was really the **final** speed — always
`4.0`, since the agent drives flat out. That made the output look like every trial started at top
speed, i.e. like the speed randomization wasn't taking effect at all.

Fixed by capturing the speed into `start_speed` immediately after setting it, before the loop
runs. The corrected log confirms the starts were genuinely varied. The bug was in the logging
only — the trials themselves always used the random speeds, so the 20/20 result is unaffected.
