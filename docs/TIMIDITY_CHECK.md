# Day 23 — Timidity check

**Task premise:** *"if the agent drives timidly, raise the speed cap / anneal off the off-track
penalty."*

This is a conditional task. Day 23 evaluates the condition rather than assuming it, because the
prescribed remedies (raising the cap, annealing the penalty) are only correct if the agent is in
fact leaving speed on the table.

---

## Method

No training, no checkpoint changes. The solved reward-v2 agent was rolled out and measured:

```
python -m analysis.collect_run --checkpoint runs/exp2_demos_s42/solved_actor.pth
  → analysis/runs/medium_seed101_solved_actor.npz
```

Medium track, `track_seed=101`, deterministic `tanh(mean)` policy. The speed profile is the
`velocity` array from that trace, against the car's speed cap of `4.0` (`env/car.py:17`).

## Result

Lap completed in **402 steps**, `lap_completed=True`, `crashed=False`.

| Metric | Value |
|---|---|
| mean speed | **3.572** / 4.0 cap |
| median speed | **4.000** |
| max speed | **4.000** |
| share of lap > 3.5 | **81.1%** |
| share of lap < 2.5 | **13.4%** |

Splitting the lap at the standing start sharpens the picture. The car needs **76 steps** to
accelerate from rest to 3.5. Once moving, it never slows down again:

| Metric | After launch ramp (n=326) |
|---|---|
| mean speed | **3.992** |
| median speed | **4.000** |
| slowest point | **3.536** |
| share > 3.5 | **100.0%** |
| share < 2.5 | **0.0%** |

All 54 sub-2.5 steps are the launch ramp. There are none anywhere else in the lap.

For reference, the tuned pure-pursuit expert laps the same track in **431** steps
(`docs/REWARD_V2_SOLVED.md`); the agent is **29 steps faster**.

## Verdict

**The agent is NOT timid.** It is pinned at the speed cap for effectively the entire lap —
median exactly 4.000, and post-launch it holds a 3.992 mean and never drops below 3.536. It is
not braking cautiously; it is barely braking at all, and it still beats the expert baseline.

The task's *"if timid"* condition is **false**, so no curriculum change and no penalty annealing
is warranted. Raising the speed cap would be the only remedy with any headroom left to give, and
that is a physics change, not a timidity fix.

**Deliverable met by measurement: the agent already attacks the track.**

---

### Note on two figures

An earlier framing of this result cited *85.3% >3.5* and *8.9% <2.5*, with the latter attributed
to corner braking. Both reproduce exactly only if the first 20 steps of the trace are dropped;
over the full 402-step lap the values are 81.1% and 13.4%. The attribution is also off: the
sub-2.5 time is entirely the standing start, not cornering — the agent's slowest post-launch
moment is 3.536. This does not weaken the verdict, it strengthens it.

Separately, `mean speed 3.572` here vs `mean_speed=3.582` in `docs/REWARD_V2_SOLVED.md` is a
sampling convention, not a discrepancy: `analysis/collect_run.py` records velocity *before*
each step, so it includes the `v=0` sample at t=0. Excluding it gives 3.581.
