# Day 26 — The locked best single-track agent

The single-track agent is now frozen. **v2**, demo-seeded SAC from `runs/exp2_demos_s42/`, laps
the medium track in ~402 steps and the default track in ~346, completes **10/10** benchmark
tracks across both configs, and beats the classical pure-pursuit controller by **~1.65x** on lap
time. It is the reliable agent, and it is the one to build on.

## The locked artifact

| | |
|---|---|
| **Weights** | `models/best_single_track_v2.pth` |
| **Config** | `models/best_single_track_v2_config.json` |
| **Provenance** | `runs/exp2_demos_s42/solved_actor.pth` (demo-seeded SAC, seed 42) |
| **Benchmark data** | `baselines/eval_v1_v2_heuristic.json` |

The locked copy is bit-for-bit the benchmarked checkpoint — both files hash to
`190026208ac4969f...`, so the numbers below describe exactly the weights in `models/`.

From the config: trained with `--curriculum none --seed 42 --demos 40` on the medium track
(`track_seed=101`) alone, seeded with 40 demonstrations (17,014 transitions, 40/40 laps, no
crashes). It first solved the target at **step 5,000** and then passed **9/9 consecutive** solved
evals (`solve_rate 1.0`), at which point the run was stopped early to free CPU. Total 45,000
steps.

## Official benchmark

Fresh run, deterministic evaluation (`tanh(mean)`, no exploration noise), 5 track seeds per
config (101, 202, 303, 404, 505), `max_steps=2000`. Averages are over **completed laps only**.

### Medium track

| Controller | Laps | Avg lap (steps) | Crashes |
|---|---|---|---|
| **v2** | **5/5** | **401.8** | 0 |
| v1 | 4/5 | 402.2 | 1 |
| pure-pursuit | 5/5 | 659.4 | 0 |

v2 vs pure-pursuit: **1.64x faster** (39.1% lower lap time).

### Default track

| Controller | Laps | Avg lap (steps) | Crashes |
|---|---|---|---|
| **v2** | **5/5** | **346.4** | 0 |
| v1 | 2/5 | 346.0 | 3 |
| pure-pursuit | 5/5 | 575.0 | 0 |

v2 vs pure-pursuit: **1.66x faster** (39.8% lower lap time).

## Margin over the heuristic

v2 laps **~1.65x faster** than pure-pursuit, and the margin is consistent across both track
types (1.64x medium, 1.66x default) rather than being an artifact of one circuit. The RL agent
beats the hand-coded baseline on speed.

Worth recording precisely: pure-pursuit is not *worse* at driving — it completed 10/10 tracks
with zero crashes, matching v2 on reliability. It is simply **slower**, because it tracks the
centerline at a conservative speed. The win here is lap time, not competence.

## v2 vs v1 — reliability is the real difference

This is why v2 is locked and v1 is not. Per-seed outcomes:

| Config | Controller | 101 | 202 | 303 | 404 | 505 |
|---|---|---|---|---|---|---|
| medium | v1 | LAP | LAP | **CRASH** | LAP | LAP |
| medium | v2 | LAP | LAP | LAP | LAP | LAP |
| default | v1 | **CRASH** | **CRASH** | **CRASH** | LAP | LAP |
| default | v2 | LAP | LAP | LAP | LAP | LAP |

**v2: 10/10. v1: 6/10.**

On lap time the two are indistinguishable — 401.8 vs 402.2 on medium, 346.4 vs 346.0 on default,
well under a 1% difference either way, and on default v1's average is fractionally *lower*. But
v1's averages are computed over only the tracks it survived: it crashes on medium seed 303 and on
3/5 default tracks. v1's apparent parity is a survivorship artifact of averaging over its
successes.

So v2's gain over v1 is **not speed, it is reliability** — same pace, four more finishes. That
also matches the start-state result in `docs/START_ROBUSTNESS.md` (20/20 varied-start laps): v2
is the agent that keeps the car on the road.

## Reproducing

```bash
PYTHONPATH=. python baselines/benchmark_agents.py
```

The script takes no arguments — seeds, configs and checkpoints are constants at the top of the
file — and it **overwrites `baselines/eval_v1_v2_heuristic.json` in place**. Copy the JSON aside
first if the current numbers matter. Note also that it loads v2 from the run path
(`runs/exp2_demos_s42/solved_actor.pth`), not from `models/best_single_track_v2.pth`; the two are
byte-identical today, so if the locked copy is ever re-pointed, update the script's `V2` constant
to keep the benchmark describing the locked artifact.

Benchmark JSON generated 2026-09-10T19:15:20Z. It stores every per-trial record (steps, lap
completion, progress, off-track rate, reward, crash flag) alongside a `summary` block; the tables
above were recomputed from the per-trial records.
