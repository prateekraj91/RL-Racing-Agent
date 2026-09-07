# Reward v2 Evaluation — v1 vs v2 vs pure-pursuit

**Task:** Day 21 (Master Plan v3) — evaluate the reward-v2 policy against the reward-v1
policy and the hand-written baseline, and decide whether v2's weights need tuning.

**Source of all numbers:** `baselines/eval_v1_v2_heuristic.json` (generated
2026-09-07 by `baselines/benchmark_agents.py`).

**Protocol:** 5 unseen track seeds (101, 202, 303, 404, 505) × 2 track configs
(`medium`, `default`) × 3 controllers, deterministic rollouts, `max_steps = 2000`.

| controller | checkpoint |
|---|---|
| v1 | `best_actor.pth` |
| v2 | `runs/exp2_demos_s42/solved_actor.pth` |
| pure_pursuit | `baselines/pure_pursuit.py` |

---

## Results

### medium (width 70, base_r 250, min_radius 80)

| controller | laps | avg lap time | avg progress | off-track rate | avg reward |
|---|---|---|---|---|---|
| v1 | 4/5 | 402.2 | 0.9031 | 0.0009 | 1335.08 |
| **v2** | **5/5** | **401.8** | **1.0018** | 0.0000 | 1402.55 |
| pure_pursuit | 5/5 | 659.4 | 1.0008 | 0.0000 | 1475.95 |

### default (narrower, tighter — unseen during v2 training)

| controller | laps | avg lap time | avg progress | off-track rate | avg reward |
|---|---|---|---|---|---|
| v1 | 2/5 | 346.0 | 0.5659 | 0.0054 | 865.36 |
| **v2** | **5/5** | 346.4 | **1.0016** | 0.0000 | 1502.66 |
| pure_pursuit | 5/5 | 575.0 | 1.0008 | 0.0000 | 1555.51 |

`avg lap time` is averaged over **completed laps only**, so v1's 346.0 on `default` is
the mean of its two successes and is not comparable at face value to v2's mean of five.

---

## VERDICT: v2 beats v1 decisively

- **v2 completes 10/10 tracks** (medium 5/5, default 5/5) against **v1's 6/10**
  (medium 4/5, default 2/5).
- v2 crashes zero times; v1 crashes 4 times (1 on medium, 3 on default).
- v2's off-track rate is exactly 0.0 on every track; v1 leaves the track on both configs.
- The gap is entirely on the tracks v1 was diagnosed as failing: v2 generalises to the
  unseen, tighter `default` geometry where v1's fixed-turn-radius behaviour washes wide
  and crashes (see `docs/DIAGNOSIS.md`). Reward v2 fixes the diagnosed failure.

## TIMID? No.

- v2 laps **~1.6× faster than pure_pursuit** on both configs: medium 401.8 vs 659.4,
  default 346.4 vs 575.0.
- Where both v1 and v2 finish, v2 is not slower: medium 401.8 vs 402.2 (v2 marginally
  faster); on the two `default` seeds v1 completes (404, 505) the two are a wash
  (v1 346.0 vs v2 348.5).
- So v2 is the **fastest controller that also completes every track**. The robustness
  gain over v1 cost no lap time, and v2 remains far ahead of the heuristic baseline.

**No weight tuning warranted.** There is no speed/robustness trade-off left to buy back.

---

## Note: total reward is not a proxy for lap quality

pure_pursuit posts the **highest avg reward on both configs** (1475.95 / 1555.51) while
being ~250 steps slower per lap than v2. That is not a better drive — it is an artefact
of the reward shape: progress is paid **per step**, so a slower lap accumulates the same
progress over more steps and banks more total reward.

Consequence: do not rank controllers by `total_reward` / `avg_reward`. **Lap time (with
lap completion as the gate) is the honest metric**, and it is what the verdict above uses.
