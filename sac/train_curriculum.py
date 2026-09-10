"""Curriculum SAC training for the reward-v2 exploration problem.

Separate from sac/train.py on purpose: that script is the reproducible baseline
trainer (Day 18) and its --config/--seed contract stays untouched.

Usage:
    python -m sac.train_curriculum --curriculum width5 --seed 42 --name c1_width5
    python -m sac.train_curriculum --curriculum none  --seed 42 --name ctl_none

Writes ONLY to runs/<name>/. The repo-root best_actor.pth / actor.pth are the v1
baseline and are never touched.

SUCCESS CRITERION (the only one that counts):
    the TARGET eval -- deterministic tanh(mean) policy, medium track
    (width 70, base_r 250, n_ctrl 10, min_radius 80), track_seed 101,
    max_steps 500 -- reports lap_completed=True AND crashed=False.
The target eval runs at every eval point regardless of which tier is training,
so the number is always comparable across runs and tiers.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from env.environment import RacingEnv
from sac.agent import SACAgent
from sac.curricula import CURRICULA, TARGETS
from sac.demo_seed import collect_demos

# ─── ACTION CONTRACT (verified against env/environment.py, env/car.py) ────────
#   action = [steering, throttle], Box(-1, 1, (2,))
#   action[0] +1 = full LEFT, -1 = full RIGHT   (x car.max_steering = 30 deg)
#   action[1] +1 = accelerate, -1 = brake       (x car.acceleration = 0.08)
#   Observation: [velocity, heading_err, signed_dist, slip, *5 rays]
# ─────────────────────────────────────────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(description="Curriculum SAC training")
    p.add_argument("--curriculum", choices=list(CURRICULA), default="width5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", required=True, help="run name -> runs/<name>/")
    p.add_argument("--warmup", type=int, default=5000,
                   help="random-action warmup steps (tier 0 only)")
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-scale", type=float, default=1.0,
                   help="multiply every tier's step budget (smoke tests)")
    p.add_argument("--reset-buffer-per-tier", action="store_true",
                   help="clear the replay buffer at each tier switch "
                        "(default: carry it forward)")
    p.add_argument("--demos", type=int, default=0,
                   help="pre-fill the buffer with N pure-pursuit episodes on the "
                        "TARGET config before training (approach 2). 0 = off.")
    p.add_argument("--demo-noise", type=float, default=0.70,
                   help="max per-episode action-noise std for demo episodes")
    p.add_argument("--demo-clean-frac", type=float, default=0.30,
                   help="fraction of demo episodes run noise-free (these lap)")
    p.add_argument("--clip-backward", action="store_true",
                   help="DIAGNOSTIC ONLY. Cancel the reward's negative-progress "
                        "term during TRAINING (eval is left untouched, so the "
                        "success criterion is unchanged). Tests the claim that "
                        "the freeze basin is caused by signed progress making "
                        "'drive backwards' as costly as 'drive forwards' is "
                        "rewarding, while standing still scores exactly 0. "
                        "Not part of any solving recipe -- this modifies the "
                        "reward and must not be used to claim a solve.")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="learning rate for all optimizers (actor, critics, alpha)")
    p.add_argument("--entropy-target", type=float, default=None,
                   help="SAC target entropy; default None -> -action_dim (the standard)")
    p.add_argument("--target", choices=list(TARGETS), default="medium",
                   help="which fixed target config defines success. 'medium' is "
                        "the historical target (seed 101, min_radius 80) and is "
                        "the default so every prior command keeps its meaning. "
                        "'tight' is the grip-experiment target (seed 24, "
                        "min_radius 60) where grip physics actually bite.")
    p.add_argument("--grip", action="store_true", default=False,
                   help="train under grip-limited physics: every RacingEnv "
                        "(target, tier training, tier eval) is built with "
                        "grip_limit=True. Default off = physics unchanged.")
    return p


def evaluate(agent, env, track_seed, max_steps):
    """Deterministic greedy rollout. Returns a dict of outcome metrics."""
    obs, _ = env.reset(options={"track_seed": track_seed})
    total_r = 0.0
    progress = 0.0
    completed = False
    crashed = False
    off_steps = 0
    speeds, steers, corner_r = [], [], []

    for n in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = agent.actor(obs_t)
            act = torch.tanh(mean).numpy()[0]

        obs, r, term, trunc, info = env.step(act)
        total_r += r
        progress = info["lap_progress"]
        speeds.append(float(obs[0]))
        steers.append(float(act[0]))
        corner_r.append(env.track.corner_radius(env.car.x, env.car.y))
        if info["crashed"]:
            crashed = True
            off_steps += 1
        if info["lap_completed"]:
            completed = True
        if term or trunc:
            break

    steps = n + 1
    # Corner-speed management is the thing under test, and MEAN speed alone
    # cannot show it: a policy that slows for corners and a policy that drives
    # uniformly slower can share a mean. So split the lap by the TRUE curvature
    # radius of the road under the car and report each half's mean speed. A
    # policy that manages corners has corner_speed well below straight_speed; a
    # policy that just crawls has them equal and both low.
    # (Ray distances cannot do this job -- at width 70 the +-90 deg rays pin
    # min(rays) at ~36px everywhere, so it carries no corner signal.)
    # The first ~80 steps are the 0 -> 4.0 acceleration ramp, and the start pose
    # sits in the tightest corner on the track (R=66.6px on the tight target),
    # so leaving them in scores the launch as "slow in a corner" and invents a
    # speed drop for a policy that has none. Measured: flat-out reads +0.96
    # with the ramp included and exactly +0.00 without it. Cut the ramp.
    RAMP = 80
    sp = np.asarray(speeds, dtype=float)
    cr = np.asarray(corner_r, dtype=float)
    if sp.size > RAMP + 20:
        sp_s, cr_s = sp[RAMP:], cr[RAMP:]
    else:
        sp_s, cr_s = sp, cr
    if sp_s.size:
        tight = cr_s <= np.percentile(cr_s, 30)   # tightest 30% of the lap
        open_ = cr_s >= np.percentile(cr_s, 70)   # most open 30%
        corner_speed = float(sp_s[tight].mean()) if tight.any() else 0.0
        straight_speed = float(sp_s[open_].mean()) if open_.any() else 0.0
    else:
        corner_speed = straight_speed = 0.0
    return {
        "steps": steps,
        "reward": total_r,
        "progress": progress,
        "completed": completed,
        "crashed": crashed,
        "lap_time": steps if completed else None,
        "off_track_rate": off_steps / max(steps, 1),
        "mean_speed": float(sp.mean()) if sp.size else 0.0,
        "min_speed": float(sp.min()) if sp.size else 0.0,
        "speed_std": float(sp.std()) if sp.size else 0.0,
        "corner_speed": corner_speed,
        "straight_speed": straight_speed,
        # >0 means the policy is slower in the tight third than the open third,
        # i.e. it is actually braking for corners rather than driving one speed.
        "speed_drop": straight_speed - corner_speed,
        "mean_steer": float(np.mean(steers)) if steers else 0.0,
        "grip_events": int(getattr(env.car, "grip_events", 0)),
        "grip_yaw_lost": float(getattr(env.car, "grip_yaw_lost", 0.0)),
    }


def main():
    args = build_parser().parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    tiers = CURRICULA[args.curriculum]
    target_cfg = TARGETS[args.target]
    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # The fixed target env -- never changes, defines success.
    target_env = RacingEnv(max_steps=target_cfg["max_steps"],
                           track_kwargs=target_cfg["track_kwargs"],
                           grip_limit=args.grip)
    target_env.action_space.seed(SEED + 999)

    agent = SACAgent(lr=args.lr, target_entropy=args.entropy_target)

    # ---- Approach 2: demonstration seeding ----------------------------------
    demo_stats = None
    if args.demos > 0:
        demo_stats = collect_demos(
            agent.replay_buffer, args.demos,
            target_cfg["track_kwargs"], target_cfg["track_seed"],
            target_cfg["max_steps"], seed=SEED,
            clean_frac=args.demo_clean_frac, noise_std=args.demo_noise,
            # Demos MUST be generated under the same physics the agent trains
            # under, or the buffer teaches a car that corners better than the
            # one it drives.
            grip_limit=args.grip,
        )
        # The run pushes more transitions than the buffer holds, so without
        # protection the demos are evicted mid-run -- exactly when the agent
        # still depends on them.
        kept = agent.replay_buffer.protect_first(len(agent.replay_buffer))
        print(f"DEMO SEED | protected {kept} demo transitions from eviction")

    print("=" * 96)
    print(f"CURRICULUM RUN  name={args.name}  curriculum={args.curriculum}  seed={SEED}")
    for t in tiers:
        print(f"  {t['name']:10s} width={t['track_kwargs']['width']:3d} "
              f"max_steps={t['max_steps']:4d}  budget={int(t['steps'] * args.steps_scale)}")
    tt = target_cfg["track_kwargs"]
    print(f"TARGET[{args.target}]: width={tt['width']} min_radius={tt['min_radius']} "
          f"max_steps={target_cfg['max_steps']} track_seed={target_cfg['track_seed']} "
          f"grip={args.grip}  -> success = lap_completed AND not crashed")
    print("=" * 96)

    global_step = 0
    episodes = 0
    crashes = 0
    history = []

    best_target_progress = -float("inf")
    best_path = os.path.join(out_dir, "best_actor.pth")
    solved_at = None
    solved_eval = None
    t_start = time.time()

    def write_record(final_best=None):
        """Persist manifest + history. Called at EVERY eval, not just at the
        end, so a run that is killed mid-flight still leaves its evidence on
        disk instead of losing the whole history."""
        manifest = {
            "run_name": args.name,
            "curriculum": args.curriculum,
            "tiers": [dict(t) for t in tiers],
            "seed": SEED,
            "warmup": args.warmup,
            "eval_every": args.eval_every,
            "batch_size": args.batch_size,
            "steps_scale": args.steps_scale,
            "reset_buffer_per_tier": args.reset_buffer_per_tier,
            "demos": args.demos,
            "demo_noise": args.demo_noise,
            "demo_clean_frac": args.demo_clean_frac,
            "clip_backward_DIAGNOSTIC": args.clip_backward,
            "grip": args.grip,
            "target": args.target,
            "demo_stats": demo_stats,
            "total_steps": global_step,
            "episodes": episodes,
            "crashes": crashes,
            "target_config": target_cfg,
            "best_target_progress": best_target_progress,
            "solved_at_step": solved_at,
            "solved_eval": solved_eval,
            "final_best_target_eval": final_best,
            "completed_run": final_best is not None,
            "wall_minutes": round((time.time() - t_start) / 60.0, 2),
            "command": (f"python -m sac.train_curriculum "
                        f"--curriculum {args.curriculum} --seed {SEED} "
                        f"--name {args.name} --demos {args.demos} "
                        f"--target {args.target}"
                        + (" --grip" if args.grip else "")),
        }
        with open(os.path.join(out_dir, "run_config.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    for tier_idx, tier in enumerate(tiers):
        budget = int(tier["steps"] * args.steps_scale)
        tk = tier["track_kwargs"]
        ms = tier["max_steps"]

        env = RacingEnv(max_steps=ms, track_kwargs=tk, grip_limit=args.grip)
        env.action_space.seed(SEED + tier_idx)
        tier_eval_env = RacingEnv(max_steps=ms, track_kwargs=tk,
                                  grip_limit=args.grip)

        if args.reset_buffer_per_tier and tier_idx > 0:
            agent.replay_buffer.buffer = []
            agent.replay_buffer.position = 0

        print(f"\n{'-' * 96}")
        print(f"TIER {tier_idx} {tier['name']}  width={tk['width']}  max_steps={ms}  "
              f"budget={budget}  buffer={len(agent.replay_buffer)}")
        print(f"{'-' * 96}")

        obs, info = env.reset(options={"track_seed": tier["track_seed"]})
        ep_prog = 0.0

        for _ in range(budget):
            # Warmup only at the very start of the whole run; later tiers keep
            # acting on-policy so the carried-forward network is not discarded.
            if global_step < args.warmup and args.demos == 0:
                action = env.action_space.sample()
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                act_t, _ = agent.actor.sample(obs_t)
                action = act_t.detach().numpy()[0]

            next_obs, reward, terminated, truncated, info = env.step(action)
            if args.clip_backward and info["progress"] < 0.0:
                # reward-v2 contributes progress * 20.0, then scales by 100.
                # Subtracting that term back out zeroes the backward penalty
                # without touching any other part of the reward.
                reward -= info["progress"] * 20.0 * 100.0
            agent.replay_buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            ep_prog = info["lap_progress"]

            if terminated or truncated:
                episodes += 1
                if info["crashed"]:
                    crashes += 1
                obs, info = env.reset(options={"track_seed": tier["track_seed"]})
                ep_prog = 0.0

            learn_after = 0 if args.demos > 0 else args.warmup
            if global_step >= learn_after and len(agent.replay_buffer) >= args.batch_size:
                agent.update(batch_size=args.batch_size)

            global_step += 1

            if global_step % args.eval_every == 0:
                tier_r = evaluate(agent, tier_eval_env, tier["track_seed"], ms)
                tgt_r = evaluate(agent, target_env, target_cfg["track_seed"],
                                 target_cfg["max_steps"])

                mins = (time.time() - t_start) / 60.0
                print(
                    f"EVAL | Step: {global_step:6d} | Tier: {tier['name']:10s} "
                    f"|| TIER prog={tier_r['progress']:7.4f} lap={str(tier_r['completed']):5s} "
                    f"crash={str(tier_r['crashed']):5s} "
                    f"|| TARGET prog={tgt_r['progress']:7.4f} "
                    f"lap={str(tgt_r['completed']):5s} crash={str(tgt_r['crashed']):5s} "
                    f"steps={tgt_r['steps']:4d} v={tgt_r['mean_speed']:5.2f} "
                    f"(cnr={tgt_r['corner_speed']:4.2f}/str={tgt_r['straight_speed']:4.2f} "
                    f"drop={tgt_r['speed_drop']:+5.2f}) grip={tgt_r['grip_events']:4d} "
                    f"R={tgt_r['reward']:9.2f} | {mins:5.1f}m"
                )

                history.append({
                    "step": global_step, "tier": tier["name"],
                    "tier_eval": tier_r, "target_eval": tgt_r,
                    "episodes": episodes, "crashes": crashes,
                })

                write_record()

                if tgt_r["progress"] > best_target_progress:
                    best_target_progress = tgt_r["progress"]
                    torch.save(agent.actor.state_dict(), best_path)

                if tgt_r["completed"] and not tgt_r["crashed"] and solved_at is None:
                    solved_at = global_step
                    solved_eval = tgt_r
                    torch.save(agent.actor.state_dict(),
                               os.path.join(out_dir, "solved_actor.pth"))
                    print(f"*** TARGET SOLVED at step {global_step}: "
                          f"lap_completed=True crashed=False "
                          f"lap_time={tgt_r['lap_time']} ***")

    torch.save(agent.actor.state_dict(), os.path.join(out_dir, "actor.pth"))

    # Final verdict eval on the frozen best-on-target checkpoint.
    final_best = None
    if os.path.exists(best_path):
        agent.actor.load_state_dict(torch.load(best_path))
        final_best = evaluate(agent, target_env, target_cfg["track_seed"],
                              target_cfg["max_steps"])
        print(f"\nFINAL TARGET EVAL (best-on-target checkpoint): {final_best}")

    write_record(final_best)

    print(f"\nSaved to {out_dir}/  (best_target_progress={best_target_progress:.4f}, "
          f"solved_at={solved_at})")


if __name__ == "__main__":
    main()
