"""SAC training loop for the racing environment.

Usage:
    python -m sac.train                      # offline, writes ./best_actor.pth + ./actor.pth
    python -m sac.train --wandb              # same, plus Weights & Biases logging
    python -m sac.train --steps 2000 --out-dir /tmp/smoke   # short smoke run

--wandb is OFF by default, so offline runs are byte-for-byte unaffected.
--out-dir defaults to "." (existing behaviour); point it elsewhere to avoid
overwriting the committed checkpoints during smoke tests.
"""

# ─── ACTION CONTRACT (verified against env/environment.py:123-128, env/car.py:39-54) ──
#
#   action = np.array([steering, throttle], dtype=np.float32)    Box(-1.0, 1.0, (2,))
#
#   action[0]  STEERING  in [-1, 1]  ->  car.steering = action[0] * car.max_steering (30°)
#                +1.0 = full LEFT   (heading angle increases, CCW on screen)
#                -1.0 = full RIGHT  (heading angle decreases, CW on screen)
#
#   action[1]  THROTTLE  in [-1, 1]  ->  car.velocity += action[1] * car.acceleration (0.08)
#                +1.0 = full accelerate       -1.0 = full brake / reverse
#                velocity clamped to [-2.0, 4.0]; friction 0.03/step decays toward 0
#
#   Observation layout (environment.py:248-257):
#     [0] velocity   [1] heading_err   [2] signed_dist   [3] slip   [4:9] 5 ray beams
# ──────────────────────────────────────────────────────────────────────────────────────

import argparse
import os

import torch

from env.environment import RacingEnv
from sac.agent import SACAgent
import random

import numpy as np

# -------------------------
# Named training configs — a run is fully specified by config name + seed.
# Add new configs here; never hardcode track params in the loop again.
# -------------------------
CONFIGS = {
    "medium": {
        "track_kwargs": {"width": 70, "base_r": 250, "n_ctrl": 10,
                         "min_radius": 80, "cx": 400, "cy": 300},
        "track_seed": 101,
        "max_steps": 500,
    },
    "default": {
        "track_kwargs": {},          # env built-in defaults (base_r=210, min_radius=70)
        "track_seed": 101,
        "max_steps": 500,
    },
}


parser = argparse.ArgumentParser(description="SAC training for RL Racing Agent")
parser.add_argument("--wandb", action="store_true",
                    help="enable Weights & Biases logging (default: off)")
parser.add_argument("--wandb-project", default="rl-racing-agent",
                    help="wandb project name")
parser.add_argument("--wandb-run-name", default=None,
                    help="optional wandb run name")
parser.add_argument("--steps", type=int, default=10000,
                    help="total environment steps")
parser.add_argument("--warmup", type=int, default=1000,
                    help="random-action warmup steps")
parser.add_argument("--eval-every", type=int, default=1000,
                    help="run a deterministic eval every N steps")
parser.add_argument("--out-dir", default=".",
                    help="directory for best_actor.pth / actor.pth")
parser.add_argument("--config", choices=list(CONFIGS), default="medium",
                    help="named training config (track params + seed + max_steps)")
parser.add_argument("--seed", type=int, default=42,
                    help="master seed — seeds torch, numpy, random, AND the env action spaces")
args = parser.parse_args()


SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


cfg = CONFIGS[args.config]
TRACK_SEED = cfg["track_seed"]
track_kwargs = cfg["track_kwargs"]
max_steps = cfg["max_steps"]

print(f"config: {args.config}  |  track_seed={TRACK_SEED}  |  max_steps={max_steps}")
print(f"track_kwargs: {track_kwargs}")

env = RacingEnv(
    max_steps=max_steps,
    verbose=False,
    track_kwargs=track_kwargs,
)

eval_env = RacingEnv(
    max_steps=max_steps,
    verbose=False,
    track_kwargs=track_kwargs,
)

# Seed the action-space RNGs — this is the one np.random.seed does NOT cover.
# Without this, env.action_space.sample() during warmup is non-reproducible.
env.action_space.seed(SEED)
eval_env.action_space.seed(SEED + 1)   # different stream so eval != train warmup

agent = SACAgent()

episodes = 0
crashes = 0


# -------------------------
# Online SAC training
# -------------------------

observation, info = env.reset(
    seed=SEED,
    options={"track_seed": TRACK_SEED}
)

total_steps = args.steps
WARMUP_STEPS = args.warmup
EVAL_EVERY = args.eval_every

os.makedirs(args.out_dir, exist_ok=True)
best_path = os.path.join(args.out_dir, "best_actor.pth")
final_path = os.path.join(args.out_dir, "actor.pth")

# -------------------------
# Optional wandb logging
# -------------------------

run = None
if args.wandb:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "algo": "SAC",
            "total_steps": total_steps,
            "warmup_steps": WARMUP_STEPS,
            "eval_every": EVAL_EVERY,
            "track_seed": TRACK_SEED,
            "track_config": track_kwargs,
            "max_episode_steps": env.max_steps,
            "batch_size": 64,
            "gamma": 0.99,
            "obs_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
            "action_space": "Box(-1,1,(2,)) [steering, throttle]",
            "seed": 42,
        },
    )
    print(f"wandb: logging to project '{args.wandb_project}' (mode={os.environ.get('WANDB_MODE', 'online')})")


def wlog(payload, step):
    """Log to wandb if enabled; no-op otherwise."""
    if run is not None:
        run.log(payload, step=step)


best_progress = -float('inf')
best_reward = 0.0
best_step = 0
best_completed = False
best_crashed = False

# Per-episode accumulators (logging only — these do not touch the training math).
ep_reward = 0.0
ep_len = 0
ep_off_track = 0
ep_speeds = []
ep_steerings = []
ep_heading_errs = []

for step in range(total_steps):

    if step == 0:
        print("WARMUP START")
        
    if step == WARMUP_STEPS:
        print("WARMUP COMPLETE | Buffer:", len(agent.replay_buffer))

    if step < WARMUP_STEPS:
        action = env.action_space.sample()
    else:
        # Convert observation to a PyTorch tensor
        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
        ).unsqueeze(0)
    
        # Ask the actor for an action
        action_tensor, _ = agent.actor.sample(
            observation_tensor
        )
    
        action = action_tensor.detach().numpy()[0]

    if step < 10:
        print("ACTION:", action)

    # Let the car/environment react
    next_observation, reward, terminated, truncated, info = env.step(
        action
    )

    # Store this experience
    agent.replay_buffer.add(
        observation,
        action,
        reward,
        next_observation,
        terminated,
    )

    # --- logging accumulators (read-only w.r.t. training) ---
    ep_reward += reward
    ep_len += 1
    if info["crashed"]:
        ep_off_track += 1
    ep_speeds.append(float(next_observation[0]))
    ep_heading_errs.append(abs(float(next_observation[1])))
    ep_steerings.append(abs(float(action[0])))

    # Move to the next state
    observation = next_observation

    # If episode ended, start another episode
    if terminated or truncated:

        episodes += 1

        if info["crashed"]:
            crashes += 1

        phase = "Warmup" if step < WARMUP_STEPS else "Training"
        print(
            f"[{phase}] Episode:",
            episodes,
            "| Lap progress:",
            round(info["lap_progress"], 4),
            "| Crashed:",
            info["crashed"],
        )

        wlog(
            {
                "episode/reward": ep_reward,
                "episode/length": ep_len,
                "episode/lap_progress": info["lap_progress"],
                "episode/lap_completed": int(info["lap_completed"]),
                "episode/lap_time": ep_len if info["lap_completed"] else None,
                "episode/off_track_rate": ep_off_track / max(ep_len, 1),
                "episode/crashed": int(info["crashed"]),
                "episode/index": episodes,
                "episode/mean_speed": float(np.mean(ep_speeds)),
                "episode/max_speed": float(np.max(ep_speeds)),
                "episode/mean_abs_steering": float(np.mean(ep_steerings)),
                "episode/mean_abs_heading_err": float(np.mean(ep_heading_errs)),
                "episode/max_abs_heading_err": float(np.max(ep_heading_errs)),
                "progress/total_episodes": episodes,
                "progress/total_crashes": crashes,
            },
            step=step,
        )

        ep_reward = 0.0
        ep_len = 0
        ep_off_track = 0
        ep_speeds = []
        ep_steerings = []
        ep_heading_errs = []

        observation, info = env.reset(
            options={"track_seed": TRACK_SEED}
        )

    # Start learning once we have enough experiences and finished warmup
    if step >= WARMUP_STEPS and len(agent.replay_buffer) >= 64:

        result = agent.update(
            batch_size=64
        )

        if step % 100 == 0:

            print(
                "Step:",
                step,
                "[Training] | Reward:",
                round(reward, 4),
                "| Buffer:",
                len(agent.replay_buffer),
                "| Actor loss:",
                round(result["actor_loss"], 3),
                "| Critic 1:",
                round(result["critic1_loss"], 3),
                "| Critic 2:",
                round(result["critic2_loss"], 3),
            )

            wlog(
                {
                    "train/actor_loss": result["actor_loss"],
                    "train/critic1_loss": result["critic1_loss"],
                    "train/critic2_loss": result["critic2_loss"],
                    "train/alpha_loss": result["alpha_loss"],
                    "train/alpha": result["alpha"],
                    "train/buffer_size": len(agent.replay_buffer),
                    "train/step_reward": reward,
                },
                step=step,
            )
    elif step < WARMUP_STEPS and step % 100 == 0:
        print(
            "Step:",
            step,
            "[Warmup] | Reward:",
            round(reward, 4),
            "| Buffer:",
            len(agent.replay_buffer),
        )

    if (step + 1) % EVAL_EVERY == 0:
            
        eval_obs, _ = eval_env.reset(options={"track_seed": TRACK_SEED})
        eval_reward = 0.0
        eval_progress = 0.0
        eval_completed = False
        eval_crashed = False
        eval_steps = 0
        off_track_steps = 0
        eval_speeds = []
        eval_steerings = []
        eval_heading_errs = []
        
        for eval_step in range(500):
            obs_t = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(obs_t)
                act = torch.tanh(mean).numpy()[0]
                
            eval_obs, r, term, trunc, info_e = eval_env.step(act)
            eval_steps += 1

            if info_e["crashed"]:
                off_track_steps += 1

            eval_reward += r
            eval_progress = info_e["lap_progress"]
            eval_speeds.append(float(eval_obs[0]))
            eval_heading_errs.append(abs(float(eval_obs[1])))
            eval_steerings.append(abs(float(act[0])))
            
            if info_e["crashed"]:
                eval_crashed = True
            if info_e["lap_completed"]:
                eval_completed = True
                
            if term or trunc:
                break

        lap_time = eval_steps if eval_completed else None
        off_track_rate = off_track_steps / eval_steps if eval_steps > 0 else 0.0

        display_step = step + 1

        print(
            f"EVAL | Step: {display_step} "
            f"| Reward: {eval_reward:.4f} "
            f"| Progress: {eval_progress:.4f} "
            f"| Completed: {eval_completed} "
            f"| Crashed: {eval_crashed} "
            f"| Lap time: {lap_time}"
            f"| Off-track rate: {off_track_rate:.2%}"
            )

        wlog(
            {
                "eval/reward": eval_reward,
                "eval/lap_progress": eval_progress,
                "eval/lap_completed": int(eval_completed),
                "eval/lap_time": lap_time,
                "eval/off_track_rate": off_track_rate,
                "eval/crashed": int(eval_crashed),
                "eval/episode_length": eval_steps,
                "eval/mean_speed": float(np.mean(eval_speeds)),
                "eval/max_speed": float(np.max(eval_speeds)),
                "eval/mean_abs_steering": float(np.mean(eval_steerings)),
                "eval/mean_abs_heading_err": float(np.mean(eval_heading_errs)),
            },
            step=step,
        )

        if eval_progress > best_progress:
            best_progress = eval_progress
            best_reward = eval_reward
            best_step = display_step
            best_completed = eval_completed
            best_crashed = eval_crashed
            torch.save(agent.actor.state_dict(), best_path)

print()
print("EVALUATION COMPLETE")
print("Training finished.")
print("Replay buffer:", len(agent.replay_buffer))

# -------------------------
# Temporary Diagnostic Evaluation
# -------------------------

print("\nBEST CHECKPOINT")
print(f"Best progress: {best_progress}")
print(f"Best reward: {best_reward}")
print(f"Best checkpoint step: {best_step}")
print(f"Completed: {best_completed}")
print(f"Crashed: {best_crashed}")

if os.path.exists(best_path):
    agent.actor.load_state_dict(torch.load(best_path))
else:
     print(f"[warn] no best checkpoint at {best_path} "
          f"(no eval beat -inf — run longer than --eval-every={EVAL_EVERY}); "
          f"skipping final diagnostic eval.")

observation, info = env.reset(options={"track_seed": TRACK_SEED})

total_reward = 0.0
max_lap_progress = 0.0
final_lap_progress = 0.0
crashed = False
lap_completed = False
final_velocity = 0.0

for _ in range(500):
    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        mean, _ = agent.actor(observation_tensor)
        action = torch.tanh(mean).numpy()[0]
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    final_lap_progress = info["lap_progress"]
    max_lap_progress = max(max_lap_progress, final_lap_progress)
        
    final_velocity = env.car.velocity
    
    if info["crashed"]:
        crashed = True
    if info["lap_completed"]:
        lap_completed = True
        
    if terminated or truncated:
        break

print("\nEVALUATION")
print(f"Total reward: {total_reward}")
print(f"Final lap progress: {final_lap_progress}")
print(f"Max lap progress: {max_lap_progress}")
print(f"Final velocity: {final_velocity}")
print(f"Crashed: {crashed}")
print(f"Lap completed: {lap_completed}")

torch.save(
    agent.actor.state_dict(),
    final_path,
)

print(f"Saved actor to {final_path}")

# -------------------------
# Save run manifest next to the checkpoint (reproducibility record)
# -------------------------
import json

manifest = {
    "config_name": args.config,
    "seed": SEED,
    "track_seed": TRACK_SEED,
    "track_kwargs": track_kwargs,
    "max_steps": max_steps,
    "total_steps": args.steps,
    "warmup": args.warmup,
    "eval_every": args.eval_every,
    "best_progress": best_progress,
    "best_step": best_step,
    "command": f"python -m sac.train --config {args.config} --seed {SEED} "
               f"--steps {args.steps} --warmup {args.warmup} --eval-every {args.eval_every}",
}

manifest_path = os.path.join(args.out_dir, "run_config.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved run manifest to {manifest_path}")

if run is not None:
    run.summary["final/lap_completed"] = int(lap_completed)
    run.summary["final/max_lap_progress"] = max_lap_progress
    run.summary["final/total_reward"] = total_reward
    run.summary["final/best_progress"] = best_progress
    run.summary["final/best_step"] = best_step
    run.finish()
