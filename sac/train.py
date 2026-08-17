import torch

from env.environment import RacingEnv
from sac.agent import SACAgent
import random

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


TRACK_SEED = 101

env = RacingEnv(
    max_steps=500,
    verbose=False,
)

eval_env = RacingEnv(
    max_steps=500,
    verbose=False,
)

agent = SACAgent()

episodes = 0
crashes = 0


# -------------------------
# Online SAC training
# -------------------------

observation, info = env.reset(
    options={"track_seed": TRACK_SEED}
)

total_steps = 10000

best_progress = -float('inf')
best_reward = 0.0
best_step = 0
best_completed = False
best_crashed = False

for step in range(total_steps):

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

    # Move to the next state
    observation = next_observation

    # If episode ended, start another episode
    if terminated or truncated:

        episodes += 1

        if info["crashed"]:
            crashes += 1

        print(
            "Episode:",
            episodes,
            "| Lap progress:",
            round(info["lap_progress"], 4),
            "| Crashed:",
            info["crashed"],
        )

        observation, info = env.reset(
            options={"track_seed": TRACK_SEED}
        )

    # Start learning once we have enough experiences
    if len(agent.replay_buffer) >= 64:

        result = agent.update(
            batch_size=64
        )

        if step % 100 == 0:

            print(
                "Step:",
                step,
                "| Reward:",
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

    if step == 0 or (step + 1) % 100 == 0:
        if step == 0:
            print("OVERFIT TEST")
            
        eval_obs, _ = eval_env.reset(options={"track_seed": TRACK_SEED})
        eval_reward = 0.0
        eval_progress = 0.0
        eval_completed = False
        eval_crashed = False
        
        for _ in range(500):
            obs_t = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(obs_t)
                act = torch.tanh(mean).numpy()[0]
                
            eval_obs, r, term, trunc, info_e = eval_env.step(act)
            eval_reward += r
            eval_progress = info_e["lap_progress"]
            
            if info_e["crashed"]:
                eval_crashed = True
            if info_e["lap_completed"]:
                eval_completed = True
                
            if term or trunc:
                break
                
        display_step = 0 if step == 0 else step + 1
        print(f"Step {display_step:<4} | Reward: {eval_reward:<8.4f} | Progress: {eval_progress:<6.4f} | Completed: {eval_completed} | Crashed: {eval_crashed}")

        if eval_progress > best_progress:
            best_progress = eval_progress
            best_reward = eval_reward
            best_step = display_step
            best_completed = eval_completed
            best_crashed = eval_crashed
            torch.save(agent.actor.state_dict(), "best_actor.pth")

print()
print("OVERFIT TEST COMPLETE")
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

agent.actor.load_state_dict(torch.load("best_actor.pth"))

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
    if final_lap_progress > max_lap_progress:
        max_lap_progress = final_lap_progress
        
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
    "actor.pth",
)

print("Saved actor to actor.pth")