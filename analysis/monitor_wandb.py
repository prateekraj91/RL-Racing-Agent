"""Query and print live W&B run telemetry for RL Racing Agent."""

import wandb
import numpy as np


def main():
    api = wandb.Api()
    run = api.run('prateekraj91-bits-pil/rl-racing-agent/nzkji5fw')
    print(f"=== W&B Run: {run.name} ({run.state}) ===")
    print(f"URL: {run.url}")
    print(f"Runtime: {run.summary.get('_runtime', 0):.1f}s | Step: {run.summary.get('_step', 0)}")

    # Summary highlights
    print("\n--- Current Metrics Summary ---")
    print(f"Eval Lap Completed: {run.summary.get('eval/target_lap_completed', 0)} "
          f"(Lap Time: {run.summary.get('eval/target_lap_time', 'N/A')}, Progress: {run.summary.get('eval/target_progress', 0):.4f})")
    print(f"Eval Target Reward: {run.summary.get('eval/target_reward', 0):.2f}")
    print(f"Episode Reward: {run.summary.get('episode/reward', 0):.2f} (Speed: {run.summary.get('episode/mean_speed', 0):.2f})")
    print(f"Entropy Alpha: {run.summary.get('train/alpha', 0):.4f} (Actor Loss: {run.summary.get('train/actor_loss', 0):.2f})")
    print(f"Episodes: {run.summary.get('progress/total_episodes', 0)} | Crashes: {run.summary.get('progress/total_crashes', 0)}")

    # Sampled DR params
    print("\n--- Recent Sampled DR Parameters ---")
    for k in sorted(run.summary.keys()):
        if k.startswith('dr/'):
            val = run.summary[k]
            if isinstance(val, float):
                print(f"  {k}: {val:.4f}")
            else:
                print(f"  {k}: {val}")


if __name__ == '__main__':
    main()
