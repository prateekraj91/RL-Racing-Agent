"""Test start-state robustness: drop the agent at varied positions/speeds/headings
and count how many laps it completes. High success = robust; low = brittle (overfit
to the trained start state)."""

import numpy as np
import torch
from env.environment import RacingEnv
from sac.agent import SACAgent

MEDIUM = {"width": 70, "base_r": 250, "n_ctrl": 10, "min_radius": 80, "cx": 400, "cy": 300}
CKPT = "runs/exp2_demos_s42/solved_actor.pth"
SEED = 101
N_TRIALS = 20

# load the agent
agent = SACAgent()
agent.actor.load_state_dict(torch.load(CKPT, map_location="cpu"))
agent.actor.eval()

def build_obs(env):
    """Rebuild the observation exactly as env.reset/step does, from current car state."""
    car, track = env.car, env.track
    err = (car.angle - track.track_heading(car.x, car.y) + 180.0) % 360.0 - 180.0
    dist = track.signed_distance(car.x, car.y)
    slip = 0.0
    rays = car.cast_rays(track)
    return np.array([car.velocity, err, dist, slip, *rays], dtype=np.float32)

def act(obs):
    with torch.no_grad():
        mean, _ = agent.actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        return torch.tanh(mean).numpy()[0]

env = RacingEnv(max_steps=2000, track_kwargs=MEDIUM)
rng = np.random.default_rng(0)

results = []
for trial in range(N_TRIALS):
    env.reset(options={"track_seed": SEED})

    # --- place the car at a VARIED start ---
    center = env.track.centerline
    idx = rng.integers(0, len(center))                 # random point on the track
    env.car.x, env.car.y = float(center[idx][0]), float(center[idx][1])
    env.car.angle = env.track.track_heading(env.car.x, env.car.y)  # face along track
    env.car.angle += float(rng.uniform(-15, 15))       # small heading noise
    env.car.velocity = float(rng.uniform(0.0, 4.0))   
    start_speed = env.car.velocity                     # capture BEFORE the loop changes it
    start_idx = idx                                     # random starting speed
    # reset lap bookkeeping so progress is measured from HERE
    env.previous_progress = env.track.get_progress(env.car.x, env.car.y)
    env.lap_progress = 0.0
    env.lap_completed = False
    env.step_count = 0

    obs = build_obs(env)
    completed, crashed = False, False
    for _ in range(2000):
        obs, r, term, trunc, info = env.step(act(obs))
        if info["lap_completed"]: completed = True
        if info["crashed"]: crashed = True
        if term or trunc: break

    results.append(completed)
    print(f"trial {trial:2d}: pos_idx={start_idx:3d} speed0={start_speed:.1f}  ->  "
          f"{'LAP' if completed else ('CRASH' if crashed else 'TIMEOUT')}")

n_ok = sum(results)
print(f"\n--- START-STATE ROBUSTNESS ---")
print(f"completed {n_ok}/{N_TRIALS} varied-start laps ({100*n_ok/N_TRIALS:.0f}%)")
print("ROBUST" if n_ok >= 0.8*N_TRIALS else "BRITTLE (overfit to trained start)")