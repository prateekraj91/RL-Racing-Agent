"""Pre-fill a SAC replay buffer with pure-pursuit demonstrations.

Approach 2 of the reward-v2 exploration fix: the agent never survives a corner,
so it never sees the +1486-scale return of a finished lap and "driving" looks
purely punishing. Seeding the buffer with expert transitions puts successful
cornering AND completed laps in the buffer from step 0, so the critic has
something to bootstrap from before the actor has learned anything.

TWO NON-OBVIOUS REQUIREMENTS, both measured:

1. THE STOCK PURE-PURSUIT GAINS PRODUCE NO FINISHES.
   baselines/pure_pursuit.py cruises at SPEED_CAP=2.6 and needs 656 steps to lap
   medium. The training/eval budget is 500, so stock demos time out at 75%
   progress -- they would seed the buffer with the exact "drove a long way, never
   finished" experience the agent already has. Retuned to SPEED_CAP=4.0,
   CORNER_SPEED=2.6, LOOKAHEAD=70 it laps medium seed 101 in 431 steps, clean.
   Those are the gains used here.

2. NOISELESS DEMOS TEACH THE CRITIC ALMOST NOTHING.
   The track is deterministic, so every clean expert episode retraces the SAME
   431 states. A critic fit only to that ribbon has no idea what happens one
   metre off it, which is exactly where the actor will be. So most episodes are
   run with Gaussian action noise and a jittered start pose, giving off-line
   recovery transitions. A fraction are kept clean so genuine lap completions
   (terminated=True at lap_completed) are present in the buffer.
"""

import math

import numpy as np

import baselines.pure_pursuit as pp
from env.environment import RacingEnv

# Retuned expert gains -- see requirement 1 above.
FAST_GAINS = {
    "SPEED_CAP": 4.0,
    "CORNER_SPEED": 2.6,
    "LOOKAHEAD": 70.0,
    "K_HEADING": 1.15,
    "K_CROSS": 0.10,
    "CORNER_ERR_LO": 8.0,
    "CORNER_ERR_HI": 35.0,
    "K_THROTTLE": 2.0,
}


def _apply_gains(gains):
    for k, v in gains.items():
        setattr(pp, k, v)


def _jitter_start(env, rng, max_lateral, max_heading_deg):
    """Displace the car off the start pose so demos include recovery states.

    Offsets perpendicular to the road, re-seeds previous_progress from the new
    position so the progress delta on step 1 is not a spurious jump.
    """
    if max_lateral <= 0 and max_heading_deg <= 0:
        return
    heading = env.track.track_heading(env.car.x, env.car.y)
    # Perpendicular to the road direction (y-up convention, matches car.angle).
    perp = math.radians(heading + 90.0)
    lateral = rng.uniform(-max_lateral, max_lateral)
    env.car.x += lateral * math.cos(perp)
    env.car.y -= lateral * math.sin(perp)
    env.car.angle += rng.uniform(-max_heading_deg, max_heading_deg)
    env.car.velocity = rng.uniform(0.0, 2.0)
    env.previous_progress = env.track.get_progress(env.car.x, env.car.y)


def collect_demos(buffer, n_episodes, track_kwargs, track_seed, max_steps,
                  seed=0, clean_frac=0.30, noise_std=0.70,
                  jitter_lateral=None, jitter_heading=12.0, verbose=True):
    """Run `n_episodes` pure-pursuit episodes and push every transition into
    `buffer`. Returns a stats dict.

    clean_frac  -- fraction of episodes run noise-free from the true start pose;
                   these are the ones that actually complete laps.
    noise_std   -- MAX Gaussian std added to the expert action. Each noisy
                   episode draws its own std uniformly from [0.05, noise_std],
                   so the buffer spans near-expert driving through to genuine
                   failures. A single fixed low std produces 0 crashes, leaving
                   the critic with no evidence that leaving the track is bad.
    jitter_lateral -- max start offset perpendicular to the road, in px.
                   Defaults to 60% of the track half-width (stays on track).
    """
    _apply_gains(FAST_GAINS)
    rng = np.random.default_rng(seed)

    half = track_kwargs.get("width", 70) / 2.0
    if jitter_lateral is None:
        jitter_lateral = 0.6 * half

    env = RacingEnv(max_steps=max_steps, track_kwargs=track_kwargs)

    n_clean = max(1, int(round(n_episodes * clean_frac)))
    stats = {"episodes": 0, "transitions": 0, "laps": 0, "crashes": 0,
             "clean_laps": 0, "progress": []}

    for ep in range(n_episodes):
        clean = ep < n_clean
        ep_noise = 0.0 if clean else float(rng.uniform(0.05, noise_std))
        obs, _ = env.reset(options={"track_seed": track_seed})
        if not clean:
            _jitter_start(env, rng, jitter_lateral, jitter_heading)
            # Rebuild the observation after the jitter so the stored s matches
            # the state the action was actually chosen from.
            obs = _observe(env)

        for _ in range(max_steps):
            action, _dbg = pp.choose_action(env)
            if not clean:
                action = np.clip(
                    action + rng.normal(0.0, ep_noise, size=2), -1.0, 1.0
                ).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            stats["transitions"] += 1
            if terminated or truncated:
                break

        stats["episodes"] += 1
        stats["progress"].append(env.lap_progress)
        if env.lap_completed:
            stats["laps"] += 1
            if clean:
                stats["clean_laps"] += 1
        if info["crashed"]:
            stats["crashes"] += 1

    stats["mean_progress"] = float(np.mean(stats["progress"]))
    stats["max_progress"] = float(np.max(stats["progress"]))
    del stats["progress"]

    if verbose:
        print(f"DEMO SEED | {stats['episodes']} episodes -> "
              f"{stats['transitions']} transitions | laps={stats['laps']} "
              f"(clean={stats['clean_laps']}) crashes={stats['crashes']} | "
              f"mean_prog={stats['mean_progress']:.4f} "
              f"max_prog={stats['max_progress']:.4f}")
    return stats


def _observe(env):
    """Rebuild the 9-d observation from current env state (post-jitter)."""
    err = env.car.angle - env.track.track_heading(env.car.x, env.car.y)
    err = (err + 180.0) % 360.0 - 180.0
    dist = env.track.signed_distance(env.car.x, env.car.y)
    rays = env.car.cast_rays(env.track)
    return np.array([env.car.velocity, err, dist, 0.0, *rays], dtype=np.float32)


if __name__ == "__main__":
    from sac.curricula import TARGET_CONFIG
    from sac.replay_buffer import ReplayBuffer

    buf = ReplayBuffer()
    collect_demos(buf, 30, TARGET_CONFIG["track_kwargs"],
                  TARGET_CONFIG["track_seed"], TARGET_CONFIG["max_steps"],
                  seed=0)
    print("buffer size:", len(buf))
