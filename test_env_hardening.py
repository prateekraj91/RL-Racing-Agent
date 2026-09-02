"""
test_env_hardening.py — Stress-tests for RacingEnv edge cases.

Runs headless (no pygame). Exercises:
  - observation dtype & shape (from both reset and step)
  - deterministic seeding (same seed → identical trajectories)
  - different seeds produce different tracks
  - off-track driving → terminated=True
  - max-step truncation → truncated=True
  - reversing (sustained braking) doesn't crash
  - spinning in place (zero speed + max steering) doesn't hang
  - extreme steering oscillation doesn't crash

Usage:
    pytest test_env_hardening.py
    python test_env_hardening.py        # standalone runner, same 10 tests
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
#   Error signals (env/track.py):
#     heading_err = wrap180(car.angle - track.track_heading(x, y))
#                   > 0  ->  car points LEFT of road    ->  correct with NEGATIVE steering
#     signed_dist = track.signed_distance(x, y)
#                   > 0  ->  car is RIGHT of centerline ->  correct with POSITIVE steering
#
#   Car min turning radius at full steer = wheelbase / tan(30°) = 86.6 px
# ──────────────────────────────────────────────────────────────────────────────────────

import numpy as np
import sys

from env.environment import RacingEnv

# ─────────────────────────────────────────────
# Action helpers — named stand-ins for the old
# discrete constants, so each test's intent
# stays readable.
# ─────────────────────────────────────────────


def act(steering=0.0, throttle=0.0):
    """Build a valid continuous action."""
    return np.array([steering, throttle], dtype=np.float32)


COAST = act(0.0, 0.0)             # was discrete action 0
ACCELERATE = act(0.0, 1.0)        # was discrete action 1
BRAKE = act(0.0, -1.0)            # was discrete action 2
STEER_LEFT = act(1.0, 0.0)        # was discrete action 3
STEER_RIGHT = act(-1.0, 0.0)      # was discrete action 4

# Compound actions (the continuous API can do in one step what the discrete
# API needed two for — the old tests poked env.car.steering directly to work
# around that).
ACCEL_HARD_LEFT = act(1.0, 1.0)
ACCEL_HARD_RIGHT = act(-1.0, 1.0)


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────


def test_obs_type_reset():
    """reset() must return np.ndarray with dtype float32 and correct shape."""
    env = RacingEnv()
    obs, info = env.reset(seed=0)

    assert isinstance(obs, np.ndarray), (
        f"reset() obs is {type(obs).__name__}, expected np.ndarray"
    )
    assert obs.dtype == np.float32, (
        f"reset() obs dtype is {obs.dtype}, expected float32"
    )
    assert obs.shape == (9,), (
        f"reset() obs shape is {obs.shape}, expected (9,)"
    )
    assert env.observation_space.contains(obs), (
        "reset() obs not contained in observation_space"
    )


def test_obs_type_step():
    """step() must return np.ndarray with dtype float32 and correct shape."""
    env = RacingEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(COAST)

    assert isinstance(obs, np.ndarray), (
        f"step() obs is {type(obs).__name__}, expected np.ndarray"
    )
    assert obs.dtype == np.float32, (
        f"step() obs dtype is {obs.dtype}, expected float32"
    )
    assert obs.shape == (9,), (
        f"step() obs shape is {obs.shape}, expected (9,)"
    )
    assert env.observation_space.contains(obs), (
        "step() obs not contained in observation_space"
    )


def test_deterministic_seed():
    """reset(seed=N) must produce identical track, pose AND trajectory both times.

    Replays a fixed action sequence after each reset, so this covers
    determinism of the whole rollout, not just the initial state.
    """

    def rollout(seed, n=200):
        env = RacingEnv(max_steps=1000)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(1234)  # same action stream every call
        states = [obs.copy()]
        rewards = []
        for _ in range(n):
            action = rng.uniform(-1.0, 1.0, 2).astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            states.append(obs.copy())
            rewards.append(reward)
            if terminated or truncated:
                break
        pose = (env.car.x, env.car.y, env.car.angle)
        return np.array(states), np.array(rewards), pose, env.track.centerline.copy()

    states1, rewards1, pose1, center1 = rollout(42)
    states2, rewards2, pose2, center2 = rollout(42)

    assert np.array_equal(center1, center2), (
        "Same seed produced different track centerlines"
    )
    assert pose1 == pose2, (
        f"Same seed produced different final car pose: {pose1} vs {pose2}"
    )
    assert states1.shape == states2.shape, (
        f"Same seed produced different trajectory lengths: "
        f"{states1.shape} vs {states2.shape}"
    )
    assert np.array_equal(states1, states2), (
        "Same seed produced a different observation trajectory"
    )
    assert np.array_equal(rewards1, rewards2), (
        "Same seed produced a different reward trajectory"
    )


def test_different_seeds_differ():
    """Different seeds must produce different tracks."""
    env = RacingEnv()

    env.reset(seed=1)
    center1 = env.track.centerline.copy()

    env.reset(seed=2)
    center2 = env.track.centerline.copy()

    assert not np.array_equal(center1, center2), (
        "Seeds 1 and 2 produced identical tracks — seeding may be broken"
    )


def test_off_track_terminates():
    """Driving off-track must set terminated=True.

    Strategy: accelerate + hard steer left. The car's minimum turning radius
    (86.6 px) is tighter than the track, so it spirals into the inside edge
    and leaves the track. We cap at 5000 steps as a safety net.
    """
    env = RacingEnv(max_steps=10000)  # high limit so truncation doesn't interfere
    env.reset(seed=99)

    terminated = False
    for i in range(5000):
        obs, reward, terminated, truncated, info = env.step(ACCEL_HARD_LEFT)

        if terminated:
            break

    assert terminated, (
        "Car never went off-track in 5000 steps of full throttle + max steering"
    )
    assert info["crashed"], (
        "Episode terminated but info['crashed'] is False"
    )


def test_max_step_truncation():
    """Coasting for max_steps must set truncated=True (not terminated).

    Uses the env's default max_steps=2000. The car starts on-track with zero
    velocity, so coasting neither moves it nor crashes it.
    """
    env = RacingEnv()  # default max_steps=2000
    max_steps = env.max_steps
    assert max_steps == 2000, f"expected default max_steps=2000, got {max_steps}"

    env.reset(seed=0)

    terminated = False
    truncated = False
    for i in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(COAST)
        if terminated or truncated:
            break

    assert truncated, (
        f"Episode did not truncate after {max_steps} steps of coasting"
    )
    assert not terminated, (
        "Episode terminated (crashed) when it should have only truncated"
    )
    assert env.step_count == max_steps, (
        f"step_count is {env.step_count}, expected {max_steps}"
    )


def test_reverse_no_crash():
    """Sustained braking (reversing) must not crash or produce NaN.

    Strategy: hold full brake for 500 steps. The car should reverse (velocity
    floors at -max_speed/2 = -2.0). The env must not raise, and observations
    must stay finite.
    """
    env = RacingEnv(max_steps=1000)
    env.reset(seed=7)

    for i in range(500):
        obs, reward, terminated, truncated, info = env.step(BRAKE)

        assert np.all(np.isfinite(obs)), (
            f"Non-finite obs at step {i} during reversing: {obs}"
        )
        assert np.isfinite(reward), (
            f"Non-finite reward at step {i} during reversing: {reward}"
        )

        if terminated or truncated:
            # It's fine if the car reverses off-track and terminates,
            # or if we hit max_steps. The point is it didn't crash.
            break


def test_spin_in_place():
    """Spinning at near-zero speed must not crash, hang, or produce NaN.

    Strategy: max steering with zero throttle. The car has zero initial
    velocity, and heading rate is proportional to speed, so it barely moves.
    """
    env = RacingEnv(max_steps=1000)
    env.reset(seed=3)

    for i in range(500):
        obs, reward, terminated, truncated, info = env.step(STEER_LEFT)

        assert np.all(np.isfinite(obs)), (
            f"Non-finite obs at step {i} during spin: {obs}"
        )
        assert np.isfinite(reward), (
            f"Non-finite reward at step {i} during spin: {reward}"
        )

        if terminated or truncated:
            break


def test_extreme_steering_oscillation():
    """Alternating max-left/max-right every step must not crash.

    Strategy: full throttle with steering slammed between +1 and -1 every
    step. This creates wild oscillation. Run 500 steps.
    """
    env = RacingEnv(max_steps=1000)
    env.reset(seed=5)

    for i in range(500):
        action = ACCEL_HARD_LEFT if i % 2 == 0 else ACCEL_HARD_RIGHT
        obs, reward, terminated, truncated, info = env.step(action)

        assert np.all(np.isfinite(obs)), (
            f"Non-finite obs at step {i} during steering oscillation: {obs}"
        )
        assert np.isfinite(reward), (
            f"Non-finite reward at step {i} during steering oscillation: {reward}"
        )

        if terminated or truncated:
            break


def test_terminated_and_truncated_mutually_exclusive():
    """terminated and truncated must never both be True on the same step.

    Strategy: run a full episode to max_steps with occasional acceleration
    to try to trigger both conditions near the boundary.
    """
    max_steps = 100
    env = RacingEnv(max_steps=max_steps)
    env.reset(seed=10)

    for i in range(max_steps + 10):  # go slightly past to be safe
        action = ACCELERATE if i % 5 == 0 else COAST
        obs, reward, terminated, truncated, info = env.step(action)

        assert not (terminated and truncated), (
            f"Both terminated AND truncated are True at step {i}"
        )

        if terminated or truncated:
            break


# ─────────────────────────────────────────────
# Standalone runner (pytest is the primary path)
# ─────────────────────────────────────────────

passed = 0
failed = 0


def run_test(name, fn):
    """Run a test function, print PASS/FAIL, track counts."""
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ {name} — {e}")
        failed += 1
    except Exception as e:
        print(f"  ✗ {name} — UNEXPECTED ERROR: {type(e).__name__}: {e}")
        failed += 1


if __name__ == "__main__":
    print("\n🏁 RacingEnv Hardening Tests\n")

    print("Observation contract:")
    run_test("obs type from reset()", test_obs_type_reset)
    run_test("obs type from step()", test_obs_type_step)

    print("\nDeterministic seeding:")
    run_test("same seed → identical trajectory", test_deterministic_seed)
    run_test("different seeds → different tracks", test_different_seeds_differ)

    print("\nTermination logic:")
    run_test("off-track → terminated=True", test_off_track_terminates)
    run_test("max steps → truncated=True", test_max_step_truncation)
    run_test("terminated & truncated mutually exclusive", test_terminated_and_truncated_mutually_exclusive)

    print("\nEdge-case stress tests:")
    run_test("sustained reversing (500 steps)", test_reverse_no_crash)
    run_test("spin in place (500 steps)", test_spin_in_place)
    run_test("extreme steering oscillation (500 steps)", test_extreme_steering_oscillation)

    print(f"\n{'='*45}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*45}\n")

    sys.exit(1 if failed else 0)
