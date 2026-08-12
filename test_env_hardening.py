"""
test_env_hardening.py — Stress-tests for RacingEnv edge cases.

Runs headless (no pygame). Exercises:
  - observation dtype & shape (from both reset and step)
  - deterministic seeding (same seed → identical state)
  - different seeds produce different tracks
  - off-track driving → terminated=True
  - max-step truncation → truncated=True
  - reversing (sustained braking) doesn't crash
  - spinning in place (zero speed + max steering) doesn't hang
  - extreme steering oscillation doesn't crash

Usage:
    python test_env_hardening.py
"""

import numpy as np
import sys

from env.environment import RacingEnv

# ─────────────────────────────────────────────
# Helpers
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
    obs, reward, terminated, truncated, info = env.step(0)  # coast

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
    """reset(seed=N) must produce identical obs/track/car state both times."""
    env = RacingEnv()

    obs1, _ = env.reset(seed=42)
    x1, y1, angle1 = env.car.x, env.car.y, env.car.angle
    center1 = env.track.centerline.copy()

    obs2, _ = env.reset(seed=42)
    x2, y2, angle2 = env.car.x, env.car.y, env.car.angle
    center2 = env.track.centerline.copy()

    assert np.array_equal(obs1, obs2), (
        "Same seed produced different observations"
    )
    assert (x1, y1, angle1) == (x2, y2, angle2), (
        f"Same seed produced different car pose: "
        f"({x1},{y1},{angle1}) vs ({x2},{y2},{angle2})"
    )
    assert np.array_equal(center1, center2), (
        "Same seed produced different track centerlines"
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

    Strategy: accelerate + hard steer right. The car will spiral outward
    and eventually leave the track. We cap at 5000 steps as a safety net —
    if it hasn't gone off-track by then, the track is unusually wide or
    the physics are broken.
    """
    env = RacingEnv(max_steps=10000)  # high limit so truncation doesn't interfere
    env.reset(seed=99)

    terminated = False
    for i in range(5000):
        obs, reward, terminated, truncated, info = env.step(1)  # accelerate
        # also steer hard right every step
        env.car.steering = env.car.max_steering

        if terminated:
            break

    assert terminated, (
        f"Car never went off-track in 5000 steps of full throttle + max steering"
    )


def test_max_step_truncation():
    """Coasting for max_steps must set truncated=True (not terminated).

    Strategy: use a small max_steps and just coast (action=0). The car
    starts on-track with zero velocity, so it won't move and won't crash.
    """
    max_steps = 50
    env = RacingEnv(max_steps=max_steps)
    env.reset(seed=0)

    terminated = False
    truncated = False
    for i in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(0)  # coast
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

    Strategy: hold brake (action=2) for 500 steps. The car should reverse.
    The env must not raise, and observations must be finite.
    """
    env = RacingEnv(max_steps=1000)
    env.reset(seed=7)

    for i in range(500):
        obs, reward, terminated, truncated, info = env.step(2)  # brake

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

    Strategy: coast (don't accelerate) but set max steering. The car has
    zero initial velocity, so it barely moves. We run 500 steps.
    """
    env = RacingEnv(max_steps=1000)
    env.reset(seed=3)

    for i in range(500):
        # Steer left without accelerating
        obs, reward, terminated, truncated, info = env.step(3)

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

    Strategy: accelerate + alternate steer-left (3) / steer-right (4)
    every step. This creates wild oscillation. Run 500 steps.
    """
    env = RacingEnv(max_steps=1000)
    env.reset(seed=5)

    for i in range(500):
        # Alternate: accelerate on even steps, steer-left on odd, steer-right on even
        if i % 2 == 0:
            obs, reward, terminated, truncated, info = env.step(4)  # steer right
        else:
            obs, reward, terminated, truncated, info = env.step(3)  # steer left

        # Also accelerate by bumping velocity directly (we can only send one action)
        if i % 3 == 0:
            obs, reward, terminated, truncated, info = env.step(1)  # accelerate

        assert np.all(np.isfinite(obs)), (
            f"Non-finite obs at step {i} during steering oscillation: {obs}"
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
        action = 1 if i % 5 == 0 else 0  # occasional throttle
        obs, reward, terminated, truncated, info = env.step(action)

        assert not (terminated and truncated), (
            f"Both terminated AND truncated are True at step {i}"
        )

        if terminated or truncated:
            break


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏁 RacingEnv Hardening Tests\n")

    print("Observation contract:")
    run_test("obs type from reset()", test_obs_type_reset)
    run_test("obs type from step()", test_obs_type_step)

    print("\nDeterministic seeding:")
    run_test("same seed → identical state", test_deterministic_seed)
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
