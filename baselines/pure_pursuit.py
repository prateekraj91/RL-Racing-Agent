"""Pure-pursuit / follow-centerline baseline for RL Racing Agent.

Uses a priority-based action selector that works within Discrete(5):
  - Large angle error  → steer (action 3 or 4)
  - Heading OK, too slow → accelerate (action 1)
  - Heading OK, too fast → brake (action 2)
  - Otherwise           → coast (action 0)

Usage:
  python -m baselines.pure_pursuit              # headless batch on all 5 tracks
  python -m baselines.pure_pursuit --visual     # pygame visualization (first track)
  python -m baselines.pure_pursuit --visual 303 # pygame visualization (track 303)
  python -m baselines.pure_pursuit --debug      # headless with per-step debug output
"""

import math
import sys

from env.environment import RacingEnv

# ─── Configuration ───────────────────────────────────────────────────────────

TRACK_SEEDS = [101, 202, 303, 404, 505]

LOOKAHEAD = 80.0        # pixels ahead on centerline to aim for
SPEED_CAP = 2.0         # target cruising speed
MIN_SPEED = 0.15        # below this, must accelerate (can't steer at v≈0)

# Proportional steering controller.
# With fixed car physics (dt*60 angular scaling), the car's effective
# turning radius is ~87px at max steering, which can follow all track
# curves (min_radius=70) with some use of the 35px track half-width.
STEERING_GAIN = 1.0     # desired_steering = gain * angle_error
STEERING_TOLERANCE = 4.0  # don't adjust steering if within this of desired

WIDTH = 800
HEIGHT = 600


# ─── Lookahead target ───────────────────────────────────────────────────────

def get_target_point(track, x, y, lookahead):
    """Find a point `lookahead` pixels ahead on the centerline.

    Uses the fractional projection `t` from track._nearest() so the
    walk starts from the car's actual nearest point on the centerline,
    not from the vertex before it.
    """
    i, t, proj, _ = track._nearest(x, y)

    cl = track.centerline
    n = len(cl)

    # Start from the nearest projection point.
    # The remaining distance in the current segment is (1 - t) * seg_len.
    next_i = (i + 1) % n
    seg_vec = cl[next_i] - cl[i]
    seg_len = math.hypot(seg_vec[0], seg_vec[1])
    remaining_in_seg = (1.0 - t) * seg_len

    if remaining_in_seg >= lookahead:
        # Target lies within this same segment.
        frac = t + lookahead / max(seg_len, 1e-9)
        target = cl[i] + frac * seg_vec
        return float(target[0]), float(target[1])

    distance = remaining_in_seg
    idx = next_i

    while distance < lookahead:
        next_idx = (idx + 1) % n
        p1 = cl[idx]
        p2 = cl[next_idx]
        seg = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if distance + seg >= lookahead:
            # Target is within this segment.
            leftover = lookahead - distance
            frac = leftover / max(seg, 1e-9)
            target = p1 + frac * (p2 - p1)
            return float(target[0]), float(target[1])

        distance += seg
        idx = next_idx

    # Fallback (shouldn't happen on a closed track).
    return float(cl[idx][0]), float(cl[idx][1])


# ─── Action selection ────────────────────────────────────────────────────────

def choose_action(env, step_count=0, debug=False):
    """Proportional steering controller exploiting steering persistence.

    Key physics facts from the environment:
      - Discrete(5): each step does exactly ONE of: coast/accel/brake/left/right
      - action 1 (accel) and action 2 (brake) do NOT change car.steering
      - action 0 (coast) decays steering *= 0.9
      - actions 3/4 change steering by ±2°
      - angular_velocity = velocity / turning_radius  (proportional to speed)
      - So: steering PERSISTS during accel/brake. Once set, we can accelerate
        and the car keeps turning — and turns FASTER as speed increases.

    Strategy:
      1. Compute desired_steering = GAIN * angle_error (clamped to ±max)
      2. If car.steering is far from desired → send steer action to adjust
      3. Otherwise → send accel/brake/coast for speed control
      4. Special case: velocity ≈ 0 → always accelerate (can't steer stopped)

    Returns (action, debug_dict).
    """
    car = env.car
    track = env.track

    # --- Lookahead target ---
    target_x, target_y = get_target_point(track, car.x, car.y, LOOKAHEAD)

    # --- Angle error ---
    dx = target_x - car.x
    dy = target_y - car.y
    target_angle = math.degrees(math.atan2(-dy, dx))

    angle_error = target_angle - car.angle
    angle_error = (angle_error + 180.0) % 360.0 - 180.0

    # --- Proportional steering target ---
    # angle_error > 0 → target is left (CCW) → need positive steering
    # angle_error < 0 → target is right (CW)  → need negative steering
    desired_steering = max(-car.max_steering,
                           min(car.max_steering, STEERING_GAIN * angle_error))
    steering_error = desired_steering - car.steering
    steering_ok = abs(steering_error) < STEERING_TOLERANCE

    # --- Dynamic speed cap ---
    # Slow down in curves: reduce effective speed cap when angle error is large.
    # At 0° error → full SPEED_CAP; at ±30°+ → reduced to MIN_SPEED.
    abs_err = abs(angle_error)
    if abs_err < 10:
        effective_cap = SPEED_CAP
    else:
        # Linear ramp-down: from SPEED_CAP at 10° to a floor of 1.0 at 45°+
        t = min((abs_err - 10) / 35.0, 1.0)
        effective_cap = SPEED_CAP * (1.0 - t) + 1.0 * t

    # --- Decision logic ---
    reason = ""

    if car.velocity < MIN_SPEED:
        # Can't steer at near-zero speed. Must accelerate first.
        action = 1
        reason = "bootstrap"

    elif not steering_ok:
        # Steering needs adjustment. Each action changes steering by ±2°.
        # But also check: if we're going too fast for this curve, brake instead.
        if car.velocity > effective_cap + 0.5:
            action = 2
            reason = "curve_brake"
        elif steering_error > 0:
            action = 4   # steering += 2 (more positive → turn left)
            reason = "adj_steer_L"
        else:
            action = 3   # steering -= 2 (more negative → turn right)
            reason = "adj_steer_R"

    elif car.velocity < effective_cap:
        # Steering is set correctly. Accelerate — steering persists,
        # and higher speed means faster turning.
        action = 1
        reason = "accelerate"

    elif car.velocity > effective_cap + 0.3:
        action = 2
        reason = "brake"

    else:
        # At target speed, heading correct.
        # Only coast if steering is near zero (don't decay steering in curves).
        if abs(car.steering) < 2.0:
            action = 0
            reason = "coast"
        else:
            # Maintain current state — don't coast (would decay steering).
            # Send a no-op accelerate (velocity is near cap, so clamped anyway).
            action = 1
            reason = "hold"

    dbg = {
        "target": (round(target_x, 1), round(target_y, 1)),
        "target_angle": round(target_angle, 1),
        "angle_error": round(angle_error, 1),
        "desired_steer": round(desired_steering, 1),
        "speed": round(car.velocity, 3),
        "steering": round(car.steering, 2),
        "action": action,
        "reason": reason,
    }

    return action, dbg


# ─── Headless evaluation ────────────────────────────────────────────────────

def run_headless(track_seed, debug=False):
    """Run one episode headlessly. Returns summary dict."""
    env = RacingEnv(max_steps=2000, verbose=False)
    obs, info = env.reset(options={"track_seed": track_seed})

    total_reward = 0.0

    for step in range(2000):
        action, dbg = choose_action(env, step_count=step, debug=debug)

        if debug:
            print(
                f"  step={step:4d}  "
                f"action={dbg['action']}({dbg['reason']:>13s})  "
                f"angle_err={dbg['angle_error']:+7.1f}°  "
                f"desired={dbg['desired_steer']:+6.1f}  "
                f"speed={dbg['speed']:5.3f}  "
                f"steering={dbg['steering']:+6.2f}  "
                f"target={dbg['target']}"
            )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    result = {
        "track_seed": track_seed,
        "steps": env.step_count,
        "lap_progress": env.lap_progress,
        "lap_completed": env.lap_completed,
        "total_reward": total_reward,
        "crashed": info.get("crashed", False),
    }

    status = "✓ LAP" if result["lap_completed"] else ("✗ CRASH" if result["crashed"] else "— TIMEOUT")
    print(
        f"Track {track_seed}: "
        f"steps={result['steps']:4d}, "
        f"lap_progress={result['lap_progress']:.4f}, "
        f"lap_completed={result['lap_completed']}, "
        f"reward={result['total_reward']:+8.4f}, "
        f"crashed={result['crashed']}  "
        f"[{status}]"
    )

    env.close()
    return result


# ─── Visual test ─────────────────────────────────────────────────────────────

def draw_car(screen, car):
    import pygame
    car_surface = pygame.Surface((40, 20), pygame.SRCALPHA)
    car_surface.fill((255, 0, 0))

    rotated = pygame.transform.rotate(car_surface, car.angle)
    rect = rotated.get_rect(center=(car.x, car.y))
    screen.blit(rotated, rect)

    # Draw heading indicator.
    hx = car.x + 25 * math.cos(math.radians(car.angle))
    hy = car.y - 25 * math.sin(math.radians(car.angle))
    pygame.draw.line(screen, (255, 200, 0), (int(car.x), int(car.y)), (int(hx), int(hy)), 2)


def run_visual(track_seed):
    """Run one episode with pygame rendering."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Pure Pursuit — Track {track_seed}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    env = RacingEnv(max_steps=2000, verbose=False)
    obs, info = env.reset(options={"track_seed": track_seed})

    running = True
    total_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Action ---
        action, dbg = choose_action(env, step_count=env.step_count)
        target_x, target_y = dbg["target"]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Draw ---
        screen.fill((30, 30, 30))

        # Track
        env.track.draw(screen)

        # Car → target line
        pygame.draw.line(
            screen, (0, 255, 255),
            (int(env.car.x), int(env.car.y)),
            (int(target_x), int(target_y)), 2
        )

        # Lookahead target dot
        pygame.draw.circle(screen, (255, 255, 0), (int(target_x), int(target_y)), 7)

        # Nearest centerline point (green)
        _, _, proj, _ = env.track._nearest(env.car.x, env.car.y)
        pygame.draw.circle(screen, (0, 255, 0), (int(proj[0]), int(proj[1])), 5)

        # Car
        draw_car(screen, env.car)

        # Rays
        env.car.draw_rays(screen, env.track)

        # HUD
        hud_lines = [
            f"step: {env.step_count}",
            f"action: {dbg['action']} ({dbg['reason']})",
            f"angle_err: {dbg['angle_error']:+.1f}°",
            f"speed: {dbg['speed']:.3f}",
            f"steering: {dbg['steering']:+.2f}",
            f"progress: {env.lap_progress:.4f}",
            f"reward: {total_reward:+.2f}",
        ]
        for idx, line in enumerate(hud_lines):
            surf = font.render(line, True, (220, 220, 220))
            screen.blit(surf, (10, 10 + idx * 18))

        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            status = "LAP!" if env.lap_completed else ("CRASHED" if info["crashed"] else "TIMEOUT")
            print(
                f"Track {track_seed}: "
                f"steps={env.step_count}, "
                f"lap_progress={env.lap_progress:.4f}, "
                f"lap_completed={env.lap_completed}, "
                f"reward={total_reward:+.4f}, "
                f"crashed={info['crashed']}  "
                f"[{status}]"
            )
            pygame.time.wait(2000)
            running = False

    env.close()
    pygame.quit()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--visual" in args:
        args.remove("--visual")
        seed = int(args[0]) if args else TRACK_SEEDS[0]
        run_visual(seed)

    else:
        debug = "--debug" in args

        print("=" * 80)
        print("Pure-Pursuit Baseline — Batch Evaluation")
        print(f"  LOOKAHEAD={LOOKAHEAD}  SPEED_CAP={SPEED_CAP}  GAIN={STEERING_GAIN}  TOL={STEERING_TOLERANCE}°")
        print("=" * 80)

        results = []
        for seed in TRACK_SEEDS:
            r = run_headless(seed, debug=debug)
            results.append(r)

        print("-" * 80)
        laps = sum(1 for r in results if r["lap_completed"])
        crashes = sum(1 for r in results if r["crashed"])
        avg_prog = sum(r["lap_progress"] for r in results) / len(results)
        avg_rew = sum(r["total_reward"] for r in results) / len(results)
        print(
            f"Summary: {laps}/{len(results)} laps completed, "
            f"{crashes} crashes, "
            f"avg_progress={avg_prog:.4f}, "
            f"avg_reward={avg_rew:+.2f}"
        )