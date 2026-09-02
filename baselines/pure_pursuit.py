"""Pure-pursuit / follow-centerline baseline for RL Racing Agent.

Continuous-action controller: a lookahead point is taken on the centerline, and
steering is driven by the heading error to that point plus a signed cross-track
term. Throttle is a proportional speed controller against a curvature-aware
speed cap (it brakes when over the cap).

Usage:
  python -m baselines.pure_pursuit                 # headless batch, medium tracks
  python -m baselines.pure_pursuit --config default
  python -m baselines.pure_pursuit --visual        # pygame visualization (first track)
  python -m baselines.pure_pursuit --visual 303    # pygame visualization (track 303)
  python -m baselines.pure_pursuit --debug         # headless with per-step debug output
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

import math
import sys

import numpy as np

from env.environment import RacingEnv

# ─── Configuration ───────────────────────────────────────────────────────────

TRACK_SEEDS = [101, 202, 303, 404, 505]

MEDIUM_TRACK = {
    "width": 70,
    "base_r": 250,
    "n_ctrl": 10,
    "min_radius": 80,
    "cx": 400,
    "cy": 300,
}

TRACK_CONFIGS = {
    "medium": MEDIUM_TRACK,
    "default": {},
}

LOOKAHEAD = 55.0        # pixels ahead on centerline to aim for

# Steering gains. steer_deg is later normalised by car.max_steering.
K_HEADING = 1.15        # deg of steer per deg of heading error to the lookahead point
K_CROSS = 0.10          # deg of steer per pixel of cross-track error

# Speed control. The car's turning radius is speed-independent (kinematic model),
# but higher speed means more distance covered per control step, so tracking error
# grows in corners — hence the curvature-aware cap.
SPEED_CAP = 2.6         # straight-line cruising cap
CORNER_SPEED = 1.5      # cap when the corner is tight
CORNER_ERR_LO = 8.0     # |heading err| below this -> full SPEED_CAP
CORNER_ERR_HI = 35.0    # |heading err| above this -> CORNER_SPEED
K_THROTTLE = 2.0        # proportional gain on (target_speed - velocity)

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
    """Continuous pure-pursuit controller.

    Steering combines two error signals the env already exposes:
      - heading error to a lookahead point on the centerline (primary)
      - signed cross-track distance from the centerline (recentring)
    Both are converted to a desired steering angle in degrees, then normalised
    by car.max_steering and clipped into the action range.

    Throttle is proportional control toward a curvature-aware speed cap, so it
    brakes (negative throttle) whenever the car is over the cap for the corner.

    Returns (action, debug_dict) where action is np.float32 array [steer, throttle].
    """
    car = env.car
    track = env.track

    # --- Lookahead target ---
    target_x, target_y = get_target_point(track, car.x, car.y, LOOKAHEAD)

    # --- Heading error to the target (y-up convention, matches car.angle) ---
    dx = target_x - car.x
    dy = target_y - car.y
    target_angle = math.degrees(math.atan2(-dy, dx))

    heading_err = (car.angle - target_angle + 180.0) % 360.0 - 180.0

    # --- Signed cross-track error (+ = right of centerline) ---
    cross_err = track.signed_distance(car.x, car.y)

    # --- Steering command (degrees, then normalised) ---
    # heading_err > 0 (pointing left of target)  -> steer right -> negative
    # cross_err  > 0 (right of centerline)       -> steer left  -> positive
    steer_deg = -K_HEADING * heading_err + K_CROSS * cross_err

    steering = float(np.clip(steer_deg / car.max_steering, -1.0, 1.0))

    # --- Curvature-aware speed cap ---
    abs_err = abs(heading_err)
    if abs_err <= CORNER_ERR_LO:
        target_speed = SPEED_CAP
    elif abs_err >= CORNER_ERR_HI:
        target_speed = CORNER_SPEED
    else:
        frac = (abs_err - CORNER_ERR_LO) / (CORNER_ERR_HI - CORNER_ERR_LO)
        target_speed = SPEED_CAP + frac * (CORNER_SPEED - SPEED_CAP)

    # --- Proportional throttle: brakes when over the cap ---
    throttle = float(
        np.clip(K_THROTTLE * (target_speed - car.velocity), -1.0, 1.0)
    )

    action = np.array([steering, throttle], dtype=np.float32)

    dbg = {
        "target": (round(target_x, 1), round(target_y, 1)),
        "heading_err": round(heading_err, 1),
        "cross_err": round(cross_err, 1),
        "target_speed": round(target_speed, 2),
        "speed": round(car.velocity, 3),
        "steering": round(steering, 3),
        "throttle": round(throttle, 3),
    }

    return action, dbg


# ─── Headless evaluation ────────────────────────────────────────────────────

def run_headless(track_seed, debug=False, track_kwargs=None, max_steps=2000):
    """Run one episode headlessly. Returns summary dict."""
    env = RacingEnv(max_steps=max_steps, verbose=False,
                    track_kwargs=track_kwargs or {})
    obs, info = env.reset(options={"track_seed": track_seed})

    total_reward = 0.0
    off_track_steps = 0

    for step in range(max_steps):
        action, dbg = choose_action(env, step_count=step, debug=debug)

        if debug:
            print(
                f"  step={step:4d}  "
                f"steer={dbg['steering']:+6.3f}  "
                f"throttle={dbg['throttle']:+6.3f}  "
                f"head_err={dbg['heading_err']:+7.1f}°  "
                f"cross={dbg['cross_err']:+6.1f}  "
                f"speed={dbg['speed']:5.3f}/{dbg['target_speed']:4.2f}  "
                f"target={dbg['target']}"
            )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info["crashed"]:
            off_track_steps += 1

        if terminated or truncated:
            break

    steps = env.step_count
    result = {
        "track_seed": track_seed,
        "steps": steps,
        "lap_progress": round(env.lap_progress, 4),
        "lap_completed": env.lap_completed,
        "lap_time": steps if env.lap_completed else None,
        "off_track_rate": round(off_track_steps / max(steps, 1), 4),
        "total_reward": round(total_reward, 4),
        "crashed": info.get("crashed", False),
    }

    status = "✓ LAP" if result["lap_completed"] else ("✗ CRASH" if result["crashed"] else "— TIMEOUT")
    print(
        f"Track {track_seed}: "
        f"steps={result['steps']:4d}, "
        f"lap_progress={result['lap_progress']:.4f}, "
        f"lap_time={str(result['lap_time']):>4s}, "
        f"off_track_rate={result['off_track_rate']:.2%}, "
        f"reward={result['total_reward']:+9.2f}  "
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


def run_visual(track_seed, track_kwargs=None):
    """Run one episode with pygame rendering."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Pure Pursuit — Track {track_seed}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    env = RacingEnv(max_steps=2000, verbose=False, track_kwargs=track_kwargs or {})
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
            f"steer: {dbg['steering']:+.3f}  throttle: {dbg['throttle']:+.3f}",
            f"head_err: {dbg['heading_err']:+.1f}°",
            f"cross_err: {dbg['cross_err']:+.1f}",
            f"speed: {dbg['speed']:.3f} / {dbg['target_speed']:.2f}",
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

    config_name = "medium"
    if "--config" in args:
        ci = args.index("--config")
        config_name = args[ci + 1]
        del args[ci:ci + 2]
    track_kwargs = TRACK_CONFIGS[config_name]

    if "--visual" in args:
        args.remove("--visual")
        seed = int(args[0]) if args else TRACK_SEEDS[0]
        run_visual(seed, track_kwargs=track_kwargs)

    else:
        debug = "--debug" in args

        print("=" * 88)
        print(f"Pure-Pursuit Baseline — Batch Evaluation  [{config_name} tracks]")
        print(f"  LOOKAHEAD={LOOKAHEAD}  K_HEADING={K_HEADING}  K_CROSS={K_CROSS}  "
              f"SPEED_CAP={SPEED_CAP}  CORNER_SPEED={CORNER_SPEED}")
        print("=" * 88)

        results = []
        for seed in TRACK_SEEDS:
            r = run_headless(seed, debug=debug, track_kwargs=track_kwargs)
            results.append(r)

        print("-" * 88)
        laps = sum(1 for r in results if r["lap_completed"])
        crashes = sum(1 for r in results if r["crashed"])
        avg_prog = sum(r["lap_progress"] for r in results) / len(results)
        avg_rew = sum(r["total_reward"] for r in results) / len(results)
        lap_times = [r["lap_time"] for r in results if r["lap_time"] is not None]
        avg_lap = sum(lap_times) / len(lap_times) if lap_times else None
        print(
            f"Summary: {laps}/{len(results)} laps completed, "
            f"{crashes} crashes, "
            f"avg_lap_time={avg_lap if avg_lap is None else round(avg_lap, 1)}, "
            f"avg_progress={avg_prog:.4f}, "
            f"avg_reward={avg_rew:+.2f}"
        )
