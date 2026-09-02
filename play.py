"""
play.py — Manual-drive mode for sanity-checking edge cases.

Arrow keys or WASD for control. Prints terminated/truncated/reward each step.
Auto-resets on episode end so you can keep playing.

Usage:
    python play.py
    python play.py --seed 42      # deterministic track
"""

import pygame
import sys
import numpy as np
from env.environment import RacingEnv

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
# ──────────────────────────────────────────────────────────────────────────────────────

def main():
    seed = None
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed = int(sys.argv[idx + 1])

    pygame.init()
    clock = pygame.time.Clock()

    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Racing Agent — Manual Play")

    env = RacingEnv(verbose=False)
    obs, _ = env.reset(seed=seed)
    episode = 1
    print(f"\n--- Episode {episode} ---")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ── Read held keys → pick action ──
        keys = pygame.key.get_pressed()

        # The continuous API takes steering and throttle together, so
        # (unlike the old discrete mapping) you can steer and accelerate
        # in the same step — no priority hack needed.
        throttle = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0       # accelerate
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle = -1.0      # brake / reverse

        steering = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = 1.0       # full left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = -1.0      # full right

        action = np.array([steering, throttle], dtype=np.float32)

        # ── Step ──
        obs, reward, terminated, truncated, info = env.step(action)

        # ── Terminal output ──
        status = ""
        if terminated:
            status = " ** TERMINATED (off-track)"
        elif truncated:
            status = " ** TRUNCATED (max steps)"

        print(
            f"step {env.step_count:4d}"
            f" | reward: {reward:+.4f}"
            f" | speed: {env.car.velocity:.2f}"
            f" | steering: {env.car.steering:.1f}"
            f"{status}"
        )

        # ── Auto-reset ──
        if terminated or truncated:
            episode += 1
            print(f"\n--- Episode {episode} ---")
            obs, _ = env.reset(seed=seed)

        # ── Draw ──
        car = env.car
        track = env.track

        screen.fill((30, 30, 30))
        track.draw(screen)

        # On-track indicator (green dot = on track, red = off)
        if track.is_on_track(car.x, car.y):
            pygame.draw.circle(screen, (0, 255, 0), (30, 30), 10)
        else:
            pygame.draw.circle(screen, (255, 0, 0), (30, 30), 10)

        # Car
        car_surface = pygame.Surface((40, 20), pygame.SRCALPHA)
        car_surface.fill((255, 0, 0))
        rotated_surface = pygame.transform.rotate(car_surface, car.angle)
        rotated_rect = rotated_surface.get_rect(center=(car.x, car.y))
        screen.blit(rotated_surface, rotated_rect)

        # Rays
        car.draw_rays(screen, track)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
