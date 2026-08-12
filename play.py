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
from env.environment import RacingEnv

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

        # Priority: throttle/brake first, then steering, else coast.
        # We can only send one discrete action per step, but steering
        # is also settable directly, so we handle both.
        action = 0  # coast

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 1  # accelerate
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 2  # brake

        # Steering is handled via actions 3/4 so it goes through
        # the same env logic (clamping, decay on coast).
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = 3  # steer left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = 4  # steer right

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
