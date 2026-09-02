import pygame
import numpy as np
from env.environment import RacingEnv
import math
from env.track import Track

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

# Constant steering angle for the turning-circle validation, in degrees.
# Normalised into the action range below.
TURN_CIRCLE_STEER_DEG = 20.0

pygame.init()

clock = pygame.time.Clock()

WIDTH = 800
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Racing Agent")

env = RacingEnv()
for i in range(20):
    env.reset()

    x, y, heading = env.track.start_pose()
    print(f"Track {i+1}: start=({x:.1f}, {y:.1f}), heading={heading:.1f}")

    screen.fill((30, 30, 30))
    env.track.draw(screen)

    pygame.display.flip()
    pygame.image.save(screen, f"track_{i+1}.png")
    pygame.time.wait(300)  # show each track for 1 second

car = env.car
track = env.track

running = True

trajectory = []

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Constant throttle + steering test.
    # The continuous API sets steering and throttle in the same step, so the
    # old post-step `env.car.steering = 20` poke is no longer needed.
    action = np.array(
        [TURN_CIRCLE_STEER_DEG / env.car.max_steering, 1.0],
        dtype=np.float32,
    )
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Obs: {observation} | Reward: {reward:.2f}")

    car = env.car

    trajectory.append((car.x, car.y))

    # Draw everything
    screen.fill((30, 30, 30))

    track.draw(screen)

    if track.is_on_track(car.x, car.y):
        status_color = (0, 255, 0)
    else:
        status_color = (255, 0, 0)

    pygame.draw.circle(screen, status_color, (30, 30), 10)

    car_surface = pygame.Surface((40, 20), pygame.SRCALPHA)
    car_surface.fill((255, 0, 0))

    rotated_surface = pygame.transform.rotate(car_surface, car.angle)

    rotated_rect = rotated_surface.get_rect(center=(car.x, car.y))

    screen.blit(rotated_surface, rotated_rect)

    car.draw_rays(screen, track)

    pygame.display.flip()
    clock.tick(60)


pygame.quit()

import matplotlib.pyplot as plt

xs = [p[0] for p in trajectory]
ys = [p[1] for p in trajectory]

plt.figure(figsize=(6, 6))
plt.plot(xs, ys)
plt.gca().set_aspect("equal")
plt.title("Turning Circle Validation")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

plt.savefig("turning_circle.png")
print("Saved turning_circle.png")



    