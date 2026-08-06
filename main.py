import pygame
from env.environment import RacingEnv
import math
from env.track import Track

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
    pygame.time.wait(300)   # show each track for 1 second

car = env.car
track = env.track

running = True

trajectory = []

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Constant throttle + steering test
    observation, reward, terminated, truncated, info = env.step(1)  # accelerate
    env.car.steering = 20  # constant steering angle
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