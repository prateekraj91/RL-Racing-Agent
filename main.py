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

car = env.car
track = env.track

running = True

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Keyboard input
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        observation, reward, terminated, truncated, info = env.step(1)
        print(f"Obs: {observation} | Reward: {reward:.2f}")


    if keys[pygame.K_s]:
        observation, reward, terminated, truncated, info = env.step(2)
        print(f"Obs: {observation} | Reward: {reward:.2f}")

    if keys[pygame.K_a]:
        observation, reward, terminated, truncated, info = env.step(3)
        print(f"Obs: {observation} | Reward: {reward:.2f}")

    if keys[pygame.K_d]:
        observation, reward, terminated, truncated, info = env.step(4)
        print(f"Obs: {observation} | Reward: {reward:.2f}")

    if not (keys[pygame.K_w] or keys[pygame.K_s] or
            keys[pygame.K_a] or keys[pygame.K_d]):
        observation, reward, terminated, truncated, info = env.step(0)

    car = env.car

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

    pygame.display.flip()
    clock.tick(60)
    

pygame.quit()