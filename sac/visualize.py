import pygame
import torch

from env.environment import RacingEnv
from sac.agent import SACAgent


TRACK_SEED = 101


env = RacingEnv(
    max_steps=500,
    verbose=False,
)

agent = SACAgent()


# -------------------------
# Load trained actor
# -------------------------

agent.actor.load_state_dict(
    torch.load(
        "actor.pth",
        map_location="cpu",
    )
)

agent.actor.eval()

print("Loaded trained actor from actor.pth")


# -------------------------
# Visual evaluation
# -------------------------

pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("SAC Racing Agent")

clock = pygame.time.Clock()

observation, info = env.reset(
    options={"track_seed": TRACK_SEED}
)

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        mean, _ = agent.actor(
            observation_tensor
        )

        action = torch.tanh(mean).numpy()[0]

        print(
            "Action:",
            action,
            "Velocity:",
            env.car.velocity,
        )

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    # -------------------------
    # Draw
    # -------------------------

    screen.fill((30, 30, 30))

    env.track.draw(screen)

    car = env.car

    car_surface = pygame.Surface(
        (40, 20),
        pygame.SRCALPHA,
    )

    car_surface.fill((255, 0, 0))

    rotated_surface = pygame.transform.rotate(
        car_surface,
        car.angle,
    )

    rotated_rect = rotated_surface.get_rect(
        center=(car.x, car.y)
    )

    screen.blit(
        rotated_surface,
        rotated_rect,
    )

    car.draw_rays(
        screen,
        env.track,
    )

    pygame.display.flip()

    clock.tick(60)

    if terminated or truncated:

        print(
            "Episode ended:",
            info,
        )

        observation, info = env.reset(
            options={"track_seed": TRACK_SEED}
        )


pygame.quit()