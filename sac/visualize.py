"""Pygame visualisation of a trained SAC actor.

Usage:
    python -m sac.visualize                                   # unchanged default
    python -m sac.visualize --checkpoint runs/exp2_demos_s42/solved_actor.pth
    python -m sac.visualize --config default --seed 303
    python -m sac.visualize --grip                        # grip-limited physics

    # the grip-trained agent in the world it was trained on (docs/GRIP_EXPERIMENT.md)
    python -m sac.visualize --checkpoint runs/grip_bites_curric_s42/solved_actor.pth \
        --config tight --seed 24 --grip

Defaults reproduce the previous hardcoded behaviour exactly: best_actor.pth on
the medium track, track_seed 101, max_steps 500.
"""

import argparse

import pygame
import torch

from env.environment import RacingEnv
from sac.agent import SACAgent
from sac.curricula import MEDIUM_TRACK_KWARGS, TIGHT_TRACK_KWARGS, TIGHT_TARGET_CONFIG


# Same named configs as sac/train.py's CONFIGS. That module runs argparse at
# import time, so its dict cannot be imported here; MEDIUM_TRACK_KWARGS in
# sac/curricula.py is the shared definition of the medium track.
CONFIGS = {
    "medium": {
        "track_kwargs": MEDIUM_TRACK_KWARGS,
        "max_steps": 500,
    },
    "default": {
        "track_kwargs": {},          # env built-in defaults (base_r=210, min_radius=70)
        "max_steps": 500,
    },
    # The grip experiment's target. Pair it with --seed 24 --grip: the track is
    # only tight enough for grip to bite on that seed (realized corner radius
    # 64.5px), and without --grip the physics it was trained under are absent.
    "tight": {
        "track_kwargs": TIGHT_TRACK_KWARGS,
        "max_steps": TIGHT_TARGET_CONFIG["max_steps"],
    },
}


parser = argparse.ArgumentParser(description="Visualise a trained SAC actor")
parser.add_argument("--checkpoint", default="best_actor.pth",
                    help="actor state_dict to load (default: best_actor.pth)")
parser.add_argument("--config", choices=list(CONFIGS), default="medium",
                    help="named track config (default: medium)")
parser.add_argument("--seed", type=int, default=101,
                    help="track seed (default: 101; use 24 with --config tight)")
parser.add_argument("--grip", action="store_true", default=False,
                    help="enable grip-limited cornering physics (default: off)")
args = parser.parse_args()


cfg = CONFIGS[args.config]
TRACK_SEED = args.seed
track_kwargs = cfg["track_kwargs"]
max_steps = cfg["max_steps"]


env = RacingEnv(
    max_steps=max_steps,
    verbose=False,
    track_kwargs=track_kwargs,
    grip_limit=args.grip,
)

agent = SACAgent()


# -------------------------
# Load trained actor
# -------------------------

agent.actor.load_state_dict(
    torch.load(
        args.checkpoint,
        map_location="cpu",
    )
)

agent.actor.eval()

print(f"Loaded trained actor from {args.checkpoint}")
print(f"config: {args.config}  |  track_seed={TRACK_SEED}  |  max_steps={max_steps}")
print(f"track_kwargs: {track_kwargs}")
print(f"grip physics: {'ON  (grip-limited cornering)' if args.grip else 'OFF (default)'}")


# -------------------------
# Visual evaluation
# -------------------------

pygame.init()

font = pygame.font.SysFont(None, 24)

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

    hud = [
        f"Speed: {env.car.velocity:.2f}",
        f"Lap progress: {info.get('lap_progress', 0.0):.3f}",
        f"Reward: {reward:.4f}",
        f"Steering: {action[0]:.2f}",
        f"Throttle: {action[1]:.2f}",
        f"Crashed: {info.get('crashed', False)}",
    ]

    y = 10

    for line in hud:
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, y))
        y += 25

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