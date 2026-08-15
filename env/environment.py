import numpy as np
import gymnasium as gym

from env.car import Car
from env.track import Track
import math

class RacingEnv(gym.Env):

    def __init__(self, max_steps=2000, verbose=False):
        super().__init__()

        self.track = Track()
        self.car = Car()

        self.max_steps = max_steps
        self.step_count = 0
        self.verbose = verbose

        # 0 = coast
        # 1 = accelerate
        # 2 = brake
        # 3 = steer left
        # 4 = steer right
        self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Box(
            low=-1000,
            high=1000,
            shape=(9,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Derive a deterministic track seed from Gymnasium's RNG.
        # self.np_random was seeded by super().reset(seed=seed) above.
        # If seed=None was passed, self.np_random is random → random track (same as before).
        # If seed=N was passed, self.np_random is deterministic → same track every time.



        if options is not None and "track_seed" in options:
            track_seed = options["track_seed"]
        else:
            track_seed = int(self.np_random.integers(0, 2**31))

        self.track = Track(seed=track_seed)


        self.car = Car()

        self.step_count = 0

        self.car.x, self.car.y, self.car.angle = self.track.start_pose()

        self.previous_progress = self.track.get_progress(
            self.car.x,
            self.car.y,
        )
        self.lap_progress = 0.0
        self.lap_completed = False

        err = (
            self.car.angle
            - self.track.track_heading(
                self.car.x,
                self.car.y,
            )
        )

        err = (err + 180.0) % 360.0 - 180.0

        dist = self.track.signed_distance(
            self.car.x,
            self.car.y,
        )

        slip = 0.0

        rays = self.car.cast_rays(self.track)

        observation = np.array(
            [
                self.car.velocity,
                err,
                dist,
                slip,
                *rays,
            ],
            dtype=np.float32,
        )

        info = {}

        return observation, info

    def step(self, action):

        # -------------------------
        # Handle action
        # -------------------------

        if action == 1:
            # Accelerate
            self.car.velocity += self.car.acceleration

        elif action == 2:
            # Brake
            self.car.velocity -= self.car.acceleration

        elif action == 3:
            # Steer left
            self.car.steering = max(
                self.car.steering - 2,
                -self.car.max_steering,
            )

        elif action == 4:
            # Steer right
            self.car.steering = min(
                self.car.steering + 2,
                self.car.max_steering,
            )

        else:
            # Coast
            self.car.steering *= 0.9

        # -------------------------
        # Update physics
        # -------------------------

        self.car.update()

        # -------------------------
        # Calculate progress
        # -------------------------

        current_progress = self.track.get_progress(
            self.car.x,
            self.car.y,
        )

        progress = current_progress - self.previous_progress

        if progress < -0.5:
            progress += 1.0
        elif progress > 0.5:
            progress -= 1.0

        
        self.lap_progress += progress
        self.previous_progress = current_progress

        if self.lap_progress >= 1.0:
            self.lap_completed = True

        # -------------------------
        # Check whether car crashed
        # -------------------------

        crashed = not self.track.is_on_track(
            self.car.x,
            self.car.y,
        )

        # -------------------------
        # Reward
        # -------------------------

        speed_reward = self.car.velocity * 0.01

        reward = progress + speed_reward

        # Off-track / crash penalty
        if crashed:
            reward -= 2.0

        # -------------------------
        # Episode termination
        # -------------------------

        self.step_count += 1

        terminated = crashed
        truncated = (not terminated) and (self.step_count >= self.max_steps)

        # -------------------------
        # Debug information
        # -------------------------

        err = (
            self.car.angle
            - self.track.track_heading(
                self.car.x,
                self.car.y,
            )
        )

        err = (err + 180.0) % 360.0 - 180.0

        dist = self.track.signed_distance(
            self.car.x,
            self.car.y,
        )

        speed = math.hypot(self.car.vx, self.car.vy)

        if speed > 1e-6:
            velocity_angle = math.degrees(
                math.atan2(-self.car.vy, self.car.vx)
            )
            slip = velocity_angle - self.car.angle
            slip = (slip + 180.0) % 360.0 - 180.0
        else:
            slip = 0.0

        rays = self.car.cast_rays(self.track)

        if self.verbose:
            print(
                "progress:",
                round(current_progress, 4),
                "delta:",
                round(progress, 4),
                "| reward:",
                round(reward, 4),
            )

            print(
                "heading err:",
                round(err, 1),
                "| dist:",
                round(dist, 1),
                "| slip:",
                round(slip, 2),
                "| rays:",
                [round(r) for r in rays],
            )

        # -------------------------
        # Observation
        # -------------------------

        observation = np.array(
            [
                self.car.velocity,
                err,
                dist,
                slip,
                *rays,
            ],
            dtype=np.float32,
        )

        info = {
            "progress": progress,
            "lap_progress": self.lap_progress,
            "speed_reward": speed_reward,
            "crashed": crashed,
            "lap_completed": self.lap_completed,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )