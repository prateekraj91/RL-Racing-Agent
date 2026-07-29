from env.car import Car
from env.track import Track
import gymnasium as gym


class RacingEnv(gym.Env):
    def __init__(self):
        self.track = Track()
        self.reset()
        self.action_space = gym.spaces.Discrete(5)
        
        self.observation_space = gym.spaces.Box(
            low=-1000,
            high=1000,
            shape=(4,),
            dtype=float,
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car = Car()

        observation = (
            self.car.x,
            self.car.y,
            self.car.angle,
            self.car.velocity,
        )

        info = {}

        return observation, info

    def step(self, action):
        # Handle actions
        if action == 1:
            self.car.velocity += self.car.acceleration

        elif action == 2:
            self.car.velocity -= self.car.acceleration

        elif action == 3:
            self.car.steering = max(
            self.car.steering - 2,
            -self.car.max_steering,
            )

        elif action == 4:
            self.car.steering = min(
            self.car.steering + 2,
            self.car.max_steering,
            )

        else:
            self.car.steering *= 0.9

        # Save previous position
        old_x = self.car.x
        old_y = self.car.y

        # Update car physics
        self.car.update()

        # Collision with track
        #if not self.track.is_on_track(self.car.x, self.car.y):
         #   self.car.x = old_x
          #  self.car.y = old_y
           # self.car.velocity = 0

        # Observation
        observation = (
            self.car.x,
            self.car.y,
            self.car.angle,
            self.car.velocity,
        )
        
        reward = 0

        if self.track.is_on_track(self.car.x, self.car.y):
            reward += 0.1
            
        reward += self.car.velocity * 0.1

        terminated = False
        truncated = False

        info = {}

        return observation, reward, terminated, truncated, info 