import math


class Car:
    def __init__(self):
        self.x = 150
        self.y = 300

        self.angle = 0

        self.velocity = 0

        self.acceleration = 0.08
        self.max_speed = 4
        self.friction = 0.03
        
        self.steering = 0
        
        self.max_steering = 30      # degrees

        self.wheelbase = 50         # pixels

        self.dt = 1 / 60            # fixed timestep

    def update(self):
        if self.velocity > 0:
            self.velocity -= self.friction
            if self.velocity < 0:
                self.velocity = 0

        elif self.velocity < 0:
            self.velocity += self.friction
            if self.velocity > 0:
                self.velocity = 0

        self.velocity = max(-self.max_speed / 2, min(self.velocity, self.max_speed))

        heading = math.radians(self.angle)

        self.x += self.velocity * math.cos(heading) * self.dt * 60
        self.y -= self.velocity * math.sin(heading) * self.dt * 60

        if abs(self.steering) > 0.01:
            turning_radius = self.wheelbase / math.tan(math.radians(self.steering))

            angular_velocity = self.velocity / turning_radius

            self.angle += math.degrees(angular_velocity * self.dt * 10)