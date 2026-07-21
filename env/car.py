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

        self.x += self.velocity * math.cos(math.radians(self.angle))
        self.y -= self.velocity * math.sin(math.radians(self.angle))