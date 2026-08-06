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

    def cast_rays(self, track, ray_angles=(-90, -45, 0, 45, 90), max_dist=200, step=4):
        distances = []
        for ra in ray_angles:
            a = math.radians(self.angle + ra)
            dx = math.cos(a)
            dy = -math.sin(a)
            dist = max_dist
            d = 0.0
            while d <= max_dist:
                px = self.x + dx * d
                py = self.y + dy * d
                if not track.is_on_track(px, py):
                    dist = d
                    break
                d += step
            distances.append(float(dist))
        return distances

    def draw_rays(self, screen, track, ray_angles=(-90, -45, 0, 45, 90), max_dist=200, step=4):
        import pygame
        for ra in ray_angles:
            a = math.radians(self.angle + ra)
            dx = math.cos(a)
            dy = -math.sin(a)
            dist = max_dist
            d = 0.0
            while d <= max_dist:
                px = self.x + dx * d
                py = self.y + dy * d
                if not track.is_on_track(px, py):
                    dist = d
                    break
                d += step
            end_x = self.x + dx * dist
            end_y = self.y + dy * dist
            pygame.draw.line(screen, (255, 60, 60),
                             (int(self.x), int(self.y)),
                             (int(end_x), int(end_y)), 2)
            pygame.draw.circle(screen, (255, 255, 0), (int(end_x), int(end_y)), 4)