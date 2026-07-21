import pygame


class Track:
    def __init__(self):
        self.outer = pygame.Rect(100, 100, 600, 400)
        self.inner = pygame.Rect(250, 200, 300, 200)

    def draw(self, screen):
        pygame.draw.rect(screen, (80, 80, 80), self.outer)
        pygame.draw.rect(screen, (30, 30, 30), self.inner)
    
    def is_on_track(self, x, y):
        point = (int(x), int(y))

        return (
            self.outer.collidepoint(point)
            and not self.inner.collidepoint(point)
        )