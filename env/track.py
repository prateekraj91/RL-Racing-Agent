import numpy as np
import pygame


class Track:
    """Procedurally generated closed-loop race track.

    Public interface (unchanged, relied on by environment.py / main.py):
        Track()                      -> constructable with no args (random track)
        track.is_on_track(x, y)      -> bool
        track.draw(screen)           -> renders the track band top-down

    New helpers:
        track.centerline             -> (N, 2) ndarray of the spline centerline
        track.start_pose()           -> (x, y, heading_deg) at the start line
    """

    def __init__(self, seed=None, width=70, base_r=210, n_ctrl=8,
                 min_radius=70, cx=400, cy=300):
        self.width = width
        self.half = width / 2.0
        self.cx, self.cy = cx, cy
        self.centerline = self._generate(seed, base_r, n_ctrl, min_radius)
        # precompute segment endpoints for the distance test
        self._a = self.centerline
        self._b = np.roll(self.centerline, -1, axis=0)

    # ---------- generation ----------
    @staticmethod
    def _catmull_rom_closed(P, samples_per_seg=40):
        n = len(P); pts = []
        for i in range(n):
            p0 = P[(i - 1) % n]; p1 = P[i]; p2 = P[(i + 1) % n]; p3 = P[(i + 2) % n]
            t = np.linspace(0, 1, samples_per_seg, endpoint=False)[:, None]
            pts.append(0.5 * ((2*p1) + (-p0 + p2)*t
                              + (2*p0 - 5*p1 + 4*p2 - p3)*t**2
                              + (-p0 + 3*p1 - 3*p2 + p3)*t**3))
        return np.vstack(pts)

    @staticmethod
    def _min_curv_radius(C):
        win = max(2, len(C) // 60)
        prev = np.roll(C, win, 0); nxt = np.roll(C, -win, 0)
        a = np.linalg.norm(C - prev, axis=1)
        b = np.linalg.norm(nxt - C, axis=1)
        c = np.linalg.norm(nxt - prev, axis=1)
        area = 0.5 * np.abs((prev[:, 0]-C[:, 0])*(nxt[:, 1]-C[:, 1])
                            - (nxt[:, 0]-C[:, 0])*(prev[:, 1]-C[:, 1]))
        area = np.where(area < 1e-6, 1e-6, area)
        return ((a*b*c) / (4*area)).min()

    def _generate(self, seed, base_r, n_ctrl, min_radius, max_tries=300):
        rng = np.random.default_rng(seed)
        for _ in range(max_tries):
            angs = np.linspace(0, 2*np.pi, n_ctrl, endpoint=False)
            angs += rng.uniform(-0.08, 0.08, n_ctrl)
            radii = base_r * rng.uniform(0.90, 1.08, n_ctrl)
            P = np.stack([self.cx + radii*np.cos(angs)*1.02,
                          self.cy + radii*np.sin(angs)*0.80], axis=1)
            C = self._catmull_rom_closed(P)
            if self._min_curv_radius(C) >= min_radius:
                return C
        raise RuntimeError("Track generator: no drivable track found in max_tries")

    # ---------- queries ----------
    def is_on_track(self, x, y):
        p = np.array([x, y], dtype=float)
        ab = self._b - self._a
        t = np.einsum('ij,ij->i', p - self._a, ab) / (np.einsum('ij,ij->i', ab, ab) + 1e-9)
        t = np.clip(t, 0.0, 1.0)
        proj = self._a + t[:, None] * ab
        dmin = np.linalg.norm(proj - p, axis=1).min()
        return bool(dmin <= self.half)

    def start_pose(self):
        p0 = self.centerline[0]
        p1 = self.centerline[1]
        heading = np.degrees(np.arctan2(-(p1[1] - p0[1]), p1[0] - p0[0]))
        return float(p0[0]), float(p0[1]), float(heading)

    # ---------- render ----------
    def draw(self, screen):
        pts = [(int(px), int(py)) for px, py in self.centerline]
        pygame.draw.lines(screen, (80, 80, 80), True, pts, int(self.width))
        pygame.draw.lines(screen, (140, 140, 140), True, pts, 2)
