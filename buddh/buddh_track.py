"""Buddh International Circuit Track representation for RL Racing Agent.

Implements the exact Track public interface required by RacingEnv:
    - track.width: float
    - track.half: float
    - track.centerline: (N, 2) ndarray
    - track.is_on_track(x, y) -> bool
    - track.start_pose() -> (x, y, heading_deg)
    - track.track_heading(x, y) -> float
    - track.corner_radius(x, y) -> float
    - track.signed_distance(x, y) -> float
    - track.get_progress(x, y) -> float
    - track.draw(screen) -> None

Includes a high-performance 2D spatial grid index for O(1) local candidate
retrieval, yielding exact distance calculations in microseconds.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


class BuddhTrack:
    """Buddh International Circuit track model with exact Gymnasium compatibility."""

    def __init__(
        self,
        data_path: str = "buddh/buddh_track_data.npz",
        manifest_path: str = "buddh/buddh_manifest.json",
        cell_size: float = 150.0,
    ):
        if not os.path.exists(data_path) or not os.path.exists(manifest_path):
            # Auto-generate if missing
            from buddh.importer import BuddhTrackImporter

            importer = BuddhTrackImporter()
            importer.save_artifacts(manifest_path, data_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        data = np.load(data_path)
        self.centerline: np.ndarray = data["centerline_sim"]
        self.centerline_metric: np.ndarray = data["centerline_metric"]
        self.left_boundary: np.ndarray = data["left_boundary_sim"]
        self.right_boundary: np.ndarray = data["right_boundary_sim"]
        self._tangents: np.ndarray = data["tangents_sim"]
        self._curv_r: np.ndarray = data["curv_radii_px"]

        self.width: float = float(self.manifest["sim_track_width_px"])
        self.half: float = self.width / 2.0
        self.cell_size: float = cell_size

        # Precompute segment endpoints for distance queries
        self._a = self.centerline
        self._b = np.roll(self.centerline, -1, axis=0)
        self._ab = self._b - self._a
        self._ab2 = np.einsum("ij,ij->i", self._ab, self._ab) + 1e-9

        # Cumulative track arc lengths for fast get_progress
        self._seg_lens = np.sqrt(self._ab2)
        self._cum_lens = np.concatenate([[0.0], np.cumsum(self._seg_lens)])
        self._total_len = float(self._cum_lens[-1])

        # Precompute heading along each segment
        dx = self._ab[:, 0]
        dy = self._ab[:, 1]
        self._headings = np.degrees(np.arctan2(-dy, dx))

        # Build 2D uniform spatial grid for fast O(1) collision & nearest queries
        self._build_spatial_grid()

    def _build_spatial_grid(self) -> None:
        """Construct flat bucket grid indexing segment bounding boxes."""
        N = len(self._a)
        # Bounding box of each segment expanded by track half-width
        seg_min_x = np.minimum(self._a[:, 0], self._b[:, 0]) - self.half - 5.0
        seg_max_x = np.maximum(self._a[:, 0], self._b[:, 0]) + self.half + 5.0
        seg_min_y = np.minimum(self._a[:, 1], self._b[:, 1]) - self.half - 5.0
        seg_max_y = np.maximum(self._a[:, 1], self._b[:, 1]) + self.half + 5.0

        self._min_gx = float(np.min(seg_min_x))
        self._min_gy = float(np.min(seg_min_y))
        max_gx = float(np.max(seg_max_x))
        max_gy = float(np.max(seg_max_y))

        self._grid_w = int(np.ceil((max_gx - self._min_gx) / self.cell_size)) + 1
        self._grid_h = int(np.ceil((max_gy - self._min_gy) / self.cell_size)) + 1

        grid_buckets = [[] for _ in range(self._grid_w * self._grid_h)]

        for i in range(N):
            x0 = int((seg_min_x[i] - self._min_gx) / self.cell_size)
            x1 = int((seg_max_x[i] - self._min_gx) / self.cell_size)
            y0 = int((seg_min_y[i] - self._min_gy) / self.cell_size)
            y1 = int((seg_max_y[i] - self._min_gy) / self.cell_size)
            x0 = max(0, min(self._grid_w - 1, x0))
            x1 = max(0, min(self._grid_w - 1, x1))
            y0 = max(0, min(self._grid_h - 1, y0))
            y1 = max(0, min(self._grid_h - 1, y1))

            for gx in range(x0, x1 + 1):
                col_offset = gx * self._grid_h
                for gy in range(y0, y1 + 1):
                    grid_buckets[col_offset + gy].append(i)

        self._grid = [
            np.array(bucket, dtype=np.int32) if bucket else np.empty(0, dtype=np.int32)
            for bucket in grid_buckets
        ]

    def _get_candidate_segments(self, x: float, y: float) -> np.ndarray:
        """Fetch candidate segment indices overlapping cell at (x, y)."""
        gx = int((x - self._min_gx) / self.cell_size)
        gy = int((y - self._min_gy) / self.cell_size)
        if gx < 0 or gx >= self._grid_w or gy < 0 or gy >= self._grid_h:
            return np.empty(0, dtype=np.int32)
        return self._grid[gx * self._grid_h + gy]

    def _nearest(self, x: float, y: float) -> Tuple[int, float, np.ndarray, float]:
        """Closest point on the centerline to (x, y).

        Returns:
            index -> segment index i (from _a[i] to _b[i])
            t     -> clamped projection param in [0, 1]
            proj  -> (2,) ndarray of closest point
            dist  -> unsigned Euclidean distance from (x, y) to proj
        """
        candidates = self._get_candidate_segments(x, y)

        if len(candidates) > 0:
            sub_a = self._a[candidates]
            sub_ab = self._ab[candidates]
            sub_ab2 = self._ab2[candidates]

            px_a = x - sub_a[:, 0]
            py_a = y - sub_a[:, 1]
            dot = px_a * sub_ab[:, 0] + py_a * sub_ab[:, 1]
            t = np.clip(dot / sub_ab2, 0.0, 1.0)

            proj_x = sub_a[:, 0] + t * sub_ab[:, 0]
            proj_y = sub_a[:, 1] + t * sub_ab[:, 1]
            dx = proj_x - x
            dy = proj_y - y
            d2 = dx * dx + dy * dy
            best_local_idx = int(np.argmin(d2))
            min_d = float(np.sqrt(d2[best_local_idx]))

            # If within safe cell boundary, return candidate result
            if min_d <= self.cell_size - self.half:
                best_global_idx = int(candidates[best_local_idx])
                best_proj = np.array(
                    [proj_x[best_local_idx], proj_y[best_local_idx]], dtype=float
                )
                return best_global_idx, float(t[best_local_idx]), best_proj, min_d

        # Fallback to full global search if far away from track
        p = np.array([x, y], dtype=float)
        ab = self._ab
        t = np.clip(np.einsum("ij,ij->i", p - self._a, ab) / self._ab2, 0.0, 1.0)
        proj = self._a + t[:, None] * ab
        d = np.linalg.norm(proj - p, axis=1)
        i = int(np.argmin(d))
        return i, float(t[i]), proj[i], float(d[i])

    def is_on_track(self, x: float, y: float) -> bool:
        """Check whether (x, y) lies within the track band (distance <= half-width)."""
        candidates = self._get_candidate_segments(x, y)
        if len(candidates) == 0:
            return False

        sub_a = self._a[candidates]
        sub_ab = self._ab[candidates]
        sub_ab2 = self._ab2[candidates]

        px_a = x - sub_a[:, 0]
        py_a = y - sub_a[:, 1]
        dot = px_a * sub_ab[:, 0] + py_a * sub_ab[:, 1]
        t = np.clip(dot / sub_ab2, 0.0, 1.0)

        proj_x = sub_a[:, 0] + t * sub_ab[:, 0]
        proj_y = sub_a[:, 1] + t * sub_ab[:, 1]
        dx = proj_x - x
        dy = proj_y - y
        min_d2 = float(np.min(dx * dx + dy * dy))
        return min_d2 <= (self.half * self.half)

    def track_heading(self, x: float, y: float) -> float:
        """Road direction nearest (x, y) in degrees (y-up coordinate frame)."""
        i, _, _, _ = self._nearest(x, y)
        return float(self._headings[i])

    def corner_radius(self, x: float, y: float) -> float:
        """Road curvature radius nearest (x, y) in simulation pixels."""
        i, _, _, _ = self._nearest(x, y)
        return float(self._curv_r[i])

    def signed_distance(self, x: float, y: float) -> float:
        """Signed perpendicular distance to centerline (+ = right of center, - = left)."""
        i, _, proj, dist = self._nearest(x, y)
        dx = self._ab[i, 0]
        dy = self._ab[i, 1]
        cross = dx * (y - proj[1]) - dy * (x - proj[0])
        sign = 1.0 if cross > 0 else -1.0
        return sign * dist

    def get_progress(self, x: float, y: float) -> float:
        """Normalized progress along the closed circuit [0.0, 1.0)."""
        i, t, _, _ = self._nearest(x, y)
        dist_along = self._cum_lens[i] + t * self._seg_lens[i]
        return float((dist_along / self._total_len) % 1.0)

    def start_pose(self) -> Tuple[float, float, float]:
        """Return (x, y, heading_deg) at official Buddh start line."""
        p0 = self.centerline[0]
        p1 = self.centerline[1]
        heading = np.degrees(np.arctan2(-(p1[1] - p0[1]), p1[0] - p0[0]))
        return float(p0[0]), float(p0[1]), float(heading)

    def draw(self, screen: Any) -> None:
        """Render track band onto a Pygame display screen."""
        if pygame is None:
            raise RuntimeError("Pygame is required for desktop rendering.")
        pts = [(int(px), int(py)) for px, py in self.centerline]
        pygame.draw.lines(screen, (70, 70, 70), True, pts, int(self.width))
        pygame.draw.lines(screen, (220, 220, 220), True, pts, 2)
