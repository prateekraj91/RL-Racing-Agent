"""Read-only loader for the saved BIC evaluation run (recorded playback).

Serves the EXISTING files verbatim (no policy call, no env step, no recompute):
  - analysis/buddh/buddh_full_rollout.csv   (7,521 recorded steps)
  - buddh/buddh_track_data.npz              (exact evaluation track geometry)
  - buddh/buddh_manifest.json               (scale, turns, provenance)
Only pure geometry (boundary offsets from the stored centerline) is derived.
"""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CSV_PATH = Path("analysis/buddh/buddh_full_rollout.csv")
NPZ_PATH = Path("buddh/buddh_track_data.npz")
MANIFEST_PATH = Path("buddh/buddh_manifest.json")
HALF_WIDTH_PX = 35.0
RAY_COLS = ["ray_l90", "ray_l45", "ray_fwd", "ray_r45", "ray_r90"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _max_centerline_distance(pts: np.ndarray, cl: np.ndarray) -> dict:
    a, b = cl, np.roll(cl, -1, axis=0)
    ab = b - a
    den = (ab * ab).sum(1)
    best = np.full(len(pts), np.inf)
    for i in range(0, len(pts), 512):
        q = pts[i : i + 512, None, :]
        t = np.clip(((q - a) * ab).sum(2) / den, 0, 1)
        proj = a + ab * t[..., None]
        best[i : i + 512] = np.hypot(*(q - proj).transpose(2, 0, 1)).min(1)
    return {"max_px": float(best.max()), "mean_px": float(best.mean()),
            "argmax_step": int(best.argmax()) + 1}


GRIP_DIR = Path("runs/grip_eval")
GRIP_LABELS = {"off": "grip OFF (baseline)", "g0p8": "grip limit 0.8 G", "g1p0": "grip limit 1.0 G", "g1p2": "grip limit 1.2 G"}
STD_KW = {"width": 70, "base_r": 250, "n_ctrl": 10, "min_radius": 80, "cx": 400, "cy": 300}  # standard_medium
STD_SEED = 1001


def _grip_run_info(run: str):
    """run id like 'bic_g1p0' / 'std_off' -> (csv path, track kind, tag). KeyError if unknown."""
    kind, _, tag = run.partition("_")
    if kind not in ("bic", "std") or tag not in GRIP_LABELS:
        raise KeyError(run)
    return GRIP_DIR / f"{run}.csv", kind, tag


@lru_cache(maxsize=16)
def load_bic_recorded(run: str = "") -> dict:
    kind, tag, csv_path = "bic", None, CSV_PATH
    if run:
        csv_path, kind, tag = _grip_run_info(run)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    def col(name, cast=float):
        if cast is bool:
            return [r[name].strip().lower() in ("true", "1") for r in rows]
        return [cast(r[name]) for r in rows]
    z = np.load(NPZ_PATH)
    man = json.loads(MANIFEST_PATH.read_text())
    if kind == "std":
        from env.track import Track
        _tr = Track(seed=STD_SEED, **STD_KW)
        cl = np.asarray(_tr.centerline, float)
        _sx, _sy, _sh = _tr.start_pose()
    else:
        cl = z["centerline_sim"].astype(float)
    n = len(cl)

    # Same boundary construction as inference.simulation.extract_track_geometry
    d = np.roll(cl, -1, 0) - np.roll(cl, 1, 0)
    nrm = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    nx, ny = -d[:, 1] / nrm, d[:, 0] / nrm
    outer = np.stack([cl[:, 0] + nx * HALF_WIDTH_PX, cl[:, 1] + ny * HALF_WIDTH_PX], 1)
    inner = np.stack([cl[:, 0] - nx * HALF_WIDTH_PX, cl[:, 1] - ny * HALF_WIDTH_PX], 1)
    pad = HALF_WIDTH_PX + 30
    bounds = {"min_x": float(np.floor(cl[:, 0].min() - pad)), "max_x": float(np.ceil(cl[:, 0].max() + pad)),
              "min_y": float(np.floor(cl[:, 1].min() - pad)), "max_y": float(np.ceil(cl[:, 1].max() + pad))}

    align = _max_centerline_distance(np.array([col("car_x"), col("car_y")]).T, cl)
    r2 = lambda a: np.round(np.asarray(a, float), 2).tolist()
    cols = {c: col(c, bool if c in ("crashed", "lap_completed") else float) for c in [
        "step", "time_s", "distance_m", "car_x", "car_y", "car_angle", "speed_px_step", "speed_kmh",
        "signed_dist_px", "signed_dist_m", "heading_err_deg", "slip_deg", "steering_action",
        "throttle_action", "steer_change", "lap_progress", "crashed", "lap_completed"]}
    cols["rays"] = [col(c) for c in RAY_COLS]
    turns = [] if kind == "std" else [{"n": t["turn_number"], "radius_m": t["radius_m"], "direction": t["direction"],
              "x": t["apex_sim_x"], "y": t["apex_sim_y"]} for t in man["turns"]]
    grip = None
    if tag is not None:
        g = {"off": None, "g0p8": 0.8, "g1p0": 1.0, "g1p2": 1.2}[tag]
        grip = {"limit_g": g, "px_step2_per_g": 9.80665 / ((1 / man["world_units_per_meter"]) * 3600.0),
                "assumption": "12 m = 70 px, 60 steps/s"}
    sp = ({"x": float(_sx), "y": float(_sy), "heading": float(_sh)} if kind == "std" else
          {"x": man["start_pose"]["x_px"], "y": man["start_pose"]["y_px"], "heading": man["start_pose"]["heading_deg"]})
    return {
        "meta": {
            "label": ("Recorded: BIC" if not run else
                      f"EXPERIMENT — {GRIP_LABELS[tag]} — {'BIC' if kind == 'bic' else 'standard track seed %d' % STD_SEED}"),
            "run": run or None, "track_kind": kind, "grip": grip,
            "steps": len(rows),
            "files": {str(p): _sha(p) for p in (csv_path, NPZ_PATH, MANIFEST_PATH)},
            "px_per_m": man["world_units_per_meter"],
            "assumed_steps_per_s": 60,
            "max_steer_deg": 30.0,
            "half_width_px": HALF_WIDTH_PX,
            "length_m": man["resampled_length_m"] if kind == "bic" else float(np.hypot(*(np.roll(cl, -1, 0) - cl).T).sum()) / man["world_units_per_meter"],
            "alignment": align,
            "turns": turns,
            "start_pose": man["start_pose"] if kind == "bic" else {"x_px": sp["x"], "y_px": sp["y"], "heading_deg": sp["heading"]},
        },
        "track": {
            "centerline": r2(cl), "inner_boundary": r2(inner), "outer_boundary": r2(outer),
            "half_width": HALF_WIDTH_PX, "width": 2 * HALF_WIDTH_PX, "bounds": bounds,
            "start_pose": sp,
        },
        "cols": cols,
    }
