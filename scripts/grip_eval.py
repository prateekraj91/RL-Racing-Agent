"""Grip-limited evaluation of the FROZEN generalist-v1 policy (experiment, no retraining).

Grip is enabled ONLY through the existing Car.grip_limit flag + Car.max_grip attribute, set on the
env's car after reset(). No env/ file is changed. Outputs go to runs/grip_eval/ only.

Physical conversion (ASSUMPTIONS, same as docs/evaluation/BIC_INDEPENDENT_AUDIT.md):
    12 m = 70 px  ->  1 px = 12/70 m = 0.171429 m ;  60 steps/s  ->  1 step = 1/60 s
    a [px/step^2] -> a * 0.171429 * 3600 [m/s^2] ;  1 G = 9.80665 m/s^2
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from buddh.buddh_track import BuddhTrack  # noqa: E402
from buddh.eval_buddh import BuddhRacingEnv, verify_frozen_model  # noqa: E402
from env.environment import RacingEnv  # noqa: E402
from sac.actor import SACActor  # noqa: E402

M_PER_PX = 1.0 / 5.833333   # = 12/70 m/px; manifest value (5.833333 px/m) so the baseline CSV is byte-identical
STEPS_PER_S = 60.0
G = 9.80665
PX_STEP2_PER_G = G / (M_PER_PX * STEPS_PER_S ** 2)   # 0.015895 px/step^2 per 1 G
OUT = Path("runs/grip_eval")
STD_KW = {"width": 70, "base_r": 250, "n_ctrl": 10, "min_radius": 80, "cx": 400, "cy": 300}  # standard_medium
STD_SEEDS = [1001, 1002, 1003, 1004, 1005]
CONDITIONS = [("off", None), ("g0p8", 0.8), ("g1p0", 1.0), ("g1p2", 1.2)]
COLS = ["step", "time_s", "distance_m", "car_x", "car_y", "car_angle", "speed_px_step", "speed_kmh",
        "signed_dist_px", "signed_dist_m", "heading_err_deg", "slip_deg", "steering_action",
        "throttle_action", "steer_change", "ray_l90", "ray_l45", "ray_fwd", "ray_r45", "ray_r90",
        "lap_progress", "crashed", "lap_completed", "reward"]


def load_actor():
    verify_frozen_model()
    a = SACActor(obs_dim=9, action_dim=2)
    a.load_state_dict(torch.load("models/generalist_v1.pth", map_location="cpu"))
    a.eval()
    return a


def run(actor, env, length_m, max_g, max_steps, reset_kwargs=None):
    obs, _ = env.reset(**(reset_kwargs or {}))
    if max_g is not None:
        env.car.grip_limit = True
        env.car.max_grip = max_g * PX_STEP2_PER_G
    else:
        env.car.grip_limit = False
    rows, prev_steer, prev_angle = [], 0.0, env.car.angle
    extra = {"a_lat_g": [], "a_dem_g": []}
    for s in range(max_steps):
        with torch.no_grad():
            mean, _ = actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            act = torch.tanh(mean).numpy()[0]
        sa, ta = float(act[0]), float(act[1])
        sd = abs(sa - prev_steer); prev_steer = sa
        obs, r, term, trunc, info = env.step(act)
        spd = float(info["speed"])
        # achieved lateral accel = v * yaw-rate ; demanded = v^2 * tan(steer)/wheelbase
        dpsi = math.radians((env.car.angle - prev_angle + 180) % 360 - 180); prev_angle = env.car.angle
        extra["a_lat_g"].append(spd * abs(dpsi) / PX_STEP2_PER_G)
        extra["a_dem_g"].append(spd ** 2 * abs(math.tan(math.radians(sa * 30.0))) / env.car.wheelbase / PX_STEP2_PER_G)
        lp = float(info["lap_progress"])
        rows.append({
            "step": s + 1, "time_s": round((s + 1) / STEPS_PER_S, 3), "distance_m": round(lp * length_m, 2),
            "car_x": round(float(env.car.x), 2), "car_y": round(float(env.car.y), 2),
            "car_angle": round(float(env.car.angle), 2), "speed_px_step": round(spd, 4),
            "speed_kmh": round(spd * STEPS_PER_S * M_PER_PX * 3.6, 2),
            "signed_dist_px": round(float(info["signed_dist"]), 3),
            "signed_dist_m": round(float(info["signed_dist"]) * M_PER_PX, 3),
            "heading_err_deg": round(float(info["heading_err"]), 3), "slip_deg": round(float(info["slip"]), 3),
            "steering_action": round(sa, 4), "throttle_action": round(ta, 4), "steer_change": round(sd, 4),
            "ray_l90": round(float(info["rays"][0]), 2), "ray_l45": round(float(info["rays"][1]), 2),
            "ray_fwd": round(float(info["rays"][2]), 2), "ray_r45": round(float(info["rays"][3]), 2),
            "ray_r90": round(float(info["rays"][4]), 2), "lap_progress": round(lp, 5),
            "crashed": bool(info["crashed"]), "lap_completed": bool(info["lap_completed"]),
            "reward": round(float(r), 4)})
        if info["crashed"] or info["lap_completed"]:
            break
    return rows, extra, env.car


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


def nearest_corner(d_m, turns, length_m):
    best = min(turns, key=lambda t: min(abs(d_m - t["d"]), length_m - abs(d_m - t["d"])))
    return best["name"], round(best["d"] - d_m, 1)


def summarize(rows, extra, car, turns, length_m, label_corners):
    n = len(rows); last = rows[-1]
    spd = np.array([r["speed_px_step"] for r in rows]); dist = np.array([r["distance_m"] for r in rows])
    out = {"steps": n, "laps_completed": int(last["lap_completed"]), "crashed": bool(last["crashed"]),
           "final_lap_progress": last["lap_progress"],
           "braking_pct": round(100.0 * sum(r["throttle_action"] < 0 for r in rows) / n, 2),
           "max_lat_g_achieved": round(float(max(extra["a_lat_g"])), 2),
           "max_lat_g_demanded_by_steering": round(float(max(extra["a_dem_g"])), 2),
           "mean_speed_px_step": round(float(spd.mean()), 3),
           "grip_events": car.grip_events, "grip_yaw_lost_sum": round(car.grip_yaw_lost, 1)}
    if last["crashed"]:
        name, off = nearest_corner(last["distance_m"], turns, length_m) if label_corners else (None, None)
        out["offtrack"] = {"distance_m": last["distance_m"], "step": n, "x": last["car_x"], "y": last["car_y"],
                           "nearest_corner": name, "corner_minus_distance_m": off,
                           "speed_px_step": last["speed_px_step"]}
    else:
        out["offtrack"] = None
    # speed ENTERING a corner = speed 50 m before the listed apex distance
    for t in turns:
        if t["name"] in ("T1", "T6", "T8"):
            tgt = t["d"] - 50.0
            out[f"entry_speed_{t['name']}"] = (round(float(np.interp(tgt, dist, spd)), 3)
                                               if dist[-1] >= tgt else None)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    actor = load_actor()
    man = json.load(open("buddh/buddh_manifest.json"))
    bic_len = man["resampled_length_m"]
    bic_turns = [{"name": f"T{t['turn_number']}", "d": t["distance_along_track_m"]} for t in man["turns"]]
    results = {"assumptions": {"m_per_px": M_PER_PX, "steps_per_s": STEPS_PER_S, "px_step2_per_G": PX_STEP2_PER_G,
                               "default_max_grip_px_step2": 0.09, "default_max_grip_G": 0.09 / PX_STEP2_PER_G},
               "runs": {}}
    track = BuddhTrack()
    for tag, g in CONDITIONS:
        t0 = time.time()
        env = BuddhRacingEnv(max_steps=15000, track=track, grip_limit=g is not None)
        rows, extra, car = run(actor, env, bic_len, g, 15000)
        write_csv(OUT / f"bic_{tag}.csv", rows)
        results["runs"][f"bic_{tag}"] = {"track": "BIC", "grip_G": g, **summarize(rows, extra, car, bic_turns, bic_len, True)}
        np.save(OUT / f"bic_{tag}_alat.npy", np.array(extra["a_lat_g"], dtype=np.float32))
        print("bic", tag, results["runs"][f"bic_{tag}"]["steps"], f"{time.time()-t0:.0f}s", flush=True)
    for seed in STD_SEEDS:
        for tag, g in CONDITIONS:
            env = RacingEnv(max_steps=3000, track_kwargs=STD_KW, grip_limit=g is not None)
            rows, extra, car = run(actor, env, 0.0, g, 3000, {"options": {"track_seed": seed}})
            cl = env.track.centerline
            length_px = float(np.sum(np.hypot(*(np.roll(cl, -1, 0) - cl).T)))
            for r in rows:
                r["distance_m"] = round(r["lap_progress"] * length_px * M_PER_PX, 2)
            if seed == STD_SEEDS[0]:
                write_csv(OUT / f"std_{tag}.csv", rows)
                np.save(OUT / f"std_{tag}_alat.npy", np.array(extra["a_lat_g"], dtype=np.float32))
            s = summarize(rows, extra, car, [], length_px * M_PER_PX, False)
            s["lap_length_px"] = round(length_px, 1)
            if s["offtrack"]:
                i = int(np.argmin(np.hypot(cl[:, 0] - rows[-1]["car_x"], cl[:, 1] - rows[-1]["car_y"])))
                s["offtrack"]["local_curv_radius_px"] = round(float(env.track._curv_r[i]), 1)
                s["offtrack"]["distance_px"] = round(rows[-1]["lap_progress"] * length_px, 1)
            results["runs"][f"std_{tag}_seed{seed}"] = {"track": f"standard_medium seed {seed}", "grip_G": g, **s}
        print("std seed", seed, flush=True)
    json.dump(results, open(OUT / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
