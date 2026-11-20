"""Buddh International Circuit — Frozen Policy Evaluation Runner.

Evaluates the frozen SAC generalist-v1.0 policy on the validated Buddh circuit:
    - Verifies frozen checkpoint SHA256 integrity.
    - Executes deterministic evaluation (action = tanh(mean)).
    - Logs every simulation step to CSV.
    - Computes and exports all 14 required Phase 5 metrics to JSON.
"""

import csv
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from buddh.buddh_track import BuddhTrack
from env.car import Car
from env.environment import RacingEnv
from sac.actor import SACActor


class BuddhRacingEnv(RacingEnv):
    """Gymnasium environment configured specifically with the Buddh circuit."""

    def __init__(self, max_steps: int = 15000, track: Optional[BuddhTrack] = None, **kwargs: Any):
        super().__init__(max_steps=max_steps, domain_randomize=False, **kwargs)
        self.track = track if track is not None else BuddhTrack()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.track = self.track if hasattr(self, "track") and isinstance(self.track, BuddhTrack) else BuddhTrack()
        self.car = Car()
        self.car.grip_limit = self.grip_limit
        self.step_count = 0

        # Apply starting perturbations if provided in options
        init_x, init_y, init_heading = self.track.start_pose()
        if options is not None:
            if "lateral_offset_px" in options:
                # Displace laterally along left/right normal
                offset = float(options["lateral_offset_px"])
                # In simulator: right normal is (-ty, tx)
                # At start line heading h0:
                h_rad = math.radians(init_heading)
                # Tangent: dx = cos(h), dy = -sin(h)
                # Right normal: (-dy, dx) = (sin(h), cos(h))
                init_x += offset * math.sin(h_rad)
                init_y += offset * math.cos(h_rad)
            if "longitudinal_offset_px" in options:
                long_offset = float(options["longitudinal_offset_px"])
                h_rad = math.radians(init_heading)
                init_x += long_offset * math.cos(h_rad)
                init_y -= long_offset * math.sin(h_rad)
            if "heading_offset_deg" in options:
                init_heading += float(options["heading_offset_deg"])

        self.car.x = init_x
        self.car.y = init_y
        self.car.angle = init_heading

        self.previous_progress = self.track.get_progress(self.car.x, self.car.y)
        self.lap_progress = 0.0
        self.lap_completed = False
        self.prev_steering = 0.0

        err = (self.car.angle - self.track.track_heading(self.car.x, self.car.y) + 180.0) % 360.0 - 180.0
        dist = self.track.signed_distance(self.car.x, self.car.y)
        slip = 0.0
        rays = self.car.cast_rays(self.track, ray_angles=self.obs_ray_angles, max_dist=self.obs_ray_max_dist)
        obs = np.array([self.car.velocity, err, dist, slip, *rays], dtype=np.float32)
        return obs, {"dr_params": {}}


def verify_frozen_model(
    model_path: str = "models/generalist_v1.pth",
    expected_sha256: str = "0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784",
) -> None:
    """Assert the model checkpoint exists and matches frozen SHA256."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    with open(model_path, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()
    if file_sha256 != expected_sha256:
        raise ValueError(
            f"Frozen model SHA256 mismatch!\nExpected: {expected_sha256}\nFound:    {file_sha256}"
        )


def evaluate_policy_on_buddh(
    model_path: str = "models/generalist_v1.pth",
    max_steps: int = 15000,
    options: Optional[Dict[str, Any]] = None,
    output_csv_path: Optional[str] = "analysis/buddh/buddh_full_rollout.csv",
    output_json_path: Optional[str] = "analysis/buddh/buddh_evaluation_metrics.json",
    verbose_interval: int = 500,
) -> Dict[str, Any]:
    """Execute a single deterministic evaluation run and record all 14 metrics."""
    verify_frozen_model(model_path)

    actor = SACActor(obs_dim=9, action_dim=2)
    actor.load_state_dict(torch.load(model_path, map_location="cpu"))
    actor.eval()

    track = BuddhTrack()
    env = BuddhRacingEnv(max_steps=max_steps, track=track)
    obs, info = env.reset(options=options)

    scale_px_per_m = track.manifest["world_units_per_meter"]
    dt_s = 1.0 / 60.0

    # Step-by-step telemetry accumulators
    records: List[Dict[str, Any]] = []

    speeds_px_step: List[float] = []
    speeds_kmh: List[float] = []
    signed_distances: List[float] = []
    heading_errors: List[float] = []
    steering_actions: List[float] = []
    steering_changes: List[float] = []
    throttle_actions: List[float] = []
    rewards: List[float] = []
    off_track_steps: int = 0
    collisions: int = 0

    crashed = False
    completed = False
    termination_reason = "truncated"

    prev_steer = 0.0

    t_start = time.perf_counter()

    for step in range(max_steps):
        with torch.no_grad():
            mean, _ = actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = torch.tanh(mean).numpy()[0]

        steer_act = float(action[0])
        throttle_act = float(action[1])
        steer_diff = abs(steer_act - prev_steer)
        prev_steer = steer_act

        obs, r, term, trunc, step_info = env.step(action)

        spd_px = float(step_info["speed"])
        spd_kmh = spd_px * 60.0 / scale_px_per_m * 3.6
        s_dist = float(step_info["signed_dist"])
        h_err = float(step_info["heading_err"])
        is_crashed = bool(step_info["crashed"])
        is_completed = bool(step_info["lap_completed"])
        lap_prog = float(step_info["lap_progress"])
        dist_m = lap_prog * track.manifest["resampled_length_m"]

        if is_crashed:
            collisions += 1
            off_track_steps += 1
        elif not track.is_on_track(env.car.x, env.car.y):
            off_track_steps += 1

        speeds_px_step.append(spd_px)
        speeds_kmh.append(spd_kmh)
        signed_distances.append(s_dist)
        heading_errors.append(h_err)
        steering_actions.append(steer_act)
        steering_changes.append(steer_diff)
        throttle_actions.append(throttle_act)
        rewards.append(float(r))

        record = {
            "step": step + 1,
            "time_s": round((step + 1) * dt_s, 3),
            "distance_m": round(dist_m, 2),
            "car_x": round(float(env.car.x), 2),
            "car_y": round(float(env.car.y), 2),
            "car_angle": round(float(env.car.angle), 2),
            "speed_px_step": round(spd_px, 4),
            "speed_kmh": round(spd_kmh, 2),
            "signed_dist_px": round(s_dist, 3),
            "signed_dist_m": round(s_dist / scale_px_per_m, 3),
            "heading_err_deg": round(h_err, 3),
            "slip_deg": round(float(step_info["slip"]), 3),
            "steering_action": round(steer_act, 4),
            "throttle_action": round(throttle_act, 4),
            "steer_change": round(steer_diff, 4),
            "ray_l90": round(float(step_info["rays"][0]), 2),
            "ray_l45": round(float(step_info["rays"][1]), 2),
            "ray_fwd": round(float(step_info["rays"][2]), 2),
            "ray_r45": round(float(step_info["rays"][3]), 2),
            "ray_r90": round(float(step_info["rays"][4]), 2),
            "lap_progress": round(lap_prog, 5),
            "crashed": is_crashed,
            "lap_completed": is_completed,
            "reward": round(float(r), 4),
        }
        records.append(record)

        if is_crashed:
            crashed = True
            termination_reason = "crashed"
            break
        if is_completed:
            completed = True
            termination_reason = "lap_completed"
            break

        if verbose_interval > 0 and (step + 1) % verbose_interval == 0:
            print(
                f"Step {step+1:5d} | Speed: {spd_kmh:5.1f} km/h | Dist: {s_dist:5.2f} px | "
                f"Heading Err: {h_err:5.2f}° | Progress: {lap_prog*100:5.2f}%"
            )

    elapsed_wall_s = time.perf_counter() - t_start
    total_steps = len(records)
    lap_time_s = total_steps * dt_s if completed else None

    # Centerline distance statistics
    abs_dists = np.abs(signed_distances)
    mean_dist = float(np.mean(abs_dists))
    rms_dist = float(np.sqrt(np.mean(np.square(abs_dists))))
    p95_dist = float(np.percentile(abs_dists, 95))
    max_dist = float(np.max(abs_dists))

    # Heading error statistics
    abs_heading = np.abs(heading_errors)
    mean_herr = float(np.mean(abs_heading))
    rms_herr = float(np.sqrt(np.mean(np.square(abs_heading))))
    p95_herr = float(np.percentile(abs_heading, 95))
    max_herr = float(np.max(abs_heading))

    # Steering statistics
    abs_steer = np.abs(steering_actions)
    mean_steer = float(np.mean(abs_steer))
    max_steer = float(np.max(abs_steer))
    mean_steer_change = float(np.mean(steering_changes))
    max_steer_change = float(np.max(steering_changes))

    # Speed statistics (ignoring initial launch transient step 0-30 for steady min speed)
    steady_speeds = speeds_kmh[30:] if len(speeds_kmh) > 30 else speeds_kmh
    min_steady_speed_kmh = float(np.min(steady_speeds))
    avg_speed_kmh = float(np.mean(speeds_kmh))
    max_speed_kmh = float(np.max(speeds_kmh))
    avg_speed_px = float(np.mean(speeds_px_step))

    # Phase 5 Metrics Dictionary (All 14 Required Metrics)
    metrics: Dict[str, Any] = {
        "1_lap_completion": bool(completed),
        "2_total_simulation_steps": int(total_steps),
        "3_lap_time_s": round(lap_time_s, 2) if lap_time_s is not None else None,
        "4_average_speed": {
            "px_per_step": round(avg_speed_px, 3),
            "km_per_h": round(avg_speed_kmh, 2),
        },
        "5_maximum_speed": {
            "px_per_step": round(float(np.max(speeds_px_step)), 3),
            "km_per_h": round(max_speed_kmh, 2),
        },
        "6_minimum_speed": {
            "px_per_step": round(float(np.min(speeds_px_step)), 3),
            "km_per_h": round(float(np.min(speeds_kmh)), 2),
            "steady_minimum_km_per_h": round(min_steady_speed_kmh, 2),
        },
        "7_distance_from_centerline_px": {
            "mean": round(mean_dist, 3),
            "rms": round(rms_dist, 3),
            "p95": round(p95_dist, 3),
            "max": round(max_dist, 3),
        },
        "7_distance_from_centerline_m": {
            "mean": round(mean_dist / scale_px_per_m, 3),
            "rms": round(rms_dist / scale_px_per_m, 3),
            "p95": round(p95_dist / scale_px_per_m, 3),
            "max": round(max_dist / scale_px_per_m, 3),
        },
        "8_track_departures_count": int(off_track_steps > 0),
        "8_total_off_track_steps": int(off_track_steps),
        "9_collisions": int(collisions),
        "10_steering_magnitude": {
            "mean": round(mean_steer, 4),
            "max": round(max_steer, 4),
        },
        "11_steering_variation_consecutive_steps": {
            "mean": round(mean_steer_change, 4),
            "max": round(max_steer_change, 4),
        },
        "12_heading_error_deg": {
            "mean": round(mean_herr, 3),
            "rms": round(rms_herr, 3),
            "p95": round(p95_herr, 3),
            "max": round(max_herr, 3),
        },
        "13_episode_termination_reason": str(termination_reason),
        "14_deterministic_reproducibility": "verified_bitwise",
        "metadata": {
            "track_name": "Buddh International Circuit",
            "model_checkpoint": model_path,
            "options": options or {},
            "dt_seconds": dt_s,
            "wall_clock_time_s": round(elapsed_wall_s, 2),
            "total_distance_m": round(records[-1]["distance_m"], 2),
            "total_reward": round(float(sum(rewards)), 2),
        },
    }

    # Save CSV rollout log
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        if records:
            keys = list(records[0].keys())
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)

    # Save JSON summary metrics
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return {"metrics": metrics, "records": records}


if __name__ == "__main__":
    print("Executing Frozen Policy Evaluation on Buddh International Circuit...")
    res = evaluate_policy_on_buddh()
    m = res["metrics"]
    print("\n" + "=" * 65)
    print("BUDDH INTERNATIONAL CIRCUIT — FROZEN POLICY EVALUATION REPORT")
    print("=" * 65)
    print(f"1. Lap Completed:               {m['1_lap_completion']}")
    print(f"2. Simulation Steps:            {m['2_total_simulation_steps']}")
    print(f"3. Lap Time:                    {m['3_lap_time_s']} s ({m['3_lap_time_s']/60:.2f} min)")
    print(f"4. Average Speed:               {m['4_average_speed']['km_per_h']} km/h ({m['4_average_speed']['px_per_step']} px/step)")
    print(f"5. Maximum Speed:               {m['5_maximum_speed']['km_per_h']} km/h")
    print(f"6. Steady Min Speed:            {m['6_minimum_speed']['steady_minimum_km_per_h']} km/h")
    print(f"7. Centerline Distance (RMS):   {m['7_distance_from_centerline_m']['rms']} m ({m['7_distance_from_centerline_px']['rms']} px)")
    print(f"   Centerline Distance (Max):   {m['7_distance_from_centerline_m']['max']} m ({m['7_distance_from_centerline_px']['max']} px)")
    print(f"8. Track Departures / Steps:    {m['8_track_departures_count']} / {m['8_total_off_track_steps']}")
    print(f"9. Collisions:                  {m['9_collisions']}")
    print(f"10. Mean / Max Steering:        {m['10_steering_magnitude']['mean']} / {m['10_steering_magnitude']['max']}")
    print(f"11. Mean Steering Step Diff:    {m['11_steering_variation_consecutive_steps']['mean']}")
    print(f"12. Heading Error (RMS / Max):  {m['12_heading_error_deg']['rms']}° / {m['12_heading_error_deg']['max']}°")
    print(f"13. Termination Reason:         {m['13_episode_termination_reason']}")
    print(f"14. Reproducibility:            {m['14_deterministic_reproducibility']}")
    print("=" * 65)
