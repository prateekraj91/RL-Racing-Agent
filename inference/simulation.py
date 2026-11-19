"""Stateful simulation manager and telemetry analysis for interactive demo.

Integrates RacingEnv with the frozen generalist-v1.0 SACActor, provides
real-time track geometry extraction, live ray sensor calculation,
online input-gradient saliency, and dual-mode grounded commentary.
"""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from env.environment import RacingEnv
from env.track import Track
from sac.actor import SACActor

logger = logging.getLogger("inference.simulation")

ROOT_DIR = Path(__file__).resolve().parent.parent

# Feature mapping contract
FEATURE_NAMES = [
    "velocity",
    "heading_error",
    "signed_distance",
    "slip",
    "ray_minus_90",
    "ray_minus_45",
    "ray_0",
    "ray_plus_45",
    "ray_plus_90",
]
RAY_INDICES = [4, 5, 6, 7, 8]
RAY_NAMES = ["ray_minus_90", "ray_minus_45", "ray_0", "ray_plus_45", "ray_plus_90"]
RAY_ANGLES = [-90.0, -45.0, 0.0, 45.0, 90.0]

# Supported Track Presets
TRACK_PRESETS: Dict[str, Dict[str, Any]] = {
    "standard_medium": {
        "id": "standard_medium",
        "name": "Standard Medium Circuit",
        "description": "Standard benchmark track with balanced corners and straights (width 70 px).",
        "track_kwargs": {
            "width": 70,
            "base_r": 250,
            "n_ctrl": 10,
            "min_radius": 80,
            "cx": 400,
            "cy": 300,
        },
        "default_seed": 1001,
        "recommended_seeds": [1001, 1002, 1003, 1004, 1005, 101],
        "max_steps": 700,
    },
    "narrow_corridor": {
        "id": "narrow_corridor",
        "name": "Narrow Precision Corridor",
        "description": "Challenging narrow track (width 40 px) requiring precise line and lateral discipline.",
        "track_kwargs": {
            "width": 40,
            "base_r": 250,
            "n_ctrl": 10,
            "min_radius": 80,
            "cx": 400,
            "cy": 300,
        },
        "default_seed": 1001,
        "recommended_seeds": [1001, 1002, 1005],
        "max_steps": 700,
    },
    "hairpin_challenge": {
        "id": "hairpin_challenge",
        "name": "Hairpin Switchback",
        "description": "Features a sharp pinch hairpin turn (hairpin_factor 0.35) demanding hard braking.",
        "track_kwargs": {
            "width": 70,
            "base_r": 250,
            "n_ctrl": 10,
            "min_radius": 80,
            "cx": 400,
            "cy": 300,
            "hairpin": True,
            "hairpin_factor": 0.35,
        },
        "default_seed": 1001,
        "recommended_seeds": [1001, 1002, 1003],
        "max_steps": 700,
    },
    "wide_oval": {
        "id": "wide_oval",
        "name": "Wide Flowing Circuit",
        "description": "Wide, flowing track (width 100 px) for high-speed cruising and smooth sweeps.",
        "track_kwargs": {
            "width": 100,
            "base_r": 250,
            "n_ctrl": 10,
            "min_radius": 80,
            "cx": 400,
            "cy": 300,
        },
        "default_seed": 101,
        "recommended_seeds": [101, 1001, 1004],
        "max_steps": 700,
    },
    "default_circuit": {
        "id": "default_circuit",
        "name": "Default Environment Circuit",
        "description": "Baseline RacingEnv track configuration (base_r 210 px, min radius 70 px).",
        "track_kwargs": {},
        "default_seed": 42,
        "recommended_seeds": [42, 101, 1001],
        "max_steps": 700,
    },
}


def extract_track_geometry(track: Track) -> Dict[str, Any]:
    """Extract full centerline and inner/outer road boundaries for canvas rendering."""
    centerline = track.centerline.tolist()
    N = len(track.centerline)
    half = float(track.half)

    # Compute continuous normal vectors at each vertex for inner/outer curbs
    inner_boundary = []
    outer_boundary = []

    C = track.centerline
    for i in range(N):
        prev_p = C[(i - 1) % N]
        next_p = C[(i + 1) % N]
        dx = float(next_p[0] - prev_p[0])
        dy = float(next_p[1] - prev_p[1])
        norm = math.hypot(dx, dy) + 1e-9
        # Perpendicular normal: (-dy/norm, dx/norm)
        nx = -dy / norm
        ny = dx / norm

        p = C[i]
        # Left boundary (+ normal * half)
        outer_boundary.append([round(float(p[0] + nx * half), 2), round(float(p[1] + ny * half), 2)])
        # Right boundary (- normal * half)
        inner_boundary.append([round(float(p[0] - nx * half), 2), round(float(p[1] - ny * half), 2)])

    all_x = [p[0] for p in centerline]
    all_y = [p[1] for p in centerline]

    start_x, start_y, start_heading = track.start_pose()

    return {
        "width": float(track.width),
        "half_width": half,
        "centerline": centerline,
        "inner_boundary": inner_boundary,
        "outer_boundary": outer_boundary,
        "start_pose": {
            "x": round(start_x, 2),
            "y": round(start_y, 2),
            "heading": round(start_heading, 2),
        },
        "bounds": {
            "min_x": math.floor(min(all_x) - half - 30),
            "max_x": math.ceil(max(all_x) + half + 30),
            "min_y": math.floor(min(all_y) - half - 30),
            "max_y": math.ceil(max(all_y) + half + 30),
        },
    }


def compute_ray_geometry(
    car_x: float,
    car_y: float,
    car_angle: float,
    ray_angles: Sequence[float],
    ray_distances: Sequence[float],
) -> List[Dict[str, Any]]:
    """Compute exact ray endpoints from car state and measured sensor distances."""
    rays = []
    for angle_offset, dist in zip(ray_angles, ray_distances):
        rad = math.radians(car_angle + angle_offset)
        dx = math.cos(rad)
        dy = -math.sin(rad)  # Screen coordinates: y is down
        end_x = car_x + dx * dist
        end_y = car_y + dy * dist

        rays.append({
            "angle_offset": float(angle_offset),
            "distance": round(float(dist), 2),
            "start": [round(car_x, 2), round(car_y, 2)],
            "end": [round(end_x, 2), round(end_y, 2)],
            "is_close": bool(dist < 50.0),
        })
    return rays


def compute_online_saliency(
    actor: SACActor,
    observation: Sequence[float],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Calculate gradient-based input saliency for observation using frozen actor.

    Gradients are computed strictly with respect to observation input tensor.
    Actor parameters remain completely frozen with requires_grad=False.
    """
    obs_tensor = torch.tensor(
        observation, dtype=torch.float32, device=device, requires_grad=True
    ).unsqueeze(0)

    # Forward pass
    mean, _ = actor(obs_tensor)
    action = torch.tanh(mean)

    steer_action = action[0, 0]
    long_action = action[0, 1]

    # Compute gradients w.r.t observation input
    grad_steer = (
        torch.autograd.grad(
            outputs=steer_action,
            inputs=obs_tensor,
            retain_graph=True,
            create_graph=False,
        )[0]
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    grad_long = (
        torch.autograd.grad(
            outputs=long_action,
            inputs=obs_tensor,
            retain_graph=False,
            create_graph=False,
        )[0]
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    saliency_steer = np.abs(grad_steer)
    saliency_long = np.abs(grad_long)

    feature_saliency_steer = {
        name: round(float(saliency_steer[i]), 5) for i, name in enumerate(FEATURE_NAMES)
    }
    feature_saliency_long = {
        name: round(float(saliency_long[i]), 5) for i, name in enumerate(FEATURE_NAMES)
    }

    ray_saliency_steer = {
        name: round(float(saliency_steer[idx]), 5)
        for idx, name in zip(RAY_INDICES, RAY_NAMES)
    }
    ray_saliency_long = {
        name: round(float(saliency_long[idx]), 5)
        for idx, name in zip(RAY_INDICES, RAY_NAMES)
    }

    top_steering_ray = max(ray_saliency_steer, key=ray_saliency_steer.get)
    top_longitudinal_ray = max(ray_saliency_long, key=ray_saliency_long.get)

    return {
        "feature_saliency_steer": feature_saliency_steer,
        "feature_saliency_long": feature_saliency_long,
        "ray_saliency_steer": ray_saliency_steer,
        "ray_saliency_long": ray_saliency_long,
        "top_steering_ray": top_steering_ray,
        "top_longitudinal_ray": top_longitudinal_ray,
        "computed_at": time.time(),
        "disclaimer": "Input-gradient sensitivity |d(action)/d(obs)|. Not human visual attention or gaze.",
    }


class SimulationSession:
    """Manages active interactive simulation state, telemetry log, and commentary."""

    def __init__(self):
        self.env: Optional[RacingEnv] = None
        self.preset_id: str = "standard_medium"
        self.preset_info: Dict[str, Any] = TRACK_PRESETS["standard_medium"]
        self.track_seed: int = 1001
        self.is_running: bool = False
        self.is_crashed: bool = False
        self.is_completed: bool = False

        # Live trajectory tracking
        self.trajectory: List[List[float]] = []
        self.history: List[Dict[str, Any]] = []
        self.commentary_log: List[Dict[str, Any]] = []

        # Current vehicle state
        self.current_obs: Optional[List[float]] = None
        self.current_action: List[float] = [0.0, 0.0]
        self.current_saliency: Optional[Dict[str, Any]] = None
        self.track_geometry: Optional[Dict[str, Any]] = None

        # Gemini call rate-limiter (max 5 calls per lap)
        self.gemini_calls_count: int = 0
        self.last_commentary_step: int = -50

    def start(
        self,
        preset_id: str = "standard_medium",
        track_seed: Optional[int] = None,
        steer_penalty_coeff: float = 0.01,
    ) -> Dict[str, Any]:
        """Initialize and start a fresh simulation session."""
        if preset_id not in TRACK_PRESETS:
            preset_id = "standard_medium"

        self.preset_id = preset_id
        self.preset_info = TRACK_PRESETS[preset_id]
        self.track_seed = (
            int(track_seed)
            if track_seed is not None
            else self.preset_info["default_seed"]
        )

        # Build fresh RacingEnv instance
        track_kwargs = dict(self.preset_info.get("track_kwargs", {}))
        max_steps = self.preset_info.get("max_steps", 700)

        self.env = RacingEnv(
            max_steps=max_steps,
            track_kwargs=track_kwargs,
            grip_limit=False,
            domain_randomize=False,
            obs_ray_angles=(-90, -45, 0, 45, 90),
            obs_ray_max_dist=200,
            steer_penalty_coeff=steer_penalty_coeff,
        )

        obs, _ = self.env.reset(options={"track_seed": self.track_seed})
        self.current_obs = [float(x) for x in obs]
        self.current_action = [0.0, 0.0]
        self.is_running = True
        self.is_crashed = False
        self.is_completed = False

        self.trajectory = [[round(self.env.car.x, 2), round(self.env.car.y, 2)]]
        self.history = []
        self.commentary_log = []
        self.gemini_calls_count = 0
        self.last_commentary_step = -50

        # Cache geometry for frontend canvas rendering
        self.track_geometry = extract_track_geometry(self.env.track)

        # Initial commentary
        self._add_commentary(
            step=0,
            event_type="session_start",
            text=(
                f"Session started on {self.preset_info['name']} (seed {self.track_seed}). "
                f"Vehicle positioned on start line at ({self.env.car.x:.1f}, {self.env.car.y:.1f})."
            ),
            source="offline_deterministic",
        )

        return self.get_state()

    def reset(self, actor: Optional[SACActor] = None) -> Dict[str, Any]:
        """Reset the current simulation with the same preset and seed."""
        return self.start(preset_id=self.preset_id, track_seed=self.track_seed)

    def step(
        self,
        actor: SACActor,
        steps: int = 1,
        calculate_saliency: bool = False,
        commentary_mode: Literal["offline", "gemini"] = "offline",
        gemini_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Advance simulation by a bounded number of steps using frozen SACActor."""
        if self.env is None:
            self.start()

        if self.is_crashed or self.is_completed:
            return self.get_state()

        step_limit = max(1, min(int(steps), 20))

        for _ in range(step_limit):
            if self.is_crashed or self.is_completed:
                break

            # 1. Evaluate frozen actor deterministically
            obs_tensor = torch.tensor(
                self.current_obs, dtype=torch.float32
            ).unsqueeze(0)
            with torch.inference_mode():
                mean, _ = actor(obs_tensor)
                action_tensor = torch.tanh(mean)

            steer_cmd = float(action_tensor[0, 0].item())
            throttle_cmd = float(action_tensor[0, 1].item())
            self.current_action = [round(steer_cmd, 6), round(throttle_cmd, 6)]

            # 2. Advance actual environment physics
            action_np = np.array([steer_cmd, throttle_cmd], dtype=np.float32)
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            self.current_obs = [float(x) for x in next_obs]

            # 3. Update trajectory
            car_x = round(float(self.env.car.x), 2)
            car_y = round(float(self.env.car.y), 2)
            self.trajectory.append([car_x, car_y])

            # 4. Status flags
            crashed = bool(info.get("crashed", False) or not self.env.track.is_on_track(car_x, car_y))
            lap_completed = bool(info.get("lap_completed", False) or self.env.lap_progress >= 1.0)

            if crashed:
                self.is_crashed = True
                self.is_running = False
            elif lap_completed:
                self.is_completed = True
                self.is_running = False
            elif truncated:
                self.is_running = False

            # Record step telemetry record
            telemetry_snapshot = self._build_telemetry_snapshot(info, reward)
            self.history.append(telemetry_snapshot)
            if len(self.history) > 300:
                self.history.pop(0)

            # 5. Driving event detection and commentary triggering
            self._evaluate_driving_events(
                telemetry_snapshot=telemetry_snapshot,
                commentary_mode=commentary_mode,
                gemini_api_key=gemini_api_key,
                actor=actor,
            )

        # Compute live saliency if requested
        if calculate_saliency or (self.env.step_count % 5 == 0):
            try:
                self.current_saliency = compute_online_saliency(
                    actor=actor, observation=self.current_obs
                )
            except Exception as e:
                logger.warning("Failed to compute online saliency: %s", e)

        return self.get_state()

    def _build_telemetry_snapshot(self, info: Dict[str, Any], reward: float) -> Dict[str, Any]:
        """Capture structured telemetry metrics for current step."""
        car = self.env.car
        return {
            "step": self.env.step_count,
            "x": round(float(car.x), 2),
            "y": round(float(car.y), 2),
            "heading": round(float(car.angle), 2),
            "velocity": round(float(car.velocity), 3),
            "steering_action": round(float(self.current_action[0]), 4),
            "steering_angle_deg": round(float(car.steering), 2),
            "throttle_action": round(float(self.current_action[1]), 4),
            "heading_error": round(float(info.get("heading_err", 0.0)), 2),
            "signed_distance": round(float(info.get("signed_dist", 0.0)), 2),
            "slip_angle": round(float(info.get("slip", 0.0)), 2),
            "lap_progress": round(float(self.env.lap_progress), 4),
            "reward": round(float(reward), 4),
            "crashed": self.is_crashed,
            "lap_completed": self.is_completed,
            "rays": [round(float(r), 2) for r in info.get("rays", [])],
        }

    def _evaluate_driving_events(
        self,
        telemetry_snapshot: Dict[str, Any],
        commentary_mode: str,
        gemini_api_key: Optional[str],
        actor: SACActor,
    ) -> None:
        """Detect notable driving milestones and add grounded commentary."""
        step = telemetry_snapshot["step"]
        if step - self.last_commentary_step < 12 and not (self.is_crashed or self.is_completed):
            return

        vel = telemetry_snapshot["velocity"]
        rays = telemetry_snapshot["rays"]
        steer_deg = telemetry_snapshot["steering_angle_deg"]
        steer_act = telemetry_snapshot["steering_action"]
        throttle = telemetry_snapshot["throttle_action"]
        dist = telemetry_snapshot["signed_distance"]
        heading_err = telemetry_snapshot["heading_error"]
        slip = telemetry_snapshot["slip_angle"]
        fwd_ray = rays[2] if len(rays) > 2 else 200.0
        min_ray = min(rays) if rays else 200.0

        event_detected: Optional[Tuple[str, str]] = None

        if self.is_crashed:
            event_detected = (
                "crash",
                f"Vehicle departed track boundary at step {step} (lateral offset {dist:.1f} px, speed {vel:.2f} px/step).",
            )
        elif self.is_completed:
            event_detected = (
                "lap_completed",
                f"Lap completed cleanly at step {step}. Mean velocity maintained at {vel:.2f} px/step across finish line.",
            )
        elif min_ray < 28.0 and abs(steer_deg) > 12.0:
            side = "Left" if rays[0] < rays[4] else "Right"
            event_detected = (
                "corner_apex",
                f"Apex clipping detected. {side} boundary clearance is {min_ray:.1f} px. Vehicle holding {steer_deg:.1f}° steering at {vel:.2f} px/step.",
            )
        elif fwd_ray < 85.0 and vel > 2.2:
            left_diag = rays[1] if len(rays) > 1 else 200.0
            event_detected = (
                "corner_approach",
                f"Corner approach detected. Vehicle speed is {vel:.2f} px/step. Left diagonal boundary is {left_diag:.1f} px away. Steering demand is {steer_act:+.2f}.",
            )
        elif abs(slip) > 4.5:
            event_detected = (
                "slip_correction",
                f"Elevated slip angle ({slip:+.1f}°) detected in turn. Counter-steering demand ({steer_deg:.1f}°) applied to stabilize trajectory.",
            )
        elif abs(steer_act) > 0.65:
            event_detected = (
                "high_steering_demand",
                f"High steering demand of {steer_deg:.1f}° (action {steer_act:+.3f}) applied to manage lateral offset ({dist:.1f} px, heading error {heading_err:.1f}°).",
            )
        elif throttle < -0.15:
            event_detected = (
                "braking_onset",
                f"Deceleration initiated (longitudinal command {throttle:+.3f}) entering turn, reducing forward velocity to {vel:.2f} px/step.",
            )
        elif fwd_ray > 140.0 and throttle > 0.70 and abs(steer_deg) < 6.0 and step > 15:
            event_detected = (
                "straight_cruising",
                f"Full throttle straight-line cruising at {vel:.2f} px/step with centered steering ({steer_deg:.1f}°) and clear forward vision.",
            )

        if event_detected:
            event_type, default_text = event_detected
            self.last_commentary_step = step

            if commentary_mode == "gemini" and gemini_api_key and self.gemini_calls_count < 5:
                # Attempt live Gemini commentary with GroundingValidator
                success = self._try_gemini_commentary(
                    event_type=event_type,
                    telemetry_snapshot=telemetry_snapshot,
                    api_key=gemini_api_key,
                )
                if not success:
                    self._add_commentary(step, event_type, default_text, source="offline_fallback")
            else:
                self._add_commentary(step, event_type, default_text, source="offline_deterministic")

    def _try_gemini_commentary(
        self,
        event_type: str,
        telemetry_snapshot: Dict[str, Any],
        api_key: str,
    ) -> bool:
        """Call Gemini Flash safely and validate claims with GroundingValidator."""
        try:
            from google import genai
            from google.genai import types
            from analysis.telemetry_commentary import GroundingValidator

            client = genai.Client(api_key=api_key)
            self.gemini_calls_count += 1

            prompt = (
                "You are an expert autonomous racing telemetry commentator. "
                "Describe the vehicle's state concisely in exactly one factual sentence. "
                "Only cite measurements present in this telemetry: "
                f"Event: {event_type}, Step: {telemetry_snapshot['step']}, "
                f"Velocity: {telemetry_snapshot['velocity']} px/step, "
                f"Steering angle: {telemetry_snapshot['steering_angle_deg']} deg (action {telemetry_snapshot['steering_action']}), "
                f"Throttle: {telemetry_snapshot['throttle_action']}, "
                f"Forward ray: {telemetry_snapshot['rays'][2]} px. "
                "Rules: NEVER mention km/h, mph, gaze, eyes, competitors, or nonexistent sensors."
            )

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=150,
                ),
            )
            raw_text = response.text.strip().replace("\n", " ")

            # Construct mock item for GroundingValidator
            item = {
                "event_id": f"live_step_{telemetry_snapshot['step']}",
                "commentary": raw_text,
                "confidence": 0.95,
                "evidence": {
                    "velocity": telemetry_snapshot["velocity"],
                    "steering_angle_deg": telemetry_snapshot["steering_angle_deg"],
                },
            }
            gt_event = {
                "event_id": f"live_step_{telemetry_snapshot['step']}",
                "telemetry_summary": {
                    "velocity_mean": telemetry_snapshot["velocity"],
                    "velocity_min": telemetry_snapshot["velocity"] - 0.1,
                    "velocity_max": telemetry_snapshot["velocity"] + 0.1,
                },
                "action_summary": {
                    "steering_deg_mean": telemetry_snapshot["steering_angle_deg"],
                    "steering_deg_min": telemetry_snapshot["steering_angle_deg"] - 1.0,
                    "steering_deg_max": telemetry_snapshot["steering_angle_deg"] + 1.0,
                },
                "sensor_summary": {},
                "saliency_triggers": {},
            }
            val_result = GroundingValidator.validate_commentary_item(item, gt_event)

            if val_result["is_valid"]:
                self._add_commentary(
                    telemetry_snapshot["step"],
                    event_type,
                    raw_text,
                    source="gemini_live",
                    metadata={"grounding_score": val_result["grounding_score"]},
                )
                return True
            else:
                logger.warning("Gemini commentary failed validation: %s", val_result["flagged_issues"])
                return False
        except Exception as exc:
            logger.warning("Gemini live commentary generation error: %s", exc)
            return False

    def _add_commentary(
        self,
        step: int,
        event_type: str,
        text: str,
        source: str = "offline_deterministic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append validated commentary record to log."""
        self.commentary_log.append({
            "step": step,
            "timestamp": time.time(),
            "event_type": event_type,
            "commentary": text,
            "source": source,
            "metadata": metadata or {},
        })
        if len(self.commentary_log) > 50:
            self.commentary_log.pop(0)

    def get_lap_review(self) -> Optional[Dict[str, Any]]:
        """Compute grounded post-lap performance review from session history."""
        if not self.history or len(self.history) < 10:
            return None

        velocities = [h["velocity"] for h in self.history]
        steering_acts = [h["steering_action"] for h in self.history]
        rewards = [h["reward"] for h in self.history]

        if len(steering_acts) > 1:
            steer_diffs = [
                abs(steering_acts[i] - steering_acts[i - 1])
                for i in range(1, len(steering_acts))
            ]
            steer_smoothness = float(np.mean(steer_diffs))
        else:
            steer_smoothness = 0.0

        all_rays = [r for h in self.history for r in (h.get("rays") or [])]
        min_boundary = min(all_rays) if all_rays else 200.0

        track_half_width = (self.track_geometry.get("width", 70.0) / 2.0) if self.track_geometry else 35.0
        off_track_count = sum(
            1 for h in self.history if abs(h.get("signed_distance", 0.0)) > track_half_width
        )

        avg_speed = float(np.mean(velocities))
        max_speed = float(np.max(velocities))
        total_reward = float(np.sum(rewards))
        total_steps = len(self.history)

        strengths: List[str] = []
        weaknesses: List[str] = []

        if max_speed >= 3.8:
            strengths.append(f"Achieved near-maximum velocity ({max_speed:.2f} px/step) on open sections.")
        if off_track_count == 0:
            strengths.append("Maintained 100% track boundary compliance with zero off-track excursions.")
        else:
            weaknesses.append(f"Exceeded nominal track boundary on {off_track_count} step(s).")

        if steer_smoothness < 0.08:
            strengths.append(f"High steering smoothness (mean Δaction of {steer_smoothness:.3f}) avoiding abrupt oscillation.")
        else:
            weaknesses.append(f"Steering oscillations detected during turn transitions (mean Δaction {steer_smoothness:.3f}).")

        if min_boundary < 20.0:
            weaknesses.append(f"Aggressive apex clipping with minimum boundary proximity dropping to {min_boundary:.1f} px.")
        elif min_boundary > 25.0:
            strengths.append(f"Safe margins preserved with minimum boundary proximity of {min_boundary:.1f} px.")

        return {
            "lap_completed": self.is_completed,
            "is_crashed": self.is_crashed,
            "total_steps": total_steps,
            "lap_time_steps": total_steps,
            "average_speed": round(avg_speed, 2),
            "maximum_speed": round(max_speed, 2),
            "steering_smoothness_score": round(max(0.0, 1.0 - steer_smoothness * 5.0) * 100, 1),
            "steering_jitter_metric": round(steer_smoothness, 4),
            "min_boundary_proximity_px": round(min_boundary, 1),
            "off_track_events": off_track_count,
            "cumulative_reward": round(total_reward, 2),
            "final_lap_progress_pct": round(self.env.lap_progress * 100.0, 1),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "verdict": (
                "EXCELLENT // CLEAN FAST LAP"
                if self.is_completed and off_track_count == 0 and avg_speed > 3.0
                else ("LAP COMPLETED WITH CORNERING STRESS" if self.is_completed else "INCOMPLETE RUN")
            ),
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialize current full simulation state for API response."""
        if self.env is None:
            self.start()

        car = self.env.car
        ray_distances = self.current_obs[4:9] if self.current_obs else [200.0] * 5

        ray_data = compute_ray_geometry(
            car_x=float(car.x),
            car_y=float(car.y),
            car_angle=float(car.angle),
            ray_angles=RAY_ANGLES,
            ray_distances=ray_distances,
        )

        return {
            "session": {
                "preset_id": self.preset_id,
                "preset_name": self.preset_info["name"],
                "track_seed": self.track_seed,
                "is_running": self.is_running,
                "is_crashed": self.is_crashed,
                "is_completed": self.is_completed,
                "step_count": self.env.step_count,
                "max_steps": self.env.max_steps,
            },
            "vehicle": {
                "x": round(float(car.x), 2),
                "y": round(float(car.y), 2),
                "angle": round(float(car.angle), 2),
                "velocity": round(float(car.velocity), 3),
                "steering": round(float(self.current_action[0]), 4),
                "steering_angle_deg": round(float(car.steering), 2),
                "throttle": round(float(self.current_action[1]), 4),
                "action_mode": (
                    "throttle"
                    if self.current_action[1] > 0
                    else ("deceleration" if self.current_action[1] < 0 else "coasting")
                ),
                "lap_progress": round(float(self.env.lap_progress), 4),
            },
            "telemetry": {
                "heading_error": round(float(self.current_obs[1]), 2) if self.current_obs else 0.0,
                "signed_distance": round(float(self.current_obs[2]), 2) if self.current_obs else 0.0,
                "slip_angle": round(float(self.current_obs[3]), 2) if self.current_obs else 0.0,
                "rays": [round(float(r), 2) for r in ray_distances],
            },
            "rays": ray_data,
            "trajectory": self.trajectory[-500:],  # Tail of trajectory for smooth rendering
            "trajectory_length": len(self.trajectory),
            "saliency": self.current_saliency,
            "commentary": self.commentary_log[-10:],
            "lap_review": self.get_lap_review() if (self.is_completed or self.is_crashed or self.env.step_count > 20) else None,
            "track": self.track_geometry,
        }


# Global singleton simulation instance for local interactive server
simulation_manager = SimulationSession()
