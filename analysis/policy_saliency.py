"""Policy Saliency Implementation for Frozen generalist-v1.0 SAC Agent.

Calculates gradient-based input saliency per simulation timestep to determine which
ray-cast beams dictate steering and longitudinal (throttle/brake) control.

MODEL FREEZE RULE:
- models/generalist_v1.pth and configs/generalist_v1.json are strictly immutable.
- Loads the frozen checkpoint with requires_grad_(False) on all parameters.
- Autograd gradients are computed solely with respect to the input observation tensor.
- Checkpoint SHA-256 hashes are verified before and after analysis.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.environment import RacingEnv
from sac.actor import SACActor
from sac.track_pools import POOL_CONFIG, TEST_SEEDS

# Immutable Reference Paths and Cryptographic Checksums
FROZEN_MODEL_PATH = os.path.join(ROOT_DIR, "models", "generalist_v1.pth")
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "generalist_v1.json")
EXPECTED_WEIGHTS_SHA256 = "0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784"
EXPECTED_CONFIG_SHA256 = "4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1"

# Results output paths
RESULTS_DIR = os.path.join(ROOT_DIR, "analysis", "results")
TIMESTEPS_CSV_PATH = os.path.join(RESULTS_DIR, "policy_saliency_timesteps.csv")
SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "policy_saliency_summary.json")

# Observation and Sensor Contract
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
RAY_FEATURE_NAMES = [
    "ray_minus_90",
    "ray_minus_45",
    "ray_0",
    "ray_plus_45",
    "ray_plus_90",
]
RAY_ANGLES = [-90.0, -45.0, 0.0, 45.0, 90.0]


def compute_sha256(filepath: str) -> str:
    """Compute the SHA-256 hexadecimal digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_integrity() -> Tuple[str, str]:
    """Verify cryptographic checksums for frozen checkpoint and configuration.
    
    Raises:
        AssertionError: If weights or configuration do not match immutable release hashes.
    """
    if not os.path.isfile(FROZEN_MODEL_PATH):
        raise FileNotFoundError(f"Frozen model missing: {FROZEN_MODEL_PATH}")
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration missing: {CONFIG_PATH}")

    actual_weights_sha = compute_sha256(FROZEN_MODEL_PATH)
    actual_config_sha = compute_sha256(CONFIG_PATH)

    assert actual_weights_sha == EXPECTED_WEIGHTS_SHA256, (
        f"Frozen model hash mismatch! Expected {EXPECTED_WEIGHTS_SHA256}, got {actual_weights_sha}"
    )
    assert actual_config_sha == EXPECTED_CONFIG_SHA256, (
        f"Frozen config hash mismatch! Expected {EXPECTED_CONFIG_SHA256}, got {actual_config_sha}"
    )

    return actual_weights_sha, actual_config_sha


def load_frozen_actor(model_path: str = FROZEN_MODEL_PATH) -> SACActor:
    """Load the frozen generalist actor, freezing all parameters in eval mode.
    
    Enforces MODEL FREEZE RULE:
    1. Loads weights with weights_only=True.
    2. Sets network to eval mode.
    3. Freezes all parameters (requires_grad_(False)).
    """
    verify_frozen_integrity()

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    obs_dim = state_dict["network.0.weight"].shape[1]
    action_dim = state_dict["mean.weight"].shape[0]

    actor = SACActor(obs_dim=obs_dim, action_dim=action_dim)
    actor.load_state_dict(state_dict)
    actor.eval()

    for param in actor.parameters():
        param.requires_grad_(False)

    return actor


def compute_action_saliency(
    actor: SACActor,
    observation: np.ndarray,
    return_presquashed: bool = True,
) -> Dict[str, Any]:
    """Calculate gradient-based input saliency for a single observation.
    
    Args:
        actor: Frozen SACActor in eval mode with requires_grad_(False).
        observation: 1D array of shape (9,) matching observation space contract.
        return_presquashed: If True, also compute latent pre-squashed gradients.
        
    Returns:
        Dictionary containing:
        - deployed_action: (steer_act, long_act) after tanh squashing
        - grad_steer: raw gradient d(action_steer) / d(obs) [shape (9,)]
        - grad_long: raw gradient d(action_long) / d(obs) [shape (9,)]
        - saliency_steer: absolute magnitude |d(action_steer) / d(obs)|
        - saliency_long: absolute magnitude |d(action_long) / d(obs)|
        - grad_mu_steer: pre-squashed gradient d(mu_0) / d(obs)
        - grad_mu_long: pre-squashed gradient d(mu_1) / d(obs)
        - saliency_mu_steer: pre-squashed absolute magnitude |d(mu_0) / d(obs)|
        - saliency_mu_long: pre-squashed absolute magnitude |d(mu_1) / d(obs)|
    """
    obs_tensor = torch.tensor(observation, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    mean, _ = actor(obs_tensor)
    deployed_action = torch.tanh(mean)

    steer_action = deployed_action[0, 0]
    long_action = deployed_action[0, 1]

    # Compute deployed action gradients
    grad_steer = torch.autograd.grad(
        outputs=steer_action,
        inputs=obs_tensor,
        retain_graph=True,
        create_graph=False,
    )[0].squeeze(0).detach().numpy()

    grad_long = torch.autograd.grad(
        outputs=long_action,
        inputs=obs_tensor,
        retain_graph=return_presquashed,
        create_graph=False,
    )[0].squeeze(0).detach().numpy()

    saliency_steer = np.abs(grad_steer)
    saliency_long = np.abs(grad_long)

    result: Dict[str, Any] = {
        "deployed_action": (float(steer_action.item()), float(long_action.item())),
        "grad_steer": grad_steer,
        "grad_long": grad_long,
        "saliency_steer": saliency_steer,
        "saliency_long": saliency_long,
    }

    if return_presquashed:
        grad_mu_steer = torch.autograd.grad(
            outputs=mean[0, 0],
            inputs=obs_tensor,
            retain_graph=True,
            create_graph=False,
        )[0].squeeze(0).detach().numpy()

        grad_mu_long = torch.autograd.grad(
            outputs=mean[0, 1],
            inputs=obs_tensor,
            retain_graph=False,
            create_graph=False,
        )[0].squeeze(0).detach().numpy()

        result["grad_mu_steer"] = grad_mu_steer
        result["grad_mu_long"] = grad_mu_long
        result["saliency_mu_steer"] = np.abs(grad_mu_steer)
        result["saliency_mu_long"] = np.abs(grad_mu_long)

    return result


def validate_gradients_finite_difference(
    actor: SACActor,
    observation: np.ndarray,
    eps: float = 1e-3,
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """Validate autograd input gradients against central finite difference perturbations.
    
    For each feature i in 0..8:
        g_fd = (action(s + eps * e_i) - action(s - eps * e_i)) / (2 * eps)
        
    Args:
        actor: Frozen SACActor in eval mode.
        observation: Observation vector (9,).
        eps: Perturbation step size.
        tolerance: Maximum acceptable discrepancy for smooth regions.
        
    Returns:
        Dictionary with max_error, mean_error, passed boolean, and per-action comparisons.
    """
    saliency = compute_action_saliency(actor, observation, return_presquashed=False)
    grad_auto_steer = saliency["grad_steer"]
    grad_auto_long = saliency["grad_long"]

    grad_fd_steer = np.zeros(9, dtype=np.float32)
    grad_fd_long = np.zeros(9, dtype=np.float32)

    for i in range(9):
        obs_plus = observation.copy()
        obs_plus[i] += eps
        obs_minus = observation.copy()
        obs_minus[i] -= eps

        with torch.no_grad():
            mean_p, _ = actor(torch.tensor(obs_plus, dtype=torch.float32).unsqueeze(0))
            act_p = torch.tanh(mean_p).numpy()[0]

            mean_m, _ = actor(torch.tensor(obs_minus, dtype=torch.float32).unsqueeze(0))
            act_m = torch.tanh(mean_m).numpy()[0]

        grad_fd_steer[i] = (act_p[0] - act_m[0]) / (2.0 * eps)
        grad_fd_long[i] = (act_p[1] - act_m[1]) / (2.0 * eps)

    diff_steer = np.abs(grad_auto_steer - grad_fd_steer)
    diff_long = np.abs(grad_auto_long - grad_fd_long)

    max_err_steer = float(np.max(diff_steer))
    mean_err_steer = float(np.mean(diff_steer))
    max_err_long = float(np.max(diff_long))
    mean_err_long = float(np.mean(diff_long))

    max_overall_err = max(max_err_steer, max_err_long)
    passed = max_overall_err <= tolerance

    return {
        "passed": passed,
        "max_overall_error": max_overall_err,
        "steering": {
            "autograd": grad_auto_steer.tolist(),
            "finite_diff": grad_fd_steer.tolist(),
            "max_error": max_err_steer,
            "mean_error": mean_err_steer,
        },
        "longitudinal": {
            "autograd": grad_auto_long.tolist(),
            "finite_diff": grad_fd_long.tolist(),
            "max_error": max_err_long,
            "mean_error": mean_err_long,
        },
    }


def run_episode_with_saliency(
    actor: SACActor,
    track_seed: int,
    track_kwargs: Dict[str, Any],
    benchmark_name: str,
    benchmark_param: Any,
    episode_id: str,
    max_steps: int = 700,
    ray_angles: Tuple[float, ...] = tuple(RAY_ANGLES),
    ray_max_dist: float = 200.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute a single episode with per-timestep input saliency recording.
    
    Args:
        actor: Frozen SACActor.
        track_seed: Seed for track generation.
        track_kwargs: Track configuration dictionary.
        benchmark_name: Benchmark suite name.
        benchmark_param: Associated parameter (width or hairpin factor).
        episode_id: Unique episode identifier.
        max_steps: Episode horizon limit.
        ray_angles: Tuple of 5 sensor ray angles.
        ray_max_dist: Maximum lookahead distance.
        
    Returns:
        Tuple of (list_of_timestep_records, episode_summary).
    """
    env = RacingEnv(
        max_steps=max_steps,
        verbose=False,
        track_kwargs=track_kwargs,
        obs_ray_angles=ray_angles,
        obs_ray_max_dist=ray_max_dist,
    )

    obs, _ = env.reset(options={"track_seed": track_seed})
    timestep_records: List[Dict[str, Any]] = []

    completed = False
    crashed = False
    steps = 0
    final_info: Dict[str, Any] = {}

    for step in range(max_steps):
        # 1. Calculate gradient-based input saliency on current observation
        sal = compute_action_saliency(actor, obs, return_presquashed=True)
        deployed_steer, deployed_long = sal["deployed_action"]

        # Track geometry context: corner tightness from ground-truth spline
        corner_r = env.track.corner_radius(env.car.x, env.car.y)
        # Threshold: radius < 150 px classified as corner / turn
        is_turn = corner_r < 150.0 or abs(env.car.steering) > 5.0
        track_section = "turn" if is_turn else "straight"

        # Action mode: positive throttle vs negative braking
        action_mode = "throttle" if deployed_long >= 0.0 else "brake"

        record: Dict[str, Any] = {
            "track_seed": track_seed,
            "benchmark": benchmark_name,
            "benchmark_param": benchmark_param,
            "episode_id": episode_id,
            "step": step,
            # Observation features
            "obs_velocity": float(obs[0]),
            "obs_heading_error": float(obs[1]),
            "obs_signed_distance": float(obs[2]),
            "obs_slip": float(obs[3]),
            "obs_ray_minus_90": float(obs[4]),
            "obs_ray_minus_45": float(obs[5]),
            "obs_ray_0": float(obs[6]),
            "obs_ray_plus_45": float(obs[7]),
            "obs_ray_plus_90": float(obs[8]),
            # Deployed actions
            "action_steering": deployed_steer,
            "action_longitudinal": deployed_long,
            "action_mode": action_mode,
            # Geometric telemetry
            "car_x": float(env.car.x),
            "car_y": float(env.car.y),
            "car_angle": float(env.car.angle),
            "car_velocity": float(env.car.velocity),
            "corner_radius": float(corner_r),
            "track_section": track_section,
            "lap_progress": float(env.lap_progress),
            # Per-feature steering gradient magnitude (S(i, steer) = |d a_steer / d s_i|)
            "saliency_steer_velocity": float(sal["saliency_steer"][0]),
            "saliency_steer_heading_error": float(sal["saliency_steer"][1]),
            "saliency_steer_signed_distance": float(sal["saliency_steer"][2]),
            "saliency_steer_slip": float(sal["saliency_steer"][3]),
            "saliency_steer_ray_minus_90": float(sal["saliency_steer"][4]),
            "saliency_steer_ray_minus_45": float(sal["saliency_steer"][5]),
            "saliency_steer_ray_0": float(sal["saliency_steer"][6]),
            "saliency_steer_ray_plus_45": float(sal["saliency_steer"][7]),
            "saliency_steer_ray_plus_90": float(sal["saliency_steer"][8]),
            # Per-feature longitudinal gradient magnitude (S(i, long) = |d a_long / d s_i|)
            "saliency_long_velocity": float(sal["saliency_long"][0]),
            "saliency_long_heading_error": float(sal["saliency_long"][1]),
            "saliency_long_signed_distance": float(sal["saliency_long"][2]),
            "saliency_long_slip": float(sal["saliency_long"][3]),
            "saliency_long_ray_minus_90": float(sal["saliency_long"][4]),
            "saliency_long_ray_minus_45": float(sal["saliency_long"][5]),
            "saliency_long_ray_0": float(sal["saliency_long"][6]),
            "saliency_long_ray_plus_45": float(sal["saliency_long"][7]),
            "saliency_long_ray_plus_90": float(sal["saliency_long"][8]),
            # Signed deployed gradients
            "grad_steer_ray_minus_90": float(sal["grad_steer"][4]),
            "grad_steer_ray_minus_45": float(sal["grad_steer"][5]),
            "grad_steer_ray_0": float(sal["grad_steer"][6]),
            "grad_steer_ray_plus_45": float(sal["grad_steer"][7]),
            "grad_steer_ray_plus_90": float(sal["grad_steer"][8]),
            "grad_long_ray_minus_90": float(sal["grad_long"][4]),
            "grad_long_ray_minus_45": float(sal["grad_long"][5]),
            "grad_long_ray_0": float(sal["grad_long"][6]),
            "grad_long_ray_plus_45": float(sal["grad_long"][7]),
            "grad_long_ray_plus_90": float(sal["grad_long"][8]),
            # Pre-squashing latent sensitivities
            "saliency_mu_steer_ray_0": float(sal["saliency_mu_steer"][6]),
            "saliency_mu_long_ray_0": float(sal["saliency_mu_long"][6]),
            # Outcome placeholder, filled on termination
            "outcome": "running",
        }
        timestep_records.append(record)

        # 2. Step environment using deterministic deployed action
        obs, reward, terminated, truncated, info = env.step(np.array([deployed_steer, deployed_long]))
        steps += 1
        final_info = info

        if terminated or truncated:
            completed = bool(info.get("lap_completed", False))
            crashed = bool(info.get("crashed", False))
            break

    # Determine episode outcome
    if completed:
        outcome = "completed"
    elif crashed:
        outcome = "crashed"
    elif steps >= max_steps:
        outcome = "timeout"
    else:
        outcome = "terminated"

    # Propagate exact outcome to all timesteps of this episode
    for rec in timestep_records:
        rec["outcome"] = outcome

    summary = {
        "episode_id": episode_id,
        "track_seed": track_seed,
        "benchmark": benchmark_name,
        "benchmark_param": benchmark_param,
        "outcome": outcome,
        "completed": completed,
        "crashed": crashed,
        "steps": steps,
        "lap_progress": round(float(final_info.get("lap_progress", 0.0)), 4),
    }

    return timestep_records, summary


def collect_controlled_dataset(actor: SACActor) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect timestep-level saliency records across controlled held-out benchmarks.
    
    Includes:
    1. Standard held-out circuits (TEST_SEEDS 1001..1010, width 70 px).
    2. Narrow corridors (width 20 px and width 40 px on representative seeds).
    3. Hairpins (pinched acute geometry f=0.50 and f=0.35 where agent completes).
    4. Failure case included without fabrication (f=0.20 seed 1001 boundary crash).
    """
    base_tk = {
        "width": 70,
        "base_r": 250,
        "n_ctrl": 10,
        "min_radius": 80,
        "cx": 400,
        "cy": 300,
    }

    all_timesteps: List[Dict[str, Any]] = []
    all_episodes: List[Dict[str, Any]] = []

    print("\n--- Phase 3: Collecting Timestep-Level Policy Saliency Dataset ---")

    # 1. Standard Geometry: All 10 held-out test seeds (1001..1010)
    print("Collecting Standard Held-Out benchmark (width 70 px)...")
    for s in TEST_SEEDS:
        ep_id = f"std_w70_s{s}"
        recs, ep_sum = run_episode_with_saliency(
            actor=actor,
            track_seed=s,
            track_kwargs=base_tk,
            benchmark_name="standard_heldout",
            benchmark_param=70,
            episode_id=ep_id,
            max_steps=700,
        )
        all_timesteps.extend(recs)
        all_episodes.append(ep_sum)
        print(f"  [{ep_id}] steps={ep_sum['steps']}, outcome={ep_sum['outcome']}")

    # 2. Narrow Corridors: Width 20 px and 40 px on representative held-out seeds
    print("\nCollecting Narrow Corridor benchmark (widths 20 px & 40 px)...")
    narrow_seeds = [1001, 1002, 1003, 1005, 1007, 1008]
    for w in [20, 40]:
        tk = dict(base_tk, width=w)
        for s in narrow_seeds:
            ep_id = f"narrow_w{w}_s{s}"
            recs, ep_sum = run_episode_with_saliency(
                actor=actor,
                track_seed=s,
                track_kwargs=tk,
                benchmark_name="narrow_corridor",
                benchmark_param=w,
                episode_id=ep_id,
                max_steps=700,
            )
            all_timesteps.extend(recs)
            all_episodes.append(ep_sum)
            print(f"  [{ep_id}] steps={ep_sum['steps']}, outcome={ep_sum['outcome']}")

    # 3. Hairpins: High-curvature geometry (f=0.50 and f=0.35)
    print("\nCollecting Hairpin benchmark (pinched acute geometry)...")
    hairpin_cases = [
        # (hairpin_factor, seed)
        (0.50, 1002),
        (0.50, 1003),
        (0.50, 1006),
        (0.50, 1007),
        (0.50, 1008),
        (0.35, 1007),
        (0.35, 1008),
        # Real failure case (crash) to fulfill requirement of recording failure outcomes without fabrication
        (0.20, 1001),
    ]

    for hf, s in hairpin_cases:
        tk = dict(base_tk, hairpin=True, hairpin_factor=hf)
        ep_id = f"hairpin_f{hf}_s{s}"
        recs, ep_sum = run_episode_with_saliency(
            actor=actor,
            track_seed=s,
            track_kwargs=tk,
            benchmark_name="hairpin",
            benchmark_param=hf,
            episode_id=ep_id,
            max_steps=700,
        )
        all_timesteps.extend(recs)
        all_episodes.append(ep_sum)
        print(f"  [{ep_id}] steps={ep_sum['steps']}, outcome={ep_sum['outcome']}")

    print(f"\nTotal episodes collected: {len(all_episodes)}")
    print(f"Total timesteps collected: {len(all_timesteps)}")
    return all_timesteps, all_episodes


def analyze_saliency_dataset(
    timesteps: List[Dict[str, Any]],
    actor: SACActor,
    n_fd_samples: int = 50,
) -> Dict[str, Any]:
    """Perform quantitative analysis on collected saliency timesteps.
    
    Computes:
    - Overall ray sensitivities for steering, throttle (>=0), and braking (<0).
    - Scaled sensitivities accounting for observation feature variances.
    - Straight vs turn sensitivity breakdown.
    - Temporal evolution across episode quartiles.
    - Finite-difference validation across a representative sample.
    """
    n_records = len(timesteps)
    assert n_records > 0, "No timesteps to analyze!"

    # 1. Feature statistics (mean and standard deviation) for normalization
    feature_arrays = {feat: np.array([r[f"obs_{feat}"] for r in timesteps], dtype=np.float64) for feat in FEATURE_NAMES}
    feature_stats = {
        feat: {
            "mean": float(np.mean(feature_arrays[feat])),
            "std": float(np.std(feature_arrays[feat], ddof=1)),
            "min": float(np.min(feature_arrays[feat])),
            "max": float(np.max(feature_arrays[feat])),
        }
        for feat in FEATURE_NAMES
    }

    # Helper to calculate average raw and scaled saliency
    def calc_mean_saliency(records: List[Dict[str, Any]], action_prefix: str) -> Dict[str, Any]:
        raw_means = {}
        scaled_means = {}
        for feat in FEATURE_NAMES:
            vals = np.array([r[f"saliency_{action_prefix}_{feat}"] for r in records], dtype=np.float64)
            m = float(np.mean(vals))
            raw_means[feat] = m
            # Scaled sensitivity: raw gradient * standard deviation of feature
            scaled_means[feat] = m * feature_stats[feat]["std"]

        # Ranking among 5 ray-cast sensors
        ray_raw_ranking = sorted(
            [(rf, raw_means[rf]) for rf in RAY_FEATURE_NAMES],
            key=lambda x: x[1],
            reverse=True,
        )
        ray_scaled_ranking = sorted(
            [(rf, scaled_means[rf]) for rf in RAY_FEATURE_NAMES],
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "raw_mean": raw_means,
            "scaled_mean": scaled_means,
            "top_ray_raw": ray_raw_ranking[0][0],
            "top_ray_raw_value": ray_raw_ranking[0][1],
            "top_ray_scaled": ray_scaled_ranking[0][0],
            "top_ray_scaled_value": ray_scaled_ranking[0][1],
            "ray_raw_ranking": ray_raw_ranking,
            "ray_scaled_ranking": ray_scaled_ranking,
        }

    # 2. Overall Steering Saliency
    overall_steer = calc_mean_saliency(timesteps, "steer")

    # 3. Overall Longitudinal Saliency
    overall_long = calc_mean_saliency(timesteps, "long")

    # 4. Partition by Throttle vs Braking
    throttle_records = [r for r in timesteps if r["action_mode"] == "throttle"]
    brake_records = [r for r in timesteps if r["action_mode"] == "brake"]

    throttle_analysis = (
        calc_mean_saliency(throttle_records, "long")
        if throttle_records
        else {"error": "no throttle records"}
    )
    brake_analysis = (
        calc_mean_saliency(brake_records, "long")
        if brake_records
        else {"error": "no brake records"}
    )

    # 5. Partition by Straight vs Turn
    straight_records = [r for r in timesteps if r["track_section"] == "straight"]
    turn_records = [r for r in timesteps if r["track_section"] == "turn"]

    straight_steer = calc_mean_saliency(straight_records, "steer") if straight_records else {}
    straight_long = calc_mean_saliency(straight_records, "long") if straight_records else {}

    turn_steer = calc_mean_saliency(turn_records, "steer") if turn_records else {}
    turn_long = calc_mean_saliency(turn_records, "long") if turn_records else {}

    # 6. Temporal Evolution across Episode Quartiles (Q1: 0-25%, Q2: 25-50%, Q3: 50-75%, Q4: 75-100%)
    quartile_bins: Dict[str, List[Dict[str, Any]]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for r in timesteps:
        prog = r["lap_progress"]
        if prog < 0.25:
            quartile_bins["Q1"].append(r)
        elif prog < 0.50:
            quartile_bins["Q2"].append(r)
        elif prog < 0.75:
            quartile_bins["Q3"].append(r)
        else:
            quartile_bins["Q4"].append(r)

    temporal_steer = {q: calc_mean_saliency(recs, "steer") for q, recs in quartile_bins.items() if recs}
    temporal_long = {q: calc_mean_saliency(recs, "long") for q, recs in quartile_bins.items() if recs}

    # 7. Finite Difference Validation on Sample of Timesteps
    step_indices = np.linspace(0, n_records - 1, min(n_fd_samples, n_records), dtype=int)
    fd_results: List[Dict[str, Any]] = []
    fd_max_errors = []
    fd_mean_errors = []

    for idx in step_indices:
        r = timesteps[idx]
        obs = np.array([r[f"obs_{feat}"] for feat in FEATURE_NAMES], dtype=np.float32)
        val = validate_gradients_finite_difference(actor, obs, eps=1e-3, tolerance=5e-3)
        fd_results.append({
            "timestep_index": int(idx),
            "track_seed": r["track_seed"],
            "step": r["step"],
            "passed": val["passed"],
            "max_overall_error": val["max_overall_error"],
            "steer_max_error": val["steering"]["max_error"],
            "long_max_error": val["longitudinal"]["max_error"],
        })
        fd_max_errors.append(val["max_overall_error"])
        fd_mean_errors.append(0.5 * (val["steering"]["mean_error"] + val["longitudinal"]["mean_error"]))

    fd_summary = {
        "n_samples_tested": len(fd_results),
        "all_passed": all(res["passed"] for res in fd_results),
        "pass_rate": float(np.mean([1.0 if res["passed"] else 0.0 for res in fd_results])),
        "max_discrepancy": float(np.max(fd_max_errors)) if fd_max_errors else 0.0,
        "mean_discrepancy": float(np.mean(fd_mean_errors)) if fd_mean_errors else 0.0,
    }

    # 8. Check finiteness, non-zero, non-NaN assertions
    all_steer_sal = np.concatenate([[r[f"saliency_steer_{f}"] for f in FEATURE_NAMES] for r in timesteps])
    all_long_sal = np.concatenate([[r[f"saliency_long_{f}"] for f in FEATURE_NAMES] for r in timesteps])

    gradient_health = {
        "has_nan": bool(np.isnan(all_steer_sal).any() or np.isnan(all_long_sal).any()),
        "has_inf": bool(np.isinf(all_steer_sal).any() or np.isinf(all_long_sal).any()),
        "all_zeros_steer": bool(np.all(all_steer_sal == 0.0)),
        "all_zeros_long": bool(np.all(all_long_sal == 0.0)),
        "steer_saliency_min": float(np.min(all_steer_sal)),
        "steer_saliency_max": float(np.max(all_steer_sal)),
        "long_saliency_min": float(np.min(all_long_sal)),
        "long_saliency_max": float(np.max(all_long_sal)),
    }

    return {
        "total_timesteps": n_records,
        "feature_statistics": feature_stats,
        "overall_steering": overall_steer,
        "overall_longitudinal": overall_long,
        "throttle_regime": {
            "n_timesteps": len(throttle_records),
            "percentage": round(100.0 * len(throttle_records) / n_records, 2),
            "analysis": throttle_analysis,
        },
        "braking_regime": {
            "n_timesteps": len(brake_records),
            "percentage": round(100.0 * len(brake_records) / n_records, 2),
            "analysis": brake_analysis,
        },
        "straight_vs_turn": {
            "n_straight": len(straight_records),
            "n_turn": len(turn_records),
            "straight_steering": straight_steer,
            "turn_steering": turn_steer,
            "straight_longitudinal": straight_long,
            "turn_longitudinal": turn_long,
        },
        "temporal_evolution": {
            "steering_by_quartile": temporal_steer,
            "longitudinal_by_quartile": temporal_long,
        },
        "finite_difference_validation": fd_summary,
        "gradient_health": gradient_health,
    }


def save_dataset_and_summary(
    timesteps: List[Dict[str, Any]],
    summary_data: Dict[str, Any],
    initial_sha: str,
    final_sha: str,
) -> None:
    """Save collected timesteps to CSV and summary analysis to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Save CSV
    if timesteps:
        fieldnames = list(timesteps[0].keys())
        with open(TIMESTEPS_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(timesteps)
        print(f"Saved {len(timesteps)} records to: {TIMESTEPS_CSV_PATH}")

    # 2. Add metadata to summary and save JSON
    full_summary = {
        "metadata": {
            "model_name": "generalist_v1",
            "weights_path": FROZEN_MODEL_PATH,
            "config_path": CONFIG_PATH,
            "sha256_before_analysis": initial_sha,
            "sha256_after_analysis": final_sha,
            "weights_unaltered": initial_sha == final_sha == EXPECTED_WEIGHTS_SHA256,
            "target_action_derivative": "deployed_action (tanh(mean))",
            "saliency_metric": "S(i, action) = |d action / d observation_i|",
        },
        "findings_key_answers": {
            "steering_highest_sensitivity_ray": summary_data["overall_steering"]["top_ray_raw"],
            "steering_highest_sensitivity_ray_scaled": summary_data["overall_steering"]["top_ray_scaled"],
            "acceleration_highest_sensitivity_ray": (
                summary_data["throttle_regime"]["analysis"]["top_ray_raw"]
                if "analysis" in summary_data["throttle_regime"]
                else None
            ),
            "braking_highest_sensitivity_ray": (
                summary_data["braking_regime"]["analysis"]["top_ray_raw"]
                if "analysis" in summary_data["braking_regime"]
                else None
            ),
        },
        "analysis_results": summary_data,
    }

    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved summary metrics to: {SUMMARY_JSON_PATH}")


def main() -> None:
    """Main CLI entrypoint for policy saliency analysis."""
    parser = argparse.ArgumentParser(description="Policy Saliency Analysis for generalist_v1")
    parser.add_argument("--validate-only", action="store_true", help="Run finite-difference validation only")
    args = parser.parse_args()

    print("=" * 80)
    print("RL RACING AGENT: POLICY SALIENCY IMPLEMENTATION")
    print("Target: generalist-v1.0 Frozen SAC Actor")
    print("=" * 80)

    # 1. Verify frozen checksums BEFORE analysis
    sha_weights_before, sha_config_before = verify_frozen_integrity()
    print(f"Pre-analysis weights SHA-256: {sha_weights_before} [VERIFIED]")
    print(f"Pre-analysis config SHA-256:  {sha_config_before} [VERIFIED]")

    # 2. Load frozen actor in evaluation mode
    actor = load_frozen_actor(FROZEN_MODEL_PATH)
    print("Loaded frozen actor successfully with requires_grad_(False) on all parameters.")

    if args.validate_only:
        dummy_obs = np.array([3.5, 12.0, -5.0, 0.0, 36.0, 52.0, 110.0, 65.0, 36.0], dtype=np.float32)
        val = validate_gradients_finite_difference(actor, dummy_obs)
        print(f"Finite difference validation passed: {val['passed']} (max error: {val['max_overall_error']:.6e})")
        return

    # 3. Collect controlled dataset across held-out tracks
    timesteps, episodes = collect_controlled_dataset(actor)

    # 4. Analyze dataset
    print("\n--- Phase 4: Saliency Analysis and Validation ---")
    summary = analyze_saliency_dataset(timesteps, actor)

    # 5. Verify frozen checksums AFTER analysis
    sha_weights_after, sha_config_after = verify_frozen_integrity()
    print(f"\nPost-analysis weights SHA-256: {sha_weights_after} [VERIFIED]")
    print(f"Post-analysis config SHA-256:  {sha_config_after} [VERIFIED]")
    assert sha_weights_before == sha_weights_after == EXPECTED_WEIGHTS_SHA256
    assert sha_config_before == sha_config_after == EXPECTED_CONFIG_SHA256

    # 6. Save deliverables
    save_dataset_and_summary(timesteps, summary, sha_weights_before, sha_weights_after)

    # 7. Print summary of key findings
    print("\n" + "=" * 80)
    print("POLICY SALIENCY EXECUTIVE FINDINGS")
    print("=" * 80)
    print(f"Total timesteps analyzed: {summary['total_timesteps']}")
    print(f"Finite-difference validation pass rate: {summary['finite_difference_validation']['pass_rate'] * 100:.1f}%")
    print(f"Max FD discrepancy: {summary['finite_difference_validation']['max_discrepancy']:.6e}")
    print("\nSensory Ray Saliency Rankings:")
    print(f"  Steering Highest Sensitivity Ray (Raw):   {summary['overall_steering']['top_ray_raw']} "
          f"({summary['overall_steering']['top_ray_raw_value']:.6f} action/px)")
    print(f"  Steering Highest Sensitivity Ray (Scaled): {summary['overall_steering']['top_ray_scaled']} "
          f"({summary['overall_steering']['top_ray_scaled_value']:.6f})")

    if "analysis" in summary["throttle_regime"] and "top_ray_raw" in summary["throttle_regime"]["analysis"]:
        th = summary["throttle_regime"]["analysis"]
        print(f"  Acceleration (Throttle) Top Ray (Raw):    {th['top_ray_raw']} "
              f"({th['top_ray_raw_value']:.6f} action/px)")
        print(f"  Acceleration (Throttle) Top Ray (Scaled):  {th['top_ray_scaled']} "
              f"({th['top_ray_scaled_value']:.6f})")

    if "analysis" in summary["braking_regime"] and "top_ray_raw" in summary["braking_regime"]["analysis"]:
        br = summary["braking_regime"]["analysis"]
        print(f"  Braking Top Ray (Raw):                    {br['top_ray_raw']} "
              f"({br['top_ray_raw_value']:.6f} action/px)")
        print(f"  Braking Top Ray (Scaled):                 {br['top_ray_scaled']} "
              f"({br['top_ray_scaled_value']:.6f})")

    print("\nStraight vs Turn Breakdown (Steering Top Ray):")
    print(f"  Straight sections: {summary['straight_vs_turn']['straight_steering'].get('top_ray_raw')}")
    print(f"  Turn sections:     {summary['straight_vs_turn']['turn_steering'].get('top_ray_raw')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
