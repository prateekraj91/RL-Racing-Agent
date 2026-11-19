"""Model loader and inference utilities for frozen SAC generalist-v1.0 actor.

Enforces cryptographic integrity checks, strict evaluation mode, parameter
freezing, and lightweight inference without loading critics or optimizers.
"""

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from sac.actor import SACActor

logger = logging.getLogger("inference.model_loader")

# Immutable release SHA-256 hashes for generalist-v1.0
EXPECTED_MODEL_SHA256 = (
    "0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784"
)
EXPECTED_CONFIG_SHA256 = (
    "4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1"
)
EXPECTED_ACTOR_PARAMETERS = 18308


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute cryptographic SHA-256 hash of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_integrity(
    file_path: Union[str, Path], expected_sha256: str, label: str = "file"
) -> str:
    """Verify that a file matches its expected cryptographic SHA-256 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If SHA-256 does not match expected value.
    """
    actual_hash = compute_sha256(file_path)
    if actual_hash != expected_sha256:
        raise RuntimeError(
            f"Integrity check failed for {label} ({file_path})!\n"
            f"  Expected SHA-256: {expected_sha256}\n"
            f"  Actual SHA-256:   {actual_hash}\n"
            "Aborting model loading to prevent execution of unverified weights."
        )
    logger.info("Verified SHA-256 integrity for %s: %s", label, actual_hash)
    return actual_hash


def load_inference_actor(
    model_path: Union[str, Path] = "models/generalist_v1.pth",
    config_path: Optional[Union[str, Path]] = "configs/generalist_v1.json",
    device: str = "cpu",
) -> Tuple[SACActor, Dict[str, Any]]:
    """Load the frozen SACActor directly from weights with integrity validation.

    Loads ONLY the actor network (no critics, optimizers, or replay buffers).
    Sets eval mode, freezes all parameters, verifies parameter count, and runs
    a warm-up forward pass under torch.inference_mode().

    Returns:
        Tuple of (actor, metadata_dict)
    """
    model_path = Path(model_path)
    logger.info("Initiating pre-flight integrity check on model: %s", model_path)
    model_sha256 = verify_file_integrity(
        model_path, EXPECTED_MODEL_SHA256, label="model checkpoint"
    )

    metadata: Dict[str, Any] = {
        "model_name": "generalist-v1.0",
        "version": "1.0.0",
        "model_path": str(model_path),
        "weights_sha256": model_sha256,
        "device": device,
        "obs_dim": 9,
        "action_dim": 2,
        "parameter_count": EXPECTED_ACTOR_PARAMETERS,
    }

    if config_path is not None:
        cfg_path = Path(config_path)
        if cfg_path.is_file():
            config_sha256 = verify_file_integrity(
                cfg_path, EXPECTED_CONFIG_SHA256, label="config file"
            )
            metadata["config_sha256"] = config_sha256
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                metadata["model_name"] = cfg_data.get("model_name", metadata["model_name"])
                metadata["version"] = cfg_data.get("version", metadata["version"])
        else:
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
    else:
        metadata["config_sha256"] = EXPECTED_CONFIG_SHA256

    # Instantiate SACActor without critics or optimizers
    obs_dim = metadata["obs_dim"]
    action_dim = metadata["action_dim"]
    actor = SACActor(obs_dim=obs_dim, action_dim=action_dim)

    # Load weights safely using weights_only=True
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(state_dict, strict=True)

    # Enforce strict evaluation and frozen parameters
    actor.eval()
    actor.requires_grad_(False)
    actor.to(device)

    # Validate exact parameter count (18,308 for the actor network)
    param_count = sum(p.numel() for p in actor.parameters())
    if param_count != EXPECTED_ACTOR_PARAMETERS:
        raise ValueError(
            f"Unexpected actor parameter count: {param_count} (expected {EXPECTED_ACTOR_PARAMETERS})"
        )
    metadata["parameter_count"] = param_count

    # Execute warm-up forward pass to prime CPU kernels and memory
    dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32, device=device)
    with torch.inference_mode():
        mean, _ = actor(dummy_input)
        _ = torch.tanh(mean)

    logger.info(
        "Successfully initialized frozen SACActor (%d parameters) on %s",
        param_count,
        device,
    )
    return actor, metadata


def predict_action(
    actor: SACActor,
    observation: Sequence[float],
    device: str = "cpu",
) -> Tuple[List[float], float]:
    """Execute deterministic action prediction for a single observation vector.

    Calculates action = tanh(mean) under torch.inference_mode().

    Returns:
        Tuple of ([steering, longitudinal], latency_ms)
    """
    t0 = time.perf_counter()
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        mean, _ = actor(obs_tensor)
        action_tensor = torch.tanh(mean)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    action = [float(action_tensor[0, 0].item()), float(action_tensor[0, 1].item())]
    return action, latency_ms


def predict_batch(
    actor: SACActor,
    observations: Sequence[Sequence[float]],
    device: str = "cpu",
) -> Tuple[List[List[float]], float]:
    """Execute deterministic action prediction for a batch of observations.

    Returns:
        Tuple of (list_of_actions, total_latency_ms) preserving input ordering.
    """
    t0 = time.perf_counter()
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.inference_mode():
        mean, _ = actor(obs_tensor)
        actions_tensor = torch.tanh(mean)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    actions = [
        [float(actions_tensor[i, 0].item()), float(actions_tensor[i, 1].item())]
        for i in range(actions_tensor.shape[0])
    ]
    return actions, latency_ms
