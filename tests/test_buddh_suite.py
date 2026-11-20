"""Unit and Integration Tests for Buddh International Circuit Evaluation Suite."""

import hashlib
import json
import os
import numpy as np
import pytest
import torch

from buddh.buddh_track import BuddhTrack
from buddh.importer import BuddhTrackImporter
from env.car import Car
from sac.actor import SACActor


FROZEN_SHA256 = "0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784"
MODEL_PATH = "models/generalist_v1.pth"
RAW_OSM_PATH = "data/osm/buddh_circuit_raw_osm.json"
MANIFEST_PATH = "buddh/buddh_manifest.json"
TRACK_DATA_PATH = "buddh/buddh_track_data.npz"


def test_frozen_weights_integrity():
    """Verify frozen weights have not been altered or retrained."""
    assert os.path.exists(MODEL_PATH), f"Model checkpoint missing at {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert file_sha256 == FROZEN_SHA256, f"Frozen weights SHA256 altered! Expected {FROZEN_SHA256}, got {file_sha256}"


def test_raw_osm_cached_data():
    """Verify raw OSM geographic cache exists and contains required tags and attribution."""
    assert os.path.exists(RAW_OSM_PATH), f"Raw OSM file missing at {RAW_OSM_PATH}"
    with open(RAW_OSM_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "elements" in data, "Malformed OSM JSON"
    relations = [el for el in data["elements"] if el["type"] == "relation"]
    assert len(relations) >= 1, "Missing relation element in OSM data"
    rel = relations[0]
    assert rel["id"] == 2532569, f"Unexpected relation ID: {rel['id']}"
    assert rel["tags"].get("name:en") == "Buddh International Circuit"


def test_manifest_consistency():
    """Verify generated manifest accurately records scale, CRS, and track dimensions."""
    assert os.path.exists(MANIFEST_PATH), f"Manifest missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["coordinate_reference_system"] == "WGS84 / UTM Zone 43N (EPSG:32643)"
    assert m["direction_of_travel"] == "clockwise"
    assert m["reference_track_width_m"] == 12.0
    assert m["sim_track_width_px"] == 70.0
    assert abs(m["world_units_per_meter"] - (70.0 / 12.0)) < 1e-4
    assert m["published_length_m"] == 5138.0
    assert abs(m["resampled_length_m"] - 5131.0) < 5.0
    assert m["length_discrepancy_pct"] < 0.25
    assert len(m["turns"]) == 16


def test_track_interface_contract():
    """Verify BuddhTrack satisfies the exact Gymnasium public interface contract."""
    track = BuddhTrack(data_path=TRACK_DATA_PATH, manifest_path=MANIFEST_PATH)
    assert track.width == 70.0
    assert track.half == 35.0
    assert track.centerline.ndim == 2 and track.centerline.shape[1] == 2

    x0, y0, h0 = track.start_pose()
    assert isinstance(x0, float) and isinstance(y0, float) and isinstance(h0, float)

    # Collision test
    assert track.is_on_track(x0, y0) is True
    assert track.is_on_track(x0 + 500.0, y0 + 500.0) is False

    # Heading & distance
    heading = track.track_heading(x0, y0)
    assert abs(heading - h0) < 1.0
    dist = track.signed_distance(x0, y0)
    assert abs(dist) < 0.1

    # Progress
    prog = track.get_progress(x0, y0)
    assert 0.0 <= prog < 0.01


def test_geometry_invariants():
    """Verify loop closure, constant track width, and absence of boundary foldings."""
    track = BuddhTrack(data_path=TRACK_DATA_PATH, manifest_path=MANIFEST_PATH)

    # Boundary width consistency
    widths = np.linalg.norm(track.left_boundary - track.right_boundary, axis=1)
    assert np.allclose(widths, 70.0, atol=1e-3), f"Width varies! Min: {widths.min()}, Max: {widths.max()}"

    # Minimum radius strictly exceeds track half-width (no boundary foldings)
    min_r = np.min(track._curv_r)
    assert min_r > track.half, f"Boundary folding detected! min_R={min_r} <= half_width={track.half}"


def test_policy_smoke_step():
    """Verify frozen SACActor produces valid bounded actions on initial Buddh state."""
    actor = SACActor(obs_dim=9, action_dim=2)
    actor.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    actor.eval()

    track = BuddhTrack(data_path=TRACK_DATA_PATH, manifest_path=MANIFEST_PATH)
    car = Car()
    car.x, car.y, car.angle = track.start_pose()

    err = (car.angle - track.track_heading(car.x, car.y) + 180.0) % 360.0 - 180.0
    dist = track.signed_distance(car.x, car.y)
    rays = car.cast_rays(track)
    obs = np.array([car.velocity, err, dist, 0.0, *rays], dtype=np.float32)

    with torch.no_grad():
        mean, _ = actor(torch.tensor(obs).unsqueeze(0))
        action = torch.tanh(mean).numpy()[0]

    assert -1.0 <= action[0] <= 1.0, f"Steering action out of bounds: {action[0]}"
    assert -1.0 <= action[1] <= 1.0, f"Throttle action out of bounds: {action[1]}"
    assert action[1] > 0.5, f"Car should accelerate from standstill on straight, got throttle {action[1]}"
