"""Recorded BIC playback: data fidelity + endpoint (read-only, no policy)."""
import csv
import hashlib
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from inference.recorded import CSV_PATH, NPZ_PATH, load_bic_recorded


def test_recorded_matches_csv_and_track():
    d = load_bic_recorded()
    assert d["meta"]["steps"] == 7521
    for f, h in d["meta"]["files"].items():
        assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h
    with open(CSV_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 7521
    assert d["cols"]["car_x"][1234] == float(rows[1234]["car_x"])  # verbatim, not recomputed
    assert d["cols"]["throttle_action"][5000] == float(rows[5000]["throttle_action"])
    # CSV positions lie on the evaluation centerline (max 1.003 m; inside the 35 px half-width)
    assert d["meta"]["alignment"]["max_px"] < 6.0
    assert len(d["track"]["centerline"]) == len(np.load(NPZ_PATH)["centerline_sim"])


def test_recorded_endpoint_is_get_only_and_reports_steps():
    from inference.app import app
    c = TestClient(app)
    r = c.get("/recorded/bic")
    assert r.status_code == 200 and r.json()["meta"]["steps"] == 7521
    assert c.post("/recorded/bic").status_code == 405
