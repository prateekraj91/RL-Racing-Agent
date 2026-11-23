"""Public-hardening tests: rollout cache (deterministic) + per-IP rate limit + max_steps cap."""
import pytest
from fastapi.testclient import TestClient

import inference.app as appmod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "ROLLOUT_CACHE_DIR", tmp_path)
    appmod._rollout_mem.clear()
    appmod._rate_hits.clear()
    with TestClient(appmod.app) as c:
        yield c


def test_cached_rollout_identical_and_not_rate_counted(client):
    body = {"preset_id": "standard_medium", "track_seed": 1001, "max_steps": 40}
    a = client.post("/simulation/rollout", json=body).json()
    n_before = sum(len(q) for q in appmod._rate_hits.values())
    b = client.post("/simulation/rollout", json=body).json()
    assert a == b
    assert sum(len(q) for q in appmod._rate_hits.values()) == n_before  # hit is free
    assert any(tmp.suffix == ".json" for tmp in appmod.ROLLOUT_CACHE_DIR.iterdir())  # disk copy


def test_rate_limit_429_on_uncached_only(client, monkeypatch):
    monkeypatch.setattr(appmod, "RATE_LIMIT_PER_MIN", 2)
    codes = [client.post("/simulation/rollout", json={"track_seed": s, "max_steps": 5}).status_code
             for s in (1, 2, 3)]
    assert codes == [200, 200, 429]
    # a cached request still succeeds while limited
    assert client.post("/simulation/rollout", json={"track_seed": 1, "max_steps": 5}).status_code == 200


def test_max_steps_capped(client):
    assert client.post("/simulation/rollout", json={"max_steps": 5000}).status_code == 422
