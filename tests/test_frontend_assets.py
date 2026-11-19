"""Test suite for static frontend asset delivery and dashboard routes."""

import pytest
from fastapi.testclient import TestClient

from inference.app import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_route_serves_html(client: TestClient):
    """Verify GET / returns 200 OK and serves index.html."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "<!DOCTYPE html>" in res.text
    assert "raceCanvas" in res.text
    assert "btnStartPause" in res.text


def test_demo_route_serves_html(client: TestClient):
    """Verify GET /demo returns 200 OK and serves index.html."""
    res = client.get("/demo")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "AUTONOMOUS RACING SIMULATOR" in res.text


def test_static_css_delivery(client: TestClient):
    """Verify GET /static/styles.css returns 200 with valid CSS."""
    res = client.get("/static/styles.css")
    assert res.status_code == 200
    assert "text/css" in res.headers["content-type"]
    assert "--bg-primary" in res.text


def test_static_js_delivery(client: TestClient):
    """Verify GET /static/app.js returns 200 with valid JavaScript."""
    res = client.get("/static/app.js")
    assert res.status_code == 200
    assert ("javascript" in res.headers["content-type"] or "text/plain" in res.headers["content-type"])
    assert "init" in res.text
    assert "raceCanvas" in res.text
