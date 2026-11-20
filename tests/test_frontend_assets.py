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


def test_static_world_js_delivery(client: TestClient):
    """Verify GET /static/world.js returns 200 with valid JavaScript coordinate module."""
    res = client.get("/static/world.js")
    assert res.status_code == 200
    assert ("javascript" in res.headers["content-type"] or "text/plain" in res.headers["content-type"])
    assert "envToWorld" in res.text
    assert "envHeadingToWorld" in res.text
    assert "envSteerToWorld" in res.text
    assert "PX_TO_WORLD" in res.text


def test_static_pbr_and_vendor_assets_delivery(client: TestClient):
    """Verify HDRI, PBR textures, vendor scripts, and license file are served."""
    # Vendor scripts
    res_rgbe = client.get("/static/vendor/RGBELoader.js")
    assert res_rgbe.status_code == 200
    assert "RGBELoader" in res_rgbe.text

    res_comp = client.get("/static/vendor/EffectComposer.js")
    assert res_comp.status_code == 200
    assert "EffectComposer" in res_comp.text

    # Licenses
    res_lic = client.get("/static/assets/LICENSES.md")
    assert res_lic.status_code == 200
    assert "Poly Haven" in res_lic.text

    # PBR asphalt texture
    res_tex = client.get("/static/assets/textures/asphalt/asphalt_diff_1k.jpg")
    assert res_tex.status_code == 200
    assert len(res_tex.content) > 10000

    # HDRI environment map
    res_hdr = client.get("/static/assets/hdr/kloofendal_48d_partly_cloudy_puresky_2k.hdr")
    assert res_hdr.status_code == 200
    assert len(res_hdr.content) > 100000

    # GLTF and Draco vendor loaders
    res_gltf = client.get("/static/vendor/GLTFLoader.js")
    assert res_gltf.status_code == 200
    assert "GLTFLoader" in res_gltf.text

    res_draco = client.get("/static/vendor/DRACOLoader.js")
    assert res_draco.status_code == 200
    assert "DRACOLoader" in res_draco.text

    # Car 3D model asset
    res_car = client.get("/static/assets/models/car/car.glb")
    assert res_car.status_code == 200
    assert len(res_car.content) > 500000


def test_checkpoint_c_features_and_dom_elements(client: TestClient):
    """Verify Checkpoint C DOM elements, styles, scripts, and CC0 props documentation."""
    # HTML DOM
    res_html = client.get("/demo")
    assert res_html.status_code == 200
    assert 'id="btnCinematicToggle"' in res_html.text
    assert 'id="btnQualityToggle"' in res_html.text
    assert 'id="cinematicBottomHud"' in res_html.text
    assert 'id="hudSpeedVal"' in res_html.text
    assert 'id="hudSteerVal"' in res_html.text
    assert 'id="hudThrottleVal"' in res_html.text

    # CSS styles
    res_css = client.get("/static/styles.css")
    assert res_css.status_code == 200
    assert ".cinematic-hud-mode" in res_css.text
    assert ".cinematic-bottom-hud" in res_css.text
    assert ".pill-quality" in res_css.text

    # JavaScript app
    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "toggleCinematicUIMode" in res_js.text
    assert "updateCinematicBottomHud" in res_js.text
    assert "buildEnvironmentProps" in res_js.text
    assert "computeCenterlineCurvature" in res_js.text
    assert "buildCurvatureBasedCurbs" in res_js.text
    assert "buildBarrierWall" in res_js.text
    assert "getRenderStats" in res_js.text

    # CC0 Environment Props Documentation in LICENSES.md
    res_lic = client.get("/static/assets/LICENSES.md")
    assert res_lic.status_code == 200
    assert "Procedural Environment Props Architecture" in res_lic.text
    assert "THREE.InstancedMesh" in res_lic.text
    assert "Tecpro" in res_lic.text
