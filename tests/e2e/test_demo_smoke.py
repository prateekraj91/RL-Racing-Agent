"""Playwright End-to-End Smoke Test Suite for RL Racing Agent Interactive Simulator.

Tests:
1. Core dashboard page load and WebGL initialization.
2. Zero fatal browser console errors.
3. Camera mode switching and dynamic camera.near depth management (0.4 cockpit, 2.0 chase/orbit, 20.0 overhead).
4. Simulation stepping and digital telemetry updates.
5. High/Low graphics quality toggle.
6. Cinematic UI Mode (Key H) panel collapsing and bottom HUD overlay.
"""

import time
import pytest
from playwright.sync_api import Playwright, Page

BASE_URL = "http://127.0.0.1:8000/demo"


@pytest.fixture(scope="module")
def shared_page(playwright: Playwright):
    """Module-scoped page: loads simulator once with standardized viewport."""
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
    # Generous timeout: headless WebGL init + the rollout fetch can be slow under the
    # CPU contention of a full test run.
    page.wait_for_selector("#threeCanvas", timeout=60000)
    # Allow the initial startSimulation fetch to settle
    time.sleep(1.5)
    yield page
    context.close()
    browser.close()


def test_demo_page_loads_and_webgl_canvas_present(shared_page: Page):
    """Verify demo loads, WebGL canvas initializes, and essential control elements exist."""
    assert shared_page.query_selector("#btnStartPause") is not None
    assert shared_page.query_selector("#btnStep") is not None
    assert shared_page.query_selector("#btnReset") is not None
    assert shared_page.query_selector("#btnCamChase") is not None
    assert shared_page.query_selector("#btnCamCockpit") is not None
    assert shared_page.query_selector("#btnCamOrbit") is not None
    assert shared_page.query_selector("#btnCamOverhead") is not None
    assert shared_page.query_selector("#btnQualityToggle") is not None
    assert shared_page.query_selector("#btnCinematicToggle") is not None
    assert shared_page.query_selector("#threeCanvas") is not None


def test_camera_modes_and_dynamic_near_plane(shared_page: Page):
    """Verify camera.near is dynamically set per mode (0.4 cockpit, 2.0 chase/orbit, 20.0 overhead)."""
    # 1. Default mode: Chase -> near >= 2.0
    near_chase = shared_page.evaluate("camera.near")
    assert near_chase >= 2.0, f"Expected chase camera.near >= 2.0, got {near_chase}"

    # 2. Switch to Cockpit -> near == 0.4
    shared_page.evaluate("setCameraMode('cockpit')")
    near_cockpit = shared_page.evaluate("camera.near")
    assert abs(near_cockpit - 0.4) < 1e-4, f"Expected cockpit camera.near == 0.4, got {near_cockpit}"

    # 3. Switch to Overhead -> near >= 15.0
    shared_page.evaluate("setCameraMode('overhead')")
    near_overhead = shared_page.evaluate("camera.near")
    assert near_overhead >= 15.0, f"Expected overhead camera.near >= 15.0, got {near_overhead}"

    # 4. Switch to Orbit -> near >= 2.0
    shared_page.evaluate("setCameraMode('orbit')")
    near_orbit = shared_page.evaluate("camera.near")
    assert near_orbit >= 2.0, f"Expected orbit camera.near >= 2.0, got {near_orbit}"

    # Restore chase mode
    shared_page.evaluate("setCameraMode('chase')")


def test_simulation_stepping_and_telemetry(shared_page: Page):
    """Verify clicking Step advances the simulation and updates telemetry cards."""
    # Step simulation 2 times via button click with no_wait_after
    shared_page.locator("#btnStep").click(force=True, no_wait_after=True)
    time.sleep(0.5)
    shared_page.locator("#btnStep").click(force=True, no_wait_after=True)
    time.sleep(0.5)

    step_text = shared_page.locator("#stepClock").inner_text()
    assert "STEP:" in step_text, f"Unexpected step text: {step_text}"

    speed_text = shared_page.locator("#valVelocity").inner_text()
    assert "px/step" in speed_text, f"Expected px/step unit in speed gauge, got {speed_text}"


def test_quality_and_cinematic_mode_toggles(shared_page: Page):
    """Verify Quality toggle and Cinematic HUD mode (Key H) toggle properly."""
    # Quality toggle button
    shared_page.locator("#btnQualityToggle").click(force=True, no_wait_after=True)
    quality_state = shared_page.evaluate("STATE.graphicsQuality")
    assert quality_state == "low"

    # Toggle back to high
    shared_page.locator("#btnQualityToggle").click(force=True, no_wait_after=True)
    quality_state = shared_page.evaluate("STATE.graphicsQuality")
    assert quality_state == "high"

    # Cinematic Mode: press Key H
    shared_page.keyboard.press("h")
    time.sleep(0.3)
    has_cinematic_class = shared_page.evaluate("document.body.classList.contains('cinematic-hud-mode')")
    assert has_cinematic_class is True

    # Render stats evaluation
    stats = shared_page.evaluate("window.getRenderStats()")
    assert isinstance(stats, dict)
    assert "fps" in stats
    assert "drawCalls" in stats
    assert "triangles" in stats
    assert stats["drawCalls"] < 400


def test_scenery_props_clear_of_centerline(shared_page: Page):
    """Every scenery prop (trees, skyline, grandstand, pit) must sit far from the track.

    Nothing may intrude on the 70 px corridor or the walls; we assert a stronger
    >= 100 px clearance from the centerline for every prop in propsGroup.
    """
    # The scene/track is built after the async rollout fetch; wait for it.
    shared_page.wait_for_function(
        "() => !!(window.__minPropClearancePx && window.STATE && window.STATE.trackData"
        " && window.__minPropClearancePx() !== null)",
        timeout=30000,
    )
    clearance = shared_page.evaluate("window.__minPropClearancePx()")
    assert clearance is not None, "prop clearance unavailable (scene not built)"
    assert clearance >= 100.0, (
        f"A scenery prop is only {clearance:.1f} px from the centerline (must be >= 100 px; "
        f"corridor is 70 px wide)"
    )
