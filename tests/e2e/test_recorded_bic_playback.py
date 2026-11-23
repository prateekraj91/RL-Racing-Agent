"""E2E: BIC recorded playback never calls the policy / simulation endpoints."""
import re
import pytest
from playwright.sync_api import Playwright

URL = "http://127.0.0.1:8000/demo?source=bic"
FORBIDDEN = re.compile(r"/(predict|step|rollout)|/simulation/|/predict_batch")


@pytest.fixture(scope="module")
def page_and_requests(playwright: Playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_context(viewport={"width": 1280, "height": 800}).new_page()
    reqs = []
    page.on("request", lambda r: reqs.append((r.method, r.url)))
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_function("window.STATE && STATE.playback.rolloutData && STATE.playback.maxStep > 0", timeout=90000)
    yield page, reqs
    browser.close()


def test_no_policy_or_sim_requests_during_playback(page_and_requests):
    page, reqs = page_and_requests
    # play at 8x, scrub, step, change camera, then run into the post-lap screen
    page.click("button.speed-pill[data-speed='8']")
    page.click("#btnStartPause")
    page.wait_for_timeout(1500)
    page.evaluate("STATE.playback.playhead = 4000")
    page.click("#btnStep"); page.click("#btnStepBack")
    page.click("#btnCamOverhead"); page.click("#btnCamChase")
    page.evaluate("STATE.playback.playhead = STATE.playback.maxStep - 1; STATE.playback.isPlaying = true")
    page.wait_for_selector("#canvasOverlay", state="visible", timeout=20000)
    bad = [(m, u) for m, u in reqs if FORBIDDEN.search(u)]
    assert not bad, f"forbidden requests during recorded playback: {bad}"
    api = [u for m, u in reqs if "/static/" not in u and u.rstrip("/").split("?")[0].endswith(("/demo", "/recorded/bic")) is False and "favicon" not in u]
    assert not api, f"unexpected non-static requests: {api}"
    assert any(u.endswith("/recorded/bic") for _, u in reqs)


def test_badge_steps_and_clearance(page_and_requests):
    page, _ = page_and_requests
    assert "RECORDED PLAYBACK" in page.inner_text("#sourceBadge") and "BIC eval" in page.inner_text("#sourceBadge")
    assert page.evaluate("STATE.playback.maxStep") == 7520
    clr = page.evaluate("window.__minPropClearancePx()")
    assert clr is not None and clr >= 100, clr
    stats = page.evaluate("window.__recordedPostLapStats")
    assert abs(stats["avgLat_m"] - 0.059) < 0.001 and abs(stats["rms_m"] - 0.142) < 0.001
    assert stats["brakingPct"] == 0
