"""Capture BIC recorded-playback screenshots + bench numbers (Playwright, local server on :8000).
Usage: PYTHONPATH=. venv/bin/python scripts/bic_playback_capture.py [--bench]
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("docs/evaluation/bic_playback"); OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8000/demo?source=bic"

def settle(page, ms=2500):
    page.wait_for_timeout(ms)

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--use-angle=metal", "--enable-gpu-rasterization", "--ignore-gpu-blocklist"])
    page = b.new_context(viewport={"width": 1280, "height": 800}).new_page()
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_function("window.STATE && STATE.playback.rolloutData && STATE.playback.maxStep>0", timeout=90000)
    settle(page, 4000)
    idx = page.evaluate("""() => { const m=window.__recordedMeta, st=STATE.playback.rolloutData.steps, o={};
      for (const k of [1,6,8]) { const t=m.turns.find(q=>q.n===k); let b=0,bd=1e9;
        st.forEach((s,i)=>{const d=Math.hypot(s.vehicle.x-t.x,s.vehicle.y-t.y); if(d<bd){bd=d;b=i}}); o[k]=b;} return o; }""")
    print("apex step idx", idx)
    def go(i, mode):
        page.evaluate(f"STATE.playback.isPlaying=false; STATE.playback.playhead={i}; setCameraMode('{mode}'); renderPlaybackFrame({i},0.016)")
    go(idx["1"] - 12, "chase"); settle(page, 3500); page.screenshot(path=str(OUT / "chase_T1.png"))
    go(idx["8"], "orbit"); settle(page, 3500); page.screenshot(path=str(OUT / "orbit_T8.png"))
    go(idx["6"], "overhead"); settle(page, 3500); page.screenshot(path=str(OUT / "overhead_full_circuit.png"))
    go(0, "chase"); settle(page, 3000); page.screenshot(path=str(OUT / "start_straight_grandstand.png"))
    page.evaluate("STATE.playback.playhead=STATE.playback.maxStep-1; STATE.playback.isPlaying=true; STATE.playback.speed=2")
    page.wait_for_selector("#canvasOverlay", state="visible", timeout=20000); settle(page, 1500)
    page.screenshot(path=str(OUT / "post_lap.png"))
    print(json.dumps(page.evaluate("window.__recordedPostLapStats")))
    b.close()
