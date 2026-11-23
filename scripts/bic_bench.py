"""Run /demo?source=bic&bench=1 headless and print the benchmark JSON."""
import json
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--use-angle=metal", "--enable-gpu-rasterization", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1280, "height": 800}).new_page()
    pg.goto("http://127.0.0.1:8000/demo?source=bic&bench=1", wait_until="domcontentloaded")
    pg.wait_for_function("window.__benchResults", timeout=120000)
    r = pg.evaluate("window.__benchResults"); print(json.dumps(r["results"], indent=1))
    json.dump(r, open("docs/evaluation/bic_playback/bench.json", "w"), indent=1)
    b.close()
