"""Static (HF Static Space) build: complete data, relative paths, and zero API calls."""
import functools
import http.server
import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

from scripts.build_static import RUNS, SEEDS, build


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    return build(tmp_path_factory.mktemp("dist") / "dist")


def test_layout_and_sizes(dist):
    assert (dist / "index.html").is_file() and (dist / "static/static_shim.js").is_file()
    assert (dist / "README.md").read_text().startswith("---\n") and "sdk: static" in (dist / "README.md").read_text()[:200]
    total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    assert total < 200e6  # far below the 1 GB Space limit
    assert not list(dist.rglob("*.pth")), "model weights must not ship in the static build"


def test_html_relative_and_shim_first(dist):
    html = (dist / "index.html").read_text()
    assert '"/static/' not in html
    assert html.index("static_shim.js") < html.index("three.min.js") < html.index("app.js")
    assert "Precomputed deterministic rollout (frozen policy)" in html
    for s in SEEDS:
        assert f'<option value="{s}">' in html


def test_data_files_complete_and_valid(dist):
    for run in RUNS:
        d = json.loads((dist / "data/recorded" / f"{run or 'bic'}.json").read_text())
        assert d["meta"]["steps"] == len(d["cols"]["step"]) > 0
        if run:
            assert d["meta"]["label"].startswith("EXPERIMENT")
    for s in SEEDS:
        d = json.loads((dist / "data/rollouts" / f"standard_medium_{s}.json").read_text())
        assert len(d["steps"]) > 100 and d["track"]


def test_served_over_http_and_no_api_paths_fetched(dist):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(dist))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        paths = ["/", "/static/app.js", "/static/recorded.js", "/static/world.js", "/static/static_shim.js",
                 "/data/recorded/bic.json", "/data/rollouts/standard_medium_1005.json"]
        for p in paths:
            assert urllib.request.urlopen(base + p).status == 200
        # The server has no API: these must 404 (proves the page cannot depend on them)
        for p in ["/recorded/bic", "/simulation/rollout", "/health", "/predict"]:
            with pytest.raises(Exception):
                urllib.request.urlopen(base + p)
    finally:
        srv.shutdown()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_shim_maps_two_calls_and_blocks_all_other_api(dist, tmp_path):
    script = tmp_path / "t.js"
    script.write_text(f"""
const calls=[]; global.window=global; global.location={{href:'http://x/'}}; global.document={{baseURI:'http://x/'}};
window.fetch=async (u)=>{{calls.push(String(u)); return {{ok:true}};}};
require({json.dumps(str(dist / 'static/static_shim.js'))});
(async()=>{{
  await fetch('/recorded/bic'); await fetch('/recorded/bic?run=bic_g1p0');
  await fetch('/simulation/rollout',{{method:'POST',body:JSON.stringify({{preset_id:'standard_medium',track_seed:1004}})}});
  await fetch('static/assets/x.jpg');
  let blocked=0;
  for (const p of ['/predict','/step','/simulation/step','/simulation/start','/simulation/state','/health','/recorded/other'])
    {{ try {{ await fetch(p); }} catch(e) {{ blocked++; }} }}
  console.log(JSON.stringify({{calls,blocked,attempts:window.__apiAttempts.length}}));
}})();
""")
    out = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout)
    assert out["calls"] == ["http://x/data/recorded/bic.json", "http://x/data/recorded/bic_g1p0.json",
                            "http://x/data/rollouts/standard_medium_1004.json", "static/assets/x.jpg"]
    assert out["blocked"] == 7 and out["attempts"] == 7


def test_tour_mode_shipped_and_static_safe(dist):
    html = (dist / "index.html").read_text()
    assert html.index("recorded.js") < html.index("tour.js")
    tour = (dist / "static/tour.js").read_text()
    assert 'get("tour") === "1"' in tour and "Escape" in tour
    # only recorded/precomputed runs: the three sources the tour touches exist in the static data set
    assert (dist / "data/recorded/bic.json").is_file() and (dist / "data/recorded/bic_g1p0.json").is_file()
    assert (dist / "data/rollouts/standard_medium_1001.json").is_file()
    # tour fetches only through the (shimmed) endpoints the app already uses
    import re
    paths = set(re.findall(r'fetch\("(/[^"?]*)', tour))
    assert paths <= {"/recorded/bic", "/simulation/rollout"}
    # timeline is strictly increasing and spans ~90 s
    ts = [int(x) for x in re.findall(r"\{ t: (\d+),", tour)]
    assert ts == sorted(ts) and ts[0] == 0 and 80 <= ts[-1] <= 90
