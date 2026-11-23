"""Build the STATIC public demo into dist/ (additional build target; the API/Docker app is unchanged).

dist/ layout:  index.html, static/ (frontend + shim), data/recorded/*.json, data/rollouts/*.json, README.md
  - data/recorded/<id>.json : exactly what GET /recorded/bic[?run=<id>] returns (BIC eval + grip runs)
  - data/rollouts/standard_medium_<seed>.json : exactly what POST /simulation/rollout returns, seeds 1001-1005
    (frozen policy, deterministic, offline commentary). Produced here, once; the static site never calls an API.

Usage:  PYTHONPATH=. python scripts/build_static.py [--out dist]
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["ROLLOUT_RATE_LIMIT_PER_MIN"] = "0"
os.environ.setdefault("ROLLOUT_CACHE_DIR", tempfile.mkdtemp(prefix="static_build_cache_"))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FRONTEND = ROOT / "frontend"
SEEDS = [1001, 1002, 1003, 1004, 1005]
RUNS = ["", "bic_g0p8", "bic_g1p0", "bic_g1p2", "std_off", "std_g0p8", "std_g1p0", "std_g1p2"]  # "" = BIC eval (grip OFF)
GITHUB = "https://github.com/prateekraj91/RL-Racing-Agent/blob/main/"
SKIP = {"static", "node_modules", "package-lock.json", "index.html", "__pycache__", ".DS_Store"}


def copy_frontend(out: Path) -> None:
    dst = out / "static"
    for item in FRONTEND.iterdir():
        if item.name in SKIP:
            continue
        (shutil.copytree(item, dst / item.name) if item.is_dir() else (dst.mkdir(exist_ok=True), shutil.copy2(item, dst / item.name)))


def transform(out: Path) -> None:
    html = (FRONTEND / "index.html").read_text()
    html = html.replace('"/static/', '"static/')
    html = html.replace('<script src="static/vendor/three.min.js">', '<script src="static/static_shim.js"></script>\n  <script src="static/vendor/three.min.js">', 1)
    html = html.replace("LIVE — standard-track rollout", "Precomputed deterministic rollout (frozen policy)")
    html = re.sub(r'<select id="trackPreset".*?</select>',
                  '<select id="trackPreset" class="select-input"><option value="standard_medium">Standard Medium (70px) — precomputed</option></select>',
                  html, count=1, flags=re.S)
    seeds = "".join(f'<option value="{s}">{s}</option>' for s in SEEDS)
    html = re.sub(r'<input type="number" id="trackSeed"[^>]*>',
                  f'<select id="trackSeed" class="select-input">{seeds}</select>', html, count=1)
    html = html.replace("Load Track</button>", "Load Rollout</button>", 1)
    (out / "index.html").write_text(html)
    for name in ("app.js", "recorded.js", "world.js"):
        f = out / "static" / name
        f.write_text(f.read_text().replace('"/static/', '"static/'))


def build_data(out: Path) -> None:
    from fastapi.testclient import TestClient
    import inference.app as appmod

    (out / "data" / "recorded").mkdir(parents=True)
    (out / "data" / "rollouts").mkdir(parents=True)
    with TestClient(appmod.app) as c:
        for run in RUNS:
            r = c.get("/recorded/bic", params={"run": run} if run else None)
            r.raise_for_status()
            (out / "data" / "recorded" / f"{run or 'bic'}.json").write_text(json.dumps(r.json(), separators=(",", ":")))
        for seed in SEEDS:
            r = c.post("/simulation/rollout", json={"preset_id": "standard_medium", "track_seed": seed,
                                                    "commentary_mode": "offline", "saliency_interval": 5})
            r.raise_for_status()
            d = r.json()
            assert d["steps"], f"empty rollout seed {seed}"
            (out / "data" / "rollouts" / f"standard_medium_{seed}.json").write_text(json.dumps(d, separators=(",", ":")))


def write_readme(out: Path) -> None:
    body = (ROOT / "README.md").read_text()
    body = re.sub(r"\]\((docs/[^)]+)\)", lambda m: f"]({GITHUB}{m.group(1)})", body)
    front = "---\ntitle: RL Racing Lab\nemoji: 🏎️\ncolorFrom: red\ncolorTo: gray\nsdk: static\npinned: false\n---\n\n"
    (out / "README.md").write_text(front + body)


def build(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copy_frontend(out)
    transform(out)
    build_data(out)
    write_readme(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    o = build(Path(a.out))
    size = sum(f.stat().st_size for f in o.rglob("*") if f.is_file())
    print(f"built {o} ({size/1e6:.1f} MB, {sum(1 for f in o.rglob('*') if f.is_file())} files)")
