# RL Racing Lab

**Live demo: https://huggingface.co/spaces/HF_USER/rl-racing-lab** *(Hugging Face Space, free CPU; may take a moment to wake)*

- A frozen SAC policy (`generalist_v1`, 9-dim observation → `[steering, throttle]`) does **path-tracking on real Buddh International Circuit geometry** (OpenStreetMap centerline, 12 m width) — kinematic bicycle model, deterministic inference.
- An **independent engineering audit** corrected several reported figures and documents what the simulator can and cannot claim: [`docs/evaluation/BIC_INDEPENDENT_AUDIT.md`](docs/evaluation/BIC_INDEPENDENT_AUDIT.md).
- **Grip finding:** with a lateral-grip limit of 0.8–1.2 G the policy leaves the track at the first corner at full speed and never brakes (0% braking in every run): [`docs/evaluation/GRIP_LIMIT_EXPERIMENT.md`](docs/evaluation/GRIP_LIMIT_EXPERIMENT.md). Not a claim about real cars — kinematic sim, assumed 12 m = 70 px and 60 steps/s.

`/demo` opens on a recorded BIC playback (chase cam, no compute). A selector switches to the grip-experiment runs (clearly labelled *EXPERIMENT*) and to a live standard-track rollout from the frozen policy. All telemetry is raw simulator output (px, px/step, degrees).

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-inference.txt
PYTHONPATH=. uvicorn inference.app:app --port 8000      # then open http://localhost:8000/demo
```

Docker (CPU, same image as the Space):

```bash
docker build -t rl-racing-lab .
docker run --rm -p 7860:7860 rl-racing-lab               # http://localhost:7860/demo , health: /health
```

Optional Gemini commentary is **off by default**; the app needs no API keys. Public hardening: `/simulation/rollout` results are cached by request (deterministic), uncached rollouts are rate-limited per IP (`ROLLOUT_RATE_LIMIT_PER_MIN`, default 20) and `max_steps` ≤ 1000.

## Tests

```bash
PYTHONPATH=. pytest -q
```

Some tests need local (git-ignored) training run directories under `runs/`.

## Reports
- BIC audit: [`docs/evaluation/BIC_INDEPENDENT_AUDIT.md`](docs/evaluation/BIC_INDEPENDENT_AUDIT.md)
- Grip-limit experiment: [`docs/evaluation/GRIP_LIMIT_EXPERIMENT.md`](docs/evaluation/GRIP_LIMIT_EXPERIMENT.md)
- Model card: [`docs/GENERALIST_MODEL_CARD.md`](docs/GENERALIST_MODEL_CARD.md)

Track data © OpenStreetMap contributors, ODbL 1.0.
