"""FastAPI inference service for frozen generalist-v1.0 SAC racing policy.

Provides synchronous endpoints executed on Starlette worker threadpool
to deliver low-latency CPU inference without event-loop blocking.
"""

from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from sac.actor import SACActor
from inference.model_loader import (
    load_inference_actor,
    predict_action,
    predict_batch,
)
from inference.simulation import (
    TRACK_PRESETS,
    simulation_manager,
)
from inference.schemas import (
    ActionResponse,
    BatchActionResponse,
    BatchObservationRequest,
    HealthResponse,
    ObservationRequest,
    SimulationResetRequest,
    SimulationRolloutRequest,
    SimulationStartRequest,
    SimulationStepRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load and warm up the frozen actor on startup."""
    model_path = os.getenv("MODEL_PATH", "models/generalist_v1.pth")
    config_path = os.getenv("CONFIG_PATH", "configs/generalist_v1.json")
    device = os.getenv("DEVICE", "cpu")

    logger.info("Initializing inference service (device=%s, model=%s)...", device, model_path)
    try:
        actor, metadata = load_inference_actor(
            model_path=model_path,
            config_path=config_path,
            device=device,
        )
        app.state.actor = actor
        app.state.metadata = metadata
        app.state.ready = True

        # Initialize simulation session
        try:
            simulation_manager.start(preset_id="standard_medium", track_seed=1001)
            logger.info("Simulation session pre-initialized successfully.")
        except Exception as sim_err:
            logger.warning("Could not pre-initialize simulation: %s", sim_err)

        logger.info(
            "Model generalist-v1.0 successfully loaded and ready for inference (%d params).",
            metadata["parameter_count"],
        )
    except Exception as exc:
        logger.exception("Failed to initialize frozen actor checkpoint: %s", exc)
        app.state.actor = None
        app.state.metadata = None
        app.state.ready = False
        raise exc

    yield

    logger.info("Shutting down inference service.")
    app.state.actor = None
    app.state.ready = False


app = FastAPI(
    title="RL Racing Agent Inference Service",
    description=(
        "Production-grade, CPU-optimized FastAPI inference backend for the frozen "
        "generalist-v1.0 Soft Actor-Critic (SAC) autonomous racing policy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Sanitize validation errors to ensure non-compliant JSON floats (NaN/Inf) serialize cleanly."""
    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return "NaN"
            if math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            return obj
        elif isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    raw_errors = jsonable_encoder(exc.errors())
    sanitized_errors = _sanitize(raw_errors)
    return JSONResponse(
        status_code=422,
        content={"detail": sanitized_errors},
    )


def get_actor_and_metadata(request: Request) -> Tuple[SACActor, dict]:
    """Dependency provider for the initialized SACActor and its metadata."""
    actor = getattr(request.app.state, "actor", None)
    metadata = getattr(request.app.state, "metadata", None)
    ready = getattr(request.app.state, "ready", False)

    if not ready or actor is None or metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not initialized or ready.",
        )
    return actor, metadata


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and model metadata",
    description="Returns service readiness and cryptographic SHA-256 verification details.",
)
def health_check(request: Request) -> HealthResponse:
    """Check health and verify loaded model parameters."""
    actor = getattr(request.app.state, "actor", None)
    metadata = getattr(request.app.state, "metadata", None)
    ready = getattr(request.app.state, "ready", False)

    if not ready or actor is None or metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unready: frozen actor checkpoint not loaded.",
        )

    return HealthResponse(
        status="ok",
        ready=True,
        model_name=metadata.get("model_name", "generalist-v1.0"),
        version=metadata.get("version", "1.0.0"),
        weights_sha256=metadata.get("weights_sha256", ""),
        config_sha256=metadata.get("config_sha256", ""),
        device=metadata.get("device", "cpu"),
        obs_dim=metadata.get("obs_dim", 9),
        action_dim=metadata.get("action_dim", 2),
        parameter_count=metadata.get("parameter_count", 18308),
    )


@app.post(
    "/predict",
    response_model=ActionResponse,
    summary="Predict action for single observation",
    description="Synchronous endpoint executed on Starlette threadpool for bounded latency.",
)
def predict(
    payload: ObservationRequest,
    actor_bundle: Tuple[SACActor, dict] = Depends(get_actor_and_metadata),
) -> ActionResponse:
    """Predict continuous actions [steering, longitudinal] for an observation vector."""
    actor, metadata = actor_bundle
    device = metadata.get("device", "cpu")

    action, latency_ms = predict_action(
        actor=actor,
        observation=payload.observation,
        device=device,
    )
    return ActionResponse.from_raw_action(action, latency_ms=latency_ms)


@app.post(
    "/predict/batch",
    response_model=BatchActionResponse,
    summary="Predict actions for batch of observations",
    description="Synchronous vectorized forward pass preserving input observation order.",
)
def predict_batch_endpoint(
    payload: BatchObservationRequest,
    actor_bundle: Tuple[SACActor, dict] = Depends(get_actor_and_metadata),
) -> BatchActionResponse:
    """Vectorized action prediction for batch observations."""
    actor, metadata = actor_bundle
    device = metadata.get("device", "cpu")

    actions_raw, latency_ms = predict_batch(
        actor=actor,
        observations=payload.observations,
        device=device,
    )
    action_responses = [
        ActionResponse.from_raw_action(act) for act in actions_raw
    ]
    return BatchActionResponse(
        actions=action_responses,
        batch_size=len(action_responses),
        total_latency_ms=round(latency_ms, 4),
    )


# ==============================================================================
# Stateful Simulation Endpoints
# ==============================================================================

@app.get(
    "/simulation/tracks",
    summary="List supported track presets",
    description="Returns available track presets with recommended evaluation seeds.",
)
def get_simulation_tracks() -> list:
    """Return catalog of available track presets."""
    return list(TRACK_PRESETS.values())


@app.post(
    "/simulation/start",
    summary="Start new simulation session",
    description="Resets RacingEnv with selected track preset and seed, initializes vehicle.",
)
def start_simulation_endpoint(
    payload: SimulationStartRequest,
    actor_bundle: Tuple[SACActor, dict] = Depends(get_actor_and_metadata),
) -> dict:
    """Initialize a fresh simulation run."""
    return simulation_manager.start(
        preset_id=payload.preset_id,
        track_seed=payload.track_seed,
        steer_penalty_coeff=payload.steer_penalty_coeff,
    )


@app.post(
    "/simulation/step",
    summary="Advance simulation by bounded steps",
    description="Runs frozen SACActor deterministically, steps RacingEnv, records telemetry and saliency.",
)
def step_simulation_endpoint(
    payload: SimulationStepRequest,
    actor_bundle: Tuple[SACActor, dict] = Depends(get_actor_and_metadata),
) -> dict:
    """Step active simulation using the loaded frozen actor."""
    actor, _ = actor_bundle
    gemini_key = os.getenv("GEMINI_API_KEY")
    return simulation_manager.step(
        actor=actor,
        steps=payload.steps,
        calculate_saliency=payload.calculate_saliency,
        commentary_mode=payload.commentary_mode,
        gemini_api_key=gemini_key,
    )


# ==============================================================================
# Public-deployment hardening (additive): rollout cache + per-IP rate limit
# ==============================================================================
import hashlib as _hashlib
import json as _json
import threading as _threading
import time as _time
from collections import OrderedDict as _OrderedDict, deque as _deque

ROLLOUT_CACHE_DIR = Path(os.getenv("ROLLOUT_CACHE_DIR", "/tmp/rollout_cache"))
ROLLOUT_CACHE_MAX = int(os.getenv("ROLLOUT_CACHE_MAX", "64"))
RATE_LIMIT_PER_MIN = int(os.getenv("ROLLOUT_RATE_LIMIT_PER_MIN", "20"))  # uncached rollouts / IP / minute
_rollout_mem: "_OrderedDict[str, dict]" = _OrderedDict()
_rollout_lock = _threading.Lock()
_rate_hits: dict = {}


def _rollout_key(payload) -> str:
    raw = _json.dumps(payload.model_dump(), sort_keys=True, default=str)
    return _hashlib.sha256(raw.encode()).hexdigest()


def _rollout_cache_get(key: str):
    with _rollout_lock:
        if key in _rollout_mem:
            _rollout_mem.move_to_end(key)
            return _rollout_mem[key]
    f = ROLLOUT_CACHE_DIR / f"{key}.json"
    try:
        if f.is_file():
            data = _json.loads(f.read_text())
            _rollout_cache_put(key, data, write_disk=False)
            return data
    except Exception:
        pass
    return None


def _rollout_cache_put(key: str, data: dict, write_disk: bool = True) -> None:
    with _rollout_lock:
        _rollout_mem[key] = data
        _rollout_mem.move_to_end(key)
        while len(_rollout_mem) > ROLLOUT_CACHE_MAX:
            _rollout_mem.popitem(last=False)
    if write_disk:
        try:
            ROLLOUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = ROLLOUT_CACHE_DIR / f"{key}.tmp"
            tmp.write_text(_json.dumps(data))
            tmp.replace(ROLLOUT_CACHE_DIR / f"{key}.json")
        except Exception:
            pass  # disk cache is best-effort


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit_or_429(request: Request) -> None:
    if RATE_LIMIT_PER_MIN <= 0:
        return
    ip, now = _client_ip(request), _time.monotonic()
    with _rollout_lock:
        q = _rate_hits.setdefault(ip, _deque())
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            retry = max(1, int(60 - (now - q[0])))
            raise HTTPException(status_code=429, detail="Rate limit exceeded for /simulation/rollout",
                                headers={"Retry-After": str(retry)})
        q.append(now)


@app.post(
    "/simulation/rollout",
    summary="Execute full deterministic simulation rollout",
    description=(
        "Executes complete episode server-side using frozen SACActor deterministically, "
        "and returns sequence of step telemetry for smooth client-side playback."
    ),
)
def rollout_simulation_endpoint(
    request: Request,
    payload: SimulationRolloutRequest,
    actor_bundle: Tuple[SACActor, dict] = Depends(get_actor_and_metadata),
) -> dict:
    """Execute complete episode rollout for client-side playback.

    Rollouts are deterministic (frozen weights, mean action), so offline-commentary results are cached
    by their full request key (memory + disk). Only cache MISSES (real compute) count against the
    per-IP rate limit. max_steps is capped at 1000 by the request schema.
    """
    actor, _ = actor_bundle
    use_cache = payload.commentary_mode == "offline"
    key = _rollout_key(payload)
    if use_cache:
        hit = _rollout_cache_get(key)
        if hit is not None:
            return hit
    _rate_limit_or_429(request)
    gemini_key = os.getenv("GEMINI_API_KEY")
    result = simulation_manager.rollout(
        actor=actor,
        preset_id=payload.preset_id,
        track_seed=payload.track_seed,
        max_steps=payload.max_steps,
        steer_penalty_coeff=payload.steer_penalty_coeff,
        saliency_interval=payload.saliency_interval,
        commentary_mode=payload.commentary_mode,
        gemini_api_key=gemini_key,
    )
    if use_cache:
        _rollout_cache_put(key, result)
    return result


@app.post(
    "/simulation/reset",
    summary="Reset active simulation",
    description="Resets vehicle to track start line, clears trajectory and event history.",
)
def reset_simulation_endpoint(
    payload: Optional[SimulationResetRequest] = None,
    actor_bundle: Tuple[SACActor, dict] = Depends(get_actor_and_metadata),
) -> dict:
    """Reset the current simulation session."""
    preset = payload.preset_id if payload and payload.preset_id else simulation_manager.preset_id
    seed = payload.track_seed if payload and payload.track_seed else simulation_manager.track_seed
    return simulation_manager.start(preset_id=preset, track_seed=seed)


@app.get(
    "/simulation/state",
    summary="Get latest simulation state",
    description="Returns current vehicle pose, live rays, telemetry, and recent trajectory.",
)
def get_simulation_state_endpoint(
    actor_bundle: Tuple[SACActor, dict] = Depends(get_actor_and_metadata),
) -> dict:
    """Return latest state of the active simulation session."""
    return simulation_manager.get_state()


@app.get(
    "/simulation/historical-saliency",
    summary="Get historical benchmark saliency reference",
    description="Returns verified per-corner saliency summary from offline analysis.",
)
def get_historical_saliency_endpoint() -> dict:
    """Return historical saliency benchmark summary from disk if available."""
    summary_path = Path("analysis/results/policy_saliency_summary.json")
    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not parse historical saliency summary: %s", exc)
    return {
        "status": "unavailable",
        "message": "Historical saliency summary not available on disk.",
    }


@app.get(
    "/recorded/bic",
    summary="Recorded BIC evaluation run (read-only)",
    description="Serves the saved BIC evaluation CSV + exact track geometry for recorded playback. "
    "Never runs the policy or the environment.",
)
def get_recorded_bic_endpoint(run: str = "") -> dict:
    """Return the saved BIC run verbatim (additive endpoint). Optional ?run=<id> selects a saved
    grip-limit experiment run from runs/grip_eval/ (e.g. bic_g1p0, std_off); default is unchanged."""
    from inference.recorded import load_bic_recorded

    try:
        return load_bic_recorded(run)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown recorded run: {run}")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Recorded BIC data missing: {exc}")


# ==============================================================================
# Static Frontend Delivery
# ==============================================================================

frontend_path = Path("frontend")
if frontend_path.is_dir():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/demo", include_in_schema=False)
    def serve_frontend():
        """Serve the interactive autonomous racing dashboard."""
        index_file = frontend_path / "index.html"
        if index_file.is_file():
            return FileResponse(str(index_file))
        raise HTTPException(
            status_code=404, detail="Frontend index.html not found."
        )
