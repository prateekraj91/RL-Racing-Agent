"""FastAPI inference service for frozen generalist-v1.0 SAC racing policy.

Provides synchronous endpoints executed on Starlette worker threadpool
to deliver low-latency CPU inference without event-loop blocking.
"""

from contextlib import asynccontextmanager
import logging
import math
import os
from typing import Tuple

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sac.actor import SACActor
from inference.model_loader import (
    load_inference_actor,
    predict_action,
    predict_batch,
)
from inference.schemas import (
    ActionResponse,
    BatchActionResponse,
    BatchObservationRequest,
    HealthResponse,
    ObservationRequest,
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
