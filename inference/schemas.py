"""Pydantic v2 schemas for the RL Racing Agent Inference Service."""

import math
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ObservationRequest(BaseModel):
    """Single observation input for policy action prediction."""

    observation: List[float] = Field(
        ...,
        description=(
            "9-element raw observation vector: "
            "[velocity, heading_error, signed_dist, slip_angle, "
            "ray_-90, ray_-45, ray_0, ray_45, ray_90]. "
            "velocity is in px/step (kinematic limits [-2.0, 4.0]). "
            "heading_error is in degrees [-180.0, 180.0]. "
            "signed_dist is signed lateral distance from centerline in pixels (positive = right, negative = left). "
            "slip_angle is tire sideslip angle in degrees [-180.0, 180.0] (velocity_angle - car.angle). "
            "ray distances are raw rangefinder distances in pixels [0.0, 200.0] (unnormalized)."
        ),
        json_schema_extra={
            "example": [3.23, -12.4, -0.7, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0]
        },
    )

    @field_validator("observation")
    @classmethod
    def validate_observation(cls, v: List[float]) -> List[float]:
        if len(v) != 9:
            raise ValueError(
                f"Observation vector must contain exactly 9 elements, got {len(v)}"
            )

        for i, val in enumerate(v):
            if val is None or math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Observation element at index {i} must be a finite float, got {val}"
                )
            if not (-1000.0 <= val <= 1000.0):
                raise ValueError(
                    f"Observation element at index {i} ({val}) is outside valid environment bounds [-1000.0, 1000.0]"
                )
        return v


class BatchObservationRequest(BaseModel):
    """Batch of observation inputs for vectorized policy action prediction."""

    observations: List[List[float]] = Field(
        ...,
        min_length=1,
        description=(
            "List of 9-element observation vectors: "
            "[velocity (px/step), heading_error (deg), signed_dist (px), slip_angle (deg), "
            "5x raw pixel ray distances [0, 200]]"
        ),
        json_schema_extra={
            "example": [
                [3.23, -12.4, -0.7, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0],
                [3.50, 0.0, 0.0, 0.0, 120.0, 150.0, 200.0, 150.0, 120.0],
            ]
        },
    )

    @field_validator("observations")
    @classmethod
    def validate_batch(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) == 0:
            raise ValueError("Batch must contain at least 1 observation")

        for b_idx, obs in enumerate(v):
            if len(obs) != 9:
                raise ValueError(
                    f"Batch item {b_idx} must contain exactly 9 elements, got {len(obs)}"
                )
            for i, val in enumerate(obs):
                if val is None or math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"Batch item {b_idx} element at index {i} must be a finite float, got {val}"
                    )
                if not (-1000.0 <= val <= 1000.0):
                    raise ValueError(
                        f"Batch item {b_idx} element at index {i} ({val}) is outside bounds [-1000.0, 1000.0]"
                    )
        return v


class ActionResponse(BaseModel):
    """Predicted action response with derived telemetry semantics."""

    action: List[float] = Field(
        ...,
        description="2-element raw continuous action [steering, longitudinal] in [-1.0, 1.0]",
    )
    steering: float = Field(
        ...,
        description="Normalized continuous steering command in [-1.0, 1.0]",
    )
    throttle: float = Field(
        ...,
        description="Normalized continuous longitudinal command in [-1.0, 1.0]",
    )
    steering_angle_deg: float = Field(
        ...,
        description="Steering angle in degrees [-30.0, 30.0] (steering * 30.0)",
    )
    action_mode: Literal["throttle", "deceleration", "coasting"] = Field(
        ...,
        description="Longitudinal action mode: throttle (>0.0), deceleration (<0.0), or coasting (==0.0)",
    )
    inference_latency_ms: Optional[float] = Field(
        None,
        description="Inference execution latency in milliseconds",
    )

    @classmethod
    def from_raw_action(
        cls, action: List[float], latency_ms: Optional[float] = None
    ) -> "ActionResponse":
        """Construct an ActionResponse from a 2-element raw action."""
        steering = float(action[0])
        throttle = float(action[1])
        steering_angle_deg = round(steering * 30.0, 4)

        if throttle > 0.0:
            action_mode: Literal["throttle", "deceleration", "coasting"] = "throttle"
        elif throttle < 0.0:
            action_mode = "deceleration"
        else:
            action_mode = "coasting"

        return cls(
            action=[round(steering, 6), round(throttle, 6)],
            steering=round(steering, 6),
            throttle=round(throttle, 6),
            steering_angle_deg=steering_angle_deg,
            action_mode=action_mode,
            inference_latency_ms=round(latency_ms, 4) if latency_ms is not None else None,
        )


class BatchActionResponse(BaseModel):
    """Batch action response for vectorized inference."""

    actions: List[ActionResponse] = Field(
        ..., description="Ordered list of predicted actions matching input batch"
    )
    batch_size: int = Field(..., description="Number of observations processed")
    total_latency_ms: Optional[float] = Field(
        None, description="Total batch inference latency in milliseconds"
    )


class HealthResponse(BaseModel):
    """Service health and model metadata response."""

    status: str = Field("ok", description="Service status")
    ready: bool = Field(True, description="Whether the model actor is initialized and ready")
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model release version")
    weights_sha256: str = Field(..., description="Cryptographic SHA-256 hash of loaded checkpoint")
    config_sha256: str = Field(..., description="Cryptographic SHA-256 hash of configuration file")
    device: str = Field("cpu", description="Execution device")
    obs_dim: int = Field(9, description="Observation feature dimension")
    action_dim: int = Field(2, description="Action vector dimension")
    parameter_count: int = Field(18308, description="Instantiated SAC actor parameter count")
