# Multi-stage Dockerfile for RL Racing Agent Frozen SAC Inference Backend
# Base Image: Python 3.11 Debian Slim Bookworm
# Target: Production CPU inference service with minimal attack surface

# ==========================================
# Build Stage: Install dependencies in venv
# ==========================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

COPY requirements-inference.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-inference.txt

# ==========================================
# Final Runtime Stage
# ==========================================
FROM python:3.11-slim-bookworm AS runner

WORKDIR /app

# Install curl for container health check
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create unprivileged system user for process isolation
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy application code and frozen model artifacts
COPY sac/__init__.py ./sac/__init__.py
COPY sac/actor.py ./sac/actor.py
COPY inference/ ./inference/
COPY models/generalist_v1.pth ./models/generalist_v1.pth
COPY configs/generalist_v1.json ./configs/generalist_v1.json

# Enforce secure file permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Runtime environment configuration
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MODEL_PATH="models/generalist_v1.pth" \
    CONFIG_PATH="configs/generalist_v1.json" \
    DEVICE="cpu"

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn single-worker production server
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
