# FastAPI Inference Backend & Docker Deployment

## 1. Overview & Architecture

This document specifies the production inference architecture and containerized deployment for the frozen `generalist-v1.0` Soft Actor-Critic (SAC) autonomous racing agent.

```
+-------------------------------------------------------------------------------+
|                             Client / Telemetry Feed                           |
+-------------------------------------------------------------------------------+
                                      |
                                      | HTTP POST /predict (JSON)
                                      v
+-------------------------------------------------------------------------------+
|                   FastAPI Inference Backend (Uvicorn / Starlette)             |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   | Input Validation (inference.schemas.ObservationRequest)               |   |
|   | - Exact 9-element feature vector                                      |   |
|   | - Rejection of NaN, Infinity, null, and non-numeric types             |   |
|   | - Value bounds validation: [-1000.0, 1000.0]                          |   |
|   +-----------------------------------------------------------------------+   |
|                                     |                                         |
|                                     v                                         |
|   +-----------------------------------------------------------------------+   |
|   | Synchronous Execution Worker (anyio.to_thread worker pool)            |   |
|   | - Prevents event-loop blocking during PyTorch operations              |   |
|   | - Sub-millisecond CPU execution (~0.034 ms raw forward pass)          |   |
|   +-----------------------------------------------------------------------+   |
|                                     |                                         |
|                                     v                                         |
|   +-----------------------------------------------------------------------+   |
|   | Frozen SACActor (sac.actor.SACActor)                                  |   |
|   | - Weights: models/generalist_v1.pth (weights_only=True)               |   |
|   | - Parameters: 18,308 weights (eval mode, requires_grad=False)         |   |
|   | - Evaluation action: a = tanh(mean) under torch.inference_mode()      |   |
|   | - NO critics, optimizers, or replay buffers loaded                    |   |
|   +-----------------------------------------------------------------------+   |
|                                     |                                         |
|                                     v                                         |
|   +-----------------------------------------------------------------------+   |
|   | Action Enrichment & Response Serialization                            |   |
|   | - Raw continuous action [steering, longitudinal] in [-1.0, 1.0]       |   |
|   | - Steering wheel angle: steering * 30.0 degrees                       |   |
|   | - Action mode: "throttle" (> 0), "deceleration" (< 0), "coasting" (0) |   |
|   | - Wall-clock latency measurement (ms)                                 |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
```

### 1.1 Model & Configuration Integrity
The service strictly validates the cryptographic SHA-256 hashes of the frozen weights and configuration on disk during the pre-flight lifespan hook before serving requests:

| Asset | Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Model Checkpoint** | `models/generalist_v1.pth` | `0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784` |
| **Configuration** | `configs/generalist_v1.json` | `4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1` |

### 1.2 Parameter Count Analysis (18,308 vs 18,310)
During loading, the service verifies that the instantiated `SACActor` network contains exactly **18,308** trainable parameters:

$$\begin{aligned}
\text{Linear}(9 \to 128): &\quad 9 \times 128 + 128 = 1,280 \\
\text{Linear}(128 \to 128): &\quad 128 \times 128 + 128 = 16,512 \\
\text{Mean Layer}(128 \to 2): &\quad 128 \times 2 + 2 = 258 \\
\text{Log-Std Layer}(128 \to 2): &\quad 128 \times 2 + 2 = 258 \\
\mathbf{Total\ Instantiated\ Actor\ Parameters}: &\quad 1,280 + 16,512 + 258 + 258 = \mathbf{18,308}
\end{aligned}$$

The configuration file `configs/generalist_v1.json` historically records `"parameter_count": 18310`. The discrepancy of $+2$ parameters arises from the training session metadata calculation, which counted `log_alpha` (a 1-element scalar temperature tensor) along with an auxiliary scalar tracked by the training wrapper. In compliance with the frozen model integrity policy, the configuration file is never modified; the service asserts the exact structural 18,308 parameters of the actor architecture.

### 1.3 Minimal Component Isolation
To ensure minimal memory footprint, rapid startup, and zero training-side attack surface:
- The inference loader imports `SACActor` directly from `sac.actor`.
- `SACAgent` is **never** instantiated.
- Critic networks (`SACCritic`), optimizers (`Adam`), and replay buffers (`ReplayBuffer`) are **not** loaded or imported into memory.

---

## 2. API Specifications

### 2.1 Health & Readiness Check
- **Endpoint**: `GET /health`
- **Description**: Verifies service readiness and cryptographic SHA-256 hashes of the loaded model.
- **Status Codes**:
  - `200 OK`: Model is loaded, verified, warmed up, and ready.
  - `503 Service Unavailable`: Model failed verification or is still initializing.

**Sample Request**:
```bash
curl -s http://localhost:8000/health
```

**Sample Response**:
```json
{
  "status": "ok",
  "ready": true,
  "model_name": "generalist_v1",
  "version": "1.0.0",
  "weights_sha256": "0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784",
  "config_sha256": "4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1",
  "device": "cpu",
  "obs_dim": 9,
  "action_dim": 2,
  "parameter_count": 18308
}
```

---

### 2.2 Single Action Prediction
- **Endpoint**: `POST /predict`
- **Description**: Predicts continuous steering and longitudinal action commands for a single 9-element observation vector.
- **Status Codes**:
  - `200 OK`: Successful inference.
  - `422 Unprocessable Entity`: Malformed request, wrong vector length, NaN/Inf, or value out of bounds.
  - `503 Service Unavailable`: Actor not ready.

**Sample Request**:
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"observation": [3.23, -12.4, -0.7, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0]}'
```

**Sample Response**:
```json
{
  "action": [0.988986, 0.966026],
  "steering": 0.988986,
  "throttle": 0.966026,
  "steering_angle_deg": 29.6696,
  "action_mode": "throttle",
  "inference_latency_ms": 0.1245
}
```

---

### 2.3 Vectorized Batch Prediction
- **Endpoint**: `POST /predict/batch`
- **Description**: Performs high-throughput vectorized forward passes for a list of observation vectors, strictly preserving order.
- **Status Codes**:
  - `200 OK`: Successful batch prediction.
  - `422 Unprocessable Entity`: Validation failure on any observation in the batch.

**Sample Request**:
```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [
      [3.23, -12.4, -0.7, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0],
      [3.80, 0.0, 0.0, 0.0, 100.0, 150.0, 200.0, 150.0, 100.0]
    ]
  }'
```

**Sample Response**:
```json
{
  "actions": [
    {
      "action": [0.988986, 0.966026],
      "steering": 0.988986,
      "throttle": 0.966026,
      "steering_angle_deg": 29.6696,
      "action_mode": "throttle",
      "inference_latency_ms": null
    },
    {
      "action": [-0.015241, 0.998412],
      "steering": -0.015241,
      "throttle": 0.998412,
      "steering_angle_deg": -0.4572,
      "action_mode": "throttle",
      "inference_latency_ms": null
    }
  ],
  "batch_size": 2,
  "total_latency_ms": 0.1842
}
```

---

## 3. Observation Contract & Telemetry Validation

The observation vector accepted by `POST /predict` and `POST /predict/batch` consists of exactly 9 floating-point values corresponding to the state vector emitted by `RacingEnv`:

| Index | Feature Key | Physical Meaning | Kinematic / Typical Range | Environment Bounds | Units & Representation |
| :---: | :--- | :--- | :---: | :---: | :--- |
| `0` | `velocity` | Longitudinal vehicle speed | $[-2.0, 4.0]$ | $[-1000.0, 1000.0]$ | Pixels / step (negative = reverse/braking) |
| `1` | `err` | Heading error relative to track tangent | $[-180.0, 180.0]$ | $[-1000.0, 1000.0]$ | Degrees |
| `2` | `dist` | Signed lateral Euclidean distance to centerline | Track-dependent (nominal $\pm \frac{W}{2}$) | $[-1000.0, 1000.0]$ | Raw pixels ($+$ = right of center, $-$ = left) |
| `3` | `slip` | Tire sideslip angle ($\theta_v - \theta_{\text{car}}$) | $[-180.0, 180.0]$ | $[-1000.0, 1000.0]$ | Degrees (wrapped $\pm 180^\circ$) |
| `4` | `ray_-90` | Left $90^\circ$ perpendicular ray-cast | $[0.0, 200.0]$ | $[-1000.0, 1000.0]$ | Raw pixels (unnormalized) |
| `5` | `ray_-45` | Left $45^\circ$ diagonal ray-cast | $[0.0, 200.0]$ | $[-1000.0, 1000.0]$ | Raw pixels (unnormalized) |
| `6` | `ray_0` | Forward $0^\circ$ centerline ray-cast | $[0.0, 200.0]$ | $[-1000.0, 1000.0]$ | Raw pixels (unnormalized) |
| `7` | `ray_45` | Right $45^\circ$ diagonal ray-cast | $[0.0, 200.0]$ | $[-1000.0, 1000.0]$ | Raw pixels (unnormalized) |
| `8` | `ray_90` | Right $90^\circ$ perpendicular ray-cast | $[0.0, 200.0]$ | $[-1000.0, 1000.0]$ | Raw pixels (unnormalized) |

### 3.1 Clarifications on Historical Configuration Discrepancies
During runtime code inspection of `env/environment.py` and `env/car.py`, several historical metadata discrepancies in `configs/generalist_v1.json` were identified and documented:
1. **Velocity Range**: `Car.update()` clamps forward/reverse velocity with:
   $$\text{velocity} \in \left[-\frac{\text{max\_speed}}{2}, \text{max\_speed}\right] = [-2.0, 4.0]\ \text{px/step}$$
   When decelerating or reversing, velocity is negative down to $-2.0\ \text{px/step}$.
2. **Slip Angle Units**: In `RacingEnv.step()`, sideslip is computed as:
   $$\theta_v = \text{math.degrees}(\text{math.atan2}(-v_y, v_x)), \quad \text{slip} = (\theta_v - \theta_{\text{car}} + 180^\circ) \pmod{360^\circ} - 180^\circ$$
   `slip` is an angle in **degrees** spanning $[-180.0^\circ, 180.0^\circ]$, **not** a dimensionless or normalized scalar in $[-1, 1]$.
3. **Signed Distance Scale & Track Widths**: Signed distance is the signed Euclidean distance in raw pixels from vehicle coordinates $(x, y)$ to the nearest centerline projection. Track widths vary across standard tracks ($W = 70\text{ px}$), domain randomization ($W \in [20, 70]\text{ px}$), and wide curriculum tiers; off-track excursions also produce larger offsets. The Gym `observation_space` specifies box bounds $[-1000.0, 1000.0]$.
4. **Ray Sensors**: All five rangefinder beams return raw Euclidean pixel distances in $[0.0, 200.0]$ (with ray-cast step size 4 pixels). They are strictly **unnormalized**.

### 3.2 Validation Rules
- **Vector Length**: Must equal 9 exactly (`len(v) == 9`).
- **Finite Numbers**: Non-finite floats (`NaN`, `Infinity`, `-Infinity`, or `null`) are rejected with `422 Unprocessable Entity`.
- **Environment Bounds**: Values must fall within the Gym `RacingEnv` observation box `[-1000.0, 1000.0]`. Valid observations from `RacingEnv` are guaranteed never to be falsely rejected.

---

## 4. Local Installation & Execution

### 4.1 Dependency Installation
Install the dedicated, lightweight inference dependencies:

```bash
pip install -r requirements-inference.txt
```

### 4.2 Start Local Server
Run Uvicorn with single worker to bind to port 8000:

```bash
PYTHONPATH=. uvicorn inference.app:app --host 0.0.0.0 --port 8000
```

### 4.3 Running API Tests
Run the test suite verifying requirements A through M:

```bash
PYTHONPATH=. pytest tests/test_inference_api.py -v
```

---

## 5. Docker Deployment

### 5.1 Dockerfile Architecture
The production container uses a hardened multi-stage build:
- **Base**: `python:3.11-slim-bookworm`.
- **Builder Stage**: Installs CPU-only PyTorch from `https://download.pytorch.org/whl/cpu` into `/opt/venv`, avoiding GPU libraries.
- **Runner Stage**:
  - Copies `/opt/venv` to `/opt/venv`.
  - Creates unprivileged user `appuser` (UID/GID isolated, non-login shell `/sbin/nologin`).
  - Sets read-only permissions on frozen weights and code.
  - Configures container health check via `curl -f http://localhost:8000/health`.

### 5.2 Build Container Image
```bash
docker build -t rl-racing-agent:v1 .
```

### 5.3 Run Container
```bash
docker run -d \
  --name rl-racing-service \
  -p 8000:8000 \
  --restart unless-stopped \
  rl-racing-agent:v1
```

### 5.4 Check Container Status & Health
```bash
docker ps --filter "name=rl-racing-service"
docker logs rl-racing-service
curl -s http://localhost:8000/health
```

### 5.5 Stop and Clean Up
```bash
docker stop rl-racing-service && docker rm rl-racing-service
```
