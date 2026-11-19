# RL Racing Agent: Interactive Simulation Dashboard (V1)

## 1. Overview & Architecture

The Interactive Web Frontend provides a real-time, browser-based motorsport engineering simulator powered by the frozen `generalist-v1.0` Soft Actor-Critic (SAC) autonomous racing policy and the authentic `RacingEnv` physics engine.

```
+---------------------------------------------------------------------------------------+
|                              Browser UI (HTML5 / Canvas)                             |
|                                                                                       |
|   +------------------------------------+  +---------------------------------------+   |
|   | 60 FPS Canvas Rendering Loop       |  | Telemetry & Saliency Gauges           |   |
|   | - Track ribbon & curb boundaries   |  | - Velocity, steering, throttle/brake   |   |
|   | - Steered vehicle body & wheels    |  | - Heading error, signed dist, slip    |   |
|   | - Dynamic trajectory trail         |  | - 5x rangefinder distance meters      |   |
|   | - 5x live lidar ray-cast beams     |  | - Live gradient saliency bar charts   |   |
|   +------------------------------------+  +---------------------------------------+   |
|                                     ^            ^                                    |
|                                     |            |                                    |
|                          JSON State |            | State Updates                      |
|                 (20-30 ticks / sec) |            |                                    |
+-------------------------------------|------------|------------------------------------+
                                      |            |
                         HTTP POST /simulation/step|
                                      v            |
+---------------------------------------------------------------------------------------+
|                    FastAPI Simulation Backend (inference/app.py)                     |
|                                                                                       |
|   +-------------------------------------------------------------------------------+   |
|   | SimulationSession Manager (inference/simulation.py)                           |   |
|   | - Owns authentic RacingEnv instance (env/environment.py)                      |   |
|   | - Manages Track geometry, car state, and trajectory history                   |   |
|   | - Detects driving milestones (corner approach, apex, braking, cruising)      |   |
|   +-------------------------------------------------------------------------------+   |
|                                     |                                                 |
|                   Observation (9,)  v  Action (2,)                                    |
|   +-------------------------------------------------------------------------------+   |
|   | Frozen SACActor (sac/actor.py)                                                |   |
|   | - Weights: models/generalist_v1.pth (18,308 weights, requires_grad=False)     |   |
|   | - Action: a = tanh(mean) under torch.inference_mode()                         |   |
|   | - Online Saliency: torch.autograd.grad(action, obs_tensor)                    |   |
|   +-------------------------------------------------------------------------------+   |
|                                     |                                                 |
|                   Event Telemetry   v  Validated Commentary                           |
|   +-------------------------------------------------------------------------------+   |
|   | Grounded Commentary Engine                                                    |   |
|   | - Offline Mode (Default): Deterministic, telemetry-verified templates        |   |
|   | - Live Mode (Opt-in): Google Gemini 3.8 Flash + GroundingValidator            |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
```

---

## 2. Track Presets & Geometry Extraction

The backend extracts procedural track curves and computes continuous normal vectors from `track.centerline` to generate inner and outer road boundaries for canvas rendering.

| Preset ID | Name | Width | Base Radius | Min Radius | Key Racing Characteristics | Recommended Seeds |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `standard_medium` | Standard Medium Circuit | $70\text{ px}$ | $250\text{ px}$ | $80\text{ px}$ | Balanced benchmark circuit with diverse corner radii | `1001`, `1002`, `1003`, `101` |
| `narrow_corridor` | Narrow Precision Corridor | $40\text{ px}$ | $250\text{ px}$ | $80\text{ px}$ | High-difficulty narrow track punishing lateral deviation | `1001`, `1002`, `1005` |
| `hairpin_challenge`| Hairpin Switchback | $70\text{ px}$ | $250\text{ px}$ | $80\text{ px}$ | Sharp pinch turn (`hairpin_factor = 0.35`) demanding heavy braking | `1001`, `1002`, `1003` |
| `wide_oval` | Wide Flowing Circuit | $100\text{ px}$ | $250\text{ px}$ | $80\text{ px}$ | High-speed sweeping turns with generous margins | `101`, `1001`, `1004` |
| `default_circuit` | Default Environment Circuit| $70\text{ px}$ | $210\text{ px}$ | $70\text{ px}$ | Baseline `RacingEnv` default geometry | `42`, `101`, `1001` |

---

## 3. Real-Time Telemetry & Lidar Sensor Beams

### 3.1 Live Vehicle Telemetry
All metrics displayed in the dashboard originate directly from Python simulation physics:
- **Velocity**: Forward longitudinal speed in pixels/step ($[-2.0, 4.0]$ kinematic bounds).
- **Steering Wheel Angle**: Front wheel angle in degrees ($\theta_{\text{wheel}} = \text{action}[0] \times 30.0^\circ$).
- **Longitudinal Command**: Throttle ($>0$), braking/deceleration ($<0$), or coasting ($=0$).
- **Heading Error**: Road tangent alignment angle in degrees ($[-180.0^\circ, 180.0^\circ]$).
- **Signed Distance**: Lateral Euclidean distance from vehicle coordinates to the centerline projection in raw pixels ($+$ right of center, $-$ left of center).
- **Slip Angle**: Tire sideslip angle in degrees ($\theta_v - \theta_{\text{car}}$, wrapped $\pm 180^\circ$).
- **Lap Progress**: Normalized distance along the closed-loop centerline ($0\%$ to $100\%$).

### 3.2 Five Lidar Sensor Beams
The vehicle casts 5 rangefinder beams at fixed angles relative to the vehicle heading:
$$\text{Angles}: \left[-90^\circ\ (\text{Left}),\ -45^\circ\ (\text{Diag L}),\ 0^\circ\ (\text{Center}),\ +45^\circ\ (\text{Diag R}),\ +90^\circ\ (\text{Right})\right]$$

Ray endpoints are computed using exact trigonometric coordinates matching `env/car.py`:
$$x_{\text{end}} = x_{\text{car}} + \cos(\theta_{\text{car}} + \theta_{\text{ray}}) \cdot d_k, \quad y_{\text{end}} = y_{\text{car}} - \sin(\theta_{\text{car}} + \theta_{\text{ray}}) \cdot d_k$$
Beams dynamically shift color based on proximity:
- **Green** ($> 90\text{ px}$): Clear corridor.
- **Amber** ($45 - 90\text{ px}$): Approaching corner or track wall.
- **Crimson** ($< 45\text{ px}$): Danger / boundary proximity.

---

## 4. Policy Explainability: Input-Gradient Saliency

The simulator calculates real-time input-gradient saliency using the frozen `SACActor`:

$$\mathbf{S}_{\text{steer}} = \left| \frac{\partial a_{\text{steer}}}{\partial \mathbf{s}} \right|, \quad \mathbf{S}_{\text{long}} = \left| \frac{\partial a_{\text{long}}}{\partial \mathbf{s}} \right|$$

- **Safety & Parameter Freeze**: All neural network weights have `requires_grad=False`. Gradients are enabled exclusively on the input observation tensor under `torch.autograd.grad`. No optimizer or weight update is ever performed.
- **Explainability Scope**: Visualized bar charts represent mathematical gradient sensitivity across the 9 observation inputs. It indicates which sensor beams most heavily dictate instantaneous throttle and steering decisions.
- **Disclaimer**: Saliency is explicitly labeled as mathematical sensitivity, **not** human visual attention or gaze.

---

## 5. Grounded Telemetry Commentary

The dashboard features an integrated racing commentary stream operating in two distinct modes:

### 5.1 Mode A: Offline Grounded Mode (Default)
- **Zero API Calls**: Does not require an internet connection or API keys.
- **Deterministic & Factual**: Formulates expert driver commentary directly from empirical measurements (e.g. corner radius, speed, steering angle, dominant ray).
- **Source Label**: Clearly tagged as `[OFFLINE GROUNDED]`.

### 5.2 Mode B: Live Gemini 3.8 Flash Mode (Opt-In)
- **Activation**: Enabled via the "Gemini Live" toggle in the control bar.
- **Security**: Requires `GEMINI_API_KEY` on the Python server backend. The API key is **never** sent to the client browser.
- **Strict Verification (`GroundingValidator`)**: Every generated sentence is validated against measured ground-truth telemetry before display. Non-existent sensors, imaginary competitors, fake corner labels, or hallucinated units (e.g. km/h, mph) are rejected.
- **Rate-Limiting**: Capped at a maximum of 5 requests per simulation run.
- **Safe Fallback**: If the key is missing, quota is exceeded, or validation fails, the system automatically falls back to offline commentary tagged as `[OFFLINE FALLBACK]`.

---

## 6. Local Execution Guide

### 6.1 Install Dependencies
```bash
pip install -r requirements-inference.txt
```

### 6.2 Start the Application
Run Uvicorn bound to port 8000:
```bash
PYTHONPATH=. uvicorn inference.app:app --host 0.0.0.0 --port 8000
```

### 6.3 Access the Web Dashboard
Open your web browser and navigate to:
**[http://127.0.0.1:8000/demo](http://127.0.0.1:8000/demo)** (or [http://127.0.0.1:8000/](http://127.0.0.1:8000/))

Interactive API Documentation:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 7. Docker Deployment

### 7.1 Build Container Image
```bash
docker build -t rl-racing-agent:v1 .
```

### 7.2 Run Container
```bash
docker run -d \
  --name rl-racing-demo \
  -p 8000:8000 \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  rl-racing-agent:v1
```

### 7.3 Health Check
```bash
curl -s http://localhost:8000/health
```

---

## 8. Known Limitations (V1)

1. **Single Local Session**: The simulation backend currently manages a single active simulation session per server process. Multi-tab concurrency shares the same simulation state.
2. **Deterministic Inference**: Inference uses $a = \tanh(\mu)$ without stochastic sampling.
3. **Headless Environment**: The server operates without display drivers or Pygame rendering; all visual rendering occurs on the client HTML5 Canvas.
