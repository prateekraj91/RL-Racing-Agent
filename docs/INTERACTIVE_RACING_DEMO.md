# RL Racing Agent: Cinematic 3D Autonomous Simulation Dashboard (V1)

## 1. Overview & Architecture

The Interactive Web Frontend provides a cinematic, high-fidelity 3D autonomous racing simulation powered by the frozen `generalist-v1.0` Soft Actor-Critic (SAC) policy and the authentic `RacingEnv` Python physics engine.

```
+---------------------------------------------------------------------------------------+
|                       Cinematic Browser UI (Three.js 3D / WebGL)                     |
|                                                                                       |
|   +------------------------------------+  +---------------------------------------+   |
|   | 3D Scene (Three.js WebGL Engine)   |  | AI Race Engineer & Telemetry Cockpit  |   |
|   | - 3D procedural asphalt ribbon     |  | - Live event streaming (Apex, Braking)|   |
|   | - Red & white alternating kerbs    |  | - Post-Lap Performance Review Card    |   |
|   | - 3D prototype car + aero wings    |  | - Digital speed, slip, steer gauges   |   |
|   | - Independent front wheel steering |  | - 5x rangefinder distance meters      |   |
|   | - Grounded active brake lights     |  | - Live gradient saliency bar charts   |   |
|   | - 5x dynamic glowing lidar beams   |  | - 2D circuit radar minimap inset      |   |
|   | - 3D glowing trajectory ribbon     |  |                                       |   |
|   | - 4x Camera modes (Chase, Cockpit) |  |                                       |   |
|   +------------------------------------+  +---------------------------------------+   |
|                                     ^            ^                                    |
|                                     |            |                                    |
|                          JSON State |            | State Updates                      |
|                 (20-30 ticks / sec) |            |                                    |
|                                     +------------+                                    |
|                                     |                                                 |
|                          HTTP POST /simulation/step                                   |
|                                     v                                                 |
+---------------------------------------------------------------------------------------+
|                    FastAPI Simulation Backend (inference/app.py)                     |
|                                                                                       |
|   +-------------------------------------------------------------------------------+   |
|   | SimulationSession Manager (inference/simulation.py)                           |   |
|   | - Owns authentic RacingEnv instance (env/environment.py)                      |   |
|   | - Manages Track geometry, car state, and trajectory history                   |   |
|   | - Detects driving milestones (corner approach, apex, braking, cruising)      |   |
|   | - Computes grounded post-lap review metrics (smoothness, min margin, speed)   |   |
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

## 2. Cinematic 3D Visual Priority & Renderer

The primary visualization is a full-screen, high-performance real-time 3D simulation built with **Three.js** and WebGL.

### 2.1 Four Cinematic Camera Modes
- **Chase Camera (`btnCamChase`)**: Cinematic trailing perspective following behind and above the car with smooth exponential damping (`lerp`) and dynamic velocity-dependent elevation.
- **Cockpit / Hood Camera (`btnCamCockpit`)**: First-person perspective anchored directly on the vehicle's nose looking down the track with front wheels and laser beams in view.
- **3D Orbit Camera (`btnCamOrbit`)**: Interactive free inspection mode using `OrbitControls` centered on the car (left-click drag to rotate, right-click to pan, scroll wheel to zoom).
- **Overhead Broadcast Camera (`btnCamOverhead`)**: Strategic satellite top-down perspective ($Y = 520$) providing an expansive tactical overview of the entire circuit.

### 2.2 Detailed 3D Racing Vehicle
- **Aerodynamic Monocoque**: Multi-component GT/Prototype racecar with front carbon splitter, aerodynamic sidepods, beveled nose cone, and rear aerodynamic diffuser.
- **Cockpit Canopy**: Tinted reflective cockpit canopy dome.
- **Rear High-Downforce GT Wing**: Dual aerodynamic pylons mounting a wide carbon fiber airfoil.
- **Steerable Wheels**:
  - 4 rubber tires with titanium multi-spoke rims and brake calipers.
  - Front wheels are parented to independent steer pivots:
    $$\theta_{\text{pivot}} = -\theta_{\text{steer}} = -\delta$$
  - Real-time wheel spin proportional to vehicle velocity.
- **Grounded Action Lighting**:
  - Twin LED projector headlights with directional cone spotlights shining onto the track.
  - Dual rear brake light bars that flare into bright crimson with a dynamic rear point light source **only** when grounded in actual deceleration/braking ($a_{\text{long}} < -0.05$).
  - Cyan chassis underglow illuminating the asphalt surface.

### 2.3 The Five Real Lidar Sensor Beams
- Exactly the 5 real rangefinder rays $[-90^\circ, -45^\circ, 0^\circ, +45^\circ, +90^\circ]$.
- Originates from the sensor pod on the car's nose and terminates at exact world endpoints $(x_{\text{end}}, y_{\text{end}})$.
- Distance-dependent coloring:
  - **Electric Cyan** ($> 90\text{ px}$): Clear track corridor.
  - **Radiant Amber** ($50 - 90\text{ px}$): Approaching apex or boundary wall.
  - **Warning Crimson** ($< 50\text{ px}$): Boundary danger.
- Pulsing concentric ring markers and spark particles at boundary hit points.

### 2.4 3D Trajectory Ribbon
- An extruded luminous cyan ribbon generated dynamically along the vehicle's historical path behind the rear axle at $Y = 0.16$.

---

## 3. AI Race Engineer & Post-Lap Review

A dedicated left sidebar hosts the **AI Race Engineer** providing real-time telemetry analysis and post-lap performance reviews.

### 3.1 Real-Time Grounded Event Stream
Analyzes continuous telemetry and generates fact-based explanations:
- **Corner Approach**: Detects forward distance reduction and steering increase (e.g. *"Corner approach detected. Forward beam is 82.0 px, speed is 3.60 px/step. Steering demand is +0.72."*).
- **Apex Clipping**: Measures minimum lateral clearance (e.g. *"Apex clipping detected. Left boundary clearance is 24.0 px holding -22.5° steering lock."*).
- **Deceleration / Braking**: Flags negative longitudinal command onset.
- **Slip Correction**: Flags elevated tire slip angles ($|\beta_{\text{slip}}| > 4.5^\circ$) and counter-steering responses.
- **Straight Cruising**: Flags full-throttle cruising with centered steering.

### 3.2 Grounded Post-Lap Performance Review Card
Generated automatically upon lap completion or on demand:
- **Lap Time**: Total duration in simulation steps.
- **Average & Top Velocity**: Speed metrics in pixels/step.
- **Steering Smoothness Score**: Normalized score based on mean consecutive steering action delta:
  $$\text{Jitter} = \frac{1}{N-1}\sum_{t=1}^{N-1} |a_{\text{steer}, t} - a_{\text{steer}, t-1}|$$
- **Minimum Boundary Proximity**: Closest proximity recorded across all 5 rays throughout the lap.
- **Off-Track Excursions**: Count of steps where $|d_{\text{signed}}| > W / 2$.
- **Grounded Observations**: Concrete strengths and areas for optimization strictly derived from measured numbers.

---

## 4. Track Presets & Geometry Extraction

The backend extracts procedural track curves and computes continuous normal vectors from `track.centerline` to generate inner and outer road boundaries for 3D extrusion.

| Preset ID | Name | Width | Base Radius | Min Radius | Key Racing Characteristics | Recommended Seeds |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `standard_medium` | Standard Medium Circuit | $70\text{ px}$ | $250\text{ px}$ | $80\text{ px}$ | Balanced benchmark circuit with diverse corner radii | `1001`, `1002`, `1003`, `101` |
| `narrow_corridor` | Narrow Precision Corridor | $40\text{ px}$ | $250\text{ px}$ | $80\text{ px}$ | High-difficulty narrow track punishing lateral deviation | `1001`, `1002`, `1005` |
| `hairpin_challenge`| Hairpin Switchback | $70\text{ px}$ | $250\text{ px}$ | $80\text{ px}$ | Sharp pinch turn (`hairpin_factor = 0.35`) demanding heavy braking | `1001`, `1002`, `1003` |
| `wide_oval` | Wide Flowing Circuit | $100\text{ px}$ | $250\text{ px}$ | $80\text{ px}$ | High-speed sweeping turns with generous margins | `101`, `1001`, `1004` |
| `default_circuit` | Default Environment Circuit| $70\text{ px}$ | $210\text{ px}$ | $70\text{ px}$ | Baseline `RacingEnv` default geometry | `42`, `101`, `1001` |

---

## 5. Policy Explainability: Input-Gradient Saliency

The simulator calculates real-time input-gradient saliency using the frozen `SACActor`:

$$\mathbf{S}_{\text{steer}} = \left| \frac{\partial a_{\text{steer}}}{\partial \mathbf{s}} \right|, \quad \mathbf{S}_{\text{long}} = \left| \frac{\partial a_{\text{long}}}{\partial \mathbf{s}} \right|$$

- **Safety & Parameter Freeze**: All neural network weights have `requires_grad=False`. Gradients are enabled exclusively on the input observation tensor under `torch.autograd.grad`. No optimizer or weight update is ever performed.
- **Explainability Scope**: Visualized bar charts represent mathematical gradient sensitivity across the 9 observation inputs.
- **Disclaimer**: Saliency is explicitly labeled as mathematical sensitivity, **not** human visual attention or gaze.

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

## 8. Verification & Performance

- **MacBook Air M4 Testing**: Runs at a steady 60 FPS in full 3D WebGL mode.
- **Authentic Physics**: 3D environment is a direct visualization of the underlying 2D continuous simulation physics; no physics fabrication exists in JavaScript.
- **Integrity**: Frozen policy `generalist-v1.0.pth` (`0d40dc47...`) and config `generalist_v1.json` (`4d76575b...`) remain completely unchanged.
