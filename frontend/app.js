/**
 * RL Racing Agent - Autonomous Simulator Dashboard
 * Real-time canvas rendering, telemetry synchronization, and simulation control.
 */

// Application State
const state = {
  isRunning: false,
  isStepping: false,
  simSpeed: 2,
  calculateSaliency: true,
  commentaryMode: "offline",
  activeSaliencyTab: "steer",
  
  // Track & Physics Data
  track: null,
  vehicle: null,
  telemetry: null,
  rays: [],
  trajectory: [],
  saliency: null,
  historicalSaliency: null,
  commentary: [],
  session: null,

  // Transform coordinates for Canvas
  scale: 1.0,
  offsetX: 0,
  offsetY: 0,
};

// DOM Elements
const canvas = document.getElementById("raceCanvas");
const ctx = canvas.getContext("2d");

const btnStartPause = document.getElementById("btnStartPause");
const startIcon = document.getElementById("startIcon");
const startText = document.getElementById("startText");
const btnStep = document.getElementById("btnStep");
const btnReset = document.getElementById("btnReset");
const btnLoadTrack = document.getElementById("btnLoadTrack");
const trackPreset = document.getElementById("trackPreset");
const trackSeed = document.getElementById("trackSeed");
const speedRange = document.getElementById("speedRange");
const speedVal = document.getElementById("speedVal");
const checkSaliency = document.getElementById("checkSaliency");
const checkGemini = document.getElementById("checkGemini");

const statusPill = document.getElementById("statusPill");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const stepClock = document.getElementById("stepClock");

const valVelocity = document.getElementById("valVelocity");
const barVelocity = document.getElementById("barVelocity");
const valSteering = document.getElementById("valSteering");
const valSteeringAction = document.getElementById("valSteeringAction");
const barSteering = document.getElementById("barSteering");
const valThrottle = document.getElementById("valThrottle");
const barThrottle = document.getElementById("barThrottle");
const valActionMode = document.getElementById("valActionMode");
const valProgress = document.getElementById("valProgress");
const barProgress = document.getElementById("barProgress");

const valHeadingErr = document.getElementById("valHeadingErr");
const valSignedDist = document.getElementById("valSignedDist");
const valSlip = document.getElementById("valSlip");
const valPos = document.getElementById("valPos");

const sensorsList = document.getElementById("sensorsList");
const saliencyBarsContainer = document.getElementById("saliencyBarsContainer");
const topSteerRay = document.getElementById("topSteerRay");
const topThrottleRay = document.getElementById("topThrottleRay");
const commentaryFeed = document.getElementById("commentaryFeed");
const commentaryModeBadge = document.getElementById("commentaryModeBadge");

const canvasOverlay = document.getElementById("canvasOverlay");
const overlayTitle = document.getElementById("overlayTitle");
const overlayMessage = document.getElementById("overlayMessage");
const btnOverlayRestart = document.getElementById("btnOverlayRestart");

const tabSteer = document.getElementById("tabSteer");
const tabLong = document.getElementById("tabLong");
const tabHist = document.getElementById("tabHist");

// Ray metadata
const RAY_CONFIGS = [
  { name: "ray_-90", label: "Ray -90° (Left)", angle: -90 },
  { name: "ray_-45", label: "Ray -45° (Diag L)", angle: -45 },
  { name: "ray_0", label: "Ray 0° (Forward)", angle: 0 },
  { name: "ray_45", label: "Ray +45° (Diag R)", angle: 45 },
  { name: "ray_90", label: "Ray +90° (Right)", angle: 90 },
];

/**
 * Initialize simulation and load initial track
 */
async function init() {
  setupEventListeners();
  await loadTrackPresets();
  await startSimulation("standard_medium", 1001);
  await fetchHistoricalSaliency();
  requestAnimationFrame(renderLoop);
}

/**
 * Fetch available track presets
 */
async function loadTrackPresets() {
  try {
    const res = await fetch("/simulation/tracks");
    if (!res.ok) return;
    const presets = await res.json();
    trackPreset.innerHTML = "";
    presets.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.name} (width ${p.track_kwargs.width || 70}px)`;
      trackPreset.appendChild(opt);
    });
    trackPreset.value = "standard_medium";
  } catch (err) {
    console.warn("Could not load track presets:", err);
  }
}

/**
 * Start or re-initialize simulation session
 */
async function startSimulation(presetId, seed) {
  try {
    setLoading(true);
    const res = await fetch("/simulation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset_id: presetId,
        track_seed: parseInt(seed, 10),
        steer_penalty_coeff: 0.01,
      }),
    });
    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    const data = await res.json();
    applyState(data);
    updateCanvasTransform();
    hideOverlay();
  } catch (err) {
    console.error("Failed to start simulation:", err);
  } finally {
    setLoading(false);
  }
}

/**
 * Step the simulation by N steps
 */
async function stepSimulation(numSteps = 1) {
  if (state.isStepping) return;
  state.isStepping = true;

  try {
    const res = await fetch("/simulation/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        steps: numSteps,
        calculate_saliency: state.calculateSaliency,
        commentary_mode: state.commentaryMode,
      }),
    });
    if (!res.ok) throw new Error(`Step error: ${res.status}`);
    const data = await res.json();
    applyState(data);

    if (state.session.is_crashed) {
      pauseSimulation();
      showOverlay("CRASH: VEHICLE OFF-TRACK", "Car departed track boundary.", true);
    } else if (state.session.is_completed) {
      pauseSimulation();
      showOverlay("LAP COMPLETED!", `Finished in ${state.session.step_count} steps.`, false);
    }
  } catch (err) {
    console.error("Step execution failed:", err);
    pauseSimulation();
  } finally {
    state.isStepping = false;
  }
}

/**
 * Reset simulation session
 */
async function resetSimulation() {
  pauseSimulation();
  hideOverlay();
  try {
    const res = await fetch("/simulation/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset_id: trackPreset.value,
        track_seed: parseInt(trackSeed.value, 10),
      }),
    });
    if (res.ok) {
      const data = await res.json();
      applyState(data);
      updateCanvasTransform();
    }
  } catch (err) {
    console.error("Reset failed:", err);
  }
}

/**
 * Apply server simulation response to local state and update UI
 */
function applyState(data) {
  state.session = data.session;
  state.vehicle = data.vehicle;
  state.telemetry = data.telemetry;
  state.rays = data.rays || [];
  state.trajectory = data.trajectory || [];
  if (data.track) state.track = data.track;
  if (data.saliency) state.saliency = data.saliency;
  if (data.commentary) updateCommentary(data.commentary);

  updateDashboardUI();
}

/**
 * Calculate canvas transform parameters to frame track
 */
function updateCanvasTransform() {
  if (!state.track || !state.track.bounds) return;

  const b = state.track.bounds;
  const trackW = b.max_x - b.min_x;
  const trackH = b.max_y - b.min_y;

  const padding = 40;
  const availW = canvas.width - padding * 2;
  const availH = canvas.height - padding * 2;

  const scaleX = availW / trackW;
  const scaleY = availH / trackH;
  state.scale = Math.min(scaleX, scaleY);

  const centeredW = trackW * state.scale;
  const centeredH = trackH * state.scale;

  state.offsetX = (canvas.width - centeredW) / 2 - b.min_x * state.scale;
  state.offsetY = (canvas.height - centeredH) / 2 - b.min_y * state.scale;
}

/**
 * World coordinate to Canvas screen coordinate
 */
function toScreen(x, y) {
  return [x * state.scale + state.offsetX, y * state.scale + state.offsetY];
}

/**
 * Canvas Rendering Loop
 */
let lastFpsTime = performance.now();
let frameCount = 0;

function renderLoop(now) {
  frameCount++;
  if (now - lastFpsTime >= 1000) {
    document.getElementById("fpsCounter").textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastFpsTime = now;
  }

  drawScene();

  if (state.isRunning && !state.isStepping) {
    stepSimulation(state.simSpeed);
  }

  requestAnimationFrame(renderLoop);
}

/**
 * Draw all visual elements onto canvas
 */
function drawScene() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.track) return;

  // 1. Draw Asphalt Track Surface
  drawTrackSurface();

  // 2. Draw Start / Finish Line
  drawStartFinishLine();

  // 3. Draw Vehicle Trajectory Trail
  drawTrajectory();

  // 4. Draw Lidar Rangefinder Beams
  drawLidarRays();

  // 5. Draw Racing Car Body & Steered Wheels
  drawVehicle();
}

/**
 * Draw road surface ribbon and curb edges
 */
function drawTrackSurface() {
  const track = state.track;
  if (!track.centerline || track.centerline.length < 2) return;

  // Outer asphalt ribbon
  ctx.beginPath();
  const [firstX, firstY] = toScreen(track.centerline[0][0], track.centerline[0][1]);
  ctx.moveTo(firstX, firstY);
  for (let i = 1; i < track.centerline.length; i++) {
    const [sx, sy] = toScreen(track.centerline[i][0], track.centerline[i][1]);
    ctx.lineTo(sx, sy);
  }
  ctx.closePath();
  ctx.lineWidth = track.width * state.scale;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#1b2230";
  ctx.stroke();

  // Outer and Inner Curbs / Boundary Edges
  if (track.outer_boundary && track.inner_boundary) {
    drawBoundaryPath(track.outer_boundary, "#3a4968", 2);
    drawBoundaryPath(track.inner_boundary, "#3a4968", 2);
  }

  // Dashed Centerline
  ctx.beginPath();
  ctx.moveTo(firstX, firstY);
  for (let i = 1; i < track.centerline.length; i++) {
    const [sx, sy] = toScreen(track.centerline[i][0], track.centerline[i][1]);
    ctx.lineTo(sx, sy);
  }
  ctx.closePath();
  ctx.lineWidth = 1.5;
  ctx.setLineDash([8, 8]);
  ctx.strokeStyle = "rgba(100, 120, 150, 0.45)";
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawBoundaryPath(points, color, width) {
  if (points.length < 2) return;
  ctx.beginPath();
  const [x0, y0] = toScreen(points[0][0], points[0][1]);
  ctx.moveTo(x0, y0);
  for (let i = 1; i < points.length; i++) {
    const [x, y] = toScreen(points[i][0], points[i][1]);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

/**
 * Draw Start / Finish Banner across track
 */
function drawStartFinishLine() {
  if (!state.track || !state.track.start_pose) return;
  const sp = state.track.start_pose;
  const [sx, sy] = toScreen(sp.x, sp.y);
  const rad = (sp.heading * Math.PI) / 180;
  const half = (state.track.half_width || 35) * state.scale;

  // Perpendicular to heading
  const nx = -Math.sin(rad);
  const ny = -Math.cos(rad);

  ctx.beginPath();
  ctx.moveTo(sx - nx * half, sy - ny * half);
  ctx.lineTo(sx + nx * half, sy + ny * half);
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#00e676";
  ctx.stroke();

  // Small Start arrow
  ctx.beginPath();
  ctx.arc(sx, sy, 4, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
}

/**
 * Draw continuous trajectory line
 */
function drawTrajectory() {
  if (state.trajectory.length < 2) return;

  ctx.beginPath();
  const [x0, y0] = toScreen(state.trajectory[0][0], state.trajectory[0][1]);
  ctx.moveTo(x0, y0);
  for (let i = 1; i < state.trajectory.length; i++) {
    const [x, y] = toScreen(state.trajectory[i][0], state.trajectory[i][1]);
    ctx.lineTo(x, y);
  }
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "rgba(0, 240, 255, 0.75)";
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

/**
 * Draw 5 live lidar rangefinder beams
 */
function drawLidarRays() {
  if (!state.rays || state.rays.length === 0) return;

  state.rays.forEach((ray) => {
    const [x0, y0] = toScreen(ray.start[0], ray.start[1]);
    const [x1, y1] = toScreen(ray.end[0], ray.end[1]);

    let rayColor = "#00e676"; // Clear (Green)
    if (ray.distance < 45) {
      rayColor = "#ff3366"; // Danger close (Red)
    } else if (ray.distance < 90) {
      rayColor = "#ffb700"; // Caution (Amber)
    }

    // Ray beam line
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.strokeStyle = rayColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // End impact dot
    ctx.beginPath();
    ctx.arc(x1, y1, ray.is_close ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = rayColor;
    ctx.shadowColor = rayColor;
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
  });
}

/**
 * Draw Vehicle with actual heading and steered front wheels
 */
function drawVehicle() {
  if (!state.vehicle) return;
  const v = state.vehicle;
  const [sx, sy] = toScreen(v.x, v.y);

  ctx.save();
  ctx.translate(sx, sy);
  // car.angle is in degrees, y is down on screen so rotate by -angle
  ctx.rotate((-v.angle * Math.PI) / 180);

  const carLength = 26;
  const carWidth = 14;

  // Car Body
  ctx.fillStyle = "#ffb700"; // Racing Gold
  ctx.beginPath();
  ctx.roundRect(-carLength / 2, -carWidth / 2, carLength, carWidth, 3);
  ctx.fill();

  // Cockpit / Windshield
  ctx.fillStyle = "#0a1120";
  ctx.fillRect(-carLength * 0.1, -carWidth * 0.35, carLength * 0.35, carWidth * 0.7);

  // Rear Wing / Spoiler
  ctx.fillStyle = "#ff3366";
  ctx.fillRect(-carLength / 2 - 2, -carWidth / 2 - 1, 3, carWidth + 2);

  // Headlights
  ctx.fillStyle = "#00f0ff";
  ctx.fillRect(carLength / 2 - 2, -carWidth / 2 + 1, 2, 3);
  ctx.fillRect(carLength / 2 - 2, carWidth / 2 - 4, 2, 3);

  // Steered Front Wheels
  const steerRad = (-v.steering_angle_deg * Math.PI) / 180;
  drawWheel(carLength * 0.35, -carWidth / 2 - 2, steerRad);
  drawWheel(carLength * 0.35, carWidth / 2 + 2, steerRad);

  // Rear Wheels (Fixed)
  drawWheel(-carLength * 0.35, -carWidth / 2 - 2, 0);
  drawWheel(-carLength * 0.35, carWidth / 2 + 2, 0);

  ctx.restore();
}

function drawWheel(x, y, steerRad) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(steerRad);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(-4, -1.5, 8, 3);
  ctx.restore();
}

/**
 * Update HTML telemetry readouts and meters
 */
function updateDashboardUI() {
  if (!state.vehicle || !state.session) return;

  const v = state.vehicle;
  const t = state.telemetry;
  const s = state.session;

  // Status Pill
  if (s.is_crashed) {
    statusDot.className = "pill-dot red";
    statusText.textContent = "CRASHED (OFF-TRACK)";
  } else if (s.is_completed) {
    statusDot.className = "pill-dot green";
    statusText.textContent = "LAP COMPLETED";
  } else if (state.isRunning) {
    statusDot.className = "pill-dot cyan";
    statusText.textContent = "RUNNING";
  } else {
    statusDot.className = "pill-dot amber";
    statusText.textContent = "PAUSED";
  }

  stepClock.textContent = `STEP: ${s.step_count} / ${s.max_steps}`;

  // Speed
  valVelocity.innerHTML = `${v.velocity.toFixed(2)} <span class="unit">px/step</span>`;
  const speedPct = Math.max(0, Math.min(100, (v.velocity / 4.0) * 100));
  barVelocity.style.width = `${speedPct}%`;

  // Steering
  valSteering.textContent = `${v.steering_angle_deg > 0 ? "+" : ""}${v.steering_angle_deg.toFixed(1)}°`;
  valSteeringAction.textContent = `(act: ${v.steering.toFixed(2)})`;
  const steerFillWidth = (Math.abs(v.steering) * 50).toFixed(1);
  if (v.steering >= 0) {
    barSteering.style.left = "50%";
    barSteering.style.width = `${steerFillWidth}%`;
  } else {
    barSteering.style.left = `${50 - parseFloat(steerFillWidth)}%`;
    barSteering.style.width = `${steerFillWidth}%`;
  }

  // Throttle / Longitudinal
  valThrottle.textContent = `${v.throttle > 0 ? "+" : ""}${v.throttle.toFixed(2)}`;
  valActionMode.textContent = v.action_mode.toUpperCase();
  const throttleFillWidth = (Math.abs(v.throttle) * 50).toFixed(1);
  if (v.throttle >= 0) {
    barThrottle.style.left = "50%";
    barThrottle.style.width = `${throttleFillWidth}%`;
    barThrottle.style.backgroundColor = "var(--accent-cyan)";
  } else {
    barThrottle.style.left = `${50 - parseFloat(throttleFillWidth)}%`;
    barThrottle.style.width = `${throttleFillWidth}%`;
    barThrottle.style.backgroundColor = "var(--accent-red)";
  }

  // Progress
  const progressPct = Math.max(0, Math.min(100, v.lap_progress * 100));
  valProgress.textContent = `${progressPct.toFixed(1)}%`;
  barProgress.style.width = `${progressPct}%`;

  // Secondary metrics
  valHeadingErr.textContent = `${t.heading_error > 0 ? "+" : ""}${t.heading_error.toFixed(1)}°`;
  valSignedDist.textContent = `${t.signed_distance > 0 ? "+" : ""}${t.signed_distance.toFixed(1)} px`;
  valSlip.textContent = `${t.slip_angle > 0 ? "+" : ""}${t.slip_angle.toFixed(1)}°`;
  valPos.textContent = `${v.x.toFixed(1)}, ${v.y.toFixed(1)}`;

  // Ray sensor meters
  renderSensorsList();

  // Saliency bars
  renderSaliency();
}

/**
 * Render 5 Lidar Rangefinder meters
 */
function renderSensorsList() {
  if (!state.rays || state.rays.length === 0) return;

  sensorsList.innerHTML = "";
  RAY_CONFIGS.forEach((cfg, idx) => {
    const ray = state.rays[idx] || { distance: 200 };
    const dist = ray.distance;
    const pct = Math.max(0, Math.min(100, (dist / 200.0) * 100));

    let barColor = "var(--accent-green)";
    if (dist < 45) barColor = "var(--accent-red)";
    else if (dist < 90) barColor = "var(--accent-amber)";

    const row = document.createElement("div");
    row.className = "sensor-row";
    row.innerHTML = `
      <span class="sensor-name">${cfg.label}</span>
      <div class="sensor-bar-bg">
        <div class="sensor-bar-fill" style="width: ${pct}%; background-color: ${barColor};"></div>
      </div>
      <span class="sensor-val" style="color: ${barColor}">${dist.toFixed(0)} px</span>
    `;
    sensorsList.appendChild(row);
  });
}

/**
 * Render Gradient Saliency bars based on active tab
 */
function renderSaliency() {
  if (state.activeSaliencyTab === "hist") {
    renderHistoricalSaliency();
    return;
  }

  if (!state.saliency) {
    saliencyBarsContainer.innerHTML = '<div style="color: var(--text-dim); padding: 8px;">Awaiting live saliency compute...</div>';
    return;
  }

  const s = state.saliency;
  const isSteer = state.activeSaliencyTab === "steer";
  const salData = isSteer ? s.feature_saliency_steer : s.feature_saliency_long;

  topSteerRay.textContent = s.top_steering_ray || "None";
  topThrottleRay.textContent = s.top_longitudinal_ray || "None";

  // Normalize bars against max in current feature set
  const maxVal = Math.max(...Object.values(salData), 0.001);

  saliencyBarsContainer.innerHTML = "";
  Object.entries(salData).forEach(([feat, val]) => {
    const pct = Math.min(100, (val / maxVal) * 100);
    const row = document.createElement("div");
    row.className = "saliency-row";
    row.innerHTML = `
      <span class="saliency-feat">${feat}</span>
      <div class="saliency-bar-bg">
        <div class="saliency-bar-fill" style="width: ${pct}%;"></div>
      </div>
      <span class="saliency-val">${val.toFixed(4)}</span>
    `;
    saliencyBarsContainer.appendChild(row);
  });
}

/**
 * Fetch and render historical benchmark saliency reference
 */
async function fetchHistoricalSaliency() {
  try {
    const res = await fetch("/simulation/historical-saliency");
    if (res.ok) {
      state.historicalSaliency = await res.json();
    }
  } catch (err) {
    console.warn("Could not load historical saliency reference:", err);
  }
}

function renderHistoricalSaliency() {
  if (!state.historicalSaliency || !state.historicalSaliency.analysis_results) {
    saliencyBarsContainer.innerHTML = '<div style="color: var(--text-dim); padding: 8px;">Historical reference not available.</div>';
    return;
  }

  const results = state.historicalSaliency.analysis_results;
  const rawMean = (results.overall_steering && results.overall_steering.raw_mean) || {};
  const maxVal = Math.max(...Object.values(rawMean), 0.001);

  if (results.overall_steering) {
    topSteerRay.textContent = results.overall_steering.top_ray_raw || "None";
  }
  if (results.overall_longitudinal) {
    topThrottleRay.textContent = results.overall_longitudinal.top_ray_raw || "None";
  }

  saliencyBarsContainer.innerHTML = "";
  Object.entries(rawMean).forEach(([feat, val]) => {
    const pct = Math.min(100, (val / maxVal) * 100);
    const row = document.createElement("div");
    row.className = "saliency-row";
    row.innerHTML = `
      <span class="saliency-feat">${feat}</span>
      <div class="saliency-bar-bg">
        <div class="saliency-bar-fill" style="width: ${pct}%; background: #00f0ff;"></div>
      </div>
      <span class="saliency-val" style="color: #00f0ff">${val.toFixed(4)}</span>
    `;
    saliencyBarsContainer.appendChild(row);
  });
}

/**
 * Update Grounded Commentary Feed
 */
function updateCommentary(items) {
  if (!items || items.length === 0) return;

  commentaryFeed.innerHTML = "";
  items.slice(-8).forEach((item) => {
    const row = document.createElement("div");
    const isGemini = item.source === "gemini_live";
    const isCrash = item.event_type === "crash";

    row.className = `commentary-item ${isGemini ? "gemini" : ""} ${isCrash ? "crash" : ""}`;
    row.innerHTML = `
      <span class="c-step">[STEP ${item.step}]</span>
      <span class="c-source">[${isGemini ? "GEMINI 3.8 FLASH" : "OFFLINE GROUNDED"}]</span>
      <span class="c-text">${item.commentary}</span>
    `;
    commentaryFeed.appendChild(row);
  });
  commentaryFeed.scrollTop = commentaryFeed.scrollHeight;
}

/**
 * Control Actions
 */
function startSimulationLoop() {
  if (state.session && (state.session.is_crashed || state.session.is_completed)) {
    resetSimulation();
  }
  state.isRunning = true;
  startIcon.textContent = "⏸";
  startText.textContent = "Pause Simulation";
  btnStartPause.className = "btn btn-secondary";
}

function pauseSimulation() {
  state.isRunning = false;
  startIcon.textContent = "▶";
  startText.textContent = "Resume Simulation";
  btnStartPause.className = "btn btn-primary";
}

function showOverlay(title, message, isCrash) {
  overlayTitle.textContent = title;
  overlayTitle.style.color = isCrash ? "var(--accent-red)" : "var(--accent-green)";
  overlayMessage.textContent = message;
  canvasOverlay.style.display = "flex";
}

function hideOverlay() {
  canvasOverlay.style.display = "none";
}

function setLoading(isLoading) {
  btnStartPause.disabled = isLoading;
  btnStep.disabled = isLoading;
  btnReset.disabled = isLoading;
  btnLoadTrack.disabled = isLoading;
}

/**
 * Setup Event Listeners
 */
function setupEventListeners() {
  btnStartPause.addEventListener("click", () => {
    if (state.isRunning) {
      pauseSimulation();
    } else {
      startSimulationLoop();
    }
  });

  btnStep.addEventListener("click", () => {
    pauseSimulation();
    stepSimulation(1);
  });

  btnReset.addEventListener("click", resetSimulation);
  btnOverlayRestart.addEventListener("click", resetSimulation);

  btnLoadTrack.addEventListener("click", () => {
    pauseSimulation();
    startSimulation(trackPreset.value, trackSeed.value);
  });

  speedRange.addEventListener("input", (e) => {
    state.simSpeed = parseInt(e.target.value, 10);
    speedVal.textContent = `${state.simSpeed}x`;
  });

  checkSaliency.addEventListener("change", (e) => {
    state.calculateSaliency = e.target.checked;
  });

  checkGemini.addEventListener("change", (e) => {
    state.commentaryMode = e.target.checked ? "gemini" : "offline";
    commentaryModeBadge.textContent = e.target.checked
      ? "LIVE GEMINI 3.8 FLASH"
      : "OFFLINE DETERMINISTIC";
    commentaryModeBadge.style.color = e.target.checked
      ? "var(--accent-purple)"
      : "var(--text-dim)";
  });

  tabSteer.addEventListener("click", () => {
    state.activeSaliencyTab = "steer";
    tabSteer.className = "tab-btn active";
    tabLong.className = "tab-btn";
    tabHist.className = "tab-btn";
    renderSaliency();
  });

  tabLong.addEventListener("click", () => {
    state.activeSaliencyTab = "long";
    tabSteer.className = "tab-btn";
    tabLong.className = "tab-btn active";
    tabHist.className = "tab-btn";
    renderSaliency();
  });

  tabHist.addEventListener("click", () => {
    state.activeSaliencyTab = "hist";
    tabSteer.className = "tab-btn";
    tabLong.className = "tab-btn";
    tabHist.className = "tab-btn active";
    renderSaliency();
  });
}

// Bootstrap on DOM loaded
document.addEventListener("DOMContentLoaded", init);
