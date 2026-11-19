/**
 * RL Racing Agent // Cinematic 3D Autonomous Simulation Dashboard
 *
 * Core Systems:
 * - Three.js 3D renderer of authentic Python RacingEnv physics (PX_TO_WORLD = 1.0)
 * - 4 Cinematic Camera Modes (Chase, Cockpit, Orbit, Overhead)
 * - Cockpit Camera: Eye placed at driver position above/ahead of center; car body hidden (wheels visible)
 * - Trajectory Ribbon: Cleared on reset and lap-wrap to eliminate diagonal crossing lines
 * - Non-finite Renderer Guards: Skips updates and warns once if any NaN/Inf is detected
 * - AI Race Engineer Feed: Appends DOM nodes, preserves last 50 items, never wipes innerHTML
 * - Policy Saliency: Live feature sensitivity bars and top steering/throttle rays
 * - 2D Circuit Radar Minimap: Matches 3D Overhead camera 1:1 without mirroring
 * - Debug Inspector Overlay (Key: D): Coordinate readouts, axes, and scale stats
 */

// ==============================================================================
// State & Configuration
// ==============================================================================

const STATE = {
  activeCameraMode: "chase", // chase | cockpit | orbit | overhead
  simRunning: false,
  simSpeed: 2,
  calculateSaliency: true,
  commentaryMode: "offline",
  presetId: "standard_medium",
  trackSeed: 1001,
  debugMode: false,

  // Track cache
  trackData: null,
  trackCenter: { x: 400.0, y: 300.0 },

  // Camera lerp targets
  cameraPos: new THREE.Vector3(0, 360, 450),
  cameraTarget: new THREE.Vector3(0, 0, 0),

  // Trajectory buffer
  trajectoryPoints: [],
  prevLapProgress: 0.0,

  // Commentary tracking
  seenCommentaryKeys: new Set(),

  // Policy Saliency
  currentSaliency: null,
  historicalSaliency: null,
  saliencyTab: "steer", // steer | long | hist
};

const FEATURE_NAMES = [
  "velocity",
  "heading_error",
  "signed_distance",
  "slip",
  "ray_minus_90",
  "ray_minus_45",
  "ray_0",
  "ray_plus_45",
  "ray_plus_90",
];

const RAY_NAMES = ["ray_minus_90", "ray_minus_45", "ray_0", "ray_plus_45", "ray_plus_90"];
const RAY_LABELS = ["L -90°", "DL -45°", "FWD 0°", "DR +45°", "R +90°"];

// Renderer Non-Finite Guards
let hasWarnedNonFinite = false;

function isValidNumber(n) {
  return typeof n === "number" && Number.isFinite(n);
}

function isValidVector2(v) {
  return Array.isArray(v) && v.length >= 2 && isValidNumber(v[0]) && isValidNumber(v[1]);
}

// ==============================================================================
// Three.js 3D Scene Architecture
// ==============================================================================

let scene, camera, renderer, controls;
let threeCanvas, minimapCanvas, minimapCtx;
let trackGroup, carGroup, carBodyGroup, trajectoryMesh, lidarGroup, debugGroup;
let frontLeftSteerPivot, frontRightSteerPivot;
let wheelMeshes = [];
let rearBrakeLights = [];
let rayBeamObjects = [];
let roadMesh, curbInnerMesh, curbOuterMesh, barrierMesh;
let carBboxHelper, trackAxesHelper, carAxesHelper;

function initThreeJS() {
  threeCanvas = document.getElementById("threeCanvas");
  minimapCanvas = document.getElementById("raceCanvas");
  if (minimapCanvas) {
    minimapCtx = minimapCanvas.getContext("2d");
  }

  const container = document.getElementById("canvasContainer");
  const width = (container && container.clientWidth > 0) ? container.clientWidth : 800;
  const height = (container && container.clientHeight > 0) ? container.clientHeight : 600;

  // 1. Scene & Atmosphere
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050811);
  scene.fog = new THREE.FogExp2(0x050811, 0.00035);

  // 2. Camera Setup (PX_TO_WORLD = 1.0, 1 world unit = 1 pixel)
  camera = new THREE.PerspectiveCamera(52, width / height, 1.0, 5000);
  camera.position.set(0, 300, 450);

  // 3. WebGL Renderer
  renderer = new THREE.WebGLRenderer({
    canvas: threeCanvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // 4. Orbit Controls
  if (THREE.OrbitControls) {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    controls.enabled = false;
  }

  // 5. Scene Hierarchy Groups
  trackGroup = new THREE.Group();
  scene.add(trackGroup);

  carGroup = new THREE.Group();
  scene.add(carGroup);

  lidarGroup = new THREE.Group();
  scene.add(lidarGroup);

  debugGroup = new THREE.Group();
  debugGroup.visible = false;
  scene.add(debugGroup);

  // 6. Lighting & Environment
  setupLighting();

  // 7. Vehicle, Sensors & Debug Helpers
  buildCarModel();
  buildLidarBeams();
  buildDebugHelpers();

  // 8. Container Resize Observer
  if (window.ResizeObserver && container) {
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        if (cr.width > 0 && cr.height > 0) {
          onCanvasResize(cr.width, cr.height);
        }
      }
    });
    ro.observe(container);
  }
  window.addEventListener("resize", () => {
    if (container) {
      onCanvasResize(container.clientWidth, container.clientHeight);
    }
  });
}

function onCanvasResize(width, height) {
  if (!renderer || !camera || width <= 0 || height <= 0) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function setupLighting() {
  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);

  const mainSun = new THREE.DirectionalLight(0xeef6ff, 1.4);
  mainSun.position.set(300, 500, 250);
  mainSun.castShadow = true;
  mainSun.shadow.mapSize.width = 2048;
  mainSun.shadow.mapSize.height = 2048;
  mainSun.shadow.camera.near = 10;
  mainSun.shadow.camera.far = 1500;
  mainSun.shadow.camera.left = -600;
  mainSun.shadow.camera.right = 600;
  mainSun.shadow.camera.top = 600;
  mainSun.shadow.camera.bottom = -600;
  scene.add(mainSun);

  const fillLight = new THREE.DirectionalLight(0x00f0ff, 0.45);
  fillLight.position.set(-300, 250, -250);
  scene.add(fillLight);

  // Subtle floor grid (ground plane)
  const gridHelper = new THREE.GridHelper(2500, 50, 0x142035, 0x0a1020);
  gridHelper.position.y = -0.5;
  scene.add(gridHelper);
}

// ==============================================================================
// 3D Track Ribbon & Boundary Generation (PX_TO_WORLD = 1.0)
// ==============================================================================

function build3DTrack(trackData) {
  if (!trackData) return;
  STATE.trackData = trackData;

  // Clear existing track geometry
  while (trackGroup.children.length > 0) {
    const obj = trackGroup.children[0];
    trackGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }

  // Calculate and update global track center
  const bounds = trackData.bounds || { min_x: 100, max_x: 700, min_y: 100, max_y: 500 };
  const cx = (bounds.min_x + bounds.max_x) / 2.0;
  const cy = (bounds.min_y + bounds.max_y) / 2.0;
  STATE.trackCenter = { x: cx, y: cy };
  setWorldTrackCenter(cx, cy);

  const innerPts = trackData.inner_boundary || [];
  const outerPts = trackData.outer_boundary || [];
  const centerPts = trackData.centerline || [];
  const n = Math.min(innerPts.length, outerPts.length);

  if (n < 4) return;

  // 1. Asphalt Road Surface Quad Strip
  const roadGeo = new THREE.BufferGeometry();
  const roadVertices = [];
  const roadNormals = [];
  const roadUvs = [];
  const roadIndices = [];

  for (let i = 0; i < n; i++) {
    const pIn = innerPts[i];
    const pOut = outerPts[i];

    if (!isValidVector2(pIn) || !isValidVector2(pOut)) {
      if (!hasWarnedNonFinite) {
        console.warn("Renderer guard: Non-finite vertex in track boundary, skipping", pIn, pOut);
        hasWarnedNonFinite = true;
      }
      continue;
    }

    const inW = envToWorld(pIn[0], pIn[1]);
    const outW = envToWorld(pOut[0], pOut[1]);

    roadVertices.push(inW.x, 0.05, inW.z);
    roadVertices.push(outW.x, 0.05, outW.z);

    roadNormals.push(0, 1, 0, 0, 1, 0);
    const progress = i / n;
    roadUvs.push(0, progress, 1, progress);

    const next = (i + 1) % n;
    const i0 = i * 2;
    const i1 = i * 2 + 1;
    const i2 = next * 2;
    const i3 = next * 2 + 1;

    roadIndices.push(i0, i1, i2);
    roadIndices.push(i1, i3, i2);
  }

  roadGeo.setAttribute("position", new THREE.Float32BufferAttribute(roadVertices, 3));
  roadGeo.setAttribute("normal", new THREE.Float32BufferAttribute(roadNormals, 3));
  roadGeo.setAttribute("uv", new THREE.Float32BufferAttribute(roadUvs, 2));
  roadGeo.setIndex(roadIndices);

  const roadMat = new THREE.MeshStandardMaterial({
    color: 0x161a22,
    roughness: 0.85,
    metalness: 0.15,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  roadMesh = new THREE.Mesh(roadGeo, roadMat);
  roadMesh.receiveShadow = true;
  trackGroup.add(roadMesh);

  // 2. Curbs (Inner and Outer with alternating Red & White segments)
  // offsetDistance in px: width=4.0 px, height=1.5 px
  curbInnerMesh = buildCurbRibbon(innerPts, -4.0, 1.5);
  curbOuterMesh = buildCurbRibbon(outerPts, 4.0, 1.5);
  if (curbInnerMesh) trackGroup.add(curbInnerMesh);
  if (curbOuterMesh) trackGroup.add(curbOuterMesh);

  // 3. Outer Barrier Wall Ribbon (wallHeight = 7.5 px)
  barrierMesh = buildBarrierWall(outerPts, 4.0, 7.5);
  if (barrierMesh) trackGroup.add(barrierMesh);

  // 4. Dashed Centerline
  if (centerPts.length > 2) {
    const centerLineMesh = buildCenterline(centerPts);
    trackGroup.add(centerLineMesh);
  }

  // 5. Start / Finish Line Banner
  if (innerPts.length > 0 && outerPts.length > 0) {
    buildStartFinishMarker(innerPts[0], outerPts[0]);
  }

  // Reset trajectory ribbon on new track
  clearTrajectoryTrail();
}

function buildCurbRibbon(boundaryPts, offsetDistance, height) {
  const n = boundaryPts.length;
  const curbGeo = new THREE.BufferGeometry();
  const vertices = [];
  const colors = [];
  const indices = [];

  for (let i = 0; i < n; i++) {
    const pt = boundaryPts[i];
    const nextPt = boundaryPts[(i + 1) % n];

    if (!isValidVector2(pt) || !isValidVector2(nextPt)) continue;

    const dx = nextPt[0] - pt[0];
    const dy = nextPt[1] - pt[1];
    const len = Math.hypot(dx, dy) || 1.0;
    const nx = (-dy / len) * offsetDistance;
    const ny = (dx / len) * offsetDistance;

    const base = envToWorld(pt[0], pt[1]);
    const ext = envToWorld(pt[0] + nx, pt[1] + ny);

    vertices.push(base.x, 0.1, base.z);
    vertices.push(ext.x, height, ext.z);

    // Alternating Red & White motorsport curbs
    const isRed = Math.floor(i / 3) % 2 === 0;
    const col = isRed ? [0.95, 0.12, 0.22] : [0.96, 0.96, 0.98];
    colors.push(...col, ...col);

    const next = (i + 1) % n;
    const i0 = i * 2;
    const i1 = i * 2 + 1;
    const i2 = next * 2;
    const i3 = next * 2 + 1;

    indices.push(i0, i1, i2);
    indices.push(i1, i3, i2);
  }

  if (vertices.length < 12) return null;

  curbGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  curbGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  curbGeo.setIndex(indices);
  curbGeo.computeVertexNormals();

  const curbMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.6,
    metalness: 0.2,
  });
  const mesh = new THREE.Mesh(curbGeo, curbMat);
  mesh.receiveShadow = true;
  return mesh;
}

function buildBarrierWall(boundaryPts, offsetDistance, wallHeight) {
  const n = boundaryPts.length;
  const wallGeo = new THREE.BufferGeometry();
  const vertices = [];
  const indices = [];

  for (let i = 0; i < n; i++) {
    const pt = boundaryPts[i];
    const nextPt = boundaryPts[(i + 1) % n];

    if (!isValidVector2(pt) || !isValidVector2(nextPt)) continue;

    const dx = nextPt[0] - pt[0];
    const dy = nextPt[1] - pt[1];
    const len = Math.hypot(dx, dy) || 1.0;
    const nx = (-dy / len) * offsetDistance;
    const ny = (dx / len) * offsetDistance;

    const pos = envToWorld(pt[0] + nx, pt[1] + ny);

    vertices.push(pos.x, 0.0, pos.z);
    vertices.push(pos.x, wallHeight, pos.z);

    const next = (i + 1) % n;
    const i0 = i * 2;
    const i1 = i * 2 + 1;
    const i2 = next * 2;
    const i3 = next * 2 + 1;

    indices.push(i0, i1, i2);
    indices.push(i1, i3, i2);
  }

  if (vertices.length < 12) return null;

  wallGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  wallGeo.setIndex(indices);
  wallGeo.computeVertexNormals();

  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x222a38,
    metalness: 0.6,
    roughness: 0.35,
  });
  const mesh = new THREE.Mesh(wallGeo, wallMat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildCenterline(centerPts) {
  const points = [];
  for (const p of centerPts) {
    if (!isValidVector2(p)) continue;
    const w = envToWorld(p[0], p[1]);
    points.push(new THREE.Vector3(w.x, 0.2, w.z));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineDashedMaterial({
    color: 0xffb700,
    dashSize: 12.0,
    gapSize: 8.0,
    linewidth: 2,
  });
  const line = new THREE.Line(geo, mat);
  line.computeLineDistances();
  return line;
}

function buildStartFinishMarker(inPt, outPt) {
  if (!isValidVector2(inPt) || !isValidVector2(outPt)) return;
  const inW = envToWorld(inPt[0], inPt[1]);
  const outW = envToWorld(outPt[0], outPt[1]);

  const sfGeo = new THREE.BufferGeometry();
  const vertices = [
    inW.x, 0.25, inW.z,
    outW.x, 0.25, outW.z,
    inW.x + 3.0, 0.25, inW.z,
    outW.x + 3.0, 0.25, outW.z,
  ];
  sfGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  sfGeo.setIndex([0, 1, 2, 1, 3, 2]);

  const sfMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const mesh = new THREE.Mesh(sfGeo, sfMat);
  trackGroup.add(mesh);
}

// ==============================================================================
// 3D Racing Vehicle Hierarchy (Accurate Scale: Length 50 px, Width 30 px, PX_TO_WORLD = 1.0)
// ==============================================================================

function buildCarModel() {
  while (carGroup.children.length > 0) {
    const obj = carGroup.children[0];
    carGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
  wheelMeshes = [];
  rearBrakeLights = [];

  // Dimensions matching Python Car (wheelbase 50 px, visual width 30 px)
  const length = 50.0;
  const width = 30.0;
  const height = 12.0;

  // Car Body Group: All body components live here so we can hide them in Cockpit view
  carBodyGroup = new THREE.Group();
  carGroup.add(carBodyGroup);

  // Materials
  const liveryMat = new THREE.MeshStandardMaterial({
    color: 0x00d4ff,
    metalness: 0.85,
    roughness: 0.22,
  });
  const carbonMat = new THREE.MeshStandardMaterial({
    color: 0x181c24,
    metalness: 0.6,
    roughness: 0.5,
  });
  const glassMat = new THREE.MeshStandardMaterial({
    color: 0x050c18,
    metalness: 0.95,
    roughness: 0.08,
    transparent: true,
    opacity: 0.85,
  });

  // 1. Main Aerodynamic Chassis / Body
  const bodyGeo = new THREE.BoxGeometry(length * 0.9, height * 0.45, width * 0.88);
  const bodyMesh = new THREE.Mesh(bodyGeo, liveryMat);
  bodyMesh.position.y = height * 0.45;
  bodyMesh.castShadow = true;
  bodyMesh.receiveShadow = true;
  carBodyGroup.add(bodyMesh);

  // 2. Cockpit Canopy / Cabin
  const cabinGeo = new THREE.BoxGeometry(length * 0.45, height * 0.45, width * 0.65);
  const cabinMesh = new THREE.Mesh(cabinGeo, glassMat);
  cabinMesh.position.set(-length * 0.05, height * 0.8, 0);
  cabinMesh.castShadow = true;
  carBodyGroup.add(cabinMesh);

  // 3. Front Splitter
  const splitterGeo = new THREE.BoxGeometry(length * 0.18, height * 0.08, width * 0.95);
  const splitter = new THREE.Mesh(splitterGeo, carbonMat);
  splitter.position.set(length * 0.46, height * 0.15, 0);
  splitter.castShadow = true;
  carBodyGroup.add(splitter);

  // 4. Rear Racing Wing
  const wingGeo = new THREE.BoxGeometry(length * 0.16, height * 0.06, width * 0.98);
  const wing = new THREE.Mesh(wingGeo, carbonMat);
  wing.position.set(-length * 0.45, height * 1.05, 0);
  wing.castShadow = true;
  carBodyGroup.add(wing);

  const wingPillarGeo = new THREE.BoxGeometry(length * 0.04, height * 0.35, width * 0.04);
  const pL = new THREE.Mesh(wingPillarGeo, carbonMat);
  pL.position.set(-length * 0.45, height * 0.85, -width * 0.32);
  const pR = new THREE.Mesh(wingPillarGeo, carbonMat);
  pR.position.set(-length * 0.45, height * 0.85, width * 0.32);
  carBodyGroup.add(pL);
  carBodyGroup.add(pR);

  // 5. Headlights
  const headLightMat = new THREE.MeshBasicMaterial({ color: 0x90e0ef });
  const headGeo = new THREE.BoxGeometry(length * 0.04, height * 0.12, width * 0.18);
  const hL = new THREE.Mesh(headGeo, headLightMat);
  hL.position.set(length * 0.46, height * 0.42, -width * 0.3);
  const hR = new THREE.Mesh(headGeo, headLightMat);
  hR.position.set(length * 0.46, height * 0.42, width * 0.3);
  carBodyGroup.add(hL);
  carBodyGroup.add(hR);

  // 6. Rear Active Brake Lights
  const brakeLightMat = new THREE.MeshBasicMaterial({ color: 0x440008 });
  const brakeLightGeo = new THREE.BoxGeometry(0.8, 1.2, 4.0);
  const brakeL = new THREE.Mesh(brakeLightGeo, brakeLightMat);
  brakeL.position.set(-length / 2 + 0.5, height * 0.5, -width * 0.32);
  const brakeR = new THREE.Mesh(brakeLightGeo, brakeLightMat);
  brakeR.position.set(-length / 2 + 0.5, height * 0.5, width * 0.32);
  carBodyGroup.add(brakeL);
  carBodyGroup.add(brakeR);
  rearBrakeLights.push(brakeL, brakeR);

  // 7. Four Wheels (Mounted on carGroup directly so wheels stay visible in Cockpit view!)
  const wheelRadius = 5.0;
  const wheelWidth = 4.0;
  const wheelRubberMat = new THREE.MeshStandardMaterial({
    color: 0x111317,
    roughness: 0.92,
    metalness: 0.08,
  });
  const wheelRimMat = new THREE.MeshStandardMaterial({
    color: 0x00f0ff,
    metalness: 0.9,
    roughness: 0.2,
  });

  function createWheel() {
    const wheelGroup = new THREE.Group();
    const tireGeo = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 18);
    tireGeo.rotateX(Math.PI / 2);
    const tireMesh = new THREE.Mesh(tireGeo, wheelRubberMat);
    tireMesh.castShadow = true;
    wheelGroup.add(tireMesh);

    const rimGeo = new THREE.CylinderGeometry(wheelRadius * 0.65, wheelRadius * 0.65, wheelWidth + 0.1, 12);
    rimGeo.rotateX(Math.PI / 2);
    const rimMesh = new THREE.Mesh(rimGeo, wheelRimMat);
    wheelGroup.add(rimMesh);

    return { group: wheelGroup, tire: tireMesh };
  }

  // Front Left Steering Pivot
  frontLeftSteerPivot = new THREE.Group();
  frontLeftSteerPivot.position.set(length * 0.32, wheelRadius, -width * 0.48);
  const wFL = createWheel();
  frontLeftSteerPivot.add(wFL.group);
  carGroup.add(frontLeftSteerPivot);
  wheelMeshes.push(wFL.tire);

  // Front Right Steering Pivot
  frontRightSteerPivot = new THREE.Group();
  frontRightSteerPivot.position.set(length * 0.32, wheelRadius, width * 0.48);
  const wFR = createWheel();
  frontRightSteerPivot.add(wFR.group);
  carGroup.add(frontRightSteerPivot);
  wheelMeshes.push(wFR.tire);

  // Rear Left
  const rearLeftGroup = new THREE.Group();
  rearLeftGroup.position.set(-length * 0.32, wheelRadius, -width * 0.48);
  const wRL = createWheel();
  rearLeftGroup.add(wRL.group);
  carGroup.add(rearLeftGroup);
  wheelMeshes.push(wRL.tire);

  // Rear Right
  const rearRightGroup = new THREE.Group();
  rearRightGroup.position.set(-length * 0.32, wheelRadius, width * 0.48);
  const wRR = createWheel();
  rearRightGroup.add(wRR.group);
  carGroup.add(rearRightGroup);
  wheelMeshes.push(wRR.tire);
}

// ==============================================================================
// 5 Dynamic Rangefinder Sensor Beams (PX_TO_WORLD = 1.0)
// ==============================================================================

function buildLidarBeams() {
  while (lidarGroup.children.length > 0) {
    const obj = lidarGroup.children[0];
    lidarGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
  rayBeamObjects = [];

  for (let i = 0; i < 5; i++) {
    // 1. Dynamic line segment
    const lineGeo = new THREE.BufferGeometry();
    const positions = new Float32Array([0, 3.5, 0, 0, 3.5, 50]);
    lineGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      linewidth: 2,
      transparent: true,
      opacity: 0.85,
    });
    const line = new THREE.Line(lineGeo, lineMat);
    lidarGroup.add(line);

    // 2. Ground Contact Ring
    const ringGeo = new THREE.RingGeometry(1.2, 1.8, 16);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    lidarGroup.add(ring);

    // 3. Spark Particle at Impact Point
    const sparkGeo = new THREE.SphereGeometry(1.0, 8, 8);
    const sparkMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const spark = new THREE.Mesh(sparkGeo, sparkMat);
    lidarGroup.add(spark);

    rayBeamObjects.push({ line, ring, spark });
  }
}

function updateLidarVisuals(rays, car) {
  if (!rays || rays.length < 5 || !carGroup) return;

  const origin = new THREE.Vector3(carGroup.position.x, 3.5, carGroup.position.z);

  for (let i = 0; i < 5; i++) {
    const ray = rays[i];
    const beamObj = rayBeamObjects[i];
    if (!beamObj || !ray) continue;

    // Renderer Guard: check start and end validity
    if (!isValidVector2(ray.start) || !isValidVector2(ray.end) || !isValidNumber(ray.distance)) {
      if (!hasWarnedNonFinite) {
        console.warn("Renderer guard: Non-finite ray geometry detected, skipping ray", ray);
        hasWarnedNonFinite = true;
      }
      continue;
    }

    const hitWorld = envToWorld(ray.end[0], ray.end[1]);
    const hit = new THREE.Vector3(hitWorld.x, 3.5, hitWorld.z);

    // Color code: Clear (>100px) Green, Closing (50-100px) Amber, Danger (<50px) Red
    let beamColor = 0x00e676;
    if (ray.distance < 50.0) {
      beamColor = 0xff2a5f;
    } else if (ray.distance < 100.0) {
      beamColor = 0xffb700;
    }

    // 1. Line Beam
    const positions = beamObj.line.geometry.attributes.position.array;
    positions[0] = origin.x;
    positions[1] = origin.y;
    positions[2] = origin.z;
    positions[3] = hit.x;
    positions[4] = hit.y;
    positions[5] = hit.z;
    beamObj.line.geometry.attributes.position.needsUpdate = true;
    beamObj.line.material.color.setHex(beamColor);

    // 2. Contact Ring & Spark
    beamObj.ring.position.set(hit.x, 0.4, hit.z);
    beamObj.ring.material.color.setHex(beamColor);
    beamObj.spark.position.set(hit.x, 0.6, hit.z);
  }
}

// ==============================================================================
// 3D Trajectory Ribbon (With Lap-Wrap & Reset Clearing)
// ==============================================================================

function clearTrajectoryTrail() {
  STATE.trajectoryPoints = [];
  if (trajectoryMesh) {
    scene.remove(trajectoryMesh);
    if (trajectoryMesh.geometry) trajectoryMesh.geometry.dispose();
    trajectoryMesh = null;
  }
}

function updateTrajectoryRibbon(car) {
  if (!car || !isValidNumber(car.x) || !isValidNumber(car.y)) return;

  // Rule 5: Clear trail buffer on lap wrap
  const currentProgress = Number(car.lap_progress) || 0.0;
  if (currentProgress < STATE.prevLapProgress - 0.5) {
    clearTrajectoryTrail();
  }
  STATE.prevLapProgress = currentProgress;

  const w = envToWorld(car.x, car.y);
  const newPt = new THREE.Vector3(w.x, 0.3, w.z);

  // Jump detection: skip / clear if distance between points > 80 px (teleport/reset)
  if (STATE.trajectoryPoints.length > 0) {
    const lastPt = STATE.trajectoryPoints[STATE.trajectoryPoints.length - 1];
    const dist = lastPt.distanceTo(newPt);
    if (dist > 80.0) {
      clearTrajectoryTrail();
    }
  }

  STATE.trajectoryPoints.push(newPt);
  if (STATE.trajectoryPoints.length > 500) {
    STATE.trajectoryPoints.shift();
  }

  if (STATE.trajectoryPoints.length < 2) return;

  if (!trajectoryMesh) {
    const geo = new THREE.BufferGeometry().setFromPoints(STATE.trajectoryPoints);
    const mat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.65,
      linewidth: 2,
    });
    trajectoryMesh = new THREE.Line(geo, mat);
    scene.add(trajectoryMesh);
  } else {
    trajectoryMesh.geometry.setFromPoints(STATE.trajectoryPoints);
  }
}

// ==============================================================================
// Cinematic Camera Director (Rule 6: Cockpit Mode Hides Body Mesh)
// ==============================================================================

function updateCameraDirector(car, velocity) {
  if (!carGroup || !camera) return;

  const carPos = carGroup.position;
  const heading = carGroup.rotation.y;
  const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
  const backward = forward.clone().negate();

  if (STATE.activeCameraMode === "cockpit") {
    if (controls) controls.enabled = false;
    // Rule 6: Hide car body mesh in cockpit mode (keep wheels visible)
    if (carBodyGroup) carBodyGroup.visible = false;

    // Driver eye position: above and ahead of body centre
    const eyePos = new THREE.Vector3(
      carPos.x + forward.x * 8.0,
      10.0,
      carPos.z + forward.z * 8.0
    );
    const lookTarget = new THREE.Vector3(
      carPos.x + forward.x * 120.0,
      8.0,
      carPos.z + forward.z * 120.0
    );

    camera.position.copy(eyePos);
    camera.lookAt(lookTarget);
    camera.up.set(0, 1, 0);

  } else {
    // Show car body mesh in chase, orbit, overhead modes
    if (carBodyGroup) carBodyGroup.visible = true;

    if (STATE.activeCameraMode === "chase") {
      if (controls) controls.enabled = false;
      const followDist = 85.0;
      const followHeight = 36.0 + Math.min(Math.abs(velocity) * 2.0, 10.0);
      const targetCamPos = new THREE.Vector3(
        carPos.x + backward.x * followDist,
        followHeight,
        carPos.z + backward.z * followDist
      );
      const lookTarget = new THREE.Vector3(
        carPos.x + forward.x * 35.0,
        10.0,
        carPos.z + forward.z * 35.0
      );

      STATE.cameraPos.lerp(targetCamPos, 0.12);
      STATE.cameraTarget.lerp(lookTarget, 0.15);

      camera.position.copy(STATE.cameraPos);
      camera.lookAt(STATE.cameraTarget);
      camera.up.set(0, 1, 0);

    } else if (STATE.activeCameraMode === "orbit") {
      if (controls) {
        controls.enabled = true;
        controls.target.set(carPos.x, 6.0, carPos.z);
        controls.update();
      }

    } else if (STATE.activeCameraMode === "overhead") {
      if (controls) controls.enabled = false;
      // Top-down circuit view centered at track origin (0, 750, 0), North is up
      camera.position.set(0, 750.0, 0);
      camera.up.set(0, 0, -1);
      camera.lookAt(0, 0, 0);
    }
  }
}

// ==============================================================================
// Debug Inspector Overlay (Key: D)
// ==============================================================================

function buildDebugHelpers() {
  trackAxesHelper = new THREE.AxesHelper(80);
  trackAxesHelper.position.set(0, 0.5, 0);
  debugGroup.add(trackAxesHelper);

  carAxesHelper = new THREE.AxesHelper(30);
  carAxesHelper.position.set(0, 3.5, 0);
  carGroup.add(carAxesHelper);
}

function updateDebugOverlay(car) {
  const overlay = document.getElementById("debugStatsOverlay");
  if (!overlay) return;

  if (!STATE.debugMode) {
    overlay.style.display = "none";
    debugGroup.visible = false;
    if (carAxesHelper) carAxesHelper.visible = false;
    return;
  }

  overlay.style.display = "block";
  debugGroup.visible = true;
  if (carAxesHelper) carAxesHelper.visible = true;

  const w = envToWorld(car.x, car.y);
  overlay.innerHTML = `
    <div class="debug-title">DEBUG INSPECTOR (KEY: D)</div>
    <div class="debug-row"><span>Scale:</span> <strong>PX_TO_WORLD = 1.0</strong></div>
    <div class="debug-row"><span>Track Width:</span> <strong>70 px</strong></div>
    <div class="debug-row"><span>Car Dimensions:</span> <strong>50L x 30W x 12H px</strong></div>
    <div class="debug-row"><span>Env Pos:</span> <strong>(${car.x.toFixed(1)}, ${car.y.toFixed(1)})</strong></div>
    <div class="debug-row"><span>Env Heading:</span> <strong>${car.angle.toFixed(1)}°</strong></div>
    <div class="debug-row"><span>World Pos:</span> <strong>(${w.x.toFixed(1)}, ${w.z.toFixed(1)})</strong></div>
    <div class="debug-row"><span>Three.js RotY:</span> <strong>${carGroup.rotation.y.toFixed(3)} rad</strong></div>
    <div class="debug-row"><span>Active Cam:</span> <strong>${STATE.activeCameraMode.toUpperCase()}</strong></div>
    <div class="debug-row"><span>Body Visible:</span> <strong>${carBodyGroup ? carBodyGroup.visible : true}</strong></div>
    <div class="debug-row"><span>FPS:</span> <strong>${currentFPS}</strong></div>
  `;
}

// ==============================================================================
// Animation Loop & Render Pipeline
// ==============================================================================

let lastFrameTime = performance.now();
let frameCount = 0;
let lastFpsUpdate = performance.now();
let currentFPS = 60;

function animate(currentTime) {
  requestAnimationFrame(animate);

  const delta = (currentTime - lastFrameTime) * 0.001;
  lastFrameTime = currentTime;

  frameCount++;
  if (currentTime - lastFpsUpdate >= 500) {
    currentFPS = Math.round((frameCount * 1000) / (currentTime - lastFpsUpdate));
    frameCount = 0;
    lastFpsUpdate = currentTime;
    const fpsEl = document.getElementById("fpsCounter");
    if (fpsEl) fpsEl.innerText = `${currentFPS} FPS`;
  }

  // Wheels spin animation proportional to velocity
  const vel = STATE.lastCarVelocity || 0.0;
  wheelMeshes.forEach((wheel) => {
    wheel.rotation.x -= vel * 0.2;
  });

  // Render 3D Scene
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

// ==============================================================================
// Simulation Backend Communication (FastAPI Client)
// ==============================================================================

async function startSimulation(presetId, trackSeed) {
  try {
    updateStatusPill("INITIALIZING", "amber");
    const res = await fetch("/simulation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset_id: presetId, track_seed: trackSeed }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    STATE.presetId = presetId;
    STATE.trackSeed = trackSeed;
    clearTrajectoryTrail();
    resetCommentaryFeed();

    build3DTrack(data.track);
    syncUIWithState(data);
    updateStatusPill("READY", "cyan");
  } catch (err) {
    console.error("Simulation start failed:", err);
    updateStatusPill("ERROR", "red");
  }
}

async function stepSimulation() {
  if (!STATE.simRunning) return;

  try {
    const res = await fetch("/simulation/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        steps: STATE.simSpeed,
        calculate_saliency: STATE.calculateSaliency,
        commentary_mode: STATE.commentaryMode,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    syncUIWithState(data);

    if (STATE.simRunning && !data.session.is_crashed && !data.session.is_completed) {
      setTimeout(stepSimulation, 30);
    }
  } catch (err) {
    console.error("Simulation step failed:", err);
    pauseSimulation();
    updateStatusPill("STEP ERROR", "red");
  }
}

async function resetSimulation() {
  try {
    const res = await fetch("/simulation/reset", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    clearTrajectoryTrail();
    resetCommentaryFeed();
    syncUIWithState(data);
    updateStatusPill("RESET", "cyan");
    hideOverlayModal();
  } catch (err) {
    console.error("Simulation reset failed:", err);
  }
}

// ==============================================================================
// State Synchronization & UI Updating
// ==============================================================================

function syncUIWithState(data) {
  if (!data) return;

  const session = data.session || {};
  const car = data.vehicle || {};
  const telemetry = data.telemetry || {};

  // Rule 4: Renderer Guard: skip update if car coordinates are non-finite
  if (!car || !isValidNumber(car.x) || !isValidNumber(car.y) || !isValidNumber(car.angle) || !isValidNumber(car.velocity)) {
    if (!hasWarnedNonFinite) {
      console.warn("Renderer guard: Non-finite vehicle state detected, skipping frame", car);
      hasWarnedNonFinite = true;
    }
    return;
  }

  STATE.lastCarVelocity = car.velocity;

  // 1. Update 3D Vehicle Transform
  const carWorld = envToWorld(car.x, car.y);
  carGroup.position.set(carWorld.x, 0.0, carWorld.z);

  // envHeadingToWorld converts degrees to radians without sign flip
  carGroup.rotation.y = envHeadingToWorld(car.angle);

  // 2. Front Wheel Steering Angle Animation
  const steerAction = Number(car.steering) || 0.0;
  const steerRad = -steerAction * 0.45; // Negative rotates front wheels inward into turn
  if (frontLeftSteerPivot) frontLeftSteerPivot.rotation.y = steerRad;
  if (frontRightSteerPivot) frontRightSteerPivot.rotation.y = steerRad;

  // 3. Dynamic Brake Light Illumination
  const isDecel = car.throttle < -0.05 || car.action_mode === "deceleration";
  rearBrakeLights.forEach((light) => {
    light.material.color.setHex(isDecel ? 0xff0020 : 0x440008);
  });

  // 4. Update 5 Rangefinder Sensor Beams
  if (data.rays) {
    updateLidarVisuals(data.rays, car);
  }

  // 5. Update 3D Trajectory Ribbon (with lap-wrap clearing)
  updateTrajectoryRibbon(car);

  // 6. Update Camera Director
  updateCameraDirector(car, car.velocity);

  // 7. Update Debug Overlay
  updateDebugOverlay(car);

  // 8. Synchronize HTML Cockpit Gauges & Telemetry Panels
  syncCockpitAndGauges(car, session, telemetry);

  // 9. AI Race Engineer Commentary Feed (Rule 7: Appending DOM nodes without wiping)
  if (data.commentary) {
    updateCommentaryFeed(data.commentary);
  }

  // 10. Policy Saliency Sensitivity Bars
  if (data.saliency) {
    STATE.currentSaliency = data.saliency;
    renderSaliencyBars();
  }

  // 11. 2D Circuit Radar Minimap
  if (STATE.trackData && minimapCtx) {
    renderRadarMinimap(STATE.trackData, car);
  }

  // 12. Lap Complete or Crash Modals
  if (session.is_completed) {
    showOverlayModal("LAP COMPLETED!", `Clean lap finished at step ${session.step_count}.`, data.lap_review);
    pauseSimulation();
  } else if (session.is_crashed) {
    showOverlayModal("CRASH DETECTED", `Vehicle departed track boundary at step ${session.step_count}.`, data.lap_review);
    pauseSimulation();
  }
}

function syncCockpitAndGauges(car, session, telemetry) {
  // Speed
  const vel = Number(car.velocity) || 0.0;
  const valSpeed = document.getElementById("valSpeed");
  const barSpeed = document.getElementById("barSpeed");
  const cockpitSpeed = document.getElementById("cockpitSpeed");
  if (valSpeed) valSpeed.innerText = vel.toFixed(2);
  if (cockpitSpeed) cockpitSpeed.innerText = vel.toFixed(2);
  if (barSpeed) barSpeed.style.width = `${Math.min(100, Math.max(0, (vel / 4.0) * 100))}%`;

  // Steering
  const steerAct = Number(car.steering) || 0.0;
  const steerDeg = Number(car.steering_angle_deg) || 0.0;
  const valSteering = document.getElementById("valSteering");
  const barSteering = document.getElementById("barSteering");
  const cockpitSteerBar = document.getElementById("cockpitSteerBar");
  if (valSteering) {
    valSteering.innerText = `${steerDeg >= 0 ? "+" : ""}${steerDeg.toFixed(1)}°`;
  }
  if (barSteering) {
    const steerWidth = Math.min(50, Math.abs(steerAct) * 50);
    barSteering.style.width = `${steerWidth}%`;
    barSteering.style.left = steerAct >= 0 ? "50%" : `${50 - steerWidth}%`;
  }
  if (cockpitSteerBar) {
    const steerWidth = Math.min(50, Math.abs(steerAct) * 50);
    cockpitSteerBar.style.width = `${steerWidth}%`;
    cockpitSteerBar.style.left = steerAct >= 0 ? "50%" : `${50 - steerWidth}%`;
  }

  // Throttle
  const throttleVal = Number(car.throttle) || 0.0;
  const valThrottle = document.getElementById("valThrottle");
  const barThrottle = document.getElementById("barThrottle");
  const cockpitThrottleBar = document.getElementById("cockpitThrottleBar");
  const cockpitActionMode = document.getElementById("cockpitActionMode");

  if (valThrottle) {
    const modeStr = car.action_mode ? car.action_mode.toUpperCase() : "COASTING";
    valThrottle.innerHTML = `${throttleVal >= 0 ? "+" : ""}${throttleVal.toFixed(2)} <span class="unit">${modeStr}</span>`;
  }
  if (cockpitActionMode) cockpitActionMode.innerText = car.action_mode ? car.action_mode.slice(0, 5).toUpperCase() : "COAST";
  if (barThrottle) {
    const throtWidth = Math.min(50, Math.abs(throttleVal) * 50);
    barThrottle.style.width = `${throtWidth}%`;
    barThrottle.style.left = throttleVal >= 0 ? "50%" : `${50 - throtWidth}%`;
    barThrottle.style.background = throttleVal >= 0 ? "var(--accent-green)" : "var(--accent-red)";
  }
  if (cockpitThrottleBar) {
    const throtWidth = Math.min(50, Math.abs(throttleVal) * 50);
    cockpitThrottleBar.style.width = `${throtWidth}%`;
    cockpitThrottleBar.style.left = throttleVal >= 0 ? "50%" : `${50 - throtWidth}%`;
    cockpitThrottleBar.style.background = throttleVal >= 0 ? "var(--accent-green)" : "var(--accent-red)";
  }

  // Lap Progress
  const valProgress = document.getElementById("valProgress");
  const barProgress = document.getElementById("barProgress");
  const progPct = (Number(car.lap_progress) || 0.0) * 100;
  if (valProgress) valProgress.innerText = `${progPct.toFixed(1)}%`;
  if (barProgress) barProgress.style.width = `${Math.min(100, Math.max(0, progPct))}%`;

  // Telemetry Readouts
  const valHeadingErr = document.getElementById("valHeadingErr");
  const valSignedDist = document.getElementById("valSignedDist");
  const valSlip = document.getElementById("valSlip");
  const valPos = document.getElementById("valPos");
  const stepClock = document.getElementById("stepClock");

  if (valHeadingErr) valHeadingErr.innerText = `${(telemetry.heading_error || 0).toFixed(1)}°`;
  if (valSignedDist) valSignedDist.innerText = `${(telemetry.signed_distance || 0).toFixed(1)} px`;
  if (valSlip) valSlip.innerText = `${(telemetry.slip_angle || 0).toFixed(1)}°`;
  if (valPos) valPos.innerText = `${car.x.toFixed(1)}, ${car.y.toFixed(1)}`;
  if (stepClock) stepClock.innerText = `STEP: ${session.step_count || 0} / ${session.max_steps || 700}`;

  // 5 Ray Sensors HUD Bar & Sidebar List
  if (telemetry.rays && telemetry.rays.length === 5) {
    for (let i = 0; i < 5; i++) {
      const rayVal = telemetry.rays[i];
      const hudTag = document.getElementById(`hudRay${i}`);
      if (hudTag) {
        const valSpan = hudTag.querySelector(".beam-val");
        if (valSpan) valSpan.innerText = `${rayVal.toFixed(0)}px`;
        hudTag.classList.toggle("close", rayVal < 50.0);
      }
    }

    const sensorsList = document.getElementById("sensorsList");
    if (sensorsList) {
      let html = "";
      for (let i = 0; i < 5; i++) {
        const rayVal = telemetry.rays[i];
        const pct = Math.min(100, Math.max(0, (rayVal / 200.0) * 100));
        let col = "var(--accent-cyan)";
        if (rayVal < 50.0) col = "var(--accent-red)";
        else if (rayVal < 100.0) col = "var(--accent-amber)";

        html += `
          <div class="sensor-row">
            <span class="sensor-lbl">${RAY_LABELS[i]}</span>
            <div class="sensor-bar-wrap">
              <div class="sensor-bar" style="width: ${pct}%; background: ${col};"></div>
            </div>
            <span class="sensor-val" style="color: ${col};">${rayVal.toFixed(1)} px</span>
          </div>
        `;
      }
      sensorsList.innerHTML = html;
    }
  }
}

// ==============================================================================
// AI Race Engineer Commentary Feed (Rule 7: Append DOM Nodes, Keep 50, No innerHTML="")
// ==============================================================================

function resetCommentaryFeed() {
  const feed = document.getElementById("commentaryFeed");
  if (!feed) return;
  STATE.seenCommentaryKeys.clear();
  feed.innerHTML = `
    <div class="commentary-item welcome-item">
      <span class="c-step">[STEP 0]</span>
      <span class="c-source">[AI ENGINEER]</span>
      <span class="c-text">Autonomous racing policy loaded. System prepped on start line. Ready to engage.</span>
    </div>
  `;
}

function updateCommentaryFeed(commentaryList) {
  const feed = document.getElementById("commentaryFeed");
  if (!feed || !Array.isArray(commentaryList)) return;

  let addedCount = 0;
  for (const item of commentaryList) {
    const key = `${item.step}_${item.event_type || ""}`;
    if (STATE.seenCommentaryKeys.has(key)) continue;
    STATE.seenCommentaryKeys.add(key);

    const div = document.createElement("div");
    div.className = "commentary-item";
    const src = (item.source || "OFFLINE").toUpperCase().replace(/_/g, " ");
    const text = item.commentary || item.text || "";
    div.innerHTML = `
      <span class="c-step">[STEP ${item.step}]</span>
      <span class="c-source">[${src}]</span>
      <span class="c-text">${text}</span>
    `;
    feed.appendChild(div);
    addedCount++;
  }

  // Keep last 50 DOM entries
  while (feed.children.length > 50) {
    feed.removeChild(feed.firstChild);
  }

  if (addedCount > 0) {
    feed.scrollTop = feed.scrollHeight;
  }
}

// ==============================================================================
// Policy Saliency Sensitivity Bars (Rule 1: Reads Backend Keys)
// ==============================================================================

function renderSaliencyBars() {
  const container = document.getElementById("saliencyBarsContainer") || document.getElementById("saliencyBars");
  const topSteerEl = document.getElementById("topSteerRay");
  const topThrottleEl = document.getElementById("topThrottleRay");
  if (!container) return;

  const sal = STATE.currentSaliency;
  if (!sal) {
    container.innerHTML = '<div class="saliency-empty">Awaiting live saliency compute...</div>';
    if (topSteerEl) topSteerEl.innerText = "--";
    if (topThrottleEl) topThrottleEl.innerText = "--";
    return;
  }

  // Read backend keys exactly as sent by inference/simulation.py
  const steerSal = sal.feature_saliency_steer || {};
  const longSal = sal.feature_saliency_long || {};

  if (topSteerEl) {
    topSteerEl.innerText = sal.top_steering_ray || "--";
  }
  if (topThrottleEl) {
    topThrottleEl.innerText = sal.top_longitudinal_ray || "--";
  }

  let activeDict = steerSal;
  if (STATE.saliencyTab === "long") activeDict = longSal;
  else if (STATE.saliencyTab === "hist" && STATE.historicalSaliency) {
    activeDict = STATE.historicalSaliency.mean_saliency_overall || steerSal;
  }

  // Find max value for normalization
  const values = Object.values(activeDict).map((v) => Number(v) || 0.0);
  const maxVal = Math.max(...values, 0.001);

  let html = "";
  FEATURE_NAMES.forEach((feat) => {
    const val = Number(activeDict[feat]) || 0.0;
    const pct = Math.min(100, Math.max(0, (val / maxVal) * 100));
    const label = feat.replace("ray_", "r_").replace("signed_distance", "dist").replace("heading_error", "head_err");

    html += `
      <div class="sal-row">
        <span class="sal-label" title="${feat}">${label}</span>
        <div class="sal-track">
          <div class="sal-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="sal-val">${val.toFixed(4)}</span>
      </div>
    `;
  });
  container.innerHTML = html;
}

async function loadHistoricalSaliency() {
  try {
    const res = await fetch("/simulation/historical-saliency");
    if (res.ok) {
      STATE.historicalSaliency = await res.json();
    }
  } catch (err) {
    console.warn("Could not load historical saliency benchmark:", err);
  }
}

// ==============================================================================
// 2D Circuit Radar Minimap (Identical Orientation to 3D Overhead View)
// ==============================================================================

function renderRadarMinimap(trackData, car) {
  if (!minimapCtx || !minimapCanvas || !trackData) return;
  const ctx = minimapCtx;
  const w = minimapCanvas.width;
  const h = minimapCanvas.height;

  ctx.clearRect(0, 0, w, h);

  const bounds = trackData.bounds || { min_x: 0, max_x: 800, min_y: 0, max_y: 600 };
  const spanX = Math.max(bounds.max_x - bounds.min_x, 100);
  const spanY = Math.max(bounds.max_y - bounds.min_y, 100);
  const scale = Math.min((w - 24) / spanX, (h - 24) / spanY);
  const ox = 12 + ((w - 24) - spanX * scale) / 2;
  const oy = 12 + ((h - 24) - spanY * scale) / 2;

  function toMap(x, y) {
    return [ox + (x - bounds.min_x) * scale, oy + (y - bounds.min_y) * scale];
  }

  // 1. Draw outer boundary
  const outer = trackData.outer_boundary || [];
  if (outer.length > 2) {
    ctx.beginPath();
    ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
    ctx.lineWidth = 1.5;
    const [p0x, p0y] = toMap(outer[0][0], outer[0][1]);
    ctx.moveTo(p0x, p0y);
    for (let i = 1; i < outer.length; i++) {
      const [px, py] = toMap(outer[i][0], outer[i][1]);
      ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.stroke();
  }

  // 2. Draw inner boundary
  const inner = trackData.inner_boundary || [];
  if (inner.length > 2) {
    ctx.beginPath();
    ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
    ctx.lineWidth = 1.5;
    const [p0x, p0y] = toMap(inner[0][0], inner[0][1]);
    ctx.moveTo(p0x, p0y);
    for (let i = 1; i < inner.length; i++) {
      const [px, py] = toMap(inner[i][0], inner[i][1]);
      ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.stroke();
  }

  // 3. Draw centerline
  const center = trackData.centerline || [];
  if (center.length > 2) {
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    const [p0x, p0y] = toMap(center[0][0], center[0][1]);
    ctx.moveTo(p0x, p0y);
    for (let i = 1; i < center.length; i++) {
      const [px, py] = toMap(center[i][0], center[i][1]);
      ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 4. Draw car position & heading dot
  if (car && isValidNumber(car.x) && isValidNumber(car.y)) {
    const [cx, cy] = toMap(car.x, car.y);

    // Car dot
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "#00f0ff";
    ctx.shadowColor = "#00f0ff";
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Heading indicator line
    const rad = (Number(car.angle) || 0) * (Math.PI / 180.0);
    const hx = cx + Math.cos(rad) * 10;
    const hy = cy - Math.sin(rad) * 10;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(hx, hy);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

// ==============================================================================
// UI Control Handlers
// ==============================================================================

function pauseSimulation() {
  STATE.simRunning = false;
  const btn = document.getElementById("btnStartPause");
  if (btn) {
    btn.innerHTML = '<span class="icon">▶</span> START';
    btn.classList.remove("active");
  }
  updateStatusPill("PAUSED", "amber");
}

function updateStatusPill(text, color) {
  const dot = document.getElementById("statusDot");
  const txt = document.getElementById("statusText");
  if (dot) dot.className = `pill-dot ${color}`;
  if (txt) txt.innerText = text;
}

function showOverlayModal(title, text, lapReview) {
  const modal = document.getElementById("canvasOverlay");
  const mTitle = document.getElementById("modalTitle");
  const mText = document.getElementById("modalText");
  const mScore = document.getElementById("modalScore");
  if (!modal) return;

  if (mTitle) mTitle.innerText = title;
  if (mText) mText.innerText = text;

  if (lapReview && mScore) {
    mScore.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 11px; margin-top: 8px;">
        Avg Speed: ${lapReview.average_speed || "--"} px/step |
        Smoothness: ${lapReview.steering_smoothness_score || "--"}%
      </div>
    `;
  }
  modal.style.display = "flex";
}

function hideOverlayModal() {
  const modal = document.getElementById("canvasOverlay");
  if (modal) modal.style.display = "none";
}

function setCameraMode(mode) {
  STATE.activeCameraMode = mode;
  ["btnCamChase", "btnCamCockpit", "btnCamOrbit", "btnCamOverhead"].forEach((id) => {
    const b = document.getElementById(id);
    if (b) b.classList.remove("active");
  });

  const activeBtnMap = {
    chase: "btnCamChase",
    cockpit: "btnCamCockpit",
    orbit: "btnCamOrbit",
    overhead: "btnCamOverhead",
  };
  const activeBtn = document.getElementById(activeBtnMap[mode]);
  if (activeBtn) activeBtn.classList.add("active");

  // Rule 6: Handle car body visibility switch immediately
  if (mode === "cockpit") {
    if (carBodyGroup) carBodyGroup.visible = false;
    if (controls) controls.enabled = false;
  } else {
    if (carBodyGroup) carBodyGroup.visible = true;
    if (mode === "orbit") {
      if (controls && carGroup) {
        controls.enabled = true;
        camera.position.set(carGroup.position.x - 70.0, 50.0, carGroup.position.z + 70.0);
        controls.target.set(carGroup.position.x, 6.0, carGroup.position.z);
        controls.update();
      }
    } else if (controls) {
      controls.enabled = false;
    }
  }

  // Force immediate camera update if car exists
  if (STATE.lastCarVelocity !== undefined) {
    updateCameraDirector(
      { x: carGroup.position.x, y: carGroup.position.z, angle: carGroup.rotation.y * 180 / Math.PI },
      STATE.lastCarVelocity
    );
  }
}

function initEventHandlers() {
  // Camera buttons
  const btnChase = document.getElementById("btnCamChase");
  const btnCockpit = document.getElementById("btnCamCockpit");
  const btnOrbit = document.getElementById("btnCamOrbit");
  const btnOverhead = document.getElementById("btnCamOverhead");

  if (btnChase) btnChase.addEventListener("click", () => setCameraMode("chase"));
  if (btnCockpit) btnCockpit.addEventListener("click", () => setCameraMode("cockpit"));
  if (btnOrbit) btnOrbit.addEventListener("click", () => setCameraMode("orbit"));
  if (btnOverhead) btnOverhead.addEventListener("click", () => setCameraMode("overhead"));

  // Start / Pause
  const btnStartPause = document.getElementById("btnStartPause");
  if (btnStartPause) {
    btnStartPause.addEventListener("click", () => {
      STATE.simRunning = !STATE.simRunning;
      if (STATE.simRunning) {
        btnStartPause.innerHTML = '<span class="icon">⏸</span> PAUSE';
        btnStartPause.classList.add("active");
        updateStatusPill("RUNNING", "green");
        hideOverlayModal();
        stepSimulation();
      } else {
        pauseSimulation();
      }
    });
  }

  // Step 1 Button
  const btnStep1 = document.getElementById("btnStep1");
  if (btnStep1) {
    btnStep1.addEventListener("click", async () => {
      pauseSimulation();
      const res = await fetch("/simulation/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          steps: 1,
          calculate_saliency: true,
          commentary_mode: STATE.commentaryMode,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        syncUIWithState(data);
      }
    });
  }

  // Reset Button
  const btnReset = document.getElementById("btnReset");
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      pauseSimulation();
      resetSimulation();
    });
  }

  // Preset Selector
  const selTrack = document.getElementById("selectTrackPreset");
  if (selTrack) {
    selTrack.addEventListener("change", (e) => {
      pauseSimulation();
      const val = e.target.value;
      const seedInput = document.getElementById("inputSeed");
      const seed = seedInput ? parseInt(seedInput.value, 10) || 1001 : 1001;
      startSimulation(val, seed);
    });
  }

  // Seed Input & New Seed Button
  const btnNewSeed = document.getElementById("btnNewSeed");
  if (btnNewSeed) {
    btnNewSeed.addEventListener("click", () => {
      const seedInput = document.getElementById("inputSeed");
      const newSeed = Math.floor(Math.random() * 900000) + 1000;
      if (seedInput) seedInput.value = newSeed;
      pauseSimulation();
      const preset = selTrack ? selTrack.value : "standard_medium";
      startSimulation(preset, newSeed);
    });
  }

  // Sim Speed Buttons
  ["btnSpeed1x", "btnSpeed2x", "btnSpeed4x"].forEach((id, idx) => {
    const btn = document.getElementById(id);
    const speeds = [1, 2, 4];
    if (btn) {
      btn.addEventListener("click", () => {
        ["btnSpeed1x", "btnSpeed2x", "btnSpeed4x"].forEach((bId) => {
          const b = document.getElementById(bId);
          if (b) b.classList.remove("active");
        });
        btn.classList.add("active");
        STATE.simSpeed = speeds[idx];
      });
    }
  });

  // Debug Toggle Button & Key D Listener
  const btnDebug = document.getElementById("btnDebugToggle");
  function toggleDebug() {
    STATE.debugMode = !STATE.debugMode;
    if (btnDebug) btnDebug.classList.toggle("active", STATE.debugMode);
    if (carGroup) {
      updateDebugOverlay({
        x: carGroup.position.x + STATE.trackCenter.x,
        y: carGroup.position.z + STATE.trackCenter.y,
        angle: (carGroup.rotation.y * 180.0) / Math.PI,
      });
    }
  }

  if (btnDebug) btnDebug.addEventListener("click", toggleDebug);
  window.addEventListener("keydown", (e) => {
    if (e.key === "d" || e.key === "D") {
      toggleDebug();
    }
  });

  // Saliency Tabs
  const tabSteer = document.getElementById("tabSteer");
  const tabLong = document.getElementById("tabLong");
  const tabHist = document.getElementById("tabHist");

  if (tabSteer) {
    tabSteer.addEventListener("click", () => {
      STATE.saliencyTab = "steer";
      [tabSteer, tabLong, tabHist].forEach((t) => t && t.classList.remove("active"));
      tabSteer.classList.add("active");
      renderSaliencyBars();
    });
  }
  if (tabLong) {
    tabLong.addEventListener("click", () => {
      STATE.saliencyTab = "long";
      [tabSteer, tabLong, tabHist].forEach((t) => t && t.classList.remove("active"));
      tabLong.classList.add("active");
      renderSaliencyBars();
    });
  }
  if (tabHist) {
    tabHist.addEventListener("click", () => {
      STATE.saliencyTab = "hist";
      [tabSteer, tabLong, tabHist].forEach((t) => t && t.classList.remove("active"));
      tabHist.classList.add("active");
      renderSaliencyBars();
    });
  }

  // Saliency Calculation Checkbox
  const checkSaliency = document.getElementById("checkSaliency");
  if (checkSaliency) {
    checkSaliency.addEventListener("change", (e) => {
      STATE.calculateSaliency = e.target.checked;
    });
  }

  // Gemini Commentary Toggle Checkbox
  const checkGemini = document.getElementById("checkGemini") || document.getElementById("checkGeminiCommentary");
  if (checkGemini) {
    checkGemini.addEventListener("change", (e) => {
      STATE.commentaryMode = e.target.checked ? "gemini" : "offline";
      const badge = document.getElementById("commentaryModeBadge");
      if (badge) badge.innerText = e.target.checked ? "GEMINI FLASH LIVE" : "OFFLINE DETERMINISTIC";
    });
  }

  // Engineer Drawer Toggle Button
  const btnToggleDrawer = document.getElementById("btnToggleDrawer");
  const drawerBody = document.getElementById("engineerDrawerBody");
  const drawerIcon = document.getElementById("drawerToggleIcon");
  if (btnToggleDrawer && drawerBody) {
    btnToggleDrawer.addEventListener("click", () => {
      const isHidden = drawerBody.style.display === "none";
      drawerBody.style.display = isHidden ? "block" : "none";
      if (drawerIcon) drawerIcon.innerText = isHidden ? "▼" : "▲";
    });
  }

  // Overlay Modal Buttons
  const btnOverlayRestart = document.getElementById("btnOverlayRestart");
  if (btnOverlayRestart) {
    btnOverlayRestart.addEventListener("click", () => {
      hideOverlayModal();
      resetSimulation();
    });
  }
}

// ==============================================================================
// Dashboard Lifecycle Initialization
// ==============================================================================

window.addEventListener("DOMContentLoaded", async () => {
  initThreeJS();
  initEventHandlers();
  loadHistoricalSaliency();

  // Create floating debug stats HUD element if missing
  if (!document.getElementById("debugStatsOverlay")) {
    const debugBox = document.createElement("div");
    debugBox.id = "debugStatsOverlay";
    debugBox.className = "debug-stats-overlay";
    debugBox.style.display = "none";
    const container = document.getElementById("canvasContainer");
    if (container) container.appendChild(debugBox);
  }

  // Start initial default simulation
  await startSimulation(STATE.presetId, STATE.trackSeed);

  // Start animation loop
  requestAnimationFrame(animate);
});
