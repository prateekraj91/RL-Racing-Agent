/**
 * RL Racing Agent // Cinematic 3D Autonomous Simulation Dashboard
 *
 * Features:
 * - Real-time Three.js 3D renderer of authentic Python RacingEnv physics
 * - 4 Cinematic Camera Modes (Chase, Cockpit, Orbit, Overhead)
 * - Steered front wheels, aerodynamic prototype vehicle chassis, dynamic wheel spin
 * - Grounded action indicators (active brake light flares & headlight projectors)
 * - 5 dynamic rangefinder sensor beams with distance-dependent colors and boundary hit rings
 * - 3D trajectory ribbon behind vehicle
 * - Real-time AI Race Engineer commentary & comprehensive post-lap review cards
 * - Real-time telemetry cockpit HUD & 2D circuit radar minimap
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

  // Track geometry cache
  trackGeometry: null,
  trackCenter: { x: 0, z: 0 },

  // Historical data
  trajectory: [],
  history: [],
  lapReview: null,

  // Camera smooth targets
  cameraPos: new THREE.Vector3(0, 40, 60),
  cameraTarget: new THREE.Vector3(0, 0, 0),

  // Saliency cache
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

// ==============================================================================
// Three.js 3D Scene Architecture
// ==============================================================================

let scene, camera, renderer, controls;
let threeCanvas, minimapCanvas, minimapCtx;
let trackGroup, carGroup, trajectoryMesh, lidarGroup;
let frontLeftSteerPivot, frontRightSteerPivot;
let wheelMeshes = [];
let rearBrakeLights = [];
let rearBrakeLightSource;
let headlightProjectors = [];
let rayBeamObjects = [];

function initThreeJS() {
  threeCanvas = document.getElementById("threeCanvas");
  minimapCanvas = document.getElementById("raceCanvas");
  if (minimapCanvas) {
    minimapCtx = minimapCanvas.getContext("2d");
  }

  const container = document.getElementById("canvasContainer");
  const width = container.clientWidth || 840;
  const height = container.clientHeight || 620;

  // Scene & Atmosphere
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07090e);
  scene.fog = new THREE.FogExp2(0x07090e, 0.0012);

  // Perspective Camera
  camera = new THREE.PerspectiveCamera(52, width / height, 0.5, 4000);
  camera.position.set(0, 50, 80);

  // High-performance WebGL Renderer
  renderer = new THREE.WebGLRenderer({
    canvas: threeCanvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // Orbit Controls (for free inspection mode)
  if (THREE.OrbitControls) {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxDistance = 1500;
    controls.minDistance = 5;
    controls.enabled = false; // Enabled only in Orbit mode
  }

  // Lighting
  setupLighting();

  // Root Groups
  trackGroup = new THREE.Group();
  scene.add(trackGroup);

  lidarGroup = new THREE.Group();
  scene.add(lidarGroup);

  // Build Car Hierarchy
  buildCarModel();

  // Build 5 Lidar Ray Emitters
  buildLidarBeams();

  // Resize handler
  window.addEventListener("resize", onWindowResize);
}

function setupLighting() {
  // Ambient Soft Light
  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);

  // Primary Stadium Floodlight (Casts Shadows)
  const sunLight = new THREE.DirectionalLight(0xdcf0ff, 1.3);
  sunLight.position.set(200, 450, 200);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 2048;
  sunLight.shadow.mapSize.height = 2048;
  sunLight.shadow.camera.near = 50;
  sunLight.shadow.camera.far = 1600;
  const d = 600;
  sunLight.shadow.camera.left = -d;
  sunLight.shadow.camera.right = d;
  sunLight.shadow.camera.top = d;
  sunLight.shadow.camera.bottom = -d;
  sunLight.shadow.bias = -0.0005;
  scene.add(sunLight);

  // Soft Cyan Rim Fill Light
  const cyanFill = new THREE.DirectionalLight(0x00f0ff, 0.45);
  cyanFill.position.set(-250, 150, -250);
  scene.add(cyanFill);

  // Ground Grid Paddock
  const grid = new THREE.GridHelper(3000, 100, 0x1a263d, 0x0c1322);
  grid.position.y = -0.05;
  scene.add(grid);
}

// ==============================================================================
// 3D Racing Vehicle Mesh Builder
// ==============================================================================

function buildCarModel() {
  carGroup = new THREE.Group();
  scene.add(carGroup);

  // Materials
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0x00d4ff,
    metalness: 0.85,
    roughness: 0.2,
  });

  const carbonMaterial = new THREE.MeshStandardMaterial({
    color: 0x111317,
    roughness: 0.6,
    metalness: 0.4,
  });

  const glassMaterial = new THREE.MeshStandardMaterial({
    color: 0x0a1626,
    metalness: 0.95,
    roughness: 0.08,
    transparent: true,
    opacity: 0.9,
  });

  const wheelRubberMat = new THREE.MeshStandardMaterial({
    color: 0x1a1a1a,
    roughness: 0.8,
    metalness: 0.1,
  });

  const wheelRimMat = new THREE.MeshStandardMaterial({
    color: 0xd0d8e8,
    metalness: 0.9,
    roughness: 0.2,
  });

  // Dimensions matching Python Car (30px length, 15px width in world coords)
  const length = 30.0;
  const width = 15.0;
  const height = 6.0;

  // 1. Lower Chassis / Undertray & Splitter
  const chassisGeo = new THREE.BoxGeometry(length, 1.2, width);
  const chassisMesh = new THREE.Mesh(chassisGeo, carbonMaterial);
  chassisMesh.position.y = 1.0;
  chassisMesh.castShadow = true;
  carGroup.add(chassisMesh);

  // Front Carbon Splitter
  const splitterGeo = new THREE.BoxGeometry(4.0, 0.4, width + 2.0);
  const splitterMesh = new THREE.Mesh(splitterGeo, carbonMaterial);
  splitterMesh.position.set(length / 2 + 1.5, 0.4, 0);
  splitterMesh.castShadow = true;
  carGroup.add(splitterMesh);

  // Rear Diffuser
  const diffuserGeo = new THREE.BoxGeometry(5.0, 1.0, width);
  const diffuserMesh = new THREE.Mesh(diffuserGeo, carbonMaterial);
  diffuserMesh.position.set(-length / 2 + 1.0, 0.8, 0);
  diffuserMesh.rotation.z = 0.15;
  carGroup.add(diffuserMesh);

  // 2. Aerodynamic Body Cocoon
  const bodyGeo = new THREE.BoxGeometry(length * 0.75, height * 0.5, width * 0.85);
  const bodyMesh = new THREE.Mesh(bodyGeo, bodyMaterial);
  bodyMesh.position.set(0, 2.2, 0);
  bodyMesh.castShadow = true;
  bodyMesh.receiveShadow = true;
  carGroup.add(bodyMesh);

  // Nose Cone (Wedge)
  const noseGeo = new THREE.CylinderGeometry(width * 0.35, width * 0.42, length * 0.35, 4);
  const noseMesh = new THREE.Mesh(noseGeo, bodyMaterial);
  noseMesh.rotation.z = Math.PI / 2;
  noseMesh.rotation.y = Math.PI / 4;
  noseMesh.position.set(length * 0.38, 1.8, 0);
  noseMesh.castShadow = true;
  carGroup.add(noseMesh);

  // 3. Cockpit Glass Canopy
  const canopyGeo = new THREE.SphereGeometry(height * 0.65, 16, 12);
  canopyGeo.scale(1.8, 0.75, 1.0);
  const canopyMesh = new THREE.Mesh(canopyGeo, glassMaterial);
  canopyMesh.position.set(1.0, 3.4, 0);
  canopyMesh.castShadow = true;
  carGroup.add(canopyMesh);

  // 4. Rear High-Downforce GT Wing
  const wingGeo = new THREE.BoxGeometry(3.0, 0.4, width * 1.15);
  const wingMesh = new THREE.Mesh(wingGeo, carbonMaterial);
  wingMesh.position.set(-length / 2 + 1.0, 5.2, 0);
  wingMesh.castShadow = true;
  carGroup.add(wingMesh);

  // Wing Struts
  const strutGeo = new THREE.BoxGeometry(0.5, 3.0, 0.4);
  const strutL = new THREE.Mesh(strutGeo, carbonMaterial);
  strutL.position.set(-length / 2 + 1.0, 3.8, width * 0.35);
  carGroup.add(strutL);
  const strutR = new THREE.Mesh(strutGeo, carbonMaterial);
  strutR.position.set(-length / 2 + 1.0, 3.8, -width * 0.35);
  carGroup.add(strutR);

  // 5. Headlights (Twin Projectors)
  const headlightMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
  const hlGeo = new THREE.BoxGeometry(0.8, 0.8, 2.0);
  const hlMeshL = new THREE.Mesh(hlGeo, headlightMat);
  hlMeshL.position.set(length / 2 - 0.2, 2.0, width * 0.32);
  carGroup.add(hlMeshL);

  const hlMeshR = new THREE.Mesh(hlGeo, headlightMat);
  hlMeshR.position.set(length / 2 - 0.2, 2.0, -width * 0.32);
  carGroup.add(hlMeshR);

  // Headlight Projector Light Cones
  const spotL = new THREE.SpotLight(0xd0f4ff, 1.5, 180, Math.PI / 6, 0.4);
  spotL.position.set(length / 2, 2.0, width * 0.3);
  spotL.target.position.set(length / 2 + 80, 0, width * 0.3);
  carGroup.add(spotL);
  carGroup.add(spotL.target);

  const spotR = new THREE.SpotLight(0xd0f4ff, 1.5, 180, Math.PI / 6, 0.4);
  spotR.position.set(length / 2, 2.0, -width * 0.3);
  spotR.target.position.set(length / 2 + 80, 0, -width * 0.3);
  carGroup.add(spotR);
  carGroup.add(spotR.target);

  // 6. Rear Brake Lights (Grounded Action Indicators)
  const brakeMat = new THREE.MeshBasicMaterial({ color: 0x550000 });
  const brakeGeo = new THREE.BoxGeometry(0.6, 0.8, 3.5);
  const brakeL = new THREE.Mesh(brakeGeo, brakeMat);
  brakeL.position.set(-length / 2 - 0.1, 2.4, width * 0.28);
  carGroup.add(brakeL);
  rearBrakeLights.push(brakeL);

  const brakeR = new THREE.Mesh(brakeGeo, brakeMat);
  brakeR.position.set(-length / 2 - 0.1, 2.4, -width * 0.28);
  carGroup.add(brakeR);
  rearBrakeLights.push(brakeR);

  // Dynamic Red Flare Point Light for Active Deceleration
  rearBrakeLightSource = new THREE.PointLight(0xff0022, 0, 40);
  rearBrakeLightSource.position.set(-length / 2 - 2, 2.4, 0);
  carGroup.add(rearBrakeLightSource);

  // 7. Steerable Front Wheels & Fixed Rear Wheels
  const wheelRadius = 3.2;
  const wheelWidth = 2.4;
  const tireGeo = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 16);
  tireGeo.rotateX(Math.PI / 2);

  const rimGeo = new THREE.CylinderGeometry(wheelRadius * 0.65, wheelRadius * 0.65, wheelWidth + 0.1, 12);
  rimGeo.rotateX(Math.PI / 2);

  function createWheelAssembly() {
    const group = new THREE.Group();
    const tire = new THREE.Mesh(tireGeo, wheelRubberMat);
    tire.castShadow = true;
    group.add(tire);
    const rim = new THREE.Mesh(rimGeo, wheelRimMat);
    group.add(rim);
    wheelMeshes.push(group);
    return group;
  }

  const axleOffsetX = length * 0.32;
  const axleOffsetZ = width * 0.52;

  // Front Left Steer Pivot
  frontLeftSteerPivot = new THREE.Group();
  frontLeftSteerPivot.position.set(axleOffsetX, wheelRadius, axleOffsetZ);
  const wheelFL = createWheelAssembly();
  frontLeftSteerPivot.add(wheelFL);
  carGroup.add(frontLeftSteerPivot);

  // Front Right Steer Pivot
  frontRightSteerPivot = new THREE.Group();
  frontRightSteerPivot.position.set(axleOffsetX, wheelRadius, -axleOffsetZ);
  const wheelFR = createWheelAssembly();
  frontRightSteerPivot.add(wheelFR);
  carGroup.add(frontRightSteerPivot);

  // Rear Left Wheel (Fixed)
  const rearL = createWheelAssembly();
  rearL.position.set(-axleOffsetX, wheelRadius, axleOffsetZ);
  carGroup.add(rearL);

  // Rear Right Wheel (Fixed)
  const rearR = createWheelAssembly();
  rearR.position.set(-axleOffsetX, wheelRadius, -axleOffsetZ);
  carGroup.add(rearR);

  // Initial pose
  carGroup.position.set(0, 0, 0);
}

// ==============================================================================
// 5 Lidar Rangefinder Beams Architecture
// ==============================================================================

function buildLidarBeams() {
  rayBeamObjects = [];

  for (let i = 0; i < 5; i++) {
    // 1. Line Beam Geometry
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, 0),
    ]);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      linewidth: 2,
      transparent: true,
      opacity: 0.85,
    });
    const lineMesh = new THREE.Line(lineGeo, lineMat);
    lidarGroup.add(lineMesh);

    // 2. Glowing Volume Cylinder Beam
    const cylGeo = new THREE.CylinderGeometry(0.2, 0.2, 1, 8);
    const cylMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.35,
    });
    const cylMesh = new THREE.Mesh(cylGeo, cylMat);
    lidarGroup.add(cylMesh);

    // 3. Boundary Hit Ring Marker on Track Curbs
    const ringGeo = new THREE.RingGeometry(1.2, 2.2, 16);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.position.y = 0.2;
    lidarGroup.add(ringMesh);

    // 4. Hit Spark Marker
    const sparkGeo = new THREE.SphereGeometry(0.6, 8, 8);
    const sparkMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const sparkMesh = new THREE.Mesh(sparkGeo, sparkMat);
    lidarGroup.add(sparkMesh);

    rayBeamObjects.push({
      line: lineMesh,
      cylinder: cylMesh,
      ring: ringMesh,
      spark: sparkMesh,
    });
  }
}

function updateLidarVisuals(rays, carPos) {
  if (!rays || rays.length < 5 || !carPos) return;

  const rawZ = carPos.z !== undefined ? carPos.z : carPos.y;
  const origin = new THREE.Vector3(
    carPos.x - STATE.trackCenter.x,
    1.6,
    rawZ - STATE.trackCenter.z
  );

  for (let i = 0; i < 5; i++) {
    const ray = rays[i];
    const beamObj = rayBeamObjects[i];
    if (!beamObj || !ray) continue;

    const hit = new THREE.Vector3(
      ray.endpoint_x - STATE.trackCenter.x,
      0.3,
      ray.endpoint_y - STATE.trackCenter.z
    );

    // Distance-dependent coloring: Cyan -> Amber -> Red
    let beamColor = 0x00f0ff;
    if (ray.distance < 50.0) {
      beamColor = 0xff2a5f;
    } else if (ray.distance < 90.0) {
      beamColor = 0xffb700;
    }

    // 1. Update Core Line
    const positions = beamObj.line.geometry.attributes.position.array;
    positions[0] = origin.x;
    positions[1] = origin.y;
    positions[2] = origin.z;
    positions[3] = hit.x;
    positions[4] = hit.y;
    positions[5] = hit.z;
    beamObj.line.geometry.attributes.position.needsUpdate = true;
    beamObj.line.material.color.setHex(beamColor);

    // 2. Update Volume Cylinder
    const dist = origin.distanceTo(hit);
    beamObj.cylinder.scale.set(1, dist, 1);
    beamObj.cylinder.position.copy(origin).lerp(hit, 0.5);
    beamObj.cylinder.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      hit.clone().sub(origin).normalize()
    );
    beamObj.cylinder.material.color.setHex(beamColor);

    // 3. Update Hit Ring Marker
    beamObj.ring.position.set(hit.x, 0.22, hit.z);
    beamObj.ring.material.color.setHex(beamColor);
    // Subtle animated pulse
    const pulseScale = 1.0 + Math.sin(Date.now() * 0.01 + i) * 0.2;
    beamObj.ring.scale.set(pulseScale, pulseScale, pulseScale);

    // 4. Update Hit Spark
    beamObj.spark.position.set(hit.x, 0.35, hit.z);
  }
}

// ==============================================================================
// 3D Track Environment Builder
// ==============================================================================

function build3DTrack(trackData) {
  // Clear previous track meshes
  while (trackGroup.children.length > 0) {
    const obj = trackGroup.children[0];
    trackGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }

  STATE.trackGeometry = trackData;
  const bounds = trackData.bounds;
  STATE.trackCenter = {
    x: (bounds.min_x + bounds.max_x) / 2,
    z: (bounds.min_y + bounds.max_y) / 2,
  };

  const inner = trackData.inner_boundary;
  const outer = trackData.outer_boundary;
  const center = trackData.centerline;
  const n = inner.length;

  // 1. Asphalt Road Ribbon
  const roadGeo = new THREE.BufferGeometry();
  const roadVertices = [];
  const roadIndices = [];

  for (let i = 0; i < n; i++) {
    const inPt = inner[i];
    const outPt = outer[i];
    roadVertices.push(
      inPt[0] - STATE.trackCenter.x, 0.06, inPt[1] - STATE.trackCenter.z,
      outPt[0] - STATE.trackCenter.x, 0.06, outPt[1] - STATE.trackCenter.z
    );

    const next = (i + 1) % n;
    const i0 = i * 2;
    const i1 = i * 2 + 1;
    const i2 = next * 2;
    const i3 = next * 2 + 1;

    roadIndices.push(i0, i1, i2);
    roadIndices.push(i1, i3, i2);
  }

  roadGeo.setAttribute("position", new THREE.Float32BufferAttribute(roadVertices, 3));
  roadGeo.setIndex(roadIndices);
  roadGeo.computeVertexNormals();

  const roadMat = new THREE.MeshStandardMaterial({
    color: 0x161922,
    roughness: 0.88,
    metalness: 0.12,
  });
  const roadMesh = new THREE.Mesh(roadGeo, roadMat);
  roadMesh.receiveShadow = true;
  trackGroup.add(roadMesh);

  // 2. Racing Curbs (Red and White Alternating Teeth)
  buildCurbStrip(inner, -1.8, 0.25);
  buildCurbStrip(outer, 1.8, 0.25);

  // 3. Glowing Centerline
  const centerPoints = center.map(
    (pt) => new THREE.Vector3(pt[0] - STATE.trackCenter.x, 0.12, pt[1] - STATE.trackCenter.z)
  );
  centerPoints.push(centerPoints[0]); // Close loop
  const centerGeo = new THREE.BufferGeometry().setFromPoints(centerPoints);
  const centerMat = new THREE.LineDashedMaterial({
    color: 0xffb700,
    dashSize: 6,
    gapSize: 4,
    linewidth: 2,
  });
  const centerLine = new THREE.Line(centerGeo, centerMat);
  centerLine.computeLineDistances();
  trackGroup.add(centerLine);

  // 4. Start / Finish Line
  buildStartFinishBanner(center[0], inner[0], outer[0]);

  // 5. Perimeter Floodlight Stadium Towers
  buildStadiumTowers(bounds);
}

function buildCurbStrip(boundaryPoints, offsetWidth, height) {
  const n = boundaryPoints.length;
  const curbGeo = new THREE.BufferGeometry();
  const curbVertices = [];
  const curbColors = [];
  const curbIndices = [];

  for (let i = 0; i < n; i++) {
    const pt = boundaryPoints[i];
    const nextPt = boundaryPoints[(i + 1) % n];
    const dx = nextPt[0] - pt[0];
    const dz = nextPt[1] - pt[1];
    const len = Math.hypot(dx, dz) || 1;
    const nx = -dz / len * offsetWidth;
    const nz = dx / len * offsetWidth;

    curbVertices.push(
      pt[0] - STATE.trackCenter.x, 0.08, pt[1] - STATE.trackCenter.z,
      pt[0] - STATE.trackCenter.x + nx, height, pt[1] - STATE.trackCenter.z + nz
    );

    // Alternating Red & White segments
    const isRed = Math.floor(i / 3) % 2 === 0;
    const col = isRed ? [0.9, 0.1, 0.2] : [0.95, 0.95, 0.95];
    curbColors.push(...col, ...col);

    const next = (i + 1) % n;
    const i0 = i * 2;
    const i1 = i * 2 + 1;
    const i2 = next * 2;
    const i3 = next * 2 + 1;

    curbIndices.push(i0, i1, i2);
    curbIndices.push(i1, i3, i2);
  }

  curbGeo.setAttribute("position", new THREE.Float32BufferAttribute(curbVertices, 3));
  curbGeo.setAttribute("color", new THREE.Float32BufferAttribute(curbColors, 3));
  curbGeo.setIndex(curbIndices);
  curbGeo.computeVertexNormals();

  const curbMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.6,
    metalness: 0.2,
  });
  const curbMesh = new THREE.Mesh(curbGeo, curbMat);
  curbMesh.castShadow = true;
  curbMesh.receiveShadow = true;
  trackGroup.add(curbMesh);
}

function buildStartFinishBanner(centerPt, inPt, outPt) {
  const sfGeo = new THREE.BufferGeometry();
  const sfVertices = [
    inPt[0] - STATE.trackCenter.x, 0.14, inPt[1] - STATE.trackCenter.z,
    outPt[0] - STATE.trackCenter.x, 0.14, outPt[1] - STATE.trackCenter.z,
    inPt[0] - STATE.trackCenter.x + 3, 0.14, inPt[1] - STATE.trackCenter.z,
    outPt[0] - STATE.trackCenter.x + 3, 0.14, outPt[1] - STATE.trackCenter.z,
  ];
  sfGeo.setAttribute("position", new THREE.Float32BufferAttribute(sfVertices, 3));
  sfGeo.setIndex([0, 1, 2, 1, 3, 2]);

  const sfMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const sfMesh = new THREE.Mesh(sfGeo, sfMat);
  trackGroup.add(sfMesh);
}

function buildStadiumTowers(bounds) {
  const minX = bounds.min_x - STATE.trackCenter.x - 60;
  const maxX = bounds.max_x - STATE.trackCenter.x + 60;
  const minZ = bounds.min_y - STATE.trackCenter.z - 60;
  const maxZ = bounds.max_y - STATE.trackCenter.z + 60;

  const towerPositions = [
    [minX, minZ],
    [maxX, minZ],
    [maxX, maxZ],
    [minX, maxZ],
  ];

  const poleGeo = new THREE.CylinderGeometry(1.2, 1.8, 90, 8);
  const poleMat = new THREE.MeshStandardMaterial({ color: 0x222a38, metalness: 0.8 });
  const lampGeo = new THREE.BoxGeometry(10, 4, 3);
  const lampMat = new THREE.MeshBasicMaterial({ color: 0xe6f4ff });

  towerPositions.forEach(([tx, tz]) => {
    const pole = new THREE.Mesh(poleGeo, poleMat);
    pole.position.set(tx, 45, tz);
    trackGroup.add(pole);

    const lamp = new THREE.Mesh(lampGeo, lampMat);
    lamp.position.set(tx, 90, tz);
    lamp.lookAt(0, 10, 0);
    trackGroup.add(lamp);
  });
}

// ==============================================================================
// 3D Trajectory Ribbon Builder
// ==============================================================================

function updateTrajectoryRibbon(trajectoryPoints) {
  if (trajectoryMesh) {
    scene.remove(trajectoryMesh);
    if (trajectoryMesh.geometry) trajectoryMesh.geometry.dispose();
  }

  if (!trajectoryPoints || trajectoryPoints.length < 2) return;

  const ribbonGeo = new THREE.BufferGeometry();
  const vertices = [];
  const ribbonWidth = 1.4;

  for (let i = 0; i < trajectoryPoints.length; i++) {
    const pt = trajectoryPoints[i];
    const x = pt[0] - STATE.trackCenter.x;
    const z = pt[1] - STATE.trackCenter.z;

    // Normal calculation along trajectory
    let nx = 0, nz = 1;
    if (i < trajectoryPoints.length - 1) {
      const next = trajectoryPoints[i + 1];
      const dx = next[0] - pt[0];
      const dz = next[1] - pt[1];
      const len = Math.hypot(dx, dz) || 1;
      nx = -dz / len;
      nz = dx / len;
    }

    vertices.push(
      x - nx * ribbonWidth, 0.16, z - nz * ribbonWidth,
      x + nx * ribbonWidth, 0.16, z + nz * ribbonWidth
    );
  }

  const indices = [];
  for (let i = 0; i < trajectoryPoints.length - 1; i++) {
    const i0 = i * 2;
    const i1 = i * 2 + 1;
    const i2 = (i + 1) * 2;
    const i3 = (i + 1) * 2 + 1;
    indices.push(i0, i1, i2);
    indices.push(i1, i3, i2);
  }

  ribbonGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  ribbonGeo.setIndex(indices);

  const ribbonMat = new THREE.MeshBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.65,
    side: THREE.DoubleSide,
  });

  trajectoryMesh = new THREE.Mesh(ribbonGeo, ribbonMat);
  scene.add(trajectoryMesh);
}

// ==============================================================================
// Cinematic Camera Director Loop
// ==============================================================================

function updateCameraDirector(carPos, carAngle, carVelocity) {
  if (!carGroup || !carPos) return;

  let carX = carPos.x;
  let carZ = carPos.z !== undefined ? carPos.z : carPos.y;
  if (carPos !== carGroup.position) {
    carX = carPos.x - STATE.trackCenter.x;
    carZ = (carPos.z !== undefined ? carPos.z : carPos.y) - STATE.trackCenter.z;
  }

  const heading = carGroup.rotation.y;
  const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
  const backward = forward.clone().negate();

  if (STATE.activeCameraMode === "chase") {
    if (controls) controls.enabled = false;
    // Damped follow camera trailing behind vehicle
    const followDist = 42.0;
    const followHeight = 16.0;
    const targetCamPos = new THREE.Vector3(
      carX + backward.x * followDist,
      followHeight + Math.abs(carVelocity) * 0.8,
      carZ + backward.z * followDist
    );
    const lookTarget = new THREE.Vector3(carX + forward.x * 15.0, 3.5, carZ + forward.z * 15.0);

    STATE.cameraPos.lerp(targetCamPos, 0.08);
    STATE.cameraTarget.lerp(lookTarget, 0.12);

    camera.position.copy(STATE.cameraPos);
    camera.lookAt(STATE.cameraTarget);

  } else if (STATE.activeCameraMode === "cockpit") {
    if (controls) controls.enabled = false;
    // Driver / Hood perspective mounted directly on vehicle
    const hoodCamPos = new THREE.Vector3(carX + forward.x * 4.0, 3.2, carZ + forward.z * 4.0);
    const lookTarget = new THREE.Vector3(carX + forward.x * 60.0, 1.5, carZ + forward.z * 60.0);

    camera.position.copy(hoodCamPos);
    camera.lookAt(lookTarget);

  } else if (STATE.activeCameraMode === "orbit") {
    if (controls) {
      controls.enabled = true;
      controls.target.lerp(new THREE.Vector3(carX, 2.0, carZ), 0.1);
      controls.update();
    }

  } else if (STATE.activeCameraMode === "overhead") {
    if (controls) controls.enabled = false;
    // Strategic satellite top-down view
    const targetCamPos = new THREE.Vector3(0, 520, 0);
    const lookTarget = new THREE.Vector3(0, 0, 0);

    STATE.cameraPos.lerp(targetCamPos, 0.06);
    camera.position.copy(STATE.cameraPos);
    camera.lookAt(lookTarget);
  }
}

function onWindowResize() {
  const container = document.getElementById("canvasContainer");
  if (!container || !renderer || !camera) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

// ==============================================================================
// 2D Circuit Minimap / Radar Renderer (#raceCanvas)
// ==============================================================================

function renderMinimapRadar(carData, trackData) {
  if (!minimapCtx || !trackData || !trackData.bounds) return;

  const w = minimapCanvas.width;
  const h = minimapCanvas.height;
  minimapCtx.clearRect(0, 0, w, h);

  const bounds = trackData.bounds;
  const pad = 12;
  const worldW = bounds.max_x - bounds.min_x || 1;
  const worldH = bounds.max_y - bounds.min_y || 1;
  const scale = Math.min((w - 2 * pad) / worldW, (h - 2 * pad) / worldH);
  const offX = pad + (w - 2 * pad - worldW * scale) / 2;
  const offY = pad + (h - 2 * pad - worldH * scale) / 2;

  function toMinimap(x, y) {
    return {
      x: offX + (x - bounds.min_x) * scale,
      y: offY + (y - bounds.min_y) * scale,
    };
  }

  // Draw Track Asphalt
  minimapCtx.beginPath();
  const inner = trackData.inner_boundary;
  const outer = trackData.outer_boundary;

  const p0 = toMinimap(outer[0][0], outer[0][1]);
  minimapCtx.moveTo(p0.x, p0.y);
  for (let i = 1; i < outer.length; i++) {
    const pt = toMinimap(outer[i][0], outer[i][1]);
    minimapCtx.lineTo(pt.x, pt.y);
  }
  minimapCtx.closePath();
  minimapCtx.fillStyle = "#161c28";
  minimapCtx.fill();
  minimapCtx.strokeStyle = "#00f0ff";
  minimapCtx.lineWidth = 1.2;
  minimapCtx.stroke();

  // Draw Centerline
  minimapCtx.beginPath();
  const c0 = toMinimap(trackData.centerline[0][0], trackData.centerline[0][1]);
  minimapCtx.moveTo(c0.x, c0.y);
  for (let i = 1; i < trackData.centerline.length; i++) {
    const pt = toMinimap(trackData.centerline[i][0], trackData.centerline[i][1]);
    minimapCtx.lineTo(pt.x, pt.y);
  }
  minimapCtx.closePath();
  minimapCtx.strokeStyle = "rgba(255, 183, 0, 0.4)";
  minimapCtx.lineWidth = 1.0;
  minimapCtx.stroke();

  // Draw Car Blip
  if (carData) {
    const carM = toMinimap(carData.x, carData.y);
    minimapCtx.beginPath();
    minimapCtx.arc(carM.x, carM.y, 4, 0, Math.PI * 2);
    minimapCtx.fillStyle = "#00f0ff";
    minimapCtx.shadowColor = "#00f0ff";
    minimapCtx.shadowBlur = 8;
    minimapCtx.fill();
    minimapCtx.shadowBlur = 0;
  }
}

// ==============================================================================
// Simulation API & Stepping Client Loop
// ==============================================================================

async function startNewSimulation(presetId, trackSeed) {
  try {
    updateStatusPill("STARTING", "amber");
    const res = await fetch("/simulation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset_id: presetId,
        track_seed: parseInt(trackSeed, 10),
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    STATE.presetId = presetId;
    STATE.trackSeed = trackSeed;

    build3DTrack(data.track);
    syncUIWithState(data);
    updateStatusPill("READY", "cyan");
  } catch (err) {
    console.error("Failed to start simulation:", err);
    updateStatusPill("ERROR", "amber");
  }
}

async function stepSimulation(stepCount = 1) {
  try {
    const res = await fetch("/simulation/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        steps: stepCount,
        calculate_saliency: STATE.calculateSaliency,
        commentary_mode: STATE.commentaryMode,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    syncUIWithState(data);
    return data;
  } catch (err) {
    console.error("Step execution failed:", err);
    return null;
  }
}

async function resetSimulation() {
  try {
    const res = await fetch("/simulation/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    syncUIWithState(data);
    updateStatusPill("PAUSED", "cyan");
    hideOverlayModal();
  } catch (err) {
    console.error("Reset failed:", err);
  }
}

// ==============================================================================
// State Synchronization & Rendering Loop
// ==============================================================================

function syncUIWithState(data) {
  if (!data) return;

  const car = data.vehicle;
  const telemetry = data.telemetry;
  const session = data.session;
  const rays = data.rays;

  // 1. Update 3D Vehicle Transform
  if (carGroup && car) {
    carGroup.position.set(
      car.x - STATE.trackCenter.x,
      1.0,
      car.y - STATE.trackCenter.z
    );
    // Heading rotation (clockwise 2D -> -angle in 3D Y-axis)
    carGroup.rotation.y = -car.angle;

    // Independent Front Wheel Steering Animation
    const steerRad = (car.steering_angle_deg * Math.PI) / 180.0;
    if (frontLeftSteerPivot) frontLeftSteerPivot.rotation.y = -steerRad;
    if (frontRightSteerPivot) frontRightSteerPivot.rotation.y = -steerRad;

    // Dynamic Wheel Spin
    wheelMeshes.forEach((w) => {
      w.rotation.x += car.velocity * 0.08;
    });

    // Grounded Action Indicator: Active Brake Light Flare
    const isBraking = car.throttle < -0.05;
    const brakeColor = isBraking ? 0xff0022 : 0x550000;
    rearBrakeLights.forEach((bl) => bl.material.color.setHex(brakeColor));
    if (rearBrakeLightSource) {
      rearBrakeLightSource.intensity = isBraking ? 2.5 : 0;
    }
  }

  // 2. Update 5 Lidar Rangefinder Beams
  if (rays) {
    updateLidarVisuals(rays, car);
  }

  // 3. Update 3D Trajectory Ribbon
  if (data.trajectory) {
    updateTrajectoryRibbon(data.trajectory);
  }

  // 4. Update 2D Circuit Minimap Radar
  if (data.track) {
    renderMinimapRadar(car, data.track);
  }

  // 5. Update Digital Telemetry Cards
  updateTelemetryGauges(car, telemetry, session);

  // 6. Update 5 Lidar Rangefinder Gauges in Sidebar & In-World HUD
  updateLidarGauges(rays);

  // 7. Update Policy Saliency Bars
  if (data.saliency) {
    STATE.currentSaliency = data.saliency;
    renderSaliencyBars();
  }

  // 8. Update AI Race Engineer Commentary Feed
  if (data.commentary) {
    updateCommentaryFeed(data.commentary);
  }

  // 9. Check Lap Completion / Review
  if (data.lap_review) {
    displayLapReviewCard(data.lap_review);
  }

  if (session.is_completed) {
    showOverlayModal("LAP COMPLETED!", `Clean lap finished in ${session.step_count} steps!`);
    STATE.simRunning = false;
    updateStartButtonState();
    updateStatusPill("LAP COMPLETED", "green");
  } else if (session.is_crashed) {
    showOverlayModal("VEHICLE CRASHED", `Exceeded track boundary at step ${session.step_count}.`);
    STATE.simRunning = false;
    updateStartButtonState();
    updateStatusPill("CRASHED", "amber");
  }
}

function updateTelemetryGauges(car, telemetry, session) {
  // Speed
  const velEl = document.getElementById("valVelocity");
  const barVel = document.getElementById("barVelocity");
  const cockpitSpeed = document.getElementById("cockpitSpeed");
  if (velEl) velEl.innerHTML = `${car.velocity.toFixed(2)} <span class="unit">px/step</span>`;
  if (barVel) barVel.style.width = `${Math.min(100, Math.max(0, (car.velocity / 4.0) * 100))}%`;
  if (cockpitSpeed) cockpitSpeed.innerText = car.velocity.toFixed(2);

  // Steering
  const steerEl = document.getElementById("valSteering");
  const barSteer = document.getElementById("barSteering");
  const cockpitSteerBar = document.getElementById("cockpitSteerBar");
  const cockpitSteerDeg = document.getElementById("cockpitSteerDeg");
  const steerDeg = car.steering_angle_deg || 0;
  if (steerEl) {
    steerEl.innerHTML = `${steerDeg >= 0 ? "+" : ""}${steerDeg.toFixed(1)}° <span class="unit">(act: ${car.steering.toFixed(2)})</span>`;
  }
  if (barSteer) {
    const steerPct = (steerDeg / 30.0) * 50; // [-50%, +50%]
    if (steerPct >= 0) {
      barSteer.style.left = "50%";
      barSteer.style.width = `${Math.min(50, steerPct)}%`;
    } else {
      barSteer.style.left = `${50 + steerPct}%`;
      barSteer.style.width = `${Math.abs(steerPct)}%`;
    }
  }
  if (cockpitSteerDeg) cockpitSteerDeg.innerText = `${steerDeg >= 0 ? "+" : ""}${steerDeg.toFixed(1)}°`;
  if (cockpitSteerBar) {
    const steerPct = (steerDeg / 30.0) * 50;
    if (steerPct >= 0) {
      cockpitSteerBar.style.left = "50%";
      cockpitSteerBar.style.width = `${Math.min(50, steerPct)}%`;
    } else {
      cockpitSteerBar.style.left = `${50 + steerPct}%`;
      cockpitSteerBar.style.width = `${Math.abs(steerPct)}%`;
    }
  }

  // Throttle / Longitudinal Action
  const throtEl = document.getElementById("valThrottle");
  const barThrot = document.getElementById("barThrottle");
  const cockpitThrotBar = document.getElementById("cockpitThrottleBar");
  const cockpitMode = document.getElementById("cockpitActionMode");
  if (throtEl) {
    throtEl.innerHTML = `${car.throttle >= 0 ? "+" : ""}${car.throttle.toFixed(2)} <span class="unit">${car.action_mode.toUpperCase()}</span>`;
  }
  if (barThrot) {
    const throtPct = car.throttle * 50;
    if (throtPct >= 0) {
      barThrot.style.left = "50%";
      barThrot.style.width = `${Math.min(50, throtPct)}%`;
      barThrot.style.background = "var(--accent-green)";
    } else {
      barThrot.style.left = `${50 + throtPct}%`;
      barThrot.style.width = `${Math.abs(throtPct)}%`;
      barThrot.style.background = "var(--accent-red)";
    }
  }
  if (cockpitMode) cockpitMode.innerText = car.action_mode.toUpperCase();
  if (cockpitThrotBar) {
    const throtPct = car.throttle * 50;
    if (throtPct >= 0) {
      cockpitThrotBar.style.left = "50%";
      cockpitThrotBar.style.width = `${Math.min(50, throtPct)}%`;
      cockpitThrotBar.style.background = "var(--accent-green)";
    } else {
      cockpitThrotBar.style.left = `${50 + throtPct}%`;
      cockpitThrotBar.style.width = `${Math.abs(throtPct)}%`;
      cockpitThrotBar.style.background = "var(--accent-red)";
    }
  }

  // Lap Progress
  const progEl = document.getElementById("valProgress");
  const barProg = document.getElementById("barProgress");
  const progPct = (car.lap_progress * 100).toFixed(1);
  if (progEl) progEl.innerText = `${progPct}%`;
  if (barProg) barProg.style.width = `${Math.min(100, Math.max(0, progPct))}%`;

  // Secondary Telemetry
  const headErrEl = document.getElementById("valHeadingErr");
  const distEl = document.getElementById("valSignedDist");
  const slipEl = document.getElementById("valSlip");
  const posEl = document.getElementById("valPos");
  const stepClock = document.getElementById("stepClock");

  if (headErrEl) headErrEl.innerText = `${telemetry.heading_error >= 0 ? "+" : ""}${telemetry.heading_error.toFixed(1)}°`;
  if (distEl) distEl.innerText = `${telemetry.signed_distance.toFixed(1)} px`;
  if (slipEl) slipEl.innerText = `${telemetry.slip_angle >= 0 ? "+" : ""}${telemetry.slip_angle.toFixed(1)}°`;
  if (posEl) posEl.innerText = `${car.x.toFixed(1)}, ${car.y.toFixed(1)}`;
  if (stepClock) stepClock.innerText = `STEP: ${session.step_count} / ${session.max_steps}`;
}

function updateLidarGauges(rays) {
  if (!rays || rays.length < 5) return;

  const listContainer = document.getElementById("sensorsList");
  if (!listContainer) return;

  let html = "";
  rays.forEach((ray, i) => {
    const dist = ray.distance;
    const pct = Math.min(100, Math.max(0, (dist / 200.0) * 100));
    let colorClass = "cyan";
    if (dist < 50.0) colorClass = "red";
    else if (dist < 90.0) colorClass = "amber";

    html += `
      <div class="sensor-row">
        <span class="sensor-name">${RAY_LABELS[i]}</span>
        <div class="sensor-bar-wrap">
          <div class="sensor-bar" style="width: ${pct}%; background: var(--accent-${colorClass});"></div>
        </div>
        <span class="sensor-dist" style="color: var(--accent-${colorClass});">${dist.toFixed(0)} px</span>
      </div>
    `;

    // Also update In-World Top HUD
    const hudTag = document.getElementById(`hudRay${i}`);
    if (hudTag) {
      hudTag.className = `lidar-hud-tag ${colorClass}`;
      const valEl = hudTag.querySelector(".beam-val");
      if (valEl) valEl.innerText = `${dist.toFixed(0)}px`;
    }
  });

  listContainer.innerHTML = html;
}

// ==============================================================================
// Policy Saliency Rendering (Live & Historical)
// ==============================================================================

function renderSaliencyBars() {
  const container = document.getElementById("saliencyBarsContainer");
  if (!container) return;

  let featureScores = {};
  if (STATE.saliencyTab === "steer" && STATE.currentSaliency) {
    featureScores = STATE.currentSaliency.steering_saliency || {};
  } else if (STATE.saliencyTab === "long" && STATE.currentSaliency) {
    featureScores = STATE.currentSaliency.throttle_saliency || {};
  } else if (STATE.saliencyTab === "hist" && STATE.historicalSaliency) {
    const raw = STATE.historicalSaliency.analysis_results?.overall_steering?.raw_mean || {};
    featureScores = raw;
  }

  const entries = Object.entries(featureScores);
  if (entries.length === 0) {
    container.innerHTML = '<div style="color: var(--text-dim); font-size: 11px; padding: 8px;">Awaiting live saliency compute...</div>';
    return;
  }

  const maxVal = Math.max(...entries.map(([, v]) => v), 0.001);
  let html = "";
  entries.forEach(([key, val]) => {
    const pct = Math.min(100, Math.max(0, (val / maxVal) * 100));
    html += `
      <div class="sal-row">
        <span class="sal-label" title="${key}">${key}</span>
        <div class="sal-track">
          <div class="sal-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="sal-val">${val.toFixed(4)}</span>
      </div>
    `;
  });
  container.innerHTML = html;

  // Update top ray highlights
  const topSteer = document.getElementById("topSteerRay");
  const topThrot = document.getElementById("topThrottleRay");
  if (topSteer && STATE.currentSaliency?.top_rays) {
    topSteer.innerText = STATE.currentSaliency.top_rays.steering || "None";
  }
  if (topThrot && STATE.currentSaliency?.top_rays) {
    topThrot.innerText = STATE.currentSaliency.top_rays.throttle || "None";
  }
}

async function loadHistoricalSaliency() {
  try {
    const res = await fetch("/simulation/historical-saliency");
    if (res.ok) {
      STATE.historicalSaliency = await res.json();
    }
  } catch (err) {
    console.warn("Historical saliency load notice:", err);
  }
}

// ==============================================================================
// AI Race Engineer Commentary & Grounded Review
// ==============================================================================

function updateCommentaryFeed(commentaryList) {
  const feed = document.getElementById("commentaryFeed");
  if (!feed || !commentaryList) return;

  feed.innerHTML = "";
  commentaryList.slice(-6).forEach((item) => {
    const eventClass = item.event_type || "default";
    const div = document.createElement("div");
    div.className = `commentary-item ${eventClass}`;
    div.innerHTML = `
      <span class="c-step">[STEP ${item.step}]</span>
      <span class="c-source">[${item.source.toUpperCase().replace("_", " ")}]</span>
      <span class="c-text">${item.commentary}</span>
    `;
    feed.appendChild(div);
  });
  feed.scrollTop = feed.scrollHeight;
}

function displayLapReviewCard(review) {
  const card = document.getElementById("lapReviewCard");
  if (!card || !review) return;

  card.style.display = "block";
  document.getElementById("reviewVerdict").innerText = review.verdict || "CLEAN LAP";
  document.getElementById("revTime").innerText = `${review.total_steps} steps`;
  document.getElementById("revAvgSpeed").innerText = `${review.average_speed} px/step`;
  document.getElementById("revMaxSpeed").innerText = `${review.maximum_speed} px/step`;
  document.getElementById("revSmoothness").innerText = `${review.steering_smoothness_score}%`;
  document.getElementById("revMinMargin").innerText = `${review.min_boundary_proximity_px} px`;
  document.getElementById("revOffTrack").innerText = review.off_track_events;

  const obsList = document.getElementById("reviewObsList");
  if (obsList) {
    obsList.innerHTML = "";
    (review.strengths || []).forEach((str) => {
      const li = document.createElement("li");
      li.style.color = "var(--accent-green)";
      li.innerText = `✓ ${str}`;
      obsList.appendChild(li);
    });
    (review.weaknesses || []).forEach((wk) => {
      const li = document.createElement("li");
      li.style.color = "var(--accent-amber)";
      li.innerText = `! ${wk}`;
      obsList.appendChild(li);
    });
  }
}

// ==============================================================================
// Animation Frame & Stepping Loop
// ==============================================================================

let lastFrameTime = performance.now();
let frameCount = 0;
let lastFpsUpdateTime = performance.now();

function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  frameCount++;
  if (now - lastFpsUpdateTime >= 1000) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdateTime));
    const fpsEl = document.getElementById("fpsCounter");
    if (fpsEl) fpsEl.innerText = `${fps} FPS`;
    frameCount = 0;
    lastFpsUpdateTime = now;
  }

  // Update Camera
  if (carGroup) {
    updateCameraDirector(carGroup.position, carGroup.rotation.y, 4.0);
  }

  // Render 3D Scene
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

// Simulation background polling loop
async function runSimulationLoop() {
  while (true) {
    if (STATE.simRunning) {
      await stepSimulation(STATE.simSpeed);
    }
    await new Promise((r) => setTimeout(r, 40));
  }
}

// ==============================================================================
// UI Control Event Listeners
// ==============================================================================

function setupEventListeners() {
  // 1. Camera Mode Buttons
  const camModes = ["Chase", "Cockpit", "Orbit", "Overhead"];
  camModes.forEach((mode) => {
    const btn = document.getElementById(`btnCam${mode}`);
    if (btn) {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".cam-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        STATE.activeCameraMode = mode.toLowerCase();

        const hintEl = document.getElementById("camHintText");
        if (hintEl) {
          if (mode === "Chase") hintEl.innerText = "CHASE CAMERA // DAMPED TRAJECTORY TRACKING";
          else if (mode === "Cockpit") hintEl.innerText = "COCKPIT CAMERA // HOOD SENSOR VIEW";
          else if (mode === "Orbit") hintEl.innerText = "3D ORBIT // DRAG TO ROTATE, SCROLL TO ZOOM";
          else if (mode === "Overhead") hintEl.innerText = "OVERHEAD // BROADCAST SATELLITE CAMERA";
        }
      });
    }
  });

  // 2. Play / Pause Button
  const btnStartPause = document.getElementById("btnStartPause");
  if (btnStartPause) {
    btnStartPause.addEventListener("click", () => {
      STATE.simRunning = !STATE.simRunning;
      updateStartButtonState();
      updateStatusPill(STATE.simRunning ? "RUNNING" : "PAUSED", STATE.simRunning ? "green" : "cyan");
    });
  }

  // 3. Step Button
  const btnStep = document.getElementById("btnStep");
  if (btnStep) {
    btnStep.addEventListener("click", async () => {
      STATE.simRunning = false;
      updateStartButtonState();
      await stepSimulation(1);
      updateStatusPill("STEPPED", "cyan");
    });
  }

  // 4. Reset Button
  const btnReset = document.getElementById("btnReset");
  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      STATE.simRunning = false;
      updateStartButtonState();
      await resetSimulation();
    });
  }

  // 5. Track Preset Selector & Load Button
  const btnLoadTrack = document.getElementById("btnLoadTrack");
  if (btnLoadTrack) {
    btnLoadTrack.addEventListener("click", async () => {
      STATE.simRunning = false;
      updateStartButtonState();
      const preset = document.getElementById("trackPreset").value;
      const seed = document.getElementById("trackSeed").value;
      await startNewSimulation(preset, seed);
    });
  }

  // 6. Sim Speed Slider
  const speedRange = document.getElementById("speedRange");
  const speedVal = document.getElementById("speedVal");
  if (speedRange && speedVal) {
    speedRange.addEventListener("input", (e) => {
      STATE.simSpeed = parseInt(e.target.value, 10);
      speedVal.innerText = `${STATE.simSpeed}x`;
    });
  }

  // 7. Saliency & Gemini Toggles
  const checkSaliency = document.getElementById("checkSaliency");
  if (checkSaliency) {
    checkSaliency.addEventListener("change", (e) => {
      STATE.calculateSaliency = e.target.checked;
    });
  }

  const checkGemini = document.getElementById("checkGemini");
  if (checkGemini) {
    checkGemini.addEventListener("change", (e) => {
      STATE.commentaryMode = e.target.checked ? "gemini" : "offline";
      const badge = document.getElementById("commentaryModeBadge");
      if (badge) {
        badge.innerText = e.target.checked ? "GEMINI 3.8 FLASH" : "OFFLINE DETERMINISTIC";
      }
    });
  }

  // 8. Saliency Tabs
  const tabSteer = document.getElementById("tabSteer");
  const tabLong = document.getElementById("tabLong");
  const tabHist = document.getElementById("tabHist");

  [tabSteer, tabLong, tabHist].forEach((tab) => {
    if (tab) {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        if (tab === tabSteer) STATE.saliencyTab = "steer";
        else if (tab === tabLong) STATE.saliencyTab = "long";
        else if (tab === tabHist) STATE.saliencyTab = "hist";
        renderSaliencyBars();
      });
    }
  });

  // 9. Overlay Restart Button
  const btnOverlayRestart = document.getElementById("btnOverlayRestart");
  if (btnOverlayRestart) {
    btnOverlayRestart.addEventListener("click", async () => {
      hideOverlayModal();
      await resetSimulation();
      STATE.simRunning = true;
      updateStartButtonState();
      updateStatusPill("RUNNING", "green");
    });
  }
}

function updateStartButtonState() {
  const icon = document.getElementById("startIcon");
  const text = document.getElementById("startText");
  const btn = document.getElementById("btnStartPause");
  if (!icon || !text || !btn) return;

  if (STATE.simRunning) {
    icon.innerText = "⏸";
    text.innerText = "Pause Simulation";
    btn.classList.add("active");
  } else {
    icon.innerText = "▶";
    text.innerText = "Start Simulation";
    btn.classList.remove("active");
  }
}

function updateStatusPill(statusText, dotColorClass) {
  const sText = document.getElementById("statusText");
  const sDot = document.getElementById("statusDot");
  if (sText) sText.innerText = statusText;
  if (sDot) sDot.className = `pill-dot ${dotColorClass}`;
}

function showOverlayModal(title, message) {
  const overlay = document.getElementById("canvasOverlay");
  const titleEl = document.getElementById("overlayTitle");
  const msgEl = document.getElementById("overlayMessage");
  if (overlay) overlay.style.display = "flex";
  if (titleEl) titleEl.innerText = title;
  if (msgEl) msgEl.innerText = message;
}

function hideOverlayModal() {
  const overlay = document.getElementById("canvasOverlay");
  if (overlay) overlay.style.display = "none";
}

// ==============================================================================
// Initialization
// ==============================================================================

window.addEventListener("DOMContentLoaded", async () => {
  initThreeJS();
  setupEventListeners();
  await loadHistoricalSaliency();

  // Load default track & initial simulation state
  await startNewSimulation("standard_medium", 1001);

  // Start 3D Render Loop
  animate();

  // Start simulation background polling loop
  runSimulationLoop();
});
