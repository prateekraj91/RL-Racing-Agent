/**
 * RL Racing Agent // Final Product Reconstruction
 * Concept-Matched Autonomous Simulation Dashboard & Telemetry Lab
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
  graphicsQuality: "high", // high | low
  cinematicMode: false,

  // Deterministic Rollout Playback Engine
  playback: {
    rolloutData: null,
    isPlaying: false,
    playhead: 0.0, // Floating simulation step index [0.0 .. maxStep]
    speed: 1.0, // Playback speed multiplier (0.5, 1.0, 2.0, 3.0, 5.0)
    nominalDt: 0.030, // Nominal step duration in seconds (33.3 steps/sec)
    maxStep: 0,
    lastStepIndex: -1,
    isLoading: false,
  },

  gltfLoaded: false,
  gltfLoading: false,

  // Track cache
  trackData: null,
  trackCenter: { x: 400.0, y: 300.0 },

  // Camera lerp targets
  cameraPos: new THREE.Vector3(0, 45, 140),
  cameraTarget: new THREE.Vector3(0, 6, 0),

  // Trajectory buffer
  trajectoryPoints: [],
  prevLapProgress: 0.0,

  // Policy Saliency
  currentSaliency: null,
  historicalSaliency: null,
  saliencyTab: "steer", // steer | long | hist
};

window.STATE = STATE;

// Telemetry History Buffer for Live Sparkline Charts (last 80 steps)
const TELEM_HISTORY = {
  maxPoints: 80,
  speed: [],
  steer: [],
  throttle: [],
  lateral: [],
  slip: [],
};

// Renderer Non-Finite Guards
let hasWarnedNonFinite = false;

function isValidNumber(n) {
  return typeof n === "number" && Number.isFinite(n);
}

function isValidVector2(v) {
  return Array.isArray(v) && v.length >= 2 && isValidNumber(v[0]) && isValidNumber(v[1]);
}

// ==============================================================================
// Three.js 3D Scene Architecture & Rendering Pipeline
// ==============================================================================

let scene, camera, renderer, controls;
let composer, renderPass, bloomPass;
let hdrEnvMap = null;
let pbrTextures = {
  asphaltDiff: null,
  asphaltNor: null,
  asphaltRough: null,
  groundDiff: null,
  groundNor: null,
  groundRough: null,
};

let mainSun = null;
let threeCanvas, minimapCanvas, minimapCtx;
let trackGroup, propsGroup, carGroup, carBodyGroup, trajectoryMesh, lidarGroup, debugGroup;
let frontLeftSteerPivot, frontRightSteerPivot;
let wheelSpinMeshes = [];
let rearBrakeLights = [];
let carEmitterNodes = [];
let wheelSpinAngle = 0.0;
let rayBeamObjects = [];
let roadMesh, curbInnerMesh, curbOuterMesh, barrierMesh, groundPlaneMesh;
let innerEdgeMesh, outerEdgeMesh, centerLineMesh;
let carBboxHelper, carPhysicsFootprint, trackAxesHelper, carAxesHelper;

function initThreeJS() {
  threeCanvas = document.getElementById("threeCanvas");
  minimapCanvas = document.getElementById("raceCanvas");
  if (minimapCanvas) {
    minimapCtx = minimapCanvas.getContext("2d");
  }

  const container = document.getElementById("canvasContainer");
  const width = (container && container.clientWidth > 0) ? container.clientWidth : (window.innerWidth || 1280);
  const height = (container && container.clientHeight > 0) ? container.clientHeight : (window.innerHeight || 800);

  // 1. Scene & Atmosphere
  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x9bc2e0, 1400.0, 4200.0);

  // 2. Camera Setup (46 deg FOV for sharp, natural perspective matching concept)
  camera = new THREE.PerspectiveCamera(46, width / height, 2.0, 5000);
  camera.position.set(0, 45, 140);
  camera.lookAt(0, 6, 0);

  // 3. WebGL Renderer
  renderer = new THREE.WebGLRenderer({
    canvas: threeCanvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  // PBR Tone Mapping & Color Management
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.physicallyCorrectLights = true;

  // High-Resolution PCF Soft Shadows
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // Attach global references for debugging & tests
  window.renderer = renderer;
  window.scene = scene;
  window.camera = camera;

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
  window.trackGroup = trackGroup;

  propsGroup = new THREE.Group();
  scene.add(propsGroup);

  carGroup = new THREE.Group();
  scene.add(carGroup);
  window.carGroup = carGroup;

  lidarGroup = new THREE.Group();
  scene.add(lidarGroup);

  debugGroup = new THREE.Group();
  debugGroup.visible = false;
  scene.add(debugGroup);

  // 6. Lighting & HDRI Skybox Environment
  setupLighting();
  loadHDRIEnvironment();
  loadPBRTextures();

  // 7. Post-Processing Pipeline
  initEffectComposer(width, height);

  // 8. Build Concept Vehicle & Sensor Visualizer
  buildCarModel();
  buildLidarBeams();
  buildDebugHelpers();

  // 9. Resize Handling
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
    const w = container ? container.clientWidth : window.innerWidth;
    const h = container ? container.clientHeight : window.innerHeight;
    onCanvasResize(w, h);
  });
}

function onCanvasResize(width, height) {
  if (!width || !height) {
    width = window.innerWidth;
    height = window.innerHeight;
  }
  if (!renderer || !camera || width <= 0 || height <= 0) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  const pr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(pr);
  renderer.setSize(width, height, false);

  if (composer) {
    composer.setPixelRatio(pr);
    composer.setSize(width, height);
  }
}

function setupLighting() {
  const ambient = new THREE.AmbientLight(0xfff6ee, 0.25);
  scene.add(ambient);

  mainSun = new THREE.DirectionalLight(0xfff8ee, 1.20);
  mainSun.position.set(260, 440, 200);
  mainSun.castShadow = true;
  mainSun.shadow.mapSize.width = 2048;
  mainSun.shadow.mapSize.height = 2048;
  mainSun.shadow.camera.near = 50;
  mainSun.shadow.camera.far = 1200;

  mainSun.shadow.camera.left = -110;
  mainSun.shadow.camera.right = 110;
  mainSun.shadow.camera.top = 110;
  mainSun.shadow.camera.bottom = -110;
  mainSun.shadow.bias = -0.00015;
  mainSun.shadow.normalBias = 0.02;
  scene.add(mainSun);
}

function loadHDRIEnvironment() {
  if (!THREE.RGBELoader) {
    scene.background = new THREE.Color(0x8ecae6);
    return;
  }

  const rgbeLoader = new THREE.RGBELoader();
  rgbeLoader.setDataType(THREE.UnsignedByteType);
  rgbeLoader.load(
    "/static/assets/hdr/kloofendal_48d_partly_cloudy_puresky_2k.hdr",
    (texture) => {
      const pmremGenerator = new THREE.PMREMGenerator(renderer);
      pmremGenerator.compileEquirectangularShader();
      hdrEnvMap = pmremGenerator.fromEquirectangular(texture).texture;

      scene.background = hdrEnvMap;
      scene.environment = hdrEnvMap;

      texture.dispose();
      pmremGenerator.dispose();
    },
    undefined,
    (err) => {
      console.warn("HDRI load notice: using procedural sky fallback", err);
      scene.background = new THREE.Color(0x8ecae6);
    }
  );
}

// ==============================================================================
// Procedural Texture Generators (Lush Natural Lawn & Illuminated City Skyline)
// ==============================================================================

function createProceduralGrassTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  // Rich base lawn green (manicured racing circuit grass)
  ctx.fillStyle = "#388434";
  ctx.fillRect(0, 0, 512, 512);

  // Micro-contrast natural blade tone patches
  const colors = ["#2d6d2a", "#3f923b", "#499e44", "#357c32", "#52aa4c", "#32772f"];
  for (let i = 0; i < 4500; i++) {
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const r = Math.random() * 3.5 + 1.2;
    ctx.fillStyle = colors[Math.floor(Math.random() * colors.length)];
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(40, 40);
  tex.anisotropy = renderer ? Math.min(renderer.capabilities.getMaxAnisotropy(), 8) : 4;
  return tex;
}

function createProceduralBuildingTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  // Dark slate-blue architectural facade
  ctx.fillStyle = "#222c3d";
  ctx.fillRect(0, 0, 256, 512);

  // Vertical architectural mullions
  ctx.fillStyle = "#18202c";
  for (let x = 0; x < 256; x += 16) {
    ctx.fillRect(x, 0, 2, 512);
  }

  // Window grid with warm white and cool cyan lit office windows
  const rows = 32;
  const cols = 16;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const isLit = Math.sin(r * 12.3 + c * 34.5) > 0.15;
      if (isLit) {
        const warm = Math.sin(r * 5.1 + c * 9.2) > -0.2;
        ctx.fillStyle = warm ? "#ffeaa7" : "#a8e6cf";
        ctx.globalAlpha = 0.65 + Math.random() * 0.35;
        ctx.fillRect(c * 16 + 3, r * 16 + 3, 9, 10);
      } else {
        ctx.fillStyle = "#111722";
        ctx.globalAlpha = 0.85;
        ctx.fillRect(c * 16 + 3, r * 16 + 3, 9, 10);
      }
    }
  }
  ctx.globalAlpha = 1.0;

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 6);
  return tex;
}

let proceduralGrassTex = null;
let proceduralBuildingTex = null;

function loadPBRTextures() {
  const loader = new THREE.TextureLoader();

  function configureTexture(tex, repeatX, repeatY) {
    if (!tex) return;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeatX, repeatY);
    tex.anisotropy = renderer ? Math.min(renderer.capabilities.getMaxAnisotropy(), 8) : 4;
  }

  pbrTextures.asphaltDiff = loader.load("/static/assets/textures/asphalt/asphalt_diff_1k.jpg", (tex) => configureTexture(tex, 1, 16));
  pbrTextures.asphaltNor = loader.load("/static/assets/textures/asphalt/asphalt_nor_1k.jpg", (tex) => configureTexture(tex, 1, 16));
  pbrTextures.asphaltRough = loader.load("/static/assets/textures/asphalt/asphalt_rough_1k.jpg", (tex) => configureTexture(tex, 1, 16));

  proceduralGrassTex = createProceduralGrassTexture();
  proceduralBuildingTex = createProceduralBuildingTexture();
  pbrTextures.groundDiff = proceduralGrassTex;
  pbrTextures.groundNor = loader.load("/static/assets/textures/ground/ground_nor_1k.jpg", (tex) => configureTexture(tex, 32, 32));
  pbrTextures.groundRough = loader.load("/static/assets/textures/ground/ground_rough_1k.jpg", (tex) => configureTexture(tex, 32, 32));
}

function initEffectComposer(width, height) {
  if (!THREE.EffectComposer || !THREE.RenderPass) return;

  const pr = Math.min(window.devicePixelRatio || 1, 2);
  const renderTarget = new THREE.WebGLRenderTarget(width * pr, height * pr, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    encoding: THREE.sRGBEncoding,
  });

  composer = new THREE.EffectComposer(renderer, renderTarget);
  composer.setPixelRatio(pr);
  renderPass = new THREE.RenderPass(scene, camera);
  composer.addPass(renderPass);

  // Subtle bloom for lights and glowing UI rays
  if (THREE.UnrealBloomPass) {
    bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(width, height), 0.22, 0.4, 0.90);
    composer.addPass(bloomPass);
  }
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

  // 1. PBR Asphalt Road Surface Quad Strip
  const roadGeo = new THREE.BufferGeometry();
  const roadVertices = [];
  const roadNormals = [];
  const roadUvs = [];
  const roadIndices = [];

  let accumulatedV = 0.0;
  for (let i = 0; i < n; i++) {
    const pIn = innerPts[i];
    const pOut = outerPts[i];

    if (!isValidVector2(pIn) || !isValidVector2(pOut)) continue;

    const inW = envToWorld(pIn[0], pIn[1]);
    const outW = envToWorld(pOut[0], pOut[1]);

    roadVertices.push(inW.x, 0.60, inW.z);
    roadVertices.push(outW.x, 0.60, outW.z);
    roadNormals.push(0, 1, 0, 0, 1, 0);

    if (i > 0) {
      const prevIn = envToWorld(innerPts[i - 1][0], innerPts[i - 1][1]);
      accumulatedV += Math.hypot(inW.x - prevIn.x, inW.z - prevIn.z) / 22.0;
    }

    roadUvs.push(0.0, accumulatedV, 1.0, accumulatedV);

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
    color: 0x24272e, // Deep charcoal racetrack asphalt
    map: pbrTextures.asphaltDiff,
    normalMap: pbrTextures.asphaltNor,
    roughnessMap: pbrTextures.asphaltRough,
    roughness: 0.76,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });
  roadMesh = new THREE.Mesh(roadGeo, roadMat);
  roadMesh.receiveShadow = true;
  trackGroup.add(roadMesh);

  // 2. Continuous Solid White Edge Markings (Inner & Outer boundaries)
  buildTrackEdgeLines(innerPts, outerPts);

  // 3. Vibrant Glowing Cyan Autonomous Guidance Centerline
  if (centerPts.length > 2) {
    centerLineMesh = buildCenterline(centerPts);
    trackGroup.add(centerLineMesh);
  }

  // 4. Red/White Corner Curbs
  const curvatures = computeCenterlineCurvature(centerPts);
  curbInnerMesh = buildCurvatureBasedCurbs(innerPts, curvatures, -4.5, 0.75);
  curbOuterMesh = buildCurvatureBasedCurbs(outerPts, curvatures, 4.5, 0.75);
  if (curbInnerMesh) trackGroup.add(curbInnerMesh);
  if (curbOuterMesh) trackGroup.add(curbOuterMesh);

  // 5. White Lower + Racing Blue Top Armco Barrier Walls
  const barrierOuterMesh = buildBarrierWall(outerPts, 4.5, 7.5);
  const barrierInnerMesh = buildBarrierWall(innerPts, -4.5, 7.5);
  if (barrierOuterMesh) trackGroup.add(barrierOuterMesh);
  if (barrierInnerMesh) trackGroup.add(barrierInnerMesh);

  // 6. Start / Finish Line Banner
  if (innerPts.length > 0 && outerPts.length > 0) {
    buildStartFinishMarker(innerPts[0], outerPts[0]);
  }

  // 7. Ground Runoff Plane (Lush natural lawn green)
  const groundGeo = new THREE.PlaneGeometry(4200, 4200);
  groundGeo.rotateX(-Math.PI / 2);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0x388334, // Natural lush circuit lawn green matching concept video
    map: proceduralGrassTex || pbrTextures.groundDiff,
    normalMap: pbrTextures.groundNor,
    roughnessMap: pbrTextures.groundRough,
    roughness: 0.82,
    metalness: 0.02,
  });
  groundPlaneMesh = new THREE.Mesh(groundGeo, groundMat);
  groundPlaneMesh.position.y = -0.40;
  groundPlaneMesh.receiveShadow = true;
  trackGroup.add(groundPlaneMesh);

  // 8. Procedural Props: Faceted Pine/Oak Trees, Lamp Posts, and Distant Horizon Skyline
  buildEnvironmentProps(trackData);

  clearTrajectoryTrail();
}

function buildTrackEdgeLines(innerPts, outerPts) {
  const n = Math.min(innerPts.length, outerPts.length);
  const lineWidth = 1.3;

  function buildSingleEdge(pts, isOuter) {
    const geo = new THREE.BufferGeometry();
    const verts = [];
    const idxs = [];

    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const nextP = pts[(i + 1) % n];
      if (!isValidVector2(p) || !isValidVector2(nextP)) continue;

      const dx = nextP[0] - p[0];
      const dy = nextP[1] - p[1];
      const len = Math.hypot(dx, dy) || 1.0;
      const nx = -dy / len;
      const ny = dx / len;

      const offsetSign = isOuter ? -1.0 : 1.0;
      const p0 = envToWorld(p[0], p[1]);
      const p1 = envToWorld(p[0] + nx * lineWidth * offsetSign, p[1] + ny * lineWidth * offsetSign);

      verts.push(p0.x, 0.65, p0.z);
      verts.push(p1.x, 0.65, p1.z);

      const next = (i + 1) % n;
      const i0 = i * 2;
      const i1 = i * 2 + 1;
      const i2 = next * 2;
      const i3 = next * 2 + 1;

      idxs.push(i0, i1, i2);
      idxs.push(i1, i3, i2);
    }

    geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    geo.setIndex(idxs);
    geo.computeVertexNormals();

    const mat = new THREE.MeshBasicMaterial({ color: 0xf5f7fb, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true;
    return mesh;
  }

  innerEdgeMesh = buildSingleEdge(innerPts, false);
  outerEdgeMesh = buildSingleEdge(outerPts, true);
  if (innerEdgeMesh) trackGroup.add(innerEdgeMesh);
  if (outerEdgeMesh) trackGroup.add(outerEdgeMesh);
}

function computeCenterlineCurvature(centerPts) {
  const n = centerPts.length;
  const curvatures = new Float32Array(n);
  if (n < 4) return curvatures;

  for (let i = 0; i < n; i++) {
    const prev = centerPts[(i - 1 + n) % n];
    const curr = centerPts[i];
    const next = centerPts[(i + 1) % n];

    const dx1 = curr[0] - prev[0];
    const dy1 = curr[1] - prev[1];
    const dx2 = next[0] - curr[0];
    const dy2 = next[1] - curr[1];

    const a1 = Math.atan2(dy1, dx1);
    const a2 = Math.atan2(dy2, dx2);

    let diff = a2 - a1;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;

    const segLen = (Math.hypot(dx1, dy1) + Math.hypot(dx2, dy2)) * 0.5 || 1.0;
    curvatures[i] = diff / segLen;
  }
  return curvatures;
}

function buildCurvatureBasedCurbs(boundaryPts, centerCurvatures, offsetDistance, height) {
  const n = boundaryPts.length;
  const curbGeo = new THREE.BufferGeometry();
  const vertices = [];
  const colors = [];
  const indices = [];

  let vertIdx = 0;
  for (let i = 0; i < n; i++) {
    const curv = Math.abs(centerCurvatures[i] || 0.0);
    if (curv <= 0.0025) continue;

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

    vertices.push(base.x, 0.62, base.z);
    vertices.push(ext.x, 0.62 + height, ext.z);

    const isRed = Math.floor(i / 2) % 2 === 0;
    const col = isRed ? [0.95, 0.12, 0.22] : [0.96, 0.96, 0.98];
    colors.push(...col, ...col);

    const nextCurv = Math.abs(centerCurvatures[(i + 1) % n] || 0.0);
    if (nextCurv > 0.0025) {
      const nextBase = envToWorld(nextPt[0], nextPt[1]);
      const nextExt = envToWorld(nextPt[0] + nx, nextPt[1] + ny);

      vertices.push(nextBase.x, 0.62, nextBase.z);
      vertices.push(nextExt.x, 0.62 + height, nextExt.z);
      colors.push(...col, ...col);

      indices.push(vertIdx, vertIdx + 1, vertIdx + 2);
      indices.push(vertIdx + 1, vertIdx + 3, vertIdx + 2);
      vertIdx += 4;
    }
  }

  if (vertices.length < 12) return null;

  curbGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  curbGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  curbGeo.setIndex(indices);
  curbGeo.computeVertexNormals();

  const curbMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.65,
    metalness: 0.15,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(curbGeo, curbMat);
  mesh.receiveShadow = true;
  return mesh;
}

function buildBarrierWall(boundaryPts, offsetDistance, wallHeight) {
  const n = boundaryPts.length;
  const wallGeo = new THREE.BufferGeometry();
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

    const pos = envToWorld(pt[0] + nx, pt[1] + ny);

    vertices.push(pos.x, -0.40, pos.z);
    colors.push(0.96, 0.96, 0.98); // White lower concrete wall

    vertices.push(pos.x, wallHeight, pos.z);
    colors.push(0.08, 0.45, 0.95); // Racing Blue top cap

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
  wallGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  wallGeo.setIndex(indices);
  wallGeo.computeVertexNormals();

  const wallMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    metalness: 0.35,
    roughness: 0.38,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(wallGeo, wallMat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildCenterline(centerPts) {
  const n = centerPts.length;
  if (n < 4) return null;
  const geo = new THREE.BufferGeometry();
  const verts = [];
  const idxs = [];
  const halfW = 0.55;

  for (let i = 0; i < n; i++) {
    const p = centerPts[i];
    const nextP = centerPts[(i + 1) % n];
    if (!isValidVector2(p) || !isValidVector2(nextP)) continue;

    const dx = nextP[0] - p[0];
    const dy = nextP[1] - p[1];
    const len = Math.hypot(dx, dy) || 1.0;
    const nx = -dy / len;
    const ny = dx / len;

    const p0 = envToWorld(p[0] - nx * halfW, p[1] - ny * halfW);
    const p1 = envToWorld(p[0] + nx * halfW, p[1] + ny * halfW);

    verts.push(p0.x, 0.68, p0.z);
    verts.push(p1.x, 0.68, p1.z);

    const next = (i + 1) % n;
    const i0 = i * 2;
    const i1 = i * 2 + 1;
    const i2 = next * 2;
    const i3 = next * 2 + 1;

    idxs.push(i0, i1, i2);
    idxs.push(i1, i3, i2);
  }

  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setIndex(idxs);
  geo.computeVertexNormals();

  const mat = new THREE.MeshBasicMaterial({
    color: 0x00f0ff,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  return mesh;
}

function buildStartFinishMarker(inPt, outPt) {
  if (!isValidVector2(inPt) || !isValidVector2(outPt)) return;
  const inW = envToWorld(inPt[0], inPt[1]);
  const outW = envToWorld(outPt[0], outPt[1]);

  const sfGeo = new THREE.BufferGeometry();
  const vertices = [
    inW.x, 0.70, inW.z,
    outW.x, 0.70, outW.z,
    inW.x + 3.0, 0.70, inW.z,
    outW.x + 3.0, 0.70, outW.z,
  ];
  sfGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  sfGeo.setIndex([0, 1, 2, 1, 3, 2]);

  const sfMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const mesh = new THREE.Mesh(sfGeo, sfMat);
  trackGroup.add(mesh);
}

// ==============================================================================
// Scenery & Environment Props (Low-poly faceted trees + City Skyline Backdrop)
// ==============================================================================

function buildEnvironmentProps(trackData) {
  if (!propsGroup) return;

  while (propsGroup.children.length > 0) {
    const obj = propsGroup.children[0];
    propsGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }

  const outerPts = trackData.outer_boundary || [];
  const centerPts = trackData.centerline || [];
  const n = outerPts.length;
  if (n < 4) return;

  const dummy = new THREE.Object3D();

  // 1. Instanced Pine Trees (Faceted Cones + Trunk)
  const pineTrunkGeo = new THREE.CylinderGeometry(0.8, 1.2, 8.0, 6);
  pineTrunkGeo.translate(0, 4.0, 0);
  const pineFoliageGeo = new THREE.ConeGeometry(6.4, 13.5, 6);
  pineFoliageGeo.translate(0, 12.5, 0);

  const trunkMat = new THREE.MeshStandardMaterial({
    color: 0x4a2e18,
    roughness: 0.9,
    metalness: 0.05,
  });

  const pineMat1 = new THREE.MeshStandardMaterial({ color: 0x1b4332, roughness: 0.78 }); // Forest Pine
  const pineMat2 = new THREE.MeshStandardMaterial({ color: 0x2d6a4f, roughness: 0.75 }); // Emerald Pine

  // 2. Instanced Deciduous Oak Trees (Faceted Dodecahedrons + Trunk)
  const oakTrunkGeo = new THREE.CylinderGeometry(0.9, 1.3, 9.0, 6);
  oakTrunkGeo.translate(0, 4.5, 0);
  const oakFoliageGeo = new THREE.DodecahedronGeometry(6.8, 1);
  oakFoliageGeo.translate(0, 13.0, 0);

  const oakMat1 = new THREE.MeshStandardMaterial({ color: 0x40916c, roughness: 0.74 }); // Mint Green
  const oakMat2 = new THREE.MeshStandardMaterial({ color: 0x52796f, roughness: 0.78 }); // Olive Green

  // 3. Lamp Posts
  const poleGeo = new THREE.CylinderGeometry(0.4, 0.6, 20.0, 6);
  poleGeo.translate(0, 10.0, 0);
  const lampFixtureGeo = new THREE.BoxGeometry(3.5, 0.6, 1.4);
  lampFixtureGeo.translate(1.5, 20.0, 0);

  const poleMat = new THREE.MeshStandardMaterial({ color: 0x3a404d, metalness: 0.7, roughness: 0.3 });
  const lampFixtureMat = new THREE.MeshStandardMaterial({
    color: 0xfffae0,
    emissive: 0xfff3b0,
    emissiveIntensity: 1.8,
  });

  // 4. Tecpro Corner Runoff Barriers
  const blockGeo = new THREE.BoxGeometry(6.5, 3.8, 3.2);
  blockGeo.translate(0, 1.9, 0);
  const redBlockMat = new THREE.MeshStandardMaterial({ color: 0xd90429, roughness: 0.5, metalness: 0.1 });
  const whiteBlockMat = new THREE.MeshStandardMaterial({ color: 0xf8f9fa, roughness: 0.5, metalness: 0.1 });

  const pine1Transforms = [];
  const pine2Transforms = [];
  const oak1Transforms = [];
  const oak2Transforms = [];
  const lampTransforms = [];
  const tecproTransforms = [];

  const curvatures = computeCenterlineCurvature(centerPts);

  for (let i = 0; i < n; i++) {
    const pt = outerPts[i];
    const nextPt = outerPts[(i + 1) % n];
    if (!isValidVector2(pt) || !isValidVector2(nextPt)) continue;

    const dx = nextPt[0] - pt[0];
    const dy = nextPt[1] - pt[1];
    const len = Math.hypot(dx, dy) || 1.0;
    const nx = -dy / len;
    const ny = dx / len;

    const curv = Math.abs(curvatures[i] || 0.0);
    const isCorner = curv > 0.0028;

    // Lamp posts along track perimeter
    if (i % 7 === 0) {
      const lx = pt[0] + nx * 30.0;
      const ly = pt[1] + ny * 30.0;
      const w = envToWorld(lx, ly);
      const angle = Math.atan2(ny, nx) + Math.PI;
      lampTransforms.push({ x: w.x, z: w.z, rotY: angle });
    }

    // Tecpro barriers on corner exits outside barrier
    if (isCorner && i % 2 === 0) {
      const bx = pt[0] + nx * 24.0;
      const by = pt[1] + ny * 24.0;
      const w = envToWorld(bx, by);
      const angle = Math.atan2(dy, dx);
      tecproTransforms.push({ x: w.x, z: w.z, rotY: angle, isRed: (i / 2) % 2 === 0 });
    }

    // Trees scattered outside track
    if (i % 2 === 0) {
      const hash = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
      const frac = hash - Math.floor(hash);
      const treeOffset = 28.0 + frac * 65.0;
      const tx = pt[0] + nx * treeOffset;
      const ty = pt[1] + ny * treeOffset;
      const w = envToWorld(tx, ty);
      const s = 0.95 + frac * 0.45;
      const rot = frac * Math.PI * 2;

      const pick = i % 4;
      if (pick === 0) pine1Transforms.push({ x: w.x, z: w.z, scale: s, rotY: rot });
      else if (pick === 1) pine2Transforms.push({ x: w.x, z: w.z, scale: s, rotY: rot });
      else if (pick === 2) oak1Transforms.push({ x: w.x, z: w.z, scale: s, rotY: rot });
      else oak2Transforms.push({ x: w.x, z: w.z, scale: s, rotY: rot });
    }
  }

  // Helper to instantiate tree meshes
  function instantiateTrees(trunkGeo, foliageGeo, foliageMat, transforms) {
    if (transforms.length === 0) return;
    const count = transforms.length;
    const trunkMesh = new THREE.InstancedMesh(trunkGeo, trunkMat, count);
    const foliageMesh = new THREE.InstancedMesh(foliageGeo, foliageMat, count);
    trunkMesh.castShadow = true;
    trunkMesh.receiveShadow = true;
    foliageMesh.castShadow = true;
    foliageMesh.receiveShadow = true;

    transforms.forEach((t, idx) => {
      dummy.position.set(t.x, 0, t.z);
      dummy.rotation.set(0, t.rotY, 0);
      dummy.scale.set(t.scale, t.scale, t.scale);
      dummy.updateMatrix();
      trunkMesh.setMatrixAt(idx, dummy.matrix);
      foliageMesh.setMatrixAt(idx, dummy.matrix);
    });

    trunkMesh.instanceMatrix.needsUpdate = true;
    foliageMesh.instanceMatrix.needsUpdate = true;
    propsGroup.add(trunkMesh);
    propsGroup.add(foliageMesh);
  }

  instantiateTrees(pineTrunkGeo, pineFoliageGeo, pineMat1, pine1Transforms);
  instantiateTrees(pineTrunkGeo, pineFoliageGeo, pineMat2, pine2Transforms);
  instantiateTrees(oakTrunkGeo, oakFoliageGeo, oakMat1, oak1Transforms);
  instantiateTrees(oakTrunkGeo, oakFoliageGeo, oakMat2, oak2Transforms);

  // Lamp Posts
  if (lampTransforms.length > 0) {
    const poleMesh = new THREE.InstancedMesh(poleGeo, poleMat, lampTransforms.length);
    const fixtureMesh = new THREE.InstancedMesh(lampFixtureGeo, lampFixtureMat, lampTransforms.length);
    poleMesh.castShadow = true;
    lampTransforms.forEach((t, idx) => {
      dummy.position.set(t.x, 0, t.z);
      dummy.rotation.set(0, t.rotY, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      poleMesh.setMatrixAt(idx, dummy.matrix);
      fixtureMesh.setMatrixAt(idx, dummy.matrix);
    });
    poleMesh.instanceMatrix.needsUpdate = true;
    fixtureMesh.instanceMatrix.needsUpdate = true;
    propsGroup.add(poleMesh);
    propsGroup.add(fixtureMesh);
  }

  // Tecpro Barriers
  const tecproRed = tecproTransforms.filter((t) => t.isRed);
  const tecproWhite = tecproTransforms.filter((t) => !t.isRed);
  if (tecproRed.length > 0) {
    const redMesh = new THREE.InstancedMesh(blockGeo, redBlockMat, tecproRed.length);
    redMesh.castShadow = true;
    tecproRed.forEach((t, idx) => {
      dummy.position.set(t.x, 0, t.z);
      dummy.rotation.set(0, t.rotY, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      redMesh.setMatrixAt(idx, dummy.matrix);
    });
    redMesh.instanceMatrix.needsUpdate = true;
    propsGroup.add(redMesh);
  }
  if (tecproWhite.length > 0) {
    const whiteMesh = new THREE.InstancedMesh(blockGeo, whiteBlockMat, tecproWhite.length);
    whiteMesh.castShadow = true;
    tecproWhite.forEach((t, idx) => {
      dummy.position.set(t.x, 0, t.z);
      dummy.rotation.set(0, t.rotY, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      whiteMesh.setMatrixAt(idx, dummy.matrix);
    });
    whiteMesh.instanceMatrix.needsUpdate = true;
    propsGroup.add(whiteMesh);
  }

  // Distant City Skyline Skyscrapers on Horizon (Creates depth matching concept)
  buildDistantCitySkyline();
}

function buildDistantCitySkyline() {
  const buildingMat = new THREE.MeshStandardMaterial({
    color: 0x3d4b60,
    map: proceduralBuildingTex,
    emissive: 0x354458,
    emissiveMap: proceduralBuildingTex,
    emissiveIntensity: 0.35,
    metalness: 0.40,
    roughness: 0.45,
  });

  const spireMat = new THREE.MeshStandardMaterial({
    color: 0x90a4ae,
    metalness: 0.85,
    roughness: 0.20,
  });

  const buildingCount = 36;
  const dummy = new THREE.Object3D();

  // Instanced main towers
  const towerGeo = new THREE.BoxGeometry(1, 1, 1);
  const towerMesh = new THREE.InstancedMesh(towerGeo, buildingMat, buildingCount);
  towerMesh.castShadow = false;
  towerMesh.receiveShadow = false;

  // Instanced setback crown towers
  const crownGeo = new THREE.BoxGeometry(1, 1, 1);
  const crownMesh = new THREE.InstancedMesh(crownGeo, buildingMat, buildingCount);
  crownMesh.castShadow = false;

  // Instanced spire needles
  const spireGeo = new THREE.CylinderGeometry(0.12, 0.4, 1.0, 8);
  const spireMesh = new THREE.InstancedMesh(spireGeo, spireMat, buildingCount);

  for (let i = 0; i < buildingCount; i++) {
    const angle = (i / buildingCount) * Math.PI * 2;
    const dist = 780.0 + (Math.sin(i * 3.3) * 0.5 + 0.5) * 420.0;
    const x = Math.cos(angle) * dist;
    const z = Math.sin(angle) * dist;

    const bWidth = 38.0 + (i % 5) * 10.0;
    const bDepth = 38.0 + ((i + 2) % 4) * 10.0;
    const bHeight = 80.0 + (Math.sin(i * 1.7) * 0.5 + 0.5) * 95.0;

    // Main base tower
    dummy.position.set(x, bHeight / 2 - 2, z);
    dummy.rotation.set(0, angle, 0);
    dummy.scale.set(bWidth, bHeight, bDepth);
    dummy.updateMatrix();
    towerMesh.setMatrixAt(i, dummy.matrix);

    // Architectural setback crown
    const crownWidth = bWidth * 0.65;
    const crownDepth = bDepth * 0.65;
    const crownHeight = bHeight * 0.35;
    dummy.position.set(x, bHeight + crownHeight / 2 - 2, z);
    dummy.scale.set(crownWidth, crownHeight, crownDepth);
    dummy.updateMatrix();
    crownMesh.setMatrixAt(i, dummy.matrix);

    // Rooftop antenna spire
    const spireHeight = 18.0 + (i % 3) * 10.0;
    dummy.position.set(x, bHeight + crownHeight + spireHeight / 2 - 2, z);
    dummy.scale.set(1.0, spireHeight, 1.0);
    dummy.updateMatrix();
    spireMesh.setMatrixAt(i, dummy.matrix);
  }

  towerMesh.instanceMatrix.needsUpdate = true;
  crownMesh.instanceMatrix.needsUpdate = true;
  spireMesh.instanceMatrix.needsUpdate = true;

  propsGroup.add(towerMesh);
  propsGroup.add(crownMesh);
  propsGroup.add(spireMesh);
}

// ==============================================================================
// 3D Racing Vehicle (Unbranded Stylized Sports Coupe Matching Concept Video)
// ==============================================================================

function buildProceduralVehicle() {
  if (carBodyGroup) {
    while (carBodyGroup.children.length > 0) {
      const obj = carBodyGroup.children[0];
      carBodyGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
    }
  } else {
    carBodyGroup = new THREE.Group();
    carGroup.add(carBodyGroup);
  }

  wheelSpinMeshes = [];
  rearBrakeLights = [];
  carEmitterNodes = [];

  const length = 50.0; // Wheelbase
  const width = 27.0;
  const height = 12.0;

  // Materials
  const liveryMat = new THREE.MeshStandardMaterial({
    color: 0x8a0c20, // Deep crimson sports car red matching concept reference frames
    metalness: 0.0,
    roughness: 0.48,
    envMapIntensity: 0.0,
  });

  const carbonMat = new THREE.MeshStandardMaterial({
    color: 0x16181c, // Matte carbon fiber
    metalness: 0.60,
    roughness: 0.40,
    envMapIntensity: 0.15,
  });

  const glassMat = new THREE.MeshPhysicalMaterial({
    color: 0x0e131d, // Dark GT cockpit glass
    metalness: 0.85,
    roughness: 0.12,
    transparent: true,
    opacity: 0.92,
  });

  // 1. Lower Cockpit Tub & Chassis
  const tubGeo = new THREE.BoxGeometry(length * 0.86, height * 0.36, width * 0.78);
  const tubMesh = new THREE.Mesh(tubGeo, liveryMat);
  tubMesh.position.y = height * 0.36;
  tubMesh.castShadow = true;
  tubMesh.receiveShadow = true;
  carBodyGroup.add(tubMesh);

  // 1b. Rear Fastback Deck & Fascia (sculpts the vertical rear body matching concept)
  const rearDeckGeo = new THREE.BoxGeometry(length * 0.28, height * 0.38, width * 0.76);
  const rearDeckMesh = new THREE.Mesh(rearDeckGeo, liveryMat);
  rearDeckMesh.position.set(-length * 0.30, height * 0.48, 0);
  rearDeckMesh.castShadow = true;
  rearDeckMesh.receiveShadow = true;
  carBodyGroup.add(rearDeckMesh);

  // 2. Sculpted Sloping Front Hood & Low Nose
  const hoodGeo = new THREE.BoxGeometry(length * 0.28, height * 0.24, width * 0.74);
  const hoodMesh = new THREE.Mesh(hoodGeo, liveryMat);
  hoodMesh.position.set(length * 0.35, height * 0.34, 0);
  hoodMesh.rotation.z = -0.16;
  hoodMesh.castShadow = true;
  carBodyGroup.add(hoodMesh);

  // Twin hood air extraction louvers in carbon
  const ventGeo = new THREE.BoxGeometry(length * 0.12, 0.4, width * 0.22);
  const ventL = new THREE.Mesh(ventGeo, carbonMat);
  ventL.position.set(length * 0.32, height * 0.44, -width * 0.20);
  ventL.rotation.z = -0.16;
  const ventR = new THREE.Mesh(ventGeo, carbonMat);
  ventR.position.set(length * 0.32, height * 0.44, width * 0.20);
  ventR.rotation.z = -0.16;
  carBodyGroup.add(ventL);
  carBodyGroup.add(ventR);

  // 3. Extended Front Carbon Splitter with Endplates
  const splitterGeo = new THREE.BoxGeometry(length * 0.18, height * 0.05, width * 0.94);
  const splitter = new THREE.Mesh(splitterGeo, carbonMat);
  splitter.position.set(length * 0.46, height * 0.12, 0);
  splitter.castShadow = true;
  carBodyGroup.add(splitter);

  // Left & right splitter dive-planes
  const diveGeo = new THREE.BoxGeometry(length * 0.08, height * 0.12, 0.6);
  const diveL = new THREE.Mesh(diveGeo, carbonMat);
  diveL.position.set(length * 0.48, height * 0.18, -width * 0.46);
  const diveR = new THREE.Mesh(diveGeo, carbonMat);
  diveR.position.set(length * 0.48, height * 0.18, width * 0.46);
  carBodyGroup.add(diveL);
  carBodyGroup.add(diveR);

  // 4. Sculpted Front Wheel Arches (Fenders)
  const fArchGeo = new THREE.BoxGeometry(length * 0.24, height * 0.34, width * 0.12);
  const fArchL = new THREE.Mesh(fArchGeo, liveryMat);
  fArchL.position.set(length * 0.32, height * 0.42, -width * 0.41);
  fArchL.castShadow = true;
  const fArchR = new THREE.Mesh(fArchGeo, liveryMat);
  fArchR.position.set(length * 0.32, height * 0.42, width * 0.41);
  fArchR.castShadow = true;
  carBodyGroup.add(fArchL);
  carBodyGroup.add(fArchR);

  // 5. Flared Rear Wheel Arches (Wide Hips over rear competition tires)
  const rArchGeo = new THREE.BoxGeometry(length * 0.26, height * 0.38, width * 0.14);
  const rArchL = new THREE.Mesh(rArchGeo, liveryMat);
  rArchL.position.set(-length * 0.32, height * 0.45, -width * 0.41);
  rArchL.castShadow = true;
  const rArchR = new THREE.Mesh(rArchGeo, liveryMat);
  rArchR.position.set(-length * 0.32, height * 0.45, width * 0.41);
  rArchR.castShadow = true;
  carBodyGroup.add(rArchL);
  carBodyGroup.add(rArchR);

  // 6. Carbon Ground-Effect Side Skirts
  const skirtGeo = new THREE.BoxGeometry(length * 0.40, height * 0.05, width * 0.08);
  const skirtL = new THREE.Mesh(skirtGeo, carbonMat);
  skirtL.position.set(0, height * 0.14, -width * 0.42);
  const skirtR = new THREE.Mesh(skirtGeo, carbonMat);
  skirtR.position.set(0, height * 0.14, width * 0.42);
  carBodyGroup.add(skirtL);
  carBodyGroup.add(skirtR);

  // 7. Fastback GT Cockpit Greenhouse
  const cabinGeo = new THREE.BoxGeometry(length * 0.44, height * 0.42, width * 0.62);
  const cabinMesh = new THREE.Mesh(cabinGeo, glassMat);
  cabinMesh.position.set(-length * 0.02, height * 0.74, 0);
  cabinMesh.castShadow = true;
  carBodyGroup.add(cabinMesh);

  // Roof panel in Rosso red
  const roofGeo = new THREE.BoxGeometry(length * 0.36, 0.4, width * 0.58);
  const roofMesh = new THREE.Mesh(roofGeo, liveryMat);
  roofMesh.position.set(-length * 0.02, height * 0.95, 0);
  roofMesh.castShadow = true;
  carBodyGroup.add(roofMesh);

  // 8. Aggressive Rear Carbon Diffuser with Aero Strakes
  const diffGeo = new THREE.BoxGeometry(length * 0.14, height * 0.08, width * 0.88);
  const diffMesh = new THREE.Mesh(diffGeo, carbonMat);
  diffMesh.position.set(-length * 0.44, height * 0.14, 0);
  diffMesh.rotation.z = 0.20;
  carBodyGroup.add(diffMesh);

  // 9. GT Airfoil Rear Wing with Vertical Endplates
  const wingGeo = new THREE.BoxGeometry(length * 0.16, height * 0.05, width * 0.96);
  const wing = new THREE.Mesh(wingGeo, carbonMat);
  wing.position.set(-length * 0.45, height * 1.05, 0);
  wing.castShadow = true;
  carBodyGroup.add(wing);

  // Dual aerodynamic wing endplates
  const endplateGeo = new THREE.BoxGeometry(length * 0.22, height * 0.20, 0.6);
  const epL = new THREE.Mesh(endplateGeo, carbonMat);
  epL.position.set(-length * 0.45, height * 1.05, -width * 0.48);
  const epR = new THREE.Mesh(endplateGeo, carbonMat);
  epR.position.set(-length * 0.45, height * 1.05, width * 0.48);
  carBodyGroup.add(epL);
  carBodyGroup.add(epR);

  // Swan-neck wing pylons
  const wingPillarGeo = new THREE.BoxGeometry(length * 0.04, height * 0.36, width * 0.04);
  const pL = new THREE.Mesh(wingPillarGeo, carbonMat);
  pL.position.set(-length * 0.44, height * 0.84, -width * 0.28);
  const pR = new THREE.Mesh(wingPillarGeo, carbonMat);
  pR.position.set(-length * 0.44, height * 0.84, width * 0.28);
  carBodyGroup.add(pL);
  carBodyGroup.add(pR);

  // 10. Aggressive Dual Front LED Headlights with Cyan Accents
  const headLightMat = new THREE.MeshBasicMaterial({ color: 0xe0f7fa });
  const headAccentMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
  const headGeo = new THREE.BoxGeometry(length * 0.04, height * 0.08, width * 0.15);
  const hL = new THREE.Mesh(headGeo, headLightMat);
  hL.position.set(length * 0.45, height * 0.38, -width * 0.28);
  const hR = new THREE.Mesh(headGeo, headLightMat);
  hR.position.set(length * 0.45, height * 0.38, width * 0.28);
  carBodyGroup.add(hL);
  carBodyGroup.add(hR);

  const dRLGeo = new THREE.BoxGeometry(length * 0.05, 0.5, width * 0.14);
  const dRLL = new THREE.Mesh(dRLGeo, headAccentMat);
  dRLL.position.set(length * 0.45, height * 0.33, -width * 0.28);
  const dRLR = new THREE.Mesh(dRLGeo, headAccentMat);
  dRLR.position.set(length * 0.45, height * 0.33, width * 0.28);
  carBodyGroup.add(dRLL);
  carBodyGroup.add(dRLR);

  // 11. Rear Emissive Active LED Taillights
  const brakeLightMat = new THREE.MeshStandardMaterial({
    color: 0xff1128,
    emissive: 0xff1128,
    emissiveIntensity: 1.5,
  });
  const brakeLightGeo = new THREE.BoxGeometry(0.8, 1.4, 4.8);
  const brakeL = new THREE.Mesh(brakeLightGeo, brakeLightMat);
  brakeL.position.set(-length * 0.44 - 0.2, height * 0.48, -width * 0.28);
  const brakeR = new THREE.Mesh(brakeLightGeo, brakeLightMat);
  brakeR.position.set(-length * 0.44 - 0.2, height * 0.48, width * 0.28);
  carBodyGroup.add(brakeL);
  carBodyGroup.add(brakeR);
  rearBrakeLights.push(brakeL, brakeR);

  // 12. Rangefinder Sensor Emitter Discs & Stalks (matching concept video)
  const stalkMat = new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.8, roughness: 0.3 });
  const emitterRingMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
  const emitterCenterMat = new THREE.MeshBasicMaterial({ color: 0x111620 });

  const emitterConfigs = [
    { x: -2.0, y: 5.5, z: -width * 0.52 },         // L -90°
    { x: length * 0.36, y: 5.5, z: -width * 0.46 }, // DL -45°
    { x: length * 0.49, y: 5.5, z: 0.0 },           // FWD 0°
    { x: length * 0.36, y: 5.5, z: width * 0.46 },  // DR +45°
    { x: -2.0, y: 5.5, z: width * 0.52 },          // R +90°
  ];

  emitterConfigs.forEach((cfg) => {
    const emitterNodeGroup = new THREE.Group();
    emitterNodeGroup.position.set(cfg.x, cfg.y, cfg.z);

    // Stalk connecting node to chassis
    const stalkGeo = new THREE.CylinderGeometry(0.35, 0.35, 2.2, 8);
    stalkGeo.rotateZ(Math.PI / 2);
    const stalk = new THREE.Mesh(stalkGeo, stalkMat);
    emitterNodeGroup.add(stalk);

    // White circular ring/disc
    const ringGeo = new THREE.RingGeometry(0.6, 1.4, 20);
    ringGeo.rotateX(-Math.PI / 2);
    const ring = new THREE.Mesh(ringGeo, emitterRingMat);
    emitterNodeGroup.add(ring);

    // Inner dark center
    const centerGeo = new THREE.CircleGeometry(0.58, 16);
    centerGeo.rotateX(-Math.PI / 2);
    const center = new THREE.Mesh(centerGeo, emitterCenterMat);
    emitterNodeGroup.add(center);

    carBodyGroup.add(emitterNodeGroup);
    carEmitterNodes.push(emitterNodeGroup);
  });

  // 13. Four GT Wheels & Rims with Yellow Brake Calipers
  const wheelRadius = 5.2;
  const wheelWidth = 3.6;
  const wheelRubberMat = new THREE.MeshStandardMaterial({
    color: 0x18181a,
    roughness: 0.85,
    metalness: 0.08,
  });
  const wheelRimMat = new THREE.MeshStandardMaterial({
    color: 0xdcdfe4, // Satin Silver / Gunmetal Rims
    metalness: 0.88,
    roughness: 0.22,
  });
  const wheelHubMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff }); // Anodized cyan centerlock nut
  const caliperMat = new THREE.MeshStandardMaterial({
    color: 0xffd600, // Vibrant Yellow Calipers
    metalness: 0.4,
    roughness: 0.3,
  });

  function createWheel() {
    const wheelGroup = new THREE.Group();

    // Spinning portion (tire + rim + spokes)
    const spinGroup = new THREE.Group();

    const tireGeo = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 20);
    tireGeo.rotateX(Math.PI / 2);
    const tireMesh = new THREE.Mesh(tireGeo, wheelRubberMat);
    tireMesh.castShadow = true;
    spinGroup.add(tireMesh);

    const rimGeo = new THREE.CylinderGeometry(wheelRadius * 0.66, wheelRadius * 0.66, wheelWidth + 0.12, 14);
    rimGeo.rotateX(Math.PI / 2);
    const rimMesh = new THREE.Mesh(rimGeo, wheelRimMat);
    spinGroup.add(rimMesh);

    // Central hub nut
    const hubGeo = new THREE.CylinderGeometry(0.8, 0.8, wheelWidth + 0.25, 8);
    hubGeo.rotateX(Math.PI / 2);
    const hubMesh = new THREE.Mesh(hubGeo, wheelHubMat);
    spinGroup.add(hubMesh);

    wheelGroup.add(spinGroup);

    // Static brake caliper (stays upright inside rim while wheel spins)
    const caliperGeo = new THREE.BoxGeometry(1.6, 2.6, 1.2);
    const caliperMesh = new THREE.Mesh(caliperGeo, caliperMat);
    caliperMesh.position.set(0, 2.2, 0);
    wheelGroup.add(caliperMesh);

    return { group: wheelGroup, spin: spinGroup };
  }

  // Front Left Steering Pivot
  frontLeftSteerPivot = new THREE.Group();
  frontLeftSteerPivot.position.set(length * 0.32, wheelRadius, -width * 0.46);
  const wFL = createWheel();
  frontLeftSteerPivot.add(wFL.group);
  carGroup.add(frontLeftSteerPivot);
  wheelSpinMeshes.push(wFL.spin);

  // Front Right Steering Pivot
  frontRightSteerPivot = new THREE.Group();
  frontRightSteerPivot.position.set(length * 0.32, wheelRadius, width * 0.46);
  const wFR = createWheel();
  frontRightSteerPivot.add(wFR.group);
  carGroup.add(frontRightSteerPivot);
  wheelSpinMeshes.push(wFR.spin);

  // Rear Left Wheel
  const rearLeftGroup = new THREE.Group();
  rearLeftGroup.position.set(-length * 0.32, wheelRadius, -width * 0.46);
  const wRL = createWheel();
  rearLeftGroup.add(wRL.group);
  carGroup.add(rearLeftGroup);
  wheelSpinMeshes.push(wRL.spin);

  // Rear Right Wheel
  const rearRightGroup = new THREE.Group();
  rearRightGroup.position.set(-length * 0.32, wheelRadius, width * 0.46);
  const wRR = createWheel();
  rearRightGroup.add(wRR.group);
  carGroup.add(rearRightGroup);
  wheelSpinMeshes.push(wRR.spin);
}

function buildCarModel() {
  while (carGroup.children.length > 0) {
    const obj = carGroup.children[0];
    carGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
  buildProceduralVehicle();
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

  const colors = [0xff2a5f, 0xffb700, 0x00f0ff, 0xffb700, 0xff2a5f];

  for (let i = 0; i < 5; i++) {
    const beamColor = colors[i];

    // 3D Solid Laser Beam Cylinder (Immune to WebGL 1px line limitations)
    const beamGeo = new THREE.CylinderGeometry(0.55, 0.55, 1.0, 10);
    const beamMat = new THREE.MeshBasicMaterial({
      color: beamColor,
      transparent: true,
      opacity: 0.95,
    });
    const beamMesh = new THREE.Mesh(beamGeo, beamMat);
    lidarGroup.add(beamMesh);

    // Ground Contact Ring at Hit Point
    const ringGeo = new THREE.RingGeometry(1.4, 2.5, 20);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: beamColor,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.95,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    lidarGroup.add(ring);

    // Spark Particle at Hit Point
    const sparkGeo = new THREE.SphereGeometry(1.1, 10, 10);
    const sparkMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const spark = new THREE.Mesh(sparkGeo, sparkMat);
    lidarGroup.add(spark);

    rayBeamObjects.push({ beam: beamMesh, ring, spark, color: beamColor });
  }
}

function updateLidarVisuals(rays, car) {
  if (!rays || rays.length < 5 || !carGroup) return;

  const origin = new THREE.Vector3(carGroup.position.x, 5.5, carGroup.position.z);

  for (let i = 0; i < 5; i++) {
    const ray = rays[i];
    const beamObj = rayBeamObjects[i];
    if (!beamObj || !ray) continue;

    if (!isValidVector2(ray.start) || !isValidVector2(ray.end) || !isValidNumber(ray.distance)) {
      continue;
    }

    // Emitter origin on car
    let rayOrigin = origin;
    if (carEmitterNodes[i]) {
      const worldNodePos = new THREE.Vector3();
      carEmitterNodes[i].getWorldPosition(worldNodePos);
      rayOrigin = worldNodePos;
    }

    const hitWorld = envToWorld(ray.end[0], ray.end[1]);
    const hit = new THREE.Vector3(hitWorld.x, 1.2, hitWorld.z);

    // Color coding matching concept: Red (<45 px), Amber (<90 px), Cyan (>=90 px)
    let beamColor = 0x00f0ff;
    if (ray.distance < 45.0) {
      beamColor = 0xff2a5f;
    } else if (ray.distance < 90.0) {
      beamColor = 0xffb700;
    }

    // Orient and scale 3D laser cylinder between rayOrigin and hit
    const deltaVec = new THREE.Vector3().subVectors(hit, rayOrigin);
    const beamLen = deltaVec.length();

    if (beamLen > 0.1) {
      beamObj.beam.visible = true;
      beamObj.beam.position.copy(rayOrigin).addScaledVector(deltaVec, 0.5);
      beamObj.beam.scale.set(1.0, beamLen, 1.0);
      beamObj.beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), deltaVec.clone().normalize());
      beamObj.beam.material.color.setHex(beamColor);
    } else {
      beamObj.beam.visible = false;
    }

    beamObj.ring.position.set(hit.x, 0.65, hit.z);
    beamObj.ring.material.color.setHex(beamColor);

    beamObj.spark.position.set(hit.x, 1.2, hit.z);
  }
}

// ==============================================================================
// 3D Trajectory Ribbon
// ==============================================================================

function clearTrajectoryTrail() {
  STATE.trajectoryPoints = [];
  if (trajectoryMesh) {
    scene.remove(trajectoryMesh);
    if (trajectoryMesh.geometry) trajectoryMesh.geometry.dispose();
    trajectoryMesh = null;
  }
}

function updatePlaybackTrajectory(steps, baseIdx, curX, curZ) {
  const windowStart = Math.max(0, baseIdx - 400);
  const pts = [];
  for (let i = windowStart; i <= baseIdx; i++) {
    const s = steps[i];
    if (s && s.vehicle && isValidNumber(s.vehicle.x) && isValidNumber(s.vehicle.y)) {
      const w = envToWorld(s.vehicle.x, s.vehicle.y);
      pts.push(new THREE.Vector3(w.x, 0.64, w.z));
    }
  }
  pts.push(new THREE.Vector3(curX, 0.64, curZ));

  if (pts.length < 2) return;

  if (!trajectoryMesh) {
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.65,
      linewidth: 2,
    });
    trajectoryMesh = new THREE.Line(geo, mat);
    scene.add(trajectoryMesh);
  } else {
    trajectoryMesh.geometry.setFromPoints(pts);
  }
}

// ==============================================================================
// Camera Director (Chase, Cockpit, Orbit, Overhead)
// ==============================================================================

function setCameraMode(mode) {
  STATE.activeCameraMode = mode;
  document.querySelectorAll(".cam-tab").forEach((btn) => btn.classList.remove("active"));
  const activeBtn = {
    chase: document.getElementById("btnCamChase"),
    cockpit: document.getElementById("btnCamCockpit"),
    orbit: document.getElementById("btnCamOrbit"),
    overhead: document.getElementById("btnCamOverhead"),
  }[mode];
  if (activeBtn) activeBtn.classList.add("active");

  if (mode === "cockpit") {
    camera.near = 0.4;
  } else if (mode === "overhead") {
    camera.near = 20.0;
  } else {
    camera.near = 2.0;
  }
  camera.updateProjectionMatrix();
}
window.setCameraMode = setCameraMode;

function updateCameraDirector(car, velocity, delta = 0.016) {
  if (!carGroup || !camera) return;

  const carPos = carGroup.position;
  const heading = carGroup.rotation.y;
  const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
  const backward = forward.clone().negate();

  if (STATE.activeCameraMode === "closeup") {
    if (controls) controls.enabled = false;
    const right = new THREE.Vector3(-forward.z, 0, forward.x);
    camera.position.set(
      carPos.x - forward.x * 52.0 + right.x * 24.0,
      22.0,
      carPos.z - forward.z * 52.0 + right.z * 24.0
    );
    camera.lookAt(carPos.x, 8.0, carPos.z);
    return;
  }

  if (STATE.activeCameraMode === "cockpit") {
    if (controls) controls.enabled = false;
    const right = new THREE.Vector3(-forward.z, 0, forward.x);
    const eyePos = new THREE.Vector3(
      carPos.x + forward.x * 2.5 - right.x * 6.5,
      14.5,
      carPos.z + forward.z * 2.5 - right.z * 6.5
    );
    const lookTarget = new THREE.Vector3(
      carPos.x + forward.x * 120.0,
      10.0,
      carPos.z + forward.z * 120.0
    );
    camera.position.copy(eyePos);
    camera.lookAt(lookTarget);
    camera.up.set(0, 1, 0);

  } else if (STATE.activeCameraMode === "overhead") {
    if (controls) controls.enabled = false;
    camera.position.set(0, 750.0, 0);
    camera.up.set(0, 0, -1);
    camera.lookAt(0, 0, 0);

  } else if (STATE.activeCameraMode === "orbit") {
    camera.up.set(0, 1, 0);
    if (controls) {
      controls.enabled = true;
      controls.target.set(carPos.x, 6.0, carPos.z);
      controls.update();
    }

  } else {
    // Chase Camera (Framing matched to concept video: car centered in lower-third, track ahead visible)
    if (controls) controls.enabled = false;
    const followDist = 120.0;
    const followHeight = 40.0 + Math.min(Math.abs(velocity) * 1.2, 6.0);
    const targetCamPos = new THREE.Vector3(
      carPos.x + backward.x * followDist,
      followHeight,
      carPos.z + backward.z * followDist
    );
    const lookTarget = new THREE.Vector3(
      carPos.x + forward.x * 48.0,
      7.0,
      carPos.z + forward.z * 48.0
    );

    const safeDelta = Math.min(Math.max(delta || 0.016, 0.001), 0.1);
    const camLerp = 1.0 - Math.exp(-8.0 * safeDelta);
    const targetLerp = 1.0 - Math.exp(-9.5 * safeDelta);

    STATE.cameraPos.lerp(targetCamPos, camLerp);
    STATE.cameraTarget.lerp(lookTarget, targetLerp);

    camera.position.copy(STATE.cameraPos);
    camera.lookAt(STATE.cameraTarget);
    camera.up.set(0, 1, 0);
  }
}

// ==============================================================================
// Animation Loop & Render Pipeline (Non-Blocking Startup)
// ==============================================================================

let lastFrameTime = performance.now();
let frameCount = 0;
let lastFpsUpdate = performance.now();
let currentFPS = 60;
let isAnimationLoopRunning = false;

function startAnimationLoop() {
  if (isAnimationLoopRunning) return;
  isAnimationLoopRunning = true;
  lastFrameTime = performance.now();
  requestAnimationFrame(animate);
}

function animate(currentTime) {
  requestAnimationFrame(animate);

  const delta = Math.min(Math.max((currentTime - lastFrameTime) * 0.001, 0.0001), 0.1);
  lastFrameTime = currentTime;

  frameCount++;
  if (currentTime - lastFpsUpdate >= 500) {
    currentFPS = Math.round((frameCount * 1000) / (currentTime - lastFpsUpdate));
    frameCount = 0;
    lastFpsUpdate = currentTime;
    window.currentFPS = currentFPS;
    const fpsEl = document.getElementById("fpsCounter");
    if (fpsEl) fpsEl.innerText = `${currentFPS} FPS`;
  }

  // Playback Clock Advance: decoupled from network speed
  if (STATE.playback.isPlaying && STATE.playback.rolloutData && STATE.playback.rolloutData.steps) {
    const stepDelta = (delta / STATE.playback.nominalDt) * STATE.playback.speed;
    STATE.playback.playhead += stepDelta;
    if (STATE.playback.playhead >= STATE.playback.maxStep) {
      STATE.playback.playhead = STATE.playback.maxStep;
      STATE.playback.isPlaying = false;
      STATE.simRunning = false;
      const startIcon = document.getElementById("startIcon");
      if (startIcon) startIcon.innerText = "↺";
      updateStatusPill("COMPLETED", "green");
      showPostLapModal();
    }
  }

  // Deterministic Rollout Playback Rendering
  if (STATE.playback.rolloutData && STATE.playback.rolloutData.steps && STATE.playback.rolloutData.steps.length > 0) {
    renderPlaybackFrame(STATE.playback.playhead, delta);
  }

  // Center shadow camera on car
  if (mainSun && carGroup) {
    mainSun.position.set(carGroup.position.x + 280, 440, carGroup.position.z + 180);
    mainSun.target = carGroup;
  }

  // Render 3D Scene
  if (STATE.graphicsQuality === "high" && composer) {
    composer.render();
  } else if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

// ==============================================================================
// Deterministic Rollout Playback & Pose Interpolation
// ==============================================================================

function renderPlaybackFrame(playhead, delta = 0.016) {
  const rollout = STATE.playback.rolloutData;
  if (!rollout || !rollout.steps || rollout.steps.length === 0) return;

  const steps = rollout.steps;
  const clampedHead = Math.min(Math.max(playhead, 0.0), steps.length - 1);
  const baseIdx = Math.floor(clampedHead);
  const nextIdx = Math.min(baseIdx + 1, steps.length - 1);
  const alpha = clampedHead - baseIdx;

  const s0 = steps[baseIdx];
  const s1 = steps[nextIdx];
  const car0 = s0.vehicle;
  const car1 = s1.vehicle;

  // 1. Position interpolation
  const pos0 = envToWorld(car0.x, car0.y);
  const pos1 = envToWorld(car1.x, car1.y);
  const curX = THREE.MathUtils.lerp(pos0.x, pos1.x, alpha);
  const curZ = THREE.MathUtils.lerp(pos0.z, pos1.z, alpha);

  if (carGroup) {
    carGroup.position.set(curX, 0.0, curZ);
  }

  // 2. Shortest-angle heading interpolation (slerp)
  const h0 = envHeadingToWorld(car0.angle);
  const h1 = envHeadingToWorld(car1.angle);
  let diff = h1 - h0;
  while (diff > Math.PI) diff -= 2 * Math.PI;
  while (diff < -Math.PI) diff += 2 * Math.PI;
  const curHeadingRad = h0 + diff * alpha;

  if (carGroup) {
    carGroup.rotation.y = curHeadingRad;
  }

  // 3. Speed & steering interpolation
  const curVelocity = THREE.MathUtils.lerp(car0.velocity, car1.velocity, alpha);
  const curSteerDeg = THREE.MathUtils.lerp(car0.steering_angle_deg, car1.steering_angle_deg, alpha);
  const curThrottle = THREE.MathUtils.lerp(car0.throttle, car1.throttle, alpha);

  // 4. Smooth continuous wheel rotation
  const wheelRadiusPx = 5.2;
  const spinDelta = (curVelocity / wheelRadiusPx) * (delta / 0.030);
  wheelSpinAngle -= spinDelta;

  const steerRad = -envSteerToWorld(curSteerDeg);
  if (frontLeftSteerPivot) frontLeftSteerPivot.rotation.y = steerRad;
  if (frontRightSteerPivot) frontRightSteerPivot.rotation.y = steerRad;
  wheelSpinMeshes.forEach((spinGroup) => {
    spinGroup.rotation.z = wheelSpinAngle;
  });

  // 5. Dynamic brake lights
  const isDecel = curThrottle < -0.05 || car0.action_mode === "deceleration";
  rearBrakeLights.forEach((light) => {
    light.material.color.setHex(isDecel ? 0xff0020 : 0xaa0015);
    light.material.emissive.setHex(isDecel ? 0xff0020 : 0xcc0015);
    light.material.emissiveIntensity = isDecel ? 2.8 : 0.8;
  });

  // 6. Rangefinder sensor beams
  if (s0.rays) {
    updateLidarVisuals(s0.rays, { x: curX, y: curZ });
  }

  // 7. Trajectory ribbon
  updatePlaybackTrajectory(steps, baseIdx, curX, curZ);

  // 8. Camera smoothing
  if (carGroup) {
    updateCameraDirector(
      { x: carGroup.position.x, y: carGroup.position.z, angle: (carGroup.rotation.y * 180.0) / Math.PI },
      curVelocity,
      delta
    );
  }

  // 9. Update Telemetry Sparkline Charts & UI
  const curEnvX = THREE.MathUtils.lerp(car0.x, car1.x, alpha);
  const curEnvY = THREE.MathUtils.lerp(car0.y, car1.y, alpha);
  const curEnvAngle = THREE.MathUtils.lerp(car0.angle, car1.angle, alpha);

  const interpolatedVehicle = {
    x: curEnvX,
    y: curEnvY,
    angle: curEnvAngle,
    velocity: curVelocity,
    steering: THREE.MathUtils.lerp(car0.steering, car1.steering, alpha),
    steering_angle_deg: curSteerDeg,
    throttle: curThrottle,
  };

  updateTelemetryCharts(interpolatedVehicle, s0.telemetry || {}, baseIdx);

  // Discrete step updates
  if (baseIdx !== STATE.playback.lastStepIndex) {
    STATE.playback.lastStepIndex = baseIdx;

    if (rollout.commentary) {
      syncPlaybackCommentary(rollout.commentary, baseIdx);
    }

    if (STATE.trackData && minimapCtx) {
      renderRadarMinimap(STATE.trackData, interpolatedVehicle);
    }
  }
}

// ==============================================================================
// Live Telemetry Sparkline Charts (Speed, Steering, Throttle/Brake, Lateral/Slip)
// ==============================================================================

function updateTelemetryCharts(car, telemetry, stepIdx) {
  const v = Number(car.velocity) || 0.0;
  const s = Number(car.steering_angle_deg) || 0.0;
  const t = Number(car.throttle) || 0.0;
  const lat = Number(telemetry.signed_distance) || 0.0;
  const slip = Number(telemetry.slip_angle) || 0.0;

  TELEM_HISTORY.speed.push(v);
  TELEM_HISTORY.steer.push(s);
  TELEM_HISTORY.throttle.push(t);
  TELEM_HISTORY.lateral.push(lat);
  TELEM_HISTORY.slip.push(slip);

  if (TELEM_HISTORY.speed.length > TELEM_HISTORY.maxPoints) {
    TELEM_HISTORY.speed.shift();
    TELEM_HISTORY.steer.shift();
    TELEM_HISTORY.throttle.shift();
    TELEM_HISTORY.lateral.shift();
    TELEM_HISTORY.slip.shift();
  }

  drawSpeedChart();
  drawSteerChart();
  drawThrottleChart();
  drawLateralSlipChart();

  // Readouts - Concept IDs & Legacy IDs
  const velEl = document.getElementById("valVelocity");
  if (velEl) velEl.innerHTML = `${v.toFixed(2)} <span class="unit">px/step</span>`;

  const valSpeed = document.getElementById("valSpeed");
  if (valSpeed) valSpeed.innerText = v.toFixed(2);

  const steerEl = document.getElementById("valSteerAngle");
  if (steerEl) steerEl.innerText = `${s >= 0 ? "+" : ""}${s.toFixed(1)}°`;

  const valSteering = document.getElementById("valSteering");
  if (valSteering) valSteering.innerText = `${s >= 0 ? "+" : ""}${s.toFixed(1)}°`;

  const throtEl = document.getElementById("valThrottle");
  if (throtEl) {
    throtEl.innerText = `${t >= 0 ? "+" : ""}${t.toFixed(2)}`;
    throtEl.className = `chart-readout ${t >= 0 ? "green" : "red"}`;
  }

  const latSlipEl = document.getElementById("valLateralSlip");
  if (latSlipEl) latSlipEl.innerText = `${lat.toFixed(1)} px • ${slip.toFixed(1)}°`;

  // Rangefinder boxes
  const rays = telemetry.rays || [];
  if (rays.length >= 5) {
    for (let i = 0; i < 5; i++) {
      const dist = Math.round(rays[i]);
      const valEl = document.getElementById(`rfVal${i}`);
      const boxEl = document.getElementById(`rfBox${i}`);
      if (valEl) valEl.innerText = dist;
      if (boxEl) {
        boxEl.classList.remove("rf-red", "rf-amber", "rf-cyan");
        if (dist < 45) boxEl.classList.add("rf-red");
        else if (dist < 90) boxEl.classList.add("rf-amber");
        else boxEl.classList.add("rf-cyan");
      }
    }
  }

  // Top step clock (Contains 'STEP:' for test assertion)
  const stepClock = document.getElementById("stepClock");
  if (stepClock) stepClock.innerText = `STEP: ${stepIdx}`;

  // Scrubber & Time
  const scrubber = document.getElementById("pbScrubber");
  if (scrubber) scrubber.value = stepIdx;
  const curTime = document.getElementById("pbCurrentTime");
  if (curTime) {
    const sec = Math.floor(stepIdx * 0.030);
    const mm = String(Math.floor(sec / 60)).padStart(2, "0");
    const ss = String(sec % 60).padStart(2, "0");
    curTime.innerText = `${mm}:${ss}`;
  }

  // Cinematic Floating Bottom HUD
  updateCinematicBottomHud(car);
}

function drawSpeedChart() {
  const canvas = document.getElementById("chartSpeed");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const data = TELEM_HISTORY.speed;
  if (data.length < 2) return;

  const maxVal = 4.0;
  const stepX = w / (TELEM_HISTORY.maxPoints - 1);
  const startX = (TELEM_HISTORY.maxPoints - data.length) * stepX;

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(0, 240, 255, 0.28)");
  grad.addColorStop(1, "rgba(0, 240, 255, 0.0)");

  ctx.beginPath();
  data.forEach((val, i) => {
    const x = startX + i * stepX;
    const y = h - (Math.min(maxVal, Math.max(0, val)) / maxVal) * (h - 8) - 4;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 2.0;
  ctx.stroke();

  ctx.lineTo(startX + (data.length - 1) * stepX, h);
  ctx.lineTo(startX, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();
}

function drawSteerChart() {
  const canvas = document.getElementById("chartSteer");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const data = TELEM_HISTORY.steer;
  if (data.length < 2) return;

  const centerY = h / 2;
  const stepX = w / (TELEM_HISTORY.maxPoints - 1);
  const startX = (TELEM_HISTORY.maxPoints - data.length) * stepX;

  // Zero baseline
  ctx.beginPath();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.lineWidth = 1;
  ctx.moveTo(0, centerY);
  ctx.lineTo(w, centerY);
  ctx.stroke();

  ctx.beginPath();
  data.forEach((val, i) => {
    const x = startX + i * stepX;
    const y = centerY - (val / 30.0) * (centerY - 6);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.strokeStyle = "#ffb700";
  ctx.lineWidth = 2.0;
  ctx.stroke();
}

function drawThrottleChart() {
  const canvas = document.getElementById("chartThrottle");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const data = TELEM_HISTORY.throttle;
  if (data.length < 2) return;

  const centerY = h / 2;
  const stepX = w / (TELEM_HISTORY.maxPoints - 1);
  const startX = (TELEM_HISTORY.maxPoints - data.length) * stepX;

  // Red zero center line
  ctx.beginPath();
  ctx.strokeStyle = "rgba(255, 42, 95, 0.4)";
  ctx.lineWidth = 1;
  ctx.moveTo(0, centerY);
  ctx.lineTo(w, centerY);
  ctx.stroke();

  // Stepped line
  ctx.beginPath();
  data.forEach((val, i) => {
    const x = startX + i * stepX;
    const y = centerY - val * (centerY - 6);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, centerY - data[i - 1] * (centerY - 6));
      ctx.lineTo(x, y);
    }
  });

  ctx.strokeStyle = "#00e676";
  ctx.lineWidth = 2.0;
  ctx.stroke();
}

function drawLateralSlipChart() {
  const canvas = document.getElementById("chartLateralSlip");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const dataLat = TELEM_HISTORY.lateral;
  const dataSlip = TELEM_HISTORY.slip;
  if (dataLat.length < 2) return;

  const centerY = h / 2;
  const stepX = w / (TELEM_HISTORY.maxPoints - 1);
  const startX = (TELEM_HISTORY.maxPoints - dataLat.length) * stepX;

  // White line: Lateral deviation
  ctx.beginPath();
  dataLat.forEach((val, i) => {
    const x = startX + i * stepX;
    const y = centerY - (val / 35.0) * (centerY - 6);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.8;
  ctx.stroke();

  // Magenta line: Slip angle
  ctx.beginPath();
  dataSlip.forEach((val, i) => {
    const x = startX + i * stepX;
    const y = centerY - (val / 15.0) * (centerY - 6);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#b388ff";
  ctx.lineWidth = 1.8;
  ctx.stroke();
}

function syncPlaybackCommentary(allCommentary, currentStep) {
  const feed = document.getElementById("commentaryFeed");
  if (!feed || !Array.isArray(allCommentary)) return;

  const visibleEvents = allCommentary.filter((item) => item.step <= currentStep);
  feed.innerHTML = "";
  for (const item of visibleEvents.slice(-40)) {
    const div = document.createElement("div");
    div.className = "eng-line";
    const text = item.commentary || item.text || "";
    div.innerHTML = `<span class="eng-step">[STEP ${item.step}]</span> ${text}`;
    feed.appendChild(div);
  }
  const curDiv = document.createElement("div");
  curDiv.className = "eng-line cursor-line";
  curDiv.innerHTML = `<span class="eng-step">[STEP: ${currentStep}]</span> <span class="blinking-cursor">|</span>`;
  feed.appendChild(curDiv);
  feed.scrollTop = feed.scrollHeight;
}

function resetCommentaryFeed() {
  const feed = document.getElementById("commentaryFeed");
  if (feed) feed.innerHTML = "";
}

// ==============================================================================
// 2D Circuit Radar Minimap
// ==============================================================================

function renderRadarMinimap(trackData, car) {
  if (!minimapCtx || !minimapCanvas || !trackData) return;
  const ctx = minimapCtx;
  const w = minimapCanvas.width;
  const h = minimapCanvas.height;

  ctx.clearRect(0, 0, w, h);

  const bounds = trackData.bounds || { min_x: 100, max_x: 700, min_y: 100, max_y: 500 };
  const scale = Math.min((w - 24) / (bounds.max_x - bounds.min_x), (h - 24) / (bounds.max_y - bounds.min_y));

  function toScreen(x, y) {
    return {
      x: 12 + (x - bounds.min_x) * scale,
      y: 12 + (y - bounds.min_y) * scale,
    };
  }

  // Draw Centerline
  const centerPts = trackData.centerline || [];
  if (centerPts.length > 2) {
    ctx.beginPath();
    centerPts.forEach((pt, i) => {
      const p = toScreen(pt[0], pt[1]);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.closePath();
    ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw Car Marker
  if (car && isValidNumber(car.x) && isValidNumber(car.y)) {
    const cp = toScreen(car.x, car.y);
    ctx.beginPath();
    ctx.arc(cp.x, cp.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#ff2a5f";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

// ==============================================================================
// Non-Blocking Viewport Loading Overlay Helpers
// ==============================================================================

function showLoadingOverlay(title = "GENERATING DETERMINISTIC EPISODE", subtitle = "Simulating rollout...") {
  const overlay = document.getElementById("simulationLoadingOverlay");
  const tEl = document.getElementById("loadingTitle");
  const sEl = document.getElementById("loadingSubtitle");
  const errBox = document.getElementById("loadingErrorBox");

  if (tEl) tEl.innerText = title;
  if (sEl) sEl.innerText = subtitle;
  if (errBox) errBox.style.display = "none";

  if (overlay) {
    overlay.style.display = "flex";
    overlay.style.opacity = "1";
    overlay.style.pointerEvents = "auto";
  }
  setPlaybackControlsEnabled(false);
}

function hideLoadingOverlay() {
  const overlay = document.getElementById("simulationLoadingOverlay");
  if (overlay) {
    overlay.style.opacity = "0";
    overlay.style.pointerEvents = "none";
    setTimeout(() => {
      if (overlay.style.opacity === "0") {
        overlay.style.display = "none";
      }
    }, 350);
  }
  setPlaybackControlsEnabled(true);
}

function showLoadingError(errMsg = "Rollout generation failed.") {
  const overlay = document.getElementById("simulationLoadingOverlay");
  const tEl = document.getElementById("loadingTitle");
  const sEl = document.getElementById("loadingSubtitle");
  const errBox = document.getElementById("loadingErrorBox");
  const errText = document.getElementById("loadingErrorText");

  if (tEl) tEl.innerText = "SIMULATION FAILED";
  if (sEl) sEl.innerText = "An error occurred while generating the deterministic episode.";
  if (errText) errText.innerText = errMsg;
  if (errBox) errBox.style.display = "block";

  if (overlay) {
    overlay.style.display = "flex";
    overlay.style.opacity = "1";
    overlay.style.pointerEvents = "auto";
  }
  setPlaybackControlsEnabled(false);
}

function setPlaybackControlsEnabled(enabled) {
  const controlsToToggle = [
    "btnStartPause",
    "btnStep",
    "btnStepBack",
    "btnReset",
    "pbScrubber",
  ];
  controlsToToggle.forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.disabled = !enabled;
      el.style.opacity = enabled ? "1.0" : "0.35";
      el.style.pointerEvents = enabled ? "auto" : "none";
    }
  });
}

// ==============================================================================
// Simulation Backend Client
// ==============================================================================

async function startSimulation(presetId = STATE.presetId, trackSeed = STATE.trackSeed) {
  try {
    updateStatusPill("GENERATING EPISODE...", "amber");
    showLoadingOverlay("GENERATING DETERMINISTIC EPISODE", `Simulating ${presetId} (seed ${trackSeed})...`);

    STATE.playback.isLoading = true;
    STATE.presetId = presetId;
    STATE.trackSeed = trackSeed;

    const res = await fetch("/simulation/rollout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset_id: presetId,
        track_seed: trackSeed,
        commentary_mode: STATE.commentaryMode,
        saliency_interval: 5,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (!data.steps || data.steps.length === 0) {
      throw new Error("Empty rollout returned from server.");
    }

    STATE.playback.rolloutData = data;
    STATE.playback.maxStep = data.steps.length - 1;
    STATE.playback.playhead = 0.0;
    STATE.playback.lastStepIndex = -1;
    STATE.playback.isPlaying = false;
    STATE.simRunning = false;
    STATE.playback.isLoading = false;

    clearTrajectoryTrail();
    resetCommentaryFeed();
    build3DTrack(data.track);

    // Initial frame
    renderPlaybackFrame(0.0, 0.016);

    // Snap chase camera to starting position
    if (STATE.activeCameraMode === "chase" && carGroup) {
      const heading = carGroup.rotation.y;
      const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
      const backward = forward.clone().negate();
      const followDist = 120.0;
      const followHeight = 40.0;
      STATE.cameraPos.set(
        carGroup.position.x + backward.x * followDist,
        followHeight,
        carGroup.position.z + backward.z * followDist
      );
      STATE.cameraTarget.set(
        carGroup.position.x + forward.x * 48.0,
        7.0,
        carGroup.position.z + forward.z * 48.0
      );
      camera.position.copy(STATE.cameraPos);
      camera.lookAt(STATE.cameraTarget);
      camera.up.set(0, 1, 0);
    }

    const startIcon = document.getElementById("startIcon");
    if (startIcon) startIcon.innerText = "▶";

    const totalTimeEl = document.getElementById("pbTotalTime");
    if (totalTimeEl) {
      const totalSec = Math.floor(STATE.playback.maxStep * 0.030);
      const mm = String(Math.floor(totalSec / 60)).padStart(2, "0");
      const ss = String(totalSec % 60).padStart(2, "0");
      totalTimeEl.innerText = `${mm}:${ss}`;
    }

    const scrubber = document.getElementById("pbScrubber");
    if (scrubber) {
      scrubber.max = STATE.playback.maxStep;
      scrubber.value = 0;
    }

    hideLoadingOverlay();
    updateStatusPill("READY (ROLLOUT LOADED)", "cyan");
  } catch (err) {
    console.error("Simulation start / rollout loading failed:", err);
    STATE.playback.isLoading = false;
    showLoadingError(err.message || "Failed to load simulation rollout.");
    updateStatusPill("ERROR", "red");
  }
}

function pauseSimulation() {
  STATE.playback.isPlaying = false;
  STATE.simRunning = false;
  const startIcon = document.getElementById("startIcon");
  if (startIcon) startIcon.innerText = "▶";
  updateStatusPill("PAUSED", "amber");
}

function stepSimulation() {
  if (!STATE.playback.rolloutData || !STATE.playback.rolloutData.steps) return;
  pauseSimulation();
  const nextStep = Math.min(STATE.playback.maxStep, Math.floor(STATE.playback.playhead) + 1);
  STATE.playback.playhead = nextStep;
  renderPlaybackFrame(STATE.playback.playhead, 0.016);
}

function resetSimulation() {
  STATE.playback.isPlaying = false;
  STATE.simRunning = false;
  STATE.playback.playhead = 0.0;
  clearTrajectoryTrail();
  renderPlaybackFrame(0.0, 0.016);
  const startIcon = document.getElementById("startIcon");
  if (startIcon) startIcon.innerText = "▶";
  updateStatusPill("READY (ROLLOUT LOADED)", "cyan");
}

function updateStatusPill(text, color = "green") {
  const dot = document.getElementById("statusDot");
  const txt = document.getElementById("statusText");
  if (dot) dot.className = `status-indicator-dot ${color}`;
  if (txt) txt.innerText = text;
}

// ==============================================================================
// Graphics Quality Switcher & Cinematic Mode
// ==============================================================================

function setGraphicsQuality(quality) {
  STATE.graphicsQuality = quality;
  const btn = document.getElementById("btnQualityToggle");
  if (btn) {
    btn.innerText = `QUALITY: ${quality.toUpperCase()}`;
  }

  if (renderer) {
    renderer.setPixelRatio(quality === "high" ? Math.min(window.devicePixelRatio || 1, 2) : 1);
    renderer.shadowMap.enabled = quality === "high";
  }
}
window.setGraphicsQuality = setGraphicsQuality;

function toggleCinematicUIMode() {
  STATE.cinematicMode = !STATE.cinematicMode;
  document.body.classList.toggle("cinematic-hud-mode", STATE.cinematicMode);

  const cinematicHud = document.getElementById("cinematicBottomHud");
  if (cinematicHud) {
    cinematicHud.style.display = STATE.cinematicMode ? "block" : "none";
  }

  const btn = document.getElementById("btnCinematicToggle");
  if (btn) {
    btn.classList.toggle("active", STATE.cinematicMode);
  }
}
window.toggleCinematicUIMode = toggleCinematicUIMode;

function updateCinematicBottomHud(car) {
  const v = Number(car.velocity) || 0.0;
  const s = Number(car.steering_angle_deg) || 0.0;
  const t = Number(car.throttle) || 0.0;

  const spdEl = document.getElementById("hudSpeedVal");
  const strEl = document.getElementById("hudSteerVal");
  const thrEl = document.getElementById("hudThrottleVal");

  if (spdEl) spdEl.innerText = v.toFixed(1);
  if (strEl) strEl.innerText = `${s >= 0 ? "+" : ""}${s.toFixed(1)}°`;
  if (thrEl) thrEl.innerText = `${t >= 0 ? "+" : ""}${t.toFixed(2)}`;
}

// ==============================================================================
// Post-Lap Analysis Modal (Matching reference_40s.png)
// ==============================================================================

function showPostLapModal() {
  const overlay = document.getElementById("canvasOverlay");
  if (!overlay) return;
  overlay.style.display = "flex";

  const rollout = STATE.playback.rolloutData;
  if (!rollout) return;

  const lapTimeEl = document.getElementById("kpiLapTime");
  if (lapTimeEl) lapTimeEl.innerHTML = `${STATE.playback.maxStep} <span class="kpi-unit">steps</span>`;

  drawLapCircuitSpeedMap();
  drawLapTraces();
}

function hidePostLapModal() {
  const overlay = document.getElementById("canvasOverlay");
  if (overlay) overlay.style.display = "none";
}

function drawLapCircuitSpeedMap() {
  const canvas = document.getElementById("lapCircuitCanvas");
  if (!canvas || !STATE.trackData) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const centerPts = STATE.trackData.centerline || [];
  if (centerPts.length < 4) return;

  const bounds = STATE.trackData.bounds || { min_x: 100, max_x: 700, min_y: 100, max_y: 500 };
  const scale = Math.min((w - 40) / (bounds.max_x - bounds.min_x), (h - 40) / (bounds.max_y - bounds.min_y));

  function toScreen(x, y) {
    return {
      x: 20 + (x - bounds.min_x) * scale,
      y: 20 + (y - bounds.min_y) * scale,
    };
  }

  // Draw colorful circuit with speed gradient
  for (let i = 0; i < centerPts.length - 1; i++) {
    const p1 = toScreen(centerPts[i][0], centerPts[i][1]);
    const p2 = toScreen(centerPts[i + 1][0], centerPts[i + 1][1]);

    const frac = i / centerPts.length;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.strokeStyle = frac > 0.4 && frac < 0.6 ? "#ff2a5f" : "#00f0ff";
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.stroke();
  }
}

function drawLapTraces() {
  const rollout = STATE.playback.rolloutData;
  if (!rollout || !rollout.steps) return;
  const steps = rollout.steps;

  // Speed Trace
  const speedCanvas = document.getElementById("lapSpeedTraceCanvas");
  if (speedCanvas) {
    const ctx = speedCanvas.getContext("2d");
    const w = speedCanvas.width;
    const h = speedCanvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = (i / steps.length) * w;
      const y = h - (s.vehicle.velocity / 4.0) * (h - 10) - 5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Steering Trace
  const steerCanvas = document.getElementById("lapSteerTraceCanvas");
  if (steerCanvas) {
    const ctx = steerCanvas.getContext("2d");
    const w = steerCanvas.width;
    const h = steerCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const centerY = h / 2;
    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = (i / steps.length) * w;
      const y = centerY - (s.vehicle.steering_angle_deg / 30.0) * (centerY - 6);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#ffb700";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// ==============================================================================
// Event Handlers & Initialization
// ==============================================================================

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

  // Quality toggle
  const btnQuality = document.getElementById("btnQualityToggle");
  if (btnQuality) {
    btnQuality.addEventListener("click", () => {
      setGraphicsQuality(STATE.graphicsQuality === "high" ? "low" : "high");
    });
  }

  // Cinematic toggle
  const btnCinematic = document.getElementById("btnCinematicToggle");
  if (btnCinematic) {
    btnCinematic.addEventListener("click", toggleCinematicUIMode);
  }

  // Play / Pause
  const btnStartPause = document.getElementById("btnStartPause");
  if (btnStartPause) {
    btnStartPause.addEventListener("click", () => {
      if (STATE.playback.isPlaying) {
        pauseSimulation();
      } else {
        if (STATE.playback.playhead >= STATE.playback.maxStep) {
          STATE.playback.playhead = 0.0;
          clearTrajectoryTrail();
        }
        STATE.playback.isPlaying = true;
        STATE.simRunning = true;
        const icon = document.getElementById("startIcon");
        if (icon) icon.innerText = "⏸";
        updateStatusPill("RUNNING", "green");
        hidePostLapModal();
      }
    });
  }

  // Step button
  const btnStep = document.getElementById("btnStep");
  if (btnStep) {
    btnStep.addEventListener("click", () => {
      stepSimulation();
    });
  }

  // Step Back button
  const btnStepBack = document.getElementById("btnStepBack");
  if (btnStepBack) {
    btnStepBack.addEventListener("click", () => {
      pauseSimulation();
      STATE.playback.playhead = Math.max(0, Math.floor(STATE.playback.playhead) - 1);
      renderPlaybackFrame(STATE.playback.playhead, 0.016);
    });
  }

  // Reset button
  const btnReset = document.getElementById("btnReset");
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      resetSimulation();
    });
  }

  // Scrubber input
  const scrubber = document.getElementById("pbScrubber");
  if (scrubber) {
    scrubber.addEventListener("input", (e) => {
      pauseSimulation();
      STATE.playback.playhead = parseFloat(e.target.value);
      renderPlaybackFrame(STATE.playback.playhead, 0.016);
    });
  }

  // Speed pills
  document.querySelectorAll(".speed-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".speed-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      const spd = parseFloat(pill.getAttribute("data-speed")) || 1.0;
      STATE.playback.speed = spd;
      const speedVal = document.getElementById("speedVal");
      if (speedVal) speedVal.innerText = `${spd}x`;
    });
  });

  // Settings popover toggle
  const btnSettings = document.getElementById("btnSettingsToggle");
  const settingsDrawer = document.getElementById("settingsDrawer");
  const btnCloseSettings = document.getElementById("btnCloseSettings");

  if (btnSettings && settingsDrawer) {
    btnSettings.addEventListener("click", () => {
      const isHidden = settingsDrawer.style.display === "none";
      settingsDrawer.style.display = isHidden ? "block" : "none";
    });
  }
  if (btnCloseSettings && settingsDrawer) {
    btnCloseSettings.addEventListener("click", () => {
      settingsDrawer.style.display = "none";
    });
  }

  // Track selector
  const selTrack = document.getElementById("trackPreset");
  const seedInput = document.getElementById("trackSeed");
  const btnLoadTrack = document.getElementById("btnLoadTrack");

  function triggerLoadTrack() {
    pauseSimulation();
    const preset = selTrack ? selTrack.value : "standard_medium";
    const seed = seedInput ? parseInt(seedInput.value, 10) || 1001 : 1001;
    startSimulation(preset, seed);
  }

  if (btnLoadTrack) btnLoadTrack.addEventListener("click", triggerLoadTrack);
  if (selTrack) selTrack.addEventListener("change", triggerLoadTrack);

  // Post-lap modal close / restart
  const btnCloseModal = document.getElementById("btnOverlayClose");
  const btnRestartModal = document.getElementById("btnOverlayRestart");
  if (btnCloseModal) btnCloseModal.addEventListener("click", hidePostLapModal);
  if (btnRestartModal) {
    btnRestartModal.addEventListener("click", () => {
      hidePostLapModal();
      resetSimulation();
      const btnStart = document.getElementById("btnStartPause");
      if (btnStart) btnStart.click();
    });
  }

  // Hook up Retry button if present
  const btnRetry = document.getElementById("btnRetryRollout");
  if (btnRetry) {
    btnRetry.addEventListener("click", () => {
      startSimulation(STATE.presetId, STATE.trackSeed);
    });
  }

  // Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;

    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      const btnStart = document.getElementById("btnStartPause");
      if (btnStart) btnStart.click();
    } else if (e.key === "ArrowRight") {
      stepSimulation();
    } else if (e.key === "ArrowLeft") {
      const btnBack = document.getElementById("btnStepBack");
      if (btnBack) btnBack.click();
    } else if (e.key === "r" || e.key === "R") {
      resetSimulation();
    } else if (e.key === "s" || e.key === "S") {
      if (settingsDrawer) {
        settingsDrawer.style.display = settingsDrawer.style.display === "none" ? "block" : "none";
      }
    } else if (e.key === "h" || e.key === "H") {
      toggleCinematicUIMode();
    } else if (e.key === "q" || e.key === "Q") {
      setGraphicsQuality(STATE.graphicsQuality === "high" ? "low" : "high");
    } else if (e.key === "1") {
      setCameraMode("chase");
    } else if (e.key === "2") {
      setCameraMode("cockpit");
    } else if (e.key === "3") {
      setCameraMode("orbit");
    } else if (e.key === "4") {
      setCameraMode("overhead");
    }
  });
}

function buildDebugHelpers() {
  trackAxesHelper = new THREE.AxesHelper(80);
  trackAxesHelper.position.set(0, 0.5, 0);
  trackAxesHelper.visible = false;
  debugGroup.add(trackAxesHelper);

  carAxesHelper = new THREE.AxesHelper(30);
  carAxesHelper.position.set(0, 3.5, 0);
  carAxesHelper.visible = false;
  carGroup.add(carAxesHelper);
}

// Render Stats Hook for Playwright E2E and Test Suites
window.getRenderStats = function() {
  return {
    fps: currentFPS,
    drawCalls: renderer ? renderer.info.render.calls : 0,
    triangles: renderer ? renderer.info.render.triangles : 0,
    textures: renderer ? renderer.info.memory.textures : 0,
    geometries: renderer ? renderer.info.memory.geometries : 0,
    quality: STATE.graphicsQuality,
    cinematic: STATE.cinematicMode,
  };
};

window.getRenderDebugInfo = window.getRenderStats;

// Initialize Application
window.addEventListener("DOMContentLoaded", () => {
  initThreeJS();
  initEventHandlers();

  // Start animation loop immediately (Non-blocking startup: 3D scene renders within milliseconds)
  startAnimationLoop();

  // Initiate simulation rollout asynchronously in background without blocking render loop
  startSimulation(STATE.presetId, STATE.trackSeed);
});
