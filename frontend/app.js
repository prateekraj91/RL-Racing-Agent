/**
 * RL Racing Agent // Cinematic 3D Autonomous Simulation Dashboard
 *
 * Phase 3 Checkpoint A:
 * - HDRI Daytime Sky & Lighting (Poly Haven 2K CC0)
 * - PBR Asphalt & Ground Runoff Materials (AmbientCG / Poly Haven CC0)
 * - Three.js Post-Processing Pipeline (EffectComposer: RenderPass -> SSAOPass -> UnrealBloomPass -> SMAAPass)
 * - ACESFilmicToneMapping & sRGB Output Encoding
 * - Sun Directional Light with Tight Shadow Camera Tracking
 * - Quality Toggle (High / Low)
 * - Physics Footprint (30 x 50 px) Wireframe in Debug Mode (D)
 * - Renderer.info Reporting (FPS, Draw Calls, Triangles, Textures, Geometries)
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

  // 60 FPS Pose Interpolation state
  interp: {
    active: false,
    prevPos: { x: 0, z: 0 },
    targetPos: { x: 0, z: 0 },
    prevHeading: 0.0,
    targetHeading: 0.0,
    prevSteer: 0.0,
    targetSteer: 0.0,
    prevVelocity: 0.0,
    targetVelocity: 0.0,
    startTime: 0,
    duration: 35.0,
  },
  gltfLoaded: false,
  gltfLoading: false,

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

window.STATE = STATE;


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
// Three.js 3D Scene Architecture & Rendering Pipeline
// ==============================================================================

let scene, camera, renderer, controls;
let composer, renderPass, ssaoPass, bloomPass, smaaPass;
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
let trackGroup, carGroup, carBodyGroup, trajectoryMesh, lidarGroup, debugGroup;
let frontLeftSteerPivot, frontRightSteerPivot;
let wheelMeshes = [];
let rearBrakeLights = [];
let gltfCarContainer = null;
let gltfRig = null;
let wheelSpinAngle = 0.0;
let rayBeamObjects = [];
let roadMesh, curbInnerMesh, curbOuterMesh, barrierMesh, groundPlaneMesh;
let propsGroup = null;
let carBboxHelper, carPhysicsFootprint, trackAxesHelper, carAxesHelper;

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

  // Distance fog gently blending the horizon into the HDRI sky without veiling the track
  scene.fog = new THREE.Fog(0x9bc2e0, 1200.0, 3800.0);

  // 2. Camera Setup (natural 48 deg FOV for reduced distortion and sharp framing)
  camera = new THREE.PerspectiveCamera(48, width / height, 2.0, 5000);
  camera.position.set(0, 300, 450);

  // 3. WebGL Renderer with High-Fidelity Color Pipeline
  renderer = new THREE.WebGLRenderer({
    canvas: threeCanvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  // PBR Tone Mapping & Color Management (punchy contrast, no washed-out highlights)
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.physicallyCorrectLights = true;

  // High-Resolution PCF Soft Shadows
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

  propsGroup = new THREE.Group();
  scene.add(propsGroup);

  carGroup = new THREE.Group();
  scene.add(carGroup);

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

  // 8. Vehicle, Sensors & Debug Helpers (Keep box car for Checkpoint A)
  buildCarModel();
  buildLidarBeams();
  buildDebugHelpers();

  // 9. Container Resize Observer
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
  if (!width || !height) {
    const container = document.getElementById("canvasContainer");
    if (container) {
      width = container.clientWidth;
      height = container.clientHeight;
    }
  }
  if (!renderer || !camera || width <= 0 || height <= 0) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);

  if (composer) {
    composer.setSize(width, height);
  }
}

function setupLighting() {
  // Balanced cool sky ambient fill (clean daylight contrast, not milky or blown out)
  const ambient = new THREE.AmbientLight(0xdbe7f4, 0.22);
  scene.add(ambient);

  // Directional sun matching HDRI sun angle with crisp contact shadows
  mainSun = new THREE.DirectionalLight(0xfff6ea, 1.55);
  mainSun.position.set(240, 420, 180);
  mainSun.castShadow = true;
  mainSun.shadow.mapSize.width = 2048;
  mainSun.shadow.mapSize.height = 2048;
  mainSun.shadow.camera.near = 50;
  mainSun.shadow.camera.far = 1100;

  // Tight shadow frustum tracking around the car
  mainSun.shadow.camera.left = -85;
  mainSun.shadow.camera.right = 85;
  mainSun.shadow.camera.top = 85;
  mainSun.shadow.camera.bottom = -85;
  mainSun.shadow.bias = -0.00015;
  mainSun.shadow.normalBias = 0.02;
  scene.add(mainSun);
}

function loadHDRIEnvironment() {
  if (!THREE.RGBELoader) {
    console.warn("RGBELoader not available, using fallback clear sky.");
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
      console.log("HDRI daytime sky loaded successfully.");
    },
    undefined,
    (err) => {
      console.warn("HDRI load failed, using sky atmospheric fallback:", err);
      scene.background = new THREE.Color(0x8ecae6);
    }
  );
}

function loadPBRTextures() {
  const texLoader = new THREE.TextureLoader();
  const res = STATE.graphicsQuality === "high" ? "2k" : "1k";
  const maxAniso = renderer && renderer.capabilities && renderer.capabilities.getMaxAnisotropy
    ? renderer.capabilities.getMaxAnisotropy()
    : 8;

  // Asphalt
  pbrTextures.asphaltDiff = texLoader.load(`/static/assets/textures/asphalt/asphalt_diff_${res}.jpg`);
  pbrTextures.asphaltDiff.wrapS = THREE.RepeatWrapping;
  pbrTextures.asphaltDiff.wrapT = THREE.RepeatWrapping;
  pbrTextures.asphaltDiff.encoding = THREE.sRGBEncoding;
  pbrTextures.asphaltDiff.anisotropy = maxAniso;
  pbrTextures.asphaltDiff.generateMipmaps = true;
  pbrTextures.asphaltDiff.minFilter = THREE.LinearMipmapLinearFilter;
  pbrTextures.asphaltDiff.magFilter = THREE.LinearFilter;

  pbrTextures.asphaltNor = texLoader.load(`/static/assets/textures/asphalt/asphalt_nor_${res}.jpg`);
  pbrTextures.asphaltNor.wrapS = THREE.RepeatWrapping;
  pbrTextures.asphaltNor.wrapT = THREE.RepeatWrapping;
  pbrTextures.asphaltNor.anisotropy = maxAniso;
  pbrTextures.asphaltNor.generateMipmaps = true;
  pbrTextures.asphaltNor.minFilter = THREE.LinearMipmapLinearFilter;
  pbrTextures.asphaltNor.magFilter = THREE.LinearFilter;

  pbrTextures.asphaltRough = texLoader.load(`/static/assets/textures/asphalt/asphalt_rough_${res}.jpg`);
  pbrTextures.asphaltRough.wrapS = THREE.RepeatWrapping;
  pbrTextures.asphaltRough.wrapT = THREE.RepeatWrapping;
  pbrTextures.asphaltRough.anisotropy = maxAniso;
  pbrTextures.asphaltRough.generateMipmaps = true;
  pbrTextures.asphaltRough.minFilter = THREE.LinearMipmapLinearFilter;
  pbrTextures.asphaltRough.magFilter = THREE.LinearFilter;

  // Ground Runoff (Rich natural grass scaling)
  pbrTextures.groundDiff = texLoader.load("/static/assets/textures/ground/ground_diff_1k.jpg");
  pbrTextures.groundDiff.wrapS = THREE.RepeatWrapping;
  pbrTextures.groundDiff.wrapT = THREE.RepeatWrapping;
  pbrTextures.groundDiff.repeat.set(50, 50);
  pbrTextures.groundDiff.encoding = THREE.sRGBEncoding;
  pbrTextures.groundDiff.anisotropy = maxAniso;
  pbrTextures.groundDiff.generateMipmaps = true;
  pbrTextures.groundDiff.minFilter = THREE.LinearMipmapLinearFilter;
  pbrTextures.groundDiff.magFilter = THREE.LinearFilter;

  pbrTextures.groundNor = texLoader.load("/static/assets/textures/ground/ground_nor_1k.jpg");
  pbrTextures.groundNor.wrapS = THREE.RepeatWrapping;
  pbrTextures.groundNor.wrapT = THREE.RepeatWrapping;
  pbrTextures.groundNor.repeat.set(50, 50);
  pbrTextures.groundNor.anisotropy = maxAniso;
  pbrTextures.groundNor.generateMipmaps = true;
  pbrTextures.groundNor.minFilter = THREE.LinearMipmapLinearFilter;
  pbrTextures.groundNor.magFilter = THREE.LinearFilter;

  pbrTextures.groundRough = texLoader.load("/static/assets/textures/ground/ground_rough_1k.jpg");
  pbrTextures.groundRough.wrapS = THREE.RepeatWrapping;
  pbrTextures.groundRough.wrapT = THREE.RepeatWrapping;
  pbrTextures.groundRough.repeat.set(50, 50);
  pbrTextures.groundRough.anisotropy = maxAniso;
  pbrTextures.groundRough.generateMipmaps = true;
  pbrTextures.groundRough.minFilter = THREE.LinearMipmapLinearFilter;
  pbrTextures.groundRough.magFilter = THREE.LinearFilter;
}

function initEffectComposer(width, height) {
  if (!THREE.EffectComposer) return;

  const pr = Math.min(window.devicePixelRatio || 1, 2);

  // Use WebGLMultisampleRenderTarget with 4x hardware MSAA at native canvas resolution
  let renderTarget;
  if (typeof THREE.WebGLMultisampleRenderTarget !== "undefined") {
    renderTarget = new THREE.WebGLMultisampleRenderTarget(
      Math.round(width * pr),
      Math.round(height * pr),
      {
        format: THREE.RGBAFormat,
        encoding: THREE.sRGBEncoding,
        samples: 4,
      }
    );
  } else {
    renderTarget = new THREE.WebGLRenderTarget(
      Math.round(width * pr),
      Math.round(height * pr),
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        encoding: THREE.sRGBEncoding,
      }
    );
  }

  composer = new THREE.EffectComposer(renderer, renderTarget);
  composer.setPixelRatio(pr);
  composer.setSize(width, height);

  // 1. Scene Render Pass (Full native resolution with 4x hardware MSAA)
  renderPass = new THREE.RenderPass(scene, camera);
  composer.addPass(renderPass);

  // 2. High-precision specular glint bloom (localized, zero milky haze)
  if (THREE.UnrealBloomPass) {
    bloomPass = new THREE.UnrealBloomPass(
      new THREE.Vector2(Math.round(width * pr * 0.5), Math.round(height * pr * 0.5)),
      0.08, // Subtle specular glint
      0.15, // Small radius
      0.92  // High threshold to avoid milky haze
    );
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

  // 1. PBR Asphalt Road Surface Quad Strip (Drivable Width = 70 px)
  const roadGeo = new THREE.BufferGeometry();
  const roadVertices = [];
  const roadNormals = [];
  const roadUvs = [];
  const roadIndices = [];

  let accumulatedV = 0.0;
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

    // Raised Y = 0.60 above ground plane (Y = -0.40) to guarantee >= 1.0 px separation & zero Z-fighting
    roadVertices.push(inW.x, 0.60, inW.z);
    roadVertices.push(outW.x, 0.60, outW.z);

    roadNormals.push(0, 1, 0, 0, 1, 0);

    if (i > 0) {
      const prevIn = envToWorld(innerPts[i - 1][0], innerPts[i - 1][1]);
      const segDist = Math.hypot(inW.x - prevIn.x, inW.z - prevIn.z);
      accumulatedV += segDist / 20.0; // Tighter, crisper asphalt aggregate
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
    color: 0x242426, // Natural dark asphalt
    map: pbrTextures.asphaltDiff,
    normalMap: pbrTextures.asphaltNor,
    roughnessMap: pbrTextures.asphaltRough,
    roughness: 0.70,
    metalness: 0.04,
    side: THREE.DoubleSide,
  });
  roadMesh = new THREE.Mesh(roadGeo, roadMat);
  roadMesh.receiveShadow = true;
  trackGroup.add(roadMesh);

  // 2. Curvature-Based Red/White Corner Curbs (Corners only, clean straightaways)
  const curvatures = computeCenterlineCurvature(centerPts);
  curbInnerMesh = buildCurvatureBasedCurbs(innerPts, curvatures, -4.0, 0.8);
  curbOuterMesh = buildCurvatureBasedCurbs(outerPts, curvatures, 4.0, 0.8);
  if (curbInnerMesh) trackGroup.add(curbInnerMesh);
  if (curbOuterMesh) trackGroup.add(curbOuterMesh);

  // 3. Dual Boundary Barrier Wall Ribbons (At exact env collision boundary: width=70 px)
  const barrierOuterMesh = buildBarrierWall(outerPts, 4.0, 7.5);
  const barrierInnerMesh = buildBarrierWall(innerPts, -4.0, 7.5);
  if (barrierOuterMesh) trackGroup.add(barrierOuterMesh);
  if (barrierInnerMesh) trackGroup.add(barrierInnerMesh);

  // 4. Dashed Centerline
  if (centerPts.length > 2) {
    const centerLineMesh = buildCenterline(centerPts);
    trackGroup.add(centerLineMesh);
  }

  // 5. Start / Finish Line Banner
  if (innerPts.length > 0 && outerPts.length > 0) {
    buildStartFinishMarker(innerPts[0], outerPts[0]);
  }

  // 6. Ground Runoff Plane (Surrounding terrain with rich natural grass)
  const groundGeo = new THREE.PlaneGeometry(3500, 3500);
  groundGeo.rotateX(-Math.PI / 2);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0x486b3a, // Vibrant natural grass green
    map: pbrTextures.groundDiff,
    normalMap: pbrTextures.groundNor,
    roughnessMap: pbrTextures.groundRough,
    roughness: 0.88,
    metalness: 0.02,
  });
  groundPlaneMesh = new THREE.Mesh(groundGeo, groundMat);
  groundPlaneMesh.position.y = -0.40;
  groundPlaneMesh.receiveShadow = true;
  trackGroup.add(groundPlaneMesh);

  // 7. Procedural Environment Props (Instanced Trees, Lamp Posts, Tecpro Barriers, Grandstands)
  buildEnvironmentProps(trackData);

  // Reset trajectory ribbon on new track
  clearTrajectoryTrail();
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
    // Only generate curbs in corner zones (|curvature| > 0.0028 rad/px)
    const curv = Math.abs(centerCurvatures[i] || 0.0);
    if (curv <= 0.0028) continue;

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
    if (nextCurv > 0.0028) {
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
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(wallGeo, wallMat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildEnvironmentProps(trackData) {
  if (!propsGroup) return;

  // Clear existing props
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

  // 1. Instanced Pine Trees (Cone + Trunk)
  const pineTrunkGeo = new THREE.CylinderGeometry(0.7, 1.1, 7.0, 5);
  pineTrunkGeo.translate(0, 3.5, 0);
  const pineFoliageGeo = new THREE.ConeGeometry(5.2, 11.0, 5);
  pineFoliageGeo.translate(0, 11.0, 0);

  const trunkMat = new THREE.MeshStandardMaterial({
    color: 0x4a2e18,
    roughness: 0.9,
    metalness: 0.05,
  });
  const pineFoliageMat = new THREE.MeshStandardMaterial({
    color: 0x1a4329,
    roughness: 0.85,
    metalness: 0.05,
  });

  // 2. Instanced Deciduous Oak Trees (Dodecahedron + Trunk)
  const oakTrunkGeo = new THREE.CylinderGeometry(0.8, 1.2, 8.0, 5);
  oakTrunkGeo.translate(0, 4.0, 0);
  const oakFoliageGeo = new THREE.DodecahedronGeometry(5.5, 1);
  oakFoliageGeo.translate(0, 11.5, 0);

  const oakFoliageMat = new THREE.MeshStandardMaterial({
    color: 0x2d6a4f,
    roughness: 0.82,
    metalness: 0.05,
  });

  // 3. Instanced Stadium / Trackside Lamp Posts
  const poleGeo = new THREE.CylinderGeometry(0.4, 0.6, 20.0, 5);
  poleGeo.translate(0, 10.0, 0);
  const lampFixtureGeo = new THREE.BoxGeometry(3.5, 0.6, 1.4);
  lampFixtureGeo.translate(1.5, 20.0, 0);

  const poleMat = new THREE.MeshStandardMaterial({
    color: 0x3a404d,
    metalness: 0.7,
    roughness: 0.3,
  });
  const lampFixtureMat = new THREE.MeshStandardMaterial({
    color: 0xfffae0,
    emissive: 0xfff3b0,
    emissiveIntensity: 1.8,
    roughness: 0.2,
  });

  // 4. Instanced Tecpro Corner Runoff Barriers
  const blockGeo = new THREE.BoxGeometry(6.5, 3.8, 3.2);
  blockGeo.translate(0, 1.9, 0);
  const redBlockMat = new THREE.MeshStandardMaterial({
    color: 0xd90429,
    roughness: 0.5,
    metalness: 0.1,
  });
  const whiteBlockMat = new THREE.MeshStandardMaterial({
    color: 0xf8f9fa,
    roughness: 0.5,
    metalness: 0.1,
  });

  // 5. Instanced Grandstand Seating Structures
  const grandstandGeo = new THREE.BoxGeometry(38.0, 12.0, 16.0);
  grandstandGeo.translate(0, 6.0, 0);
  const canopyGeo = new THREE.BoxGeometry(40.0, 1.2, 18.0);
  canopyGeo.translate(0, 13.0, 0);

  const grandstandMat = new THREE.MeshStandardMaterial({
    color: 0x222831,
    roughness: 0.6,
    metalness: 0.3,
  });
  const canopyMat = new THREE.MeshStandardMaterial({
    color: 0x00b4d8,
    roughness: 0.4,
    metalness: 0.2,
  });

  const pineTransforms = [];
  const oakTransforms = [];
  const lampTransforms = [];
  const tecproTransforms = [];
  const grandstandTransforms = [];

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

    // A. Lamp Posts: Offset >= 25 px outside barrier (boundary + 32 px = barrier + 28 px)
    if (i % 6 === 0) {
      const lx = pt[0] + nx * 32.0;
      const ly = pt[1] + ny * 32.0;
      const w = envToWorld(lx, ly);
      const angle = Math.atan2(ny, nx) + Math.PI;
      lampTransforms.push({ x: w.x, z: w.z, rotY: angle });
    }

    // B. Tecpro Corner Runoff Barriers: Placed on corner exits outside barrier
    if (isCorner && i % 2 === 0) {
      const bx = pt[0] + nx * 28.0;
      const by = pt[1] + ny * 28.0;
      const w = envToWorld(bx, by);
      const angle = Math.atan2(dy, dx);
      tecproTransforms.push({ x: w.x, z: w.z, rotY: angle, isRed: (i / 2) % 2 === 0 });
    }

    // C. Grandstands: Placed along long straightaways outside barrier
    if (!isCorner && i % 18 === 0 && grandstandTransforms.length < 5) {
      const gx = pt[0] + nx * 48.0;
      const gy = pt[1] + ny * 48.0;
      const w = envToWorld(gx, gy);
      const angle = Math.atan2(dy, dx);
      grandstandTransforms.push({ x: w.x, z: w.z, rotY: angle });
    }

    // D. Trees along track perimeter: Offset >= 30 px to 85 px outside barrier
    if (i % 3 === 0) {
      const hash = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
      const frac = hash - Math.floor(hash);
      const treeOffset = 30.0 + frac * 55.0;
      const tx = pt[0] + nx * treeOffset;
      const ty = pt[1] + ny * treeOffset;
      const w = envToWorld(tx, ty);
      const s = 0.85 + frac * 0.45;
      const rot = frac * Math.PI * 2;

      if (i % 2 === 0) {
        pineTransforms.push({ x: w.x, z: w.z, scale: s, rotY: rot });
      } else {
        oakTransforms.push({ x: w.x, z: w.z, scale: s, rotY: rot });
      }
    }
  }

  // E. Distant Background Landscape Trees
  for (let j = 0; j < 40; j++) {
    const angle = (j / 40) * Math.PI * 2;
    const dist = 320.0 + (Math.sin(j * 3.7) * 0.5 + 0.5) * 260.0;
    const x = Math.cos(angle) * dist;
    const z = Math.sin(angle) * dist;
    const s = 1.0 + (j % 5) * 0.15;
    if (j % 2 === 0) {
      pineTransforms.push({ x, z, scale: s, rotY: angle });
    } else {
      oakTransforms.push({ x, z, scale: s, rotY: angle });
    }
  }

  // Instantiate Pine Trees
  const pineCount = pineTransforms.length;
  if (pineCount > 0) {
    const pineTrunkMesh = new THREE.InstancedMesh(pineTrunkGeo, trunkMat, pineCount);
    const pineFoliageMesh = new THREE.InstancedMesh(pineFoliageGeo, pineFoliageMat, pineCount);
    pineTrunkMesh.castShadow = true;
    pineTrunkMesh.receiveShadow = true;
    pineFoliageMesh.castShadow = true;
    pineFoliageMesh.receiveShadow = true;

    pineTransforms.forEach((t, idx) => {
      dummy.position.set(t.x, 0, t.z);
      dummy.rotation.set(0, t.rotY, 0);
      dummy.scale.set(t.scale, t.scale, t.scale);
      dummy.updateMatrix();
      pineTrunkMesh.setMatrixAt(idx, dummy.matrix);
      pineFoliageMesh.setMatrixAt(idx, dummy.matrix);
    });

    pineTrunkMesh.instanceMatrix.needsUpdate = true;
    pineFoliageMesh.instanceMatrix.needsUpdate = true;
    propsGroup.add(pineTrunkMesh);
    propsGroup.add(pineFoliageMesh);
  }

  // Instantiate Oak Trees
  const oakCount = oakTransforms.length;
  if (oakCount > 0) {
    const oakTrunkMesh = new THREE.InstancedMesh(oakTrunkGeo, trunkMat, oakCount);
    const oakFoliageMesh = new THREE.InstancedMesh(oakFoliageGeo, oakFoliageMat, oakCount);
    oakTrunkMesh.castShadow = true;
    oakTrunkMesh.receiveShadow = true;
    oakFoliageMesh.castShadow = true;
    oakFoliageMesh.receiveShadow = true;

    oakTransforms.forEach((t, idx) => {
      dummy.position.set(t.x, 0, t.z);
      dummy.rotation.set(0, t.rotY, 0);
      dummy.scale.set(t.scale, t.scale, t.scale);
      dummy.updateMatrix();
      oakTrunkMesh.setMatrixAt(idx, dummy.matrix);
      oakFoliageMesh.setMatrixAt(idx, dummy.matrix);
    });

    oakTrunkMesh.instanceMatrix.needsUpdate = true;
    oakFoliageMesh.instanceMatrix.needsUpdate = true;
    propsGroup.add(oakTrunkMesh);
    propsGroup.add(oakFoliageMesh);
  }

  // Instantiate Lamp Posts
  const lampCount = lampTransforms.length;
  if (lampCount > 0) {
    const poleMesh = new THREE.InstancedMesh(poleGeo, poleMat, lampCount);
    const fixtureMesh = new THREE.InstancedMesh(lampFixtureGeo, lampFixtureMat, lampCount);
    poleMesh.castShadow = true;
    poleMesh.receiveShadow = true;

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

  // Instantiate Tecpro Barriers
  const tecproRed = tecproTransforms.filter((t) => t.isRed);
  const tecproWhite = tecproTransforms.filter((t) => !t.isRed);

  if (tecproRed.length > 0) {
    const redMesh = new THREE.InstancedMesh(blockGeo, redBlockMat, tecproRed.length);
    redMesh.castShadow = true;
    redMesh.receiveShadow = true;
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
    whiteMesh.receiveShadow = true;
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

  // Instantiate Grandstands
  const gsCount = grandstandTransforms.length;
  if (gsCount > 0) {
    const standMesh = new THREE.InstancedMesh(grandstandGeo, grandstandMat, gsCount);
    const canopyMesh = new THREE.InstancedMesh(canopyGeo, canopyMat, gsCount);
    standMesh.castShadow = true;
    standMesh.receiveShadow = true;
    canopyMesh.castShadow = true;
    canopyMesh.receiveShadow = true;

    grandstandTransforms.forEach((t, idx) => {
      dummy.position.set(t.x, 0, t.z);
      dummy.rotation.set(0, t.rotY, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      standMesh.setMatrixAt(idx, dummy.matrix);
      canopyMesh.setMatrixAt(idx, dummy.matrix);
    });

    standMesh.instanceMatrix.needsUpdate = true;
    canopyMesh.instanceMatrix.needsUpdate = true;
    propsGroup.add(standMesh);
    propsGroup.add(canopyMesh);
  }
}

function buildCenterline(centerPts) {
  const points = [];
  for (const p of centerPts) {
    if (!isValidVector2(p)) continue;
    const w = envToWorld(p[0], p[1]);
    points.push(new THREE.Vector3(w.x, 0.68, w.z));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineDashedMaterial({
    color: 0xffffff, // Natural high-contrast white road markings
    dashSize: 14.0,
    gapSize: 10.0,
    linewidth: 3,
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
// 3D Racing Vehicle Hierarchy (Accurate Scale: Wheelbase 50 px, Width 30 px)
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

  wheelMeshes = [];
  rearBrakeLights = [];

  const length = 50.0;
  const width = 30.0;
  const height = 12.0;

  const liveryMat = new THREE.MeshPhysicalMaterial({
    color: 0x00d4ff,
    metalness: 0.85,
    roughness: 0.18,
    clearcoat: 0.9,
    clearcoatRoughness: 0.1,
  });
  const carbonMat = new THREE.MeshStandardMaterial({
    color: 0x181c24,
    metalness: 0.6,
    roughness: 0.5,
  });
  const glassMat = new THREE.MeshPhysicalMaterial({
    color: 0x050c18,
    metalness: 0.9,
    roughness: 0.05,
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

  // 6. Rear Emissive Active Brake Lights
  const brakeLightMat = new THREE.MeshStandardMaterial({
    color: 0x440008,
    emissive: 0x220004,
    emissiveIntensity: 0.5,
  });
  const brakeLightGeo = new THREE.BoxGeometry(0.8, 1.2, 4.0);
  const brakeL = new THREE.Mesh(brakeLightGeo, brakeLightMat);
  brakeL.position.set(-length / 2 + 0.5, height * 0.5, -width * 0.32);
  const brakeR = new THREE.Mesh(brakeLightGeo, brakeLightMat);
  brakeR.position.set(-length / 2 + 0.5, height * 0.5, width * 0.32);
  carBodyGroup.add(brakeL);
  carBodyGroup.add(brakeR);
  rearBrakeLights.push(brakeL, brakeR);

  // 7. Four Wheels
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

function loadGLTFVehicle() {
  if (STATE.gltfLoaded || STATE.gltfLoading) return;
  STATE.gltfLoading = true;

  if (typeof THREE.GLTFLoader === "undefined") {
    console.warn("GLTFLoader not available, using procedural vehicle");
    STATE.gltfLoading = false;
    return;
  }

  const gltfLoader = new THREE.GLTFLoader();
  if (typeof THREE.DRACOLoader !== "undefined") {
    const dracoLoader = new THREE.DRACOLoader();
    dracoLoader.setDecoderPath("/static/vendor/draco/gltf/");
    gltfLoader.setDRACOLoader(dracoLoader);
  }

  gltfLoader.load(
    "/static/assets/models/car/car.glb",
    (gltf) => {
      try {
        setupGLTFCar(gltf);
        STATE.gltfLoaded = true;
        STATE.gltfLoading = false;
        console.log("GLTF sports car model loaded and rigged successfully");
      } catch (err) {
        console.warn("Error setting up GLTF car, falling back to procedural", err);
        STATE.gltfLoading = false;
      }
    },
    undefined,
    (err) => {
      console.warn("Failed to load car.glb, using procedural vehicle fallback", err);
      STATE.gltfLoading = false;
    }
  );
}

function setupGLTFCar(gltf) {
  const model = gltf.scene;

  // Wheel nodes and steering wheel
  const wheelFL = model.getObjectByName("wheel_fl");
  const wheelFR = model.getObjectByName("wheel_fr");
  const wheelRL = model.getObjectByName("wheel_rl");
  const wheelRR = model.getObjectByName("wheel_rr");
  const steeringWheel = model.getObjectByName("steering_wheel");
  const brakeMesh = model.getObjectByName("lights_red");
  const bodyMesh = model.getObjectByName("body");

  // Create dedicated steering pivots for front wheels
  let flPivot = null;
  let frPivot = null;

  if (wheelFL && wheelFL.parent) {
    flPivot = new THREE.Group();
    flPivot.name = "steer_pivot_fl";
    flPivot.position.copy(wheelFL.position);
    wheelFL.parent.add(flPivot);
    wheelFL.position.set(0, 0, 0);
    flPivot.add(wheelFL);
  }

  if (wheelFR && wheelFR.parent) {
    frPivot = new THREE.Group();
    frPivot.name = "steer_pivot_fr";
    frPivot.position.copy(wheelFR.position);
    wheelFR.parent.add(frPivot);
    wheelFR.position.set(0, 0, 0);
    frPivot.add(wheelFR);
  }

  // High-fidelity unbranded sports racing car PBR materials
  const bodyMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xd91626, // Racing Rosso / vibrant clean red
    metalness: 0.35,
    roughness: 0.18,
    clearcoat: 1.0,
    clearcoatRoughness: 0.06,
    envMap: hdrEnvMap,
    envMapIntensity: 1.35,
  });

  const glassMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x111622,
    metalness: 0.1,
    roughness: 0.05,
    transmission: 0.65,
    transparent: true,
    opacity: 0.60,
    envMap: hdrEnvMap,
    envMapIntensity: 1.5,
  });

  const rimMaterial = new THREE.MeshStandardMaterial({
    color: 0xdcdfe4, // Satin Silver Rims
    metalness: 0.90,
    roughness: 0.20,
    envMap: hdrEnvMap,
    envMapIntensity: 1.2,
  });

  const tireMaterial = new THREE.MeshStandardMaterial({
    color: 0x18181a, // Matte Black Rubber
    metalness: 0.05,
    roughness: 0.82,
  });

  const headlightMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0.6,
    roughness: 0.08,
    transmission: 0.85,
    transparent: true,
    opacity: 0.80,
    envMap: hdrEnvMap,
    envMapIntensity: 1.5,
  });

  const taillightMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xc80815,
    emissive: 0x220004,
    emissiveIntensity: 0.4,
    metalness: 0.15,
    roughness: 0.15,
    clearcoat: 0.8,
  });

  const trimMaterial = new THREE.MeshStandardMaterial({
    color: 0x1a1a1c, // Dark trim / diffusers / splitters
    metalness: 0.35,
    roughness: 0.45,
  });

  const chromeMaterial = new THREE.MeshStandardMaterial({
    color: 0xeeeeee,
    metalness: 0.95,
    roughness: 0.08,
    envMap: hdrEnvMap,
    envMapIntensity: 1.4,
  });

  const brakeMaterials = [];
  model.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;

      const n = (child.name || "").toLowerCase();

      if (n === "body") {
        child.material = bodyMaterial;
      } else if (n === "glass") {
        child.material = glassMaterial;
      } else if (n.startsWith("tire")) {
        child.material = tireMaterial;
      } else if (n.startsWith("rim_") || n.startsWith("wheel")) {
        child.material = rimMaterial;
      } else if (n === "lights") {
        child.material = headlightMaterial;
      } else if (n === "lights_red" || n === "brakes" || n === "steering_red_lights") {
        const mat = taillightMaterial.clone();
        child.material = mat;
        brakeMaterials.push(mat);
      } else if (n === "chrome") {
        child.material = chromeMaterial;
      } else if (n.includes("carbon") || n === "grills" || n === "plastic_gray" || n === "wipers") {
        child.material = trimMaterial;
      } else {
        if (child.material) {
          if (hdrEnvMap) {
            child.material.envMap = hdrEnvMap;
            child.material.envMapIntensity = 1.0;
          }
          child.material.needsUpdate = true;
        }
      }
    }
  });

  // Center offset & uniform scaling
  // Model raw wheelbase = 1.495 - (-1.155) = 2.650 m.
  // Wheelbase midpoint = +0.170 m along Z.
  // Uniform scale to 50.0 px (env.car.wheelbase): 50.0 / 2.650 = 18.8679245
  const scale = 50.0 / 2.650;

  const innerPivot = new THREE.Group();
  innerPivot.position.set(0, 0, -0.170);
  innerPivot.add(model);

  const orientGroup = new THREE.Group();
  orientGroup.rotation.y = -Math.PI / 2; // Orient forward along Three.js +X
  orientGroup.scale.set(scale, scale, scale);
  orientGroup.add(innerPivot);

  gltfCarContainer = new THREE.Group();
  gltfCarContainer.name = "gltf_car_container";
  gltfCarContainer.add(orientGroup);

  gltfRig = {
    flWheel: wheelFL,
    frWheel: wheelFR,
    rlWheel: wheelRL,
    rrWheel: wheelRR,
    wheels: [wheelFL, wheelFR, wheelRL, wheelRR].filter(Boolean),
    flPivot: flPivot,
    frPivot: frPivot,
    steeringWheel: steeringWheel,
    brakeMaterials: brakeMaterials,
    body: bodyMesh,
    container: gltfCarContainer,
  };

  // Hide procedural fallback parts, add GLTF car to carGroup
  if (carBodyGroup) carBodyGroup.visible = false;
  if (frontLeftSteerPivot) frontLeftSteerPivot.visible = false;
  if (frontRightSteerPivot) frontRightSteerPivot.visible = false;
  wheelMeshes.forEach((w) => {
    if (w.parent) w.parent.visible = false;
  });

  carGroup.add(gltfCarContainer);
}

function buildCarModel() {
  while (carGroup.children.length > 0) {
    const obj = carGroup.children[0];
    carGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
  gltfCarContainer = null;
  gltfRig = null;
  STATE.gltfLoaded = false;
  STATE.gltfLoading = false;

  // 1. Build procedural fallback immediately
  buildProceduralVehicle();

  // 2. Load GLTF sports car model asynchronously
  loadGLTFVehicle();
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
  STATE.interp.active = false;
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
  const newPt = new THREE.Vector3(w.x, 0.64, w.z);

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

function updateCameraDirector(car, velocity, delta = 0.016) {
  if (!carGroup || !camera) return;

  const carPos = carGroup.position;
  const heading = carGroup.rotation.y;
  const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
  const backward = forward.clone().negate();

  if (STATE.activeCameraMode === "cockpit") {
    if (controls) controls.enabled = false;

    // Cockpit mode: Driver eye placed in driver's seat looking out over hood and dash (near=0.4)
    if (camera.near !== 0.4) {
      camera.near = 0.4;
      camera.updateProjectionMatrix();
    }

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
    if (camera.near !== 20.0) {
      camera.near = 20.0;
      camera.updateProjectionMatrix();
    }
    if (controls) controls.enabled = false;
    camera.position.set(0, 750.0, 0);
    camera.up.set(0, 0, -1);
    camera.lookAt(0, 0, 0);

  } else {
    // chase or orbit (near >= 2.0)
    if (camera.near !== 2.0) {
      camera.near = 2.0;
      camera.updateProjectionMatrix();
    }

    if (STATE.activeCameraMode === "chase") {
      if (controls) controls.enabled = false;
      // 2.6 car lengths behind (130 px), 45 px above (sharp proportions, car fills frame comfortably)
      const followDist = 130.0;
      const followHeight = 45.0 + Math.min(Math.abs(velocity) * 1.5, 8.0);
      const targetCamPos = new THREE.Vector3(
        carPos.x + backward.x * followDist,
        followHeight,
        carPos.z + backward.z * followDist
      );
      const lookTarget = new THREE.Vector3(
        carPos.x + forward.x * 52.0,
        10.0,
        carPos.z + forward.z * 52.0
      );

      // Frame-delta-based camera smoothing: 1 - exp(-decay * delta)
      const safeDelta = Math.min(Math.max(delta || 0.016, 0.001), 0.1);
      const camLerp = 1.0 - Math.exp(-7.5 * safeDelta);
      const targetLerp = 1.0 - Math.exp(-9.0 * safeDelta);

      STATE.cameraPos.lerp(targetCamPos, camLerp);
      STATE.cameraTarget.lerp(lookTarget, targetLerp);

      camera.position.copy(STATE.cameraPos);
      camera.lookAt(STATE.cameraTarget);
      camera.up.set(0, 1, 0);

    } else if (STATE.activeCameraMode === "orbit") {
      camera.up.set(0, 1, 0);
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
// Debug Inspector Overlay (Key: D) & Physics Footprint Wireframe
// ==============================================================================

function buildDebugHelpers() {
  trackAxesHelper = new THREE.AxesHelper(80);
  trackAxesHelper.position.set(0, 0.5, 0);
  debugGroup.add(trackAxesHelper);

  carAxesHelper = new THREE.AxesHelper(30);
  carAxesHelper.position.set(0, 3.5, 0);
  carGroup.add(carAxesHelper);

  // Exact physics footprint wireframe box (30x50 px, env.car.wheelbase 50 px)
  const footprintGeo = new THREE.BoxGeometry(50.0, 12.0, 30.0);
  const footprintEdges = new THREE.EdgesGeometry(footprintGeo);
  carPhysicsFootprint = new THREE.LineSegments(
    footprintEdges,
    new THREE.LineBasicMaterial({ color: 0xffb700, linewidth: 2 })
  );
  carPhysicsFootprint.position.y = 6.0;
  carPhysicsFootprint.visible = false;
  carGroup.add(carPhysicsFootprint);
}

function updateDebugOverlay(car) {
  const overlay = document.getElementById("debugStatsOverlay");
  if (!overlay) return;

  if (!STATE.debugMode) {
    overlay.style.display = "none";
    debugGroup.visible = false;
    if (carAxesHelper) carAxesHelper.visible = false;
    if (carPhysicsFootprint) carPhysicsFootprint.visible = false;
    return;
  }

  overlay.style.display = "block";
  debugGroup.visible = true;
  if (carAxesHelper) carAxesHelper.visible = true;
  if (carPhysicsFootprint) carPhysicsFootprint.visible = true;

  const w = envToWorld(car.x, car.y);
  const calls = renderer ? renderer.info.render.calls : 0;
  const tris = renderer ? renderer.info.render.triangles.toLocaleString() : "0";
  const textures = renderer ? renderer.info.memory.textures : 0;
  const geometries = renderer ? renderer.info.memory.geometries : 0;

  overlay.innerHTML = `
    <div class="debug-title">DEBUG INSPECTOR (KEY: D)</div>
    <div class="debug-row"><span>Vehicle Model:</span> <strong>${gltfCarContainer ? "glTF Sports Car (Rigged)" : "Procedural Box"}</strong></div>
    <div class="debug-row"><span>Scale:</span> <strong>PX_TO_WORLD = 1.0</strong></div>
    <div class="debug-row"><span>Track Width:</span> <strong>70 px</strong></div>
    <div class="debug-row"><span>Car Dimensions:</span> <strong>50L x 30W x 12H px</strong></div>
    <div class="debug-row"><span>Wheelbase:</span> <strong>50 px (Physics)</strong></div>
    <div class="debug-row"><span>Footprint:</span> <strong>30x50 px (Wireframe)</strong></div>
    <div class="debug-row"><span>Env Pos:</span> <strong>(${car.x.toFixed(1)}, ${car.y.toFixed(1)})</strong></div>
    <div class="debug-row"><span>Env Heading:</span> <strong>${car.angle.toFixed(1)}°</strong></div>
    <div class="debug-row"><span>World Pos:</span> <strong>(${w.x.toFixed(1)}, ${w.z.toFixed(1)})</strong></div>
    <div class="debug-row"><span>Three.js RotY:</span> <strong>${carGroup.rotation.y.toFixed(3)} rad</strong></div>
    <div class="debug-row"><span>Active Cam:</span> <strong>${STATE.activeCameraMode.toUpperCase()}</strong></div>
    <div class="debug-row"><span>FPS:</span> <strong>${currentFPS}</strong></div>
    <div class="debug-row"><span>Draw Calls:</span> <strong>${calls} (&lt; 400)</strong></div>
    <div class="debug-row"><span>Triangles:</span> <strong>${tris} (&lt; 1.5M)</strong></div>
    <div class="debug-row"><span>Textures:</span> <strong>${textures}</strong></div>
    <div class="debug-row"><span>Geometries:</span> <strong>${geometries}</strong></div>
  `;
}

// ==============================================================================
// Graphics Quality Switcher (High / Low)
// ==============================================================================

function setGraphicsQuality(quality) {
  STATE.graphicsQuality = quality;
  const btnQuality = document.getElementById("btnQualityToggle");
  const qualityDot = document.getElementById("qualityDot");
  const qualityText = document.getElementById("qualityText");

  if (quality === "low") {
    renderer.shadowMap.enabled = false;
    if (mainSun) mainSun.castShadow = false;
    loadPBRTextures();
    if (roadMesh && roadMesh.material) {
      roadMesh.material.map = pbrTextures.asphaltDiff;
      roadMesh.material.normalMap = pbrTextures.asphaltNor;
      roadMesh.material.roughnessMap = pbrTextures.asphaltRough;
      roadMesh.material.needsUpdate = true;
    }
    if (btnQuality) btnQuality.className = "pill pill-quality low";
    if (qualityDot) qualityDot.className = "pill-dot";
    if (qualityText) qualityText.innerText = "QUALITY: LOW (FAST)";
  } else {
    renderer.shadowMap.enabled = true;
    if (mainSun) mainSun.castShadow = true;
    loadPBRTextures();
    if (roadMesh && roadMesh.material) {
      roadMesh.material.map = pbrTextures.asphaltDiff;
      roadMesh.material.normalMap = pbrTextures.asphaltNor;
      roadMesh.material.roughnessMap = pbrTextures.asphaltRough;
      roadMesh.material.needsUpdate = true;
    }
    if (btnQuality) btnQuality.className = "pill pill-quality active";
    if (qualityDot) qualityDot.className = "pill-dot green";
    if (qualityText) qualityText.innerText = "QUALITY: HIGH (CRISP PBR)";
  }
}

// ==============================================================================
// Cinematic UI Mode (Key: H)
// ==============================================================================

function toggleCinematicUIMode() {
  STATE.cinematicMode = !STATE.cinematicMode;
  document.body.classList.toggle("cinematic-hud-mode", STATE.cinematicMode);

  const btnCinematic = document.getElementById("btnCinematicToggle");
  const cinematicDot = document.getElementById("cinematicDot");
  const bottomHud = document.getElementById("cinematicBottomHud");

  if (STATE.cinematicMode) {
    if (btnCinematic) btnCinematic.classList.add("active");
    if (cinematicDot) cinematicDot.className = "pill-dot cyan active";
    if (bottomHud) bottomHud.style.display = "block";
  } else {
    if (btnCinematic) btnCinematic.classList.remove("active");
    if (cinematicDot) cinematicDot.className = "pill-dot cyan";
    if (bottomHud) bottomHud.style.display = "none";
  }

  // Trigger resize observer / canvas resize adaptation
  setTimeout(() => onCanvasResize(), 60);
  setTimeout(() => onCanvasResize(), 380);
}

function updateCinematicBottomHud(car, session) {
  const hudSpeed = document.getElementById("hudSpeedVal");
  const hudSteer = document.getElementById("hudSteerVal");
  const hudThrottle = document.getElementById("hudThrottleVal");
  const hudProgFill = document.getElementById("hudProgFill");
  const hudProgVal = document.getElementById("hudProgVal");

  if (hudSpeed) hudSpeed.innerText = (Number(car.velocity) || 0.0).toFixed(1);
  if (hudSteer) {
    const steerDeg = Number(car.steering_angle_deg) || 0.0;
    hudSteer.innerText = `${steerDeg >= 0 ? "+" : ""}${steerDeg.toFixed(1)}°`;
  }
  if (hudThrottle) {
    const throt = Number(car.throttle) || 0.0;
    hudThrottle.innerText = `${throt >= 0 ? "+" : ""}${throt.toFixed(2)}`;
    hudThrottle.style.color = throt >= 0 ? "var(--accent-green)" : "var(--accent-red)";
  }
  if (hudProgVal && hudProgFill) {
    const progPct = Math.min(100, Math.max(0, (Number(car.lap_progress) || 0.0) * 100));
    hudProgVal.innerText = `${progPct.toFixed(1)}%`;
    hudProgFill.style.width = `${progPct}%`;
  }
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

  const delta = Math.min(Math.max((currentTime - lastFrameTime) * 0.001, 0.0001), 0.1);
  lastFrameTime = currentTime;

  frameCount++;
  if (currentTime - lastFpsUpdate >= 500) {
    currentFPS = Math.round((frameCount * 1000) / (currentTime - lastFpsUpdate));
    frameCount = 0;
    lastFpsUpdate = currentTime;
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
      const startText = document.getElementById("startText");
      if (startIcon) startIcon.innerText = "↺";
      if (startText) startText.innerText = "Replay";
      const startBtn = document.getElementById("btnStartPause");
      if (startBtn) startBtn.classList.remove("active");
      const isCrashed = STATE.playback.rolloutData.session.is_crashed;
      const isCompleted = STATE.playback.rolloutData.session.is_completed;
      updateStatusPill(isCrashed ? "CRASHED" : (isCompleted ? "COMPLETED" : "ENDED"), isCrashed ? "red" : "green");
      if (isCompleted) {
        showOverlayModal("LAP COMPLETED!", `Clean lap finished at step ${STATE.playback.maxStep}.`, STATE.playback.rolloutData.lap_review);
      } else if (isCrashed) {
        showOverlayModal("CRASH DETECTED", `Vehicle departed track boundary at step ${STATE.playback.maxStep}.`, STATE.playback.rolloutData.lap_review);
      }
    }
  }

  // Deterministic Rollout Playback Rendering
  if (STATE.playback.rolloutData && STATE.playback.rolloutData.steps && STATE.playback.rolloutData.steps.length > 0) {
    renderPlaybackFrame(STATE.playback.playhead, delta);
  }

  // Keep sunlight and shadow camera centered around vehicle for crisp shadows
  if (mainSun && carGroup) {
    mainSun.position.set(carGroup.position.x + 300, 480, carGroup.position.z + 200);
    mainSun.target = carGroup;
  }

  // Render 3D Scene through Post-Processing Pipeline
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

  // 1. Position interpolation (world units)
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
  STATE.lastCarVelocity = curVelocity;

  // 4. Smooth continuous wheel rotation using interpolated speed
  const wheelRadiusPx = 6.755;
  const spinDelta = (curVelocity / wheelRadiusPx) * (delta / 0.030);
  wheelSpinAngle -= spinDelta;

  const steerRad = -envSteerToWorld(curSteerDeg);
  if (gltfRig) {
    if (gltfRig.flPivot) gltfRig.flPivot.rotation.y = steerRad;
    if (gltfRig.frPivot) gltfRig.frPivot.rotation.y = steerRad;
    if (gltfRig.steeringWheel) gltfRig.steeringWheel.rotation.z = -steerRad * 2.5;
    if (gltfRig.wheels) {
      gltfRig.wheels.forEach((wheel) => {
        wheel.rotation.x = wheelSpinAngle;
      });
    }
  } else {
    if (frontLeftSteerPivot) frontLeftSteerPivot.rotation.y = steerRad;
    if (frontRightSteerPivot) frontRightSteerPivot.rotation.y = steerRad;
    wheelMeshes.forEach((wheel) => {
      wheel.rotation.x = wheelSpinAngle;
    });
  }

  // 5. Dynamic brake lights
  const isDecel = curThrottle < -0.05 || car0.action_mode === "deceleration";
  if (gltfRig && gltfRig.brakeMaterials) {
    gltfRig.brakeMaterials.forEach((mat) => {
      mat.emissive.setHex(isDecel ? 0xff0020 : 0x220004);
      mat.emissiveIntensity = isDecel ? 2.5 : 0.4;
    });
  }
  rearBrakeLights.forEach((light) => {
    light.material.color.setHex(isDecel ? 0xff0020 : 0x440008);
    light.material.emissive.setHex(isDecel ? 0xff0020 : 0x220004);
    light.material.emissiveIntensity = isDecel ? 1.5 : 0.4;
  });

  // 6. Rangefinder sensor rays
  if (s0.rays) {
    updateLidarVisuals(s0.rays, { x: curX, y: curZ });
  }

  // 7. Trajectory ribbon (synchronized up to baseIdx + current interpolated pose)
  updatePlaybackTrajectory(steps, baseIdx, curX, curZ);

  // 8. Frame-delta-based camera smoothing
  if (carGroup) {
    updateCameraDirector(
      { x: carGroup.position.x, y: carGroup.position.z, angle: (carGroup.rotation.y * 180.0) / Math.PI },
      curVelocity,
      delta
    );
  }

  // 9. Synchronize digital UI, dials, step counter, charts, saliency, and engineer feed
  const curEnvX = THREE.MathUtils.lerp(car0.x, car1.x, alpha);
  const curEnvY = THREE.MathUtils.lerp(car0.y, car1.y, alpha);
  const curEnvAngle = THREE.MathUtils.lerp(car0.angle, car1.angle, alpha);
  const curProgress = THREE.MathUtils.lerp(car0.lap_progress, car1.lap_progress, alpha);

  const interpolatedVehicle = {
    x: curEnvX,
    y: curEnvY,
    angle: curEnvAngle,
    velocity: curVelocity,
    steering: THREE.MathUtils.lerp(car0.steering, car1.steering, alpha),
    steering_angle_deg: curSteerDeg,
    throttle: curThrottle,
    action_mode: curThrottle > 0 ? "throttle" : (curThrottle < 0 ? "deceleration" : "coasting"),
    lap_progress: curProgress,
  };

  const currentSession = {
    ...rollout.session,
    step_count: baseIdx,
    max_steps: rollout.session.max_steps || steps.length - 1,
  };

  syncCockpitAndGauges(interpolatedVehicle, currentSession, s0.telemetry);

  // Discrete step updates (saliency, commentary feed, minimap)
  if (baseIdx !== STATE.playback.lastStepIndex) {
    STATE.playback.lastStepIndex = baseIdx;

    if (s0.saliency) {
      STATE.currentSaliency = s0.saliency;
      renderSaliencyBars();
    }

    if (rollout.commentary) {
      syncPlaybackCommentary(rollout.commentary, baseIdx);
    }

    if (STATE.trackData && minimapCtx) {
      renderRadarMinimap(STATE.trackData, interpolatedVehicle);
    }

    if (STATE.debugMode) {
      updateDebugOverlay(interpolatedVehicle);
    }
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

function syncPlaybackCommentary(allCommentary, currentStep) {
  const feed = document.getElementById("commentaryFeed");
  if (!feed || !Array.isArray(allCommentary)) return;

  const visibleEvents = allCommentary.filter((item) => item.step <= currentStep);
  feed.innerHTML = "";
  for (const item of visibleEvents.slice(-50)) {
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
  }
  feed.scrollTop = feed.scrollHeight;
}

// ==============================================================================
// Simulation Backend Communication (FastAPI Client)
// ==============================================================================

async function startSimulation(presetId = STATE.presetId, trackSeed = STATE.trackSeed) {
  try {
    updateStatusPill("SIMULATING ROLLOUT...", "amber");
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

    // Render initial frame at step 0
    renderPlaybackFrame(0.0, 0.016);

    // Snap chase camera immediately on initial session load to start framing
    if (STATE.activeCameraMode === "chase" && carGroup) {
      const heading = carGroup.rotation.y;
      const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
      const backward = forward.clone().negate();
      const followDist = 175.0;
      const followHeight = 60.0;
      STATE.cameraPos.set(
        carGroup.position.x + backward.x * followDist,
        followHeight,
        carGroup.position.z + backward.z * followDist
      );
      STATE.cameraTarget.set(
        carGroup.position.x + forward.x * 45.0,
        12.0,
        carGroup.position.z + forward.z * 45.0
      );
      camera.position.copy(STATE.cameraPos);
      camera.lookAt(STATE.cameraTarget);
      camera.up.set(0, 1, 0);
    }

    const startBtn = document.getElementById("btnStartPause");
    if (startBtn) {
      startBtn.classList.remove("active");
      const icon = document.getElementById("startIcon");
      const text = document.getElementById("startText");
      if (icon) icon.innerText = "▶";
      if (text) text.innerText = "Start Simulation";
    }

    updateStatusPill("READY (ROLLOUT LOADED)", "cyan");
  } catch (err) {
    console.error("Simulation start / rollout loading failed:", err);
    STATE.playback.isLoading = false;
    updateStatusPill("ROLLOUT ERROR", "red");
  }
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
  STATE.playback.lastStepIndex = -1;
  clearTrajectoryTrail();
  resetCommentaryFeed();
  renderPlaybackFrame(0.0, 0.016);

  const startBtn = document.getElementById("btnStartPause");
  if (startBtn) {
    startBtn.classList.remove("active");
    const icon = document.getElementById("startIcon");
    const text = document.getElementById("startText");
    if (icon) icon.innerText = "▶";
    if (text) text.innerText = "Start Simulation";
  }
  updateStatusPill("READY", "cyan");
  hideOverlayModal();
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

  // 1. Update 3D Vehicle Transform & Interpolation Targets
  const carWorld = envToWorld(car.x, car.y);
  const headingRad = envHeadingToWorld(car.angle);
  const steerAngleDeg = Number(car.steering_angle_deg) || 0.0;
  const now = performance.now();

  if (!STATE.interp.active) {
    STATE.interp.prevPos = { x: carWorld.x, z: carWorld.z };
    STATE.interp.targetPos = { x: carWorld.x, z: carWorld.z };
    STATE.interp.prevHeading = headingRad;
    STATE.interp.targetHeading = headingRad;
    STATE.interp.prevSteer = steerAngleDeg;
    STATE.interp.targetSteer = steerAngleDeg;
    STATE.interp.prevVelocity = car.velocity;
    STATE.interp.targetVelocity = car.velocity;
    STATE.interp.startTime = now;
    STATE.interp.duration = 35.0;
    STATE.interp.active = true;

    carGroup.position.set(carWorld.x, 0.0, carWorld.z);
    carGroup.rotation.y = headingRad;
  } else {
    STATE.interp.prevPos = { x: carGroup.position.x, z: carGroup.position.z };
    STATE.interp.targetPos = { x: carWorld.x, z: carWorld.z };
    STATE.interp.prevHeading = carGroup.rotation.y;
    STATE.interp.targetHeading = headingRad;
    STATE.interp.prevSteer = STATE.interp.targetSteer;
    STATE.interp.targetSteer = steerAngleDeg;
    STATE.interp.prevVelocity = STATE.lastCarVelocity || 0.0;
    STATE.interp.targetVelocity = car.velocity;
    STATE.interp.duration = Math.max(now - STATE.interp.startTime, 20.0);
    STATE.interp.startTime = now;
  }

  // 2. Dynamic Emissive Brake Light Illumination
  const isDecel = car.throttle < -0.05 || car.action_mode === "deceleration";
  if (gltfRig && gltfRig.brakeMaterials) {
    gltfRig.brakeMaterials.forEach((mat) => {
      mat.emissive.setHex(isDecel ? 0xff0020 : 0x220004);
      mat.emissiveIntensity = isDecel ? 2.5 : 0.4;
    });
  }
  rearBrakeLights.forEach((light) => {
    light.material.color.setHex(isDecel ? 0xff0020 : 0x440008);
    light.material.emissive.setHex(isDecel ? 0xff0020 : 0x220004);
    light.material.emissiveIntensity = isDecel ? 1.5 : 0.4;
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
  // Cinematic Floating Bottom HUD
  updateCinematicBottomHud(car, session);

  // Speed
  const vel = Number(car.velocity) || 0.0;
  const valSpeed = document.getElementById("valSpeed");
  const valVelocity = document.getElementById("valVelocity");
  const barSpeed = document.getElementById("barSpeed");
  const barVelocity = document.getElementById("barVelocity");
  const cockpitSpeed = document.getElementById("cockpitSpeed");
  if (valSpeed) valSpeed.innerText = vel.toFixed(2);
  if (valVelocity) valVelocity.innerHTML = `${vel.toFixed(2)} <span class="unit">px/step</span>`;
  if (cockpitSpeed) cockpitSpeed.innerText = vel.toFixed(2);
  if (barSpeed) barSpeed.style.width = `${Math.min(100, Math.max(0, (vel / 4.0) * 100))}%`;
  if (barVelocity) barVelocity.style.width = `${Math.min(100, Math.max(0, (vel / 4.0) * 100))}%`;

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
  if (STATE.playback) {
    STATE.playback.isPlaying = false;
  }
  const btn = document.getElementById("btnStartPause");
  if (btn) {
    btn.classList.remove("active");
    const icon = document.getElementById("startIcon");
    const text = document.getElementById("startText");
    if (icon) icon.innerText = "▶";
    if (text) text.innerText = "Resume";
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

  // Set camera.near per mode to prevent Z-fighting and near-clipping:
  // 0.4 in cockpit, >= 2.0 in chase/orbit, and 20.0 in overhead
  if (mode === "cockpit") {
    camera.near = 0.4;
    camera.updateProjectionMatrix();
    if (controls) controls.enabled = false;
    camera.up.set(0, 1, 0);
  } else if (mode === "overhead") {
    camera.near = 20.0;
    camera.updateProjectionMatrix();
    if (controls) controls.enabled = false;
    camera.position.set(0, 750.0, 0);
    camera.up.set(0, 0, -1);
    camera.lookAt(0, 0, 0);
  } else {
    // chase or orbit
    camera.near = 2.0;
    camera.updateProjectionMatrix();
    camera.up.set(0, 1, 0);
    if (mode === "orbit") {
      if (controls && carGroup) {
        controls.enabled = true;
        camera.position.set(carGroup.position.x - 70.0, 50.0, carGroup.position.z + 70.0);
        controls.target.set(carGroup.position.x, 6.0, carGroup.position.z);
        controls.update();
      }
    } else if (mode === "chase") {
      if (controls) controls.enabled = false;
      if (carGroup) {
        const heading = carGroup.rotation.y;
        const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
        const backward = forward.clone().negate();
        const vel = STATE.lastCarVelocity || 0;
        const followDist = 175.0;
        const followHeight = 60.0 + Math.min(Math.abs(vel) * 2.0, 10.0);
        STATE.cameraPos.set(
          carGroup.position.x + backward.x * followDist,
          followHeight,
          carGroup.position.z + backward.z * followDist
        );
        STATE.cameraTarget.set(
          carGroup.position.x + forward.x * 45.0,
          12.0,
          carGroup.position.z + forward.z * 45.0
        );
        camera.position.copy(STATE.cameraPos);
        camera.lookAt(STATE.cameraTarget);
      }
    }
  }

  // Force immediate camera update if car exists
  if (STATE.lastCarVelocity !== undefined) {
    updateCameraDirector(
      { x: carGroup.position.x, y: carGroup.position.z, angle: (carGroup.rotation.y * 180.0) / Math.PI },
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

  // Quality Toggle Button
  const btnQuality = document.getElementById("btnQualityToggle");
  function toggleQuality() {
    const newQual = STATE.graphicsQuality === "high" ? "low" : "high";
    setGraphicsQuality(newQual);
  }
  if (btnQuality) {
    btnQuality.addEventListener("click", toggleQuality);
  }

  // Cinematic Mode Toggle Button
  const btnCinematic = document.getElementById("btnCinematicToggle");
  if (btnCinematic) {
    btnCinematic.addEventListener("click", toggleCinematicUIMode);
  }

  // Start / Pause
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
        btnStartPause.classList.add("active");
        const icon = document.getElementById("startIcon");
        const text = document.getElementById("startText");
        if (icon) icon.innerText = "⏸";
        if (text) text.innerText = "Pause";
        updateStatusPill("PLAYING", "green");
        hideOverlayModal();
      }
    });
  }

  // Step Button
  const btnStep1 = document.getElementById("btnStep") || document.getElementById("btnStep1");
  if (btnStep1) {
    btnStep1.addEventListener("click", () => {
      pauseSimulation();
      stepSimulation();
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

  // Track Selector & Seed Controls
  const selTrack = document.getElementById("trackPreset") || document.getElementById("selectTrackPreset");
  const seedInput = document.getElementById("trackSeed") || document.getElementById("inputSeed");
  const btnLoadTrack = document.getElementById("btnLoadTrack");

  function triggerLoadTrack() {
    pauseSimulation();
    const preset = selTrack ? selTrack.value : "standard_medium";
    const seed = seedInput ? parseInt(seedInput.value, 10) || 1001 : 1001;
    startSimulation(preset, seed);
  }

  if (btnLoadTrack) {
    btnLoadTrack.addEventListener("click", triggerLoadTrack);
  }
  if (selTrack) {
    selTrack.addEventListener("change", triggerLoadTrack);
  }

  const btnNewSeed = document.getElementById("btnNewSeed");
  if (btnNewSeed) {
    btnNewSeed.addEventListener("click", () => {
      const newSeed = Math.floor(Math.random() * 900000) + 1000;
      if (seedInput) seedInput.value = newSeed;
      triggerLoadTrack();
    });
  }

  // Speed Slider & Speed Buttons
  const speedRange = document.getElementById("speedRange");
  const speedVal = document.getElementById("speedVal");
  if (speedRange) {
    speedRange.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      const speeds = [0.5, 1.0, 2.0, 3.0, 5.0];
      const spd = speeds[Math.min(speeds.length - 1, Math.max(0, val - 1))];
      STATE.playback.speed = spd;
      STATE.simSpeed = Math.round(spd);
      if (speedVal) speedVal.innerText = `${spd}x`;
    });
  }

  ["btnSpeed1x", "btnSpeed2x", "btnSpeed4x"].forEach((id, idx) => {
    const btn = document.getElementById(id);
    const speeds = [1.0, 2.0, 4.0];
    if (btn) {
      btn.addEventListener("click", () => {
        ["btnSpeed1x", "btnSpeed2x", "btnSpeed4x"].forEach((bId) => {
          const b = document.getElementById(bId);
          if (b) b.classList.remove("active");
        });
        btn.classList.add("active");
        STATE.playback.speed = speeds[idx];
        STATE.simSpeed = Math.round(speeds[idx]);
        if (speedVal) speedVal.innerText = `${speeds[idx]}x`;
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
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;
    if (e.key === "d" || e.key === "D") {
      toggleDebug();
    } else if (e.key === "h" || e.key === "H") {
      toggleCinematicUIMode();
    } else if (e.key === "q" || e.key === "Q") {
      toggleQuality();
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
