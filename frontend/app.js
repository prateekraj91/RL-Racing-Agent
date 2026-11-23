/**
 * RL Racing Agent // Final Product Reconstruction
 * Concept-Matched Autonomous Simulation Dashboard & Telemetry Lab
 */

// ==============================================================================
// State & Configuration
// ==============================================================================

// Data source: "live" (policy rollout via API) or "bic" (recorded BIC evaluation CSV; no API calls).
const SOURCE = (() => {
  try { return new URLSearchParams(window.location.search).get("source") === "bic" ? "bic" : "live"; }
  catch (e) { return "live"; }
})();

const STATE = {
  source: SOURCE,
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

  // Camera lerp targets (smoothed actual camera state)
  cameraPos: new THREE.Vector3(0, 45, 140),
  cameraTarget: new THREE.Vector3(0, 6, 0),
  cameraUp: new THREE.Vector3(0, 1, 0),
  cameraFov: 52,
  // Snapshot of the camera at the last mode switch (for the ease-in/out blend)
  camFromPos: new THREE.Vector3(0, 45, 140),
  camFromTarget: new THREE.Vector3(0, 6, 0),
  camFromUp: new THREE.Vector3(0, 1, 0),
  camFromFov: 52,

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
let prevCarWorldX = null, prevCarWorldZ = null; // for distance-based wheel spin
// Camera smoothing + orbit state
let orbitAngle = 0.0;
let orbitUserTookOver = false;
let chaseForwardX = 1.0, chaseForwardZ = 0.0; // car heading averaged over ~25 steps
let cameraTransT = 1.0; // transition timer since last mode switch (seconds); 1 = settled
const CAMERA_TRANSITION_S = 0.6;
const smoothstep01 = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
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
  scene.fog = new THREE.Fog(0xa9c4de, 1400.0, 4200.0); // near/far rescaled per-track in build3DTrack

  // 2. Camera Setup (46 deg FOV for sharp, natural perspective matching concept)
  camera = new THREE.PerspectiveCamera(46, width / height, 2.0, 9000);
  camera.position.set(0, 45, 140);
  camera.lookAt(0, 6, 0);

  // 3. WebGL Renderer
  renderer = new THREE.WebGLRenderer({
    canvas: threeCanvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(highPixelRatio());

  // PBR Tone Mapping & Color Management (concept: ACES @ exposure 0.62)
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.62;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.physicallyCorrectLights = false;

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
    // First drag in orbit hands control to the mouse (stops auto-rotation).
    controls.addEventListener("start", () => {
      if (STATE.activeCameraMode === "orbit") orbitUserTookOver = true;
    });
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

  // 6. Lighting & Physical Sky Environment
  setupLighting();
  setupSkyEnvironment();
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

// High-quality pixel ratio. BIC (large circuit) is set to 0.8 (native bench: High avg 59.9 fps);
// override for native A/B benchmarking with ?pr=<0.5..2>.
function highPixelRatio() {
  try {
    const q = parseFloat(new URLSearchParams(window.location.search).get("pr"));
    if (Number.isFinite(q) && q >= 0.5 && q <= 2) return q;
  } catch (e) {}
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return STATE.source === "bic" ? Math.min(dpr, 0.8) : dpr;
}

function onCanvasResize(width, height) {
  if (!width || !height) {
    width = window.innerWidth;
    height = window.innerHeight;
  }
  if (!renderer || !camera || width <= 0 || height <= 0) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  const pr = highPixelRatio();
  renderer.setPixelRatio(pr);
  renderer.setSize(width, height, false);

  if (composer) {
    composer.setPixelRatio(pr);
    composer.setSize(width, height);
  }
}

function setupLighting() {
  // Sky/ground hemisphere fill (concept: 0xcfe3ff over 0x4a5a32 @ 0.55)
  const hemi = new THREE.HemisphereLight(0xcfe3ff, 0x4a5a32, 0.55);
  scene.add(hemi);

  // Warm key sun with soft shadows. A tight frustum follows the car (see animate()).
  mainSun = new THREE.DirectionalLight(0xfff1dc, 3.1);
  mainSun.position.set(260, 440, 200);
  mainSun.castShadow = true;
  mainSun.shadow.mapSize.width = 2048;
  mainSun.shadow.mapSize.height = 2048;
  mainSun.shadow.camera.near = 10;
  mainSun.shadow.camera.far = 1400;

  // Tight shadow frustum around the car for crisp contact shadows
  mainSun.shadow.camera.left = -150;
  mainSun.shadow.camera.right = 150;
  mainSun.shadow.camera.top = 150;
  mainSun.shadow.camera.bottom = -150;
  mainSun.shadow.bias = -0.0004;
  mainSun.shadow.normalBias = 0.6;
  scene.add(mainSun);
  scene.add(mainSun.target);
}

// Shared sun direction (concept: elevation ~26 deg, azimuth 215 deg).
const SUN_DIR = new THREE.Vector3().setFromSphericalCoords(
  1,
  THREE.MathUtils.degToRad(90 - 26),
  THREE.MathUtils.degToRad(215)
);
let skyMesh = null;

// three.js Preetham Sky shader, inlined (vendored three r128 has no Sky addon).
function makeSkyMaterial() {
  const vert = `
    uniform vec3 sunPosition; uniform float rayleigh; uniform float turbidity; uniform float mieCoefficient; uniform vec3 up;
    varying vec3 vWorldPosition; varying vec3 vSunDirection; varying float vSunfade; varying vec3 vBetaR; varying vec3 vBetaM; varying float vSunE;
    const float e = 2.71828182845904523536028747135266249775724709369995957;
    const vec3 totalRayleigh = vec3( 5.804542996261093E-6, 1.3562911419845635E-5, 3.0265902468824876E-5 );
    const vec3 MieConst = vec3( 1.8399918514433978E14, 2.7798023919660528E14, 4.0790479543861094E14 );
    const float cutoffAngle = 1.6110731556870734; const float steepness = 1.5; const float EE = 1000.0;
    float sunIntensity( float zac ) { zac = clamp( zac, -1.0, 1.0 ); return EE * max( 0.0, 1.0 - pow( e, -( ( cutoffAngle - acos( zac ) ) / steepness ) ) ); }
    vec3 totalMie( float T ) { float c = ( 0.2 * T ) * 10E-18; return 0.434 * c * MieConst; }
    void main() {
      vec4 wp = modelMatrix * vec4( position, 1.0 ); vWorldPosition = wp.xyz;
      gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); gl_Position.z = gl_Position.w;
      vSunDirection = normalize( sunPosition ); vSunE = sunIntensity( dot( vSunDirection, up ) );
      vSunfade = 1.0 - clamp( 1.0 - exp( ( sunPosition.y / 450000.0 ) ), 0.0, 1.0 );
      float rc = rayleigh - ( 1.0 * ( 1.0 - vSunfade ) );
      vBetaR = totalRayleigh * rc; vBetaM = totalMie( turbidity ) * mieCoefficient;
    }`;
  const frag = `
    varying vec3 vWorldPosition; varying vec3 vSunDirection; varying float vSunfade; varying vec3 vBetaR; varying vec3 vBetaM; varying float vSunE;
    uniform float mieDirectionalG; uniform vec3 up; uniform float exposure;
    const float pi = 3.141592653589793238462643383279502884197169;
    const float rayleighZenithLength = 8.4E3; const float mieZenithLength = 1.25E3;
    const float sunAngularDiameterCos = 0.999956676946448443553574619906976478926848692873900859324;
    const float THREE_OVER_SIXTEENPI = 0.05968310365946075; const float ONE_OVER_FOURPI = 0.07957747154594767;
    float rayleighPhase( float c ) { return THREE_OVER_SIXTEENPI * ( 1.0 + pow( c, 2.0 ) ); }
    float hgPhase( float c, float g ) { float g2 = pow( g, 2.0 ); float inv = 1.0 / pow( 1.0 - 2.0 * g * c + g2, 1.5 ); return ONE_OVER_FOURPI * ( ( 1.0 - g2 ) * inv ); }
    void main() {
      vec3 direction = normalize( vWorldPosition - cameraPosition );
      float zenithAngle = acos( max( 0.0, dot( up, direction ) ) );
      float inv = 1.0 / ( cos( zenithAngle ) + 0.15 * pow( 93.885 - ( ( zenithAngle * 180.0 ) / pi ), -1.253 ) );
      float sR = rayleighZenithLength * inv; float sM = mieZenithLength * inv;
      vec3 Fex = exp( -( vBetaR * sR + vBetaM * sM ) );
      float cosTheta = dot( direction, vSunDirection );
      float rPhase = rayleighPhase( cosTheta * 0.5 + 0.5 ); vec3 betaRTheta = vBetaR * rPhase;
      float mPhase = hgPhase( cosTheta, mieDirectionalG ); vec3 betaMTheta = vBetaM * mPhase;
      vec3 Lin = pow( vSunE * ( ( betaRTheta + betaMTheta ) / ( vBetaR + vBetaM ) ) * ( 1.0 - Fex ), vec3( 1.5 ) );
      Lin *= mix( vec3( 1.0 ), pow( vSunE * ( ( betaRTheta + betaMTheta ) / ( vBetaR + vBetaM ) ) * Fex, vec3( 1.0 / 2.0 ) ), clamp( pow( 1.0 - dot( up, vSunDirection ), 5.0 ), 0.0, 1.0 ) );
      vec3 L0 = vec3( 0.1 ) * Fex;
      float sundisk = smoothstep( sunAngularDiameterCos, sunAngularDiameterCos + 0.00002, cosTheta );
      L0 += ( vSunE * 19000.0 * Fex ) * sundisk;
      vec3 texColor = ( Lin + L0 ) * 0.04 + vec3( 0.0, 0.0003, 0.00075 );
      vec3 retColor = pow( texColor, vec3( 1.0 / ( 1.2 + ( 1.2 * vSunfade ) ) ) );
      // Tone-map + encode to match the rest of the scene (raw ShaderMaterial isn't auto-processed).
      vec3 col = retColor * exposure;
      col = ( col * ( 2.51 * col + 0.03 ) ) / ( col * ( 2.43 * col + 0.59 ) + 0.14 ); // ACES
      col = clamp( col, 0.0, 1.0 );
      col = pow( col, vec3( 1.0 / 2.2 ) ); // sRGB OETF
      gl_FragColor = vec4( col, 1.0 );
    }`;
  return new THREE.ShaderMaterial({
    uniforms: {
      turbidity: { value: 2.6 },
      rayleigh: { value: 2.2 },
      mieCoefficient: { value: 0.004 },
      mieDirectionalG: { value: 0.82 },
      sunPosition: { value: SUN_DIR.clone().multiplyScalar(450000) },
      up: { value: new THREE.Vector3(0, 1, 0) },
      // Maps the Preetham HDR sky radiance into display range (separate from the
      // renderer's 0.62 scene exposure, since this raw ShaderMaterial self-tonemaps).
      // Tuned so the sky reads as the concept's pale blue (~#b3bcc0) vs ref_11s.
      exposure: { value: 0.30 },
    },
    vertexShader: vert,
    fragmentShader: frag,
    side: THREE.BackSide,
    depthWrite: false,
  });
}

function setupSkyEnvironment() {
  const skyMat = makeSkyMaterial();
  skyMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), skyMat);
  skyMesh.scale.setScalar(20000);
  scene.add(skyMesh);
  scene.background = null; // sky mesh is the background

  // PMREM environment baked from the sky for soft image-based lighting
  try {
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envScene = new THREE.Scene();
    const sky2 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), makeSkyMaterial());
    sky2.scale.setScalar(8000);
    envScene.add(sky2);
    hdrEnvMap = pmrem.fromScene(envScene, 0.02).texture;
    scene.environment = hdrEnvMap;
    pmrem.dispose();
  } catch (err) {
    console.warn("Sky PMREM env notice:", err);
  }
}

// Generic canvas texture helper (ported from concept main.js)
function canvasTex(w, h, draw, repeat) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  // Canvas is authored in sRGB; without this it is read as linear and washes out.
  t.encoding = THREE.sRGBEncoding;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (repeat) t.repeat.set(repeat[0], repeat[1]);
  t.anisotropy = renderer ? Math.min(renderer.capabilities.getMaxAnisotropy(), 8) : 4;
  return t;
}

// ==============================================================================
// Procedural Texture Generators (Lush Natural Lawn & Illuminated City Skyline)
// ==============================================================================

function createProceduralGrassTexture() {
  // Concept-style lawn: saturated base + high-frequency blade speckle (hides tiling).
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#56763a"; // concept base green
  ctx.fillRect(0, 0, 512, 512);

  for (let i = 0; i < 30000; i++) {
    const r = 60 + Math.random() * 40;
    const g = 100 + Math.random() * 60;
    const b = 35 + Math.random() * 25;
    ctx.fillStyle = `rgba(${r | 0},${g | 0},${b | 0},0.55)`;
    ctx.fillRect(Math.random() * 512, Math.random() * 512, 2, 2 + Math.random() * 3);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.encoding = THREE.sRGBEncoding;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(90, 90); // large plane; repeat tuned in build3DTrack-independent of normal maps
  tex.anisotropy = renderer ? Math.min(renderer.capabilities.getMaxAnisotropy(), 8) : 4;
  return tex;
}

// Low-frequency tonal variation layer to break up grass tiling (second scale).
function createGrassMacroTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#56763a";
  ctx.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 90; i++) {
    const x = Math.random() * 256, y = Math.random() * 256, r = 30 + Math.random() * 70;
    const light = Math.random() > 0.5;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, light ? "rgba(120,150,70,0.30)" : "rgba(40,60,28,0.30)");
    g.addColorStop(1, "rgba(86,118,58,0)");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.encoding = THREE.sRGBEncoding;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  return tex;
}

// Concept asphalt: grain + a rubbered-in dark racing line across the ribbon width (u).
function createProceduralAsphaltTexture() {
  return canvasTex(512, 512, (g, w, h) => {
    g.fillStyle = "#3f4249"; // dark charcoal asphalt: high contrast vs grass (concept)
    g.fillRect(0, 0, w, h);
    for (let i = 0; i < 26000; i++) {
      const v = (40 + Math.random() * 50) | 0;
      g.fillStyle = `rgba(${v},${v + 2},${v + 5},${0.35 + Math.random() * 0.5})`;
      g.fillRect(Math.random() * w, Math.random() * h, 1 + Math.random() * 2, 1 + Math.random() * 2);
    }
    // rubbered racing line darkening down the centre of the ribbon (u = x)
    const gr = g.createLinearGradient(0, 0, w, 0);
    gr.addColorStop(0.25, "rgba(0,0,0,0)");
    gr.addColorStop(0.5, "rgba(10,10,12,0.30)");
    gr.addColorStop(0.75, "rgba(0,0,0,0)");
    g.fillStyle = gr;
    g.fillRect(0, 0, w, h);
  }, [1, 1]);
}

function createProceduralBuildingTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  // Pale blue-grey facade so the distant skyline reads as a soft hazy backdrop
  ctx.fillStyle = "#7c8794";
  ctx.fillRect(0, 0, 256, 512);

  // Window grid: cool/warm lit panes over a mid blue-grey (concept winTex)
  const rows = 23;
  const cols = 13;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = Math.random();
      ctx.fillStyle = v > 0.82 ? "#e9e2c4" : v > 0.4 ? "#3c4d61" : "#55697f";
      ctx.fillRect(c * 20 + 8, r * 22 + 8, 13, 14);
    }
  }
  ctx.globalAlpha = 1.0;

  const tex = new THREE.CanvasTexture(canvas);
  tex.encoding = THREE.sRGBEncoding;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 6);
  // Mipmaps + anisotropy so the window grid doesn't alias/pixelate on near faces.
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = renderer ? Math.min(renderer.capabilities.getMaxAnisotropy(), 8) : 4;
  return tex;
}

let proceduralGrassTex = null;
let proceduralGrassMacroTex = null;
let proceduralBuildingTex = null;
let proceduralAsphaltTex = null;
let proceduralCrowdTex = null;

// Speckled crowd texture for the grandstand (ported from concept main.js).
function createProceduralCrowdTexture() {
  return canvasTex(512, 64, (g, w, h) => {
    g.fillStyle = "#2b2f36";
    g.fillRect(0, 0, w, h);
    const cols = ["#e64a3b", "#f2c14e", "#3b82f6", "#f5f5f5", "#22c55e", "#111", "#ff7a1a"];
    for (let i = 0; i < 2600; i++) {
      g.fillStyle = cols[(Math.random() * cols.length) | 0];
      g.fillRect(Math.random() * w, Math.random() * h, 3, 4);
    }
  }, [6, 1]);
}

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
  proceduralGrassMacroTex = createGrassMacroTexture();
  proceduralAsphaltTex = createProceduralAsphaltTexture();
  proceduralBuildingTex = createProceduralBuildingTexture();
  proceduralCrowdTex = createProceduralCrowdTexture();
  pbrTextures.groundDiff = proceduralGrassTex;
}

function initEffectComposer(width, height) {
  if (!THREE.EffectComposer || !THREE.RenderPass) return;

  const pr = highPixelRatio();
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

  // 1a. Darker tarmac runoff apron (slightly wider than the corridor, just above ground)
  const runoffMat = new THREE.MeshStandardMaterial({
    color: 0x55595f,
    roughness: 0.95,
    metalness: 0.0,
    envMapIntensity: 0.4,
    side: THREE.DoubleSide,
  });
  const runoffMesh = buildTrackRibbon(innerPts, outerPts, 7.0, 0.30, runoffMat, 100);
  if (runoffMesh) {
    runoffMesh.receiveShadow = true;
    trackGroup.add(runoffMesh);
  }

  // 1b. Asphalt road surface with rubbered-in racing line (concept canvas texture)
  const roadMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: proceduralAsphaltTex || pbrTextures.asphaltDiff,
    roughness: 0.82,
    metalness: 0.0,
    envMapIntensity: 0.5,
    side: THREE.DoubleSide,
  });
  roadMesh = buildTrackRibbon(innerPts, outerPts, 0.0, 0.60, roadMat, 24);
  if (roadMesh) {
    roadMesh.receiveShadow = true;
    trackGroup.add(roadMesh);
  }

  // 2. Continuous Solid White Edge Markings (Inner & Outer boundaries)
  buildTrackEdgeLines(innerPts, outerPts);

  // 3. (No painted centreline — concept has none; the rubbered racing line reads instead.)

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

  // 6. Start / finish checker line + gantry with glowing panel at the real start pose
  buildStartGantry(trackData);


  // 7. Ground: huge grass plane receiving shadows (base), plus a low-frequency
  //    tonal overlay to break up texture tiling (concept "second scale").
  const GROUND = Math.max(16000, (Math.hypot(bounds.max_x - bounds.min_x, bounds.max_y - bounds.min_y) || 0) * 2.2);
  const groundGeo = new THREE.PlaneGeometry(GROUND, GROUND);
  groundGeo.rotateX(-Math.PI / 2);
  if (proceduralGrassTex) proceduralGrassTex.repeat.set(120 * GROUND / 16000, 120 * GROUND / 16000);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: proceduralGrassTex || pbrTextures.groundDiff,
    roughness: 1.0,
    metalness: 0.0,
    envMapIntensity: 0.6,
  });
  groundPlaneMesh = new THREE.Mesh(groundGeo, groundMat);
  groundPlaneMesh.position.y = -0.40;
  groundPlaneMesh.receiveShadow = true;
  trackGroup.add(groundPlaneMesh);

  if (proceduralGrassMacroTex) {
    const macroGeo = new THREE.PlaneGeometry(GROUND, GROUND);
    macroGeo.rotateX(-Math.PI / 2);
    const macroMat = new THREE.MeshBasicMaterial({
      map: proceduralGrassMacroTex,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
    });
    const macroMesh = new THREE.Mesh(macroGeo, macroMat);
    macroMesh.position.y = -0.38;
    trackGroup.add(macroMesh);
  }

  // Rescale fog to the real track size (concept Fog(0xa9c4de, 3000, 11000)).
  const spanX = bounds.max_x - bounds.min_x;
  const spanY = bounds.max_y - bounds.min_y;
  const diag = Math.hypot(spanX, spanY) || 800;
  if (scene.fog) {
    scene.fog.near = diag * 1.4;
    scene.fog.far = diag * 6.5;
  }

  // 8. Procedural Props: trees + skyline (into propsGroup)
  buildEnvironmentProps(trackData);
  // 8b. Grandstand + pit along the start straight (also into propsGroup, after the
  //     clear in buildEnvironmentProps, so the prop-clearance test covers them too)
  buildGrandstand(trackData);

  clearTrajectoryTrail();
}

// Build a quad-strip ribbon between inner/outer boundary point lists (concept ribbon()).
// `widen` expands both edges outward across the track; UV u spans width, v = length/vDivisor.
function buildTrackRibbon(innerPts, outerPts, widen, y, material, vDivisor) {
  const n = Math.min(innerPts.length, outerPts.length);
  if (n < 4) return null;
  const geo = new THREE.BufferGeometry();
  const pos = [];
  const uv = [];
  const idx = [];
  let cum = 0.0;
  let prevInW = null;

  for (let i = 0; i < n; i++) {
    const pIn = innerPts[i];
    const pOut = outerPts[i];
    if (!isValidVector2(pIn) || !isValidVector2(pOut)) continue;

    let inW = envToWorld(pIn[0], pIn[1]);
    let outW = envToWorld(pOut[0], pOut[1]);

    if (widen && widen !== 0) {
      const ax = outW.x - inW.x;
      const az = outW.z - inW.z;
      const al = Math.hypot(ax, az) || 1.0;
      const ux = ax / al;
      const uz = az / al;
      inW = { x: inW.x - ux * widen, z: inW.z - uz * widen };
      outW = { x: outW.x + ux * widen, z: outW.z + uz * widen };
    }

    if (prevInW) cum += Math.hypot(inW.x - prevInW.x, inW.z - prevInW.z) / (vDivisor || 24);
    prevInW = inW;

    pos.push(inW.x, y, inW.z);
    pos.push(outW.x, y, outW.z);
    uv.push(0.0, cum, 1.0, cum);

    const next = (i + 1) % n;
    const i0 = i * 2, i1 = i * 2 + 1, i2 = next * 2, i3 = next * 2 + 1;
    idx.push(i0, i1, i2, i1, i3, i2);
  }

  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  // Flatten normals to up so the flat ribbon lights evenly
  const nr = geo.attributes.normal;
  for (let i = 0; i < nr.count; i++) nr.setXYZ(i, 0, 1, 0);
  nr.needsUpdate = true;
  return new THREE.Mesh(geo, material);
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
    // Concept curb red #d8231d (low blue) — avoids the magenta cast under bloom
    const col = isRed ? [0.847, 0.137, 0.113] : [0.96, 0.96, 0.98];
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
    roughness: 0.6,
    metalness: 0.0,
    envMapIntensity: 0.3,
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

function buildStartGantry(trackData) {
  const inner = trackData.inner_boundary || [];
  const outer = trackData.outer_boundary || [];
  if (inner.length < 1 || outer.length < 1) return;
  if (!isValidVector2(inner[0]) || !isValidVector2(outer[0])) return;

  const inW = envToWorld(inner[0][0], inner[0][1]);
  const outW = envToWorld(outer[0][0], outer[0][1]);
  const cx = (inW.x + outW.x) / 2;
  const cz = (inW.z + outW.z) / 2;

  // across = edge-to-edge (start line), tangent = along the track
  let ax = outW.x - inW.x, az = outW.z - inW.z;
  const aLen = Math.hypot(ax, az) || 1; ax /= aLen; az /= aLen;
  const tx = -az, tz = ax; // perpendicular (track forward)
  const halfW = aLen / 2;
  const acrossAngle = Math.atan2(az, ax);

  // Checkered start/finish line across the corridor
  const checkTex = canvasTex(256, 32, (g, w, h) => {
    for (let x = 0; x < 16; x++) for (let y = 0; y < 2; y++) {
      g.fillStyle = (x + y) % 2 ? "#111" : "#f5f5f5";
      g.fillRect(x * 16, y * 16, 16, 16);
    }
  }, [1, 1]);
  const checkGeo = new THREE.PlaneGeometry(aLen + 2, 9);
  checkGeo.rotateX(-Math.PI / 2);
  const checkMesh = new THREE.Mesh(checkGeo, new THREE.MeshStandardMaterial({ map: checkTex, roughness: 0.7 }));
  checkMesh.position.set(cx, 0.9, cz);
  checkMesh.rotation.y = -acrossAngle;
  checkMesh.receiveShadow = true;
  trackGroup.add(checkMesh);

  // Gantry: two posts just outside the corridor, a beam, and a glowing cyan panel
  const gm = new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.5, metalness: 0.6 });
  const postH = 46;
  for (const s of [-1, 1]) {
    const px = cx + ax * (halfW + 7) * s;
    const pz = cz + az * (halfW + 7) * s;
    const post = new THREE.Mesh(new THREE.BoxGeometry(4, postH, 4), gm);
    post.position.set(px, postH / 2, pz);
    post.castShadow = true;
    trackGroup.add(post);
  }
  const beam = new THREE.Mesh(new THREE.BoxGeometry(5, 9, aLen + 22), gm);
  beam.position.set(cx, postH, cz);
  beam.rotation.y = -Math.atan2(tz, tx);
  beam.castShadow = true;
  trackGroup.add(beam);

  const panel = new THREE.Mesh(
    new THREE.BoxGeometry(5.5, 5.5, aLen + 2),
    new THREE.MeshStandardMaterial({ color: 0x0a0d12, emissive: 0x18d4ff, emissiveIntensity: 1.4 })
  );
  panel.position.set(cx, postH - 0.5, cz);
  panel.rotation.y = -Math.atan2(tz, tx);
  trackGroup.add(panel);
}

// Grandstand + pit building along the start straight, built from the real start pose.
function buildGrandstand(trackData) {
  const inner = trackData.inner_boundary || [];
  const outer = trackData.outer_boundary || [];
  const bounds = trackData.bounds || { min_x: 100, max_x: 700, min_y: 100, max_y: 500 };
  if (inner.length < 1 || outer.length < 1) return;
  if (!isValidVector2(inner[0]) || !isValidVector2(outer[0])) return;

  const startInner = envToWorld(inner[0][0], inner[0][1]);
  const startOuter = envToWorld(outer[0][0], outer[0][1]);

  // out = the true outward perpendicular (inner -> outer across the track), NOT the
  // radial from the bounds centre (which can run nearly tangent at the start).
  let ox = startOuter.x - startInner.x, oz = startOuter.z - startInner.z;
  const ol = Math.hypot(ox, oz) || 1; ox /= ol; oz /= ol;
  const fx = -oz, fz = ox; // tangent (along the track)
  const ry = Math.atan2(-fz, fx); // aligns a box's local +X with fwd

  const STAND_LEN = 170; // compact so it doesn't wrap into the curved track

  // --- Pit building: long white block + cyan light band, just outside the wall ---
  const pit = new THREE.Mesh(
    new THREE.BoxGeometry(STAND_LEN - 20, 18, 14),
    new THREE.MeshStandardMaterial({ color: 0xf2f4f6, roughness: 0.5, metalness: 0.05 })
  );
  const pitX = startOuter.x + ox * 90, pitZ = startOuter.z + oz * 90; // ~125px from centerline
  pit.position.set(pitX, 9, pitZ);
  pit.rotation.y = ry;
  pit.castShadow = true; pit.receiveShadow = true;
  propsGroup.add(pit);

  const band = new THREE.Mesh(
    new THREE.BoxGeometry(STAND_LEN - 18, 3, 15),
    new THREE.MeshStandardMaterial({ color: 0x0a0d12, emissive: 0x19c9ff, emissiveIntensity: 0.9 })
  );
  band.position.set(pitX, 19, pitZ);
  band.rotation.y = ry;
  propsGroup.add(band);

  // --- Grandstand: tiered crowd rows + roof, further out ---
  const standBase = { x: startOuter.x + ox * 142, z: startOuter.z + oz * 142 };
  const crowdMat = new THREE.MeshStandardMaterial({ map: proceduralCrowdTex, roughness: 0.9 });
  const concreteMat = new THREE.MeshStandardMaterial({ color: 0xd9dde2, roughness: 0.7 });
  const ROWS = 6;
  let topY = 0;
  for (let r = 0; r < ROWS; r++) {
    const h = 12 + r * 10;
    topY = Math.max(topY, h);
    const seg = new THREE.Mesh(new THREE.BoxGeometry(STAND_LEN, h, 15), [concreteMat, concreteMat, concreteMat, concreteMat, crowdMat, concreteMat]);
    seg.position.set(standBase.x + ox * (r * 13), h / 2, standBase.z + oz * (r * 13));
    seg.rotation.y = ry;
    seg.castShadow = true; seg.receiveShadow = true;
    propsGroup.add(seg);
  }
  // Roof slab over the stand
  const roof = new THREE.Mesh(
    new THREE.BoxGeometry(STAND_LEN + 20, 4, 95),
    new THREE.MeshStandardMaterial({ color: 0xeef1f4, roughness: 0.4, metalness: 0.3 })
  );
  const roofOut = (ROWS * 13) / 2;
  roof.position.set(standBase.x + ox * roofOut, topY + 26, standBase.z + oz * roofOut);
  roof.rotation.y = ry;
  roof.castShadow = true;
  propsGroup.add(roof);
  // Roof support pillars at the front
  const pillarMat = concreteMat;
  for (let i = -1; i <= 1; i++) {
    const px = standBase.x + fx * (i * STAND_LEN * 0.1);
    const pz = standBase.z + fz * (i * STAND_LEN * 0.1);
    const pil = new THREE.Mesh(new THREE.BoxGeometry(4, topY + 26, 4), pillarMat);
    pil.position.set(px, (topY + 26) / 2, pz);
    pil.castShadow = true;
    propsGroup.add(pil);
  }
}

// Coarse distance from an env-space point to the nearest centerline vertex (px).
function trackDistEnv(centerPts, x, y) {
  let best = 1e9;
  for (let i = 0; i < centerPts.length; i += 3) {
    const p = centerPts[i];
    const dx = p[0] - x, dy = p[1] - y;
    const d = dx * dx + dy * dy;
    if (d < best) best = d;
  }
  return Math.sqrt(best);
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

  const bounds = trackData.bounds || { min_x: 100, max_x: 700, min_y: 100, max_y: 500 };
  const ecx = (bounds.min_x + bounds.max_x) / 2;
  const ecy = (bounds.min_y + bounds.max_y) / 2;

  const dummy = new THREE.Object3D();
  const col = new THREE.Color();

  // Deterministic RNG (seeded) so scenery is stable across reloads.
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

  // Shared trunk + two foliage silhouettes (pine cone / round). Foliage base white
  // so the per-instance green tint (setColorAt + instanceColor) shows directly.
  const trunkGeo = new THREE.CylinderGeometry(0.9, 1.3, 9.0, 6);
  trunkGeo.translate(0, 4.5, 0);
  const pineGeo = new THREE.ConeGeometry(7.0, 15.0, 7);
  pineGeo.translate(0, 13.0, 0);
  const roundGeo = new THREE.IcosahedronGeometry(8.0, 1);
  roundGeo.translate(0, 12.5, 0);

  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x5b4330, roughness: 1.0, envMapIntensity: 0.4 });
  const leafMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, flatShading: true, envMapIntensity: 0.5 });

  // Gather tree placements OUTSIDE the corridor (>=150 px from centerline),
  // scattered outward from the outer boundary so the infield stays open.
  const trees = []; // { x, z, scale, rotY, round }
  const LARGE = !!trackData.large_circuit; // BIC: ~5 km lap, 5,131 boundary points
  const MAX_TREES = LARGE ? 1500 : 620;
  const stride = LARGE ? Math.max(1, Math.floor(n / (MAX_TREES / 3))) : 1;
  for (let i = 0; i < n && trees.length < MAX_TREES; i += stride) {
    const pt = outerPts[i];
    const nextPt = outerPts[(i + stride) % n];
    if (!isValidVector2(pt) || !isValidVector2(nextPt)) continue;

    const dx = nextPt[0] - pt[0];
    const dy = nextPt[1] - pt[1];
    const len = Math.hypot(dx, dy) || 1.0;
    let nx = -dy / len;
    let ny = dx / len;
    // Force the normal to point away from the track centre (outward).
    if ((pt[0] - ecx) * nx + (pt[1] - ecy) * ny < 0) { nx = -nx; ny = -ny; }

    const perTree = 3;
    for (let k = 0; k < perTree && trees.length < MAX_TREES; k++) {
      const offset = 45 + rnd() * 430;
      const jitter = (rnd() - 0.5) * len * 2.2;
      const tx = pt[0] + nx * offset + (dx / len) * jitter;
      const ty = pt[1] + ny * offset + (dy / len) * jitter;
      if (trackDistEnv(centerPts, tx, ty) < 150) continue; // keep off the track
      const w = envToWorld(tx, ty);
      trees.push({
        x: w.x,
        z: w.z,
        scale: 0.7 + rnd() * 0.7,
        rotY: rnd() * Math.PI * 2,
        round: rnd() < 0.55,
      });
    }
  }

  if (trees.length > 0) {
    const count = trees.length;
    const trunkMesh = new THREE.InstancedMesh(trunkGeo, trunkMat, count);
    const pineMesh = new THREE.InstancedMesh(pineGeo, leafMat, count);
    const roundMesh = new THREE.InstancedMesh(roundGeo, leafMat, count);
    [trunkMesh, pineMesh, roundMesh].forEach((m) => { m.castShadow = true; m.receiveShadow = true; });

    let pineN = 0, roundN = 0;
    trees.forEach((t) => {
      dummy.position.set(t.x, 0, t.z);
      dummy.rotation.set(0, t.rotY, 0);
      dummy.scale.set(t.scale, t.scale, t.scale);
      dummy.updateMatrix();
      // Green HSL ramp per instance (concept main.js)
      col.setHSL(0.24 + rnd() * 0.08, 0.45 + rnd() * 0.2, 0.22 + rnd() * 0.12);
      // Trunk follows the same transform slot index as its foliage mesh
      if (t.round) {
        roundMesh.setMatrixAt(roundN, dummy.matrix);
        roundMesh.setColorAt(roundN, col);
        trunkMesh.setMatrixAt(pineN + roundN, dummy.matrix);
        roundN++;
      } else {
        pineMesh.setMatrixAt(pineN, dummy.matrix);
        pineMesh.setColorAt(pineN, col);
        trunkMesh.setMatrixAt(pineN + roundN, dummy.matrix);
        pineN++;
      }
    });
    pineMesh.count = pineN;
    roundMesh.count = roundN;
    trunkMesh.count = count;

    trunkMesh.instanceMatrix.needsUpdate = true;
    pineMesh.instanceMatrix.needsUpdate = true;
    roundMesh.instanceMatrix.needsUpdate = true;
    if (pineMesh.instanceColor) pineMesh.instanceColor.needsUpdate = true;
    if (roundMesh.instanceColor) roundMesh.instanceColor.needsUpdate = true;

    propsGroup.add(trunkMesh, pineMesh, roundMesh);
  }

  // Distant City Skyline Skyscrapers on Horizon (depth / backdrop)
  buildDistantCitySkyline(trackData);
}

function buildDistantCitySkyline(trackData) {
  // Pale, self-lit blue-grey towers so they read as a soft hazy skyline even when
  // back-lit; pushed far out and fogged, they sit low behind the tree line.
  const buildingMat = new THREE.MeshStandardMaterial({
    color: 0xb6c2d1,
    map: proceduralBuildingTex,
    emissive: 0x9fb0c4,
    emissiveMap: proceduralBuildingTex,
    emissiveIntensity: 0.85,
    metalness: 0.10,
    roughness: 0.65,
    envMapIntensity: 0.6,
  });

  const LARGE = !!(trackData && trackData.large_circuit);
  const COUNT = LARGE ? 52 : 80; // BIC: 160 -> 52 (-67%) to cut High-quality cost
  const dummy = new THREE.Object3D();
  const towerGeo = new THREE.BoxGeometry(1, 1, 1);
  const towerMesh = new THREE.InstancedMesh(towerGeo, buildingMat, COUNT);
  const crownGeo = new THREE.BoxGeometry(1, 1, 1);
  const crownMesh = new THREE.InstancedMesh(crownGeo, buildingMat, COUNT);
  towerMesh.castShadow = crownMesh.castShadow = false;
  towerMesh.receiveShadow = crownMesh.receiveShadow = false;

  // Seeded RNG for a stable skyline.
  let s = 1337;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;

  for (let i = 0; i < COUNT; i++) {
    // Bias the cluster toward the far side, with some all around.
    let x, z;
    let sizeK = 1;
    if (LARGE) {
      // Large circuit: towers ring the lap, 1.2-2.4 km outside random centerline points
      // (fixed-seed RNG), and are rejected if they would crowd any other part of the track.
      const cl = trackData.centerline;
      let ok = false;
      for (let tries = 0; tries < 40 && !ok; tries++) {
        const ci = Math.floor(rnd() * cl.length);
        const a = cl[ci], b = cl[(ci + 7) % cl.length];
        const dl = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
        let nx = -(b[1] - a[1]) / dl, ny = (b[0] - a[0]) / dl;
        const bnd = trackData.bounds;
        const mx = (bnd.min_x + bnd.max_x) / 2, my = (bnd.min_y + bnd.max_y) / 2;
        if ((a[0] - mx) * nx + (a[1] - my) * ny < 0) { nx = -nx; ny = -ny; }
        const off = 1500 + rnd() * 1300;
        const ex = a[0] + nx * off, ey = a[1] + ny * off;
        if (trackDistEnv(cl, ex, ey) < 900) continue;
        const w = envToWorld(ex, ey);
        x = w.x; z = w.z; ok = true;
      }
      if (!ok) { x = 1e5; z = 1e5; } // parked far away; never near the track
      sizeK = 1.0;
    } else {
      const angle = (i < 56)
        ? (-2.4 + rnd() * 2.2)              // back arc
        : (rnd() * Math.PI * 2);           // scattered
      const dist = 3300 + rnd() * 2500;    // 3300–5800 world units (far, fogged, behind trees)
      x = Math.cos(angle) * dist;
      z = Math.sin(angle) * dist;
    }

    const bW = (110 + rnd() * 200) * sizeK;
    const bD = (110 + rnd() * 200) * sizeK;
    const bH = (120 + rnd() * 460) * sizeK; // 120–580 (lower, so they sit near the tree line)

    dummy.position.set(x, bH / 2 - 2, z);
    dummy.rotation.set(0, rnd() * 0.6, 0);
    dummy.scale.set(bW, bH, bD);
    dummy.updateMatrix();
    towerMesh.setMatrixAt(i, dummy.matrix);

    const cW = bW * 0.6, cD = bD * 0.6, cH = bH * 0.3;
    dummy.position.set(x, bH + cH / 2 - 2, z);
    dummy.scale.set(cW, cH, cD);
    dummy.updateMatrix();
    crownMesh.setMatrixAt(i, dummy.matrix);
  }

  towerMesh.instanceMatrix.needsUpdate = true;
  crownMesh.instanceMatrix.needsUpdate = true;
  propsGroup.add(towerMesh, crownMesh);
}

// ==============================================================================
// 3D Racing Vehicle (Unbranded Stylized Sports Coupe Matching Concept Video)
// ==============================================================================

// Env physics constants (see env.car / world.js): wheelbase 50 px, nominal width 30 px.
const CAR_WHEELBASE_PX = 50.0;
const CAR_WIDTH_PX = 30.0;
const CAR_WHEEL_RADIUS = 9.0;
const CAR_TRACK_Z = 17.0; // left/right wheel offset
const ROAD_TOP_Y = 0.60;  // asphalt ribbon top (cars sit here)

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
  // Clear previously added wheels / footprint from carGroup
  [frontLeftSteerPivot, frontRightSteerPivot, carPhysicsFootprint].forEach((o) => {
    if (o && o.parent) o.parent.remove(o);
  });
  carGroup.children
    .filter((c) => c.userData && c.userData.isWheel)
    .forEach((c) => carGroup.remove(c));

  wheelSpinMeshes = [];
  rearBrakeLights = [];
  carEmitterNodes = []; // rays originate from the car centre (fallback) — no visible emitters

  // ---- Materials (concept main.js) ----
  const paint = new THREE.MeshPhysicalMaterial({
    color: 0xb3150e, metalness: 0.15, roughness: 0.42,
    clearcoat: 0.6, clearcoatRoughness: 0.1, envMapIntensity: 0.4,
  });
  const black = new THREE.MeshStandardMaterial({ color: 0x0d0e10, roughness: 0.55, metalness: 0.3, envMapIntensity: 0.3 });
  const glass = new THREE.MeshPhysicalMaterial({ color: 0x0a0f16, metalness: 0.2, roughness: 0.04, clearcoat: 1.0, envMapIntensity: 0.6 });

  // ---- Smooth extruded body (side profile x,y extruded across width z) ----
  const sh = new THREE.Shape();
  [[-41, 4], [-42, 12], [-35, 15.5], [-20, 16.5], [-4, 16.5], [16, 15.5], [33, 12.5], [41.5, 8.5], [41, 3.5]]
    .forEach((p, i) => (i ? sh.lineTo(p[0], p[1]) : sh.moveTo(p[0], p[1])));
  sh.closePath();
  const bodyGeo = new THREE.ExtrudeGeometry(sh, { depth: CAR_WIDTH_PX, bevelEnabled: true, bevelThickness: 3, bevelSize: 2.5, bevelSegments: 5, curveSegments: 10 });
  bodyGeo.translate(0, 0, -CAR_WIDTH_PX / 2);
  const body = new THREE.Mesh(bodyGeo, paint);
  body.castShadow = true; body.receiveShadow = true;
  carBodyGroup.add(body);

  // ---- Glass canopy (lower + narrower so it doesn't read as a tall box) ----
  const cs = new THREE.Shape();
  [[-21, 15.5], [-10, 21], [5, 21], [18, 15]].forEach((p, i) => (i ? cs.lineTo(p[0], p[1]) : cs.moveTo(p[0], p[1])));
  cs.closePath();
  const cabGeo = new THREE.ExtrudeGeometry(cs, { depth: 17, bevelEnabled: true, bevelThickness: 2, bevelSize: 1.5, bevelSegments: 4 });
  cabGeo.translate(0, 0, -8.5);
  const cab = new THREE.Mesh(cabGeo, glass);
  cab.castShadow = true;
  carBodyGroup.add(cab);

  const roof = new THREE.Mesh(new THREE.BoxGeometry(11, 1.0, 15), paint);
  roof.position.set(-2, 22.0, 0);
  carBodyGroup.add(roof);

  // ---- Rear wing (blended into the deck) + struts, diffuser, splitter, skirts ----
  // A sloped rear deck fairing bridges the cabin down to the wing.
  const deck = new THREE.Mesh(new THREE.BoxGeometry(16, 1.4, 26), paint);
  deck.position.set(-31, 16.5, 0); deck.rotation.z = 0.18; deck.castShadow = true;
  carBodyGroup.add(deck);
  const wing = new THREE.Mesh(new THREE.BoxGeometry(7, 1.1, 30), black);
  wing.position.set(-39, 18.5, 0); wing.rotation.z = 0.06; wing.castShadow = true;
  carBodyGroup.add(wing);
  for (const s of [-1, 1]) {
    const strut = new THREE.Mesh(new THREE.BoxGeometry(2.4, 4.0, 1.2), black);
    strut.position.set(-38, 16.5, 8 * s);
    carBodyGroup.add(strut);
  }
  const diff = new THREE.Mesh(new THREE.BoxGeometry(4, 3, 26), black);
  diff.position.set(-42.5, 3.5, 0);
  carBodyGroup.add(diff);
  const split = new THREE.Mesh(new THREE.BoxGeometry(6, 1.2, 32), black);
  split.position.set(41, 2.4, 0);
  carBodyGroup.add(split);
  for (const s of [-1, 1]) {
    const skirt = new THREE.Mesh(new THREE.BoxGeometry(46, 3, 1.5), black);
    skirt.position.set(0, 3.5, 17.6 * s);
    carBodyGroup.add(skirt);
  }

  // ---- Emissive tail lights (brighten on braking) + head lights ----
  for (const s of [-1, 1]) {
    const tl = new THREE.Mesh(
      new THREE.BoxGeometry(1, 2.2, 10),
      new THREE.MeshStandardMaterial({ color: 0x300000, emissive: 0xff1a12, emissiveIntensity: 1.2 })
    );
    tl.position.set(-44.2, 12.3, 10 * s);
    carBodyGroup.add(tl);
    rearBrakeLights.push(tl);

    const hl = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1.8, 8),
      new THREE.MeshStandardMaterial({ color: 0x333333, emissive: 0xeaf6ff, emissiveIntensity: 2.0 })
    );
    hl.position.set(43, 9.5, 11 * s); hl.rotation.y = -0.25 * s;
    carBodyGroup.add(hl);
  }

  // ---- Wheels: wheelbase 50 px (axles at +/-25), radius 9 ----
  const tireMat = new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.9, envMapIntensity: 0.2 });
  const rimMat = new THREE.MeshStandardMaterial({ color: 0xb8bec6, metalness: 1.0, roughness: 0.25, envMapIntensity: 0.6 });
  const calMat = new THREE.MeshStandardMaterial({ color: 0xffc400, roughness: 0.4 });

  function createWheel() {
    const group = new THREE.Group();
    const spin = new THREE.Group();
    const tg = new THREE.CylinderGeometry(CAR_WHEEL_RADIUS, CAR_WHEEL_RADIUS, 7, 28); tg.rotateX(Math.PI / 2);
    const tire = new THREE.Mesh(tg, tireMat); tire.castShadow = true; spin.add(tire);
    const rg = new THREE.CylinderGeometry(6.2, 6.2, 7.4, 24); rg.rotateX(Math.PI / 2);
    spin.add(new THREE.Mesh(rg, rimMat));
    for (let i = 0; i < 5; i++) {
      const sp = new THREE.Mesh(new THREE.BoxGeometry(1.8, 11, 8), black);
      sp.rotation.z = (i * Math.PI * 2) / 5; spin.add(sp);
    }
    group.add(spin);
    const cal = new THREE.Mesh(new THREE.BoxGeometry(4, 4, 8.2), calMat);
    cal.position.set(3, 2, 0); group.add(cal); // static caliper
    return { group, spin };
  }

  // [x, z, isFront]
  const wheelDefs = [
    [CAR_WHEELBASE_PX / 2, CAR_TRACK_Z, true],
    [CAR_WHEELBASE_PX / 2, -CAR_TRACK_Z, true],
    [-CAR_WHEELBASE_PX / 2, CAR_TRACK_Z, false],
    [-CAR_WHEELBASE_PX / 2, -CAR_TRACK_Z, false],
  ];
  const fronts = [];
  for (const [x, z, isFront] of wheelDefs) {
    const piv = new THREE.Group();
    piv.position.set(x, CAR_WHEEL_RADIUS, z);
    piv.userData.isWheel = true;
    const w = createWheel();
    piv.add(w.group);
    carGroup.add(piv);
    wheelSpinMeshes.push(w.spin);
    if (isFront) fronts.push(piv);
  }
  frontLeftSteerPivot = fronts[0];
  frontRightSteerPivot = fronts[1];

  // ---- Debug-only physics footprint wireframe: 30 (z) x 50 (x) px ----
  // Drawn over the model (depthTest off) so the collision box is always visible in debug.
  const fpGeo = new THREE.BoxGeometry(CAR_WHEELBASE_PX, 14, CAR_WIDTH_PX);
  const fpEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(fpGeo),
    new THREE.LineBasicMaterial({ color: 0x33ff88, depthTest: false, transparent: true })
  );
  fpEdges.position.set(0, 7, 0);
  fpEdges.renderOrder = 999;
  fpEdges.visible = STATE.debugMode;
  carPhysicsFootprint = fpEdges;
  carGroup.add(fpEdges);
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

const RAY_MAX = 200; // env obs_ray_max_dist
let sweepMesh = null;
const BEAM_Y = 7.0;

// Real-distance colour tiers (concept): red <45, amber <90, cyan beyond.
function rayCol(d) {
  return d < 45 ? 0xff4d4d : d < 90 ? 0xffb02e : 0x67e3ff;
}

function buildLidarBeams() {
  while (lidarGroup.children.length > 0) {
    const obj = lidarGroup.children[0];
    lidarGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
  rayBeamObjects = [];

  for (let i = 0; i < 5; i++) {
    // Thin bright core
    const core = new THREE.Mesh(
      new THREE.CylinderGeometry(0.5, 0.5, 1, 8),
      new THREE.MeshBasicMaterial({ color: 0x67e3ff, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    // Soft wider additive glow
    const glow = new THREE.Mesh(
      new THREE.CylinderGeometry(2.6, 2.6, 1, 8),
      new THREE.MeshBasicMaterial({ color: 0x67e3ff, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    // White hit sphere
    const hit = new THREE.Mesh(
      new THREE.SphereGeometry(2.3, 14, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    // Flat ground ring in the beam colour
    const ringGeo = new THREE.RingGeometry(3.6, 5.6, 28);
    ringGeo.rotateX(-Math.PI / 2);
    const ring = new THREE.Mesh(
      ringGeo,
      new THREE.MeshBasicMaterial({ color: 0x67e3ff, transparent: true, opacity: 0.85, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    lidarGroup.add(core, glow, hit, ring);
    rayBeamObjects.push({ core, glow, hit, ring });
  }

  // Optional illustrative scan sweep (rotating translucent purple fan) — clearly distinct
  // from the measured beams (wide flat wedge, not a thin line).
  sweepMesh = new THREE.Mesh(
    new THREE.CircleGeometry(RAY_MAX, 36, 0, 0.32),
    new THREE.MeshBasicMaterial({ color: 0xd678ff, transparent: true, opacity: 0.16, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false })
  );
  sweepMesh.rotation.x = -Math.PI / 2;
  sweepMesh.visible = false;
  lidarGroup.add(sweepMesh);
}

const _tmpV3 = new THREE.Vector3();
function updateLidarVisuals(rays) {
  if (!rays || rays.length < 5 || !carGroup || rayBeamObjects.length < 5) return;
  const mode = STATE.activeCameraMode;
  const upY = new THREE.Vector3(0, 1, 0);
  const labelParts = [];
  const container = document.getElementById("canvasContainer");
  const W = container ? container.clientWidth : window.innerWidth;
  const H = container ? container.clientHeight : window.innerHeight;
  camera.updateMatrixWorld();

  for (let i = 0; i < 5; i++) {
    const ray = rays[i];
    const o = rayBeamObjects[i];
    if (!ray || !isValidVector2(ray.start) || !isValidVector2(ray.end) || !isValidNumber(ray.distance)) continue;

    const sW = envToWorld(ray.start[0], ray.start[1]);
    const hW = envToWorld(ray.end[0], ray.end[1]);
    const origin = _tmpV3.set(sW.x, BEAM_Y, sW.z);
    const hit = new THREE.Vector3(hW.x, BEAM_Y, hW.z);
    const col = rayCol(ray.distance);

    const dv = new THREE.Vector3().subVectors(hit, origin);
    const len = dv.length();
    if (len > 0.1) {
      const q = new THREE.Quaternion().setFromUnitVectors(upY, dv.clone().normalize());
      const mid = origin.clone().addScaledVector(dv, 0.5);
      for (const m of [o.core, o.glow]) {
        m.visible = true;
        m.position.copy(mid);
        m.scale.set(1, len, 1);
        m.quaternion.copy(q);
        m.material.color.setHex(col);
      }
    } else {
      o.core.visible = o.glow.visible = false;
    }
    o.hit.position.set(hit.x, BEAM_Y, hit.z);
    o.ring.position.set(hit.x, 0.72, hit.z);
    o.ring.material.color.setHex(col);

    if (mode === "orbit") {
      const p = new THREE.Vector3(hit.x, BEAM_Y, hit.z).project(camera);
      if (p.z < 1) {
        const sx = (p.x + 1) / 2 * W;
        const sy = (1 - p.y) / 2 * H;
        const c = "#" + col.toString(16).padStart(6, "0");
        labelParts.push(
          `<div class="ray-label" style="left:${sx.toFixed(0)}px;top:${sy.toFixed(0)}px;color:${c};border-color:${c}66">${ray.distance.toFixed(0)} px</div>`
        );
      }
    }
  }

  // Illustrative sweep + labels only in the orbit "sensor" view.
  if (sweepMesh) {
    sweepMesh.visible = mode === "orbit";
    if (sweepMesh.visible) {
      sweepMesh.position.set(carGroup.position.x, 2.0, carGroup.position.z);
      sweepMesh.rotation.z -= 0.05;
    }
  }
  const labels = document.getElementById("rayLabels");
  if (labels) labels.innerHTML = mode === "orbit" ? labelParts.join("") : "";
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

const TRAIL_LEN = 260;
// Fading cyan ribbon of the last ~260 poses, slightly above the road, brighter near the car.
// Rebuilt each frame from the contiguous window [baseIdx-259 .. baseIdx] + the interpolated
// head, so it never draws a diagonal across the track on wrap/scrub.
function updatePlaybackTrajectory(steps, baseIdx, alpha, curX, curZ) {
  const start = Math.max(0, baseIdx - TRAIL_LEN + 1);
  const pts = [];
  for (let i = start; i <= baseIdx; i++) {
    const v = steps[i] && steps[i].vehicle;
    if (v && isValidNumber(v.x) && isValidNumber(v.y)) {
      const w = envToWorld(v.x, v.y);
      pts.push([w.x, w.z]);
    }
  }
  pts.push([curX, curZ]); // interpolated head pose

  if (pts.length < 2) {
    if (trajectoryMesh) trajectoryMesh.visible = false;
    return;
  }

  const half = 1.7, y = 0.66;
  const n = pts.length;
  const pos = new Float32Array(n * 2 * 3);
  const col = new Float32Array(n * 2 * 3);
  const idx = [];
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    const prev = pts[Math.max(0, i - 1)];
    const next = pts[Math.min(n - 1, i + 1)];
    let dx = next[0] - prev[0], dz = next[1] - prev[1];
    const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
    const nx = -dz, nz = dx; // perpendicular
    const o = i * 6;
    pos[o] = p[0] + nx * half; pos[o + 1] = y; pos[o + 2] = p[1] + nz * half;
    pos[o + 3] = p[0] - nx * half; pos[o + 4] = y; pos[o + 5] = p[1] - nz * half;
    const a = i / (n - 1); // tail (0) -> head (1): brighter near the car
    const r = 0.25 * a, g = 0.85 * a, b = 1.0 * a;
    col[o] = r; col[o + 1] = g; col[o + 2] = b;
    col[o + 3] = r; col[o + 4] = g; col[o + 5] = b;
    if (i > 0) { const base = (i - 1) * 2; idx.push(base, base + 2, base + 1, base + 1, base + 2, base + 3); }
  }

  if (!trajectoryMesh) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx);
    const mat = new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.9,
      blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
    });
    trajectoryMesh = new THREE.Mesh(geo, mat);
    scene.add(trajectoryMesh);
  } else {
    const g = trajectoryMesh.geometry;
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(idx);
  }
  trajectoryMesh.visible = true;
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

  // Near plane: 0.4 only in cockpit, >=15 overhead, 2.0 elsewhere (set here too so
  // tests reading camera.near right after a switch see the right value).
  camera.near = mode === "cockpit" ? 0.4 : mode === "overhead" ? 20.0 : 2.0;
  camera.updateProjectionMatrix();

  // Snapshot the current camera pose and start a smooth ease-in/out transition.
  STATE.camFromPos.copy(STATE.cameraPos);
  STATE.camFromTarget.copy(STATE.cameraTarget);
  STATE.camFromUp.copy(STATE.cameraUp);
  STATE.camFromFov = STATE.cameraFov;
  cameraTransT = 0.0;

  if (mode === "orbit") {
    orbitUserTookOver = false;
    orbitAngle = Math.atan2(chaseForwardZ, chaseForwardX) + Math.PI + 0.6; // start behind/side
  }
  // Toggle the sensor legend (orbit only)
  const legend = document.getElementById("sensorLegend");
  if (legend) legend.style.display = mode === "orbit" ? "block" : "none";
  const labels = document.getElementById("rayLabels");
  if (labels && mode !== "orbit") labels.innerHTML = "";
}
window.setCameraMode = setCameraMode;

// Desired camera pose per mode (position, look target, fov, up, near).
function desiredCameraPose(mode, carPos, delta) {
  const h = carGroup.rotation.y;
  const fwd = new THREE.Vector3(Math.cos(h), 0, -Math.sin(h));
  const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
  const lagFwd = new THREE.Vector3(chaseForwardX, 0, chaseForwardZ);
  if (lagFwd.lengthSq() < 1e-6) lagFwd.copy(fwd);

  if (mode === "cockpit") {
    return {
      pos: new THREE.Vector3(carPos.x + fwd.x * 6, carPos.y + 16, carPos.z + fwd.z * 6),
      target: new THREE.Vector3(carPos.x + fwd.x * 140, carPos.y + 9, carPos.z + fwd.z * 140),
      fov: 72, up: new THREE.Vector3(0, 1, 0), near: 0.4, useControls: false,
    };
  }
  if (mode === "overhead") {
    const b = (STATE.trackData && STATE.trackData.bounds) || { min_x: 0, max_x: 800, min_y: 0, max_y: 600 };
    const c = envToWorld((b.min_x + b.max_x) / 2, (b.min_y + b.max_y) / 2);
    const spanX = b.max_x - b.min_x, spanZ = b.max_y - b.min_y;
    const fovV = 46;
    const aspect = (camera && camera.aspect) || 1.6;
    const needV = (spanZ / 2) / Math.tan((fovV * Math.PI) / 360);
    const needH = (spanX / 2) / (Math.tan((fovV * Math.PI) / 360) * aspect);
    const height = Math.max(needV, needH) * 1.12 + 120;
    return {
      pos: new THREE.Vector3(c.x, height, c.z),
      target: new THREE.Vector3(c.x, 0, c.z),
      // Very high camera (large circuit): raise near so the 24-bit depth buffer can still
      // separate ground / runoff / asphalt (0.4-1.0 apart); standard tracks keep near = 20.
      fov: fovV, up: new THREE.Vector3(0, 0, -1), near: height > 3000 ? height * 0.4 : 20,
      far: Math.max(9000, height * 1.6), useControls: false,
    };
  }
  if (mode === "orbit") {
    orbitAngle += (delta || 0.016) * 0.28; // slow auto-rotation
    return {
      pos: new THREE.Vector3(carPos.x + Math.cos(orbitAngle) * 175, carPos.y + 120, carPos.z + Math.sin(orbitAngle) * 175),
      target: new THREE.Vector3(carPos.x, carPos.y + 8, carPos.z),
      fov: 48, up: new THREE.Vector3(0, 1, 0), near: 2, useControls: orbitUserTookOver,
    };
  }
  // chase — lagged cinematic heading
  return {
    pos: new THREE.Vector3(carPos.x - lagFwd.x * 185, carPos.y + 52, carPos.z - lagFwd.z * 185),
    target: new THREE.Vector3(carPos.x + lagFwd.x * 70, carPos.y + 14, carPos.z + lagFwd.z * 70),
    fov: 52, up: new THREE.Vector3(0, 1, 0), near: 2, useControls: false,
  };
}

function updateCameraDirector(car, velocity, delta = 0.016) {
  if (!carGroup || !camera) return;

  const safeDelta = Math.min(Math.max(delta || 0.016, 0.001), 0.1);
  const carPos = carGroup.position;
  const d = desiredCameraPose(STATE.activeCameraMode, carPos, delta);

  const wantFar = d.far || 9000; // overhead on a large circuit needs a deeper far plane
  if (camera.near !== d.near || camera.far !== wantFar) {
    camera.near = d.near;
    camera.far = wantFar;
    camera.updateProjectionMatrix();
  }

  cameraTransT = Math.min(CAMERA_TRANSITION_S, cameraTransT + safeDelta);

  if (d.useControls && controls) {
    // Orbit after the user grabbed the view: OrbitControls drives the camera.
    controls.enabled = true;
    controls.target.set(carPos.x, carPos.y + 8, carPos.z);
    controls.update();
    STATE.cameraPos.copy(camera.position);
    STATE.cameraTarget.copy(controls.target);
    STATE.cameraFov = camera.fov;
    STATE.cameraUp.copy(camera.up);
    return;
  }

  // Keep controls enabled in orbit to capture the first drag, but don't let it move
  // the camera until the user takes over.
  if (controls) controls.enabled = STATE.activeCameraMode === "orbit";

  // Ease-in/out blend from the pose at the last switch toward the (live) desired pose.
  // smoothstep gives zero velocity at both ends → no jump on switch, no snap on arrival.
  const u = smoothstep01(cameraTransT / CAMERA_TRANSITION_S);
  STATE.cameraPos.copy(STATE.camFromPos).lerp(d.pos, u);
  STATE.cameraTarget.copy(STATE.camFromTarget).lerp(d.target, u);
  STATE.cameraUp.copy(STATE.camFromUp).lerp(d.up, u);
  if (STATE.cameraUp.lengthSq() > 1e-6) STATE.cameraUp.normalize();
  STATE.cameraFov = STATE.camFromFov + (d.fov - STATE.camFromFov) * u;

  camera.position.copy(STATE.cameraPos);
  camera.up.copy(STATE.cameraUp);
  camera.lookAt(STATE.cameraTarget);
  if (Math.abs(camera.fov - STATE.cameraFov) > 0.01) {
    camera.fov = STATE.cameraFov;
    camera.updateProjectionMatrix();
  }
}

// ==============================================================================
// Benchmark Mode (/demo?bench=1)
// ==============================================================================

const BENCH = { active: false, recording: false, samples: [], lastInfo: null };

function percentile(sortedAsc, p) {
  if (sortedAsc.length === 0) return 0;
  const idx = Math.min(sortedAsc.length - 1, Math.floor((p / 100) * sortedAsc.length));
  return sortedAsc[idx];
}

function summarizePhase(label, quality, samples, info) {
  // Split the first 1 s (startup / any residual compile) from the steady-state window.
  const startup = [];
  const steady = [];
  let cum = 0;
  for (const ms of samples) {
    if (cum < 1000) startup.push(ms); else steady.push(ms);
    cum += ms;
  }
  const body = steady.length > 2 ? steady : samples;
  const sorted = [...body].sort((a, b) => a - b);
  const n = sorted.length;
  const sumMs = sorted.reduce((a, b) => a + b, 0);
  const avgFps = n > 0 ? (1000 * n) / sumMs : 0;
  const maxMs = n > 0 ? sorted[n - 1] : 0;
  const minFps = maxMs > 0 ? 1000 / maxMs : 0;
  const startupMax = startup.length ? Math.max(...startup) : 0;
  return {
    phase: label,
    quality,
    frames: n,
    duration_s: +(sumMs / 1000).toFixed(2),
    avg_fps: +avgFps.toFixed(1),
    min_fps: +minFps.toFixed(1),
    frame_time_ms: {
      p50: +percentile(sorted, 50).toFixed(2),
      p95: +percentile(sorted, 95).toFixed(2),
      p99: +percentile(sorted, 99).toFixed(2),
      max: +maxMs.toFixed(2),
    },
    startup: { frames: startup.length, max_ms: +startupMax.toFixed(2) },
    renderer_info: info || null,
  };
}

// Pre-compile all materials/post-processing for every camera mode and quality so the
// first rendered frame of playback doesn't hitch on shader compilation.
function warmupShaders() {
  if (!renderer || !scene || !camera) return;
  try {
    const prevMode = STATE.activeCameraMode;
    const prevQuality = STATE.graphicsQuality;
    for (const q of ["high", "low"]) {
      setGraphicsQuality(q);
      for (const m of ["chase", "cockpit", "orbit", "overhead"]) {
        setCameraMode(m);
        renderer.compile(scene, camera);
        if (q === "high" && composer) composer.render();
        else renderer.render(scene, camera);
      }
    }
    setGraphicsQuality(prevQuality);
    setCameraMode(prevMode);
  } catch (e) {
    console.warn("[warmup] shader pre-compile skipped:", e);
  }
}

async function runBenchmark(durationMs = 20000) {
  // Wait for the rollout to be ready.
  const t0 = performance.now();
  while ((!STATE.playback.rolloutData || !STATE.playback.rolloutData.steps) &&
         performance.now() - t0 < 20000) {
    await new Promise((r) => setTimeout(r, 200));
  }
  if (!STATE.playback.rolloutData) {
    console.warn("[bench] rollout never loaded; aborting benchmark");
    return;
  }

  BENCH.active = true;
  setCameraMode("chase");

  const phases = [
    { label: "High quality", quality: "high" },
    { label: "Low quality", quality: "low" },
  ];
  const results = [];

  for (const ph of phases) {
    setGraphicsQuality(ph.quality);
    // Restart playback from the top for a comparable 20 s window.
    STATE.playback.playhead = 0.0;
    STATE.playback.lastStepIndex = -1;
    STATE.playback.speed = 1.0;
    STATE.playback.isPlaying = true;
    STATE.simRunning = true;

    showBenchOverlay(`Recording ${ph.label}… (${durationMs / 1000}s)`, results);
    // Warm up this quality's pipeline (compile any passes) before recording.
    if (renderer && scene && camera) {
      renderer.compile(scene, camera);
      if (ph.quality === "high" && composer) composer.render(); else renderer.render(scene, camera);
    }
    // Let the quality switch settle before recording.
    await new Promise((r) => setTimeout(r, 600));
    BENCH.samples = [];
    BENCH.lastInfo = null;
    if (renderer && renderer.info) renderer.info.autoReset = false;
    BENCH.recording = true;
    await new Promise((r) => setTimeout(r, durationMs));
    BENCH.recording = false;
    if (renderer && renderer.info) renderer.info.autoReset = true;

    results.push(summarizePhase(ph.label, ph.quality, BENCH.samples, BENCH.lastInfo));
    showBenchOverlay("Benchmark running…", results);
  }

  BENCH.active = false;
  STATE.playback.isPlaying = false;

  const payload = {
    benchmark: "rl-racing-lab chase cam",
    seed: STATE.trackSeed,
    preset: STATE.presetId,
    viewport: [window.innerWidth, window.innerHeight],
    device_pixel_ratio: window.devicePixelRatio || 1,
    results,
  };
  console.log("[bench] RESULTS JSON:\n" + JSON.stringify(payload, null, 2));
  window.__benchResults = payload;
  showBenchOverlay("Benchmark complete — results below (also logged to console)", results, true);
}

function showBenchOverlay(status, results, done) {
  let el = document.getElementById("benchOverlay");
  if (!el) {
    el = document.createElement("div");
    el.id = "benchOverlay";
    el.style.cssText =
      "position:fixed;top:16px;left:50%;transform:translateX(-50%);z-index:9999;" +
      "background:rgba(9,13,20,0.92);border:1px solid rgba(140,200,255,0.25);border-radius:12px;" +
      "padding:16px 20px;color:#e8eef6;font:13px/1.5 'DejaVu Sans Mono',monospace;min-width:460px;" +
      "box-shadow:0 10px 40px rgba(0,0,0,0.5);backdrop-filter:blur(8px)";
    document.body.appendChild(el);
  }
  const rows = results.map((r) => {
    const i = r.renderer_info || {};
    return (
      `<div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.12)">` +
      `<b style="color:#67e3ff">${r.phase} (${r.quality})</b><br>` +
      `avg fps <b>${r.avg_fps}</b> &nbsp; min fps <b>${r.min_fps}</b> &nbsp; frames ${r.frames}<br>` +
      `frame time ms — p50 ${r.frame_time_ms.p50} · p95 ${r.frame_time_ms.p95} · p99 ${r.frame_time_ms.p99} · max ${r.frame_time_ms.max}<br>` +
      `startup (first 1s, excluded): ${r.startup ? r.startup.frames : 0} frames, max ${r.startup ? r.startup.max_ms : 0} ms<br>` +
      `draws ${i.draw_calls ?? "–"} · tris ${i.triangles ?? "–"} · tex ${i.textures ?? "–"} · geo ${i.geometries ?? "–"}` +
      `</div>`
    );
  });
  el.innerHTML =
    `<div style="color:${done ? "#5ff2b5" : "#ffd27a"};font-weight:700;letter-spacing:.04em">${status}</div>` +
    rows.join("");
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
  const rawFrameMs = currentTime - lastFrameTime;
  lastFrameTime = currentTime;

  // Benchmark sampler (see runBenchmark): record real frame time + renderer.info.
  // info.autoReset is disabled during recording so totals accumulate across all
  // composer passes for the frame; we read the previous frame's totals, then reset.
  if (BENCH.recording && rawFrameMs > 0 && rawFrameMs < 2000) {
    BENCH.samples.push(rawFrameMs);
    if (renderer && renderer.info) {
      BENCH.lastInfo = {
        draw_calls: renderer.info.render.calls,
        triangles: renderer.info.render.triangles,
        textures: renderer.info.memory.textures,
        geometries: renderer.info.memory.geometries,
      };
      renderer.info.reset();
    }
  }

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
      if (BENCH.active) {
        // Loop the rollout continuously during benchmarking.
        STATE.playback.playhead = 0.0;
        STATE.playback.lastStepIndex = -1;
      } else {
        STATE.playback.playhead = STATE.playback.maxStep;
        STATE.playback.isPlaying = false;
        STATE.simRunning = false;
        const startIcon = document.getElementById("startIcon");
        if (startIcon) startIcon.innerText = "↺";
        updateStatusPill("COMPLETED", "green");
        showPostLapModal();
      }
    }
  }

  // Deterministic Rollout Playback Rendering
  if (STATE.playback.rolloutData && STATE.playback.rolloutData.steps && STATE.playback.rolloutData.steps.length > 0) {
    renderPlaybackFrame(STATE.playback.playhead, delta);
  }

  // Sun + tight shadow frustum follow the car, along the shared sun direction
  if (mainSun && carGroup) {
    const cx = carGroup.position.x;
    const cz = carGroup.position.z;
    mainSun.position.set(cx + SUN_DIR.x * 900, SUN_DIR.y * 900, cz + SUN_DIR.z * 900);
    mainSun.target.position.set(cx, 0, cz);
    mainSun.target.updateMatrixWorld();
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
    carGroup.position.set(curX, ROAD_TOP_Y, curZ); // tyres sit on the asphalt (no float/sink)
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

  // Lagged forward vector: car heading averaged over the last ~25 steps (cinematic chase).
  let lfx = 0, lfz = 0;
  for (let k = 0; k < 25; k++) {
    const sv = steps[Math.max(0, baseIdx - k)];
    if (!sv || !sv.vehicle) continue;
    const a = envHeadingToWorld(sv.vehicle.angle);
    lfx += Math.cos(a);
    lfz += -Math.sin(a);
  }
  const lfl = Math.hypot(lfx, lfz) || 1;
  chaseForwardX = lfx / lfl;
  chaseForwardZ = lfz / lfl;

  // 3. Speed & steering interpolation
  const curVelocity = THREE.MathUtils.lerp(car0.velocity, car1.velocity, alpha);
  const curSteerDeg = THREE.MathUtils.lerp(car0.steering_angle_deg, car1.steering_angle_deg, alpha);
  const curThrottle = THREE.MathUtils.lerp(car0.throttle, car1.throttle, alpha);

  // 4. Wheel spin from interpolated ground distance / wheel radius (all four wheels)
  if (prevCarWorldX !== null) {
    const moved = Math.hypot(curX - prevCarWorldX, curZ - prevCarWorldZ);
    const dir = curVelocity < 0 ? 1 : -1; // forward motion rolls the wheel forward
    wheelSpinAngle += dir * (moved / CAR_WHEEL_RADIUS);
  }
  prevCarWorldX = curX; prevCarWorldZ = curZ;

  // Front wheels yaw from steering_angle_deg telemetry (deg -> rad via world.js), not the action
  const steerRad = -envSteerToWorld(curSteerDeg);
  if (frontLeftSteerPivot) frontLeftSteerPivot.rotation.y = steerRad;
  if (frontRightSteerPivot) frontRightSteerPivot.rotation.y = steerRad;
  wheelSpinMeshes.forEach((spinGroup) => {
    spinGroup.rotation.z = wheelSpinAngle;
  });

  // 5. Dynamic brake lights
  const isDecel = curThrottle < -0.05 || car0.action_mode === "deceleration";
  rearBrakeLights.forEach((light) => {
    light.material.color.setHex(0x3a0000);
    light.material.emissive.setHex(isDecel ? 0xff1a12 : 0xbb1109);
    light.material.emissiveIntensity = isDecel ? 2.2 : 0.7;
  });

  // 6. Rangefinder sensor beams
  if (s0.rays) {
    updateLidarVisuals(s0.rays);
  }

  // 7. Trajectory ribbon
  updatePlaybackTrajectory(steps, baseIdx, alpha, curX, curZ);

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

  if (STATE.source === "bic" && window.recordedHudUpdate) window.recordedHudUpdate(baseIdx, alpha);
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

  drawSpeedChart(stepIdx);
  drawSteerChart(stepIdx);
  drawThrottleChart(stepIdx);
  drawLateralSlipChart(stepIdx);

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

// Lab sparklines: plot the rolling window of the REAL step values around the current
// index (not per-render-frame samples, which flatline when paused and oversample when
// fast). Matches the concept's chart() approach.
const CHART_WIN = 80;
const thrOfStep = (s) => (s.action && s.action[1] !== undefined) ? s.action[1] : ((s.vehicle && s.vehicle.throttle) || 0);

function chartFromSteps(canvasId, stepIdx, series, opts) {
  const canvas = document.getElementById(canvasId);
  const rollout = STATE.playback.rolloutData;
  if (!canvas || !rollout || !rollout.steps) return;
  const steps = rollout.steps;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const lo = opts.lo, hi = opts.hi;
  const base = Math.max(0, Math.min(steps.length - 1, Math.floor(stepIdx)));
  const yOf = (v) => h - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (h - 8) - 4;

  if (opts.zero && lo < 0 && hi > 0) {
    const zy = yOf(0);
    ctx.strokeStyle = "rgba(255,255,255,0.14)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(w, zy); ctx.stroke();
  }
  for (const srs of series) {
    ctx.beginPath();
    for (let i = 0; i < CHART_WIN; i++) {
      const s = steps[Math.max(0, base - CHART_WIN + 1 + i)];
      const x = (i / (CHART_WIN - 1)) * w;
      const y = yOf(srs.fn(s));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = srs.color; ctx.lineWidth = srs.width || 2.0; ctx.stroke();
    if (srs.fill) {
      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, srs.fill); g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g; ctx.fill();
    }
  }
}

function drawSpeedChart(stepIdx) {
  chartFromSteps("chartSpeed", stepIdx, [{ fn: (s) => s.vehicle.velocity, color: "#67e3ff", fill: "rgba(103,227,255,0.28)" }], { lo: 0, hi: 4.4 });
}
function drawSteerChart(stepIdx) {
  chartFromSteps("chartSteer", stepIdx, [{ fn: (s) => s.vehicle.steering_angle_deg, color: "#ffb02e" }], { lo: -32, hi: 32, zero: true });
}
function drawThrottleChart(stepIdx) {
  chartFromSteps("chartThrottle", stepIdx, [
    { fn: (s) => Math.max(0, thrOfStep(s)), color: "#3ee08f" },
    { fn: (s) => Math.min(0, thrOfStep(s)), color: "#ff4d4d" },
  ], { lo: -1.05, hi: 1.05, zero: true });
}
function drawLateralSlipChart(stepIdx) {
  chartFromSteps("chartLateralSlip", stepIdx, [
    { fn: (s) => (s.telemetry && s.telemetry.signed_distance) || 0, color: "#ffffff", width: 1.8 },
    { fn: (s) => ((s.telemetry && s.telemetry.slip_angle) || 0) * 2, color: "#b388ff", width: 1.8 },
  ], { lo: -30, hi: 30, zero: true });
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

  // Corner markers (recorded circuit: T1/T6/T8 = tightest corners from the audit)
  if (trackData.markers) {
    ctx.font = "bold 11px DejaVu Sans Mono, monospace";
    trackData.markers.forEach((m) => {
      const p = toScreen(m.x, m.y);
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffb02e"; ctx.fill();
      ctx.fillStyle = "#ffd98a"; ctx.fillText(m.label, p.x + 7, p.y + 4);
    });
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

    // Pre-compile shaders for every mode/quality so the first playback frames don't hitch.
    warmupShaders();

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
    renderer.setPixelRatio(quality === "high" ? highPixelRatio() : 1);
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

// Compute post-lap KPIs from the REAL rollout step telemetry (not the live session's
// 300-step window), matching the concept's "computed from logged telemetry".
function computeLapStats(steps) {
  const half = (STATE.trackData && STATE.trackData.half_width) || 35.0;
  let vSum = 0, vMax = 0, brake = 0, depart = 0, devSum = 0, steerDeltaSum = 0;
  let prevSteer = null;
  for (let i = 0; i < steps.length; i++) {
    const v = steps[i].vehicle || {};
    const t = steps[i].telemetry || {};
    const vel = v.velocity || 0;
    vSum += vel; if (vel > vMax) vMax = vel;
    const thr = (steps[i].action && steps[i].action[1] !== undefined) ? steps[i].action[1] : (v.throttle || 0);
    if (thr < -0.05) brake++;
    const dev = Math.abs(t.signed_distance || 0);
    devSum += dev;
    if (dev > half) depart++;
    const st = v.steering_angle_deg || 0;
    if (prevSteer !== null) steerDeltaSum += Math.abs(st - prevSteer);
    prevSteer = st;
  }
  const n = steps.length || 1;
  const crashed = !!(STATE.playback.rolloutData && STATE.playback.rolloutData.session &&
    STATE.playback.rolloutData.session.is_crashed);
  return {
    lapSteps: STATE.playback.maxStep,
    avgSpeed: vSum / n,
    maxSpeed: vMax,
    departures: depart,
    collisions: crashed ? 1 : 0,
    brakingPct: (brake / n) * 100,
    avgLatDev: devSum / n,
    steerDelta: steerDeltaSum / Math.max(1, n - 1),
  };
}

function showPostLapModal() {
  const overlay = document.getElementById("canvasOverlay");
  if (!overlay) return;
  overlay.style.display = "flex";

  const rollout = STATE.playback.rolloutData;
  if (!rollout || !rollout.steps) return;
  if (STATE.source === "bic" && window.recordedPostLap) { window.recordedPostLap(); drawLapCircuitSpeedMap(); drawLapTraces(); return; }
  const st = computeLapStats(rollout.steps);

  const setKpi = (id, val, unit) => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = unit ? `${val} <span class="kpi-unit">${unit}</span>` : `${val}`;
  };
  setKpi("kpiLapTime", st.lapSteps, "steps");
  setKpi("kpiAvgSpeed", st.avgSpeed.toFixed(2), "px/st");
  setKpi("kpiMaxSpeed", st.maxSpeed.toFixed(2), "px/st");
  setKpi("kpiDepartures", st.departures);
  setKpi("kpiCollisions", st.collisions);
  setKpi("kpiBraking", st.brakingPct.toFixed(1), "%");
  setKpi("kpiLatDev", st.avgLatDev.toFixed(1), "px");
  setKpi("kpiSteerDelta", st.steerDelta.toFixed(2) + "°");

  drawLapCircuitSpeedMap();
  drawLapTraces();
}

function hidePostLapModal() {
  const overlay = document.getElementById("canvasOverlay");
  if (overlay) overlay.style.display = "none";
}

// Speed-along-the-lap map: the REAL driven path coloured by REAL speed (red slow -> cyan fast).
function drawLapCircuitSpeedMap() {
  const canvas = document.getElementById("lapCircuitCanvas");
  const rollout = STATE.playback.rolloutData;
  if (!canvas || !STATE.trackData || !rollout || !rollout.steps) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const bounds = STATE.trackData.bounds || { min_x: 100, max_x: 700, min_y: 100, max_y: 500 };
  const pad = 22;
  const scale = Math.min((w - 2 * pad) / (bounds.max_x - bounds.min_x), (h - 2 * pad) / (bounds.max_y - bounds.min_y));
  const ox = (w - scale * (bounds.max_x - bounds.min_x)) / 2;
  const oz = (h - scale * (bounds.max_y - bounds.min_y)) / 2;
  const toScreen = (x, y) => ({ x: ox + (x - bounds.min_x) * scale, y: oz + (y - bounds.min_y) * scale });

  // Faint full centreline underlay
  const cl = STATE.trackData.centerline || [];
  if (cl.length > 2) {
    ctx.beginPath();
    cl.forEach((p, i) => { const s = toScreen(p[0], p[1]); i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
    ctx.closePath();
    ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = 10; ctx.lineCap = "round"; ctx.stroke();
  }

  const steps = rollout.steps;
  const VLO = 2.2, VHI = 4.0;
  ctx.lineWidth = 6; ctx.lineCap = "round";
  for (let i = 1; i < steps.length; i++) {
    const a = steps[i - 1].vehicle, b = steps[i].vehicle;
    if (!a || !b) continue;
    const p0 = toScreen(a.x, a.y), p1 = toScreen(b.x, b.y);
    const u = Math.min(1, Math.max(0, (b.velocity - VLO) / (VHI - VLO)));
    ctx.strokeStyle = `hsl(${(u * 190).toFixed(0)},90%,${(55 - u * 5).toFixed(0)}%)`;
    ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
  }
}

function drawLapTraces() {
  const rollout = STATE.playback.rolloutData;
  if (!rollout || !rollout.steps) return;
  const steps = rollout.steps;
  const nS = steps.length;
  const thrOf = (s) => (s.action && s.action[1] !== undefined) ? s.action[1] : ((s.vehicle && s.vehicle.throttle) || 0);

  // Speed trace with braking zones shaded (where longitudinal action < 0).
  const speedCanvas = document.getElementById("lapSpeedTraceCanvas");
  if (speedCanvas) {
    const ctx = speedCanvas.getContext("2d");
    const w = speedCanvas.width, h = speedCanvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(255,255,255,0.03)"; ctx.fillRect(0, 0, w, h);
    // braking-zone shading
    ctx.fillStyle = "rgba(255,77,77,0.16)";
    for (let i = 0; i < nS; i++) {
      if (thrOf(steps[i]) < -0.05) ctx.fillRect((i / nS) * w, 0, w / nS + 1, h);
    }
    // gridlines
    ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
    for (let g = 1; g < 4; g++) { ctx.beginPath(); ctx.moveTo(0, (h * g) / 4); ctx.lineTo(w, (h * g) / 4); ctx.stroke(); }
    // speed line (2..4.2)
    const lo = 2, hi = 4.2;
    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = (i / nS) * w;
      const y = h - ((s.vehicle.velocity - lo) / (hi - lo)) * (h - 8) - 4;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = "#67e3ff"; ctx.lineWidth = 2.2; ctx.stroke();
    ctx.font = "11px DejaVu Sans Mono, monospace"; ctx.fillStyle = "#8fa3b8";
    ctx.fillText(hi.toString(), 5, 13); ctx.fillText(lo.toString(), 5, h - 5);
  }

  // Steering-angle trace.
  const steerCanvas = document.getElementById("lapSteerTraceCanvas");
  if (steerCanvas) {
    const ctx = steerCanvas.getContext("2d");
    const w = steerCanvas.width, h = steerCanvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(255,255,255,0.03)"; ctx.fillRect(0, 0, w, h);
    const centerY = h / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, centerY); ctx.lineTo(w, centerY); ctx.stroke();
    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = (i / nS) * w;
      const y = centerY - (s.vehicle.steering_angle_deg / 32.0) * (centerY - 6);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = "#ffb02e"; ctx.lineWidth = 2.2; ctx.stroke();
    ctx.font = "11px DejaVu Sans Mono, monospace"; ctx.fillStyle = "#8fa3b8";
    ctx.fillText("32", 5, 13); ctx.fillText("-32", 5, h - 5);
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
    } else if (e.key === "d" || e.key === "D") {
      toggleDebugMode();
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

// Debug overlay toggle (key D): axes + the 30x50 px physics-footprint wireframe.
function toggleDebugMode() {
  STATE.debugMode = !STATE.debugMode;
  if (debugGroup) debugGroup.visible = STATE.debugMode;
  if (trackAxesHelper) trackAxesHelper.visible = STATE.debugMode;
  if (carAxesHelper) carAxesHelper.visible = STATE.debugMode;
  if (carPhysicsFootprint) carPhysicsFootprint.visible = STATE.debugMode;
  const btn = document.getElementById("btnDebugToggle");
  if (btn) btn.style.display = STATE.debugMode ? "inline-flex" : "none";
}
window.toggleDebugMode = toggleDebugMode;

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

// Minimum distance (env px) from any scenery prop (trees, skyline, grandstand, pit)
// to the track centerline. Used by the e2e prop-clearance test. Returns null until
// the scene + track are built.
window.__minPropClearancePx = function () {
  if (!propsGroup || !STATE.trackData || !STATE.trackData.centerline) return null;
  const cl = STATE.trackData.centerline;
  if (!cl.length) return null;

  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  let min = Infinity;

  function consider(worldX, worldZ) {
    const env = worldToEnv(worldX, worldZ); // {x, y} in env px
    let best = Infinity;
    for (let i = 0; i < cl.length; i++) {
      const dx = cl[i][0] - env.x, dy = cl[i][1] - env.y;
      const d = dx * dx + dy * dy;
      if (d < best) best = d;
    }
    const dist = Math.sqrt(best);
    if (dist < min) min = dist;
  }

  propsGroup.traverse((obj) => {
    if (obj.isInstancedMesh) {
      for (let i = 0; i < obj.count; i++) {
        obj.getMatrixAt(i, m);
        p.setFromMatrixPosition(m);
        consider(p.x, p.z);
      }
    } else if (obj.isMesh) {
      consider(obj.position.x, obj.position.z);
    }
  });

  return (min === Infinity) ? null : min;
};

// Initialize Application
window.addEventListener("DOMContentLoaded", () => {
  initThreeJS();
  initEventHandlers();

  // Start animation loop immediately (Non-blocking startup: 3D scene renders within milliseconds)
  startAnimationLoop();

  // Initiate simulation rollout asynchronously in background without blocking render loop
  // Recorded source (?source=bic) never starts a live rollout: recorded.js loads the saved run.
  if (STATE.source !== "bic") startSimulation(STATE.presetId, STATE.trackSeed);

  // Benchmark mode: /demo?bench=1 records 20s at High then Low in chase cam.
  try {
    const params = new URLSearchParams(window.location.search);
    if (params.get("bench") === "1") {
      runBenchmark(20000);
    }
  } catch (e) { /* no-op */ }
});
