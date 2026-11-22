import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { buildSim, HALF_W, RAY_MAX, VMAX } from './sim.js';

const W = 1920, H = 1080, SPS = 75; // sim steps per video second
const S = buildSim();
const { T, steps, stats } = S;
const M = T.M;
const LAP = stats.lapSteps;

// ---------- renderer / scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('c'), antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.62;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, W / H, 1, 30000);

// sky + environment lighting
const sky = new Sky();
sky.scale.setScalar(25000);
const su = sky.material.uniforms;
su.turbidity.value = 2.6; su.rayleigh.value = 2.2; su.mieCoefficient.value = 0.004; su.mieDirectionalG.value = 0.82;
const sunDir = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - 26), THREE.MathUtils.degToRad(215));
su.sunPosition.value.copy(sunDir);
scene.add(sky);
const pmrem = new THREE.PMREMGenerator(renderer);
const envScene = new THREE.Scene(); const sky2 = sky.clone(); envScene.add(sky2);
scene.environment = pmrem.fromScene(envScene, 0.02).texture;
scene.environmentIntensity = 0.55;
scene.fog = new THREE.Fog(0xa9c4de, 3000, 11000);

const sun = new THREE.DirectionalLight(0xfff1dc, 3.2);
sun.castShadow = true;
sun.shadow.mapSize.set(1536, 1536);
const sc = sun.shadow.camera; sc.left = -320; sc.right = 320; sc.top = 320; sc.bottom = -320; sc.near = 10; sc.far = 3000;
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.6;
scene.add(sun, sun.target);
scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x4a5a32, 0.55));

// ---------- textures ----------
function canvasTex(w, h, draw, rep) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  t.wrapS = t.wrapT = THREE.RepeatWrapping; if (rep) t.repeat.set(rep[0], rep[1]); return t;
}
let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const asphaltTex = canvasTex(512, 512, (g, w, h) => {
  g.fillStyle = '#3b3e43'; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 26000; i++) { const v = 40 + rnd() * 50 | 0; g.fillStyle = `rgba(${v},${v + 2},${v + 5},${0.35 + rnd() * 0.5})`; g.fillRect(rnd() * w, rnd() * h, 1 + rnd() * 2, 1 + rnd() * 2); }
  // rubbered-in racing line darkening
  const gr = g.createLinearGradient(0, 0, w, 0); gr.addColorStop(0.25, 'rgba(0,0,0,0)'); gr.addColorStop(0.5, 'rgba(10,10,12,.28)'); gr.addColorStop(0.75, 'rgba(0,0,0,0)');
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
});
const grassTex = canvasTex(512, 512, (g, w, h) => {
  g.fillStyle = '#56763a'; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 30000; i++) { const r = 60 + rnd() * 40, gg = 100 + rnd() * 60, b = 35 + rnd() * 25; g.fillStyle = `rgba(${r | 0},${gg | 0},${b | 0},.55)`; g.fillRect(rnd() * w, rnd() * h, 2, 2 + rnd() * 3); }
}, [140, 140]);
const curbTex = canvasTex(64, 128, (g, w, h) => { g.fillStyle = '#d8231d'; g.fillRect(0, 0, w, h); g.fillStyle = '#f2f2f2'; g.fillRect(0, 0, w, h / 2); });
const armcoTex = canvasTex(256, 64, (g, w, h) => {
  g.fillStyle = '#c9ced4'; g.fillRect(0, 0, w, h); g.fillStyle = '#1f6fd1'; g.fillRect(0, 0, w, 18); g.fillStyle = '#dfe3e8'; g.fillRect(0, 26, w, 6);
  g.fillStyle = '#8a9099'; g.fillRect(0, h - 6, w, 6);
});
const winTex = canvasTex(256, 512, (g, w, h) => {
  g.fillStyle = '#7c8794'; g.fillRect(0, 0, w, h);
  for (let y = 8; y < h; y += 22) for (let x = 8; x < w; x += 20) { const v = rnd(); g.fillStyle = v > .82 ? '#e9e2c4' : v > .4 ? '#3c4d61' : '#55697f'; g.fillRect(x, y, 13, 14); }
});
const crowdTex = canvasTex(512, 64, (g, w, h) => {
  g.fillStyle = '#2b2f36'; g.fillRect(0, 0, w, h);
  const cols = ['#e64a3b', '#f2c14e', '#3b82f6', '#f5f5f5', '#22c55e', '#111', '#ff7a1a'];
  for (let i = 0; i < 2600; i++) { g.fillStyle = cols[(rnd() * cols.length) | 0]; g.fillRect(rnd() * w, rnd() * h, 3, 4); }
});

// ---------- track meshes ----------
function ribbon(off0, off1, y, mat, vScale, idxFilter) {
  const pos = [], uv = [], ind = [];
  let cum = 0, n = 0;
  const runs = [];
  let cur = null;
  for (let i = 0; i <= M; i++) {
    const k = i % M;
    if (!idxFilter || idxFilter(k)) { if (!cur) { cur = []; runs.push(cur); } cur.push(k); } else cur = null;
  }
  for (const run of runs) {
    const base = n; cum = 0;
    run.forEach((k, j) => {
      const p = T.pts[k], l = T.left[k];
      if (j > 0) { const q = T.pts[run[j - 1]]; cum += Math.hypot(p[0] - q[0], p[1] - q[1]); }
      pos.push(p[0] + l[0] * off0, y, p[1] + l[1] * off0, p[0] + l[0] * off1, y, p[1] + l[1] * off1);
      uv.push(0, cum / vScale, 1, cum / vScale);
      if (j > 0) { const a = base + (j - 1) * 2; ind.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
      n += 2;
    });
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(ind); geo.computeVertexNormals();
  // ensure normals point up
  const nr = geo.attributes.normal; for (let i = 0; i < nr.count; i++) nr.setXYZ(i, 0, 1, 0);
  const m = new THREE.Mesh(geo, mat); m.receiveShadow = true; scene.add(m); return m;
}
function wall(off, h, mat, vScale) {
  const pos = [], uv = [], ind = []; let cum = 0;
  for (let i = 0; i <= M; i++) {
    const k = i % M, p = T.pts[k], l = T.left[k];
    if (i > 0) { const q = T.pts[(i - 1) % M]; cum += Math.hypot(p[0] - q[0], p[1] - q[1]); }
    const x = p[0] + l[0] * off, z = p[1] + l[1] * off;
    pos.push(x, 0, z, x, h, z); uv.push(cum / vScale, 0, cum / vScale, 1);
    if (i > 0) { const a = (i - 1) * 2; ind.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(ind); geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, mat); m.castShadow = true; m.receiveShadow = true; scene.add(m);
}
const ground = new THREE.Mesh(new THREE.PlaneGeometry(20000, 20000), new THREE.MeshStandardMaterial({ map: grassTex, roughness: 1 }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
// runoff tarmac margin (darker, slightly wider than corridor)
ribbon(-HALF_W - 6, HALF_W + 6, 0.3, new THREE.MeshStandardMaterial({ color: 0x55585c, roughness: 0.95 }), 100);
ribbon(-HALF_W, HALF_W, 0.6, new THREE.MeshStandardMaterial({ map: asphaltTex, roughness: 0.82, metalness: 0.0 }), 150);
const lineMat = new THREE.MeshStandardMaterial({ color: 0xf4f4f4, roughness: 0.6 });
ribbon(HALF_W - 3.2, HALF_W - 1.2, 0.8, lineMat, 100);
ribbon(-HALF_W + 1.2, -HALF_W + 3.2, 0.8, lineMat, 100);
const corner = (k) => Math.abs(S.ks[k]) > 1 / 260;
const curbMat = new THREE.MeshStandardMaterial({ map: curbTex, roughness: 0.55 });
ribbon(HALF_W, HALF_W + 11, 1.0, curbMat, 22, corner);
ribbon(-HALF_W - 11, -HALF_W, 1.0, curbMat, 22, corner);
const armco = new THREE.MeshStandardMaterial({ map: armcoTex, roughness: 0.45, metalness: 0.5, side: THREE.DoubleSide });
wall(HALF_W + 62, 13, armco, 80);
wall(-HALF_W - 62, 13, armco, 80);
// start line checker
{
  const k = 6, p = T.pts[k], l = T.left[k], t = T.tan[k];
  const chk = canvasTex(256, 32, (g, w, h) => { for (let x = 0; x < 16; x++) for (let y = 0; y < 2; y++) { g.fillStyle = (x + y) % 2 ? '#111' : '#f5f5f5'; g.fillRect(x * 16, y * 16, 16, 16); } });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(70, 9), new THREE.MeshStandardMaterial({ map: chk, roughness: .7 }));
  m.rotation.x = -Math.PI / 2; m.rotation.z = -Math.atan2(l[1], l[0]); m.position.set(p[0], 0.9, p[1]); m.receiveShadow = true; scene.add(m);
  // gantry
  const gm = new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: .5, metalness: .6 });
  for (const s of [-1, 1]) { const post = new THREE.Mesh(new THREE.BoxGeometry(5, 64, 5), gm); post.position.set(p[0] + l[0] * 52 * s, 32, p[1] + l[1] * 52 * s); post.castShadow = true; scene.add(post); }
  const beam = new THREE.Mesh(new THREE.BoxGeometry(6, 12, 112), gm);
  beam.position.set(p[0], 64, p[1]); beam.rotation.y = -Math.atan2(t[1], t[0]); beam.castShadow = true; scene.add(beam);
  const panel = new THREE.Mesh(new THREE.BoxGeometry(6.5, 7, 80), new THREE.MeshStandardMaterial({ color: 0x111, emissive: 0x18d4ff, emissiveIntensity: 1.4 }));
  panel.position.copy(beam.position); panel.rotation.copy(beam.rotation); scene.add(panel);
}
// distance to track (coarse) for placing scenery
function trackDist(x, z) { let b = 1e9; for (let i = 0; i < M; i += 6) { const p = T.pts[i]; const d = (p[0] - x) ** 2 + (p[1] - z) ** 2; if (d < b) b = d; } return Math.sqrt(b); }
// trees
{
  const trunkG = new THREE.CylinderGeometry(3, 4.5, 40, 6); trunkG.translate(0, 20, 0);
  const pineG = new THREE.ConeGeometry(30, 120, 8); pineG.translate(0, 95, 0);
  const roundG = new THREE.IcosahedronGeometry(42, 1); roundG.translate(0, 70, 0);
  const trunkM = new THREE.MeshStandardMaterial({ color: 0x5b4330, roughness: 1 });
  const leafM = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .95, flatShading: true });
  const N = 900;
  const trunks = new THREE.InstancedMesh(trunkG, trunkM, N), pines = new THREE.InstancedMesh(pineG, leafM, N), rounds = new THREE.InstancedMesh(roundG, leafM, N);
  let a = 0, b = 0, c = 0; const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), col = new THREE.Color();
  for (let tries = 0; tries < 6000 && a < N; tries++) {
    const x = -2600 + rnd() * 5200, z = -1900 + rnd() * 3700;
    const d = trackDist(x, z); if (d < 150) continue;
    if (x > -900 && x < 400 && z < -760 && z > -1150) continue; // grandstand zone
    const s = 0.7 + rnd() * 0.7; q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), rnd() * 6.28);
    m4.compose(new THREE.Vector3(x, 0, z), q, new THREE.Vector3(s, s, s));
    trunks.setMatrixAt(a++, m4);
    col.setHSL(0.24 + rnd() * 0.08, 0.45 + rnd() * .2, 0.22 + rnd() * 0.12);
    if (rnd() < 0.45) { pines.setMatrixAt(b, m4); pines.setColorAt(b++, col); } else { rounds.setMatrixAt(c, m4); rounds.setColorAt(c++, col); }
  }
  trunks.count = a; pines.count = b; rounds.count = c;
  for (const im of [trunks, pines, rounds]) { im.castShadow = true; im.receiveShadow = true; scene.add(im); }
}
// buildings (skyline on the far side + scattered)
{
  const bm = new THREE.MeshStandardMaterial({ map: winTex, roughness: .4, metalness: .3 });
  for (let i = 0; i < 140; i++) {
    const ang = rnd() * Math.PI * 2, r = 3800 + rnd() * 3000;
    let x = Math.cos(ang) * r * 1.2, z = Math.sin(ang) * r;
    if (i < 60) { x = -2600 + rnd() * 5600; z = -3600 - rnd() * 2200; }
    const w = 160 + rnd() * 240, d = 160 + rnd() * 240, h = 160 + rnd() * (i < 60 ? 900 : 380);
    const g = new THREE.BoxGeometry(w, h, d); g.translate(0, h / 2, 0);
    const uvs = g.attributes.uv; for (let j = 0; j < uvs.count; j++) uvs.setXY(j, uvs.getX(j) * w / 160, uvs.getY(j) * h / 300);
    const m = new THREE.Mesh(g, bm); m.position.set(x, 0, z); m.rotation.y = rnd() * 0.6; scene.add(m);
  }
}
// grandstand + pit building along the main straight
{
  const gm = new THREE.MeshStandardMaterial({ map: crowdTex, roughness: .9 });
  const cm = new THREE.MeshStandardMaterial({ color: 0xd9dde2, roughness: .7 });
  for (let r = 0; r < 7; r++) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(1000, 12 + r * 12, 22), [cm, cm, cm, cm, gm, cm]);
    b.position.set(-250, (12 + r * 12) / 2, -820 - r * 22); b.castShadow = true; b.receiveShadow = true; scene.add(b);
  }
  const roof = new THREE.Mesh(new THREE.BoxGeometry(1020, 5, 190), new THREE.MeshStandardMaterial({ color: 0xeef1f4, roughness: .4, metalness: .4 }));
  roof.position.set(-250, 150, -880); roof.castShadow = true; scene.add(roof);
  for (let i = 0; i < 6; i++) { const p = new THREE.Mesh(new THREE.BoxGeometry(5, 150, 5), cm); p.position.set(-740 + i * 196, 75, -975); scene.add(p); }
  const pit = new THREE.Mesh(new THREE.BoxGeometry(900, 40, 60), new THREE.MeshStandardMaterial({ color: 0xf2f4f6, roughness: .5 }));
  pit.position.set(-300, 20, -600); pit.castShadow = true; pit.receiveShadow = true; scene.add(pit);
  const band = new THREE.Mesh(new THREE.BoxGeometry(902, 6, 62), new THREE.MeshStandardMaterial({ color: 0x111, emissive: 0x19c9ff, emissiveIntensity: .9 }));
  band.position.set(-300, 36, -600); scene.add(band);
}

// ---------- car ----------
const car = new THREE.Group(); scene.add(car);
const paint = new THREE.MeshPhysicalMaterial({ color: 0xb3150e, metalness: .15, roughness: .42, clearcoat: .6, clearcoatRoughness: .1 });
const black = new THREE.MeshStandardMaterial({ color: 0x0d0e10, roughness: .55, metalness: .3 });
const glass = new THREE.MeshPhysicalMaterial({ color: 0x0a0f16, metalness: .2, roughness: .03, clearcoat: 1 });
{
  const sh = new THREE.Shape();
  [[-41, 4], [-42, 12], [-35, 15.5], [-20, 16.5], [-4, 16.5], [16, 15.5], [33, 12.5], [41.5, 8.5], [41, 3.5]].forEach((p, i) => i ? sh.lineTo(p[0], p[1]) : sh.moveTo(p[0], p[1]));
  sh.closePath();
  const bg = new THREE.ExtrudeGeometry(sh, { depth: 30, bevelEnabled: true, bevelThickness: 3, bevelSize: 2.5, bevelSegments: 5, curveSegments: 8 });
  bg.translate(0, 0, -15);
  const body = new THREE.Mesh(bg, paint); body.castShadow = true; car.add(body);
  const cs = new THREE.Shape();
  [[-22, 16], [-9, 25.5], [6, 25.5], [19, 15.5]].forEach((p, i) => i ? cs.lineTo(p[0], p[1]) : cs.moveTo(p[0], p[1])); cs.closePath();
  const cg = new THREE.ExtrudeGeometry(cs, { depth: 22, bevelEnabled: true, bevelThickness: 2, bevelSize: 1.5, bevelSegments: 4 }); cg.translate(0, 0, -11);
  const cab = new THREE.Mesh(cg, glass); cab.castShadow = true; car.add(cab);
  const roof = new THREE.Mesh(new THREE.BoxGeometry(14, 1.2, 20), paint); roof.position.set(-1.5, 27.2, 0); car.add(roof);
  const wingM = black;
  const wing = new THREE.Mesh(new THREE.BoxGeometry(7, 1.2, 32), wingM); wing.position.set(-38, 21.5, 0); wing.rotation.z = 0.08; wing.castShadow = true; car.add(wing);
  for (const s of [-1, 1]) { const st = new THREE.Mesh(new THREE.BoxGeometry(2.4, 5, 1.2), wingM); st.position.set(-37, 18.5, 8 * s); car.add(st); }
  const diff = new THREE.Mesh(new THREE.BoxGeometry(4, 3, 26), black); diff.position.set(-42.5, 3.5, 0); car.add(diff);
  const split = new THREE.Mesh(new THREE.BoxGeometry(6, 1.2, 32), black); split.position.set(41, 2.4, 0); car.add(split);
  for (const s of [-1, 1]) {
    const tl = new THREE.Mesh(new THREE.BoxGeometry(1, 2.2, 10), new THREE.MeshStandardMaterial({ color: 0x300, emissive: 0xff1a12, emissiveIntensity: 3 }));
    tl.position.set(-44.2, 12.3, 10 * s); car.add(tl);
    const hl = new THREE.Mesh(new THREE.BoxGeometry(1, 1.8, 8), new THREE.MeshStandardMaterial({ color: 0x333, emissive: 0xeaf6ff, emissiveIntensity: 2.2 }));
    hl.position.set(43, 9.5, 11 * s); hl.rotation.y = -0.25 * s; car.add(hl);
    const skirt = new THREE.Mesh(new THREE.BoxGeometry(46, 3, 1.5), black); skirt.position.set(0, 3.5, 17.6 * s); car.add(skirt);
  }
}
const tireG = new THREE.CylinderGeometry(9, 9, 7, 28); tireG.rotateX(Math.PI / 2);
const rimG = new THREE.CylinderGeometry(6.2, 6.2, 7.4, 24); rimG.rotateX(Math.PI / 2);
const tireM = new THREE.MeshStandardMaterial({ color: 0x141414, roughness: .9 });
const rimM = new THREE.MeshStandardMaterial({ color: 0xb8bec6, metalness: 1, roughness: .25 });
const wheels = [], pivots = [];
for (const [x, z, front] of [[25, 17, 1], [25, -17, 1], [-25, 17, 0], [-25, -17, 0]]) {
  const piv = new THREE.Group(); piv.position.set(x, 9, z); car.add(piv);
  const wg = new THREE.Group(); piv.add(wg);
  const t = new THREE.Mesh(tireG, tireM); t.castShadow = true; wg.add(t);
  wg.add(new THREE.Mesh(rimG, rimM));
  for (let i = 0; i < 5; i++) { const sp = new THREE.Mesh(new THREE.BoxGeometry(1.8, 11, 8), black); sp.rotation.z = i * Math.PI * 2 / 5; wg.add(sp); }
  const cal = new THREE.Mesh(new THREE.BoxGeometry(4, 4, 8.2), new THREE.MeshStandardMaterial({ color: 0xffc400, roughness: .4 })); cal.position.set(3, 2, 0); piv.add(cal);
  wheels.push(wg); if (front) pivots.push(piv);
}

// ---------- sensors / trail ----------
const rayGroup = new THREE.Group(); scene.add(rayGroup);
const beams = [], glows = [], hits = [], rings = [];
for (let i = 0; i < 5; i++) {
  const bm = new THREE.Mesh(new THREE.BoxGeometry(1, 0.9, 0.9), new THREE.MeshBasicMaterial({ color: 0x67e3ff, transparent: true, opacity: .95, blending: THREE.AdditiveBlending, depthWrite: false }));
  const gl = new THREE.Mesh(new THREE.BoxGeometry(1, 4, 4), new THREE.MeshBasicMaterial({ color: 0x67e3ff, transparent: true, opacity: .16, blending: THREE.AdditiveBlending, depthWrite: false }));
  const ht = new THREE.Mesh(new THREE.SphereGeometry(2.6, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  const rg = new THREE.Mesh(new THREE.RingGeometry(4.5, 6, 32), new THREE.MeshBasicMaterial({ color: 0x67e3ff, transparent: true, opacity: .8, side: THREE.DoubleSide, depthWrite: false }));
  rg.rotation.x = -Math.PI / 2;
  rayGroup.add(bm, gl, ht, rg); beams.push(bm); glows.push(gl); hits.push(ht); rings.push(rg);
}
const sweep = new THREE.Mesh(new THREE.CircleGeometry(RAY_MAX, 32, 0, 0.32), new THREE.MeshBasicMaterial({ color: 0xd678ff, transparent: true, opacity: .16, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
sweep.rotation.x = -Math.PI / 2; scene.add(sweep);
const TRAIL = 260;
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(TRAIL * 2 * 3), 3));
trailGeo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(TRAIL * 2 * 3), 3));
{ const ind = []; for (let i = 0; i < TRAIL - 1; i++) { const a = i * 2; ind.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); } trailGeo.setIndex(ind); }
const trail = new THREE.Mesh(trailGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: .9, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
scene.add(trail);

const rayCol = d => d < 45 ? 0xff4d4d : d < 90 ? 0xffb02e : 0x67e3ff;
function stepAt(kf) {
  const k = Math.floor(kf), f = kf - k;
  const a = steps[Math.max(0, k)], b = steps[Math.min(steps.length - 1, k + 1)];
  let dh = b.h - a.h; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
  return { ...a, x: a.x + (b.x - a.x) * f, z: a.z + (b.z - a.z) * f, h: a.h + dh * f, dist: a.dist + (b.dist - a.dist) * f, steer: a.steer + (b.steer - a.steer) * f };
}
function placeCar(kf, opts = {}) {
  const st = stepAt(kf);
  car.position.set(st.x, 0.6, st.z); car.rotation.y = -st.h;
  for (const w of wheels) w.rotation.z = -st.dist / 9;
  for (const p of pivots) p.rotation.y = -st.steer * Math.PI / 180;
  // sun follows
  sun.position.set(st.x + sunDir.x * 1200, sunDir.y * 1200, st.z + sunDir.z * 1200); sun.target.position.set(st.x, 0, st.z);
  // rays
  rayGroup.visible = !!opts.rays;
  if (opts.rays) {
    const k = Math.floor(kf), rs = steps[k].rays;
    rs.forEach((r, i) => {
      const ox = st.x + Math.cos(st.h) * 30, oz = st.z + Math.sin(st.h) * 30;
      const ang = st.h + r.a * Math.PI / 180, ex = ox + Math.cos(ang) * r.d, ez = oz + Math.sin(ang) * r.d;
      const mx = (ox + ex) / 2, mz = (oz + ez) / 2, len = r.d, y = 7;
      const c = rayCol(r.d);
      for (const m of [beams[i], glows[i]]) { m.position.set(mx, y, mz); m.scale.set(len, 1, 1); m.rotation.set(0, -ang, 0); m.material.color.setHex(c); }
      hits[i].visible = rings[i].visible = r.hit;
      hits[i].position.set(ex, y, ez); rings[i].position.set(ex, 1.2, ez); rings[i].material.color.setHex(c);
      r._sx = ex; r._sz = ez;
    });
  }
  // trail
  trail.visible = !!opts.trail;
  if (opts.trail) {
    const pos = trailGeo.attributes.position, col = trailGeo.attributes.color;
    for (let i = 0; i < TRAIL; i++) {
      const q = stepAt(Math.max(0, kf - (TRAIL - 1 - i) * 1.0)), l = [Math.sin(q.h), -Math.cos(q.h)];
      const a = i / (TRAIL - 1);
      pos.setXYZ(i * 2, q.x + l[0] * 1.6, 1.5, q.z + l[1] * 1.6); pos.setXYZ(i * 2 + 1, q.x - l[0] * 1.6, 1.5, q.z - l[1] * 1.6);
      col.setXYZ(i * 2, .25 * a, .85 * a, 1 * a); col.setXYZ(i * 2 + 1, .25 * a, .85 * a, 1 * a);
    }
    pos.needsUpdate = col.needsUpdate = true; trailGeo.computeBoundingSphere();
  }
  return st;
}

// ---------- overlays helpers ----------
const $ = id => document.getElementById(id);
const layers = ['title', 'chase', 'lab', 'sens', 'rw', 'ana', 'end'];
function show(name) { layers.forEach(l => $('L-' + l).style.display = l === name ? 'block' : 'none'); const wm = $('wm'); wm.style.display = (name === 'end' || name === 'title') ? 'none' : 'block'; const right = name === 'lab' || name === 'rw'; wm.style.left = right ? 'auto' : '28px'; wm.style.right = right ? '28px' : 'auto'; wm.style.bottom = name === 'lab' ? '1046px' : '22px'; wm.style.top = 'auto'; wm.style.fontSize = name === 'lab' ? '11px' : '13px'; if (name === 'lab') { wm.style.bottom = '34px'; wm.style.right = '50px'; } }
const ease = t => t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
const fmtS = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
// track bounds for 2D maps
let bx0 = 1e9, bx1 = -1e9, bz0 = 1e9, bz1 = -1e9; for (const p of T.pts) { bx0 = Math.min(bx0, p[0]); bx1 = Math.max(bx1, p[0]); bz0 = Math.min(bz0, p[1]); bz1 = Math.max(bz1, p[1]); }
function mapper(w, h, pad) { const s = Math.min((w - 2 * pad) / (bx1 - bx0), (h - 2 * pad) / (bz1 - bz0)); const ox = (w - s * (bx1 - bx0)) / 2, oz = (h - s * (bz1 - bz0)) / 2; return (x, z) => [ox + (x - bx0) * s, oz + (z - bz0) * s]; }
function drawMini(cv, st) {
  const g = cv.getContext('2d'), mp = mapper(cv.width, cv.height, 10); g.clearRect(0, 0, cv.width, cv.height);
  g.lineWidth = 7; g.strokeStyle = 'rgba(255,255,255,.12)'; g.beginPath(); T.pts.forEach((p, i) => { const [x, y] = mp(p[0], p[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.closePath(); g.stroke();
  g.lineWidth = 2; g.strokeStyle = 'rgba(103,227,255,.75)'; g.stroke();
  const [cx, cy] = mp(st.x, st.z); g.fillStyle = '#ff3b2f'; g.beginPath(); g.arc(cx, cy, 6, 0, 7); g.fill(); g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke();
}
function chart(cv, k, key, lo, hi, color, opts = {}) {
  const g = cv.getContext('2d'), w = cv.width, h = cv.height, N = 300; g.clearRect(0, 0, w, h);
  g.fillStyle = 'rgba(255,255,255,.03)'; g.fillRect(0, 0, w, h);
  g.strokeStyle = 'rgba(255,255,255,.08)'; g.lineWidth = 1;
  for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(0, h * i / 4); g.lineTo(w, h * i / 4); g.stroke(); }
  if (lo < 0 && hi > 0) { const y0 = h - (0 - lo) / (hi - lo) * h; g.strokeStyle = 'rgba(255,255,255,.22)'; g.beginPath(); g.moveTo(0, y0); g.lineTo(w, y0); g.stroke(); }
  const series = Array.isArray(key) ? key : [[key, color]];
  for (const [kk, cc] of series) {
    g.beginPath();
    for (let i = 0; i < N; i++) { const q = steps[Math.max(0, k - N + 1 + i)]; const v = typeof kk === 'function' ? kk(q) : q[kk]; const x = i / (N - 1) * w, y = h - (v - lo) / (hi - lo) * h; i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.strokeStyle = cc; g.lineWidth = 2.2; g.stroke();
    if (opts.fill) { g.lineTo(w, h); g.lineTo(0, h); g.closePath(); const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, cc.replace('1)', '.28)')); gr.addColorStop(1, cc.replace('1)', '0)')); g.fillStyle = gr; g.fill(); }
  }
}
// engineer events (from the logged telemetry)
const brakeStarts = []; for (let k = 1; k < steps.length - 200; k++) if (steps[k].long < -0.3 && steps[k - 1].long >= -0.3) brakeStarts.push(k);
const events = [];
let lastEnd = -1e9;
for (const b of brakeStarts) {
  if (b < lastEnd + 15) continue;
  const pre = steps[b - 8], side = pre.rays[1].d < pre.rays[3].d ? 'left' : 'right';
  const turnDir = steps[b + 15].steer > 0 ? 'Right' : 'Left';
  events.push({ k: b - 8, txt: `${turnDir}-hander ahead. Forward ray ${pre.rays[2].d.toFixed(0)} px, DR +45° ${pre.rays[3].d.toFixed(0)} px, DL −45° ${pre.rays[1].d.toFixed(0)} px. Policy starts braking (long ${fmtS(steps[b].long)}).` });
  const sg = Math.sign(steps[b + 15].steer); let ap = b; for (let j = b; j < b + 110; j++) { if (Math.sign(steps[j].steer) !== sg && j > b + 20) break; if (steps[j].steer * sg > steps[ap].steer * sg) ap = j; }
  const A = steps[ap];
  events.push({ k: ap, txt: `Apex: steering ${Math.abs(A.steer).toFixed(1)}° ${A.steer > 0 ? 'right' : 'left'}, slip ${Math.abs(A.slip).toFixed(1)}°, speed ${A.v.toFixed(2)} px/step, ${Math.abs(A.signed).toFixed(0)} px ${A.signed * (A.steer > 0 ? -1 : 1) > 0 ? 'inside' : 'outside'} of centre.` });
  let ex = ap + 12; while (ex < ap + 160 && steps[ex].long < 0.6) ex++;
  lastEnd = ex;
  events.push({ k: ex, txt: `Exit: throttle ${fmtS(steps[ex].long)}, speed ${A.v.toFixed(2)} → ${steps[Math.min(ex + 40, steps.length - 1)].v.toFixed(2)} px/step. No track departure.` });
}
events.sort((a, b) => a.k - b.k);

// ---------- scenes ----------
const START = steps[0];
const SCENES = {
  title: { dur: 4, setup() { show('title'); }, frame(t) {
    const st = placeCar(0, {}); const a = 2.05 + t * 0.07, r = 215;
    camera.fov = 30; camera.setViewOffset(W, H, -430, 40, W, H);
    camera.position.set(st.x + Math.cos(a) * r, 34, st.z + Math.sin(a) * r); camera.lookAt(st.x + 4, 12, st.z);
    const e = ease((t - .5) / 1.0);
    $('L-title').style.opacity = e; $('L-title').style.transform = `translateY(${(1 - e) * 18}px)`;
  } },
  chase: { dur: 8, k0: LAP - 330, setup() { show('chase'); }, frame(t) {
    const kf = this.k0 + t * SPS, st = placeCar(kf, { trail: true });
    // lagged heading for the camera
    let hx = 0, hz = 0; for (let i = 0; i < 18; i++) { const q = stepAt(Math.max(0, kf - i * 1.5)); hx += Math.cos(q.h); hz += Math.sin(q.h); }
    const hc = Math.atan2(hz, hx);
    camera.fov = 52; camera.clearViewOffset();
    camera.position.set(st.x - Math.cos(hc) * 185, 52, st.z - Math.sin(hc) * 185);
    camera.lookAt(st.x + Math.cos(hc) * 70, 14, st.z + Math.sin(hc) * 70);
    const s = steps[Math.floor(kf)];
    $('hSpd').textContent = s.v.toFixed(2);
    $('hStr').textContent = (s.steer >= 0 ? '+' : '−') + Math.abs(s.steer).toFixed(1) + '°';
    const sb = $('hStrB'); sb.style.left = (50 + Math.min(0, s.steer) / 30 * 50) + '%'; sb.style.width = Math.abs(s.steer) / 30 * 50 + '%';
    $('hLong').textContent = fmtS(s.long);
    const lb = $('hLongB'); lb.style.background = s.long >= 0 ? '#3ee08f' : '#ff4d4d'; lb.style.left = (50 + Math.min(0, s.long) * 50) + '%'; lb.style.width = Math.abs(s.long) * 50 + '%';
    $('hLap').textContent = (s.prog * 100).toFixed(1) + '%';
    drawMini($('mmc'), st);
  } },
  lab: { dur: 9, k0: 790, setup() { show('lab'); }, frame(t) {
    const kf = this.k0 + t * SPS, st = placeCar(kf, { trail: true, rays: true }), k = Math.floor(kf), s = steps[k];
    let hx = 0, hz = 0; for (let i = 0; i < 18; i++) { const q = stepAt(Math.max(0, kf - i * 1.5)); hx += Math.cos(q.h); hz += Math.sin(q.h); }
    const hc = Math.atan2(hz, hx);
    camera.fov = 50; camera.setViewOffset(W, H, 232, 70, W, H);
    camera.position.set(st.x - Math.cos(hc) * 200, 78, st.z - Math.sin(hc) * 200);
    camera.lookAt(st.x + Math.cos(hc) * 60, 10, st.z + Math.sin(hc) * 60);
    $('lStep').textContent = k % LAP;
    chart($('ch0'), k, 'v', 0, 4.4, 'rgba(103,227,255,1)', { fill: true }); $('cv0').textContent = s.v.toFixed(2) + ' px/step';
    chart($('ch1'), k, 'steer', -32, 32, 'rgba(255,176,46,1)'); $('cv1').textContent = (s.steer >= 0 ? '+' : '−') + Math.abs(s.steer).toFixed(1) + '°';
    chart($('ch2'), k, [[q => Math.max(0, q.long), 'rgba(62,224,143,1)'], [q => Math.min(0, q.long), 'rgba(255,77,77,1)']], -1.05, 1.05); $('cv2').textContent = fmtS(s.long);
    chart($('ch3'), k, [['signed', 'rgba(214,120,255,1)'], [q => q.slip * 3, 'rgba(255,255,255,1)']], -30, 30); $('cv3').textContent = `${s.signed.toFixed(1)} px · ${s.slip.toFixed(1)}°`;
    $('raysbox').innerHTML = s.rays.map((r, i) => `<div class="rb"><div class="a">${['L −90°', 'DL −45°', 'FWD 0°', 'DR +45°', 'R +90°'][i]}</div><div class="d" style="color:#${rayCol(r.d).toString(16).padStart(6, '0')}">${r.d.toFixed(0)}</div></div>`).join('');
    const vis = events.filter(e => e.k <= k && e.k >= this.k0 - 200).slice(-4);
    $('feedL').innerHTML = vis.map((e, i) => {
      const isNew = i === vis.length - 1; const n = isNew ? Math.min(e.txt.length, Math.floor((k - e.k) * 3.2)) : e.txt.length;
      return `<div class="fl ${isNew ? '' : 'old'}"><span class="s">[STEP ${e.k % LAP}]</span><span>${e.txt.slice(0, n)}${isNew && n < e.txt.length ? '▍' : ''}</span></div>`;
    }).join('');
  } },
  sens: { dur: 6, k0: LAP + 1180, setup() { show('sens'); }, frame(t) {
    const kf = this.k0 + t * SPS, st = placeCar(kf, { rays: true, trail: true });
    const a = st.h + 2.5 - t * 0.16, r = 175;
    camera.fov = 48; camera.clearViewOffset();
    camera.position.set(st.x + Math.cos(a) * r, 120, st.z + Math.sin(a) * r);
    camera.lookAt(st.x + Math.cos(st.h) * 55, 0, st.z + Math.sin(st.h) * 55);
    sweep.visible = true; sweep.position.set(st.x, 2, st.z); sweep.rotation.z = -(t * 3.2) % (Math.PI * 2);
    const rs = steps[Math.floor(kf)].rays; camera.updateMatrixWorld();
    $('rlabels').innerHTML = rs.map(rr => {
      const v = new THREE.Vector3(rr._sx, 8, rr._sz).project(camera); const x = (v.x + 1) / 2 * W, y = (1 - v.y) / 2 * H;
      const c = '#' + rayCol(rr.d).toString(16).padStart(6, '0');
      return `<div class="rl" style="left:${x}px;top:${y}px;color:${c};border-color:${c}66">${rr.d.toFixed(0)} px</div>`;
    }).join('');
  } },
  rw: { dur: 8, k0: 2 * LAP + 300, setup() { show('rw'); $('rwLen').textContent = `${T.length.toFixed(0)} px · ${(T.length * 0.18 / 1000).toFixed(2)} km`; $('rwRad').textContent = `${stats.minRadius.toFixed(0)} px · ${(stats.minRadius * .18).toFixed(1)} m`; }, frame(t) {
    const kf = this.k0 + t * SPS, st = placeCar(kf, { trail: true });
    const cx = (bx0 + bx1) / 2, cz = (bz0 + bz1) / 2;
    const p = ease((t - 2.6) / 4.5);
    const a = -1.2 + t * 0.06, r = 2900 - p * 900, hgt = 3200 - p * 1700;
    camera.fov = 40; camera.clearViewOffset();
    camera.position.set(cx + Math.cos(a) * r * (1 - (1 - p) * .9), hgt, cz + Math.sin(a) * r * (1 - (1 - p) * .9));
    camera.lookAt(cx + (st.x - cx) * p * .35, 0, cz + (st.z - cz) * p * .35);
    car.scale.setScalar(2.6 - 1.0 * p); // keep the car readable from altitude
    // map overlay draws the OSM-style centerline, then fades out
    const mapA = 1 - ease((t - 2.4) / 1.4);
    const g = $('map').getContext('2d'); g.clearRect(0, 0, W, H);
    if (mapA > 0) {
      g.globalAlpha = mapA; g.fillStyle = '#0b1118'; g.fillRect(0, 0, W, H);
      g.strokeStyle = 'rgba(103,227,255,.07)'; g.lineWidth = 1;
      for (let x = 0; x < W; x += 80) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
      for (let y = 0; y < H; y += 80) { g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
      const mp = mapper(W, H, 170);
      seed = 99; g.strokeStyle = 'rgba(255,255,255,.08)'; g.lineWidth = 3;
      for (let i = 0; i < 90; i++) { const x = rnd() * W, y = rnd() * H, ang = (rnd() < .5 ? 0.35 : -1.22) + (rnd() - .5) * .15, L = 120 + rnd() * 380; g.lineWidth = rnd() < .2 ? 5 : 2; g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(ang) * L, y + Math.sin(ang) * L); g.stroke(); }
      const n = Math.floor(M * Math.min(1, t / 2.0));
      g.beginPath(); for (let i = 0; i <= n; i++) { const q = T.pts[i % M], [x, y] = mp(q[0], q[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.strokeStyle = '#67e3ff'; g.lineWidth = 5; g.shadowColor = '#67e3ff'; g.shadowBlur = 18; g.stroke(); g.shadowBlur = 0;
      for (let i = 0; i <= n; i += 60) { const q = T.pts[i % M], [x, y] = mp(q[0], q[1]); g.fillStyle = '#fff'; g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.fill(); }
      g.font = '600 16px DejaVu Sans Mono'; g.fillStyle = 'rgba(255,255,255,.5)'; g.fillText('way · highway=raceway · nodes resampled @ 4 px', 1400, 1000);
      g.globalAlpha = 1;
    }
    const stage = t < 1.2 ? 0 : t < 2.4 ? 1 : t < 4.2 ? 2 : 3;
    [...$('psteps').children].forEach((c, i) => c.className = i === stage ? 'on' : '');
    $('rwInfo').style.opacity = ease((t - 4.5) / .8);
  } },
  ana: { dur: 6.5, setup() {
    show('ana'); car.scale.setScalar(1);
    const st = stats, tl = [['Lap time', st.lapSteps, 'steps'], ['Avg speed', st.avgV.toFixed(2), 'px/st'], ['Max speed', st.maxV.toFixed(2), 'px/st'], ['Departures', st.departures, ''], ['Collisions', st.collisions, ''], ['Braking', st.brakePct.toFixed(1), '%'], ['Avg |lat dev|', st.avgDev.toFixed(1), 'px'], ['Steer Δ/step', st.smooth.toFixed(2), '°']];
    this.tl = tl;
  }, frame(t) {
    const st = placeCar(LAP - 2, {}); const cx = (bx0 + bx1) / 2, cz = (bz0 + bz1) / 2;
    camera.fov = 40; camera.clearViewOffset(); camera.position.set(cx + 400, 2600, cz + 1500); camera.lookAt(cx, 0, cz);
    const e = ease(t / 0.8); $('ana').style.opacity = e; $('ana').style.transform = `scale(${0.97 + 0.03 * e})`;
    $('tiles').innerHTML = this.tl.map(([l, v, u], i) => { const a = ease((t - 0.4 - i * 0.12) / 0.5); return `<div class="tile" style="opacity:${a}"><div class="lbl" style="font-size:11px">${l}</div><div class="n">${v}<small>${u}</small></div></div>`; }).join('');
    const p = ease((t - 0.8) / 3.2), lap = steps.slice(0, LAP), n = Math.floor(LAP * p);
    // speed map
    const mc = $('aMap'), g = mc.getContext('2d'), mp = mapper(mc.width, mc.height, 24); g.clearRect(0, 0, mc.width, mc.height);
    g.fillStyle = 'rgba(255,255,255,.03)'; g.fillRect(0, 0, mc.width, mc.height);
    g.lineWidth = 12; g.strokeStyle = 'rgba(255,255,255,.08)'; g.beginPath(); T.pts.forEach((q, i) => { const [x, y] = mp(q[0], q[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.closePath(); g.stroke();
    g.lineWidth = 6; g.lineCap = 'round';
    for (let i = 1; i < n; i++) { const a = lap[i - 1], b = lap[i]; const u = (b.v - 2.2) / (VMAX - 2.2); g.strokeStyle = `hsl(${(1 - u) * 0 + u * 190},90%,${55 - u * 5}%)`; const [x0, y0] = mp(a.x, a.z), [x1, y1] = mp(b.x, b.z); g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); }
    g.font = '600 13px Inter'; g.fillStyle = '#8fa3b8'; g.fillText('slow', 24, mc.height - 16); g.fillText('fast', 160, mc.height - 16);
    const gr = g.createLinearGradient(60, 0, 150, 0); gr.addColorStop(0, 'hsl(0,90%,55%)'); gr.addColorStop(1, 'hsl(190,90%,50%)'); g.fillStyle = gr; g.fillRect(62, mc.height - 26, 90, 10);
    // traces
    const trace = (cv, key, lo, hi, col, shade) => {
      const c = cv.getContext('2d'), w = cv.width, h = cv.height; c.clearRect(0, 0, w, h); c.fillStyle = 'rgba(255,255,255,.03)'; c.fillRect(0, 0, w, h);
      if (shade) { c.fillStyle = 'rgba(255,77,77,.16)'; for (let i = 0; i < n; i++) if (lap[i].long < -0.05) c.fillRect(i / LAP * w, 0, w / LAP + 1, h); }
      c.strokeStyle = 'rgba(255,255,255,.08)'; for (let i = 1; i < 4; i++) { c.beginPath(); c.moveTo(0, h * i / 4); c.lineTo(w, h * i / 4); c.stroke(); }
      c.beginPath(); for (let i = 0; i < n; i++) { const x = i / LAP * w, y = h - (lap[i][key] - lo) / (hi - lo) * h; i ? c.lineTo(x, y) : c.moveTo(x, y); } c.strokeStyle = col; c.lineWidth = 2.4; c.stroke();
      c.font = '12px DejaVu Sans Mono'; c.fillStyle = '#8fa3b8'; c.fillText(hi.toString(), 6, 14); c.fillText(lo.toString(), 6, h - 6);
    };
    trace($('aSpd'), 'v', 2, 4.2, '#67e3ff', true); trace($('aStr'), 'steer', -32, 32, '#ffb02e');
  } },
  end: { dur: 3.5, setup() { show('end'); }, frame(t) { $('end').style.opacity = ease(t / .7); } },
};
window.SCENE_LIST = Object.entries(SCENES).map(([k, v]) => [k, v.dur]);
let curScene = null;
window.setScene = (name) => { curScene = SCENES[name]; sweep.visible = false; car.scale.setScalar(1); curScene.setup(); };
window.renderAt = (t) => { curScene.frame(t); camera.updateProjectionMatrix(); renderer.render(scene, camera); return true; };
window.READY = true;
