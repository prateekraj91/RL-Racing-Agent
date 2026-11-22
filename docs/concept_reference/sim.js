// Scripted concept lap. Units follow the real RacingEnv conventions:
// distances in px, speed in px/step, angles in degrees. Not policy output.
export const HALF_W = 35;          // track half-width (env corridor 70 px)
export const WHEELBASE = 50;       // env.car.wheelbase
export const RAY_ANGLES = [-90, -45, 0, 45, 90];
export const RAY_MAX = 200;
export const VMAX = 4.0;

function wrap(i, n) { return ((i % n) + n) % n; }

export function buildTrack() {
  const C = [[-1100, -720], [-300, -730], [600, -720], [1060, -640], [1260, -330], [1270, 150],
    [1180, 500], [920, 590], [760, 420], [720, 140], [480, 40], [250, 190], [40, 70],
    [-200, 330], [-600, 590], [-1020, 560], [-1270, 260], [-1320, -200], [-1260, -560]];
  const n = C.length, SUB = 240, raw = [];
  for (let i = 0; i < n; i++) {
    const p0 = C[wrap(i - 1, n)], p1 = C[i], p2 = C[wrap(i + 1, n)], p3 = C[wrap(i + 2, n)];
    for (let k = 0; k < SUB; k++) {
      const t = k / SUB, t2 = t * t, t3 = t2 * t;
      raw.push([0, 1].map(d => 0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3)));
    }
  }
  const N = raw.length;
  // resample to uniform 4 px spacing
  const cum = [0];
  for (let i = 1; i <= N; i++) {
    const a = raw[i - 1], b = raw[i % N];
    cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  const L = cum[N], DS = 4, M = Math.round(L / DS), pts = [];
  let j = 0;
  for (let k = 0; k < M; k++) {
    const s = (k / M) * L;
    while (cum[j + 1] < s) j++;
    const f = (s - cum[j]) / (cum[j + 1] - cum[j]);
    const a = raw[j], b = raw[(j + 1) % N];
    pts.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
  }
  return finishPolyline(pts, L / M);
}

function finishPolyline(pts, ds) {
  const M = pts.length, tan = [], left = [], head = [], kap = [];
  for (let i = 0; i < M; i++) {
    const a = pts[wrap(i - 1, M)], b = pts[wrap(i + 1, M)];
    let tx = b[0] - a[0], tz = b[1] - a[1];
    const l = Math.hypot(tx, tz); tx /= l; tz /= l;
    tan.push([tx, tz]); left.push([tz, -tx]); head.push(Math.atan2(tz, tx));
  }
  for (let i = 0; i < M; i++) {
    let dh = head[wrap(i + 1, M)] - head[wrap(i - 1, M)];
    while (dh > Math.PI) dh -= 2 * Math.PI;
    while (dh < -Math.PI) dh += 2 * Math.PI;
    kap.push(dh / (2 * ds));
  }
  return { pts, tan, left, head, kap, ds, M, length: ds * M };
}

function smooth(arr, w) {
  const M = arr.length, out = new Array(M).fill(0);
  for (let i = 0; i < M; i++) {
    let s = 0, ws = 0;
    for (let k = -w; k <= w; k++) { const g = Math.exp(-(k * k) / (2 * (w / 2) ** 2)); s += arr[wrap(i + k, M)] * g; ws += g; }
    out[i] = s / ws;
  }
  return out;
}

function raySeg(ox, oz, dx, dz, ax, az, bx, bz) {
  const ex = bx - ax, ez = bz - az, den = dx * ez - dz * ex;
  if (Math.abs(den) < 1e-9) return Infinity;
  const t = ((ax - ox) * ez - (az - oz) * ex) / den;
  const u = ((ax - ox) * dz - (az - oz) * dx) / den;
  return (t >= 0 && u >= 0 && u <= 1) ? t : Infinity;
}

export function buildSim() {
  const T = buildTrack();
  const { M } = T;
  const edgeL = T.pts.map((p, i) => [p[0] + HALF_W * T.left[i][0], p[1] + HALF_W * T.left[i][1]]);
  const edgeR = T.pts.map((p, i) => [p[0] - HALF_W * T.left[i][0], p[1] - HALF_W * T.left[i][1]]);
  // racing line: shift toward the inside of corners
  const ks = smooth(T.kap, 50);
  const off = ks.map(k => Math.max(-22, Math.min(22, -k * 3600)));
  const offS = smooth(off, 25);
  const pathPts = T.pts.map((p, i) => [p[0] + offS[i] * T.left[i][0], p[1] + offS[i] * T.left[i][1]]);
  const P = finishPolyline(pathPts, T.ds); // approx (spacing slightly varies)
  // true cumulative distance along path
  const cum = [0];
  for (let i = 1; i <= M; i++) { const a = pathPts[i - 1], b = pathPts[i % M]; cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
  const PL = cum[M];
  const kp = smooth(P.kap, 6);
  // speed profile
  const A = 0.075, ACC = 0.02, BRK = 0.05;
  let v = kp.map(k => Math.min(VMAX, Math.sqrt(A / Math.max(1e-6, Math.abs(k)))));
  for (let it = 0; it < 3; it++) {
    for (let i = 1; i <= M; i++) { const a = wrap(i - 1, M), b = wrap(i, M), ds = cum[i] - cum[i - 1]; v[b] = Math.min(v[b], Math.sqrt(v[a] ** 2 + 2 * ACC * ds)); }
    for (let i = M; i >= 1; i--) { const a = wrap(i, M), b = wrap(i - 1, M), ds = cum[i] - cum[i - 1]; v[b] = Math.min(v[b], Math.sqrt(v[a] ** 2 + 2 * BRK * ds)); }
  }
  const sample = (arr, s) => {
    s = ((s % PL) + PL) % PL;
    let lo = 0, hi = M;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m; }
    const f = (s - cum[lo]) / (cum[lo + 1] - cum[lo]);
    return [lo, f];
  };
  const lerp = (a, b, f) => a + (b - a) * f;
  const angLerp = (a, b, f) => { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * f; };

  // integrate several laps worth of steps
  const steps = [];
  let s = 0, cIdx = 0, dist = 0;
  const NSTEPS = 6000;
  for (let k = 0; k < NSTEPS; k++) {
    const [i, f] = sample(null, s), i2 = wrap(i + 1, M);
    const x = lerp(pathPts[i][0], pathPts[i2][0], f), z = lerp(pathPts[i][1], pathPts[i2][1], f);
    const h = angLerp(P.head[i], P.head[i2], f);
    const vk = lerp(v[i], v[i2], f), kk = lerp(kp[i], kp[i2], f);
    // nearest centerline index (local search)
    let best = 1e18, bi = cIdx;
    for (let d = -40; d <= 40; d++) { const q = T.pts[wrap(cIdx + d, M)], dd = (q[0] - x) ** 2 + (q[1] - z) ** 2; if (dd < best) { best = dd; bi = wrap(cIdx + d, M); } }
    cIdx = bi;
    const lc = T.left[cIdx], cp = T.pts[cIdx];
    const signed = (x - cp[0]) * lc[0] + (z - cp[1]) * lc[1];
    let herr = (h - T.head[cIdx]) * 180 / Math.PI; while (herr > 180) herr -= 360; while (herr < -180) herr += 360;
    // rays
    const rays = RAY_ANGLES.map(a => {
      const ang = h + a * Math.PI / 180, dx = Math.cos(ang), dz = Math.sin(ang);
      let t = RAY_MAX;
      for (let d = -70; d <= 70; d++) {
        const a0 = wrap(cIdx + d, M), a1 = wrap(a0 + 1, M);
        t = Math.min(t, raySeg(x, z, dx, dz, edgeL[a0][0], edgeL[a0][1], edgeL[a1][0], edgeL[a1][1]));
        t = Math.min(t, raySeg(x, z, dx, dz, edgeR[a0][0], edgeR[a0][1], edgeR[a1][0], edgeR[a1][1]));
      }
      return { a, d: t, hit: t < RAY_MAX, ex: x + dx * t, ez: z + dz * t };
    });
    const steer = Math.max(-30, Math.min(30, Math.atan(WHEELBASE * kk) * 180 / Math.PI));
    steps.push({ k, x, z, h, v: vk, steer, signed, herr, kap: kk, rays, s, dist, lap: Math.floor(s / PL), prog: (s % PL) / PL, cIdx });
    s += vk; dist += vk;
  }
  // longitudinal action, slip, yaw rate, accel from consecutive steps
  for (let k = 0; k < NSTEPS; k++) {
    const a = steps[k], b = steps[Math.min(k + 1, NSTEPS - 1)];
    const dv = b.v - a.v;
    a.acc = dv;
    a.long = dv > 1e-4 ? Math.min(1, Math.max(dv / ACC, 0.25 + 0.5 * a.v / VMAX)) : dv < -1e-4 ? Math.max(-1, dv / BRK) : 0.35;
    a.lat = a.v * a.v * a.kap;
    a.slip = 7 * a.lat / A;
    a.yaw = a.v * a.kap * 180 / Math.PI;
  }
  // smooth the action signal a touch
  const lg = steps.map(q => q.long);
  for (let k = 0; k < NSTEPS; k++) { let sm = 0, c = 0; for (let d = -2; d <= 2; d++) { const q = lg[Math.max(0, Math.min(NSTEPS - 1, k + d))]; sm += q; c++; } steps[k].long = sm / c; }
  // one-lap stats (lap 0)
  const lap = steps.filter(q => q.lap === 0);
  const sp = lap.map(q => q.v);
  const stats = {
    lapSteps: lap.length,
    avgV: sp.reduce((a, b) => a + b, 0) / sp.length,
    maxV: Math.max(...sp),
    departures: lap.filter(q => Math.abs(q.signed) > HALF_W).length,
    collisions: 0,
    smooth: lap.slice(1).reduce((a, q, i) => a + Math.abs(q.steer - lap[i].steer), 0) / (lap.length - 1),
    brakePct: 100 * lap.filter(q => q.long < -0.05).length / lap.length,
    avgDev: lap.reduce((a, q) => a + Math.abs(q.signed), 0) / lap.length,
    maxSlip: Math.max(...lap.map(q => Math.abs(q.slip))),
    minRadius: Math.min(...T.kap.map(k => 1 / Math.max(1e-6, Math.abs(k)))),
  };
  return { T, edgeL, edgeR, pathPts, steps, stats, PL, ks };
}
