/**
 * Tour mode: /demo?tour=1 plays an automated ~90 s sequence for screen recording.
 * Uses ONLY the existing recorded / precomputed runs (BIC eval, BIC grip 1.0 G, precomputed standard-track
 * rollout seed 1001). No new data, no policy calls beyond the existing /simulation/rollout (cached) in the API
 * build; in the static build every fetch is mapped to static files by static_shim.js. Esc exits.
 *
 * Timeline (seconds): 0 title/orbit · 6 BIC chase+cinematic HUD · 30 T8 orbit + sensor labels ·
 * 42 lab layout (precomputed std-track rollout: charts + engineer feed) · 52 grip 1.0 G into T1, hold on failure ·
 * 70 overhead BIC lap -> post-lap analysis · 82 end card.
 */
(function () {
  let active = false;
  try { active = new URLSearchParams(window.location.search).get("tour") === "1"; } catch (e) { /* ignore */ }
  if (!active) return;

  const SPACE_URL = "huggingface.co/spaces/prateekraj91/rl-racing-lab";
  const REPO_URL = "github.com/prateekraj91/RL-Racing-Agent";
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const body = document.body;
  body.classList.add("tour-mode", "tour-nohud", "tour-nolabels");

  // ---- overlay elements -------------------------------------------------------------------------
  const fade = document.createElement("div");
  fade.id = "tourFade"; fade.className = "tour-fade on";
  const cap = document.createElement("div");
  cap.id = "tourCaption"; cap.className = "tour-caption";
  const end = document.createElement("div");
  end.id = "tourEnd"; end.className = "tour-end";
  end.innerHTML = `<div class="te-title">Next: grip-aware retraining</div>
    <div class="te-sub">live demo + code</div>
    <div class="te-url">${SPACE_URL}</div><div class="te-url">${REPO_URL}</div>`;
  body.append(fade, cap, end);

  let capTimer = 0;
  function caption(text, opts = {}) {
    clearTimeout(capTimer);
    cap.classList.remove("show");
    capTimer = setTimeout(() => {
      if (!text) return;
      cap.textContent = text;
      cap.classList.toggle("title", !!opts.title);
      cap.classList.toggle("high", !!opts.high);
      cap.classList.remove("top");
      cap.classList.add("show");
    }, 450);
  }
  async function fadeTo(black, ms = 380) { fade.style.transitionDuration = ms + "ms"; fade.classList.toggle("on", black); await sleep(ms + 20); }
  async function cut(fn) { await fadeTo(true); await fn(); await sleep(250); await fadeTo(false, 450); }

  // ---- playback / camera helpers ----------------------------------------------------------------
  // The playhead is driven by WALL-CLOCK time (not frame deltas), so the sequence keeps its timing at any
  // frame rate. The normal playback clock is paused (isPlaying=false); renderPlaybackFrame still draws.
  let drv = null;
  function drive(from, to, seconds, then) {
    STATE.playback.isPlaying = false;
    drv = { from, to, seconds, t: performance.now(), then, done: false };
  }
  function driveTick() {
    if (!drv) return;
    const k = Math.min(1, (performance.now() - drv.t) / 1000 / drv.seconds);
    STATE.playback.playhead = drv.from + (drv.to - drv.from) * k;
    if (k >= 1 && !drv.done) { drv.done = true; if (drv.then) drv.then(); }
  }
  function hold() { drv = null; STATE.playback.isPlaying = false; }
  function jump(step, mode) {
    STATE.playback.isPlaying = false;
    clearTrajectoryTrail();
    STATE.playback.playhead = step; STATE.playback.lastStepIndex = -1;
    renderPlaybackFrame(step, 0.016);
    const h = carGroup.rotation.y;
    chaseForwardX = Math.cos(h); chaseForwardZ = -Math.sin(h);
    setCameraMode(mode);
    const d = desiredCameraPose(mode, carGroup.position, 0.016);
    STATE.cameraPos.copy(d.pos); STATE.cameraTarget.copy(d.target); STATE.cameraUp.copy(d.up); STATE.cameraFov = d.fov;
    STATE.camFromPos.copy(d.pos); STATE.camFromTarget.copy(d.target); STATE.camFromUp.copy(d.up); STATE.camFromFov = d.fov;
    cameraTransT = CAMERA_TRANSITION_S;
    camera.position.copy(d.pos); camera.up.copy(d.up); camera.fov = d.fov; camera.near = d.near; camera.far = d.far || 9000;
    camera.updateProjectionMatrix(); camera.lookAt(d.target);
    renderPlaybackFrame(step, 0.016);
  }
  function setHud(on) { body.classList.toggle("tour-nohud", !on); }
  function setLabels(on) { body.classList.toggle("tour-nolabels", !on); }
  function setCinematic(on) { if (STATE.cinematicMode !== on) toggleCinematicUIMode(); }
  const modalOff = () => { window.__tourNoModal = true; hidePostLapModal(); };

  async function toBic(run) {
    modalOff();
    await window.recordedLoad(run);
    STATE.playback.nominalDt = 1 / 60;
    STATE.playback.isPlaying = false;
  }
  async function toPrecomputed() {
    modalOff();
    STATE.source = "live";
    body.classList.remove("recorded-source");
    const h = $("recordedHud"); if (h) h.remove();
    STATE.playback.nominalDt = 0.030; STATE.playback.speed = 1.0;
    await startSimulation("standard_medium", 1001);
    STATE.playback.isPlaying = false;
  }

  // ---- the sequence -----------------------------------------------------------------------------
  const STEPS = [
    { t: 0, run: async () => {
      jump(0, "orbit"); setCinematic(true); setHud(false); setLabels(false);
      await fadeTo(false, 900);
      caption("RL Racing Lab: a frozen SAC policy on a real F1 circuit", { title: true });
    } },
    { t: 6, run: async () => { caption(""); await cut(async () => {
      jump(250, "chase"); setHud(true); setLabels(false); drive(250, 1690, 23.5);
    }); caption("Buddh International Circuit, imported from OpenStreetMap · 0 departures · mean deviation 0.059 m"); } },
    { t: 30, run: async () => { caption(""); await cut(async () => {
      jump(2780, "orbit"); setLabels(true); drive(2780, 3140, 11.5);
    }); caption("5 rangefinders, raw pixel distances"); } },
    { t: 42, run: async () => { caption(""); await cut(async () => {
      await toPrecomputed(); setCinematic(false); setHud(false); setLabels(false);
      jump(0, "chase"); drive(0, 386, 9.5);
    }); caption("Precomputed deterministic rollout (frozen policy) · live charts + grounded engineer feed", { high: true }); } },
    { t: 52, run: async () => { caption(""); await cut(async () => {
      await toBic("bic_g1p0"); setCinematic(true); setHud(true); setLabels(false);
      jump(330, "chase"); drive(330, STATE.playback.maxStep, 5.2);
    }); setTimeout(() => caption("Realistic grip (~1 G): off track at T1, full speed, 0% braking"), 1500); } },
    { t: 70, run: async () => { caption(""); await cut(async () => {
      await toBic(""); window.__tourNoModal = false; setHud(false); setLabels(false);
      jump(0, "overhead"); drive(0, STATE.playback.maxStep, 6, () => { showPostLapModal(); cap.classList.add("top"); }); // full lap in 6 s, then post-lap analysis
    }); caption("It learned path-tracking with infinite grip. It never learned to brake."); } },
    { t: 82, run: async () => { caption(""); await fadeTo(true, 700); hidePostLapModal(); hold();
      body.classList.add("tour-ended"); end.classList.add("show"); } },
  ];

  let t0 = 0, idx = 0, busy = false, stopped = false;
  function tick() {
    if (stopped) return;
    driveTick();
    const T = (performance.now() - t0) / 1000;
    if (!busy && idx < STEPS.length && T >= STEPS[idx].t) {
      const s = STEPS[idx++]; busy = true;
      Promise.resolve(s.run()).catch((e) => console.error("[tour] step failed", e)).finally(() => { busy = false; });
    }
    requestAnimationFrame(tick);
  }

  function exitTour() {
    stopped = true;
    window.location.href = window.location.pathname; // back to the normal landing page
  }
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") exitTour(); });
  window.__tour = { exit: exitTour, steps: STEPS.map((s) => s.t), T: () => (t0 ? (performance.now() - t0) / 1000 : -1) };

  // Start once the initial BIC run is loaded; warm the data the later stages need.
  window.addEventListener("recorded-loaded", async function once() {
    window.removeEventListener("recorded-loaded", once);
    try {
      fetch("/recorded/bic?run=bic_g1p0").then((r) => r.arrayBuffer()).catch(() => {});
      fetch("/simulation/rollout", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset_id: "standard_medium", track_seed: 1001, commentary_mode: "offline", saliency_interval: 5 }) })
        .then((r) => r.arrayBuffer()).catch(() => {});
    } catch (e) { /* warm-up is best effort */ }
    await sleep(1200);
    t0 = performance.now();
    window.__tourStarted = true;
    requestAnimationFrame(tick);
  });
})();
