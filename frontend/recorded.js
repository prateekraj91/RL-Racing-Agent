/**
 * Recorded playback source: "Recorded: BIC" (?source=bic).
 *
 * Plays the SAVED BIC evaluation run (analysis/buddh/buddh_full_rollout.csv, 7,521 steps)
 * through the existing playback clock / interpolation / renderer. It never calls the policy
 * or any /predict, /step or /rollout endpoint: the only request is GET /recorded/bic, which
 * serves the stored CSV + track verbatim. Fields the CSV lacks are shown as "not recorded".
 * Only pose interpolation between recorded steps is performed (by renderPlaybackFrame).
 */
(function () {
  const NA = "not recorded";
  let REC = null; // { meta, cols }
  // Optional experiment run (?source=bic&run=bic_g1p0 | std_g0p8 ...). Same honest-unit rules as BIC playback.
  const RUN = (() => { try { return new URLSearchParams(window.location.search).get("run") || ""; } catch (e) { return ""; } })();
  const RUN_TAG = { off: "grip OFF (baseline)", g0p8: "grip limit 0.8 G", g1p0: "grip limit 1.0 G", g1p2: "grip limit 1.2 G" };
  const runLabel = () => { const [k, t] = RUN.split("_"); return `EXPERIMENT — ${RUN_TAG[t] || RUN}${k === "std" ? " (standard track)" : ""}`; };
  const isGrip = () => !!RUN;

  const $ = (id) => document.getElementById(id);
  const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : NA);
  const mmss = (sec) => `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;

  function setBadge() {
    const b = $("sourceBadge"), sw = $("sourceSwitch");
    if (!b) return;
    if (STATE.source === "bic") {
      b.className = "source-badge recorded";
      b.textContent = RUN ? runLabel() : "RECORDED PLAYBACK — BIC eval";
      if (sw) { sw.textContent = "◂ switch to LIVE rollout"; sw.href = window.location.pathname; }
    } else {
      b.className = "source-badge live";
      b.textContent = "LIVE — policy rollout";
      if (sw) { sw.textContent = "Recorded: BIC ▸"; sw.href = "?source=bic"; }
    }
  }

  function buildRollout(data) {
    const c = data.cols, m = data.meta, n = m.steps;
    const ang = [-90, -45, 0, 45, 90];
    const steps = new Array(n);
    for (let i = 0; i < n; i++) {
      const x = c.car_x[i], y = c.car_y[i], a = c.car_angle[i];
      const dists = [c.rays[0][i], c.rays[1][i], c.rays[2][i], c.rays[3][i], c.rays[4][i]];
      const rays = ang.map((off, k) => {
        const r = ((a + off) * Math.PI) / 180;
        return { angle_offset: off, distance: dists[k], start: [x, y],
                 end: [x + Math.cos(r) * dists[k], y - Math.sin(r) * dists[k]], is_close: dists[k] < 50 };
      });
      steps[i] = {
        vehicle: { x, y, angle: a, velocity: c.speed_px_step[i],
                   steering: c.steering_action[i] * m.max_steer_deg,
                   steering_angle_deg: c.steering_action[i] * m.max_steer_deg, // action x 30 deg lock (env.car.max_steering)
                   throttle: c.throttle_action[i] },
        action: [c.steering_action[i], c.throttle_action[i]],
        telemetry: { signed_distance: c.signed_dist_px[i], slip_angle: c.slip_deg[i], rays: dists },
        rays,
      };
    }
    return { steps, recorded: true, session: { is_crashed: c.crashed.some(Boolean) } };
  }

  function buildHud(m) {
    const hud = document.createElement("div");
    hud.id = "recordedHud"; hud.className = "recorded-hud";
    hud.innerHTML = `
      <div class="rh-title">${RUN ? runLabel() : "RECORDED RUN · BIC eval"}</div>
      <div class="rh-row"><span>STEP</span><b id="rhStep">1 / ${m.steps}</b></div>
      <div class="rh-row"><span>ELAPSED*</span><b id="rhElapsed">0.00 s</b></div>
      <div class="rh-bar"><div id="rhBarFill"></div></div>
      <div class="rh-row"><span>LAP PROGRESS</span><b id="rhLap">0.0 %</b></div>
      <div class="rh-sep"></div>
      <div class="rh-row"><span>SPEED</span><b id="rhSpd">0.00 px/step</b></div>
      <div class="rh-row"><span></span><b id="rhSpdM">0.000 m/step</b></div>
      <div class="rh-row sec"><span>≈ km/h (assumes 60 steps/s)</span><b id="rhKmh">0.0</b></div>
      <div class="rh-row"><span>LAT DEV</span><b id="rhLat">0 px</b></div>
      <div class="rh-row"><span>HEADING ERR</span><b id="rhHdg">0.00°</b></div>
      <div class="rh-row"><span>YAW-RATE</span><b id="rhSlip">0.00°</b></div>
      <div class="rh-note">kinematic yaw-rate (not tyre slip)</div>
      <div class="rh-row"><span>STEER ACTION</span><b id="rhSteer">0.00</b></div>
      ${RUN ? `<div class="rh-row"><span>LAT ACCEL (derived)</span><b id="rhAlat">0.00 G</b></div>
      <div class="rh-row"><span>GRIP LIMIT</span><b>${m.grip && m.grip.limit_g ? m.grip.limit_g.toFixed(1) + " G" : "OFF"}</b></div>
      <div class="rh-note na">Derived: lat accel = speed × |Δheading| per step, in G assuming 12 m = 70 px and 60 steps/s. Grip = existing Car.grip_limit understeer clamp (no speed loss). Not recorded: tyre slip, commentary. * assumes 60 steps/s.</div>`
      : `<div class="rh-note na">Not recorded: lateral accel, tyre slip/grip, commentary. * assumes 60 steps/s (physics has no time base); kinematic model, no grip limit.</div>`}`;
    document.body.appendChild(hud);
  }

  window.recordedHudUpdate = function (i) {
    if (!REC) return;
    const c = REC.cols, m = REC.meta;
    const s = (id, t) => { const e = $(id); if (e) e.textContent = t; };
    s("rhStep", `${c.step[i]} / ${m.steps}`);
    s("rhElapsed", `${fmt(c.time_s[i], 2)} s`);
    const lp = Math.min(1, Math.max(0, c.lap_progress[i]));
    const f = $("rhBarFill"); if (f) f.style.width = `${(lp * 100).toFixed(1)}%`;
    s("rhLap", `${(lp * 100).toFixed(1)} %`);
    s("rhSpd", `${fmt(c.speed_px_step[i], 3)} px/step`);
    s("rhSpdM", `${fmt(c.speed_px_step[i] / m.px_per_m, 3)} m/step`);
    s("rhKmh", fmt(c.speed_kmh[i], 1));
    s("rhLat", `${c.signed_dist_px[i] >= 0 ? "+" : ""}${fmt(c.signed_dist_px[i], 2)} px · ${fmt(c.signed_dist_m[i], 3)} m`);
    s("rhHdg", `${fmt(c.heading_err_deg[i], 2)}°`);
    s("rhSlip", `${fmt(c.slip_deg[i], 2)}°`);
    if (RUN && i > 0) {
      let d = c.car_angle[i] - c.car_angle[i - 1]; d = ((d + 540) % 360) - 180;
      s("rhAlat", `${((c.speed_px_step[i] * Math.abs(d) * Math.PI / 180) / (9.80665 / ((1 / m.px_per_m) * 3600))).toFixed(2)} G`);
    }
    s("rhSteer", `${c.steering_action[i] >= 0 ? "+" : ""}${fmt(c.steering_action[i], 3)} (×${m.max_steer_deg}°)`);
    // Overrides of the live-sim readouts that assume a different time base / labels
    s("stepClock", `STEP: ${c.step[i]} / ${m.steps}`);
    s("pbCurrentTime", mmss(c.time_s[i]));
    s("valLateralSlip", `${fmt(c.signed_dist_px[i], 1)} px • ${fmt(c.slip_deg[i], 1)}°`);
  };

  function mean(a) { let t = 0; for (const v of a) t += v; return t / a.length; }

  window.recordedPostLap = function () {
    const c = REC.cols, m = REC.meta, n = m.steps;
    const absM = c.signed_dist_m.map(Math.abs);
    const avgLat = mean(absM);
    const rms = Math.sqrt(mean(c.signed_dist_m.map((v) => v * v)));
    const maxLat = Math.max(...absM);
    const braking = c.throttle_action.filter((t) => t < 0).length;
    const departures = c.signed_dist_px.filter((d) => Math.abs(d) > m.half_width_px).length;
    const crashes = c.crashed.filter(Boolean).length;
    const jitter = mean(c.steer_change.map(Math.abs)) * m.max_steer_deg;
    const set = (id, html) => { const e = $(id); if (e) e.innerHTML = html; };
    const u = (v, unit) => `${v} <span class="kpi-unit">${unit}</span>`;
    set("kpiLapTime", u(n.toLocaleString("en-US"), `steps · ${c.time_s[n - 1].toFixed(2)} s*`));
    set("kpiAvgSpeed", u(mean(c.speed_px_step).toFixed(3), "px/step"));
    set("kpiMaxSpeed", u(Math.max(...c.speed_px_step).toFixed(3), "px/step"));
    set("kpiDepartures", `${departures}`);
    set("kpiCollisions", `${crashes}`);
    set("kpiBraking", u(((braking / n) * 100).toFixed(1), `% (${braking}/${n})`));
    set("kpiLatDev", u(avgLat.toFixed(3), "m mean"));
    set("kpiSteerDelta", u(jitter.toFixed(2), "°/step"));
    const row = document.querySelector(".post-lap-kpi-row");
    if (row && !$("kpiRms")) {
      [["RMS LAT DEV", "kpiRms"], ["MAX LAT DEV", "kpiMaxLat"]].forEach(([lab, id]) => {
        const d = document.createElement("div"); d.className = "kpi-card";
        d.innerHTML = `<span class="kpi-label">${lab}</span><span class="kpi-value" id="${id}"></span>`;
        row.appendChild(d);
      });
    }
    set("kpiRms", u(rms.toFixed(3), "m"));
    set("kpiMaxLat", u(maxLat.toFixed(3), "m"));
    const badge = document.querySelector(".lap-complete-badge"); if (badge) badge.textContent = "RECORDED LAP COMPLETE";
    const tag = document.querySelector(".computed-tag"); if (tag) tag.textContent = `computed from recorded CSV (${n.toLocaleString("en-US")} rows)`;
    const foot = document.querySelector(".post-lap-footer");
    if (foot && !$("recordedNote")) {
      const d = document.createElement("div"); d.id = "recordedNote"; d.className = "rh-note wide";
      d.textContent = "* assumes 60 steps/s (no time base in the physics). Speeds are native px/step. Yaw-rate is kinematic, not tyre slip. " +
        (isGrip() ? "Kinematic model with the existing grip-limit understeer clamp (experiment). Braking = throttle < 0." :
        "Kinematic model, no grip limit. Braking = throttle < 0 (never occurs).");
      foot.prepend(d);
    }
    window.__recordedPostLapStats = { avgLat_m: avgLat, rms_m: rms, maxLat_m: maxLat, brakingPct: (braking / n) * 100,
      departures, crashes, steps: n, avgSpeed: mean(c.speed_px_step), maxSpeed: Math.max(...c.speed_px_step), jitterDeg: jitter };
  };

  function setupSpeedPills() {
    const wrap = document.querySelector(".pb-speed-selector");
    if (!wrap) return;
    wrap.innerHTML = [0.5, 1, 2, 4, 8].map((v) => `<button class="speed-pill${v === 2 ? " active" : ""}" data-speed="${v}">${v}x</button>`).join("");
    wrap.querySelectorAll(".speed-pill").forEach((p) => p.addEventListener("click", () => {
      wrap.querySelectorAll(".speed-pill").forEach((q) => q.classList.remove("active"));
      p.classList.add("active");
      STATE.playback.speed = parseFloat(p.dataset.speed);
    }));
  }

  async function loadRecordedBIC() {
    updateStatusPill("LOADING RECORDED RUN...", "amber");
    showLoadingOverlay("LOADING RECORDED BIC RUN", "Reading saved evaluation CSV (no policy calls)...");
    const res = await fetch(RUN ? `/recorded/bic?run=${encodeURIComponent(RUN)}` : "/recorded/bic");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    REC = data;
    const m = data.meta;

    const track = Object.assign({}, data.track, {
      large_circuit: m.track_kind !== "std",
      markers: (m.track_kind === "std" ? [] : [1, 6, 8]).map((k) => {
        const t = m.turns.find((q) => q.n === k);
        return { label: `T${k}`, x: t.x, y: t.y, radius_m: t.radius_m };
      }),
    });

    const rollout = buildRollout(data);
    STATE.playback.rolloutData = rollout;
    STATE.playback.maxStep = rollout.steps.length - 1;
    STATE.playback.nominalDt = 1 / m.assumed_steps_per_s; // 1x == 60 steps/s (assumed)
    STATE.playback.speed = 2.0;
    STATE.playback.playhead = 0.0;
    STATE.playback.lastStepIndex = -1;
    STATE.playback.isPlaying = false;
    STATE.playback.isLoading = false;
    STATE.simRunning = false;

    clearTrajectoryTrail();
    build3DTrack(track);
    window.__recordedMeta = m;

    // UI chrome for the recorded source
    document.body.classList.add("recorded-source");
    buildHud(m);
    setupSpeedPills();
    const mini = $("raceCanvas");
    if (mini) {
      const panel = document.createElement("div");
      panel.id = "recordedMinimap"; panel.className = "recorded-minimap";
      panel.innerHTML = `<div class="rh-title">${m.track_kind === "std" ? "STANDARD TRACK · seed 1001" : "CIRCUIT · T1 / T6 / T8 = tightest"}</div>`;
      mini.width = 150; mini.height = 240;
      panel.appendChild(mini);
      document.body.appendChild(panel);
    }
    const t = document.querySelector("#telemetrySidebar .telemetry-chart-block:nth-child(4) .chart-title");
    if (t) { t.textContent = "LAT DEV • KINEMATIC YAW-RATE (NOT TYRE SLIP)"; t.style.fontSize = "9px"; }
    const disc = document.querySelector(".concept-disclaimer");
    if (disc) disc.textContent = (RUN ? `${runLabel().toUpperCase()} • FROZEN GENERALIST-V1 • NO LIVE INFERENCE` : "RECORDED BIC EVAL • FROZEN GENERALIST-V1 • NO LIVE INFERENCE");
    const total = $("pbTotalTime"); if (total) total.textContent = mmss(m.steps / m.assumed_steps_per_s);
    const sc = $("pbScrubber"); if (sc) { sc.max = STATE.playback.maxStep; sc.value = 0; }
    const startIcon = $("startIcon"); if (startIcon) startIcon.innerText = "▶";

    renderPlaybackFrame(0.0, 0.016);
    warmupShaders();
    if (carGroup) {
      const h = carGroup.rotation.y;
      const fwd = new THREE.Vector3(Math.cos(h), 0, -Math.sin(h));
      STATE.cameraPos.set(carGroup.position.x - fwd.x * 185, 52, carGroup.position.z - fwd.z * 185);
      STATE.cameraTarget.set(carGroup.position.x + fwd.x * 70, 14, carGroup.position.z + fwd.z * 70);
      camera.position.copy(STATE.cameraPos); camera.lookAt(STATE.cameraTarget); camera.up.set(0, 1, 0);
    }
    hideLoadingOverlay();
    updateStatusPill("READY (RECORDED RUN LOADED)", "cyan");
  }

  window.addEventListener("DOMContentLoaded", () => {
    setBadge();
    if (STATE.source !== "bic") return;
    loadRecordedBIC().catch((err) => {
      console.error("Recorded BIC load failed:", err);
      showLoadingError(err.message || "Failed to load recorded run.");
      updateStatusPill("ERROR", "red");
    });
  });
})();
