/**
 * Static-build shim (dist/ only; injected by scripts/build_static.py — never loaded by the API server).
 * Replaces the two backend calls the frontend makes with relative static data files and BLOCKS every
 * other API path, so the public static demo can never contact /predict, /step, /rollout, /simulation
 * or /recorded endpoints.
 *   GET  /recorded/bic[?run=ID]   -> data/recorded/<ID|bic>.json         (recorded playback, verbatim API payload)
 *   POST /simulation/rollout      -> data/rollouts/<preset>_<seed>.json  (precomputed deterministic rollouts)
 */
(function () {
  window.STATIC_BUILD = true;
  window.__apiAttempts = []; // blocked API calls (test hook); should stay empty
  window.__staticFetches = []; // mapped static data fetches (test hook)
  const orig = window.fetch.bind(window);
  const API = /^\/(predict|step|simulation|recorded|health|ready|saliency|docs|openapi)/;
  const rel = (p) => new URL(p, document.baseURI).href;

  window.fetch = async function (input, init) {
    const raw = typeof input === "string" ? input : input && input.url ? input.url : String(input);
    const u = new URL(raw, window.location.href);
    const path = u.pathname;
    if (path === "/recorded/bic") {
      const run = u.searchParams.get("run") || "bic";
      const target = rel(`data/recorded/${encodeURIComponent(run)}.json`);
      window.__staticFetches.push(target);
      return orig(target);
    }
    if (path === "/simulation/rollout") {
      let body = {};
      try { body = JSON.parse((init && init.body) || "{}"); } catch (e) { /* ignore */ }
      const preset = body.preset_id || "standard_medium";
      const seed = body.track_seed || 1001;
      const target = rel(`data/rollouts/${encodeURIComponent(preset)}_${encodeURIComponent(seed)}.json`);
      window.__staticFetches.push(target);
      return orig(target);
    }
    if (API.test(path)) {
      window.__apiAttempts.push(raw);
      throw new Error("API disabled in static build: " + path);
    }
    return orig(input, init);
  };
})();
