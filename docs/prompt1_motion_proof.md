# Prompt 1 — smooth motion: measurement & proof

**Date:** 2026-11-22 · **Branch:** `feat/visual-uplift` · **Seed:** 1001 / `standard_medium`

> **Headline:** The smooth-motion architecture Prompt 1 specifies (server-side full-episode
> rollout + client-side frame-`dt` interpolation) is **already implemented** in the codebase.
> This document measures the current behaviour, diagnoses the original jerk, and **proves** the
> rendered car motion is continuous — **0 zero-movement frames, 0 zero-then-jump events** over a
> 10 s playback. No env physics, obs/action contract, or frozen policy was touched. Measurements
> were taken by running the app, not by reading code.

---

## 1. Latency, frame time, and saliency cadence (item 1)

Measured against the live server (`uvicorn inference.app:app`, CPU), 30–50 samples each.

| Path | What it is | Latency |
|------|------------|---------|
| **`POST /simulation/rollout`** | **The design in use** — one request for the whole episode | **388 steps in 3342 ms** (8.6 ms/step amortized), then **zero network during playback** |
| `POST /simulation/step` (1 step) | What a per-step-over-network loop would cost | avg **25.9 ms**, p50 17.3, **p95 124.5**, max **159.4 ms** |
| `POST /simulation/step` + saliency **every** step | "saliency every step" cost | avg 16.5 ms, p50 12.8, max 39.2 ms |
| `POST /predict` (actor forward only) | SAC actor inference alone | avg **1.19 ms**, p50 0.99, server-reported 0.05 ms |

**Saliency cadence:** in the rollout, **78 / 388 steps** carry saliency — **1 every ~5 steps**,
matching the requested `saliency_interval=5`. **Saliency is NOT computed every step.**

**Frame time (in-browser, foreground pane):** avg **29.6 ms/frame (~34 fps)** at `graphicsQuality:
high` (post-processing composer: bloom + SSAO + SMAA). This is render throughput, independent of
motion smoothness (see §3 caveat).

### Diagnosis of the original jerk
A per-step **network** loop pays **17–160 ms per step with p95 124 ms and high variance**, while
the render loop runs every ~16–30 ms. So the renderer repeatedly has **no new step (car frozen),
then a step lands and the car jumps** — exactly "zero-movement frames followed by jumps". The
actor itself is ~1 ms, so the cost is physics + commentary + serialization + network + variance,
not inference. The current single-rollout design removes the per-frame network entirely.

---

## 2. What satisfies the Prompt 1 spec (already in the tree)

| Spec item | Where | Status |
|-----------|-------|--------|
| 2. Additive `POST /simulation/rollout {preset_id/track_seed/max_steps}` returning every step's telemetry, rays, actions, and saliency every N | `inference/app.py:289`, `inference/simulation.py:381` (`rollout`), schema `inference/schemas.py:198` | **Present.** Returns `steps[]` with `vehicle, telemetry, rays, observation, action, reward, saliency, events`; saliency throttled to `saliency_interval`. |
| 3. pytest: rollout output identical to stepping one at a time for the same seed | `tests/test_simulation_api.py:303` `test_simulation_rollout_endpoint_and_parity` | **Present & passing** (positions, headings, actions, observations, telemetry, rays asserted equal). |
| 4. Frontend playback clock (steps/s × speed); per-frame pose lerp between `floor(t)` and `floor(t)+1`; shortest-arc heading; wheels from interpolated distance; camera smoothing from frame `dt`; Play/Pause/Step/Reset, charts, HUD, engineer feed share the playback index | `frontend/app.js` — `animate()` `:1705`, `renderPlaybackFrame()` `:1759`, `updateCameraDirector()` `:1608` | **Present.** Fractional `playhead += (dt / nominalDt) * speed`; position `lerp`; shortest-arc heading; wheel spin from interpolated velocity; exponential camera smoothing on clamped `dt`; all UI reads the same `playhead`. |

The `rollout` method reuses the exact `start()` + `step()` physics/actor path, so parity is
structural, not coincidental.

---

## 3. Proof: per-frame car displacement over 10 s (item 5)

Sampled `window.carGroup.position` every `requestAnimationFrame` during live playback at **1× speed**
(the real render path, driven by `animate()` → `renderPlaybackFrame()`).

**Foreground pane (representative):**

| Metric | Value |
|--------|-------|
| Frames sampled (10 s) | 339 |
| Avg frame dt | 29.6 ms (~34 fps) |
| **Displacement / frame — avg** | **3.11 px** |
| Displacement p50 / p95 / p99 | 2.50 / 4.51 / 6.67 px |
| Displacement max | 13.33 px (one long-`dt` frame) |
| **Zero-movement frames** | **0** |
| **Zero-then-jump events** | **0** |
| Playhead advance | 2.23 → 304.97 (continuous) |

**Time-scaling check (background-throttled pane, ~1 fps):** even when the browser throttled rAF
to ~1 fps (dt ≈ 1016 ms/frame), the car did **not** teleport — displacement stayed ~2.5–4.7 px
because `animate()` **clamps `dt` to ≤ 0.1 s**, so one slow frame advances at most ~3.3 steps.
Correlation `corr(dt, displacement) = 0.767` with a near-constant `px/ms` (0.0009–0.0046, varying
with velocity, not stutter) confirms displacement tracks **elapsed time** — smooth time-scaled
motion, not frame-coupled jumps.

**Interpretation of `max/avg = 4.29`:** the max-displacement frame is simply the longest-`dt`
frame; because displacement scales with `dt`, covering more ground over a longer frame is correct,
not a jerk. The success criterion — *no zero-movement frames followed by jumps* — holds.

> **Caveat:** the ~34 fps figure is the in-app browser pane (headless, high-quality composer).
> It is a rendering-throughput number, separate from the motion-interpolation smoothness Prompt 1
> targets; a visible desktop GPU should render higher. If perceived smoothness still matters on
> the target machine, that's a graphics-quality/perf follow-up, not a motion-logic fix.

---

## 4. Conclusion

- **No code change was required for Prompt 1** — the jerk was already resolved by the
  server-side rollout + client interpolation architecture.
- Verified by running: single rollout request, local interpolation, saliency every 5 steps,
  parity test green, and **0 zero-movement / 0 zero-then-jump** frames over 10 s.
- Frozen model/config hashes unchanged (checked at task end).
