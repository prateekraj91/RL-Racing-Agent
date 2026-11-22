# Prompt 2 — concept world port (built from the real centerline via world.js)

**Branch:** `feat/visual-uplift` · **Date:** 2026-11-22

The 3D scene in `frontend/app.js` was rebuilt from the concept (`docs/concept_reference/main.js`),
driven by the **real** track geometry (`rollout.track` centerline / inner+outer boundaries) through
`frontend/world.js` (`envToWorld`, 1 unit = 1 px). The ground/road/props code was **replaced**, not
layered. No env physics, obs/action contracts, inference, or `sim.js` numbers were touched.

Side-by-side deliverables: `docs/visual_gap_assets/p2_chase_vs_ref11s.jpg`,
`docs/visual_gap_assets/p2_overhead_vs_ref33s.jpg`.

## What changed

| Area | Before | After (concept-matched) |
|------|--------|-------------------------|
| **Sky** | HDRI file / flat blue fallback | Inlined Preetham **Sky** shader (turbidity 2.6, rayleigh 2.2, mie 0.004/g 0.82), sun elevation ~26°/az 215°, `BackSide` dome |
| **Environment** | PMREM from HDRI | **PMREM baked from the Sky**; per-material `envMapIntensity` (no `environmentIntensity` in r128) |
| **Tone map** | ACES, exposure 1.0 | ACES, **exposure 0.72** (0.62 crushed the skyline; 0.72 keeps contrast) |
| **Fog** | `Fog(0x9bc2e0, 1400, 4200)` fixed | `Fog(0xa9c4de, …)` **rescaled per track** (near = diag·1.1, far = diag·4.2) |
| **Sun** | dir 1.2, `physicallyCorrectLights=true` | warm dir **3.1** (`0xfff1dc`), soft PCF shadows, **tight frustum follows the car** along the shared sun dir; **hemisphere** fill `0.55` |
| **Ground** | 4200 plane, repeat 40 (tiled), PBR normal/rough | **16000 plane**, concept grass canvas + a low-freq **macro overlay** (second scale) to kill tiling |
| **Road** | single strip, flat grey | **runoff apron** (darker, wider) + asphalt ribbon with the **rubbered racing-line** canvas texture, via a shared `buildTrackRibbon()` |
| **Curbs** | vertex red `[0.95,0.12,0.22]` (pink) | **`#d8231d`** red/white, curvature-gated only |
| **Walls** | — | white-concrete + racing-blue Armco at the real boundary |
| **Centreline** | glowing cyan strip | **removed** (concept has none; rubbered line reads) |
| **Start** | tiny white quad | **checker start line + gantry** (posts, beam, glowing cyan panel) at the real start pose |
| **Trees** | hugged the boundary, could fall **inside** the loop, foliage untinted | scattered **outward** (normal forced away from centre), **≥150 px off the centerline**, green **per-instance `setColorAt` + `instanceColor.needsUpdate`** |
| **Depth** | near 2 fixed | road ≥1 px above ground; **camera.near 0.4 only in cockpit, 2.0 elsewhere** |

## Backlog bugs fixed (from `docs/prompt2_backlog.md`)

- **B1 white trees** — the old trees used flat colored materials; the new instanced foliage uses a
  white base material tinted per-instance with the concept green HSL ramp via `setColorAt`, and
  crucially sets `instanceColor.needsUpdate = true`. Trees now render green.
- **B2 pink/magenta** — root cause found: "reds" were authored with a large **blue** channel and
  exaggerated by the bloom pass. Lidar close-ray `0xff2a5f` (R255 **G42 B95** = magenta),
  curb `[0.95,0.12,0.22]`, brake `0xff0020`. Replaced with low-blue concept values: rays
  `0xff4d4d / 0xffb02e / 0x67e3ff`, curb `#d8231d`, brake `0xff1a12`, and the car paint
  `0x8a0c20 → 0xb3150e` (the pink body was the same too-much-blue problem, exposed once the sun
  got brighter; now a clearcoat `MeshPhysicalMaterial`).
- **B3 trees inside the loop** — placement normal is now forced to point away from the track centre
  and every candidate within 150 px of the centerline is rejected, so the infield stays open.

## fps

**fps could not be reliably measured in the automated browser pane.** `requestAnimationFrame`
throttles to ~1 Hz when the pane is not the active foreground surface (same limitation seen in
Prompt 1), and timing `composer.render()` synchronously only measures CPU command submission
(~0.5 ms) because WebGL GPU work is async — so neither yields a trustworthy frame rate here.

Per the instruction to **defer fps work to the end of Prompt 2**, the ≥55 fps-in-High target and
per-change fps reporting (SSAO/GTAO at half res, bloom at half res, composer pixel ratio 0.8) are
tracked for that final optimization pass and not done now.

## Verification

- Frozen hashes unchanged (`0d40dc47…76784`, `4d76575b…63bdb1`).
- Screenshots captured at seed 1001 / `standard_medium`; cache-busted with an asset `?v=` query
  (the browser pane caches `app.js` and `/demo` aggressively).

## Known residuals (candidates for later polish, not blocking)

- Chase car reads a touch dark; sky is slightly pale at the horizon (concept is too).
- Overhead is a pure top-down; the concept frame is a 3/4 aerial, so the asphalt contrast reads
  less punchy than ref_33s even though the ribbon + open infield are correct.
- No grandstand / pit building yet (not in Prompt 2 scope).
