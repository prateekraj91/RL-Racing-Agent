# Prompt 2 backlog — colour / scenery defects (DO NOT FIX IN PROMPT 1)

Logged during the visual gap report. These are visual-correctness bugs to fix in a later
visual pass, after smooth motion (Prompt 1) lands. Investigate the real cause before editing —
hypotheses below are starting points, not conclusions.

## B1 — Trees render white instead of green
- **Symptom:** in overhead/3D views the foliage is white (see `docs/visual_gap_assets/app_overhead.jpg`).
- **Hypothesis:** per-instance colour isn't being applied — `InstancedMesh.setColorAt(...)`
  without `instanceColor.needsUpdate = true`, and/or the leaf `MeshStandardMaterial` base
  `color` left white so there's no vertex/instance tint to modulate. Concept tints each tree
  with an HSL green ramp (`concept_reference/main.js:170-171`).
- **Fix target (Prompt 2):** ensure `setColorAt` + `instanceColor.needsUpdate` and a sensible
  base material colour, matching the concept ramp.

## B2 — Pink/magenta on curbs, brake lights, and sensor rays
- **Symptom:** curbs, car brake lights, and rangefinder rays read as hot pink/magenta instead
  of red / red+white / cyan-amber-red tiers.
- **Hypothesis:** a missing or failed texture load (curb texture) falling back to a tint, or a
  wrong `colorSpace` on a canvas/texture, or an over-driven / wrong `emissive` value. Concept
  uses `#d8231d` curb red and `emissive 0xff1a12` brake red in sRGB.
- **Fix target (Prompt 2):** find the actual cause (check texture load success, `colorSpace`,
  emissive hex + intensity, tone-mapping interaction) — don't just recolour the symptom.

## B3 — Trees placed inside the track loop
- **Symptom:** the infield is packed with trees (overhead shot); concept keeps the infield
  open and scatters trees outside the ribbon.
- **Fix target (Prompt 2):** exclude any tree within **>= 150 px of the centerline** from
  placement (concept uses a `trackDist > 150` keep-out, `concept_reference/main.js:165`).

See `docs/visual_gap_report.md` §7 and §9 for context and the concept port references.
