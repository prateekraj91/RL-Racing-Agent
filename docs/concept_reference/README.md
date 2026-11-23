# Concept reference (visual target)

Source of the 45 s concept video. Three.js r170, vanilla ES modules.

- `main.js`  scene: sky (three/addons Sky + PMREM env), sun + soft shadows, ground,
  track ribbons (runoff, asphalt, edge lines, curvature-based curbs, Armco walls),
  start gantry, instanced trees, skyline buildings, grandstand + pit building,
  procedural car, sensor beams + hit rings, trail, cameras, HUD/dashboard drawing.
- `index.html` overlay layout + CSS (HUD strip, lab dashboard, sensor legend,
  real-world import panel, post-lap analysis).
- `sim.js`   SCRIPTED lap used only to drive the concept. NOT policy output.
  Do not use its numbers anywhere in the real app.
- `frames/`  reference stills from the video.

Use this for art direction, layout and scene-building code. The real app must
render real RacingEnv / frozen-policy telemetry only.
