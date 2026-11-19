/**
 * Coordinate Transform Module for RL Racing Agent.
 *
 * Converts 2D RacingEnv simulation pixel coordinates to Three.js 3D world units.
 *
 * RacingEnv coordinate frame:
 * - Distance: Pixels (px). Standard canvas size is 800 x 600 px.
 * - Angle: DEGREES [0, 360). 0° = +X (Right), 90° = -Y (Up on screen), 180° = -X (Left), 270° = +Y (Down on screen).
 * - Car dimensions in env: wheelbase = 50 px, nominal visual width = 30 px.
 * - Nominal track width: 70 px (half-width = 35 px).
 *
 * Three.js coordinate frame:
 * - Ground plane: X and Z axes. Y is Up (+Y).
 * - Scale factor: PX_TO_WORLD = 1.0 (1 world unit = 1 pixel).
 * - Overhead Camera: at (0, Y, 0) looking down at (0, 0, 0) with up=(0, 0, -1).
 *   Screen Right = +X (env +X).
 *   Screen Down  = +Z (env +Y).
 * - Car Heading:
 *   In env: dx = v * cos(heading), dy = -v * sin(heading).
 *   In Three.js: dX = dx, dZ = dy.
 *   Forward unit vector for car oriented along local +X rotated by theta around +Y:
 *   (cos(theta), 0, -sin(theta)).
 *   Therefore, rotation.y = heading_deg * (Math.PI / 180.0) without sign flip.
 */

const PX_TO_WORLD = 1.0;

const WORLD_STATE = {
  trackCenter: { x: 400.0, y: 300.0 },
};

function setWorldTrackCenter(cx, cy) {
  WORLD_STATE.trackCenter.x = Number(cx) || 400.0;
  WORLD_STATE.trackCenter.y = Number(cy) || 300.0;
}

function envToWorld(envX, envY) {
  return {
    x: (Number(envX) - WORLD_STATE.trackCenter.x) * PX_TO_WORLD,
    z: (Number(envY) - WORLD_STATE.trackCenter.y) * PX_TO_WORLD,
  };
}

function envHeadingToWorld(angleDeg) {
  const deg = Number(angleDeg) || 0.0;
  return deg * (Math.PI / 180.0);
}

function worldToEnv(worldX, worldZ) {
  return {
    x: (Number(worldX) / PX_TO_WORLD) + WORLD_STATE.trackCenter.x,
    y: (Number(worldZ) / PX_TO_WORLD) + WORLD_STATE.trackCenter.y,
  };
}

// Export for Node/CommonJS test environments if available
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    PX_TO_WORLD,
    WORLD_STATE,
    setWorldTrackCenter,
    envToWorld,
    envHeadingToWorld,
    worldToEnv,
  };
}
