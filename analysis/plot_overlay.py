"""Day 24 racing-line comparison: LEARNED vs CENTERLINE vs HEURISTIC.

Overlays the three reference lines on the medium track (seed 101):
    - centerline                  (dashed gray)      -- the geometric reference
    - pure-pursuit (heuristic)    (solid orange)     -- baselines/pure_pursuit.py
    - v2 (learned)                (speed-colored)    -- the star of the figure
v1 is drawn thin/faint as a bonus "before" reference; it sits within ~1px of v2,
so it is deliberately de-emphasised rather than competing for attention.

All three lines track the centerline within a few pixels of a 70px-wide track,
so at track scale they are visually near-identical. The right-hand panels carry
the actual signal: signed offset from the centerline, and speed, both against
lap progress, with the track's corners shaded.

Usage:
    python -m analysis.plot_overlay
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from env.environment import RacingEnv

MEDIUM_TRACK = {"width": 70, "base_r": 250, "n_ctrl": 10,
                "min_radius": 80, "cx": 400, "cy": 300}
SEED = 101

C_PP = "#ff7f0e"      # pure-pursuit / heuristic
C_V1 = "#d62728"      # v1
C_V2 = "#2c7d59"      # v2 legend proxy (mid-viridis-ish, colorbar carries the truth)

# -------------------------
# Rebuild the exact track (same seed+config) for its geometry
# -------------------------
env = RacingEnv(max_steps=2000, verbose=False, track_kwargs=MEDIUM_TRACK)
env.reset(options={"track_seed": SEED})
track = env.track
center = track.centerline
half = track.half

nxt = np.roll(center, -1, axis=0)
prev = np.roll(center, 1, axis=0)
tangent = nxt - prev
tangent /= (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9)
normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
edge_left = center + half * normal
edge_right = center - half * normal


def closed(a):
    return np.vstack([a, a[0]])


# -------------------------
# Centerline curvature -> where the corners are, on a lap-progress axis
# -------------------------
u = np.roll(tangent, -1, axis=0)
dtheta = np.degrees(np.arctan2(
    tangent[:, 0] * u[:, 1] - tangent[:, 1] * u[:, 0],
    np.einsum("ij,ij->i", tangent, u),
))
seg_len = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
curvature = np.abs(dtheta) / (seg_len + 1e-9)          # deg per px
node_progress = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]]) / seg_len.sum()
is_corner = curvature > np.percentile(curvature, 75)   # tightest quartile


def corner_spans(mask, prog):
    """Contiguous [start, end] progress spans where mask is True."""
    spans, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = prog[i]
        elif not m and start is not None:
            spans.append((start, prog[i]))
            start = None
    if start is not None:
        spans.append((start, 1.0))
    return spans


SPANS = corner_spans(is_corner, node_progress)


# -------------------------
# Load the driven paths
# -------------------------
v2 = np.load("analysis/runs/medium_seed101_solved_actor.npz")
v1 = np.load("analysis/runs/medium_seed101.npz")
pp = np.load("analysis/runs/medium_seed101_pure_pursuit.npz")


def signed_offset(run):
    return np.array([track.signed_distance(x, y) for x, y in zip(run["x"], run["y"])])


# -------------------------
# Figure: big overlay (left) + offset/speed diagnostics (right)
# -------------------------
fig = plt.figure(figsize=(15.5, 8.2), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
ax = fig.add_subplot(gs[:, 0])
ax_off = fig.add_subplot(gs[0, 1])
ax_spd = fig.add_subplot(gs[1, 1], sharex=ax_off)

# --- track: edges (solid gray) + centerline (dashed) ---
ax.plot(*closed(edge_left).T, color="gray", linewidth=1)
ax.plot(*closed(edge_right).T, color="gray", linewidth=1)
ax.plot(*closed(center).T, color="darkgray", linewidth=1.2, linestyle="--",
        label="centerline", zorder=2)

# --- v1 (bonus, de-emphasised: it overlaps v2 within ~1px) ---
ax.plot(v1["x"], v1["y"], color=C_V1, linewidth=1.0, alpha=0.45,
        linestyle="--", label="v1 (earlier reward, ≈ v2)", zorder=3)

# --- pure-pursuit / HEURISTIC line ---
# outlined so it stays legible where v2 is viridis-yellow (its top speed)
ax.plot(pp["x"], pp["y"], color=C_PP, linewidth=2.2, alpha=1.0,
        label="pure-pursuit (heuristic)", zorder=4,
        path_effects=[pe.Stroke(linewidth=3.8, foreground="0.15"), pe.Normal()])

# --- v2 (learned) — solid, colored by speed, the star of the figure ---
pts = np.array([v2["x"], v2["y"]]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap="viridis", linewidth=3.2, zorder=5)
lc.set_array(v2["velocity"])
ax.add_collection(lc)
cbar = fig.colorbar(lc, ax=ax, orientation="horizontal",
                    fraction=0.045, pad=0.02, aspect=45)
cbar.set_label("v2 (learned) speed  [px/step]")

# --- start marker ---
ax.plot(v2["x"][0], v2["y"][0], "o", color="lime", markersize=11,
        markeredgecolor="black", markeredgewidth=0.8, label="start", zorder=6)

handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Line2D([], [], color=C_V2, linewidth=3.2))
labels.insert(0, "v2 (learned, speed-colored)")
leg_pp = Line2D([], [], color=C_PP, linewidth=2.2,
                path_effects=[pe.Stroke(linewidth=3.8, foreground="0.15"), pe.Normal()])
handles[labels.index("pure-pursuit (heuristic)")] = leg_pp
ax.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.92)

fig.suptitle("Racing line comparison — learned vs centerline vs heuristic   |   "
             f"medium track, seed {SEED}, default physics (grip off)",
             fontsize=13)
ax.set_xlabel("x [px]")
ax.set_ylabel("y [px]")
ax.set_aspect("equal")
ax.invert_yaxis()          # pygame y-down

# -------------------------
# Right panels: the divergence is only a few px, so plot it directly
# -------------------------
for a in (ax_off, ax_spd):
    for s0, s1 in SPANS:
        a.axvspan(s0, s1, color="0.90", zorder=0)

ax_off.axhline(0, color="darkgray", linestyle="--", linewidth=1.2, zorder=1)
ax_off.plot(v1["lap_progress"], signed_offset(v1), color=C_V1, linewidth=0.9,
            alpha=0.45, linestyle="--", zorder=2)
ax_off.plot(pp["lap_progress"], signed_offset(pp), color=C_PP, linewidth=1.6, zorder=3)
ax_off.plot(v2["lap_progress"], signed_offset(v2), color=C_V2, linewidth=1.8, zorder=4)
ax_off.set_ylabel("signed offset from\ncenterline [px]")
ax_off.set_ylim(-8, 8)
ax_off.set_title("Offset from centerline   (track edge at ±35 px, far off-scale)",
                 fontsize=10)
ax_off.text(0.015, 0.06, "shaded = tightest-quartile corners",
            transform=ax_off.transAxes, fontsize=8, color="0.35")
ax_off.grid(alpha=0.25)

ax_spd.plot(v1["lap_progress"], v1["velocity"], color=C_V1, linewidth=0.9,
            alpha=0.45, linestyle="--", zorder=2)
ax_spd.plot(pp["lap_progress"], pp["velocity"], color=C_PP, linewidth=1.6,
            label="pure-pursuit (heuristic)", zorder=3)
ax_spd.plot(v2["lap_progress"], v2["velocity"], color=C_V2, linewidth=1.8,
            label="v2 (learned)", zorder=4)
ax_spd.axhline(4.0, color="0.5", linestyle=":", linewidth=1.0)
ax_spd.text(0.985, 4.0, " car max_speed 4.0", ha="right", va="bottom",
            fontsize=8, color="0.4")
ax_spd.set_ylabel("speed [px/step]")
ax_spd.set_xlabel("lap progress")
ax_spd.set_ylim(0, 4.45)
ax_spd.set_xlim(0, 1.0)
ax_spd.set_title("Speed profile  (no corner braking in the learned line)", fontsize=10)
ax_spd.legend(loc="lower right", fontsize=8, framealpha=0.92)
ax_spd.grid(alpha=0.25)

out = "analysis/figs/racing_line_overlay.png"
plt.savefig(out, dpi=150)
plt.close()
print("saved:", out)
