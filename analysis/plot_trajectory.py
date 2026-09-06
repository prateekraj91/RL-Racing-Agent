import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.environment import RacingEnv


MEDIUM_TRACK = {
    "width": 70, "base_r": 250, "n_ctrl": 10,
    "min_radius": 80, "cx": 400, "cy": 300,
}

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["medium", "default"], default="default")
parser.add_argument("--seed", type=int, default=101)
args = parser.parse_args()

if args.config == "medium":
    track_kwargs = MEDIUM_TRACK
    run_name = f"medium_seed{args.seed}"
else:
    track_kwargs = {}
    run_name = f"default_seed{args.seed}"


# -------------------------
# Rebuild the SAME track (same seed + config) to get its geometry
# -------------------------

env = RacingEnv(max_steps=2000, verbose=False, track_kwargs=track_kwargs)
env.reset(options={"track_seed": args.seed})
track = env.track

center = track.centerline          # (N, 2)
half = track.half                  # half track width


# -------------------------
# Compute the two road edges by offsetting the centerline
# perpendicular to its own direction
# -------------------------

# direction along the centerline at each point (next - prev)
nxt = np.roll(center, -1, axis=0)
prev = np.roll(center, 1, axis=0)
tangent = nxt - prev
tangent /= (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9)

# perpendicular = rotate tangent 90 degrees: (dx,dy) -> (-dy,dx)
normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

edge_left = center + half * normal
edge_right = center - half * normal


# -------------------------
# Load the driven path
# -------------------------

data = np.load(f"analysis/runs/{run_name}.npz")
x = data["x"]
y = data["y"]
velocity = data["velocity"]
progress = data["lap_progress"]


# -------------------------
# Draw
# -------------------------

fig, ax = plt.subplots(figsize=(8, 7))

# track edges (close the loop by appending the first point)
def closed(a):
    return np.vstack([a, a[0]])

ax.plot(*closed(edge_left).T, color="gray", linewidth=1)
ax.plot(*closed(edge_right).T, color="gray", linewidth=1)
ax.plot(*closed(center).T, color="lightgray", linewidth=1, linestyle="--")

# driven path, colored by speed
sc = ax.scatter(x, y, c=velocity, cmap="viridis", s=8, zorder=3)
plt.colorbar(sc, ax=ax, label="velocity")

# start (green dot) and crash/end (red X)
ax.plot(x[0], y[0], "o", color="lime", markersize=12, label="start", zorder=4)
ax.plot(x[-1], y[-1], "X", color="red", markersize=16, label="end / crash", zorder=5)

ax.set_title(f"Driven path — {run_name}  (final progress {progress[-1]:.2f})")
ax.set_aspect("equal")
ax.invert_yaxis()          # pygame y grows downward; match it
ax.legend(loc="upper right")
plt.tight_layout()

out = f"analysis/figs/trajectory_{run_name}.png"
plt.savefig(out, dpi=120)
plt.close()
print("saved:", out)