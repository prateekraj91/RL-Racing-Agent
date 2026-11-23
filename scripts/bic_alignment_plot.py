"""Plot the saved BIC CSV positions against the exact evaluation track; print max distance."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inference.recorded import load_bic_recorded, HALF_WIDTH_PX

d = load_bic_recorded(); m = d["meta"]
cl = np.array(d["track"]["centerline"]); inn = np.array(d["track"]["inner_boundary"]); out = np.array(d["track"]["outer_boundary"])
x, y = np.array(d["cols"]["car_x"]), np.array(d["cols"]["car_y"])
al = m["alignment"]
fig, ax = plt.subplots(figsize=(7, 11))
for b in (inn, out): ax.plot(b[:, 0], b[:, 1], color="#888", lw=.8)
ax.plot(cl[:, 0], cl[:, 1], color="#27c", lw=.6, label="centerline (npz)")
ax.plot(x, y, color="#e33", lw=.8, label="CSV car_x/car_y (7,521 steps)")
ax.set_aspect("equal"); ax.invert_yaxis(); ax.legend(loc="lower right")
ax.set_title(f"BIC CSV vs eval track  |  max dist {al['max_px']:.3f} px = {al['max_px']/m['px_per_m']:.3f} m  (half-width {HALF_WIDTH_PX:.0f} px)")
fig.savefig("docs/evaluation/bic_playback/csv_vs_track_alignment.png", dpi=110, bbox_inches="tight")
print(al, "max_m", al["max_px"] / m["px_per_m"]); print(m["files"])
