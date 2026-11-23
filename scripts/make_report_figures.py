"""Figures for docs/REPORT.md, generated only from recorded CSV/NPZ/JSON (no new experiments)."""
import json, shutil
from pathlib import Path
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)
PX_M = 12.0 / 70.0


class T:
    """Minimal column-store over csv (no pandas dependency)."""
    def __init__(self, path):
        rows = list(csv.DictReader(open(path)))
        self.cols = {}
        for k in rows[0]:
            v = [r[k] for r in rows]
            try: self.cols[k] = np.array(v, float)
            except ValueError: self.cols[k] = np.array(v, object)
    def __getattr__(self, k): return self.cols[k]
    def __len__(self): return len(next(iter(self.cols.values())))


def read_csv(p): return T(p)

z = np.load("buddh/buddh_track_data.npz")
man = json.load(open("buddh/buddh_manifest.json"))
bic = read_csv("analysis/buddh/buddh_full_rollout.csv")
turns = man["turns"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": .25})

# 1. track + driven path
fig, ax = plt.subplots(figsize=(6.5, 6.5))
L, R, C = z["left_boundary_sim"], z["right_boundary_sim"], z["centerline_sim"]
for b in (L, R): ax.plot(b[:, 0], b[:, 1], c="0.45", lw=.8)
ax.plot(C[:, 0], C[:, 1], c="0.7", lw=.5, ls="--", label="OSM centerline")
ax.plot(bic.car_x, bic.car_y, c="tab:red", lw=.9, label="driven path (7,521 steps)")
for t in turns:
    if t["turn_number"] in (1, 6, 8):
        ax.annotate(f"T{t['turn_number']}", (t["apex_sim_x"], t["apex_sim_y"]), color="navy", weight="bold")
ax.plot(*C[0], "g^", label="start")
ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
ax.set_title("BIC: imported centerline, 70 px corridor, and driven path"); ax.legend(loc="best", fontsize=7)
fig.savefig(OUT / "fig1_bic_track_path.png", bbox_inches="tight"); plt.close(fig)

# 2. speed vs distance, grip off vs on
fig, ax = plt.subplots(figsize=(8, 3.6))
runs = {"grip off": "bic_off", "0.8 G": "bic_g0p8", "1.0 G": "bic_g1p0", "1.2 G": "bic_g1p2"}
for lab, f in runs.items():
    d = read_csv(f"runs/grip_eval/{f}.csv")
    ax.plot(d.distance_m, d.speed_px_step, lw=1.4 if lab == "grip off" else 1, label=lab)
for t in turns:
    ax.axvline(t["distance_along_track_m"], c="0.6", lw=.5, ls=":")
    if t["turn_number"] in (1, 6, 8):
        ax.text(t["distance_along_track_m"], 4.12, f"T{t['turn_number']}", ha="center", fontsize=7)
ax.set_xlabel("lap distance (m, assumes 12 m = 70 px)"); ax.set_ylabel("speed (px/step)")
ax.set_ylim(0, 4.3); ax.legend(ncol=4, fontsize=7, loc="lower right")
ax.set_title("Speed vs lap distance: grip runs end at the first corner (T1) still at 4.00 px/step; no braking")
fig.savefig(OUT / "fig2_speed_vs_distance_grip.png", bbox_inches="tight"); plt.close(fig)

# 3. lateral deviation distribution
d = np.abs(bic.signed_dist_px)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(d, bins=80, color="tab:blue"); ax.set_yscale("log")
ax.axvline(d.mean(), c="k", ls="--", label=f"mean {d.mean():.3f} px ({d.mean()*PX_M:.3f} m)")
rms = np.sqrt((bic.signed_dist_px**2).mean())
ax.axvline(rms, c="tab:red", ls="--", label=f"RMS {rms:.3f} px ({rms*PX_M:.3f} m)")
ax.axvline(d.max(), c="tab:green", ls="--", label=f"max {d.max():.2f} px ({d.max()*PX_M:.2f} m)")
ax.set_xlabel("|lateral deviation| (px)"); ax.set_ylabel("steps (log)"); ax.legend(fontsize=7)
ax.set_title("BIC nominal lap: lateral deviation (mean != RMS)")
fig.savefig(OUT / "fig3_lateral_deviation.png", bbox_inches="tight"); plt.close(fig)

# 4. steering histogram
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(bic.steering_action, bins=100, range=(-1, 1), color="tab:purple"); ax.set_yscale("log")
sat = (np.abs(bic.steering_action) > 0.95).mean() * 100
ax.set_xlabel("steering action [-1, 1]"); ax.set_ylabel("steps (log)")
ax.set_title(f"BIC steering action histogram ({sat:.2f}% of steps |a|>0.95)")
fig.savefig(OUT / "fig4_steering_histogram.png", bbox_inches="tight"); plt.close(fig)

# 5. robustness
r = read_csv("analysis/buddh/robustness_summary.csv")
lab = [str(t)[6:8] + " " + str(t)[9:].replace("_", " ") for t in r.trial_id]
fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(r)); w = .4
ax.bar(x - w/2, r.rms_dist_px, w, label="RMS lateral dev (px)")
ax.bar(x + w/2, r.max_dist_px, w, label="max lateral dev (px)")
ax.axhline(35, c="r", ls="--", lw=.8, label="half corridor width (35 px)")
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("px"); ax.legend(fontsize=7)
ax.set_title(f"Robustness: {int(sum(str(c)=='True' for c in r.completed))}/{len(r)} trials completed, {int(r.off_track_steps.sum())} off-track steps, {int(r.collisions.sum())} collisions")
fig.savefig(OUT / "fig5_robustness_trials.png", bbox_inches="tight"); plt.close(fig)

shutil.copy("docs/evaluation/bic_playback/chase_T1.png", OUT / "fig6_live_demo_screenshot.png")
print("mean|d| px", d.mean(), "rms", rms, "sat%", sat, "steer mean", bic.steering_action.mean())
print("turn dists", [(t["turn_number"], t["distance_along_track_m"]) for t in turns])
