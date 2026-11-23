"""Chart: BIC speed vs lap distance, grip off vs on, corners marked. Reads runs/grip_eval/ only."""
import csv, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
man = json.load(open("buddh/buddh_manifest.json"))
def load(t):
    r = list(csv.DictReader(open(f"runs/grip_eval/bic_{t}.csv")))
    return [float(x["distance_m"]) for x in r], [float(x["speed_px_step"]) for x in r]
fig, ax = plt.subplots(figsize=(11, 4.2))
for t, lab, c in [("off", "grip OFF (baseline, completes lap)", "#444"), ("g0p8", "0.8 G", "#d95f02"),
                  ("g1p0", "1.0 G", "#1b9e77"), ("g1p2", "1.2 G", "#7570b3")]:
    d, v = load(t); ax.plot(d, v, label=lab, color=c, lw=1.6 if t == "off" else 1.2)
    if t != "off": ax.plot(d[-1], v[-1], "x", color="red", ms=9)
for tr in man["turns"]:
    ax.axvline(tr["distance_along_track_m"], color="#bbb", lw=.7, ls=":")
    ax.text(tr["distance_along_track_m"], 4.32, f"T{tr['turn_number']}", ha="center", fontsize=7,
            color="#c00" if tr["turn_number"] in (1, 6, 8) else "#666")
ax.set_ylim(0, 4.5); ax.set_xlim(0, 3500)
ax.set_xlabel("lap distance (m)  [lap_progress × 5131 m]"); ax.set_ylabel("speed (px/step)")
ax.set_title("BIC — frozen policy speed vs distance: all grip-limited runs leave the track at T1 (red ×); policy never slows")
ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
fig.savefig("docs/evaluation/grip_limit/speed_vs_distance.png", dpi=140)
