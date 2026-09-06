import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # save to file, don't try to open a window
import matplotlib.pyplot as plt


RUN_PATH = "analysis/runs/medium_seed101.npz"
FIG_DIR = "analysis/figs"


data = np.load(RUN_PATH)
steer = data["steer"]
throttle = data["throttle"]

n = len(steer)


# -------------------------
# Numbers (the actual diagnosis)
# -------------------------

pct_steer_neg = 100.0 * np.mean(steer < 0)
pct_steer_pos = 100.0 * np.mean(steer > 0)
pct_throttle_neg = 100.0 * np.mean(throttle < 0)

print("--- ACTION STATS (medium, seed 101) ---")
print(f"steps: {n}")
print()
print(f"steer   min/mean/max: {steer.min():.3f} / {steer.mean():.3f} / {steer.max():.3f}")
print(f"  % steps steering LEFT  (<0): {pct_steer_neg:.1f}%")
print(f"  % steps steering RIGHT (>0): {pct_steer_pos:.1f}%")
print()
print(f"throttle min/mean/max: {throttle.min():.3f} / {throttle.mean():.3f} / {throttle.max():.3f}")
print(f"  % steps braking (throttle<0): {pct_throttle_neg:.1f}%")


# -------------------------
# Figures
# -------------------------

os.makedirs(FIG_DIR, exist_ok=True)

# Steering histogram
plt.figure(figsize=(6, 4))
plt.hist(steer, bins=40, color="steelblue", edgecolor="black")
plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
plt.title("Steering distribution (medium, seed 101)")
plt.xlabel("steer  (action[0])   <0 = left,  >0 = right")
plt.ylabel("step count")
plt.xlim(-1, 1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/steer_hist.png", dpi=120)
plt.close()

# Throttle histogram
plt.figure(figsize=(6, 4))
plt.hist(throttle, bins=40, color="seagreen", edgecolor="black")
plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
plt.title("Throttle distribution (medium, seed 101)")
plt.xlabel("throttle  (action[1])   <0 = brake,  >0 = accelerate")
plt.ylabel("step count")
plt.xlim(-1, 1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/throttle_hist.png", dpi=120)
plt.close()

# -------------------------
# Speed across the lap
# -------------------------

velocity = data["velocity"]
lap_progress = data["lap_progress"]

plt.figure(figsize=(7, 4))
plt.plot(lap_progress, velocity, color="darkorange", linewidth=1.5)
plt.title("Speed vs lap position (medium, seed 101)")
plt.xlabel("lap progress  (0 = start, 1 = finish)")
plt.ylabel("velocity")
plt.ylim(0, max(velocity) * 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/speed_profile.png", dpi=120)
plt.close()

print(f"speed min/mean/max: {velocity.min():.3f} / {velocity.mean():.3f} / {velocity.max():.3f}")
print(f"saved: {FIG_DIR}/speed_profile.png")

print()
print(f"saved: {FIG_DIR}/steer_hist.png")
print(f"saved: {FIG_DIR}/throttle_hist.png")