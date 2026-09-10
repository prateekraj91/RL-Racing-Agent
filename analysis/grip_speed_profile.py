"""Grip experiment, step 3: does the grip-trained agent BRAKE FOR CORNERS?

Captures one deterministic (tanh(mean), no sampling) lap of the grip-trained
actor in the exact world it was trained in -- the tight track (seed 24,
min_radius 60), grip physics ON -- logging per step the position, speed,
lap progress and the GROUND-TRUTH road curvature at the car's position
(Track.corner_radius, not the ray sensor: at width 70 the lateral rays are
pinned at ~36px everywhere and say nothing about the corner ahead).

Then does the same for the non-grip reward-v2 agent on ITS world (medium,
seed 101, grip OFF) purely as a baseline contrast, and answers:

    corr(curvature, speed) strongly negative -> slows for tight corners
    corr near zero                           -> uniform cruise, just cautious

Capture + plot only. No training, no checkpoint writes.

Usage:  python -m analysis.grip_speed_profile
"""

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.environment import RacingEnv
from sac.agent import SACAgent

# ---------------------------------------------------------------- config

GRIP_CKPT = "runs/grip_bites_curric_s42/solved_actor.pth"
GRIP_CFG = dict(
    track_kwargs={"base_r": 250, "n_ctrl": 10, "min_radius": 60,
                  "cx": 400, "cy": 300, "width": 70},
    track_seed=24,
    max_steps=700,
    grip_limit=True,
)

# The reward-v2 winner (docs/REWARD_V2_SOLVED.md), on the medium target it was
# trained for, grip OFF. Baseline contrast only.
V2_CKPT = "runs/exp2_demos_s42/solved_actor.pth"
V2_CFG = dict(
    track_kwargs={"base_r": 250, "n_ctrl": 10, "min_radius": 80,
                  "cx": 400, "cy": 300, "width": 70},
    track_seed=101,
    max_steps=500,
    grip_limit=False,
)

V_CAP = 4.0        # speed cap the agents saturate at
MAX_GRIP = 0.09    # Car.max_grip

# A corner is road the car CANNOT take flat out under grip physics: the
# tightest arc holdable at v is R = v^2/max_grip, so R < V_CAP^2/max_grip is
# a corner. Same physical threshold for both tracks -> the comparison is fair.
R_CRIT = V_CAP ** 2 / MAX_GRIP     # 177.8 px

# ---------------------------------------------------------------- rollout


def deterministic_action(agent, observation):
    obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, _ = agent.actor(obs_t)
        return torch.tanh(mean).numpy()[0]


def capture_lap(checkpoint, cfg):
    """One deterministic lap. State is logged AFTER each step so position,
    speed, curvature and lap_progress all describe the same instant."""
    env = RacingEnv(
        max_steps=cfg["max_steps"],
        verbose=False,
        track_kwargs=cfg["track_kwargs"],
        grip_limit=cfg["grip_limit"],
    )
    agent = SACAgent()
    agent.actor.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    agent.actor.eval()

    obs, _ = env.reset(options={"track_seed": cfg["track_seed"]})

    log = {k: [] for k in ("x", "y", "velocity", "lap_progress",
                           "corner_radius", "curvature", "steer", "throttle")}
    terminated = truncated = False
    info = {}

    while not (terminated or truncated):
        action = deterministic_action(agent, obs)
        obs, _reward, terminated, truncated, info = env.step(action)

        r = env.track.corner_radius(env.car.x, env.car.y)
        log["x"].append(float(env.car.x))
        log["y"].append(float(env.car.y))
        log["velocity"].append(float(env.car.velocity))
        log["lap_progress"].append(float(info["lap_progress"]))
        log["corner_radius"].append(r)
        log["curvature"].append(1.0 / r)
        log["steer"].append(float(action[0]))
        log["throttle"].append(float(action[1]))

    out = {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}
    out["lap_completed"] = bool(info.get("lap_completed", False))
    out["crashed"] = bool(info.get("crashed", False))
    out["grip_events"] = int(getattr(env.car, "grip_events", 0))
    out["steps"] = len(out["x"])
    # the track's own curvature profile, for the shaded corner bands
    out["track_curv_radius"] = env.track._curv_r.copy()
    out["centerline"] = env.track.centerline.copy()
    return out

# ---------------------------------------------------------------- stats


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(a):
    """Average-rank transform (ties averaged), so Spearman = Pearson of ranks."""
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


def _spearman(a, b):
    return _pearson(_rank(a), _rank(b))


def launch_cut(v, frac=0.9):
    """Index of the first step where speed first reaches frac * median speed.

    The car starts from rest, so the opening acceleration ramp is a speed
    change that has nothing to do with corners. Reported both ways.
    """
    target = frac * float(np.median(v))
    idx = np.argmax(v >= target)
    return int(idx) if v[idx] >= target else 0


def analyse(run, label):
    v = run["velocity"]
    k = run["curvature"]
    R = run["corner_radius"]
    cut = launch_cut(v)

    corner = R < R_CRIT
    # tertile split within this track, as a threshold-free cross-check
    q33 = np.percentile(R, 33.3)
    q67 = np.percentile(R, 66.7)

    st = {
        "label": label,
        "steps": run["steps"],
        "lap_completed": run["lap_completed"],
        "crashed": run["crashed"],
        "grip_events": run["grip_events"],
        "mean_speed": float(v.mean()),
        "speed_std": float(v.std()),
        "min_R": float(R.min()),
        "launch_cut": cut,
        "pearson_full": _pearson(k, v),
        "spearman_full": _spearman(k, v),
        "pearson_nolaunch": _pearson(k[cut:], v[cut:]),
        "spearman_nolaunch": _spearman(k[cut:], v[cut:]),
        "corner_frac": float(corner.mean()),
        "corner_speed": float(v[corner].mean()) if corner.any() else float("nan"),
        "straight_speed": float(v[~corner].mean()) if (~corner).any() else float("nan"),
        "tight_third_speed": float(v[R <= q33].mean()),
        "open_third_speed": float(v[R >= q67].mean()),
    }
    # same, excluding the standing-start ramp
    vc, Rc = v[cut:], R[cut:]
    cc = Rc < R_CRIT
    th = run["throttle"]
    thc = th[cut:]
    # Speed is set ONLY by throttle here: Car.update() applies a constant 0.03
    # friction and steering does not scrub speed, so holding v needs
    # throttle = friction/acceleration = 0.375. Anything below that is a lift,
    # anything below 0 is a real brake. Throttle is the CONTROL; speed is its
    # integral, so throttle shows the decision without the lag.
    st["corr_curv_throttle"] = _pearson(k[cut:], thc)
    st["corner_throttle"] = float(thc[cc].mean()) if cc.any() else float("nan")
    st["straight_throttle"] = float(thc[~cc].mean()) if (~cc).any() else float("nan")
    st["brake_frac"] = float((thc < 0).mean())
    st["brake_frac_corner"] = float((thc[cc] < 0).mean()) if cc.any() else float("nan")
    st["brake_frac_straight"] = float((thc[~cc] < 0).mean()) if (~cc).any() else float("nan")
    st["at_cap_frac"] = float((vc >= 3.95).mean())
    st["corner_speed_nolaunch"] = float(vc[cc].mean()) if cc.any() else float("nan")
    st["straight_speed_nolaunch"] = float(vc[~cc].mean()) if (~cc).any() else float("nan")
    st["tight_third_speed_nolaunch"] = float(vc[Rc <= np.percentile(Rc, 33.3)].mean())
    st["open_third_speed_nolaunch"] = float(vc[Rc >= np.percentile(Rc, 66.7)].mean())
    return st

# ---------------------------------------------------------------- plot

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CORNER_BAND = "#dfe7f2"


def _corner_spans(progress, corner_mask):
    """Contiguous [start, end] progress spans where corner_mask is True."""
    spans = []
    i = 0
    n = len(corner_mask)
    while i < n:
        if corner_mask[i]:
            j = i
            while j + 1 < n and corner_mask[j + 1]:
                j += 1
            spans.append((progress[i], progress[j]))
            i = j + 1
        else:
            i += 1
    return spans


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def make_figure(grip, v2, gs, vs, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 12.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2.2, 1.2, 2.4]})
    fig.patch.set_facecolor(SURFACE)

    p = grip["lap_progress"]
    spans = _corner_spans(p, grip["corner_radius"] < R_CRIT)

    def shade(ax, sp, color):
        for a, b in sp:
            ax.axvspan(a, b, color=color, lw=0, zorder=0)

    # --- panel 1: grip agent speed -------------------------------------------
    ax = axes[0]
    _style(ax)
    shade(ax, spans, CORNER_BAND)
    ax.plot(p, grip["velocity"], color=BLUE, lw=2.0, zorder=3)
    ax.axhline(4.0, color=MUTED, lw=1.2, ls=":", zorder=2)
    ax.text(0.985, 4.06, "speed cap 4.0", color=MUTED, fontsize=8.5, ha="right")
    ax.axhline(gs["mean_speed"], color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.text(0.008, gs["mean_speed"] + 0.08, f"lap mean {gs['mean_speed']:.2f}",
            color=INK2, fontsize=9)
    ax.text(0.245, 0.30,
            "shaded = corner: road radius < 177.8px, the tightest arc a car can\n"
            "hold flat out under grip physics (R = v_cap\u00b2 / max_grip)",
            color=INK2, fontsize=8.5)
    ax.set_ylabel("speed (px/step)", color=INK2, fontsize=10)
    ax.set_ylim(0, 4.45)
    ax.set_title("Grip-trained agent \u2014 tight track (seed 24), grip ON   \u00b7   "
                 f"corner {gs['corner_speed_nolaunch']:.2f} vs straight "
                 f"{gs['straight_speed_nolaunch']:.2f} px/step "
                 "(standing start excluded)",
                 color=INK, fontsize=12.5, loc="left", pad=10)

    # --- panel 2: the control signal that produced it ------------------------
    ax = axes[1]
    _style(ax)
    shade(ax, spans, CORNER_BAND)
    ax.plot(p, grip["throttle"], color=BLUE, lw=1.6, zorder=3)
    ax.axhline(0.375, color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.text(0.008, 0.44, "0.375 = throttle that just holds speed against friction",
            color=INK2, fontsize=8.5)
    ax.axhline(0.0, color="#c3c2b7", lw=1.2, zorder=2)
    ax.text(0.008, -0.30, "below 0 = braking", color=INK2, fontsize=8.5)
    ax.set_ylabel("throttle (action)", color=INK2, fontsize=10)
    ax.set_ylim(-1.1, 1.15)
    ax.set_title("The decision behind it \u2014 throttle   \u00b7   "
                 f"corr(curvature, throttle) = {gs['corr_curv_throttle']:+.3f}   \u00b7   "
                 f"brakes on {100*gs['brake_frac_corner']:.0f}% of corner steps, "
                 f"{100*gs['brake_frac_straight']:.0f}% of straight steps",
                 color=INK, fontsize=11, loc="left", pad=8)

    # --- panel 3: the road underneath ----------------------------------------
    ax = axes[2]
    _style(ax)
    shade(ax, spans, CORNER_BAND)
    ax.plot(p, grip["curvature"] * 1000.0, color=INK2, lw=1.6, zorder=3)
    ax.axhline(1000.0 / R_CRIT, color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.set_ylabel("road curvature\n(1000/R)", color=INK2, fontsize=10)
    ax.set_title(f"Road curvature at the car   \u00b7   corr(curvature, speed) = "
                 f"{gs['pearson_nolaunch']:+.3f} excl. start "
                 f"({gs['pearson_full']:+.3f} whole lap)",
                 color=INK, fontsize=11, loc="left", pad=8)

    # --- panel 4: the non-grip baseline on its own track ---------------------
    ax = axes[3]
    _style(ax)
    p2 = v2["lap_progress"]
    shade(ax, _corner_spans(p2, v2["corner_radius"] < R_CRIT), "#f5e2d8")
    ax.plot(p2, v2["velocity"], color=ORANGE, lw=2.0, zorder=3)
    ax.axhline(4.0, color=MUTED, lw=1.2, ls=":", zorder=2)
    ax.axhline(vs["mean_speed"], color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.text(0.008, vs["mean_speed"] - 0.32, f"lap mean {vs['mean_speed']:.2f}",
            color=INK2, fontsize=9)
    ax.text(0.30, 0.30, "flat out at the cap through its own corners \u2014 "
                        "never once brakes", color=INK2, fontsize=8.5)
    ax.set_ylabel("speed (px/step)", color=INK2, fontsize=10)
    ax.set_xlabel("lap progress", color=INK2, fontsize=10)
    ax.set_ylim(0, 4.45)
    ax.set_xlim(0, 1.0)
    ax.set_title("BASELINE \u2014 non-grip reward-v2 agent, medium track (seed 101), grip OFF\n"
                 f"corr(curvature, speed) = {vs['pearson_nolaunch']:+.3f} excl. start"
                 f"   \u00b7   corner {vs['corner_speed_nolaunch']:.2f} vs straight "
                 f"{vs['straight_speed_nolaunch']:.2f} px/step \u2014 no corner effect at all",
                 color=INK, fontsize=11, loc="left", pad=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

# ---------------------------------------------------------------- main


def _report(st):
    print(f"\n--- {st['label']} ---")
    print(f"  steps={st['steps']}  lap_completed={st['lap_completed']}  "
          f"crashed={st['crashed']}  grip_events={st['grip_events']}")
    print(f"  mean speed        {st['mean_speed']:.3f}  (std {st['speed_std']:.3f})")
    print(f"  tightest road R   {st['min_R']:.1f}px   corner steps "
          f"{100*st['corner_frac']:.1f}%")
    print(f"  corr(curvature, speed)  Pearson {st['pearson_full']:+.3f}   "
          f"Spearman {st['spearman_full']:+.3f}")
    print(f"    excl. standing start (first {st['launch_cut']} steps): "
          f"Pearson {st['pearson_nolaunch']:+.3f}   "
          f"Spearman {st['spearman_nolaunch']:+.3f}")
    print(f"  corner vs straight (R<{R_CRIT:.1f}px): "
          f"{st['corner_speed']:.3f} vs {st['straight_speed']:.3f}  "
          f"(gap {st['straight_speed']-st['corner_speed']:+.3f})")
    print(f"    excl. standing start: {st['corner_speed_nolaunch']:.3f} vs "
          f"{st['straight_speed_nolaunch']:.3f}  "
          f"(gap {st['straight_speed_nolaunch']-st['corner_speed_nolaunch']:+.3f})")
    print(f"  tightest third vs most-open third: {st['tight_third_speed']:.3f} vs "
          f"{st['open_third_speed']:.3f}  "
          f"(gap {st['open_third_speed']-st['tight_third_speed']:+.3f})")
    print(f"    excl. standing start: {st['tight_third_speed_nolaunch']:.3f} vs "
          f"{st['open_third_speed_nolaunch']:.3f}  "
          f"(gap {st['open_third_speed_nolaunch']-st['tight_third_speed_nolaunch']:+.3f})")
    print(f"  -- the control signal (throttle; 0.375 holds speed, <0 = brake) --")
    print(f"  corr(curvature, throttle) excl. start: {st['corr_curv_throttle']:+.3f}")
    print(f"  mean throttle  corner {st['corner_throttle']:+.3f}  "
          f"straight {st['straight_throttle']:+.3f}")
    print(f"  steps actually BRAKING (throttle<0): overall "
          f"{100*st['brake_frac']:.1f}%   in corners {100*st['brake_frac_corner']:.1f}%"
          f"   on straights {100*st['brake_frac_straight']:.1f}%")
    print(f"  steps pinned at the 4.0 speed cap: {100*st['at_cap_frac']:.1f}%")


def main():
    print(f"grip agent : {GRIP_CKPT}")
    print(f"             tight seed 24, min_radius 60, width 70, "
          f"max_steps 700, grip_limit=True")
    grip = capture_lap(GRIP_CKPT, GRIP_CFG)

    print(f"v2 baseline: {V2_CKPT}")
    print(f"             medium seed 101, min_radius 80, width 70, "
          f"max_steps 500, grip_limit=False")
    v2 = capture_lap(V2_CKPT, V2_CFG)

    gs = analyse(grip, "GRIP agent (tight track, grip ON)")
    vs = analyse(v2, "V2 baseline (medium track, grip OFF)")

    os.makedirs("analysis/runs", exist_ok=True)
    np.savez(
        "analysis/runs/grip_agent_lap.npz",
        x=grip["x"], y=grip["y"], velocity=grip["velocity"],
        lap_progress=grip["lap_progress"], corner_radius=grip["corner_radius"],
        curvature=grip["curvature"], steer=grip["steer"], throttle=grip["throttle"],
        lap_completed=grip["lap_completed"], crashed=grip["crashed"],
        grip_events=grip["grip_events"], centerline=grip["centerline"],
        track_curv_radius=grip["track_curv_radius"],
    )
    np.savez(
        "analysis/runs/v2_agent_lap_baseline.npz",
        x=v2["x"], y=v2["y"], velocity=v2["velocity"],
        lap_progress=v2["lap_progress"], corner_radius=v2["corner_radius"],
        curvature=v2["curvature"], steer=v2["steer"], throttle=v2["throttle"],
        lap_completed=v2["lap_completed"], crashed=v2["crashed"],
    )

    fig_path = "analysis/figs/grip_speed_profile.png"
    make_figure(grip, v2, gs, vs, fig_path)

    _report(gs)
    _report(vs)
    print("\nsaved:  analysis/runs/grip_agent_lap.npz")
    print("saved:  analysis/runs/v2_agent_lap_baseline.npz")
    print("saved: ", fig_path)


if __name__ == "__main__":
    main()
