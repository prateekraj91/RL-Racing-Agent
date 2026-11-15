"""Publication-Quality Plotting for Diversity Scaling Experiment.

Generates:
1. analysis/results/diversity_heldout_completion.png
2. analysis/results/diversity_heldout_laptime.png
3. analysis/results/diversity_narrow_completion.png
4. analysis/results/diversity_hairpin_progress.png
"""

import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "figure.autolayout": True,
        "figure.dpi": 300,
    })


def main():
    json_path = "analysis/results/diversity_scaling_summary.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run analysis/eval_diversity_scaling.py first.")
        sys.exit(1)

    with open(json_path, "r") as f:
        data = json.load(f)

    set_pub_style()
    os.makedirs("analysis/results", exist_ok=True)

    models_meta = data["metadata"]["evaluated_models"]
    models_meta = sorted(models_meta, key=lambda x: x["pool_size"])

    dr_models = [m for m in models_meta if m["pool_size"] > 1]
    baseline_model = next((m for m in models_meta if m["pool_size"] == 1), None)

    dr_pools = [m["pool_size"] for m in dr_models]
    dr_names = [m["name"] for m in dr_models]
    all_pools = [m["pool_size"] for m in models_meta]
    all_names = [m["name"] for m in models_meta]

    test_pool_label = f"TEST_SEEDS ({min(data['metadata']['test_seeds'])}..{max(data['metadata']['test_seeds'])})"
    pool_summary = data["track_pool_summary"]
    width_summary = data["width_sweep_summary"]
    hairpin_summary = data["hairpin_summary"]

    # -------------------------------------------------------------------------
    # Plot 1: Held-out Completion Rate vs Training-Pool Size
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    dr_comp = [pool_summary[n][test_pool_label]["completion_rate"] * 100 for n in dr_names]

    n_test = len(data["metadata"]["test_seeds"])
    dr_errs = [100.0 * np.sqrt(max(0.0, (cr/100.0)*(1.0 - cr/100.0) / n_test)) for cr in dr_comp]

    ax.plot(dr_pools, dr_comp, marker="o", color="#1f77b4", linewidth=2.5, markersize=8,
            label="Multi-Track Models (Calibrated DR, 200k steps)")
    ax.errorbar(dr_pools, dr_comp, yerr=dr_errs, fmt="none", ecolor="#1f77b4", elinewidth=1.5, capsize=4)

    for x, y in zip(dr_pools, dr_comp):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, weight="bold")

    if baseline_model:
        b_name = baseline_model["name"]
        b_comp = pool_summary[b_name][test_pool_label]["completion_rate"] * 100
        ax.scatter([1], [b_comp], marker="D", s=80, color="#d95f02", zorder=5,
                   label="Baseline: Single-Track v2 (No DR, 45k steps)")
        ax.annotate(f"{b_comp:.1f}%", (1, b_comp), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9, weight="bold", color="#d95f02")

    ax.set_title("Held-Out Generalization vs Training Track Diversity", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Number of Procedural Training Tracks ($N$)", fontsize=11, labelpad=8)
    ax.set_ylabel("Lap Completion Rate (%)", fontsize=11, labelpad=8)
    ax.set_xticks(all_pools)
    ax.set_ylim(-5, 115)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    p1 = "analysis/results/diversity_heldout_completion.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[Saved] {p1}")

    # -------------------------------------------------------------------------
    # Plot 2: Held-out Mean Lap Time vs Training-Pool Size (with explicit failure reporting)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    dr_times = [pool_summary[n][test_pool_label]["avg_lap_time"] for n in dr_names]
    dr_stds = [pool_summary[n][test_pool_label]["std_lap_time"] for n in dr_names]

    ax1.plot(dr_pools, dr_times, marker="s", color="#2ca02c", linewidth=2.2, markersize=8,
             label="Multi-Track DR Series (200k steps)")
    ax1.errorbar(dr_pools, dr_times, yerr=dr_stds, fmt="none", ecolor="#2ca02c", elinewidth=1.5, capsize=4)

    for x, y, s in zip(dr_pools, dr_times, dr_stds):
        ax1.annotate(f"{y:.1f} ± {s:.1f} steps", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5)

    if baseline_model:
        b_name = baseline_model["name"]
        b_time = pool_summary[b_name][test_pool_label]["avg_lap_time"]
        b_std = pool_summary[b_name][test_pool_label]["std_lap_time"]
        ax1.scatter([1], [b_time], marker="D", s=80, color="#d95f02", zorder=5,
                    label="Baseline: Single-Track v2 (No DR, 45k steps)")
        ax1.annotate(f"{b_time:.1f} ± {b_std:.1f} steps", (1, b_time), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8.5, color="#d95f02")

    ax1.set_title("Held-Out Lap Time & Failure Audit vs Training Diversity", fontsize=13, weight="bold", pad=12)
    ax1.set_ylabel("Completed Lap Time (simulation steps)\n[Lower = Faster]", fontsize=11, labelpad=8)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Subplot 2: Failures count (crashes + timeouts)
    failures = [pool_summary[n][test_pool_label]["total"] - pool_summary[n][test_pool_label]["completed"] for n in all_names]
    colors = ["#d95f02" if p == 1 else "#1f77b4" for p in all_pools]
    bars = ax2.bar(all_pools, failures, color=colors, width=1.4, alpha=0.85, label="Failed Episodes (DNF)")
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{int(h)}/10", (bar.get_x() + bar.get_width()/2, h), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=9, weight="bold")

    ax2.set_xlabel("Number of Procedural Training Tracks ($N$)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Failures (DNF)", fontsize=11, labelpad=8)
    ax2.set_xticks(all_pools)
    ax2.set_ylim(0, max(failures + [3]) + 2)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    p2 = "analysis/results/diversity_heldout_laptime.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[Saved] {p2}")

    # -------------------------------------------------------------------------
    # Plot 3: Narrow-Track Completion Rate vs Training-Pool Size
    # INCLUDES ALL EVALUATED WIDTHS: [70, 40, 36, 32, 28, 25, 20]
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    widths = [70, 40, 36, 32, 28, 25, 20]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(widths)))

    for w, c in zip(widths, colors):
        w_key = f"width_{w}"
        rates = [width_summary[name].get(w_key, {}).get("completion_rate", 0.0) * 100 for name in all_names]
        ax.plot(all_pools, rates, marker="o", linewidth=2.0, color=c, label=f"Width {w} px")

    ax.set_title("Zero-Shot Narrow-Track Completion vs Training Pool Diversity", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Number of Procedural Training Tracks ($N$)\n[N=1: Single-Track (No DR); N>=5: Calibrated DR]",
                  fontsize=10.5, labelpad=8)
    ax.set_ylabel("Completion Rate on Held-Out Tracks (%)", fontsize=11, labelpad=8)
    ax.set_xticks(all_pools)
    ax.set_ylim(-5, 115)
    ax.legend(title="Track Width", bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    p3 = "analysis/results/diversity_narrow_completion.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[Saved] {p3}")

    # -------------------------------------------------------------------------
    # Plot 4: Hairpin Generalization (Completion & Progress) vs Training-Pool Size
    # -------------------------------------------------------------------------
    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True)

    factors = [0.5, 0.35, 0.2]
    f_colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    for f_val, c in zip(factors, f_colors):
        p_key = f"pinch_{f_val}"

        # DR models series
        dr_c_rates = [hairpin_summary[n].get(p_key, {}).get("completion_rate", 0.0) * 100 for n in dr_names]
        dr_progs = [hairpin_summary[n].get(p_key, {}).get("avg_progress", 0.0) * 100 for n in dr_names]

        ax_c.plot(dr_pools, dr_c_rates, marker="o", linewidth=2.2, color=c, label=f"Pinch {f_val} (DR)")
        ax_p.plot(dr_pools, dr_progs, marker="s", linewidth=2.2, color=c, label=f"Pinch {f_val} (DR)")

        # Single-track baseline point
        if baseline_model:
            b_name = baseline_model["name"]
            b_c = hairpin_summary[b_name].get(p_key, {}).get("completion_rate", 0.0) * 100
            b_p = hairpin_summary[b_name].get(p_key, {}).get("avg_progress", 0.0) * 100
            ax_c.scatter([1], [b_c], marker="D", s=50, color=c, alpha=0.8, edgecolors="k")
            ax_p.scatter([1], [b_p], marker="D", s=50, color=c, alpha=0.8, edgecolors="k")

    # Annotate specific data points of interest
    # Pool 10 pinch 0.5
    p10_c = hairpin_summary["Pool 10 (Calibrated DR)"]["pinch_0.5"]["completion_rate"] * 100
    ax_c.annotate(f"P10: {p10_c:.0f}%", (10, p10_c), textcoords="offset points", xytext=(-5, 10),
                  ha="center", fontsize=8.5, weight="bold", color="#2ca02c")

    # Pool 25 pinch 0.5 and 0.35
    p25_c50 = hairpin_summary["Pool 25 (Calibrated DR)"]["pinch_0.5"]["completion_rate"] * 100
    ax_c.annotate(f"P25: {p25_c50:.0f}%", (25, p25_c50), textcoords="offset points", xytext=(0, 10),
                  ha="center", fontsize=8.5, weight="bold", color="#2ca02c")
    p25_c35 = hairpin_summary["Pool 25 (Calibrated DR)"]["pinch_0.35"]["completion_rate"] * 100
    ax_c.annotate(f"P25: {p25_c35:.0f}%", (25, p25_c35), textcoords="offset points", xytext=(0, 10),
                  ha="center", fontsize=8.5, weight="bold", color="#ff7f0e")

    ax_c.set_title("(a) Hairpin Completion Rate", fontsize=12, weight="bold")
    ax_c.set_xlabel("Training Pool Size ($N$)\n[Diamond = Single-Track Baseline]", fontsize=10)
    ax_c.set_ylabel("Completion Rate (%)", fontsize=10)
    ax_c.set_xticks(all_pools)
    ax_c.set_ylim(-5, 115)
    ax_c.legend(framealpha=0.9, fontsize=8.5)

    ax_p.set_title("(b) Hairpin Mean Lap Progress", fontsize=12, weight="bold")
    ax_p.set_xlabel("Training Pool Size ($N$)\n[Diamond = Single-Track Baseline]", fontsize=10)
    ax_p.set_ylabel("Mean Lap Progress (%)", fontsize=10)
    ax_p.set_xticks(all_pools)
    ax_p.set_ylim(-5, 115)
    ax_p.legend(framealpha=0.9, fontsize=8.5)

    fig.suptitle("Unseen Hairpin Generalization across Varying Pinch Severity", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    p4 = "analysis/results/diversity_hairpin_progress.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[Saved] {p4}")


if __name__ == "__main__":
    main()
