"""Plot ablation study results: factorial effects and condition comparisons.

Generates:
  - analysis/figs/ablation_condition_bars.png  (completion rates by condition)
  - analysis/figs/ablation_factorial_effects.png  (main + interaction effects)
  - analysis/figs/ablation_narrow_heatmap.png  (width sweep heatmap)
  - analysis/figs/ablation_hairpin_bars.png  (hairpin benchmark by condition)

Usage:
    python -m analysis.plot_ablation
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "analysis/figs"
RESULTS_DIR = "analysis/results"

# Condition metadata matching experimental design
CONDITIONS = [
    ("A", True,  200, 0.01),
    ("B", True,  200, 0.00),
    ("C", True,  300, 0.01),
    ("D", True,  300, 0.00),
    ("E", False, 200, 0.01),
    ("F", False, 200, 0.00),
    ("G", False, 300, 0.01),
    ("H", False, 300, 0.00),
]

CONDITION_LABELS = {
    "A": "A\nDR·200·Steer",
    "B": "B\nDR·200·NoSteer",
    "C": "C\nDR·300·Steer",
    "D": "D\nDR·300·NoSteer",
    "E": "E\nNoDR·200·Steer",
    "F": "F\nNoDR·200·NoSteer",
    "G": "G\nNoDR·300·Steer",
    "H": "H\nNoDR·300·NoSteer",
}

# Color palette: DR conditions in blues, NoDR in oranges
COLORS = {
    "A": "#2563eb", "B": "#60a5fa",
    "C": "#1d4ed8", "D": "#93c5fd",
    "E": "#ea580c", "F": "#fb923c",
    "G": "#c2410c", "H": "#fdba74",
}


def load_results():
    summary_path = os.path.join(RESULTS_DIR, "ablation_summary.json")
    effects_path = os.path.join(RESULTS_DIR, "ablation_factorial_effects.json")

    with open(summary_path) as f:
        summary = json.load(f)
    with open(effects_path) as f:
        effects = json.load(f)

    return summary, effects


def plot_condition_bars(summary, out_path):
    """Bar chart of held-out completion rates for all 8 conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cr = summary["condition_results"]

    labels = []
    rates = []
    colors = []
    for label, _, _, _ in CONDITIONS:
        if label in cr and "standard_heldout" in cr[label]:
            labels.append(CONDITION_LABELS[label])
            rates.append(cr[label]["standard_heldout"]["completion_rate"] * 100)
            colors.append(COLORS[label])

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, edgecolor="white", linewidth=1.5, width=0.7)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Held-Out Completion Rate (%)", fontsize=12)
    ax.set_title("Ablation Study: Held-Out Track Completion by Condition (Width=70)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563eb", label="DR On"),
        Patch(facecolor="#ea580c", label="DR Off"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_factorial_effects(effects, out_path):
    """Horizontal bar chart of main effects and interactions for completion_rate."""
    cr_effects = effects.get("completion_rate", {})
    if "note" in cr_effects:
        print(f"Skipping factorial effects plot: {cr_effects['note']}")
        return

    main = cr_effects.get("main_effects", {})
    interactions = cr_effects.get("two_factor_interactions", {})
    three_way = cr_effects.get("three_factor_interaction", {})

    labels = []
    values = []
    colors_list = []

    # Main effects
    for factor, val in main.items():
        labels.append(f"Main: {factor}")
        values.append(val * 100)
        colors_list.append("#2563eb" if val >= 0 else "#dc2626")

    # Two-factor interactions
    for pair, val in interactions.items():
        labels.append(f"Interact: {pair}")
        values.append(val * 100)
        colors_list.append("#16a34a" if val >= 0 else "#dc2626")

    # Three-factor
    for triple, val in three_way.items():
        if val is not None:
            labels.append(f"3-way: {triple}")
            values.append(val * 100)
            colors_list.append("#9333ea" if val >= 0 else "#dc2626")

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors_list, edgecolor="white", linewidth=1)

    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + 0.5 if val >= 0 else bar.get_width() - 0.5
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}pp", ha=ha, va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Effect Size (percentage points)", fontsize=11)
    ax.set_title("Factorial Effects on Held-Out Completion Rate", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_narrow_heatmap(summary, out_path):
    """Heatmap of completion rates across conditions × corridor widths."""
    cr = summary["condition_results"]
    widths = [w for w in [70, 40, 36, 32, 28, 25, 20] if any(f"width_{w}" in cr.get(l, {}) for l in cr)]
    cond_labels = []
    data = []

    for label, _, _, _ in CONDITIONS:
        if label not in cr:
            continue
        row = []
        for w in widths:
            key = f"width_{w}"
            val = cr[label].get(key, {}).get("completion_rate", 0) * 100
            row.append(val)
        data.append(row)
        cond_labels.append(label)

    if not data:
        print("No data for narrow heatmap")
        return

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([f"W={w}" for w in widths], fontsize=10)
    ax.set_yticks(range(len(cond_labels)))
    ax.set_yticklabels(cond_labels, fontsize=11)

    for i in range(len(cond_labels)):
        for j in range(len(widths)):
            val = data[i, j]
            color = "white" if val < 40 or val > 80 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("Corridor Width Generalization Completion Rate (%)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Completion %", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_hairpin_bars(summary, out_path):
    """Grouped bar chart of hairpin completion rates by condition and pinch factor."""
    cr = summary["condition_results"]
    factors = [0.50, 0.35, 0.20]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, hf in enumerate(factors):
        ax = axes[idx]
        labels = []
        rates = []
        colors = []
        for label, _, _, _ in CONDITIONS:
            if label not in cr:
                continue
            key = f"hairpin_{hf}"
            val = cr[label].get(key, {}).get("completion_rate", 0) * 100
            labels.append(label)
            rates.append(val)
            colors.append(COLORS[label])

        x = np.arange(len(labels))
        bars = ax.bar(x, rates, color=colors, edgecolor="white", linewidth=1, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(f"Hairpin Factor = {hf:.2f}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("Completion Rate (%)", fontsize=11)
    fig.suptitle("Ablation Study: Hairpin Generalization", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary, effects = load_results()

    plot_condition_bars(summary, os.path.join(OUT_DIR, "ablation_condition_bars.png"))
    plot_factorial_effects(effects, os.path.join(OUT_DIR, "ablation_factorial_effects.png"))
    plot_narrow_heatmap(summary, os.path.join(OUT_DIR, "ablation_narrow_heatmap.png"))
    plot_hairpin_bars(summary, os.path.join(OUT_DIR, "ablation_hairpin_bars.png"))

    print("\nAll ablation plots generated.")


if __name__ == "__main__":
    main()
