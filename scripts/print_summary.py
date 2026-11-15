import json

with open("analysis/results/diversity_scaling_summary.json") as f:
    d = json.load(f)

models = [m["name"] for m in d["metadata"]["evaluated_models"]]
print("=== SUITE 1: HELD-OUT TEST (W=70) ===")
for m in models:
    s = d["track_pool_summary"][m]["TEST_SEEDS (1001..1010)"]
    c = s["completion_rate"] * 100
    lt = s["avg_lap_time"]
    sd = s["std_lap_time"]
    p = s["avg_progress"]
    print(f"{m:28s} | compl: {c:5.1f}% | avg_lap: {lt} +- {sd} | prog: {p:.4f}")

print("\n=== SUITE 2: NARROW WIDTH SWEEP (HELD-OUT SEEDS) ===")
widths = d["metadata"]["width_levels"]
header = "Model".ljust(28) + " | " + " | ".join(f"W={w:2d}" for w in widths)
print(header)
print("-" * len(header))
for m in models:
    row = []
    for w in widths:
        s = d["width_sweep_summary"][m][f"width_{w}"]
        row.append(f"{s['completion_rate']*100:4.0f}%")
    print(f"{m:28s} | " + " | ".join(row))

print("\n=== SUITE 3: HAIRPIN PINCH SWEEP (HELD-OUT SEEDS) ===")
factors = d["metadata"]["hairpin_factors"]
header_h = "Model".ljust(28) + " | " + " | ".join(f"P={f:4.2f} (Compl / Prog)" for f in factors)
print(header_h)
print("-" * len(header_h))
for m in models:
    row = []
    for f in factors:
        s = d["hairpin_summary"][m][f"pinch_{f}"]
        c = s["completion_rate"] * 100
        pr = s["avg_progress"] * 100
        row.append(f"{c:4.0f}% / {pr:4.1f}%")
    print(f"{m:28s} | " + " | ".join(row))
