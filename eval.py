#!/usr/bin/env python3
"""Reproduce the headline generalization eval of the frozen policy from a fresh clone.

    python eval.py --weights models/generalist_v1.pth

Runs the 110 deterministic held-out episodes (10 test tracks x {standard, 7 corridor
widths, 3 hairpin pinch factors}) plus the 20 randomized-start trials, reusing the exact
episode runners of analysis/verify_generalist_reproducibility.py (no forked logic).
Read-only: nothing under analysis/ or models/ is written. ~3 min on CPU.
Exit code is non-zero if results differ from the committed repetition-1 reference
(only checked for the frozen weights).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from analysis import verify_generalist_reproducibility as V  # noqa: E402
from sac.actor import SACActor  # noqa: E402
from sac.track_pools import POOL_CONFIG, TEST_SEEDS  # noqa: E402

REF = os.path.join(ROOT, "analysis/results/generalist_reproducibility_summary.json")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/generalist_v1.pth")
    ap.add_argument("--config", default="configs/generalist_v1.json")
    ap.add_argument("--json", help="optionally write the summary here")
    a = ap.parse_args()

    sd = torch.load(a.weights, map_location="cpu", weights_only=True)
    actor = SACActor(obs_dim=sd["network.0.weight"].shape[1], action_dim=sd["mean.weight"].shape[0])
    actor.load_state_dict(sd)
    actor.eval()
    frozen = sha256(a.weights) == V.EXPECTED_SHA256
    print(f"weights {a.weights}  sha256 {sha256(a.weights)[:12]}…  ({'FROZEN generalist_v1' if frozen else 'not the frozen policy'})")

    cfg = json.load(open(a.config))
    sc = cfg["observation_space"]["sensor_configuration"]
    rays, dist = tuple(sc["obs_ray_angles"]), float(sc["obs_ray_max_dist"])
    base = dict(cfg["nominal_track_parameters"])
    ms = POOL_CONFIG["max_steps"]

    slices = {}

    def run(name, tk):
        eps = [V.run_single_episode(actor, s, tk, rays, dist, max_steps=ms) for s in TEST_SEEDS]
        slices[name] = (V.compute_slice_statistics(eps), eps)
        st = slices[name][0]
        print(f"  {name:<22} {st['n_completed']:>2}/{st['n_total']} completed  crash={st['n_crashed']}  timeout={st['n_timeout']}  mean_lap={st['mean_lap_time']}")

    print("Held-out circuits (TEST_SEEDS 1001-1010):")
    run("standard_w70", base)
    for w in V.WIDTH_LEVELS:
        run(f"narrow_w{w}", dict(base, width=w))
    for hf in V.HAIRPIN_FACTORS:
        run(f"hairpin_f{hf}", dict(base, hairpin=True, hairpin_factor=hf))
    rng = np.random.default_rng(1001)
    starts = [V.run_randomized_start_episode(actor, s, rng, base, rays, dist, max_steps=1000)
              for s in TEST_SEEDS for _ in range(2)]
    st = V.compute_slice_statistics(starts)
    slices["randomized_starts"] = (st, starts)
    print(f"  {'randomized_starts':<22} {st['n_completed']:>2}/{st['n_total']} completed")

    bench = [e for k, (_, eps) in slices.items() if k != "randomized_starts" for e in eps]
    ov = V.compute_slice_statistics(bench)
    print(f"\nHEADLINE: deterministic benchmark {ov['n_completed']}/{ov['n_total']} completed "
          f"({100*ov['completion_rate']:.1f}%), crashes={ov['n_crashed']}, timeouts={ov['n_timeout']}")

    if args_json := a.json:
        json.dump({k: v[0] for k, v in slices.items()} | {"overall": ov}, open(args_json, "w"), indent=2)

    if frozen and os.path.exists(REF):
        ref = json.load(open(REF))["repetition_summaries"]["repetition_1"]
        mine = {"standard_heldout": slices["standard_w70"][0], "randomized_starts": slices["randomized_starts"][0],
                "overall_deterministic_benchmark": ov}
        for w in V.WIDTH_LEVELS:
            mine[f"narrow_w_{w}"] = slices[f"narrow_w{w}"][0]
        for hf in V.HAIRPIN_FACTORS:
            mine[f"hairpin_f_{hf}"] = slices[f"hairpin_f{hf}"][0]
        bad = [k for k, v in mine.items() if v != ref[k]]
        print("MATCHES committed reference (repetition 1): " + ("YES, all slices identical" if not bad else "NO -> " + ", ".join(bad)))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
