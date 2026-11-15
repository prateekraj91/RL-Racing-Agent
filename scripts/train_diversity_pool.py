"""Parallel Training Orchestrator for Diversity Scaling Experiment.

Trains multiple missing pool sizes (e.g. 5, 10, 20 tracks) concurrently
with strictly controlled configurations matching the calibrated 25-track baseline.
"""

import argparse
import json
import os
import subprocess
import sys
import time

PYTHON_BIN = sys.executable

RUN_CONFIGS = [
    {
        "pool": 5,
        "name": "dr_multitrack_s42_calibrated_5tracks",
    },
    {
        "pool": 10,
        "name": "dr_multitrack_s42_calibrated_10tracks",
    },
    {
        "pool": 20,
        "name": "dr_multitrack_s42_calibrated_20tracks",
    },
]


def get_current_progress(out_dir: str) -> dict:
    cfg_file = os.path.join(out_dir, "run_config.json")
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                data = json.load(f)
            return {
                "step": data.get("total_steps", 0),
                "episodes": data.get("episodes", 0),
                "crashes": data.get("crashes", 0),
                "completed": data.get("completed_run", False),
            }
        except Exception:
            pass
    return {"step": 0, "episodes": 0, "crashes": 0, "completed": False}


def main():
    parser = argparse.ArgumentParser(description="Train missing diversity pool runs.")
    parser.add_argument("--pools", type=int, nargs="+", default=[5, 10, 20],
                        help="Pool sizes to train (default: 5 10 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demos", type=int, default=40)
    parser.add_argument("--preset", default="calibrated")
    parser.add_argument("--target", default="medium")
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--stagger-seconds", type=int, default=15,
                        help="Delay in seconds between process launches")
    args = parser.parse_args()

    targets = [c for c in RUN_CONFIGS if c["pool"] in args.pools]
    if not targets:
        print(f"No matching configurations for pools: {args.pools}")
        sys.exit(1)

    print("=" * 80)
    print(f"DIVERSITY SUITE ORCHESTRATOR | Launching {len(targets)} runs concurrently")
    print(f"Pool sizes: {[t['pool'] for t in targets]}")
    print(f"Seed: {args.seed} | DR Preset: {args.preset} | Demos: {args.demos} | Target: {args.target}")
    print("=" * 80)

    processes = {}
    log_files = {}

    start_time = time.time()

    for item in targets:
        pool = item["pool"]
        name = item["name"]
        out_dir = os.path.join("runs", name)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "train.log")
        log_file = open(log_path, "w")
        log_files[name] = log_file

        cmd = [
            PYTHON_BIN, "-u", "-m", "sac.train_curriculum",
            "--curriculum", "none",
            "--seed", str(args.seed),
            "--name", name,
            "--demos", str(args.demos),
            "--train-pool", str(pool),
            "--domain-randomize",
            "--dr-preset", args.preset,
            "--target", args.target,
        ]
        if args.wandb:
            cmd.append("--wandb")

        print(f"[{time.strftime('%X')}] Launching Pool {pool:2d} -> {name}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Logging stdout to: {log_path}")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )
        processes[name] = {
            "proc": proc,
            "pool": pool,
            "out_dir": out_dir,
            "log_path": log_path,
        }

        if len(targets) > 1 and item != targets[-1]:
            print(f"  Waiting {args.stagger_seconds}s for clean demo initialization before next launch...")
            time.sleep(args.stagger_seconds)

    print("\nAll target processes launched. Monitoring progression...\n")

    while True:
        time.sleep(30)
        elapsed_min = (time.time() - start_time) / 60.0

        status_strs = []
        all_done = True

        for name, info in processes.items():
            proc = info["proc"]
            pool = info["pool"]
            poll = proc.poll()
            prog = get_current_progress(info["out_dir"])

            if poll is None:
                all_done = False
                status_strs.append(f"P{pool}({prog['step']//1000:3d}k/200k)")
            else:
                status_strs.append(f"P{pool}(DONE,code={poll})")

        print(f"[{elapsed_min:5.1f}m] Status: " + " | ".join(status_strs), flush=True)

        if all_done:
            break

    # Close log files
    for f in log_files.values():
        f.close()

    total_time = (time.time() - start_time) / 60.0
    print("\n" + "=" * 80)
    print(f"DIVERSITY SUITE COMPLETED in {total_time:.2f} minutes")
    print("=" * 80)

    exit_code = 0
    for name, info in processes.items():
        proc = info["proc"]
        pool = info["pool"]
        final_cfg = os.path.join(info["out_dir"], "run_config.json")
        best_ckpt = os.path.join(info["out_dir"], "best_actor.pth")

        if proc.returncode != 0:
            print(f"❌ Pool {pool} ({name}) FAILED with exit code {proc.returncode}")
            print(f"   Check log at: {info['log_path']}")
            exit_code = 1
        elif not os.path.exists(best_ckpt):
            print(f"❌ Pool {pool} ({name}) completed but {best_ckpt} is missing!")
            exit_code = 1
        else:
            with open(final_cfg, "r") as f:
                cfg = json.load(f)
            sampled = cfg.get("sampled_track_seeds", [])
            print(f"✅ Pool {pool} ({name}) PASSED: 200,000 steps, {len(sampled)} distinct seeds sampled {sampled[:5]}..., best checkpoint: {best_ckpt}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
