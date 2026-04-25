"""
Master precompute orchestrator. Runs every cache-generation script in sequence
so the notebook can load in under 30 seconds.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/precompute_all.py --device mps
    python3 scripts/precompute_all.py --device cpu     # slower, no GPU needed
    python3 scripts/precompute_all.py --skip kv        # skip the 605MB KV cache rebuild
"""
import argparse
import os
import subprocess
import sys
import time


STAGES = [
    ("kv", "scripts/extract_kv_cache.py",
     "Qwen3-4B KV cache + self-study queries (~1 min, 605 MB output)"),
    ("qa", "scripts/precompute_qa_results.py",
     "QA at 6 ratios + verbatim repeats (~45-75 min, ~100 KB output)"),
    ("task_queries", "scripts/extract_task_queries.py",
     "Task-prompt Q vectors for Part G (~1 min, ~25 MB output)"),
    ("layer_heatmap", "scripts/precompute_layer_heatmap.py",
     "Part C layer × ratio heatmap for 4 fast methods (~15 min, 8 KB output)"),
    ("pareto", "scripts/precompute_pareto.py",
     "Part F default Pareto scatter (~1 min, <1 KB output)"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps",
                        help="Device to pass to each stage (mps/cuda/cpu)")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=[s[0] for s in STAGES],
                        help="Stage short-names to skip")
    args = parser.parse_args()

    # Ensure caches directory exists
    os.makedirs("data/cached_kv", exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    total_t0 = time.time()

    for short, script_path, desc in STAGES:
        if short in args.skip:
            print(f"\n[SKIP]  {short}: {desc}")
            continue

        print(f"\n{'='*60}")
        print(f"[STAGE] {short}")
        print(f"  {desc}")
        print(f"  Script: {script_path}")
        print(f"{'='*60}")

        cmd = [sys.executable, script_path]
        # Scripts that actually touch the model accept --device
        if short in ("kv", "qa", "task_queries"):
            cmd.extend(["--device", args.device])

        stage_t0 = time.time()
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        result = subprocess.run(cmd, cwd=repo_root, env=env)
        stage_elapsed = time.time() - stage_t0

        if result.returncode != 0:
            print(f"\n[ERROR] {short} failed with exit code {result.returncode} "
                  f"after {stage_elapsed:.0f}s")
            sys.exit(result.returncode)

        print(f"\n[DONE]  {short} completed in {stage_elapsed:.0f}s")

    total = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"All precomputes complete in {total/60:.1f} min")
    print(f"{'='*60}")
    print("\nCache files:")
    for fn in ["Qwen3-4B.pt", "Qwen3-4B_qa_results.pt",
               "Qwen3-4B_task_queries.pt", "Qwen3-4B_layer_heatmap.pt",
               "Qwen3-4B_pareto_default.pt"]:
        path = f"data/cached_kv/{fn}"
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
