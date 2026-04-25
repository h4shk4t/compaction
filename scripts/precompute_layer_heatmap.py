"""
Precompute Part C's per-layer × per-ratio compaction-quality grid for ALL
methods, once. Notebook then loads the result instead of recomputing on each
method-dropdown change (which would take ~10 minutes on CPU).

This script takes ~30-60 minutes on CPU (slow because AM-OMP is iterative),
but you only run it once.

Usage:
    python scripts/precompute_layer_heatmap.py
    python scripts/precompute_layer_heatmap.py --methods AM-HighestAttnKeys,Random,Truncation
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from compaction.algorithms import (
    HighestAttentionKeysCompaction,
    KVMergerCompaction,
    OMPCompaction,
    RandomSubsetKeysCompaction,
    TruncationCompaction,
)

ALGO_INSTANCES = {
    "AM-HighestAttnKeys": HighestAttentionKeysCompaction(
        score_method="rms", nnls_iters=2, nnls_lower_bound=0.05,
        nnls_upper_bound=20.0, c2_method="lsq", beta_method="nnls",
    ),
    "AM-OMP": OMPCompaction(
        nnls_iters=0, nnls_upper_bound=1096.63, drop_key_beta_cutoff=-7,
        c2_method="lsq",
    ),
    "KVMerger": KVMergerCompaction(c2_method="merge", beta_method="zero"),
    "Random": RandomSubsetKeysCompaction(beta_method="nnls", c2_method="lsq"),
    "Truncation": TruncationCompaction(beta_method="nnls", c2_method="lsq"),
}

RATIOS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90])


def cosine_similarity(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def compute_grid_for_method(cached_kv, algo):
    """Compute (n_layers, n_ratios) cosine-similarity grid for one method."""
    n_layers = cached_kv["n_layers"]
    n_kv_heads = cached_kv["n_kv_heads"]
    gqa = cached_kv["n_q_heads"] // n_kv_heads
    hd = cached_kv["head_dim"]
    T = cached_kv["seq_len"]
    scale = hd ** -0.5

    grid = np.zeros((n_layers, len(RATIOS)), dtype=np.float32)

    for ri, r in enumerate(RATIOS):
        t = max(1, int(round(T * r)))
        for li in range(n_layers):
            K = cached_kv[f"K_{li}"].float()
            V = cached_kv[f"V_{li}"].float()
            Q = cached_kv[f"Q_{li}"].float()
            cosines = []
            for kv_h in range(n_kv_heads):
                K_h = K[kv_h]
                V_h = V[kv_h]
                Q_h = Q[kv_h*gqa:(kv_h+1)*gqa].reshape(-1, hd)
                try:
                    C1, beta, C2, _ = algo.compute_compacted_cache(K_h, V_h, Q_h, t)
                except Exception:
                    C1 = K_h[:t]
                    beta = torch.zeros(t)
                    C2 = V_h[:t]
                fo = torch.softmax((Q_h @ K_h.T) * scale, dim=-1) @ V_h
                co = torch.softmax((Q_h @ C1.T) * scale + beta.float().unsqueeze(0), dim=-1) @ C2.float()
                cosines.append(cosine_similarity(fo, co))
            grid[li, ri] = float(np.mean(cosines))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(ALGO_INSTANCES.keys()),
                        help="Comma-separated method names")
    parser.add_argument("--input", default="data/cached_kv/Qwen3-4B.pt")
    parser.add_argument("--output", default="data/cached_kv/Qwen3-4B_layer_heatmap.pt")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    print(f"Methods: {methods}")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")

    print("\nLoading cached KV...")
    t0 = time.time()
    cached_kv = torch.load(args.input, map_location="cpu", weights_only=False)
    print(f"  {time.time() - t0:.1f}s")

    save_data = {
        "ratios": RATIOS,
        "n_layers": cached_kv["n_layers"],
        "n_kv_heads": cached_kv["n_kv_heads"],
        "methods": methods,
    }

    for name in methods:
        if name not in ALGO_INSTANCES:
            print(f"WARNING: unknown method {name}, skipping")
            continue
        print(f"\n── {name} ──")
        t0 = time.time()
        grid = compute_grid_for_method(cached_kv, ALGO_INSTANCES[name])
        elapsed = time.time() - t0
        print(f"  grid shape: {grid.shape}, elapsed: {elapsed:.1f}s")
        print(f"  range: [{grid.min():.3f}, {grid.max():.3f}]")
        save_data[f"grid_{name}"] = grid

    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(save_data, args.output)
    print(f"  Size: {os.path.getsize(args.output) / 1024:.1f} KB")
    print("\nDone!")


if __name__ == "__main__":
    main()
