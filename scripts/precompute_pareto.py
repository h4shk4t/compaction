"""
Precompute Part F Speed–Quality Pareto data at the notebook's default view
(layer=12, head=4, all 5 methods including OMP, 4 keep ratios, 3 trials).

Running this once means the notebook's Part F renders instantly on load.
If the user changes the layer/head dropdown the cell falls back to live
compute.

Usage:
    python scripts/precompute_pareto.py
    python scripts/precompute_pareto.py --layer 12 --head 4 --device mps
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def cosine_similarity(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cached_kv/Qwen3-4B.pt")
    parser.add_argument("--output", default="data/cached_kv/Qwen3-4B_pareto_default.pt")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Layer:   {args.layer}")
    print(f"Head:    {args.head}")
    print(f"Trials:  {args.trials}")

    print("\nLoading cached KV...")
    cached_kv = torch.load(args.input, map_location="cpu", weights_only=False)
    T = cached_kv["seq_len"]
    hd = cached_kv["head_dim"]
    gqa = cached_kv["n_q_heads"] // cached_kv["n_kv_heads"]
    scale = hd ** -0.5

    K_h = cached_kv[f"K_{args.layer}"].float()[args.head]
    V_h = cached_kv[f"V_{args.layer}"].float()[args.head]
    Q_h = cached_kv[f"Q_{args.layer}"].float()[
        args.head * gqa:(args.head + 1) * gqa
    ].reshape(-1, hd)

    full_out = torch.softmax((Q_h @ K_h.T) * scale, dim=-1) @ V_h

    ratios = [0.05, 0.15, 0.30, 0.60]
    rows = []
    for method_name, algo in ALGO_INSTANCES.items():
        print(f"\n── {method_name} ──")
        method_t0 = time.time()
        for r in ratios:
            t_target = max(1, min(T - 1, int(round(T * r))))
            times = []
            C1 = beta = C2 = None
            for trial in range(args.trials):
                t0 = time.perf_counter()
                try:
                    C1, beta, C2, _ = algo.compute_compacted_cache(K_h, V_h, Q_h, t_target)
                    times.append(time.perf_counter() - t0)
                except Exception:
                    times.append(float("nan"))
            times_sorted = sorted(t for t in times if t == t)
            median_ms = 1000.0 * times_sorted[len(times_sorted) // 2] if times_sorted else float("nan")

            if C1 is not None:
                comp_out = torch.softmax(
                    (Q_h @ C1.T) * scale + beta.float().unsqueeze(0), dim=-1,
                ) @ C2.float()
                cos = cosine_similarity(full_out, comp_out)
            else:
                cos = float("nan")

            rows.append({
                "method": method_name,
                "keep_ratio": r,
                "time_ms": median_ms,
                "cosine": cos,
            })
        print(f"  {time.time() - method_t0:.1f}s for {len(ratios)} ratios")

    save_data = {
        "layer": args.layer,
        "head": args.head,
        "trials": args.trials,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(save_data, args.output)
    print(f"\nSaved {len(rows)} rows to {args.output} ({os.path.getsize(args.output)} bytes)")


if __name__ == "__main__":
    main()
