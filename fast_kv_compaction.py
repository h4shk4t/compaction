# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "torch",
#   "transformers",
#   "accelerate",
#   "datasets",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    import sys
    from typing import Dict, List, Optional, Tuple

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    return np, pd, plt, random, sys, torch


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fast KV Compaction via Attention Matching

    **Paper**: [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284) (Zweiger, Fu, Guo, Kim 2026)

    This interactive notebook explains the core idea of Attention Matching (AM) and lets you compare real compaction algorithms from the paper's codebase on synthetic KV caches.

    ---

    ## The Problem

    Large language models store a **KV cache** during generation: every token's key and value vectors are kept so the model can attend to them. For long contexts, this cache becomes the memory bottleneck.

    **Naive approaches** to summarizing or shrinking the cache (eviction, truncation, random dropping) lose information. The paper asks: *can we build a smaller set of keys and values that reproduces the same attention behavior?*

    ## The Attention Matching Idea

    Instead of just picking a subset of tokens, AM constructs a compact cache $(C_1, \beta, C_2)$ where:

    Add more compaction formulae

    - $C_1$ (shape $t \times d$) = **compacted keys** (selected or merged from original keys)
    - $\beta$ (shape $t$) = **bias terms** that correct the partition function so $\sum_j \exp(q \cdot C_{1,j} / \sqrt{d} + \beta_j) \approx \sum_j \exp(q \cdot K_j / \sqrt{d})$
    - $C_2$ (shape $t \times d$) = **compacted values**, solved via least-squares regression so that $\text{softmax}(q C_1^\top / \sqrt{d} + \beta) \cdot C_2 \approx \text{softmax}(q K^\top / \sqrt{d}) \cdot V$

    The key insight: $C_2 \neq V[\text{indices}]$ in general. The least-squares solve lets dropped tokens' information "bleed into" the retained values. And $\beta$ ensures the attention distribution's normalization stays correct.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Controls are rendered in the sidebar (always visible on the right)
    return


@app.cell
def _(mo):
    scenario = mo.ui.dropdown(
        options=["needle", "recency", "clustered"],
        value="needle",
        label="Synthetic scenario",
    )
    method = mo.ui.dropdown(
        options=[
            "AM-HighestAttnKeys",
            "AM-OMP",
            "KVMerger",
            "Random",
            "Truncation",
        ],
        value="AM-HighestAttnKeys",
        label="Compaction method",
    )
    seq_len = mo.ui.slider(32, 256, value=128, step=16, label="Sequence length")
    n_heads = mo.ui.slider(1, 8, value=4, step=1, label="Heads")
    d_head = mo.ui.slider(8, 64, value=32, step=8, label="Head dim")
    keep_ratio = mo.ui.slider(0.05, 1.0, value=0.25, step=0.05, label="Keep ratio")
    noise = mo.ui.slider(0.0, 0.5, value=0.10, step=0.05, label="Noise")
    seed = mo.ui.slider(0, 999, value=42, step=1, label="Seed")

    qa_ratio_selector = mo.ui.dropdown(
        options={"5%": 0.05, "10%": 0.10, "25%": 0.25, "50%": 0.50, "75%": 0.75},
        value="25%",
        label="QA compression ratio",
    )

    mo.sidebar(
        [
            mo.md("## Synthetic Controls"),
            scenario,
            method,
            seq_len,
            n_heads,
            d_head,
            keep_ratio,
            noise,
            seed,
            mo.md("---"),
            mo.md("## QA Demo"),
            qa_ratio_selector,
        ]
    )
    return d_head, keep_ratio, method, n_heads, noise, qa_ratio_selector, scenario, seed, seq_len


@app.cell
def _(np, random, torch):
    def set_seed(s: int) -> None:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def make_synthetic_kv(
        scenario: str,
        seq_len: int,
        n_heads: int,
        d_head: int,
        noise: float,
        seed: int,
    ):
        """Create a toy multi-head attention problem.

        Returns Q, K, V each of shape (H, T, D) and the focus token index.
        """
        set_seed(seed)
        h, t, d = n_heads, seq_len, d_head

        q = torch.randn(h, t, d)
        k = torch.randn(h, t, d)
        v = torch.randn(h, t, d)

        if scenario == "needle":
            focus = max(1, t // 3)
            q[:, -max(4, t // 8) :, :] += 1.5
            k[:, focus : focus + 1, :] += 3.0
            v[:, focus : focus + 1, :] += 2.0
        elif scenario == "recency":
            focus = t - 1
            q[:, -max(8, t // 4) :, :] += 1.2
            k[:, -max(8, t // 4) :, :] += torch.linspace(
                0.2, 2.0, steps=max(8, t // 4)
            ).view(1, -1, 1)
            v[:, -max(8, t // 4) :, :] += 1.0
        else:  # clustered
            focus = t // 2
            cluster = slice(max(0, focus - 5), min(t, focus + 6))
            k[:, cluster, :] += 1.5
            v[:, cluster, :] += 1.0
            q[:, cluster, :] += 0.75

        if noise > 0:
            q = q + noise * torch.randn_like(q)
            k = k + noise * torch.randn_like(k)
            v = v + noise * torch.randn_like(v)

        return q, k, v, focus

    return make_synthetic_kv, set_seed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithms for computing compacted keys

    <insert the algorithm details, defaults, advantages/disadvantages here\>
    """)
    return


@app.cell
def _(sys):
    # Ensure the repo root is on the path so we can import compaction.*
    import os
    import pathlib

    # __file__ may not exist in all marimo contexts; fall back to cwd
    try:
        repo_root = str(pathlib.Path(__file__).resolve().parent)
    except NameError:
        repo_root = os.getcwd()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from compaction.algorithms import (
        HighestAttentionKeysCompaction,
        KVMergerCompaction,
        OMPCompaction,
        RandomSubsetKeysCompaction,
        TruncationCompaction,
    )

    # Instantiate algorithms with the paper's recommended defaults
    ALGO_INSTANCES = {
        "AM-HighestAttnKeys": HighestAttentionKeysCompaction(
            score_method="rms",
            nnls_iters=2,
            nnls_lower_bound=0.05,
            nnls_upper_bound=20.0,
            c2_method="lsq",
            beta_method="nnls",
        ),
        "AM-OMP": OMPCompaction(
            nnls_iters=0,
            nnls_upper_bound=1096.63,  # exp(7)
            drop_key_beta_cutoff=-7,
            c2_method="lsq",
        ),
        "KVMerger": KVMergerCompaction(
            c2_method="merge",
            beta_method="zero",
        ),
        "Random": RandomSubsetKeysCompaction(
            beta_method="nnls",
            c2_method="lsq",
        ),
        "Truncation": TruncationCompaction(
            beta_method="nnls",
            c2_method="lsq",
        ),
    }
    return (ALGO_INSTANCES,)


@app.cell
def _(ALGO_INSTANCES, set_seed, torch):
    def am_output(q, K, V, C1, beta, C2):
        """Compute full and compacted attention outputs for a single head.

        q: (T, D) or (n, D) query vectors
        K, V: (T, D) original keys/values
        C1: (t, D) compacted keys
        beta: (t,) bias terms
        C2: (t, D) compacted values

        Returns (full_out, compact_out) each (n, D).
        """
        scale = q.shape[-1] ** -0.5

        # Full attention
        full_scores = (q @ K.T).float() * scale
        full_weights = torch.softmax(full_scores, dim=-1)
        full_out = full_weights @ V.float()

        # Compacted attention (with beta!)
        comp_scores = (q @ C1.T).float() * scale + beta.float().unsqueeze(0)
        comp_weights = torch.softmax(comp_scores, dim=-1)
        comp_out = comp_weights @ C2.float()

        return full_out, comp_out, full_weights, comp_weights

    def cosine_similarity(a, b):
        a = a.reshape(-1).float()
        b = b.reshape(-1).float()
        denom = (a.norm() * b.norm()).clamp_min(1e-12)
        return float(torch.dot(a, b) / denom)

    def run_compaction(method_name, q, k, v, keep_ratio, seed_val):
        """Run a compaction algorithm from the repository on multi-head tensors.

        q, k, v: (H, T, D)
        Returns a dict with per-head and aggregated results.
        """
        set_seed(seed_val)
        H, T, D = q.shape
        t = max(1, int(round(T * keep_ratio)))
        algo = ALGO_INSTANCES[method_name]

        # If t >= T, no compaction needed — return perfect identity result
        if t >= T:
            scale = D ** -0.5
            full_scores = (q[0] @ k[0].T).float() * scale
            full_weights = torch.softmax(full_scores, dim=-1)
            full_out = torch.stack([
                torch.softmax((q[h] @ k[h].T).float() * scale, dim=-1) @ v[h].float()
                for h in range(H)
            ])
            return {
                "full_out": full_out,
                "compact_out": full_out.clone(),
                "full_weights": full_weights,
                "compact_weights": full_weights.clone(),
                "token_score": full_weights.mean(dim=0),
                "kept": torch.arange(T),
                "cosine_similarity": 1.0,
                "per_head_cosine": [1.0] * H,
                "keep_tokens": T,
            }

        all_full_out = []
        all_comp_out = []
        all_full_w = []
        all_comp_w = []
        all_indices = []
        all_scores = []
        per_head_cosine = []

        for h in range(H):
            K_h = k[h]  # (T, D)
            V_h = v[h]  # (T, D)
            Q_h = q[h]  # (T, D) — use actual queries from the synthetic setup

            try:
                C1, beta, C2, indices = algo.compute_compacted_cache(K_h, V_h, Q_h, t)
            except Exception:
                # Numerical failure (rank-deficient matrix at high keep ratios
                # with small synthetic data). Fall back to direct subset selection.
                indices = list(range(t))
                C1 = K_h[:t]
                beta = torch.zeros(t)
                C2 = V_h[:t]

            full_out_h, comp_out_h, full_w_h, comp_w_h = am_output(
                Q_h, K_h, V_h, C1, beta, C2
            )

            all_full_out.append(full_out_h)
            all_comp_out.append(comp_out_h)
            all_full_w.append(full_w_h)
            all_comp_w.append(comp_w_h)
            all_indices.append(indices)

            # Token-level attention score for visualization (mean attention per key)
            scores = full_w_h.mean(dim=0)  # (T,)
            all_scores.append(scores)

            per_head_cosine.append(cosine_similarity(full_out_h, comp_out_h))

        full_out = torch.stack(all_full_out)
        comp_out = torch.stack(all_comp_out)

        # Aggregate indices across heads — use head 0 for visualization
        kept_indices = (
            torch.tensor(sorted(all_indices[0]))
            if isinstance(all_indices[0], list)
            else torch.tensor(sorted(all_indices[0].tolist()))
        )

        # Mean token score across heads
        token_score = torch.stack(all_scores).mean(dim=0)

        return {
            "full_out": full_out,
            "compact_out": comp_out,
            "full_weights": all_full_w[0],  # head 0 for viz
            "compact_weights": all_comp_w[0],
            "token_score": token_score,
            "kept": kept_indices,
            "cosine_similarity": cosine_similarity(full_out, comp_out),
            "per_head_cosine": per_head_cosine,
            "keep_tokens": t,
        }

    return cosine_similarity, run_compaction


@app.cell
def _(d_head, make_synthetic_kv, n_heads, noise, scenario, seed, seq_len):
    q, k, v, focus = make_synthetic_kv(
        scenario=scenario.value,
        seq_len=seq_len.value,
        n_heads=n_heads.value,
        d_head=d_head.value,
        noise=noise.value,
        seed=seed.value,
    )
    return focus, k, q, v


@app.cell
def _(k, keep_ratio, method, pd, q, run_compaction, seed, seq_len, v):
    primary_result = run_compaction(
        method.value, q, k, v, keep_ratio.value, seed.value
    )
    random_result = run_compaction("Random", q, k, v, keep_ratio.value, seed.value)
    truncation_result = run_compaction(
        "Truncation", q, k, v, keep_ratio.value, seed.value
    )

    summary = pd.DataFrame(
        [
            {
                "method": method.value,
                "keep_tokens": primary_result["keep_tokens"],
                "keep_ratio": primary_result["keep_tokens"] / seq_len.value,
                "cosine_similarity": primary_result["cosine_similarity"],
            },
            {
                "method": "Random",
                "keep_tokens": random_result["keep_tokens"],
                "keep_ratio": random_result["keep_tokens"] / seq_len.value,
                "cosine_similarity": random_result["cosine_similarity"],
            },
            {
                "method": "Truncation",
                "keep_tokens": truncation_result["keep_tokens"],
                "keep_ratio": truncation_result["keep_tokens"] / seq_len.value,
                "cosine_similarity": truncation_result["cosine_similarity"],
            },
        ]
    )
    summary
    return primary_result, summary


@app.cell(hide_code=True)
def _(method, mo, summary):
    mo.md(f"""
    ## Quality Comparison

    Using the paper's actual metric: cosine similarity between $\\text{{softmax}}(q K^\\top / \\sqrt{{d}}) V$ and $\\text{{softmax}}(q C_1^\\top / \\sqrt{{d}} + \\beta) C_2$.

    - **{method.value}**: {summary.iloc[0]['cosine_similarity']:.4f} cosine similarity at {summary.iloc[0]['keep_ratio']:.0%} keep ratio
    - **Random baseline**: {summary.iloc[1]['cosine_similarity']:.4f}
    - **Truncation baseline**: {summary.iloc[2]['cosine_similarity']:.4f}

    The gap between the AM method and baselines demonstrates the value of attention matching: selecting important keys, correcting the partition function with $\\beta$, and fitting $C_2$ via least squares.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Token Importance

    The plot below shows mean attention mass per token position (averaged across heads and query positions). Dashed lines mark the tokens retained by the selected compaction method. The solid line marks the "focus" token in the synthetic scenario.
    """)
    return


@app.cell
def _(focus, method, plt, primary_result, scenario, seq_len):
    _token_score = primary_result["token_score"].detach().cpu().numpy()
    _kept = primary_result["kept"].detach().cpu().numpy()

    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.plot(_token_score, marker="o", markersize=3, linewidth=1.5, color="#2563eb")
    for pos in _kept:
        _ax.axvline(pos, linestyle="--", alpha=0.25, color="#16a34a")
    _ax.axvline(
        int(focus), linestyle="-", linewidth=2, color="#dc2626", label="focus token"
    )
    _ax.set_title(
        f"Token importance & retained positions  [{method.value} on '{scenario.value}']"
    )
    _ax.set_xlabel("Token position")
    _ax.set_ylabel("Mean attention mass")
    _ax.set_xlim(0, max(0, seq_len.value - 1))
    _ax.legend(loc="upper right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attention Maps: Full vs Compacted

    Left: the full attention matrix for head 0.
    Right: attention through the compacted cache $(C_1, \beta, C_2)$. Note how the compacted version preserves the dominant attention patterns despite using far fewer key positions. The compacted attention is computed as $\text{softmax}(q C_1^\top / \sqrt{d} + \beta)$, where the $\beta$ bias terms help match the original partition function.
    """)
    return


@app.cell
def _(keep_ratio, method, plt, primary_result):
    _full = primary_result["full_weights"].detach().cpu().numpy()
    _compact = primary_result["compact_weights"].detach().cpu().numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _im0 = _axes[0].imshow(_full, aspect="auto", cmap="Blues")
    _axes[0].set_title("Full attention (head 0)")
    _axes[0].set_xlabel("Key position")
    _axes[0].set_ylabel("Query position")
    plt.colorbar(_im0, ax=_axes[0], fraction=0.046, pad=0.04)

    _im1 = _axes[1].imshow(_compact, aspect="auto", cmap="Blues")
    _axes[1].set_title(f"Compacted attention (head 0) [{method.value}]")
    _axes[1].set_xlabel("Compacted key index")
    _axes[1].set_ylabel("Query position")
    plt.colorbar(_im1, ax=_axes[1], fraction=0.046, pad=0.04)

    _fig.suptitle(
        f"Attention map comparison  [keep_ratio={keep_ratio.value:.2f}]",
        fontsize=13,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-Head Compaction Quality

    Different attention heads respond very differently to the same compression ratio. Some heads are "easy" (high cosine similarity even at aggressive compression), while others are "hard". This motivates the paper's **non-uniform head budgets**: giving more cache to hard heads and less to easy ones.

    The bar chart below shows per-head cosine similarity for the current settings.
    """)
    return


@app.cell
def _(keep_ratio, method, n_heads, plt, primary_result):
    _per_head = primary_result["per_head_cosine"]

    _fig, _ax = plt.subplots(figsize=(8, 3))
    _x = range(len(_per_head))
    _colors = ["#dc2626" if c < 0.9 else "#16a34a" if c > 0.99 else "#2563eb" for c in _per_head]
    _ax.bar(_x, _per_head, color=_colors, edgecolor="white", linewidth=0.5)
    _ax.set_xlabel("Head index")
    _ax.set_ylabel("Cosine similarity")
    _ax.set_title(
        f"Per-head compaction quality  [{method.value}, keep={keep_ratio.value:.0%}, {n_heads.value} heads]"
    )
    _ax.set_ylim(min(0.5, min(_per_head) - 0.05), 1.02)
    _ax.axhline(1.0, linestyle=":", color="gray", alpha=0.5)
    _ax.set_xticks(list(_x))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Real Model Analysis: Qwen3-4B

    Everything above used synthetic data. Below, we load **actual KV cache and query vectors** extracted from Qwen3-4B (4 billion parameters, 36 layers, 8 KV heads) processing a short article.

    The cached data was extracted by running a single forward pass over the article with hooks that captured:
    - **K, V** (shape `8 × 305 × 128` per layer) — the KV cache
    - **Q** (shape `32 × 305 × 128` per layer) — the actual query vectors the model computed

    Qwen3-4B uses **Grouped Query Attention (GQA)**: 32 query heads share 8 KV heads (4 Q heads per KV head). For compaction, we run the algorithms per-KV-head, using the corresponding Q heads as the training queries.
    """)
    return


@app.cell
def _(mo, torch):
    # Load the cached KV data from the extraction script
    import os as _os

    _cache_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd(),
        "data", "cached_kv", "Qwen3-4B.pt",
    )

    if _os.path.exists(_cache_path):
        cached_kv = torch.load(_cache_path, map_location="cpu", weights_only=False)
        _info = (
            f"Loaded: **{cached_kv['model_name']}** — "
            f"{cached_kv['n_layers']} layers, {cached_kv['n_kv_heads']} KV heads, "
            f"{cached_kv['n_q_heads']} Q heads, "
            f"seq_len={cached_kv['seq_len']}, head_dim={cached_kv['head_dim']}"
        )
        mo.output.replace(mo.md(_info))
    else:
        cached_kv = None
        mo.output.replace(
            mo.md(
                f"**Cache file not found** at `{_cache_path}`. "
                "Run `python scripts/extract_kv_cache.py` first."
            )
        )
    return (cached_kv,)


@app.cell
def _(mo):
    real_keep_ratio = mo.ui.slider(
        0.05, 0.90, value=0.25, step=0.05, label="Keep ratio (real model)"
    )
    real_method = mo.ui.dropdown(
        options=["AM-HighestAttnKeys", "AM-OMP", "KVMerger", "Random", "Truncation"],
        value="AM-HighestAttnKeys",
        label="Method (real model)",
    )
    mo.hstack([real_method, real_keep_ratio], justify="start", gap=1)

    mo.sidebar(
        [
            mo.md("## Real model"),
            real_method,
            real_keep_ratio,
        ]
    )
    return real_keep_ratio, real_method


@app.cell
def _(
    ALGO_INSTANCES,
    cached_kv,
    cosine_similarity,
    np,
    pd,
    real_keep_ratio,
    real_method,
    torch,
):
    # Per-layer compaction quality on real Qwen3-4B KV cache
    #
    # For each layer:
    #   1. Get K, V (shape: n_kv_heads × T × D) and Q (shape: n_q_heads × T × D)
    #   2. For each KV head, run the compaction algorithm using the corresponding
    #      Q heads as training queries (4 Q heads per KV head in GQA)
    #   3. Measure cosine similarity between full and compacted attention output
    #
    # This tells us: which layers are easy/hard to compress?

    if cached_kv is None:
        layer_quality = pd.DataFrame()
    else:
        _n_layers = cached_kv["n_layers"]
        _n_kv_heads = cached_kv["n_kv_heads"]
        _n_q_heads = cached_kv["n_q_heads"]
        _gqa_groups = _n_q_heads // _n_kv_heads  # 32 / 8 = 4
        _head_dim = cached_kv["head_dim"]
        _T = cached_kv["seq_len"]
        _t = max(1, int(round(_T * real_keep_ratio.value)))
        _scale = _head_dim ** -0.5

        _algo = ALGO_INSTANCES[real_method.value]
        _rows = []

        for _layer_idx in range(_n_layers):
            _K = cached_kv[f"K_{_layer_idx}"].float()  # (n_kv_heads, T, D)
            _V = cached_kv[f"V_{_layer_idx}"].float()  # (n_kv_heads, T, D)
            _Q = cached_kv[f"Q_{_layer_idx}"].float()  # (n_q_heads, T, D)

            _head_cosines = []

            for _kv_h in range(_n_kv_heads):
                _K_h = _K[_kv_h]  # (T, D)
                _V_h = _V[_kv_h]  # (T, D)

                # Gather the Q heads that correspond to this KV head
                # GQA: Q heads [_kv_h * _gqa_groups .. (_kv_h+1) * _gqa_groups)
                _q_start = _kv_h * _gqa_groups
                _q_end = _q_start + _gqa_groups
                _Q_h = _Q[_q_start:_q_end].reshape(-1, _head_dim)  # (gqa_groups * T, D)

                try:
                    _C1, _beta, _C2, _indices = _algo.compute_compacted_cache(
                        _K_h, _V_h, _Q_h, _t
                    )
                except Exception:
                    # Fallback for numerical issues
                    _C1 = _K_h[:_t]
                    _beta = torch.zeros(_t)
                    _C2 = _V_h[:_t]

                # Measure quality: compare full vs compacted attention output
                # Use the same Q_h queries for evaluation
                _full_scores = (_Q_h @ _K_h.T) * _scale
                _full_w = torch.softmax(_full_scores, dim=-1)
                _full_out = _full_w @ _V_h

                _comp_scores = (_Q_h @ _C1.T) * _scale + _beta.float().unsqueeze(0)
                _comp_w = torch.softmax(_comp_scores, dim=-1)
                _comp_out = _comp_w @ _C2.float()

                _head_cosines.append(cosine_similarity(_full_out, _comp_out))

            _mean_cos = np.mean(_head_cosines)
            _min_cos = min(_head_cosines)
            _rows.append({
                "layer": _layer_idx,
                "mean_cosine": _mean_cos,
                "min_cosine": _min_cos,
            })

        layer_quality = pd.DataFrame(_rows)

    layer_quality
    return (layer_quality,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per-Layer Compaction Quality

    Each bar shows the mean cosine similarity (across 8 KV heads) when compacting that layer's KV cache. Layers with low similarity are "hard" — they lose important information during compression. This is why the paper uses **non-uniform budgets**: give more cache budget to hard layers, less to easy ones.
    """)
    return


@app.cell
def _(cached_kv, layer_quality, plt, real_keep_ratio, real_method):
    _fig = None
    if cached_kv is not None and len(layer_quality) > 0:
        _fig, _ax = plt.subplots(figsize=(14, 4))
        _colors = [
            "#dc2626" if c < 0.9 else "#16a34a" if c > 0.99 else "#2563eb"
            for c in layer_quality["mean_cosine"]
        ]
        _ax.bar(
            layer_quality["layer"],
            layer_quality["mean_cosine"],
            color=_colors,
            edgecolor="white",
            linewidth=0.5,
        )
        _ax.set_xlabel("Layer index")
        _ax.set_ylabel("Mean cosine similarity")
        _ax.set_title(
            f"Per-layer compaction quality on Qwen3-4B  "
            f"[{real_method.value}, keep={real_keep_ratio.value:.0%}]"
        )
        _ax.set_ylim(
            min(0.5, layer_quality["mean_cosine"].min() - 0.05), 1.02
        )
        _ax.axhline(1.0, linestyle=":", color="gray", alpha=0.5)
        _ax.set_xticks(range(0, len(layer_quality), 2))
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Layer × Compression Heatmap

    The heatmap below sweeps 8 compression ratios across all 36 layers. Each cell shows the mean cosine similarity for that (layer, ratio) combination.

    **What to look for:**
    - **Vertical bands of red** = layers that are hard to compress at any ratio (these need more budget)
    - **Horizontal gradient** = expected — quality drops as you compress more
    - **Bright spots in the low-ratio columns** = layers that stay accurate even at aggressive compression (these are "free" to compress)

    This takes ~20 seconds to compute since it runs 36 layers × 8 ratios × 8 heads = 2,304 compaction calls.
    """)
    return


@app.cell
def _(ALGO_INSTANCES, cached_kv, cosine_similarity, np, real_method, torch):
    # Cross-ablation: compute cosine similarity for every (layer, keep_ratio) pair
    # Only depends on real_method (not real_keep_ratio), so changing the ratio
    # slider above doesn't re-trigger this expensive cell.

    if cached_kv is None:
        heatmap_data = None
    else:
        _n_layers = cached_kv["n_layers"]
        _n_kv_heads = cached_kv["n_kv_heads"]
        _gqa_groups = cached_kv["n_q_heads"] // _n_kv_heads
        _head_dim = cached_kv["head_dim"]
        _T = cached_kv["seq_len"]
        _scale = _head_dim ** -0.5
        _algo = ALGO_INSTANCES[real_method.value]

        _ratios = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90])
        _grid = np.zeros((_n_layers, len(_ratios)))

        for _ri, _r in enumerate(_ratios):
            _t = max(1, int(round(_T * _r)))
            for _li in range(_n_layers):
                _K = cached_kv[f"K_{_li}"].float()
                _V = cached_kv[f"V_{_li}"].float()
                _Q = cached_kv[f"Q_{_li}"].float()

                _cosines = []
                for _kv_h in range(_n_kv_heads):
                    _K_h = _K[_kv_h]
                    _V_h = _V[_kv_h]
                    _Q_h = _Q[_kv_h * _gqa_groups:(_kv_h + 1) * _gqa_groups].reshape(-1, _head_dim)

                    try:
                        _C1, _beta, _C2, _ = _algo.compute_compacted_cache(_K_h, _V_h, _Q_h, _t)
                    except Exception:
                        _C1 = _K_h[:_t]
                        _beta = torch.zeros(_t)
                        _C2 = _V_h[:_t]

                    _fo = torch.softmax((_Q_h @ _K_h.T) * _scale, dim=-1) @ _V_h
                    _co = torch.softmax((_Q_h @ _C1.T) * _scale + _beta.float().unsqueeze(0), dim=-1) @ _C2.float()
                    _cosines.append(cosine_similarity(_fo, _co))

                _grid[_li, _ri] = np.mean(_cosines)

        heatmap_data = {"grid": _grid, "ratios": _ratios}
    return (heatmap_data,)


@app.cell
def _(cached_kv, heatmap_data, plt, real_method):
    _fig = None
    if heatmap_data is not None:
        _grid = heatmap_data["grid"]
        _ratios = heatmap_data["ratios"]
        _n_layers = _grid.shape[0]

        _fig, _ax = plt.subplots(figsize=(10, 8))
        _im = _ax.imshow(
            _grid,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
            origin="lower",
        )
        _ax.set_xlabel("Keep ratio")
        _ax.set_ylabel("Layer index")
        _ax.set_title(
            f"Layer × Compression Quality — {real_method.value} on {cached_kv['model_name']}"
        )
        _ax.set_xticks(range(len(_ratios)))
        _ax.set_xticklabels([f"{r:.0%}" for r in _ratios])
        _ax.set_yticks(range(0, _n_layers, 2))
        plt.colorbar(_im, ax=_ax, label="Mean cosine similarity", shrink=0.8)

        # Annotate cells with values
        for _yi in range(_n_layers):
            for _xi in range(len(_ratios)):
                _val = _grid[_yi, _xi]
                _color = "white" if _val < 0.75 else "black"
                _ax.text(
                    _xi, _yi, f"{_val:.2f}",
                    ha="center", va="center", fontsize=5.5, color=_color,
                )

        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optimized Head Budgets vs Compaction Quality

    The paper precomputes **per-head influence curves** — for each head, how much does perplexity increase as you compress it? A greedy solver then allocates budget to minimize total perplexity: heads with steep curves (hard to compress) get more budget.

    Below we load the precomputed `optimized_agnostic.json` budget and compare it with our compaction quality measurements. The left panel shows the solver's budget allocation (what it *thinks* needs more cache). The right panel shows the actual compaction quality at 25% keep ratio (what *actually* degrades).

    If the solver is well-calibrated, the two should correlate: heads with low quality (red in right panel) should get high budget (bright in left panel).
    """)
    return


@app.cell
def _(cached_kv, heatmap_data, np, plt):
    import json as _json
    import os as _os

    _fig = None

    # Load optimized budget
    _budget_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd(),
        "head_budget_optimization", "head_budgets", "Qwen3-4B", "optimized_agnostic.json",
    )

    if cached_kv is not None and _os.path.exists(_budget_path) and heatmap_data is not None:
        with open(_budget_path) as _f:
            _budget = _json.load(_f)

        _n_layers = cached_kv["n_layers"]
        _n_kv_heads = cached_kv["n_kv_heads"]

        # Build budget grid (layers × heads)
        _budget_grid = np.zeros((_n_layers, _n_kv_heads))
        for _li in range(_n_layers):
            for _hi in range(_n_kv_heads):
                _key = f"L{_li}H{_hi}"
                _budget_grid[_li, _hi] = _budget.get(_key, 0.0)

        # Build quality grid at ~25% keep ratio from heatmap_data
        # Find the ratio column closest to 0.20
        _ratios = heatmap_data["ratios"]
        _quality_grid_full = heatmap_data["grid"]  # (n_layers, n_ratios)
        _col_idx = int(np.argmin(np.abs(_ratios - 0.20)))

        # For quality we only have mean-across-heads per layer from heatmap_data.
        # Let's show the budget grid vs quality-per-layer as a side-by-side.

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # Left: budget allocation (layers × heads)
        _im1 = _ax1.imshow(
            _budget_grid,
            aspect="auto",
            cmap="YlOrRd",
            origin="lower",
        )
        _ax1.set_xlabel("KV Head index")
        _ax1.set_ylabel("Layer index")
        _ax1.set_title("Optimized budget allocation\n(brighter = more cache budget)")
        _ax1.set_xticks(range(_n_kv_heads))
        _ax1.set_yticks(range(0, _n_layers, 2))
        plt.colorbar(_im1, ax=_ax1, label="Proportion of total budget", shrink=0.8)

        # Right: compaction quality heatmap (layers × ratios) — reuse existing data
        _im2 = _ax2.imshow(
            _quality_grid_full,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
            origin="lower",
        )
        _ax2.set_xlabel("Keep ratio")
        _ax2.set_ylabel("Layer index")
        _ax2.set_title("Compaction quality (mean cosine sim)\n(red = hard to compress)")
        _ax2.set_xticks(range(len(_ratios)))
        _ax2.set_xticklabels([f"{r:.0%}" for r in _ratios])
        _ax2.set_yticks(range(0, _n_layers, 2))
        plt.colorbar(_im2, ax=_ax2, label="Mean cosine similarity", shrink=0.8)

        _fig.suptitle(
            "Does the solver's budget match actual compaction difficulty?",
            fontsize=13, y=1.01,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(ALGO_INSTANCES, cached_kv, np, plt, torch):
    # Scatter plot: budget vs quality per layer (aggregated across heads)
    import json as _json
    import os as _os

    _fig = None

    _budget_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd(),
        "head_budget_optimization", "head_budgets", "Qwen3-4B", "optimized_agnostic.json",
    )

    if cached_kv is not None and _os.path.exists(_budget_path):
        with open(_budget_path) as _f:
            _budget = _json.load(_f)

        _n_layers = cached_kv["n_layers"]
        _n_kv_heads = cached_kv["n_kv_heads"]

        # Per-layer: sum of budget across heads
        _layer_budgets = []
        for _li in range(_n_layers):
            _total = sum(_budget.get(f"L{_li}H{_hi}", 0.0) for _hi in range(_n_kv_heads))
            _layer_budgets.append(_total)

        # Per-layer: mean compaction quality at 20% from our cached heatmap
        # Recompute quickly from cached_kv at a fixed ratio
        _gqa = cached_kv["n_q_heads"] // _n_kv_heads
        _hd = cached_kv["head_dim"]
        _T = cached_kv["seq_len"]
        _t = max(1, int(round(_T * 0.20)))
        _scale = _hd ** -0.5

        # Reuse the class from ALGO_INSTANCES to avoid reimporting
        _HAK_cls = type(ALGO_INSTANCES["AM-HighestAttnKeys"])
        _algo = _HAK_cls(
            score_method="rms", nnls_iters=2, nnls_lower_bound=0.05,
            nnls_upper_bound=20.0, c2_method="lsq", beta_method="nnls",
        )

        def _cos(a, b):
            a, b = a.reshape(-1).float(), b.reshape(-1).float()
            return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))

        _layer_quality = []
        for _li in range(_n_layers):
            _K = cached_kv[f"K_{_li}"].float()
            _V = cached_kv[f"V_{_li}"].float()
            _Q = cached_kv[f"Q_{_li}"].float()
            _cosines = []
            for _kv_h in range(_n_kv_heads):
                _Q_h = _Q[_kv_h * _gqa:(_kv_h + 1) * _gqa].reshape(-1, _hd)
                try:
                    _C1, _beta, _C2, _ = _algo.compute_compacted_cache(
                        _K[_kv_h], _V[_kv_h], _Q_h, _t
                    )
                except Exception:
                    _C1, _beta, _C2 = _K[_kv_h][:_t], torch.zeros(_t), _V[_kv_h][:_t]
                _fo = torch.softmax((_Q_h @ _K[_kv_h].T) * _scale, dim=-1) @ _V[_kv_h]
                _co = torch.softmax((_Q_h @ _C1.T) * _scale + _beta.float().unsqueeze(0), dim=-1) @ _C2.float()
                _cosines.append(_cos(_fo, _co))
            _layer_quality.append(np.mean(_cosines))

        # Scatter: x = quality (higher = easier), y = budget (higher = solver thinks it's hard)
        _fig, _ax = plt.subplots(figsize=(8, 5))
        _ax.scatter(
            _layer_quality, _layer_budgets,
            c=range(_n_layers), cmap="viridis", s=60, edgecolors="white", linewidth=0.5,
        )
        for _li in range(_n_layers):
            _ax.annotate(
                str(_li), (_layer_quality[_li], _layer_budgets[_li]),
                fontsize=6, ha="center", va="bottom",
            )
        _ax.set_xlabel("Compaction quality (mean cosine sim at 20% keep)")
        _ax.set_ylabel("Solver budget allocation (sum across heads)")
        _ax.set_title("Budget vs Actual Compaction Difficulty per Layer")
        _ax.axhline(np.mean(_layer_budgets), linestyle=":", color="gray", alpha=0.5, label="mean budget")
        _ax.axvline(np.mean(_layer_quality), linestyle=":", color="gray", alpha=0.5, label="mean quality")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # QA Demo: How Compression Affects Model Answers

    Below we load **pre-computed QA results** from Qwen3-4B answering multiple-choice questions about the article at different compression ratios. Use the slider to see how the model's answers, accuracy, and perplexity change as you compress more aggressively.

    The results were generated using the full AM pipeline: compaction with `AM-HighestAttnKeys` + optimized non-uniform head budgets + self-study query generation.
    """)
    return


@app.cell
def _(mo, torch):
    import os as _os

    _qa_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd(),
        "data", "cached_kv", "Qwen3-4B_qa_results.pt",
    )

    if _os.path.exists(_qa_path):
        qa_data = torch.load(_qa_path, map_location="cpu", weights_only=False)
        _ratios = qa_data["ratios"]
        mo.output.replace(
            mo.md(
                f"Loaded QA results: **{qa_data['model_name']}**, "
                f"{len(qa_data['questions'])} questions, "
                f"{len(_ratios)} compression ratios ({', '.join(f'{r:.0%}' for r in _ratios)})"
            )
        )
    else:
        qa_data = None
        mo.output.replace(
            mo.md(
                f"**QA results not found** at `{_qa_path}`. "
                "Run `PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/precompute_qa_results.py --device mps` first."
            )
        )
    return (qa_data,)


@app.cell
def _():
    # qa_ratio_selector is defined in the sidebar controls cell above
    return


@app.cell
def _(mo, qa_data, qa_ratio_selector):
    # Display QA results for the selected ratio vs baseline
    if qa_data is None or qa_ratio_selector is None:
        mo.output.replace(mo.md("*QA data not loaded.*"))
    else:
        _ratio = qa_ratio_selector.value
        _orig = qa_data["original_qa"]
        _comp = qa_data["ratio_results"][_ratio]["qa_results"]
        _orig_repeat = qa_data["original_repeat"]
        _comp_repeat = qa_data["ratio_results"][_ratio]["repeat_result"]

        _orig_acc = sum(r["correct"] for r in _orig)
        _comp_acc = sum(r["correct"] for r in _comp)

        # Build comparison table
        _rows = []
        for _i, (_o, _c) in enumerate(zip(_orig, _comp)):
            _rows.append(
                f"| {_i+1}. {_o['question'][:50]}... | {_o['gold_letter']} "
                f"| {_o['parsed_choice']} {'✓' if _o['correct'] else '✗'} | {_o['perplexity']:.2f} "
                f"| {_c['parsed_choice']} {'✓' if _c['correct'] else '✗'} | {_c['perplexity']:.2f} |"
            )
        _table = "\n".join(_rows)

        mo.output.replace(
            mo.md(f"""
    ### Results at **{_ratio:.0%}** keep ratio vs full cache

    | Question | Gold | Full cache | PPL | Compacted ({_ratio:.0%}) | PPL |
    |----------|------|-----------|-----|------------------------|-----|
    {_table}

    **Accuracy**: Full cache {_orig_acc}/{len(_orig)} → Compacted {_comp_acc}/{len(_comp)}

    **Verbatim repeat test**:
    - Full cache: {_orig_repeat['word_recall']:.1%} word recall, perplexity {_orig_repeat['perplexity']:.2f}
    - Compacted: {_comp_repeat['word_recall']:.1%} word recall, perplexity {_comp_repeat['perplexity']:.2f}
    """)
        )
    return


@app.cell
def _(mo, qa_data, qa_ratio_selector):
    # Word-level diff between the original article and the compacted model's repeat attempt.
    # Green = correct word, Red = wrong/changed word, Gray = missing word from original.
    import difflib as _difflib

    _output = None
    if qa_data is not None and qa_ratio_selector is not None:
        _ratio = qa_ratio_selector.value
        _comp_repeat = qa_data["ratio_results"][_ratio]["repeat_result"]
        _article = qa_data["article_text"]

        # Tokenize into words
        _orig_words = _article.split()
        _comp_words = _comp_repeat["generated_text"].split()

        # Use SequenceMatcher to get word-level diff opcodes
        _sm = _difflib.SequenceMatcher(None, _orig_words, _comp_words)
        _html_parts = []

        for _tag, _i1, _i2, _j1, _j2 in _sm.get_opcodes():
            if _tag == "equal":
                # Words match — green
                for _w in _orig_words[_i1:_i2]:
                    _html_parts.append(
                        f'<span style="color:#16a34a">{_w}</span>'
                    )
            elif _tag == "replace":
                # Words changed — show original as strikethrough gray, replacement as red
                for _w in _orig_words[_i1:_i2]:
                    _html_parts.append(
                        f'<span style="color:#9ca3af;text-decoration:line-through">{_w}</span>'
                    )
                for _w in _comp_words[_j1:_j2]:
                    _html_parts.append(
                        f'<span style="background:#fecaca;color:#dc2626;font-weight:bold">{_w}</span>'
                    )
            elif _tag == "delete":
                # Words missing from compacted output — gray strikethrough
                for _w in _orig_words[_i1:_i2]:
                    _html_parts.append(
                        f'<span style="background:#e5e7eb;color:#6b7280;text-decoration:line-through">{_w}</span>'
                    )
            elif _tag == "insert":
                # Extra words in compacted output — red background
                for _w in _comp_words[_j1:_j2]:
                    _html_parts.append(
                        f'<span style="background:#fecaca;color:#dc2626;font-weight:bold">{_w}</span>'
                    )

        _diff_html = " ".join(_html_parts)

        _legend = (
            '<div style="margin-bottom:12px;font-size:0.85em">'
            '<span style="color:#16a34a">■ correct</span> &nbsp; '
            '<span style="background:#fecaca;color:#dc2626;font-weight:bold">■ wrong/inserted</span> &nbsp; '
            '<span style="color:#9ca3af;text-decoration:line-through">■ replaced (original)</span> &nbsp; '
            '<span style="background:#e5e7eb;color:#6b7280;text-decoration:line-through">■ missing</span>'
            '</div>'
        )

        _output = mo.vstack([
            mo.md(f"### Verbatim Repeat Diff: Original Article vs {_ratio:.0%} Compacted Output"),
            mo.md(
                f"Word recall: **{_comp_repeat['word_recall']:.1%}** "
                f"({_comp_repeat['matched_words']}/{_comp_repeat['total_words']} words), "
                f"perplexity: **{_comp_repeat['perplexity']:.2f}**"
            ),
            mo.Html(_legend),
            mo.Html(
                f'<div style="line-height:1.8;font-size:0.95em;max-width:800px">{_diff_html}</div>'
            ),
        ])
    _output
    return


@app.cell
def _(np, plt, qa_data):
    # Summary plot: accuracy and perplexity vs compression ratio
    _fig = None
    if qa_data is not None:
        _ratios = sorted(qa_data["ratio_results"].keys())
        _accuracies = [qa_data["ratio_results"][r]["accuracy"] for r in _ratios]
        _repeat_recalls = [qa_data["ratio_results"][r]["repeat_result"]["word_recall"] for r in _ratios]

        # Mean perplexity across questions
        _mean_ppls = []
        for r in _ratios:
            _ppls = [q["perplexity"] for q in qa_data["ratio_results"][r]["qa_results"]
                     if not np.isnan(q["perplexity"])]
            _mean_ppls.append(np.mean(_ppls) if _ppls else float("nan"))

        _repeat_ppls = [qa_data["ratio_results"][r]["repeat_result"]["perplexity"] for r in _ratios]

        # Baseline values
        _orig_acc = qa_data["original_accuracy"]
        _orig_recall = qa_data["original_repeat"]["word_recall"]
        _orig_ppls = [q["perplexity"] for q in qa_data["original_qa"] if not np.isnan(q["perplexity"])]
        _orig_mean_ppl = np.mean(_orig_ppls) if _orig_ppls else float("nan")

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Accuracy
        _ax1.plot(_ratios, _accuracies, "o-", color="#2563eb", linewidth=2, label="Compacted")
        _ax1.axhline(_orig_acc, linestyle="--", color="#16a34a", label="Full cache")
        _ax1.set_xlabel("Keep ratio")
        _ax1.set_ylabel("QA Accuracy")
        _ax1.set_title("QA Accuracy vs Compression")
        _ax1.set_ylim(-0.05, 1.15)
        _ax1.legend(fontsize=8)
        _ax1.grid(True, alpha=0.3)

        # Mean perplexity
        _ax2.plot(_ratios, _mean_ppls, "s-", color="#7c3aed", linewidth=2, label="Compacted (QA)")
        _ax2.plot(_ratios, _repeat_ppls, "^-", color="#d97706", linewidth=2, label="Compacted (repeat)")
        _ax2.axhline(_orig_mean_ppl, linestyle="--", color="#16a34a", label="Full cache (QA)")
        _ax2.set_xlabel("Keep ratio")
        _ax2.set_ylabel("Perplexity")
        _ax2.set_title("Perplexity vs Compression")
        _ax2.legend(fontsize=8)
        _ax2.grid(True, alpha=0.3)

        # Word recall
        _ax3.plot(_ratios, _repeat_recalls, "D-", color="#dc2626", linewidth=2, label="Compacted")
        _ax3.axhline(_orig_recall, linestyle="--", color="#16a34a", label="Full cache")
        _ax3.set_xlabel("Keep ratio")
        _ax3.set_ylabel("Word Recall")
        _ax3.set_title("Verbatim Repeat Recall vs Compression")
        _ax3.set_ylim(-0.05, 1.15)
        _ax3.legend(fontsize=8)
        _ax3.grid(True, alpha=0.3)

        _fig.suptitle(f"End-to-End Impact of KV Cache Compaction — {qa_data['model_name']}", fontsize=13)
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compression-Quality Tradeoff

    The sweep below runs every available method at 12 compression ratios and plots quality vs cache size. This is the central result: AM methods (HighestAttnKeys and OMP) consistently outperform naive baselines, especially at aggressive compression (low keep ratio).
    """)
    return


@app.cell
def _(ALGO_INSTANCES, k, np, pd, q, run_compaction, seed, v):
    _ratios = np.linspace(0.05, 0.90, 12)
    _sweep_rows = []
    for _method_name in ALGO_INSTANCES:
        for _r in _ratios:
            _res = run_compaction(_method_name, q, k, v, float(_r), seed.value)
            _sweep_rows.append(
                {
                    "method": _method_name,
                    "keep_ratio": float(_r),
                    "cosine_similarity": _res["cosine_similarity"],
                }
            )
    sweep = pd.DataFrame(_sweep_rows)
    sweep
    return (sweep,)


@app.cell
def _(plt, sweep):
    _method_styles = {
        "AM-HighestAttnKeys": ("#2563eb", "o", "-"),
        "AM-OMP": ("#7c3aed", "s", "-"),
        "KVMerger": ("#d97706", "^", "--"),
        "Random": ("#6b7280", "x", ":"),
        "Truncation": ("#dc2626", "D", ":"),
    }

    _fig, _ax = plt.subplots(figsize=(9, 5))
    for _method, _grp in sweep.groupby("method"):
        _grp = _grp.sort_values("keep_ratio")
        _color, _marker, _ls = _method_styles.get(
            _method, ("#000000", "o", "-")
        )
        _ax.plot(
            _grp["keep_ratio"],
            _grp["cosine_similarity"],
            marker=_marker,
            linestyle=_ls,
            color=_color,
            label=_method,
            markersize=5,
            linewidth=2,
        )
    _ax.set_xlabel("Keep ratio (fraction of original cache)")
    _ax.set_ylabel("Cosine similarity to full attention output")
    _ax.set_title("Compression-Quality Tradeoff: All Methods")
    _ax.legend(loc="lower right")
    _ax.set_ylim(None, 1.02)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why $\beta$ and $C_2$ Matter

    A common question: why not just keep a subset of keys and their original values? The answer is that Attention Matching adds two critical corrections:

    **1. Partition function correction ($\beta$)**

    When we drop keys, the denominator of softmax changes: $\sum_{j \in \text{kept}} \exp(q \cdot K_j / \sqrt{d}) \neq \sum_{j=1}^{T} \exp(q \cdot K_j / \sqrt{d})$

    The bias terms $\beta$ correct for this: they're solved via non-negative least squares (NNLS) so that the compacted partition function matches the original. Without $\beta$, the attention weights are miscalibrated.

    **2. Value fitting ($C_2$)**

    Even with correct attention weights, using $V[\text{indices}]$ as values loses information from dropped tokens. Instead, AM solves:

    $$C_2 = \arg\min_{C_2} \| \text{softmax}(q C_1^\top / \sqrt{d} + \beta) \cdot C_2 - \text{softmax}(q K^\top / \sqrt{d}) \cdot V \|^2$$

    This least-squares fit lets the compacted values carry information from *all* original tokens, not just the retained ones.

    The ablation below shows the impact: we compare the full AM method against variants with $\beta=0$ (no bias correction) and $C_2 = V[\text{indices}]$ (direct value selection).
    """)
    return


@app.cell
def _(
    ALGO_INSTANCES,
    cosine_similarity,
    k,
    keep_ratio,
    pd,
    q,
    seed,
    set_seed,
    torch,
    v,
):
    # Ablation: show impact of beta and C2 fitting
    # Use the HighestAttentionKeysCompaction class from the already-imported ALGO_INSTANCES
    _HAK = type(ALGO_INSTANCES["AM-HighestAttnKeys"])

    set_seed(seed.value)
    _H, _T, _D = q.shape
    _t = max(1, int(round(_T * keep_ratio.value)))
    _scale = _D ** -0.5

    # Full AM (beta + C2 fitting)
    _algo_full = _HAK(
        score_method="rms", nnls_iters=2, nnls_lower_bound=0.05,
        nnls_upper_bound=20.0, c2_method="lsq", beta_method="nnls",
    )
    # No beta (beta=0, but C2 still fitted)
    _algo_no_beta = _HAK(
        score_method="rms", nnls_iters=0, c2_method="lsq", beta_method="zero",
    )
    # No C2 fitting (beta fitted, but C2 = V[indices])
    _algo_no_c2 = _HAK(
        score_method="rms", nnls_iters=2, nnls_lower_bound=0.05,
        nnls_upper_bound=20.0, c2_method="direct", beta_method="nnls",
    )
    # Neither (just subset selection)
    _algo_naive = _HAK(
        score_method="rms", nnls_iters=0, c2_method="direct", beta_method="zero",
    )

    _ablation_rows = []
    for _label, _algo in [
        ("Full AM (beta + C2 fit)", _algo_full),
        ("No beta (C2 fit only)", _algo_no_beta),
        ("No C2 fit (beta only)", _algo_no_c2),
        ("Naive subset", _algo_naive),
    ]:
        _cosines = []
        for _h in range(_H):
            _C1, _beta, _C2, _idx = _algo.compute_compacted_cache(
                k[_h], v[_h], q[_h], _t
            )
            # Full output
            _fs = (q[_h] @ k[_h].T).float() * _scale
            _fw = torch.softmax(_fs, dim=-1)
            _fo = _fw @ v[_h].float()
            # Compact output
            _cs = (q[_h] @ _C1.T).float() * _scale + _beta.float().unsqueeze(0)
            _cw = torch.softmax(_cs, dim=-1)
            _co = _cw @ _C2.float()
            _cosines.append(cosine_similarity(_fo, _co))
        _ablation_rows.append({
            "variant": _label,
            "mean_cosine": sum(_cosines) / len(_cosines),
            "min_cosine": min(_cosines),
        })

    ablation_df = pd.DataFrame(_ablation_rows)
    ablation_df
    return (ablation_df,)


@app.cell
def _(ablation_df, plt):
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _colors = ["#2563eb", "#7c3aed", "#d97706", "#6b7280"]
    _ax.barh(
        ablation_df["variant"],
        ablation_df["mean_cosine"],
        color=_colors,
        edgecolor="white",
        height=0.6,
    )
    _ax.set_xlabel("Mean cosine similarity (across heads)")
    _ax.set_title("Ablation: Impact of beta and C2 fitting")
    _ax.set_xlim(min(0.5, ablation_df["mean_cosine"].min() - 0.05), 1.02)
    _ax.axvline(1.0, linestyle=":", color="gray", alpha=0.5)
    for _i, _row in ablation_df.iterrows():
        _ax.text(
            _row["mean_cosine"] + 0.005,
            _i,
            f"{_row['mean_cosine']:.4f}",
            va="center",
            fontsize=9,
        )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm Comparison: How They Choose

    Each compaction method selects (or constructs) the retained keys differently:

    | Method | Key Selection | Beta | C2 |
    |--------|--------------|------|-----|
    | **AM-HighestAttnKeys** | Top-$t$ keys by RMS attention score | NNLS fit | Least-squares fit |
    | **AM-OMP** | Greedy OMP approximating the partition function | NNLS fit | Least-squares fit |
    | **KVMerger** | Merge consecutive similar keys via Gaussian kernel averaging | Zero | Merged from original values |
    | **Random** | Random selection without replacement | NNLS fit | Least-squares fit |
    | **Truncation** | Keep the first $t$ tokens | NNLS fit | Least-squares fit |

    Note that Random and Truncation still benefit from AM's beta and C2 fitting -- even random key selection becomes much better when paired with proper value reconstruction. The key insight: **how you reconstruct matters as much as what you keep**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Going Further

    This notebook demonstrates the core ideas on synthetic data. To see the real impact:

    - **Run on a real model**: Use `examples/qa_demo.py` to compact a Qwen3-4B model's KV cache and test QA accuracy before/after.
    - **Non-uniform budgets**: The `head_budget_optimization/` directory contains tools for computing optimal per-head compression ratios.
    - **Chunked compaction**: For very long contexts that don't fit in memory, the codebase supports chunked compaction via `compaction/compaction_methods/chunked.py`.
    - **Full evaluation**: `evaluation/run_qa_evaluation.py` benchmarks methods on QuALITY, LongHealth, and other long-document QA tasks.

    ```bash
    # Try the quick demo (requires a GPU for the full model):
    python -m examples.qa_demo --model Qwen/Qwen3-4B --target-size 0.1
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
