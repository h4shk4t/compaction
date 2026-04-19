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

    mo.sidebar(
        [
            mo.md("## Controls"),
            scenario,
            method,
            seq_len,
            n_heads,
            d_head,
            keep_ratio,
            noise,
            seed,
        ]
    )
    return d_head, keep_ratio, method, n_heads, noise, scenario, seed, seq_len


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
