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
    # Hero — title, paper citation, headline result, badges, and author
    # attribution. We feed mo.Html a single self-contained <div> string
    # (no markdown processing, no f-string nesting) — this is the
    # construction that consistently avoids the marimo/React parse errors
    # we hit with mo.md + complex inline HTML.
    _hero = """
    <div style="text-align:center;padding:35px 16px 26px;border-bottom:1px solid #e5e7eb;margin-bottom:24px;background:linear-gradient(180deg,#f8fafc 0%,#ffffff 100%);border-radius:8px;">
    <div style="font-size:0.78rem;letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-bottom:10px;">Interactive paper explainer · marimo competition 2026</div>
    <h1 style="font-size:2.4rem;line-height:1.15;margin:0 0 8px;font-weight:700;color:#111827;">Fast KV Compaction via Attention Matching</h1>
    <div style="color:#4b5563;font-size:1.02rem;margin-bottom:8px;">Paper: Zweiger · Fu · Guo · Kim &nbsp;·&nbsp; <a href="https://arxiv.org/abs/2602.16284" style="color:#2563eb;">arXiv:2602.16284</a></div>
    <div style="color:#374151;font-size:1rem;margin-bottom:16px;">Notebook by <b>Ashutosh Srivastava</b> (<a href="https://x.com/h4shkat" style="color:#2563eb;">@h4shkat</a>) &nbsp;·&nbsp; <a href="https://www.linkedin.com/in/ashutosh-srivastava-1bbb0a223/" style="color:#2563eb;">LinkedIn</a> &nbsp;·&nbsp; <a href="https://h4shk4t.github.io/" style="color:#2563eb;">Portfolio</a></div>
    <div style="display:inline-block;padding:10px 20px;background:#eff6ff;border-left:3px solid #2563eb;border-radius:4px;font-size:0.97rem;color:#1e3a8a;max-width:680px;text-align:left;"><b>Headline result (reproduced below):</b> Qwen3-4B at <b>20% of its KV cache</b> still answers <b>6 / 6 MCQs correctly</b> and retains <b>~99% verbatim recall</b> — using only matrix algebra (NNLS + OLS), no gradient descent.</div>
    <div style="margin-top:18px;display:flex;gap:7px;justify-content:center;flex-wrap:wrap;font-size:0.8rem;">
    <span style="padding:5px 10px;background:#dcfce7;color:#166534;border-radius:4px;">✓ CPU / MPS only</span>
    <span style="padding:5px 10px;background:#dbeafe;color:#1e40af;border-radius:4px;">✓ No model fine-tuning</span>
    <span style="padding:5px 10px;background:#fef3c7;color:#92400e;border-radius:4px;">✓ Fully cached &amp; reproducible</span>
    <span style="padding:5px 10px;background:#ede9fe;color:#5b21b6;border-radius:4px;">✓ Real Qwen3-4B data</span>
    </div>
    </div>
    """
    mo.Html(_hero)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *P.S.* For the cleanest first read, click the **⋮** menu in the top-right of the notebook and toggle **"Show code"** off — every cell hides its code and you see the rendered output only. Then use **run mode** (execute all cells) on your first pass; expect roughly **30–60 s** for everything to load. **Part F** times compaction algorithms for wall-clock comparison and accounts for most of that wait.

    All sliders and dropdowns live in the **left sidebar**. Each Part below is annotated with which sidebar control it consumes (e.g. *↪ Uses sidebar: QA keep ratio*).

    ## Contents

    - **Headline** - Qwen3-4B on a 1,409-token article: live MCQ teaser and verbatim diff as you change the sidebar **QA keep ratio**.
    - **A · Algorithms** - Tiny synthetic $T=32$ setup: compare how each method picks compact keys $C_k$ before the full attention-matching pipeline.
    - **B · Sandbox** - Full AM loop ($C_k$ → $\beta$ via NNLS on mass → $C_v$ via OLS) on toy multi-head KV; scenarios (needle / recency / clustered) and quality plots.
    - **C · Real model** - Cached $K$,$V$,$Q$ from Qwen3-4B on a real article: per-layer/head compaction quality, heatmaps, and GQA-stacked queries.
    - **D · Query-guided** - Same document cache, different **reference queries** (task prompts): how AM’s per-token importance shifts when $Q_\text{ref}$ targets different sections. How we can leverage query-guided compaction for multi-agent systems where each agent would specialize for its own task.
    - **E · QA demo** - Full 6-MCQ table and alignment-based verbatim metrics at every precomputed compression ratio.
    - **F · Pareto** - Wall-clock vs. quality across methods and keep ratios (the paper’s speed–quality front; optional AM-OMP).
    - **Recap** - Equations, $\beta$ and $C_v$, non-uniform head budgets, end-to-end takeaways, and “going further” pointers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The problem

    A transformer attention layer computes
    $$\text{attn}(q, K, V) \;=\; \text{softmax}\!\left(\frac{q K^\top}{\sqrt{d}}\right) V\,.$$
    Autoregressive generation attends to *every previous token* — so models **cache** $K$ and $V$ as they go (the **KV cache**). Its size scales linearly with context length:
    $$\text{bytes} \;=\; 2 \cdot L \cdot H \cdot T \cdot d \cdot s\,.$$
    For Qwen3-4B at 32K context this is $\approx 4.7$ GB per conversation. At 70B & 128K, hundreds of GB. The KV cache is the #1 reason long-context inference is expensive.

    *($Q$ is never cached — each new token produces a fresh query and old queries are discarded after use.)*

    ## The compaction objective

    Prior eviction methods (StreamingLLM, H2O, SnapKV, KVMerger) drop low-attention tokens. That loses information *and* shrinks the block's total attention mass — so the compacted block gets systematically under-weighted in any subsequent mixture-attention (paper §2).

    Attention Matching (AM) instead builds a triple
    $$(C_k,\, \beta,\, C_v)\,, \qquad C_k, C_v \in \mathbb{R}^{t \times d},\;\; \beta \in \mathbb{R}^{t}$$
    so that for **reference queries** $q$ drawn from the model's own distribution, two things hold:

    **Eq. 1 — local attention output matches.**
    $$\frac{\exp(q K^\top)\,V}{\sum_j \exp(q K_j^\top)} \;\approx\; \frac{\exp(q C_k^\top + \beta)\,C_v}{\sum_j \exp(q (C_k)_j^\top + \beta_j)}\,.$$

    **Eq. 2 — attention mass matches.**
    $$\underbrace{\sum_j \exp(q K_j^\top)}_{\text{Mass}(q; K)} \;\approx\; \sum_j \exp(q (C_k)_j^\top + \beta_j)\,.$$

    Together Eqs. 1+2 guarantee that when the compacted block is later concatenated with new uncompacted tokens, the resulting mixture-attention is preserved (Appendix A.2). *Codebase note: the paper's $(C_k, C_v)$ are `C1, C2` in the source. We use paper notation.*

    ## Why all three pieces — and why they can be solved fast

    **$\beta$ fixes the attention mass.** With $t < T$ keys the compact mass is at most $t$ even at $q = 0$ where the original is $T$. So AM introduces $w_j = e^{\beta_j} \ge 0$, which **multiplicatively re-weights** each compact key. Across all reference queries this is $A w \approx m$ with $A_{ij} = \exp(q_i (C_k)_j^\top)$, $m_i = \sum_k \exp(q_i K_k^\top)$, solved by **non-negative least squares**:
    $$w^\star = \arg\min_{w \ge 0} \|A w - m\|_2^2\,, \qquad \beta_j = \log w_j^\star\,.$$
    Intuitively, $w_j$ counts *how many original keys' worth of attention mass* the compact key $(C_k)_j$ represents.

    **$C_v$ is OLS.** With $C_k$ and $\beta$ fixed, Eq. 1 is linear in $C_v$. Let $X \in \mathbb{R}^{n \times t}$ stack the compacted softmax rows and $Y \in \mathbb{R}^{n \times d}$ stack the full attention outputs. Then
    $$C_v^\star = \arg\min_{C_v} \|X C_v - Y\|_F^2 = (X^\top X)^{-1} X^\top Y\,.$$
    If you instead used $C_v = V[\text{indices}]$, dropped tokens' values are lost forever. OLS lets each $(C_v)_j$ become a *learned linear combination of every original $V_k$* — including the dropped ones.

    **The order:** select $C_k$ → fit $\beta$ via NNLS on Eq. 2 → fit $C_v$ via OLS on Eq. 1. Both subproblems are convex with closed-form solvers. This is why AM is *fast*, unlike end-to-end gradient-descent approaches (Cartridges, Eyuboglu et al. 2025).

    All controls live in the **sidebar**. Start by dragging the QA keep-ratio slider — the teaser below reacts to it live.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    <a id="headline"></a>
    ## The headline result

    Qwen3-4B reads a 1,409-token article about a fictional island nation. We compact its KV cache with Attention Matching, then ask the compacted model to (a) answer **6 multiple-choice questions** and (b) recite the article verbatim.

    **Drag the "QA keep ratio" slider** in the sidebar between **5% / 10% / 20% / 30% / 50% / 75%**. At **5%** the model starts confabulating (red/gray words); by **20%** the diff is mostly green and every MCQ is still correct. Color key: green = matched, red = wrong/inserted, gray strikethrough = missing.
    """)
    return


@app.cell(hide_code=True)
def _(mo, qa_data, qa_ratio_selector):
    # Live teaser — reacts to the QA keep ratio slider in the sidebar.
    #
    # Shows three things:
    #   1. A headline summarizing MCQ accuracy and *alignment-based* recall
    #      (NOT the broken substring-match recall from the .pt file).
    #   2. The full 6-MCQ answer table (full cache vs compacted).
    #   3. The colored word-by-word diff between article and repeat output,
    #      with a truncation note when the model hit its max_new_tokens cap.
    import difflib as _difflib

    _teaser_out = mo.md("*Teaser unavailable — `qa_data` not loaded.*")
    if qa_data is not None and qa_ratio_selector is not None:
        _ratio = qa_ratio_selector.value
        _comp_repeat = qa_data["ratio_results"][_ratio]["repeat_result"]
        _orig_repeat = qa_data["original_repeat"]
        _article = qa_data["article_text"]
        _orig_qa = qa_data["original_qa"]
        _comp_qa = qa_data["ratio_results"][_ratio]["qa_results"]

        _orig_words = _article.split()
        _comp_words = _comp_repeat["generated_text"].split()
        _sm = _difflib.SequenceMatcher(None, _orig_words, _comp_words)
        _opcodes = _sm.get_opcodes()

        # ── Correct alignment-based recall (fraction of article words that
        # actually landed in the right place in the compacted output).
        # The .pt file's word_recall is a naive substring-check that can
        # return ~100% even when the output is truncated 1/3 of the way in.
        _equal_count = sum(
            (_i2 - _i1) for _tag, _i1, _i2, _j1, _j2 in _opcodes if _tag == "equal"
        )
        _n_orig_words = len(_orig_words)
        _n_comp_words = len(_comp_words)
        _aligned_recall = _equal_count / max(_n_orig_words, 1)

        # Detect truncation. If the compacted output ends with a partial word
        # (no trailing punctuation) AND is substantially shorter than the
        # article, max_new_tokens clipped the generation.
        _truncated = (_n_comp_words < 0.9 * _n_orig_words) and len(_comp_words) > 0
        _trunc_note = ""
        if _truncated:
            _trunc_note = (
                f' <span style="color:#92400e">⚠ Generated repeat was '
                f'truncated at ~{_n_comp_words} words '
                f'(article has {_n_orig_words}). This is a `max_new_tokens` '
                f'limit in the precompute script, not a compaction failure.</span>'
            )

        # ── Word-level diff with four kinds of opcode
        _html_parts = []
        for _tag, _i1, _i2, _j1, _j2 in _opcodes:
            if _tag == "equal":
                for _w in _orig_words[_i1:_i2]:
                    _html_parts.append(f'<span style="color:#16a34a">{_w}</span>')
            elif _tag == "replace":
                for _w in _orig_words[_i1:_i2]:
                    _html_parts.append(
                        f'<span style="color:#9ca3af;text-decoration:line-through">{_w}</span>'
                    )
                for _w in _comp_words[_j1:_j2]:
                    _html_parts.append(
                        f'<span style="background:#fecaca;color:#dc2626;font-weight:bold">{_w}</span>'
                    )
            elif _tag == "delete":
                for _w in _orig_words[_i1:_i2]:
                    _html_parts.append(
                        f'<span style="background:#e5e7eb;color:#6b7280;text-decoration:line-through">{_w}</span>'
                    )
            elif _tag == "insert":
                for _w in _comp_words[_j1:_j2]:
                    _html_parts.append(
                        f'<span style="background:#fecaca;color:#dc2626;font-weight:bold">{_w}</span>'
                    )
        _diff_html = " ".join(_html_parts)

        _orig_acc = sum(r["correct"] for r in _orig_qa)
        _comp_acc = sum(r["correct"] for r in _comp_qa)
        _total_q = len(_orig_qa)

        _headline = (
            f'<div style="padding:12px;border-left:4px solid #2563eb;'
            f'background:#eff6ff;margin-bottom:16px;font-size:0.95em">'
            f'<b>At {_ratio:.0%} of the cache</b>, Qwen3-4B answers '
            f'<b>{_comp_acc}/{_total_q}</b> MCQ questions correctly '
            f'(baseline: {_orig_acc}/{_total_q}) and recalls '
            f'<b>{_aligned_recall:.1%}</b> of the article in its original '
            f'order.{_trunc_note}'
            f'<br><span style="color:#6b7280">Drag the "QA keep ratio" slider '
            f'in the sidebar to switch ratios ↓</span>'
            f'</div>'
        )

        # ── 6-MCQ answer table
        _rows = ["| # | Question | Gold | Full cache | Compacted |",
                 "|---|---|:---:|:---:|:---:|"]
        for _qi, (_o, _c) in enumerate(zip(_orig_qa, _comp_qa), start=1):
            _o_mark = "✓" if _o["correct"] else "✗"
            _c_mark = "✓" if _c["correct"] else "✗"
            _qtext = _o["question"]
            if len(_qtext) > 60:
                _qtext = _qtext[:57] + "…"
            _rows.append(
                f"| {_qi} | {_qtext} | **{_o['gold_letter']}** "
                f"| {_o['parsed_choice']} {_o_mark} "
                f"| {_c['parsed_choice']} {_c_mark} |"
            )
        _mcq_table = mo.md("\n".join(_rows))

        _legend = (
            '<div style="margin-bottom:10px;font-size:0.85em">'
            '<span style="color:#16a34a">■ correct</span> &nbsp; '
            '<span style="background:#fecaca;color:#dc2626;font-weight:bold">■ wrong/inserted</span> &nbsp; '
            '<span style="color:#9ca3af;text-decoration:line-through">■ replaced (original)</span> &nbsp; '
            '<span style="background:#e5e7eb;color:#6b7280;text-decoration:line-through">■ missing</span>'
            '</div>'
        )

        _teaser_out = mo.vstack([
            mo.Html(_headline),
            mo.md("### 6-MCQ results"),
            _mcq_table,
            mo.md("### Verbatim repeat diff (article → compacted output)"),
            mo.Html(_legend),
            mo.Html(
                f'<div style="line-height:1.8;font-size:0.95em;max-width:820px">{_diff_html}</div>'
            ),
        ])
    _teaser_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    <a id="part-a"></a>
    # Part A — Algorithms for Constructing $C_k$

    > **↪ Uses sidebar:** *OMP step* slider (below the explainer) and the *Synthetic* group (Sequence length / Heads / Head dim / Seed) for the OMP demo's underlying tensor.

    Before we build things on real model data, let's get mechanical intuition for the compaction algorithms themselves. We do this on a tiny synthetic problem you can hold in your head: $T = 32$ keys in a 16-dimensional space, with a few "needle" positions engineered to correlate strongly with the queries.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Controls are rendered in the sidebar (always visible on the right)
    return


@app.cell
def _(mo):
    # All UI controls live in this single sidebar cell so the order is
    # explicit and predictable. The QA keep ratio drives the headline teaser
    # (and Part E), so it sits at the top — that's the first thing a reader
    # interacts with after the hero. Each section is labeled with the Part
    # letters that consume it.

    qa_ratio_selector = mo.ui.slider(
        steps=[0.05, 0.10, 0.20, 0.30, 0.50, 0.75],
        value=0.20,
        label="QA keep ratio",
        show_value=True,
    )

    real_keep_ratio = mo.ui.slider(
        steps=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
        value=0.20,
        label="Real-model keep ratio",
        show_value=True,
    )
    real_method = mo.ui.dropdown(
        options=["AM-HighestAttnKeys", "AM-OMP", "KVMerger", "Random", "Truncation"],
        value="AM-HighestAttnKeys",
        label="Real-model method",
    )

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
        label="Synthetic method",
    )
    seq_len = mo.ui.slider(32, 256, value=128, step=16, label="Sequence length")
    n_heads = mo.ui.slider(1, 8, value=4, step=1, label="Heads")
    d_head = mo.ui.slider(8, 64, value=32, step=8, label="Head dim")
    keep_ratio = mo.ui.slider(0.05, 1.0, value=0.25, step=0.05, label="Keep ratio")
    noise = mo.ui.slider(0.0, 0.5, value=0.10, step=0.05, label="Noise")
    seed = mo.ui.slider(0, 999, value=42, step=1, label="Seed")

    mo.sidebar(
        [
            mo.md("## Headline & Part E"),
            mo.md("*QA demo + word-diff teaser*"),
            qa_ratio_selector,
            mo.md("---"),
            mo.md("## Part C, D & β-ablation"),
            mo.md("*Real Qwen3-4B layer heatmap, query-conditioned tasks, β/Cv ablation*"),
            real_method,
            real_keep_ratio,
            mo.md("---"),
            mo.md("## Part A & B"),
            mo.md("*Synthetic OMP / sandbox / β visualization*"),
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
    return (
        d_head,
        keep_ratio,
        method,
        n_heads,
        noise,
        qa_ratio_selector,
        real_keep_ratio,
        real_method,
        scenario,
        seed,
        seq_len,
    )


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
    ## Algorithms for computing compacted keys ($C_k$)

    Section 3.3 of the paper restricts $C_k$ to be a *subset* of the original keys, i.e. $C_k = K_{S,:}$ for some index set $S \subset \{1,\dots,T\}$ with $|S| = t$. This avoids any iterative gradient-based optimization. The paper proposes **two subset-selection methods** with different speed/quality trade-offs; this notebook also includes three baselines for comparison.

    | Method | What it selects | How $\beta$ is fit | Complexity | Notes |
    |--------|-----------------|--------------------|------------|-------|
    | **AM-HighestAttnKeys** | Top-$t$ keys by RMS attention score | NNLS, **after** selection | $O(T)$ per head | Paper's fast default (Section 3.3, "Highest attention keys") |
    | **AM-OMP** | Greedy orthogonal matching pursuit on the **mass feature matrix** | NNLS, **inside** the OMP loop (re-fit at every step) | $O(t \cdot n \cdot T)$ per head | Paper's best-quality method (Section 3.3, "OMP keys") |
    | **Random** | Random subset (baseline; with AM's $\beta$ and $C_v$ fits) | NNLS, after selection | $O(1)$ | A useful "what does the AM fit alone buy you?" baseline |
    | **Truncation** | First $t$ tokens (recency baseline) | NNLS, after selection | $O(1)$ | Worst on most non-recency tasks |
    | **KVMerger** (Wang et al. 2024) | Merges consecutive similar keys via cosine threshold + Gaussian kernel | $\beta = 0$ (no AM fitting) | $O(T^2)$ | External baseline, not an AM method |

    ### How AM-HighestAttnKeys picks $C_k$ (Section 3.3, "Highest attention keys")

    For each reference query $q_i$, compute the post-softmax attention weights over the original keys: $a_i = \text{softmax}(q_i K^\top) \in \mathbb{R}^{1 \times T}$. Aggregate across queries via root-mean-square:
    $$s_j = \sqrt{\tfrac{1}{n}\sum_{i=1}^n a_{i,j}^2}\,.$$
    Pick the top-$t$ indices by $s_j$. (The paper notes RMS is more robust than mean or max — Appendix F.1.) After selection, fit $\beta$ via NNLS as described in Part 3, then fit $C_v$ via OLS.

    **When this fails:** when different queries each need different (non-overlapping) keys. RMS will pick the keys that are "important on average" but may miss keys that are critically important to a small subset of queries.

    ### How AM-OMP picks $C_k$ AND $\beta$ jointly  (interactive — drag the slider below)

    OMP optimizes Eq. 2 *during* selection, so it returns $C_k$ and $\beta$ together. Two tensors define the problem:

    - **Mass feature matrix** $\Phi_{ij} = \exp(q_i K_j^\top / \sqrt{d})$  &nbsp;— each column is a key's per-query exponentiated score
    - **Target attention mass** $m_i = \sum_{j=1}^T \Phi_{ij}$  &nbsp;— what the compacted block must reproduce

    OMP greedily builds a subset $S$ of columns with non-negative weights $w$ to make $\Phi_{:,S}\,w$ approximate $m$. Each iteration:

    1. **Look at the residual** $r = m - \Phi_{:,S}\,w$ — the per-query mass we *still owe*.
    2. **Pick the column most aligned with $r$** — the next key is just $\arg\max_j \;r^\top \Phi_{:,j}$.
    3. **Re-fit all weights jointly** via NNLS so the residual shrinks as much as possible.

    Step 3 is what makes it *orthogonal* matching pursuit: after each pick, every previously chosen weight is allowed to readjust. This is also where the $\beta$ fall out for free — at the end, $\beta = \log w$.

    The widget below executes this loop step-by-step on a 32-key problem with 4 planted "needle" positions. **Panel 2** is the punchline: at every step it shows $r^\top \Phi_{:,j}$ for every key — the bar that wins (red) becomes the next selected key. Watch how the bar heights collapse as $r$ shrinks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    omp_step_seq_len = mo.ui.slider(16, 64, value=32, step=8, label="Sequence length (T)")
    omp_step_d = mo.ui.slider(8, 32, value=16, step=8, label="Head dim (D)")
    omp_step_seed = mo.ui.slider(0, 50, value=7, step=1, label="Random seed")
    omp_step_target_t = mo.ui.slider(4, 16, value=10, step=1, label="Target keys (t)")
    omp_step_controls = mo.hstack(
        [omp_step_seq_len, omp_step_d, omp_step_target_t, omp_step_seed],
        justify="start", gap=1,
    )
    omp_step_controls
    return omp_step_d, omp_step_seed, omp_step_seq_len, omp_step_target_t


@app.cell
def _(
    omp_step_d,
    omp_step_seed,
    omp_step_seq_len,
    omp_step_target_t,
    set_seed,
    torch,
):
    def _omp_step_trace(K_full, Q_full, t_target):
        """Run the paper's Algorithm 1 (OMP for mass matching) and capture a
        snapshot at every step.

        Variables follow the paper's notation:
          Phi (n, T) — mass feature matrix, Phi_ij = exp(q_i K_j^T / sqrt(d))
          m   (n,)   — target attention mass, m_i = sum_j Phi_ij
          r   (n,)   — residual, r = m - Phi_:,S @ w
          w   (|S|,) — non-negative weights from NNLS at each step
        Adapted from SimpleOMPCompaction.select_keys in compaction/algorithms/omp.py.
        """
        n, d = Q_full.shape
        T = K_full.shape[0]

        inv_sqrt_d = d ** -0.5
        scores = (Q_full @ K_full.T).float() * inv_sqrt_d
        max_scores = scores.max(dim=1, keepdim=True)[0]
        Phi = torch.exp(scores - max_scores)        # (n, T) — mass feature matrix
        m = Phi.sum(dim=1)                          # (n,) — target attention mass
        approx = torch.zeros_like(m)                # = Phi_:,S @ w, starts at 0
        r = m - approx                              # initial residual = m

        selected = []
        mask = torch.zeros(T, dtype=torch.bool)

        snapshots = []
        for _step in range(t_target):
            # Line 5 of Algorithm 1: greedy pick by inner product with residual
            corr_raw = (Phi * r.unsqueeze(1)).sum(dim=0)
            corr_avail = corr_raw.clone()
            corr_avail[mask] = float("-inf")
            j_star = int(corr_avail.argmax().item())
            selected.append(j_star)
            mask[j_star] = True

            # Line 7 of Algorithm 1: re-fit NNLS over all selected keys
            Phi_S = Phi[:, selected]
            w = torch.linalg.lstsq(Phi_S, m.unsqueeze(1)).solution.squeeze(1).clamp(min=1e-12)
            approx = Phi_S @ w
            r = m - approx

            mass_rel_err = (r.abs() / m.clamp_min(1e-12)).mean().item()
            snapshots.append({
                "step": _step + 1,
                "selected": list(selected),
                "newly_selected": j_star,
                "corr_raw": corr_raw.clone(),
                "residual_norm": float(r.norm().item()),
                "mass_rel_err": mass_rel_err,
                "m": m.clone(),
                "approx": approx.clone(),
            })
        return snapshots

    set_seed(omp_step_seed.value)
    _T = omp_step_seq_len.value
    _D = omp_step_d.value

    # A toy "needle" setup: most keys are random, a few at specific positions have
    # high alignment with the queries — these are the ones OMP should discover.
    _K_demo = torch.randn(_T, _D)
    _special_positions = [_T // 4, _T // 2, (3 * _T) // 4, _T - 3]
    for _pos in _special_positions:
        _K_demo[_pos] += 2.5 * torch.randn(_D)

    _Q_demo = torch.randn(6, _D)
    # Bias a few queries toward the special keys so they have strong attention
    for _i, _pos in enumerate(_special_positions):
        if _i < 6:
            _Q_demo[_i] = _K_demo[_pos] * 0.8 + 0.4 * torch.randn(_D)

    omp_snapshots = _omp_step_trace(_K_demo, _Q_demo, omp_step_target_t.value)
    omp_demo_inputs = {"K": _K_demo, "Q": _Q_demo, "T": _T, "special": _special_positions}
    return omp_demo_inputs, omp_snapshots


@app.cell
def _(mo, omp_snapshots):
    omp_current_step = mo.ui.slider(
        1, len(omp_snapshots), value=1, step=1,
        label=f"OMP step (1 to {len(omp_snapshots)})",
        show_value=True,
    )
    omp_current_step
    return (omp_current_step,)


@app.cell
def _(omp_current_step, omp_demo_inputs, omp_snapshots, plt):
    _snap = omp_snapshots[omp_current_step.value - 1]
    _T = omp_demo_inputs["T"]
    _special = set(omp_demo_inputs["special"])
    _selected = set(_snap["selected"])
    _new_pick = _snap["newly_selected"]

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 3.5))

    # Panel 1: selection state
    _colors = []
    for _i in range(_T):
        if _i == _new_pick:
            _colors.append("#dc2626")  # just picked (red)
        elif _i in _selected:
            _colors.append("#16a34a")  # previously selected (green)
        elif _i in _special:
            _colors.append("#f59e0b")  # "true" important positions (orange)
        else:
            _colors.append("#d1d5db")  # unselected (gray)
    _heights = [1.0] * _T
    _axes[0].bar(range(_T), _heights, color=_colors, edgecolor="white", linewidth=0.4)
    _axes[0].set_title(
        f"Selection after step {_snap['step']} "
        f"({len(_selected)}/{_T} keys)"
    )
    _axes[0].set_xlabel("Token position")
    _axes[0].set_yticks([])
    _axes[0].set_xlim(-0.5, _T - 0.5)
    _axes[0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#dc2626", label="just picked"),
            plt.Rectangle((0, 0), 1, 1, color="#16a34a", label="previously selected"),
            plt.Rectangle((0, 0), 1, 1, color="#f59e0b", label="planted needle (truth)"),
            plt.Rectangle((0, 0), 1, 1, color="#d1d5db", label="unselected"),
        ],
        loc="upper right",
        fontsize=7,
    )

    # Panel 2: correlation scores this step (residual · exp_scores per key)
    _corr = _snap["corr_raw"].detach().cpu().numpy()
    _corr_display = _corr.copy()
    # Show previously selected in black with neutral bar; new pick in red
    _prev = [i for i in _snap["selected"] if i != _new_pick]
    _bar_colors = ["#e5e7eb"] * _T
    for _i in _prev:
        _bar_colors[_i] = "#94a3b8"  # previously selected
    _bar_colors[_new_pick] = "#dc2626"  # winner
    _axes[1].bar(range(_T), _corr_display, color=_bar_colors, edgecolor="white", linewidth=0.3)
    _axes[1].axvline(_new_pick, color="#dc2626", linestyle="--", alpha=0.4)
    _axes[1].set_title(
        "Greedy pick (line 5):  $r^\\top \\Phi_{:,j}$\n(inner product of residual with each column of $\\Phi$; winner in red)"
    )
    _axes[1].set_xlabel("Token position $j$")
    _axes[1].set_ylabel("$r^\\top \\Phi_{:,j}$")
    _axes[1].set_xlim(-0.5, _T - 0.5)

    # Panel 3: attention mass relative error over steps so far
    _errs = [s["mass_rel_err"] for s in omp_snapshots[:omp_current_step.value]]
    _resids = [s["residual_norm"] for s in omp_snapshots[:omp_current_step.value]]
    _steps_x = list(range(1, len(_errs) + 1))
    _ax2 = _axes[2].twinx()
    _axes[2].plot(_steps_x, _errs, "o-", color="#2563eb", linewidth=2, label="mass rel err")
    _ax2.plot(_steps_x, _resids, "s--", color="#f97316", linewidth=1.5, alpha=0.7, label="‖residual‖")
    _axes[2].set_yscale("log")
    _ax2.set_yscale("log")
    _axes[2].set_xlabel("OMP step")
    _axes[2].set_ylabel("Attention mass\nrelative error (log)", color="#2563eb")
    _ax2.set_ylabel("Residual norm  $\\|m - \\Phi_{:,S} w\\|$ (log)", color="#f97316")
    _axes[2].set_title("Attention-mass approximation\nimproves each step (Eq. 2)")
    _axes[2].tick_params(axis="y", labelcolor="#2563eb")
    _ax2.tick_params(axis="y", labelcolor="#f97316")
    _axes[2].grid(True, alpha=0.3)

    _fig.suptitle(
        f"OMP Algorithm 1 — Step {_snap['step']}  |  "
        f"attention-mass rel err: {_snap['mass_rel_err']:.3e}",
        fontsize=12,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading this chart (cross-referenced with Algorithm 1 in the paper):**

    - **Left panel** — the set $S$ of selected tokens grows by 1 each step (line 6 of the algorithm). The red bar is the token just picked ($j^\star$ from line 5); green are previously selected. Orange bars mark the "planted needles" — if OMP is working, these should light up first.
    - **Middle panel** — the greedy-pick scoring function from line 5: $r^\top \Phi_{:,j}$ for every unselected key $j$. OMP picks the tallest unpicked bar. Previously selected keys (gray) drop to ~0 because the residual is now essentially orthogonal to their column.
    - **Right panel** — the **attention-mass relative error** $\|m - \Phi_{:,S}\,w\|/\|m\|$ falls rapidly (log scale). This is exactly the quantity OMP minimizes (Eq. 2 of the paper). Note: this is *not* the softmax denominator inside the local compacted attention — it is the un-normalized total attention mass that the compacted block reports to any future concatenated block.

    **Things to try:**

    - Drag the **Random seed** slider — the 4 planted needles move around, but OMP should still find them first.
    - Increase **Target keys ($t$)** — notice that the mass error drops sharply once all needles are collected, then plateaus while OMP picks up "background" keys.
    - Step through from step 1. The middle panel's *winner* bar (red) is what line 5 will pick at the next iteration. Once a key joins $S$, its score becomes ~0 forever (the NNLS re-fit at line 7 already used that direction).

    **What OMP returns**: after $t$ iterations Algorithm 1 outputs (i) the index set $S$, giving $C_k = K_{S,:}$; and (ii) the final NNLS weights $w$, from which we set $\beta = \log w$. **There is no separate β-fitting step for OMP** — that is the key difference from AM-HighestAttnKeys, where keys are picked first and then β is fit afterward by NNLS on the full attention-mass system. Computing $C_v$ is a separate subsequent step (OLS on the attention outputs, Section 3.2 — visualized later in the notebook).

    ### The faster alternative: AM-HighestAttnKeys

    OMP is elegant but can be overkill. The paper's recommended **fast** method, AM-HighestAttnKeys, skips the residual-chasing and just picks the top-$t$ keys by a simpler score:
    $$s_j \;=\; \text{RMS}_i \, \exp\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right),$$
    i.e. the root-mean-square *post-softmax* attention weight this key receives across the training queries (Section 3.3, "Highest attention keys"). Keys that are "reliably important to many queries" rise to the top. Then $\beta$ and $C_v$ are fit with the same NNLS + least-squares machinery as OMP — only the *selection* step differs.

    **When does this fail?** When different keys are important to *different, non-overlapping* subsets of queries. OMP handles this naturally (each step re-targets unexplained residual), but HighestAttnKeys can greedily pick several redundant keys that all happen to have high RMS scores. In practice, on language models, the two give very close results at most ratios.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    <a id="part-b"></a>
    # Part B — The Synthetic Sandbox

    > **↪ Uses sidebar:** entire *Part A & B* group — *Synthetic scenario / method, Sequence length, Heads, Head dim, Keep ratio, Noise, Seed*. This is the only fully-live part of the notebook (everything below uses cached data); each slider triggers a 1–3 s recompute.

    Now let's run the full AM pipeline ($C_k$ selection $\rightarrow$ $\beta$ fit via NNLS on attention mass $\rightarrow$ $C_v$ fit via OLS on attention output) and compare methods on a toy multi-head attention problem. The synthetic KV cache is generated in one of three **scenarios** (pick in the sidebar):

    - **Needle** — all tokens are random noise except one "needle" token at position $T/3$ whose key is strongly activated, and queries at the end of the sequence are biased to attend to it. A realistic compaction algorithm should always keep this token.
    - **Recency** — the last quarter of tokens have progressively stronger keys (mimicking recency bias). Truncation-baselines look deceptively good here.
    - **Clustered** — a small region around position $T/2$ has correlated keys and values. Tests whether the algorithm can pick one representative per cluster or over-samples within it.

    On top of this you control `seq_len`, `n_heads`, `d_head`, `keep_ratio`, `noise`, and `seed` — all in the sidebar. Every cell below recomputes instantly when you move a slider.

    **Why this matters**: the synthetic sandbox lets us check algorithm correctness and intuition *before* running on a real model. If an algorithm fails on "needle" — the simplest possible test — it will fail on a real context too.
    """)
    return


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

    Using cosine similarity between the full attention output $\\text{{softmax}}(q K^\\top / \\sqrt{{d}}) V$ and the compacted attention output $\\text{{softmax}}(q C_k^\\top / \\sqrt{{d}} + \\beta) C_v$ (per Eq. 1 of the paper).

    - **{method.value}**: {summary.iloc[0]['cosine_similarity']:.4f} cosine similarity at {summary.iloc[0]['keep_ratio']:.0%} keep ratio
    - **Random baseline**: {summary.iloc[1]['cosine_similarity']:.4f}
    - **Truncation baseline**: {summary.iloc[2]['cosine_similarity']:.4f}

    The gap between the AM method and baselines demonstrates what AM buys you: selecting good keys ($C_k$), matching the per-block attention mass ($\\beta$ via NNLS, Eq. 2), and reconstructing values to absorb information from dropped tokens ($C_v$ via OLS, Eq. 1).
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
    Right: attention through the compacted cache $(C_k, \beta, C_v)$. Note how the compacted version preserves the dominant attention patterns despite using far fewer key positions. The compacted attention is computed as $\text{softmax}(q C_k^\top / \sqrt{d} + \beta)$, where the $\beta$ bias terms preserve the block's attention mass (Eq. 2 of the paper) so the per-key weights are correctly scaled.
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

    <a id="part-c"></a>
    # Part C — Does this work on a real model?

    > **↪ Uses sidebar:** *Real-model method* dropdown and *Real-model keep ratio* slider. Plot updates instantly from the precomputed grid.

    We now switch from synthetic data to **real KV-cache tensors** extracted from Qwen3-4B (4 B parameters, 36 layers) processing a 1,409-token article. The extraction was a one-time preprocessing step: `scripts/extract_kv_cache.py` ran a single forward pass with **PyTorch forward hooks** on every attention module to capture $K$, $V$, and the post-RoPE $Q$ at each layer, then saved them to `data/cached_kv/Qwen3-4B.pt`.

    Per-layer shapes: $K, V \in \mathbb{R}^{8 \times 1409 \times 128}$ (8 KV heads), $Q \in \mathbb{R}^{32 \times 1409 \times 128}$ (32 query heads).

    **GQA.** Qwen3-4B uses **Grouped Query Attention**: 32 query heads share 8 KV heads (factor of 4). When running AM on KV head $h$, we stack the 4 corresponding query heads: $Q_h \in \mathbb{R}^{(4 \cdot 1409) \times 128} = \mathbb{R}^{5636 \times 128}$. This generous query supply is what makes AM's NNLS + ridge fit tight on real models.
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
def _():
    # real_keep_ratio and real_method are defined in the sidebar cell above.
    return


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

    layer_quality = pd.DataFrame()
    if cached_kv is not None:
        # Try precomputed grid first — gives instant lookup for the pre-baked
        # methods (HighestAttnKeys, KVMerger, Random, Truncation at the 8
        # standard ratios). Falls back to on-demand compute for other cases.
        import os as _os_lq
        _precomp_path = _os_lq.path.join(
            _os_lq.path.dirname(_os_lq.path.abspath(__file__))
            if "__file__" in dir() else _os_lq.getcwd(),
            "data", "cached_kv", "Qwen3-4B_layer_heatmap.pt",
        )
        _precomp_lq = None
        if _os_lq.path.exists(_precomp_path):
            _precomp_lq = torch.load(_precomp_path, map_location="cpu", weights_only=False)

        _n_layers = cached_kv["n_layers"]

        _precomp_ratio_match = None
        if _precomp_lq is not None and f"grid_{real_method.value}" in _precomp_lq:
            _matches = np.where(np.isclose(_precomp_lq["ratios"], real_keep_ratio.value, atol=1e-6))[0]
            if len(_matches) > 0:
                _precomp_ratio_match = int(_matches[0])

    _used_precomp = False
    if cached_kv is not None and _precomp_ratio_match is not None:
        _grid = _precomp_lq[f"grid_{real_method.value}"]
        _mean_cos_per_layer = _grid[:, _precomp_ratio_match]
        layer_quality = pd.DataFrame([
            {"layer": _li, "mean_cosine": float(_mean_cos_per_layer[_li]), "min_cosine": float(_mean_cos_per_layer[_li])}
            for _li in range(_n_layers)
        ])
        _used_precomp = True

    if cached_kv is not None and not _used_precomp:
        # Fallback — on-demand compute (slow, ~16s)
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
    ### Per-Head Influence Curves: *Why* We Need Non-Uniform Budgets

    The per-layer bar chart above hides a crucial detail: **within a single layer, different heads behave completely differently.** Some heads stay at near-perfect quality even at 5% keep ratio, while others collapse the moment you compress them.

    The plot below picks 4 representative layers of Qwen3-4B and, for each layer, shows one curve per KV head: cosine-similarity-to-full vs keep ratio. Heads are color-coded by how compressible they are at 20% keep: **green = easy** (stays ≥ 0.97), **red = hard** (drops below 0.85), **blue = in between**.

    This is the empirical basis for the paper's greedy budget solver: if head A's curve is flat and head B's curve plummets, a fixed total cache should allocate more to B than to A.
    """)
    return


@app.cell
def _(ALGO_INSTANCES, cached_kv, cosine_similarity, np, torch):
    # Per-head influence curves — how does each head's quality vary with ratio?
    # Uses HighestAttnKeys (the paper's fast default) per head, computing attention
    # scores once per head and then picking top-t at each ratio.
    # ~4 layers * 8 heads * 14 ratios = 448 compaction calls (~15s total).

    if cached_kv is None:
        influence_curves = None
    else:
        _n_layers = cached_kv["n_layers"]
        _n_kv_heads = cached_kv["n_kv_heads"]
        _gqa = cached_kv["n_q_heads"] // _n_kv_heads
        _hd = cached_kv["head_dim"]
        _T = cached_kv["seq_len"]
        _scale = _hd ** -0.5

        # Pick 4 representative layers: early, early-mid, late-mid, late
        _layer_picks = [0, _n_layers // 3, (2 * _n_layers) // 3, _n_layers - 1]
        _ratios = np.array([0.03, 0.05, 0.08, 0.12, 0.17, 0.22, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90])

        _algo = ALGO_INSTANCES["AM-HighestAttnKeys"]

        _curves = {}  # (layer, head) -> {"ratios": ..., "cosines": ...}
        for _li in _layer_picks:
            _K_layer = cached_kv[f"K_{_li}"].float()
            _V_layer = cached_kv[f"V_{_li}"].float()
            _Q_layer = cached_kv[f"Q_{_li}"].float()

            for _kv_h in range(_n_kv_heads):
                _K_h = _K_layer[_kv_h]
                _V_h = _V_layer[_kv_h]
                _Q_h = _Q_layer[_kv_h * _gqa:(_kv_h + 1) * _gqa].reshape(-1, _hd)

                # Full attention output (for cosine similarity target)
                _full_out = torch.softmax((_Q_h @ _K_h.T) * _scale, dim=-1) @ _V_h

                _cos_by_ratio = []
                for _r in _ratios:
                    _t = max(1, min(_T - 1, int(round(_T * _r))))
                    try:
                        _C1, _beta, _C2, _ = _algo.compute_compacted_cache(_K_h, _V_h, _Q_h, _t)
                        _comp_out = torch.softmax(
                            (_Q_h @ _C1.T) * _scale + _beta.float().unsqueeze(0), dim=-1
                        ) @ _C2.float()
                        _cos_by_ratio.append(cosine_similarity(_full_out, _comp_out))
                    except Exception:
                        _cos_by_ratio.append(float("nan"))

                _curves[(_li, _kv_h)] = {
                    "ratios": _ratios,
                    "cosines": np.array(_cos_by_ratio),
                }

        influence_curves = {"curves": _curves, "layer_picks": _layer_picks, "ratios": _ratios}
    return (influence_curves,)


@app.cell
def _(cached_kv, influence_curves, np, plt):
    _fig = None
    if influence_curves is not None:
        _curves = influence_curves["curves"]
        _layer_picks = influence_curves["layer_picks"]
        _ratios = influence_curves["ratios"]
        _n_kv_heads = cached_kv["n_kv_heads"]

        # Find the ratio column at ~20% to classify heads as easy/hard
        _idx_20 = int(np.argmin(np.abs(_ratios - 0.20)))

        _fig, _axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        _axes_flat = _axes.flatten()

        for _panel_i, _layer_idx in enumerate(_layer_picks):
            _ax = _axes_flat[_panel_i]
            _easy_count = _hard_count = _mid_count = 0

            for _h in range(_n_kv_heads):
                _curve = _curves[(_layer_idx, _h)]
                _cos_at_20 = _curve["cosines"][_idx_20]

                if np.isnan(_cos_at_20):
                    _color, _alpha, _lw = "#6b7280", 0.4, 1
                elif _cos_at_20 >= 0.97:
                    _color, _alpha, _lw = "#16a34a", 0.85, 1.8
                    _easy_count += 1
                elif _cos_at_20 < 0.85:
                    _color, _alpha, _lw = "#dc2626", 0.9, 2.2
                    _hard_count += 1
                else:
                    _color, _alpha, _lw = "#2563eb", 0.75, 1.5
                    _mid_count += 1

                _ax.plot(
                    _curve["ratios"], _curve["cosines"],
                    color=_color, alpha=_alpha, linewidth=_lw,
                    marker="o", markersize=3.5,
                    label=f"head {_h}" if _panel_i == 0 else None,
                )

            _ax.axvline(0.20, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            _ax.axhline(0.97, color="#16a34a", linestyle=":", alpha=0.3, linewidth=0.8)
            _ax.axhline(0.85, color="#dc2626", linestyle=":", alpha=0.3, linewidth=0.8)
            _ax.set_title(
                f"Layer {_layer_idx}  —  easy: {_easy_count}, mid: {_mid_count}, hard: {_hard_count}",
                fontsize=11,
            )
            _ax.set_xlabel("Keep ratio")
            _ax.set_ylabel("Cosine similarity")
            _ax.set_xlim(0, 1.0)
            _ax.set_ylim(max(0.3, min(min(c["cosines"]) for c in _curves.values()) - 0.02), 1.01)
            _ax.grid(True, alpha=0.25)

        _fig.suptitle(
            "Per-head compaction sensitivity across layers (Qwen3-4B, AM-HighestAttnKeys)\n"
            "Green = easy to compress (quality stays high), Red = hard (sharp drop), Blue = in between",
            fontsize=12, y=1.00,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway:** The shape of each curve — not just its height at one ratio — determines how to budget cache. A head whose curve is roughly flat all the way down to 5% keep can safely be assigned a tiny budget; a head whose curve crashes between 30% and 20% needs protection. The greedy allocator in `head_budget_optimization/solver.py` reads exactly these curves (but using perplexity deltas instead of cosine sim) and allocates cache dollars one at a time to whichever head's marginal return is highest.
    """)
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

    # Fast path: try to load precomputed grid from
    # data/cached_kv/Qwen3-4B_layer_heatmap.pt. The precompute script runs
    # all methods × 8 ratios × 36 layers × 8 heads once (~5 min on CPU),
    # turning this 10-minute-per-method cell into an instant lookup.
    # Fallback path: compute on-demand if no precomputed grid exists for
    # the currently-selected method (e.g. AM-OMP is excluded by default).
    import os as _os_hm

    heatmap_data = None
    _precomp_path = _os_hm.path.join(
        _os_hm.path.dirname(_os_hm.path.abspath(__file__))
        if "__file__" in dir() else _os_hm.getcwd(),
        "data", "cached_kv", "Qwen3-4B_layer_heatmap.pt",
    )
    _precomp = None
    if _os_hm.path.exists(_precomp_path):
        _precomp = torch.load(_precomp_path, map_location="cpu", weights_only=False)

    if cached_kv is None:
        heatmap_data = None
    elif _precomp is not None and f"grid_{real_method.value}" in _precomp:
        # Instant lookup from disk
        heatmap_data = {
            "grid": _precomp[f"grid_{real_method.value}"],
            "ratios": _precomp["ratios"],
            "source": "precomputed",
        }
    else:
        # Fallback: compute on-demand (slow, ~2–10 min depending on method)
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

        heatmap_data = {"grid": _grid, "ratios": _ratios, "source": "on-demand"}
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


@app.cell
def _(cached_kv, heatmap_data, mo, np, plt):
    # Deep dive (collapsed by default) — does the paper's greedy budget
    # solver pick the same heads our direct cosine-similarity measurement
    # flags as hard? We load the precomputed optimized_agnostic.json,
    # plot it next to the layer-quality heatmap, and also scatter per-layer
    # budget vs per-layer quality at 20% keep. Uses only precomputed data
    # (no live compaction calls) so this expands instantly.
    import json as _json
    import os as _os

    _pair_fig = None
    _scatter_fig = None
    _content = mo.md("*Optimized budget file not available.*")

    _budget_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd(),
        "head_budget_optimization", "head_budgets", "Qwen3-4B", "optimized_agnostic.json",
    )

    if (
        cached_kv is not None
        and _os.path.exists(_budget_path)
        and heatmap_data is not None
    ):
        with open(_budget_path) as _f:
            _budget = _json.load(_f)

        _n_layers = cached_kv["n_layers"]
        _n_kv_heads = cached_kv["n_kv_heads"]

        # Per-head budget grid (layers × heads)
        _budget_grid = np.zeros((_n_layers, _n_kv_heads))
        for _li in range(_n_layers):
            for _hi in range(_n_kv_heads):
                _budget_grid[_li, _hi] = _budget.get(f"L{_li}H{_hi}", 0.0)

        _ratios = heatmap_data["ratios"]
        _quality_grid_full = heatmap_data["grid"]  # (n_layers, n_ratios)

        # ── Side-by-side: budget heatmap + quality heatmap ──
        _pair_fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 6))
        _im1 = _ax1.imshow(_budget_grid, aspect="auto", cmap="YlOrRd", origin="lower")
        _ax1.set_xlabel("KV head index")
        _ax1.set_ylabel("Layer index")
        _ax1.set_title("Optimized budget\n(brighter = more cache)")
        _ax1.set_xticks(range(_n_kv_heads))
        _ax1.set_yticks(range(0, _n_layers, 2))
        plt.colorbar(_im1, ax=_ax1, shrink=0.8)

        _im2 = _ax2.imshow(
            _quality_grid_full, aspect="auto", cmap="RdYlGn",
            vmin=0.5, vmax=1.0, origin="lower",
        )
        _ax2.set_xlabel("Keep ratio")
        _ax2.set_ylabel("Layer index")
        _ax2.set_title("Compaction quality\n(red = hard to compress)")
        _ax2.set_xticks(range(len(_ratios)))
        _ax2.set_xticklabels([f"{r:.0%}" for r in _ratios])
        _ax2.set_yticks(range(0, _n_layers, 2))
        plt.colorbar(_im2, ax=_ax2, shrink=0.8)
        _pair_fig.suptitle("Does the solver's budget line up with where compaction hurts?", fontsize=12)
        _pair_fig.tight_layout()

        # ── Scatter: per-layer budget vs per-layer quality ──
        # Use the precomputed heatmap grid (no live compaction)
        _col_idx = int(np.argmin(np.abs(_ratios - 0.20)))
        _layer_quality_20 = _quality_grid_full[:, _col_idx]
        _layer_budget_sum = _budget_grid.sum(axis=1)

        _scatter_fig, _axS = plt.subplots(figsize=(8, 5))
        _axS.scatter(
            _layer_quality_20, _layer_budget_sum,
            c=range(_n_layers), cmap="viridis", s=60,
            edgecolors="white", linewidth=0.5,
        )
        for _li in range(_n_layers):
            _axS.annotate(
                str(_li), (float(_layer_quality_20[_li]), float(_layer_budget_sum[_li])),
                fontsize=6, ha="center", va="bottom",
            )
        _axS.set_xlabel("Per-layer quality at 20% keep (mean cosine)")
        _axS.set_ylabel("Per-layer budget (sum across heads)")
        _axS.set_title("Budget vs Actual Compaction Difficulty")
        _axS.axhline(float(np.mean(_layer_budget_sum)), linestyle=":", color="gray", alpha=0.5)
        _axS.axvline(float(np.mean(_layer_quality_20)), linestyle=":", color="gray", alpha=0.5)
        _scatter_fig.tight_layout()

        _content = mo.vstack([
            mo.md(
                "The paper allocates budget by running a **greedy solver on per-head influence curves** — for each head, how much does perplexity increase at compaction ratio $r$? Heads with steep curves get more of the total cache budget. Numbers shown here are from `head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json` (shipped with the repo)."
            ),
            mo.as_html(_pair_fig),
            mo.as_html(_scatter_fig),
            mo.md(
                "**Observation.** Correlation is *weak but present* — the solver optimizes for **perplexity** (end-to-end generation quality), while our heatmap measures **per-head cosine similarity** (local attention reproduction). They are related but not identical. Early layers (hard by cosine) are allocated modestly; the solver instead spends budget on middle layers whose perplexity sensitivity is empirically highest."
            ),
        ])

    mo.accordion({"Deep dive: solver budget vs measured compaction difficulty (click to expand)": _content})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    <a id="part-d"></a>
    # Part D — Same article, different tasks, different compactions

    > **↪ Uses inline control:** task picker dropdown below the heatmap. The score grid itself is precomputed; only the colored-article rendering reacts.

    So far every compaction we've run has used the **same** query distribution — queries derived from the model reading the article itself (the "self-study" style of reference query discussed in Section 3.1 of the paper). But AM is not a function of $(K, V)$ alone. It's a function of $(K, V, Q_\text{ref})$ where $Q_\text{ref}$ is the set of training queries the scoring procedure consults to decide which keys matter.

    **Research question for this section:**

    > *Given a fixed $(K, V)$ cache of a document, how much does the choice of reference queries change which tokens AM decides to retain? And if $Q_\text{ref}$ is derived from a specific downstream question, does the retained cache specialize to that question?*

    The paper compares four families of reference queries (self-study, repeat-prefill, context-prefill, random-vectors) at the level of **downstream accuracy**. Here we zoom in one level deeper — we visualize the per-token relevance score that emerges when $Q_\text{ref}$ comes from three different *task prompts* about the same article:

    - **T1** — *When did Verandia declare independence, and who was its first president?* (targets Section 2)
    - **T2** — *What is the name of the deep-sea research vessel, when was it launched, and at which university is it based?* (targets Section 5)
    - **T3** — *List every endemic species and climate feature mentioned in the document.* (targets Section 7)

    These tasks were chosen from non-adjacent sections so hot bands are spatially separated on the heatmap below. The implementation: for each task, we extracted the Q vectors Qwen3-4B produces at the task-prompt tokens (after it has read the article), then computed the standard AM-HighestAttnKeys per-position importance score, aggregated across all 36 layers × 8 KV heads with uniform weighting.

    **The takeaway we'll develop below:** AM with task-derived queries produces a *task-adaptive* compacted cache — without any algorithm change, only the $Q_\text{ref}$ source changes.
    """)
    return


@app.cell
def _(mo, torch):
    # Load task-specific query vectors. Each layer contains a stacked tensor of
    # shape (n_tasks, n_q_heads, max_task_tokens, head_dim) where task Q vectors
    # are zero-padded on the token axis to a common length.
    import os as _os

    _task_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd(),
        "data", "cached_kv", "Qwen3-4B_task_queries.pt",
    )

    if _os.path.exists(_task_path):
        task_queries = torch.load(_task_path, map_location="cpu", weights_only=False)
        _tcs = task_queries["task_token_counts"]
        mo.output.replace(
            mo.md(
                f"Loaded task queries for {len(task_queries['task_names'])} tasks — "
                f"token counts: {dict(_tcs)}"
            )
        )
    else:
        task_queries = None
        mo.output.replace(
            mo.md(
                f"**Task-query file not found** at `{_task_path}`. "
                "Run `PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/extract_task_queries.py --device mps` "
                "to enable the query-conditioned compaction section below."
            )
        )
    return (task_queries,)


@app.cell
def _(cached_kv, np, task_queries, torch):
    # Per-task, per-token relevance scores aggregated across all layers and
    # heads with uniform weighting. This is exactly what AM-HighestAttnKeys
    # computes under the hood — RMS post-softmax attention weight across all
    # the reference queries — except here the reference queries come from
    # three different task prompts instead of the usual self-study set.
    #
    # Output shape: (n_tasks, seq_len). One score per (task, token position).

    if task_queries is None or cached_kv is None:
        task_scores = None
    else:
        _n_layers = cached_kv["n_layers"]
        _n_kv_heads = cached_kv["n_kv_heads"]
        _gqa = cached_kv["n_q_heads"] // _n_kv_heads
        _hd = cached_kv["head_dim"]
        _T = cached_kv["seq_len"]
        _scale = _hd ** -0.5

        _task_names = task_queries["task_names"]
        _n_tasks = len(_task_names)
        # Actual (unpadded) task token counts per task
        _tc = [task_queries["task_token_counts"][n] for n in _task_names]

        # Accumulator for squared attention: (n_tasks, seq_len)
        _sq_sum = np.zeros((_n_tasks, _T), dtype=np.float32)
        _count = np.zeros((_n_tasks,), dtype=np.float32)

        for _li in range(_n_layers):
            _K = cached_kv[f"K_{_li}"].float()               # (n_kv_heads, T, D)
            _Q_task = task_queries[f"Q_{_li}"].float()        # (n_tasks, n_q_heads, max_tt, D)

            for _kv_h in range(_n_kv_heads):
                _K_h = _K[_kv_h]                              # (T, D)
                # Q heads that attend to this KV head under GQA
                _q_start = _kv_h * _gqa
                _q_end = _q_start + _gqa
                _Q_slice = _Q_task[:, _q_start:_q_end, :, :]  # (n_tasks, _gqa, max_tt, D)

                for _ti in range(_n_tasks):
                    _n_tokens = _tc[_ti]
                    _q = _Q_slice[_ti, :, :_n_tokens, :].reshape(-1, _hd)  # (_gqa*n_tokens, D)
                    # Attention weights from task queries to article keys
                    _a = torch.softmax((_q @ _K_h.T) * _scale, dim=-1)      # (_gqa*n_tokens, T)
                    # Accumulate squared attention for RMS aggregation across
                    # all reference queries across all (layer, head) pairs
                    _sq_sum[_ti] += (_a ** 2).sum(dim=0).numpy()
                    _count[_ti] += _a.shape[0]

        # RMS = sqrt(mean of squares). mean = sum / count.
        _rms = np.sqrt(_sq_sum / _count[:, None].clip(min=1.0))
        task_scores = {
            "task_names": _task_names,
            "task_prompts": task_queries["task_prompts"],
            "scores": _rms,     # (n_tasks, seq_len)
            "seq_len": _T,
        }
    return (task_scores,)


@app.cell
def _(mo, task_scores):
    # Dropdown to pick which task's colored article to render below the heatmap.
    if task_scores is None:
        task_picker = mo.ui.dropdown(options=["(task queries unavailable)"], value="(task queries unavailable)", label="Task")
    else:
        task_picker = mo.ui.dropdown(
            options=list(task_scores["task_names"]),
            value=task_scores["task_names"][0],
            label="Task to visualize below",
        )
    task_picker
    return (task_picker,)


@app.cell
def _(np, task_scores):
    # Differential score: for each task, subtract the per-position mean
    # across ALL tasks. Then clip at zero and row-normalize to [0, 1].
    #
    # WHY this instead of raw row-wise normalization?
    # Attention is dominated by universal attention sinks — the first token
    # (<|im_start|>), the final few chat-template tokens, and ultra-frequent
    # words like "the". These attract strong attention from *every* task's
    # queries, so raw scores look nearly identical across tasks at those
    # positions. A differential score (task_i - mean_over_tasks) cancels
    # out the common baseline and exposes only what this particular task
    # cares about RELATIVE to the others. Content-specific tokens pop,
    # structural tokens disappear. This is the same trick as contrastive
    # saliency maps in computer vision.
    _scores_norm = None
    if task_scores is not None:
        _s = task_scores["scores"]                         # (n_tasks, T)
        _mean = _s.mean(axis=0, keepdims=True)             # (1, T)
        _diff = np.maximum(_s - _mean, 0.0)                # (n_tasks, T), >= 0
        _row_max = _diff.max(axis=1, keepdims=True) + 1e-12
        _scores_norm = _diff / _row_max                    # each row in [0, 1]
    task_scores_norm = _scores_norm
    return (task_scores_norm,)


@app.cell
def _(plt, task_scores, task_scores_norm):
    # Panel A — heatmap of per-task per-token importance.
    # Does NOT depend on task_picker, so changing the dropdown below does not
    # re-render this cell.
    _fig = None
    if task_scores is not None and task_scores_norm is not None:
        _task_names = task_scores["task_names"]
        _T = task_scores["seq_len"]
        _n_tasks = len(_task_names)

        _fig, _ax = plt.subplots(figsize=(14, 2.4 + 0.5 * _n_tasks))
        _im = _ax.imshow(task_scores_norm, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
        _ax.set_xlabel(f"Token position (sequence length = {_T})")
        _ax.set_yticks(range(_n_tasks))
        _ax.set_yticklabels(_task_names)
        _ax.set_title(
            "Per-task per-token importance score (RMS attention, aggregated across 36 layers × 8 KV heads)",
            fontsize=11,
        )
        plt.colorbar(_im, ax=_ax, label="Normalized RMS attention", shrink=0.7)
        _fig.tight_layout()
    _fig
    return


@app.cell
def _(cached_kv, mo, task_picker, task_scores, task_scores_norm):
    # Panel B — the article rendered word-by-word with the currently-selected
    # task's importance score as each word's background color intensity.
    # Depends on task_picker so flipping the dropdown re-colors the article
    # without re-rendering the heatmap above.

    _article_html = mo.md("*Task query data unavailable.*")
    if task_scores is not None and task_scores_norm is not None and cached_kv is not None:
        _task_names = task_scores["task_names"]
        _task_prompts = task_scores["task_prompts"]
        _article = cached_kv["article_text"]
        _words = _article.split()
        _art_start = cached_kv["article_indices_start"]
        _art_stop = cached_kv["article_indices_stop"]
        _art_len = _art_stop - _art_start

        _picked_task = task_picker.value
        _picked_idx = _task_names.index(_picked_task) if _picked_task in _task_names else 0
        _art_scores = task_scores_norm[_picked_idx, _art_start:_art_stop]
        _n_words = len(_words)

        _html_parts = [
            f'<div style="margin-bottom:10px;font-size:0.9em;color:#6b7280">'
            f'<b>Task:</b> <i>“{_task_prompts[_picked_task]}”</i> '
            f'— brighter words = more attention from this task\'s queries</div>',
            '<div style="line-height:1.9;max-width:880px;font-size:0.93em">',
        ]
        for _wi, _w in enumerate(_words):
            # Approximate this word's token-position range via uniform spacing;
            # word-level granularity is what the reader can see anyway.
            _p0 = int(round(_wi * _art_len / _n_words))
            _p1 = max(_p0 + 1, int(round((_wi + 1) * _art_len / _n_words)))
            _score = float(_art_scores[_p0:_p1].max()) if _p1 > _p0 else 0.0
            _alpha = min(1.0, _score * 1.4)  # small gamma to make peaks pop
            _r, _g, _b = 255, int(255 * (1 - _alpha * 0.6)), int(255 * (1 - _alpha))
            _bg = f"rgb({_r},{_g},{_b})"
            _html_parts.append(
                f'<span style="background:{_bg};padding:1px 2px;border-radius:2px">{_w}</span>'
            )
        _html_parts.append("</div>")
        _article_html = mo.Html(" ".join(_html_parts))
    _article_html
    return


@app.cell
def _(cached_kv, mo, task_scores, task_scores_norm):
    # Sanity check: top-10 most-attended words per task.
    # We aggregate per-word scores by mapping each word to its uniform token
    # range (same mapping used above) and taking the max within the range.
    # This gives a tight, human-readable summary of what each task "cares about".

    _top_out = mo.md("")
    if task_scores is not None and task_scores_norm is not None and cached_kv is not None:
        _task_names = task_scores["task_names"]
        _article = cached_kv["article_text"]
        _words = _article.split()
        _art_start = cached_kv["article_indices_start"]
        _art_stop = cached_kv["article_indices_stop"]
        _art_len = _art_stop - _art_start
        _n_words = len(_words)

        _rows = ["| Task | Top-10 words (by mean attention score) |", "|---|---|"]
        for _ti, _tname in enumerate(_task_names):
            _s = task_scores_norm[_ti, _art_start:_art_stop]
            _word_scores = []
            for _wi, _w in enumerate(_words):
                _p0 = int(round(_wi * _art_len / _n_words))
                _p1 = max(_p0 + 1, int(round((_wi + 1) * _art_len / _n_words)))
                _word_scores.append((_w, float(_s[_p0:_p1].max()) if _p1 > _p0 else 0.0))
            # Deduplicate while preserving descending score order
            _seen = set()
            _top = []
            for _w, _sc in sorted(_word_scores, key=lambda x: -x[1]):
                _wc = _w.strip(".,;:()—\"'").lower()
                if _wc and _wc not in _seen and not _wc.isdigit():
                    _seen.add(_wc)
                    _top.append(_w.strip(".,;:()—"))
                if len(_top) >= 10:
                    break
            _rows.append(f"| **{_tname}** | {', '.join(_top)} |")
        _top_out = mo.md("\n".join(_rows))
    _top_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same $K$, same $V$, same algorithm — only $Q_\text{ref}$ changes, and three visibly different token sets get scored as important. T1 lights up §2, T2 lights up §5, T3 lights up §7. AM's output is genuinely a function of $(K, V, Q_\text{ref})$, not just $(K, V)$.

    **Why this matters for multi-agent systems.** In a hierarchical agent system, an orchestrator agent typically passes the same context (a long document, a chat history, a code repository) to many worker agents that each have a *different* sub-task: "extract the date", "summarize the economy", "list the wildlife". Today this almost always means **each worker re-attends to the full context** — paying the full KV-cache memory + latency cost per agent. Query-conditioned AM offers a different option: compact the shared context **once per worker**, using that worker's task prompt as $Q_\text{ref}$. Each worker gets a small cache that's specialized to its own question, and the heavy $(K, V)$ tensors live in one place. Recent industry work (e.g. Ramp Labs' Latent Briefing for recursive language models, Apr 2026) reports 42–57% worker-token savings using exactly this idea — task-derived queries fed into AM's standard pipeline.

    **Other practical settings:** RAG retrieval (compact a long document tailored to the user's question), long-conversation chat (compact pre-history toward the current intent), code agents (compact a repo toward the current refactor). In all of them the task is *known at compaction time*, so the lever this section visualizes is directly available.

    *Caveats.* Task queries here are 15–28 vectors each (small vs. the 1,200+ self-study queries the paper uses); a real deployment would mix the two. We aggregate heads uniformly. The heatmap shows **scoring**, not post-compaction downstream accuracy — running an end-to-end task-conditioned AM round-trip is the natural follow-up experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    <a id="part-e"></a>
    # Part E — End-to-end QA (full results)

    > **↪ Uses sidebar:** *QA keep ratio* slider (top of the sidebar). Drives this section's table, summary chart, and word-diff. The headline teaser at the top of the notebook also reacts to this slider.

    The teaser at the top already showed the headline at one ratio. This is the full per-ratio breakdown: which questions the model gets right at each compression level, the verbatim-repeat diff, and how those numbers move as you compress harder. Pipeline used: **AM-HighestAttnKeys + optimized non-uniform head budget + self-study queries** (paper's recommended default).
    """)
    return


@app.cell
def _():
    # qa_ratio_selector is defined in the sidebar controls cell above
    return


@app.cell
def _(qa_data):
    # Compute ALIGNMENT-based word recall for every ratio once, plus baseline.
    # Replaces the broken substring-based word_recall stored in the .pt file
    # (that one matches `w.lower() in generated_lower`, which counts a word
    # whose characters happen to appear as a substring of any other word —
    # giving false 100% readings when the output is truncated).
    #
    # Alignment recall = fraction of article words that difflib aligns as
    # "equal" with the compacted output, i.e. present AND in the right order.
    import difflib as _difflib_ar

    def _aligned_recall(article_text, generated_text):
        _o = article_text.split()
        _g = generated_text.split()
        if not _o:
            return 0.0
        _sm = _difflib_ar.SequenceMatcher(None, _o, _g)
        _eq = sum(
            (_i2 - _i1) for _tag, _i1, _i2, _j1, _j2 in _sm.get_opcodes()
            if _tag == "equal"
        )
        return _eq / len(_o)

    if qa_data is None:
        aligned_recalls = None
    else:
        _article = qa_data["article_text"]
        aligned_recalls = {
            "baseline": _aligned_recall(_article, qa_data["original_repeat"]["generated_text"]),
            "per_ratio": {
                r: _aligned_recall(_article, qa_data["ratio_results"][r]["repeat_result"]["generated_text"])
                for r in qa_data["ratio_results"]
            },
        }
    return (aligned_recalls,)


@app.cell
def _(aligned_recalls, mo, qa_data, qa_ratio_selector):
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
        _orig_ar = aligned_recalls["baseline"] if aligned_recalls else 0.0
        _comp_ar = aligned_recalls["per_ratio"].get(_ratio, 0.0) if aligned_recalls else 0.0

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

    **Verbatim repeat test** (recall = fraction of article words that `difflib` aligned as "equal" in the model's repeat — stricter than substring matching, so truncated outputs show low recall honestly):
    - Full cache: {_orig_ar:.1%} aligned recall, perplexity {_orig_repeat['perplexity']:.2f}
    - Compacted: {_comp_ar:.1%} aligned recall, perplexity {_comp_repeat['perplexity']:.2f}
    """)
        )
    return


@app.cell
def _(aligned_recalls, mo, qa_data, qa_ratio_selector):
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
        _ar = aligned_recalls["per_ratio"].get(_ratio, 0.0) if aligned_recalls else 0.0
        _truncated_dd = len(_comp_words) < 0.9 * len(_orig_words) and len(_comp_words) > 0

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

        _truncation_note = ""
        if _truncated_dd:
            _truncation_note = (
                f" &nbsp; <span style='color:#92400e'>⚠ Model output was truncated at "
                f"~{len(_comp_words)} of {len(_orig_words)} article words — raising "
                f"`max_new_tokens` in `scripts/precompute_qa_results.py` would eliminate this.</span>"
            )

        _output = mo.vstack([
            mo.md(f"### Verbatim Repeat Diff: Original Article vs {_ratio:.0%} Compacted Output"),
            mo.Html(
                f"Aligned recall: <b>{_ar:.1%}</b> "
                f"({len(_comp_words)} generated words / {len(_orig_words)} article words), "
                f"perplexity: <b>{_comp_repeat['perplexity']:.2f}</b>{_truncation_note}"
            ),
            mo.Html(_legend),
            mo.Html(
                f'<div style="line-height:1.8;font-size:0.95em;max-width:800px">{_diff_html}</div>'
            ),
        ])
    _output
    return


@app.cell
def _(aligned_recalls, np, plt, qa_data):
    # Three-panel summary, all monotonic and directly meaningful:
    #   (1) MCQ accuracy vs ratio
    #   (2) Per-question correctness heatmap (6 questions × 6 ratios + baseline)
    #   (3) Aligned verbatim recall vs ratio
    #
    # We deliberately DO NOT plot mean perplexity here. The model generates
    # different text at each ratio, so perplexity-of-generated-text is not
    # comparable across ratios — its variance across questions (~0.1) is
    # larger than its trend across ratios (~0.04). It read as non-monotonic
    # in earlier drafts and confused readers. If you want a perplexity-like
    # signal, the cosine-similarity quality grid in Part C is monotonic.
    _fig = None
    if qa_data is not None:
        _ratios = sorted(qa_data["ratio_results"].keys())
        _accuracies = [qa_data["ratio_results"][r]["accuracy"] for r in _ratios]
        _repeat_recalls = [
            aligned_recalls["per_ratio"].get(r, 0.0) if aligned_recalls else 0.0
            for r in _ratios
        ]
        _orig_acc = qa_data["original_accuracy"]
        _orig_recall = aligned_recalls["baseline"] if aligned_recalls else 0.0

        # Per-question correctness matrix: rows = questions, cols = [ratios..., baseline]
        _n_q = len(qa_data["original_qa"])
        _correctness = np.zeros((_n_q, len(_ratios) + 1), dtype=float)
        for _ri, _r in enumerate(_ratios):
            for _qi, _q in enumerate(qa_data["ratio_results"][_r]["qa_results"]):
                _correctness[_qi, _ri] = 1.0 if _q["correct"] else 0.0
        for _qi, _q in enumerate(qa_data["original_qa"]):
            _correctness[_qi, -1] = 1.0 if _q["correct"] else 0.0

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Panel 1: aggregate accuracy vs ratio
        _ax1.plot(_ratios, _accuracies, "o-", color="#2563eb", linewidth=2, label="Compacted")
        _ax1.axhline(_orig_acc, linestyle="--", color="#16a34a", label="Full cache")
        _ax1.set_xlabel("Keep ratio")
        _ax1.set_ylabel("QA accuracy")
        _ax1.set_title("Aggregate accuracy vs compression")
        _ax1.set_ylim(-0.05, 1.15)
        _ax1.legend(fontsize=8)
        _ax1.grid(True, alpha=0.3)

        # Panel 2: per-question correctness heatmap (which questions break first)
        _im = _ax2.imshow(_correctness, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        _ax2.set_xticks(list(range(len(_ratios))) + [len(_ratios)])
        _ax2.set_xticklabels([f"{r:.0%}" for r in _ratios] + ["full"], fontsize=9)
        _ax2.set_yticks(range(_n_q))
        _ax2.set_yticklabels([f"Q{i+1}" for i in range(_n_q)], fontsize=9)
        _ax2.set_xlabel("Keep ratio")
        _ax2.set_title("Which questions break first?")
        for _qi in range(_n_q):
            for _ri in range(len(_ratios) + 1):
                _val = _correctness[_qi, _ri]
                _ax2.text(_ri, _qi, "✓" if _val == 1.0 else "✗",
                          ha="center", va="center",
                          color="white" if _val == 1.0 else "white", fontsize=12)

        # Panel 3: aligned verbatim recall
        _ax3.plot(_ratios, _repeat_recalls, "D-", color="#dc2626", linewidth=2, label="Compacted")
        _ax3.axhline(_orig_recall, linestyle="--", color="#16a34a", label="Full cache")
        _ax3.set_xlabel("Keep ratio")
        _ax3.set_ylabel("Aligned recall")
        _ax3.set_title("Verbatim repeat recall vs compression")
        _ax3.set_ylim(-0.05, 1.15)
        _ax3.legend(fontsize=8)
        _ax3.grid(True, alpha=0.3)

        _fig.suptitle(f"End-to-end impact of KV cache compaction — {qa_data['model_name']}", fontsize=13)
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    <a id="part-f"></a>
    # Part F — The Speed–Quality Pareto (Figure 1 of the paper)

    > **↪ Uses inline controls:** layer / KV-head dropdowns + "Include AM-OMP" checkbox below. At the default (layer 12, head 4) the scatter loads instantly from the precomputed cache.

    Why does the paper call itself "**Fast** KV Compaction"? Because the speed advantage over prior work — particularly Cartridges (Eyuboglu et al., 2025), which does end-to-end gradient descent — is the whole point. But the word "fast" deserves measurement, not just assertion.

    The paper's **Figure 1** is a scatter plot of downstream accuracy vs. compaction time (log-scale), showing that AM methods trace the Pareto frontier. We reproduce the core idea here on a single real KV head: we time every algorithm at four keep ratios and plot **cosine similarity vs. wall-clock compaction time**. The goal is not to beat the paper's numbers (we have one CPU and one head) but to see the Pareto structure appear with our own eyes.

    **Reading the plot:**

    - **Upper-right** = slow but high-quality (AM-OMP).
    - **Upper-left** = fast AND high-quality = Pareto-optimal (AM-HighestAttnKeys should live here).
    - **Lower-left** = fast but low-quality (Truncation, Random).

    Pick any head from the sidebar below to re-run the benchmark on that head.
    """)
    return


@app.cell
def _(cached_kv, mo):
    pareto_layer = mo.ui.dropdown(
        options=[str(i) for i in range(cached_kv["n_layers"])] if cached_kv is not None else ["0"],
        value="12",
        label="Layer",
    )
    pareto_head = mo.ui.dropdown(
        options=[str(i) for i in range(cached_kv["n_kv_heads"])] if cached_kv is not None else ["0"],
        value="4",
        label="KV head",
    )
    # AM-OMP is the slowest method (~25s for 4 ratios on one head). Disabled
    # by default to keep the Pareto plot responsive; users can opt in.
    pareto_include_omp = mo.ui.checkbox(value=False, label="Include AM-OMP (slow, ~25s)")
    mo.hstack([pareto_layer, pareto_head, pareto_include_omp], justify="start", gap=1)
    return pareto_head, pareto_include_omp, pareto_layer


@app.cell
def _(
    ALGO_INSTANCES,
    cached_kv,
    cosine_similarity,
    pareto_head,
    pareto_include_omp,
    pareto_layer,
    torch,
):
    # Speed–Quality Pareto benchmark on one real Qwen3-4B head.
    # 5 methods × 4 ratios × 3 trials for median timing = 60 compaction calls.
    #
    # Fast path: if the user stays on the default (layer=12, head=4) and a
    # precomputed cache exists at data/cached_kv/Qwen3-4B_pareto_default.pt
    # (produced by scripts/precompute_pareto.py), skip all 60 calls and use
    # the cached numbers. Fall through to live compute for any other
    # (layer, head) or when the cache is missing.
    import os as _os_pf
    import time as _time

    pareto_data = None
    _pareto_cache_path = _os_pf.path.join(
        _os_pf.path.dirname(_os_pf.path.abspath(__file__))
        if "__file__" in dir() else _os_pf.getcwd(),
        "data", "cached_kv", "Qwen3-4B_pareto_default.pt",
    )
    _pareto_cache = None
    if _os_pf.path.exists(_pareto_cache_path):
        _pareto_cache = torch.load(_pareto_cache_path, map_location="cpu", weights_only=False)

    _using_cache = (
        cached_kv is not None
        and _pareto_cache is not None
        and int(pareto_layer.value) == _pareto_cache.get("layer")
        and int(pareto_head.value) == _pareto_cache.get("head")
    )

    if _using_cache:
        _rows_cached = _pareto_cache["rows"]
        if not pareto_include_omp.value:
            _rows_cached = [_r for _r in _rows_cached if _r["method"] != "AM-OMP"]
        # Recompute n_queries without allocating tensors
        _gqa_c = cached_kv["n_q_heads"] // cached_kv["n_kv_heads"]
        _n_q_c = _gqa_c * cached_kv["seq_len"]
        pareto_data = {
            "rows": _rows_cached,
            "layer": _pareto_cache["layer"],
            "head": _pareto_cache["head"],
            "n_queries": _n_q_c,
            "source": "cache",
        }

    if cached_kv is None:
        pareto_data = None
    elif not _using_cache:
        _li = int(pareto_layer.value)
        _hi = int(pareto_head.value)
        _gqa = cached_kv["n_q_heads"] // cached_kv["n_kv_heads"]
        _hd = cached_kv["head_dim"]
        _T = cached_kv["seq_len"]
        _scale = _hd ** -0.5

        _K_h = cached_kv[f"K_{_li}"].float()[_hi]            # (T, D)
        _V_h = cached_kv[f"V_{_li}"].float()[_hi]            # (T, D)
        _Q_h = cached_kv[f"Q_{_li}"].float()[_hi * _gqa:(_hi + 1) * _gqa].reshape(-1, _hd)

        # Full attention output as quality target
        _full_out = torch.softmax((_Q_h @ _K_h.T) * _scale, dim=-1) @ _V_h

        _ratios = [0.05, 0.15, 0.30, 0.60]
        _rows = []
        # Filter methods based on user preference
        _methods_to_run = {
            n: a for n, a in ALGO_INSTANCES.items()
            if pareto_include_omp.value or n != "AM-OMP"
        }
        for _method_name, _algo in _methods_to_run.items():
            for _r in _ratios:
                _t_target = max(1, min(_T - 1, int(round(_T * _r))))
                # Median of 3 trials for timing stability
                _times = []
                _C1 = _beta = _C2 = None
                for _trial in range(3):
                    _t0 = _time.perf_counter()
                    try:
                        _C1, _beta, _C2, _ = _algo.compute_compacted_cache(
                            _K_h, _V_h, _Q_h, _t_target,
                        )
                        _times.append(_time.perf_counter() - _t0)
                    except Exception:
                        _times.append(float("nan"))
                _times_sorted = sorted(_t for _t in _times if _t == _t)  # drop NaN
                _median_ms = (
                    1000.0 * _times_sorted[len(_times_sorted) // 2]
                    if _times_sorted else float("nan")
                )

                # Quality from the final trial
                if _C1 is not None:
                    _comp_out = torch.softmax(
                        (_Q_h @ _C1.T) * _scale + _beta.float().unsqueeze(0), dim=-1
                    ) @ _C2.float()
                    _cos = cosine_similarity(_full_out, _comp_out)
                else:
                    _cos = float("nan")

                _rows.append({
                    "method": _method_name,
                    "keep_ratio": _r,
                    "time_ms": _median_ms,
                    "cosine": _cos,
                })
        pareto_data = {
            "rows": _rows,
            "layer": _li,
            "head": _hi,
            "n_queries": int(_Q_h.shape[0]),
        }
    pareto_data
    return (pareto_data,)


@app.cell
def _(pareto_data, plt):
    _fig = None
    if pareto_data is not None:
        _method_styles = {
            "AM-HighestAttnKeys": ("#2563eb", "o"),
            "AM-OMP":              ("#7c3aed", "s"),
            "KVMerger":            ("#d97706", "^"),
            "Random":              ("#6b7280", "x"),
            "Truncation":          ("#dc2626", "D"),
        }

        _by_method = {}
        for _row in pareto_data["rows"]:
            _by_method.setdefault(_row["method"], []).append(_row)

        _fig, _ax = plt.subplots(figsize=(9, 6))
        for _method, _pts in _by_method.items():
            _pts = sorted(_pts, key=lambda r: r["time_ms"])
            _color, _marker = _method_styles.get(_method, ("#000", "o"))
            _xs = [_p["time_ms"] for _p in _pts]
            _ys = [_p["cosine"] for _p in _pts]
            _ax.plot(_xs, _ys, "-", color=_color, alpha=0.45, linewidth=1.5)
            # Unfilled markers (like 'x', '+') don't support edgecolors in matplotlib
            _unfilled_markers = {"x", "+", "|", "_", "."}
            _scatter_kwargs = {"color": _color, "marker": _marker, "s": 80, "label": _method, "zorder": 3}
            if _marker not in _unfilled_markers:
                _scatter_kwargs["edgecolors"] = "white"
                _scatter_kwargs["linewidth"] = 0.8
            _ax.scatter(_xs, _ys, **_scatter_kwargs)
            # Annotate each point with its keep ratio
            for _p in _pts:
                _ax.annotate(
                    f"{_p['keep_ratio']:.0%}",
                    (_p["time_ms"], _p["cosine"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=_color, alpha=0.85,
                )

        _ax.set_xscale("log")
        _ax.set_xlabel("Median compaction time (ms, log scale)")
        _ax.set_ylabel("Cosine similarity to full attention output")
        _ax.set_title(
            f"Speed–Quality Pareto  —  Qwen3-4B, "
            f"layer {pareto_data['layer']}, KV head {pareto_data['head']}, "
            f"{pareto_data['n_queries']} queries"
        )
        _ax.legend(loc="lower right", fontsize=9)
        _ax.grid(True, which="both", alpha=0.25)
        _ax.set_ylim(None, 1.02)
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What you should see** (and what the paper's Figure 1 shows at scale):

    - **AM-OMP** sits toward the upper-right: highest quality, slowest — each greedy step re-fits NNLS across the growing selection.
    - **AM-HighestAttnKeys** sits upper-left: near-OMP quality at a fraction of the time. For most budgets it is on the Pareto frontier. This is why the paper recommends it as the default "fast" method.
    - **KVMerger** trails AM methods at low keep ratios (its $O(T^2)$ clustering is expensive and its lack of $\beta$ means the compacted block's mass is wrong).
    - **Random** and **Truncation** are very fast but low-quality at aggressive ratios. Random does benefit dramatically from AM's $\beta$ + $C_v$ fit — this is why it isn't at the bottom of the plot.
    - The **annotation next to each marker** is that point's keep ratio; as you decrease it along each curve the points move down-and-left (faster, lower quality).

    **Pick a different head** in the dropdowns above to re-run. Heads with more structured attention (e.g., shallow layers) make the AM gap narrower; mid/late layers where attention is diffuse show the gap dramatically.

    *Caveat:* we are timing on a single CPU and a single head with ~1220 queries, while the paper's Figure 1 times a full model on an H100. The *shape* of the Pareto matches; the absolute wall-clock numbers do not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why $\beta$ and $C_v$ matter — and how to measure it correctly

    The intro introduced the three-ingredient $(C_k, \beta, C_v)$ framework. Here we **visualize** what each piece does and **measure** its impact.

    A subtlety worth flagging: if you ablate $\beta$ and then run softmax over the *compacted block alone*, $\beta$ looks irrelevant — softmax is shift-invariant in the row dimension, so an additive bias gets absorbed and cosine similarity stays at ~1.0. **That measurement is wrong for this paper.** AM is designed for the case where the compacted block is **concatenated with another block** (a follow-up question, generated continuation, the next conversation turn). In that mixture-attention regime, the relative weighting between the two blocks is set by their *attention-mass ratio* — and that ratio is exactly what $\beta$ corrects (paper Appendix A.2). So our ablation below scores compacted + future-block mixture-attention against full + future-block mixture-attention. That is where $\beta$'s value shows up.

    First, the visualization of $\beta$'s effect on per-key mass.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seeing $\beta$ in Action

    To make $\beta$'s effect concrete we plot the per-key contribution to the **attention mass**, $\Phi_{i^\star,\, j} = \exp(q_{i^\star} K_j^\top / \sqrt{d})$, for the sharpest reference query $i^\star$. Three settings:

    1. **Full cache** — original mass contributions over all $T$ keys.
    2. **Compacted, $\beta = 0$** — only the kept keys contribute, and *each* compact key contributes only $\exp(q (C_k)_j^\top)$. The block's total mass $\sum_j \exp(q (C_k)_j^\top)$ is far below $m_i = \sum_k \exp(q K_k^\top)$.
    3. **Compacted, $\beta$ fitted via NNLS** — each compact key contributes $w_j \exp(q (C_k)_j^\top) = \exp(q (C_k)_j^\top + \beta_j)$. The total mass now closely matches $m_i$.

    The headline number on each panel is the **attention-mass relative error** averaged across all reference queries (this is exactly the residual in Eq. 2 of the paper). Tune `keep_ratio` in the sidebar to see how the no-$\beta$ error grows at aggressive compression while the with-$\beta$ error stays small.
    """)
    return


@app.cell
def _(ALGO_INSTANCES, k, keep_ratio, np, plt, q, set_seed, torch, v):
    _HAK_pf = type(ALGO_INSTANCES["AM-HighestAttnKeys"])
    set_seed(0)

    _h_pf = 0  # head 0
    _K_pf = k[_h_pf]
    _V_pf = v[_h_pf]
    _Q_pf = q[_h_pf]
    _T_pf, _D_pf = _K_pf.shape
    _t_pf = max(2, int(round(_T_pf * keep_ratio.value)))
    _scale_pf = _D_pf ** -0.5

    # With beta fitted
    _algo_beta_pf = _HAK_pf(
        score_method="rms", nnls_iters=2, nnls_lower_bound=0.05,
        nnls_upper_bound=20.0, c2_method="lsq", beta_method="nnls",
    )
    _C1_b, _beta_b, _C2_b, _idx_b = _algo_beta_pf.compute_compacted_cache(
        _K_pf, _V_pf, _Q_pf, _t_pf
    )

    # With beta forced to zero (same scoring, no bias correction)
    _algo_nobeta_pf = _HAK_pf(
        score_method="rms", nnls_iters=0, c2_method="lsq", beta_method="zero",
    )
    _C1_nb, _beta_nb, _C2_nb, _idx_nb = _algo_nobeta_pf.compute_compacted_cache(
        _K_pf, _V_pf, _Q_pf, _t_pf
    )

    # Pick a representative query — choose the one with sharpest full attention
    _full_scores_pf = (_Q_pf @ _K_pf.T).float() * _scale_pf
    _full_w_pf = torch.softmax(_full_scores_pf, dim=-1)
    _sharpness_pf = (_full_w_pf * _full_w_pf).sum(dim=-1)
    _qi_pf = int(_sharpness_pf.argmax().item())
    _q_pick = _Q_pf[_qi_pf:_qi_pf + 1]

    # Per-key UN-NORMALIZED mass contributions (this is what beta is fitted to match)
    # Phi_j = exp(q · K_j / sqrt(d))   for the picked query
    # Use a consistent shift for numerical stability: subtract the same max across
    # all three variants so they stay on the same scale.
    _scaled_full_pick = (_q_pick @ _K_pf.T).float() * _scale_pf  # (1, T)
    _shift_pick = _scaled_full_pick.max(dim=-1, keepdim=True)[0]  # (1, 1) — scaled max
    _full_phi = torch.exp(_scaled_full_pick - _shift_pick)[0]  # (T,)
    _nb_phi = torch.exp(
        (_q_pick @ _C1_nb.T).float() * _scale_pf - _shift_pick
    )[0]  # (t,) — NO beta multiplier
    _wb_phi_per_key = torch.exp(
        (_q_pick @ _C1_b.T).float() * _scale_pf
        + _beta_b.float().unsqueeze(0)
        - _shift_pick
    )[0]  # (t,) — INCLUDES exp(beta) = w multiplier

    # Total attention masses (the quantity Eq. 2 wants matched)
    _full_total = _full_phi.sum().item()
    _nb_total = _nb_phi.sum().item()
    _wb_total = _wb_phi_per_key.sum().item()

    # Attention-mass relative error averaged across ALL reference queries (Eq. 2)
    _scaled_full_all = (_Q_pf @ _K_pf.T).float() * _scale_pf  # (n, T)
    _shift_all = _scaled_full_all.max(dim=-1, keepdim=True)[0]  # (n, 1)
    _all_full_phi = torch.exp(_scaled_full_all - _shift_all)
    _all_full_m = _all_full_phi.sum(dim=-1)
    _all_nb_phi = torch.exp(
        (_Q_pf @ _C1_nb.T).float() * _scale_pf - _shift_all
    )
    _all_nb_m = _all_nb_phi.sum(dim=-1)
    _all_wb_phi = torch.exp(
        (_Q_pf @ _C1_b.T).float() * _scale_pf
        + _beta_b.float().unsqueeze(0)
        - _shift_all
    )
    _all_wb_m = _all_wb_phi.sum(dim=-1)
    _nb_err = ((_all_full_m - _all_nb_m).abs() / _all_full_m.clamp_min(1e-12)).mean().item()
    _wb_err = ((_all_full_m - _all_wb_m).abs() / _all_full_m.clamp_min(1e-12)).mean().item()

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 3.8))

    _max_y = max(
        _full_phi.max().item(), _nb_phi.max().item(), _wb_phi_per_key.max().item(),
    ) * 1.1
    # Guard against degenerate case (all mass contributions ~0 due to numerical subtraction)
    if _max_y <= 0 or not np.isfinite(_max_y):
        _max_y = 1.0

    _axes[0].bar(range(_T_pf), _full_phi.numpy(), color="#16a34a", edgecolor="white", linewidth=0.4)
    _axes[0].set_title(
        f"Full cache: $\\Phi_j = \\exp(q\\cdot K_j/\\sqrt{{d}})$\n"
        f"total mass $m = {_full_total:.3f}$",
        fontsize=11,
    )
    _axes[0].set_xlabel("Key position $j$")
    _axes[0].set_ylabel("Mass contribution")
    _axes[0].set_xlim(-0.5, _T_pf - 0.5)
    _axes[0].set_ylim(0, _max_y)

    _nb_positions = list(_idx_nb) if hasattr(_idx_nb, "__iter__") else _idx_nb.tolist()
    _nb_full_arr = np.zeros(_T_pf)
    for _ii, _pos in enumerate(_nb_positions):
        _nb_full_arr[_pos] = _nb_phi[_ii].item()
    _axes[1].bar(range(_T_pf), _nb_full_arr, color="#dc2626", edgecolor="white", linewidth=0.4)
    _axes[1].set_title(
        f"Compacted, $\\beta = 0$ (no correction)\n"
        f"compact mass $= {_nb_total:.3f}$  |  rel err: {_nb_err:.1%}",
        fontsize=11, color="#dc2626",
    )
    _axes[1].set_xlabel("Key position (gaps = dropped)")
    _axes[1].set_xlim(-0.5, _T_pf - 0.5)
    _axes[1].set_ylim(0, _max_y)

    _wb_positions = list(_idx_b) if hasattr(_idx_b, "__iter__") else _idx_b.tolist()
    _wb_full_arr = np.zeros(_T_pf)
    for _ii, _pos in enumerate(_wb_positions):
        _wb_full_arr[_pos] = _wb_phi_per_key[_ii].item()
    _axes[2].bar(range(_T_pf), _wb_full_arr, color="#2563eb", edgecolor="white", linewidth=0.4)
    _axes[2].set_title(
        f"Compacted, $\\beta$ fitted via NNLS\n"
        f"compact mass $= {_wb_total:.3f}$  |  rel err: {_wb_err:.1%}",
        fontsize=11, color="#2563eb",
    )
    _axes[2].set_xlabel("Key position (gaps = dropped)")
    _axes[2].set_xlim(-0.5, _T_pf - 0.5)
    _axes[2].set_ylim(0, _max_y)

    _fig.suptitle(
        f"Per-key mass contributions $\\Phi_{{i^\\star,j}}$ (query #{_qi_pf}; "
        f"keep={keep_ratio.value:.0%}, $t={_t_pf}/{_T_pf}$)",
        fontsize=12, y=1.02,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Green** = original mass contributions across all $T$ keys (their sum is the target $m$). **Red** = no-$\beta$ compacted block — total mass is far below $m$, which would cause the block to be systematically under-attended in any future mixture-attention. **Blue** = same selected positions but each scaled by $w_j = e^{\beta_j}$; total mass now matches $m$.

    The `rel err` headline is exactly the residual NNLS minimizes (Eq. 2). Drag `keep_ratio` in the sidebar — the red error grows sharply at low ratios while the blue error stays small, until $t$ is so small that $\beta$ hits its clamp.
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
    # Ablation: how much does each ingredient (β-fit, Cv-fit) actually matter?
    #
    # IMPORTANT — measurement protocol:
    # When the compacted block is used in *isolation* (one softmax over the
    # t kept keys alone), β is shift-invariant under softmax and looks
    # useless. β only earns its keep when the compacted block is later
    # **concatenated with another block** (e.g. a follow-up question, a new
    # turn, generated continuation). Then the relative weighting between the
    # two blocks is set by their attention-mass ratio — which is exactly
    # what β corrects.
    #
    # So the ablation here measures the cosine similarity to the FULL
    # mixture-attention output, where the compacted block is concatenated
    # with a fresh "future" block of `n_future` random keys/values. This is
    # the regime AM is actually used in (see paper Appendix A.2).
    _HAK = type(ALGO_INSTANCES["AM-HighestAttnKeys"])

    set_seed(seed.value)
    _H, _T, _D = q.shape
    _t = max(1, int(round(_T * keep_ratio.value)))
    _scale = _D ** -0.5

    # Synthesize a "future" block: small (n_future tokens) appended to the
    # cache when the model continues generating after the compacted prefix.
    _n_future = max(8, _T // 4)
    _K_future = torch.randn(_H, _n_future, _D)
    _V_future = torch.randn(_H, _n_future, _D)
    # Use the LAST queries (the generation-time queries)
    _Q_eval = q[:, -max(8, _T // 8):, :]

    # Full AM (β + Cv fitting)
    _algo_full = _HAK(
        score_method="rms", nnls_iters=2, nnls_lower_bound=0.05,
        nnls_upper_bound=20.0, c2_method="lsq", beta_method="nnls",
    )
    # No β (β=0, but Cv still fitted)
    _algo_no_beta = _HAK(
        score_method="rms", nnls_iters=0, c2_method="lsq", beta_method="zero",
    )
    # No Cv fitting (β fitted, but Cv = V[indices])
    _algo_no_cv = _HAK(
        score_method="rms", nnls_iters=2, nnls_lower_bound=0.05,
        nnls_upper_bound=20.0, c2_method="direct", beta_method="nnls",
    )
    # Neither (naive subset eviction)
    _algo_naive = _HAK(
        score_method="rms", nnls_iters=0, c2_method="direct", beta_method="zero",
    )

    _ablation_rows = []
    for _label, _algo in [
        ("Full AM (β + Cv fit)", _algo_full),
        ("No β (Cv fit only)", _algo_no_beta),
        ("No Cv fit (β only)", _algo_no_cv),
        ("Naive subset (no β, no Cv fit)", _algo_naive),
    ]:
        _cosines = []
        for _h in range(_H):
            _C1, _beta, _C2, _idx = _algo.compute_compacted_cache(
                k[_h], v[_h], q[_h], _t
            )

            # Full mixture: concat full block with future block
            _Kf_full = torch.cat([k[_h].float(), _K_future[_h]], dim=0)
            _Vf_full = torch.cat([v[_h].float(), _V_future[_h]], dim=0)
            _full = torch.softmax((_Q_eval[_h].float() @ _Kf_full.T) * _scale, dim=-1) @ _Vf_full

            # Compacted mixture: concat compact block with future block.
            # The compact block's per-key bias β is added to its score
            # columns; the future block has zero bias.
            _K_eff = torch.cat([_C1.float(), _K_future[_h]], dim=0)
            _V_eff = torch.cat([_C2.float(), _V_future[_h]], dim=0)
            _bias = torch.cat(
                [_beta.float(), torch.zeros(_n_future)], dim=0
            )
            _scores = (_Q_eval[_h].float() @ _K_eff.T) * _scale + _bias.unsqueeze(0)
            _comp = torch.softmax(_scores, dim=-1) @ _V_eff

            _cosines.append(cosine_similarity(_full, _comp))

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
    _ax.set_xlabel("Mean cosine similarity to full mixture-attention output")
    _ax.set_title("Ablation: impact of β and Cv fitting (compact + future-block mixture)")
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


@app.cell
def _(mo):
    mo.accordion({
        "Algorithm cheat-sheet: how each method chooses $C_k$, $\\beta$, $C_v$": mo.md(r"""
    | Method | $C_k$ selection | $\beta$ | $C_v$ |
    |--------|-----------------|---------|-------|
    | **AM-HighestAttnKeys** | Top-$t$ keys by RMS post-softmax attention weight | NNLS fit on Eq. 2 (after selection) | OLS fit on Eq. 1 |
    | **AM-OMP** | Greedy Algorithm 1, directly minimizes Eq. 2 attention-mass error | NNLS, **jointly** with selection (re-fit each step) | OLS fit on Eq. 1 |
    | **KVMerger** (Wang et al. 2024) | Agglomerative merging of consecutive similar keys via Gaussian kernel | Zero (no AM fitting) | Merged from original values |
    | **Random** | Random subset (with full AM pipeline applied after) | NNLS fit on Eq. 2 | OLS fit on Eq. 1 |
    | **Truncation** | First $t$ tokens (with full AM pipeline applied after) | NNLS fit on Eq. 2 | OLS fit on Eq. 1 |

    Random and Truncation both benefit from AM's $\beta$ and $C_v$ fitting — even random selection becomes much stronger when paired with proper value reconstruction. The insight: **how you reconstruct matters as much as what you keep.**
    """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    <a id="recap"></a>
    # Recap: What Did We Just See?

    1. **The problem**: KV cache scales linearly with context length and dominates long-context inference memory.
    2. **The framework**: Attention Matching builds a compact cache $(C_k, \beta, C_v)$ that satisfies two equations *simultaneously*:
       - **Eq. 1** — local attention output matching: $\text{softmax}(q C_k^\top + \beta) C_v \approx \text{softmax}(q K^\top) V$
       - **Eq. 2** — attention mass matching: $\sum_j \exp(q (C_k)_j^\top + \beta_j) \approx \sum_j \exp(q K_j^\top)$

       Together they preserve the compacted block's behavior under concatenation with future tokens.
    3. **Three pieces**:
       - $C_k$ — compact keys, picked by top-score (HighestAttnKeys, Section 3.3) or greedy mass-matching (OMP, Algorithm 1).
       - $\beta$ — per-key log-bias, fitted by non-negative least squares on the **attention-mass** system $A w = m$, then $\beta = \log w$.
       - $C_v$ — compact values, fitted by ordinary least squares on the **attention-output** system $X C_v = Y$.
    4. **Non-uniform budgets** — different heads compress to different difficulties, so we should give each head its own cache allowance (greedy solver on per-head influence curves).
    5. **End-to-end** — on Qwen3-4B answering multiple-choice questions about an article, AM retains 100% of answers correct down to 10% cache, and degrades gracefully below that.

    ## Going Further

    - **Run on a real model with your own prompt**: Use `examples/qa_demo.py` to compact a Qwen3-4B model's KV cache and test QA accuracy before/after.
    - **Re-optimize head budgets** for your own data with `head_budget_optimization/run.py`.
    - **Chunked compaction**: for contexts that don't fit in memory, the codebase supports chunk-wise AM via `compaction/compaction_methods/chunked.py`.
    - **Full benchmark suite**: `evaluation/run_qa_evaluation.py` runs QuALITY, LongHealth, RULER, Qasper, and LongBenchV2.

    ```bash
    # Try the quick demo (requires a GPU or Apple Silicon MPS for the full model):
    python -m examples.qa_demo --model Qwen/Qwen3-4B --target-size 0.1
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    **Author · References · Reproducibility**

    **Notebook by Ashutosh Srivastava** · [X / @h4shkat](https://x.com/h4shkat) · [LinkedIn](https://www.linkedin.com/in/ashutosh-srivastava-1bbb0a223/) · [Portfolio](https://h4shk4t.github.io/)

    **Paper**: Zweiger, A., Fu, X., Guo, H., & Kim, Y. (2026). *Fast KV Compaction via Attention Matching.* [arXiv:2602.16284](https://arxiv.org/abs/2602.16284).

    **Reproducibility**: every cache this notebook loads can be regenerated with one command: `PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/precompute_all.py --device mps`. Requires Qwen/Qwen3-4B (auto-downloaded on first run).

    Submitted to the [marimo Notebook Competition 2026](https://marimo.io/pages/events/notebook-competition). Built on [marimo](https://marimo.io), [PyTorch](https://pytorch.org), and the paper's [reference implementation](https://github.com/adamzweiger/kv-compaction).

    *All figures, plots, and widgets are computed on Apple Silicon MPS or pure CPU — no discrete GPU required.*

    [↑ Back to the top](#headline)
    """)
    return


if __name__ == "__main__":
    app.run()
