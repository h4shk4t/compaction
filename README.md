# Fast KV Compaction via Attention Matching — Interactive Explainer

**Submission for the marimo Notebook Competition 2026.**

By **Ashutosh Srivastava** &nbsp;·&nbsp; [@h4shkat](https://x.com/h4shkat) &nbsp;·&nbsp; [LinkedIn](https://www.linkedin.com/in/ashutosh-srivastava-1bbb0a223/) &nbsp;·&nbsp; [Portfolio](https://h4shk4t.github.io/)

Paper: Zweiger, Fu, Guo, Kim (2026). *Fast KV Compaction via Attention Matching.* [arXiv:2602.16284](https://arxiv.org/abs/2602.16284).

---

## What this is

An interactive walkthrough of the paper's **Attention Matching (AM)** algorithm — the math, the code, and the empirical results — using cached tensors from the actual Qwen3-4B model so every plot and slider responds in real time without needing to load the model.

The dramatic headline: **at 20% of Qwen3-4B's KV cache, the model still answers 6/6 multiple-choice questions correctly and recites the article with ~99% verbatim accuracy** — using only matrix algebra (NNLS + OLS), no gradient descent.

## How to read

Open `notebook.py` in marimo or molab. For the cleanest first read, click the **⋮** menu (top-right) and toggle **"Show code"** off.

All controls live in the **left sidebar**. Drag the **QA keep ratio** slider at the top first — the headline teaser reacts to it live.

## Bundle layout (~27 MB total)

```
notebook.py                                # the marimo notebook (open this)
pyproject.toml                             # minimal dependencies
data/cached_kv/                            # 4 small cache files bundled directly
  Qwen3-4B_qa_results.pt                   # MCQ + verbatim repeat at 6 ratios (96 KB)
  Qwen3-4B_task_queries.pt                 # task-prompt Q vectors for Part D (24 MB)
  Qwen3-4B_layer_heatmap.pt                # Part C layer × ratio quality grid (8 KB)
  Qwen3-4B_pareto_default.pt               # Part F default Pareto scatter (3 KB)
                                           # Qwen3-4B.pt (605 MB) — auto-downloaded
                                           # on first run from the Hugging Face dataset
                                           # at h4shk4t/fast-kv-compaction-cache.
head_budget_optimization/head_budgets/     # optimized non-uniform head budget JSON
compaction/algorithms/                     # the paper's reference compaction algos
scripts/                                   # one-shot scripts that produced the caches
                                           # (not needed to run the notebook)
```

## First-run note (Qwen3-4B.pt)

The largest cache file is 605 MB — over molab's 100 MB single-upload cap. The notebook downloads it on first run from a public Hugging Face dataset (`https://huggingface.co/datasets/h4shk4t/fast-kv-compaction-cache`) into `data/cached_kv/Qwen3-4B.pt`. Expect a one-time 30–60 second wait; subsequent runs use the local cache instantly.

If the download fails (rare; retry first), you can also regenerate the file from scratch with:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/extract_kv_cache.py --device mps
```

## What's in the notebook

- **Headline teaser** — Qwen3-4B 6-MCQ table + word-by-word diff at any ratio (sidebar slider).
- **Part A** — interactive OMP residual visualization on a tiny synthetic problem.
- **Part B** — full AM pipeline on toy multi-head KV (live compute, the only non-cached section).
- **Part C** — real-model per-layer × per-ratio compaction quality heatmap.
- **Part D** — *same article, different tasks, different compactions* — the novel contribution: a clean visualization of how AM is actually a function of $(K, V, Q_\text{ref})$, not just $(K, V)$. Frames query-conditioned compaction as the natural lever for multi-agent / RAG systems.
- **Part E** — full per-ratio QA breakdown (which questions break first, verbatim recall vs ratio).
- **Part F** — speed/quality Pareto reproducing the paper's Figure 1.
- **β + Cv ablation** — visualization of attention-mass correction + mixture-attention ablation showing why all three pieces matter.
- **Recap + footer** with paper citation and reproducibility command.

## Reproducing the caches

If you want to regenerate everything from scratch (requires Qwen3-4B + MPS or GPU):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/precompute_all.py --device mps
```

Total runtime ≈ 2–3 hours on Apple Silicon MPS.

## Compute requirements

The notebook itself runs on **CPU only** (or MPS if available — none of the cells require it). Memory peak ≈ 1.2 GB while the cached KV tensors are loaded. No GPU needed because the heavy work — running Qwen3-4B forward passes — already happened during cache extraction.
