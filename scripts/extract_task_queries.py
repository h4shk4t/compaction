"""
Extract task-prompt query vectors from Qwen3-4B.

For the Verandia article cached in data/cached_kv/Qwen3-4B.pt we already have
query vectors corresponding to the article tokens themselves (these are the
"self-study" queries the model produces when reading the article). But the
notebook's Part G needs a *different* kind of query: the queries the model
produces when it is specifically asked a question about the article.

This script does:
1. Load Qwen3-4B and tokenize `{article}\\n\\n{task_prompt}` for each of
   several pre-specified task prompts.
2. Run a forward pass with hooks on every attention layer to capture the Q
   vectors (RoPE applied — this matters because the task tokens live at
   higher positions than the article tokens and therefore have different
   rotary phases).
3. Slice out *only* the Q vectors at positions corresponding to the task
   prompt portion of the input — i.e., everything after the article ends.
4. Pad task-query sets to a common length (max_task_tokens) so they stack
   cleanly into a single tensor per layer.
5. Save to data/cached_kv/Qwen3-4B_task_queries.pt.

Usage:
    python scripts/extract_task_queries.py
    python scripts/extract_task_queries.py --device mps
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from evaluation.utils import load_model_and_tokenizer, format_context
from scripts.extract_kv_cache import ARTICLE, extract_queries_via_hooks


# Three task prompts targeting non-adjacent sections so hot bands on the
# Part G heatmap will be spatially separated.
TASK_PROMPTS = {
    "T1_independence": (
        "When did Verandia declare independence, and who was its first "
        "president?"
    ),
    "T2_research_vessel": (
        "What is the name of the deep-sea research vessel, when was it "
        "launched, and at which university is it based?"
    ),
    "T3_wildlife": (
        "List every endemic species and climate feature mentioned in the "
        "document."
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Extract task-prompt query vectors")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Device: {args.device}")
    print(f"Model:  {args.model}")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, device="cpu")
    if args.device != "cpu":
        print(f"  Moving model to {args.device}...")
        model = model.to(args.device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Tokenize article-only to find the common-prefix length ───────────
    # format_context wraps the article in a chat template that ends in an
    # assistant-start suffix. So:
    #   format_context(article) = [prefix | article_text | suffix]
    #   format_context(article + task) = [prefix | article_text | task_text | suffix]
    # The LONGEST COMMON PREFIX between the two token sequences equals
    # [prefix | article_text] — which is exactly where the task tokens start
    # in the combined sequence. This method is robust to any chat-template
    # idiosyncrasies (no hard-coding of prefix/suffix lengths).
    article_only_fmt = format_context(tokenizer, ARTICLE, model_name=args.model)
    article_only_ids = tokenizer(
        article_only_fmt, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0].tolist()
    print(f"\nArticle-only tokenized length: {len(article_only_ids)} tokens")

    # ── For each task: format article+task, forward pass, slice task tokens ─
    per_task_queries = {}  # task_name -> list[Tensor (1, n_q_heads, n_task_tokens, head_dim)] per layer
    per_task_token_counts = {}

    for task_name, task_prompt in TASK_PROMPTS.items():
        print(f"\n── Task '{task_name}' ──")
        combined = ARTICLE + "\n\n" + task_prompt
        formatted = format_context(tokenizer, combined, model_name=args.model)
        input_ids_cpu = tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        input_ids = input_ids_cpu.to(args.device)
        total_len = input_ids.shape[1]
        combined_ids_list = input_ids_cpu[0].tolist()

        # Find longest common prefix with article-only tokens
        common = 0
        for a, b in zip(article_only_ids, combined_ids_list):
            if a != b:
                break
            common += 1
        task_positions = slice(common, total_len)
        n_task_tokens = total_len - common
        print(f"  Total tokens: {total_len}")
        print(f"  Common prefix with article-only: {common}")
        print(f"  Task-portion tokens: {n_task_tokens} (positions {common}..{total_len})")

        t0 = time.time()
        _past_kv, query_vectors = extract_queries_via_hooks(model, input_ids)
        print(f"  Forward pass: {time.time() - t0:.1f}s")

        # Slice each layer's Q to task tokens only and record
        task_Q_per_layer = []
        for q_layer in query_vectors:
            # q_layer shape: (1, n_q_heads, total_len, head_dim)
            q_task = q_layer[:, :, task_positions, :].contiguous()
            task_Q_per_layer.append(q_task)
        per_task_queries[task_name] = task_Q_per_layer
        per_task_token_counts[task_name] = n_task_tokens

    # ── Pad to a common max_task_tokens and stack ──────────────────────────
    max_task_tokens = max(per_task_token_counts.values())
    n_layers = len(next(iter(per_task_queries.values())))
    print(f"\nPadding task queries to max_task_tokens={max_task_tokens}")

    # Build save_data: for each layer, a stacked tensor of shape
    #   (n_tasks, n_q_heads, max_task_tokens, head_dim)
    # with zero-padding on the token axis where a task was shorter.
    task_names = list(TASK_PROMPTS.keys())
    save_data = {
        "model_name": args.model,
        "task_names": task_names,
        "task_prompts": TASK_PROMPTS,
        "task_token_counts": per_task_token_counts,
        "max_task_tokens": max_task_tokens,
        "article_only_len": len(article_only_ids),
    }

    for layer_idx in range(n_layers):
        # Reference shape from any task's tensor at this layer
        q_ref = per_task_queries[task_names[0]][layer_idx]
        _, n_q_heads, _, head_dim = q_ref.shape
        stacked = torch.zeros(
            len(task_names), n_q_heads, max_task_tokens, head_dim,
            dtype=q_ref.dtype,
        )
        for t_idx, task_name in enumerate(task_names):
            q = per_task_queries[task_name][layer_idx].squeeze(0)  # (n_q_heads, n_task_tokens, head_dim)
            n = q.shape[1]
            stacked[t_idx, :, :n, :] = q
        save_data[f"Q_{layer_idx}"] = stacked

    # ── Save ──────────────────────────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        model_short = args.model.split("/")[-1]
        os.makedirs("data/cached_kv", exist_ok=True)
        output_path = f"data/cached_kv/{model_short}_task_queries.pt"

    print(f"\nSaving to {output_path}...")
    torch.save(save_data, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.1f} MB")

    print("\nDone! Summary:")
    for name, count in per_task_token_counts.items():
        print(f"  {name}: {count} task tokens")


if __name__ == "__main__":
    main()
