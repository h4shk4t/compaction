"""
Pre-compute QA results at multiple compression ratios for notebook visualization.

This script:
1. Loads Qwen3-4B and encodes the demo article
2. Answers QA questions using the FULL (uncompacted) cache as baseline
3. For each of several keep ratios, compacts the cache and:
   a. Answers the same questions
   b. Computes perplexity of the generated answers
   c. Runs a verbatim repeat test
4. Saves all results to a .pt file the notebook can load

Usage:
    python scripts/precompute_qa_results.py
    python scripts/precompute_qa_results.py --device mps
    python scripts/precompute_qa_results.py --ratios 0.05,0.10,0.25,0.50,0.75
"""
import argparse
import sys
import os
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from evaluation.utils import (
    load_model_and_tokenizer,
    extract_full_kv_cache,
    format_question,
    parse_model_choice,
    compute_perplexity_on_compacted_cache,
)
from evaluation.configs.utils import load_query_config
from compaction.compaction_methods.registry import get_compaction_method
from models.generate import generate_with_compacted_cache_batch


# ── Article and questions ────────────────────────────────────────────────────
# Single source of truth: ARTICLE lives in extract_kv_cache.py. We import it
# here so the KV-cache extraction and QA pre-compute stay in perfect sync.
from scripts.extract_kv_cache import ARTICLE

QUESTIONS = [
    # Section 2 — Independence
    {
        "question": "When did Verandia declare independence?",
        "options": [
            "June 15, 1972",
            "March 3, 1987",
            "January 1, 2000",
            "August 22, 1965",
        ],
        "gold_label": 2,  # B
    },
    # Section 4 — Marine Reserve
    {
        "question": "What is the Korvath Marine Reserve primarily named after?",
        "options": [
            "A species of coral found near the island",
            "The body of water surrounding Verandia",
            "Verandia's first president, Elena Korvath",
            "A local Verandian Creole word for ocean",
        ],
        "gold_label": 3,  # C
    },
    # Section 5 — Research Vessel
    {
        "question": "What is the name of the deep-sea research vessel operated by the National University of Verandia?",
        "options": [
            "RV Ocean Explorer",
            "RV Deep Current",
            "RV Reef Guardian",
            "RV Coral Pioneer",
        ],
        "gold_label": 4,  # D
    },
    # Section 4 — Marine Reserve size (numeric retrieval)
    {
        "question": "How large is the Korvath Marine Reserve?",
        "options": [
            "1,200 square kilometers",
            "12,000 square kilometers",
            "120,000 square kilometers",
            "487 square kilometers",
        ],
        "gold_label": 2,  # B
    },
    # Section 6 — Languages
    {
        "question": "Which two languages are official in Verandia?",
        "options": [
            "English and French",
            "English and Samoan",
            "English and Verandian Creole",
            "Verandian Creole and Hindi",
        ],
        "gold_label": 3,  # C
    },
    # Section 7 — Endemic wildlife
    {
        "question": "Which of the following is an endemic species of Verandia?",
        "options": [
            "The Verandian fruit bat (Pteropus verandensis)",
            "The Galapagos tortoise",
            "The snow leopard",
            "The giant panda",
        ],
        "gold_label": 1,  # A
    },
]

REPEAT_PROMPT = "Please repeat the context above verbatim, word for word, without any additions or omissions."


def ask_questions(model, tokenizer, questions, compacted_cache, original_seq_len, model_name):
    """Answer questions and compute perplexity for each answer."""
    prompts = [
        format_question(tokenizer, q["question"], q["options"], model_name)
        for q in questions
    ]

    answers = generate_with_compacted_cache_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        compacted_cache=compacted_cache,
        # MCQ answers usually fit in ~500-1500 tokens (including a <think>
        # block). 2048 is fine for these, but we cap a bit higher to avoid
        # truncating the thinking of harder questions.
        max_new_tokens=3072,
        original_seq_len=original_seq_len,
    )

    results = []
    for q, answer, prompt in zip(questions, answers, prompts):
        choice = parse_model_choice(answer)
        correct = choice == q["gold_label"] if choice else False
        letter = chr(64 + choice) if choice else "?"
        gold_letter = chr(64 + q["gold_label"])

        # Compute perplexity of the generated answer
        answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
        device = next(model.parameters()).device
        try:
            ppl, log_ppl = compute_perplexity_on_compacted_cache(
                model=model,
                tokenizer=tokenizer,
                compacted_cache=compacted_cache,
                generated_token_ids=answer_token_ids,
                question_prompt=prompt,
                device=str(device),
                original_seq_len=original_seq_len,
            )
        except Exception as e:
            print(f"  Perplexity computation failed: {e}")
            ppl, log_ppl = float("nan"), float("nan")

        results.append({
            "question": q["question"],
            "options": q["options"],
            "gold_label": q["gold_label"],
            "gold_letter": gold_letter,
            "answer_text": answer.strip(),
            "parsed_choice": letter,
            "correct": correct,
            "perplexity": ppl,
            "log_perplexity": log_ppl,
        })
    return results


def repeat_test(model, tokenizer, article, compacted_cache, original_seq_len, model_name,
                max_new_tokens=4096):
    """Run verbatim repeat test and compute word recall.

    max_new_tokens must be large enough to cover the full article plus any
    <think> prefix the model emits. For a 1409-token article we default to
    4096, which gives the model roughly 2.5× the article's token budget.
    """
    prompt = format_question(tokenizer, REPEAT_PROMPT, options=None, model_name=model_name)
    answers = generate_with_compacted_cache_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        compacted_cache=compacted_cache,
        max_new_tokens=max_new_tokens,
        original_seq_len=original_seq_len,
    )
    generated = answers[0].strip()

    # Strip thinking block if present
    generated_clean = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()

    # Word-level recall
    article_words = article.split()
    generated_lower = generated_clean.lower()
    matched = sum(1 for w in article_words if w.lower() in generated_lower)
    word_recall = matched / len(article_words) if article_words else 0.0

    # Perplexity of the repeat
    answer_token_ids = tokenizer.encode(generated, add_special_tokens=False)
    device = next(model.parameters()).device
    try:
        ppl, log_ppl = compute_perplexity_on_compacted_cache(
            model=model,
            tokenizer=tokenizer,
            compacted_cache=compacted_cache,
            generated_token_ids=answer_token_ids,
            question_prompt=prompt,
            device=str(device),
            original_seq_len=original_seq_len,
        )
    except Exception as e:
        print(f"  Repeat perplexity failed: {e}")
        ppl, log_ppl = float("nan"), float("nan")

    return {
        # Keep the full generated repeat so the notebook's alignment-based
        # recall can score it fairly. Truncation here would make the
        # verbatim-repeat diff look much worse than it is.
        "generated_text": generated_clean,
        "word_recall": word_recall,
        "matched_words": matched,
        "total_words": len(article_words),
        "perplexity": ppl,
        "log_perplexity": log_ppl,
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-compute QA results at multiple compression ratios")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ratios", default="0.05,0.10,0.20,0.30,0.50,0.75",
                        help="Comma-separated keep ratios to evaluate")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ratios = [float(r.strip()) for r in args.ratios.split(",")]

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Device: {args.device}")
    print(f"Model:  {args.model}")
    print(f"Ratios: {ratios}")
    print()

    # ── Step 1: Load model ────────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, device="cpu")
    if args.device != "cpu":
        model = model.to(args.device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Extract full KV cache ─────────────────────────────────────
    print("\nExtracting KV cache...")
    seq_len, past_key_values, article_indices, formatted_context, _ = extract_full_kv_cache(
        model, tokenizer, ARTICLE, args.device, model_name=args.model,
    )
    article_len = len(article_indices)
    print(f"  Total tokens: {seq_len}, Article tokens: {article_len}")

    # ── Step 3: Baseline — full cache ─────────────────────────────────────
    print("\n=== BASELINE (full cache) ===")
    original_cache = tuple(
        (k, torch.zeros_like(k[:, :, :, 0]), v)
        for k, v in past_key_values
    )

    print("  Answering questions...")
    original_qa = ask_questions(
        model, tokenizer, QUESTIONS, original_cache, seq_len, args.model,
    )
    for r in original_qa:
        print(f"    Q: {r['question'][:60]}... → {r['parsed_choice']} (gold: {r['gold_letter']}) "
              f"{'OK' if r['correct'] else 'WRONG'}  ppl={r['perplexity']:.2f}")

    print("  Repeat test...")
    original_repeat = repeat_test(
        model, tokenizer, ARTICLE, original_cache, seq_len, args.model,
    )
    print(f"    Word recall: {original_repeat['word_recall']:.1%}, ppl={original_repeat['perplexity']:.2f}")
    del original_cache

    # ── Step 4: Compacted caches at each ratio ────────────────────────────
    algorithm_kwargs = {
        "algorithm": "highest_attention_keys",
        "score_method": "rms",
        "nnls_iters": 2,
        "nnls_lower_bound": 0.05,
        "nnls_upper_bound": 20.0,
        "c2_method": "lsq",
        "precomputed_budget_path": "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json",
    }
    compaction_method = get_compaction_method(
        "AM-HighestAttnKeys", method_kwargs=algorithm_kwargs,
    )
    query_config = load_query_config("repeat")
    non_article_tokens = seq_len - article_len

    ratio_results = {}
    for ratio in ratios:
        print(f"\n=== RATIO {ratio:.0%} ===")
        target_article_tokens = max(1, int(article_len * ratio))
        actual_target = target_article_tokens + non_article_tokens

        # Compact
        t0 = time.time()
        compacted_cache, stats = compaction_method.compact_kv_cache(
            past_key_values=past_key_values,
            target_size=actual_target,
            indices=article_indices,
            query_config=query_config,
            model=model,
            tokenizer=tokenizer,
            formatted_context=formatted_context,
            compute_stats=False,
        )
        compact_time = time.time() - t0
        print(f"  Compacted in {compact_time:.1f}s")

        # Compute compacted cache size for logging
        compacted_seq_len = compacted_cache[0][0].shape[-2] if compacted_cache else 0

        # QA
        print("  Answering questions...")
        qa_results = ask_questions(
            model, tokenizer, QUESTIONS, compacted_cache, seq_len, args.model,
        )
        for r in qa_results:
            print(f"    Q: {r['question'][:60]}... → {r['parsed_choice']} (gold: {r['gold_letter']}) "
                  f"{'OK' if r['correct'] else 'WRONG'}  ppl={r['perplexity']:.2f}")

        # Repeat
        print("  Repeat test...")
        rpt = repeat_test(
            model, tokenizer, ARTICLE, compacted_cache, seq_len, args.model,
        )
        print(f"    Word recall: {rpt['word_recall']:.1%}, ppl={rpt['perplexity']:.2f}")

        ratio_results[ratio] = {
            "keep_ratio": ratio,
            "target_article_tokens": target_article_tokens,
            "compacted_seq_len": compacted_seq_len,
            "compaction_time": compact_time,
            "qa_results": qa_results,
            "repeat_result": rpt,
            "accuracy": sum(r["correct"] for r in qa_results) / len(qa_results),
        }
        del compacted_cache

    # ── Step 5: Save everything ───────────────────────────────────────────
    save_data = {
        "model_name": args.model,
        "article_text": ARTICLE,
        "questions": QUESTIONS,
        "seq_len": seq_len,
        "article_len": article_len,
        "original_qa": original_qa,
        "original_repeat": original_repeat,
        "original_accuracy": sum(r["correct"] for r in original_qa) / len(original_qa),
        "ratios": ratios,
        "ratio_results": ratio_results,
    }

    if args.output:
        output_path = args.output
    else:
        model_short = args.model.split("/")[-1]
        os.makedirs("data/cached_kv", exist_ok=True)
        output_path = f"data/cached_kv/{model_short}_qa_results.pt"

    print(f"\nSaving to {output_path}...")
    torch.save(save_data, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.1f} MB")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Original accuracy: {save_data['original_accuracy']:.0%}")
    for ratio in ratios:
        r = ratio_results[ratio]
        print(f"  {ratio:5.0%} keep: accuracy={r['accuracy']:.0%}, "
              f"repeat_recall={r['repeat_result']['word_recall']:.1%}, "
              f"compact_time={r['compaction_time']:.1f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
