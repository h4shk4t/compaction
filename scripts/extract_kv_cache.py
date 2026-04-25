"""
Extract KV cache and query vectors from a Qwen3 model for notebook analysis.

This script:
1. Loads Qwen3-4B (in bfloat16 on MPS/CUDA/CPU)
2. Encodes a short article via the chat template
3. Runs a single forward pass
4. Hooks into each attention layer to capture the query vectors (Q)
5. Saves K, V, Q per layer to a .pt file the notebook can load

Usage:
    python scripts/extract_kv_cache.py
    python scripts/extract_kv_cache.py --model Qwen/Qwen3-4B --device mps
    python scripts/extract_kv_cache.py --device cpu  # slower but always works

The output file will be saved to data/cached_kv/<model_short_name>.pt
"""
import argparse
import sys
import os
import time

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from evaluation.utils import load_model_and_tokenizer, extract_full_kv_cache


# ── The article we'll encode ─────────────────────────────────────────────────
# Expanded ~2000-token Verandia dossier with 8 clearly-named sections. Each
# section contains specific named entities and facts so query-specific
# compaction can target different sections with minimal overlap.
#
# This is the single source of truth — precompute_qa_results.py and
# extract_task_queries.py both import ARTICLE from here.

ARTICLE = """\
Section 1 — Geography and Discovery.
The small island nation of Verandia is located in the southern Pacific Ocean, \
approximately 1,400 kilometers northeast of New Zealand and 2,800 kilometers \
east of Australia. Its total land area spans 1,856 square kilometers across \
one main island, called Verand Major, and 14 smaller volcanic outcroppings \
collectively known as the Kiri Archipelago. The island was first charted by \
the Dutch navigator Hendrik van Droost in 1782, who named it after his \
patron Admiral Verandus van Hoeven. From 1810 to 1987 the island was a \
colony of the British Empire, administered as part of the Western Pacific \
Territories.

Section 2 — Independence and Governance.
Verandia declared independence from colonial rule on March 3, 1987, \
following a decade of peaceful self-determination movements led by the \
Verandian National Congress. Its first president, Elena Korvath, served two \
consecutive terms from 1987 to 1999. Under Korvath's leadership Verandia \
adopted a parliamentary constitutional system with a directly elected \
35-seat Legislative Assembly. The head of state is the president, elected \
to a five-year term and limited to two consecutive terms. Executive power \
rests with the Prime Minister, who is nominated from the majority bloc in \
the Assembly. The current Prime Minister is Samuel Tenari, in office since \
2024.

Section 3 — Economy.
Verandia's economy has undergone a dramatic transformation since \
independence. Under colonial administration the island depended almost \
exclusively on sugar cane exports, which at peak production in 1958 \
accounted for 87 percent of all foreign-currency earnings. Korvath's \
economic diversification plan shuttered the last commercial sugar plantation \
in 1994 and redirected investment toward three pillars: eco-tourism, \
sustainable fisheries, and a nascent technology sector. Today the technology \
sector, centered on remote-work platforms and marine-monitoring sensors, \
employs approximately 4,200 people and contributes 22 percent of GDP. \
Sustainable fisheries, managed through a quota system introduced in 2001, \
produce an annual catch of 18,500 tons, primarily tuna, snapper, and farmed \
oysters. Eco-tourism draws around 110,000 visitors per year, generating \
revenue equivalent to 31 percent of GDP.

Section 4 — The Korvath Marine Reserve.
In 2003 Verandia established the Korvath Marine Reserve, named after the \
former president, which spans 12,000 square kilometers around the island's \
coral reef system. The reserve contains the largest contiguous section of \
healthy staghorn coral in the southern Pacific. The reserve became a UNESCO \
World Heritage Site in 2008 and attracts roughly 40,000 visitors per year. \
Revenue from the reserve's entrance fees funds the majority of the island's \
public school system; approximately 74 percent of education funding in 2024 \
came from marine reserve receipts. Visitor numbers are capped at 250 per day \
across six access points to prevent ecological damage.

Section 5 — Capital and Education.
The capital of Verandia is Port Alani, home to approximately 21,000 \
residents and located on the southeastern coast of Verand Major. Port Alani \
hosts the National University of Verandia, which opened its doors in 1995 \
and has since grown to an enrollment of 3,400 students. The university is \
best known for its marine biology program, which collaborates with research \
institutions in New Zealand, Japan, Australia, and the United States. The \
university operates a deep-sea research vessel called the RV Coral Pioneer, \
launched in 2019 and funded through a combination of government grants and \
private donations. The Coral Pioneer can descend to 2,500 meters and is \
equipped with two remotely operated underwater vehicles. The university \
also runs a smaller coastal-research vessel called the RV Reef Sentinel, \
active since 2011.

Section 6 — Demographics and Language.
Verandia's current population stands at approximately 58,000 people, \
distributed roughly 36 percent urban (Port Alani and two smaller towns) and \
64 percent rural across the main island and outlying atolls. The median age \
is 34.2 years. Life expectancy at birth is 79.4 years, among the highest in \
the region. The official languages are English and Verandian Creole, a \
contact language that developed in the 19th century from English, Samoan, \
and Hindi elements brought by indentured plantation workers. The literacy \
rate stands at 98.1 percent for adults over 15, attributable in large part \
to the universal primary-and-secondary education program funded by the \
Korvath Marine Reserve. Approximately 68 percent of residents identify as \
bilingual.

Section 7 — Climate and Wildlife.
Verandia has a tropical monsoon climate, with an annual rainfall of 2,400 \
millimeters falling mostly between November and March. Average temperatures \
range from 22 degrees Celsius in the cooler months to 29 degrees Celsius in \
the warmest. The island hosts several endemic species, including the \
Verandian fruit bat (Pteropus verandensis), the crimson-tufted rail \
(Rallina korvathi, a ground-dwelling bird named after the first president), \
and the coral-reef-dwelling Verandian angelfish (Centropyge verandiensis). \
Marine biodiversity within the Korvath Reserve includes 487 documented \
coral species and at least 1,240 species of reef fish. The island's coral \
reef system is considered a hope spot by international marine conservation \
groups due to its resilience against bleaching events driven by climate \
change.

Section 8 — Sports and Cuisine.
The national sport of Verandia is outrigger canoe racing, introduced in the \
late 19th century by Polynesian migrants and codified in national rules in \
1972. The annual Korvath Regatta, held each February across a 40-kilometer \
offshore course, draws competitors from twelve Pacific nations and remains \
the island's largest public event. Verandian cuisine is dominated by \
seafood; raw marinated snapper, smoked tuna, and oyster chowder appear on \
nearly every restaurant menu, accompanied by staple starches of breadfruit, \
taro, and coconut-flavored rice. The national dish, known locally as \
pili-pili (Verandian Creole for small-fish stew), combines reef-caught \
parrotfish with green chilies, lime, and coconut cream.
"""


def extract_queries_via_hooks(model, input_ids):
    """
    Run a forward pass and capture query vectors from every attention layer
    using PyTorch forward hooks.

    How hooks work:
    - We register a function on each attention module.
    - When model(input_ids) runs, each attention layer's forward() is called.
    - Our hook fires right AFTER the forward, giving us access to the inputs
      that were passed to the attention layer.
    - We grab the query_states from inside the attention computation.

    But there's a subtlety: the hook sees the module's inputs/outputs, not
    intermediate variables like query_states. So instead, we'll use a
    forward PRE-hook that patches the module to save query_states, or
    simpler: we re-derive Q from the hidden states using the layer's
    own projection weights.

    Actually, the cleanest approach: use a forward hook on the attention
    module and recompute Q from the input hidden_states. This is exactly
    what the model does internally (line 211 of modeling_qwen3.py):
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

    Returns
    -------
    past_key_values : tuple
        The KV cache from the forward pass
    query_vectors : list of Tensor
        Q vectors per layer, each shape (1, num_q_heads, seq_len, head_dim)
    """
    query_vectors = []
    hooks = []

    def make_hook(layer_idx):
        """Create a hook that captures query vectors for one layer."""
        def hook_fn(module, args, kwargs, output):
            # The first positional arg to Qwen3Attention.forward() is hidden_states
            hidden_states = args[0] if args else kwargs.get("hidden_states")

            # Recompute Q exactly as the model does (see modeling_qwen3.py line 211)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            q = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

            # Apply RoPE (rotary positional embeddings)
            # We need cos/sin from position_embeddings
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                # position_embeddings might be in positional args
                # Check the forward signature — it's usually a kwarg
                pass

            if position_embeddings is not None:
                cos, sin = position_embeddings
                # Apply rotary embedding to Q (same function the model uses)
                from models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
                q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)  # K doesn't matter, we only want Q
                q = q_rot

            query_vectors.append(q.detach().cpu())

        return hook_fn

    # Register hooks on each attention layer
    for layer_idx, layer in enumerate(model.model.layers):
        attn_module = layer.self_attn
        # Use register_forward_hook with_kwargs=True so we can access kwargs
        h = attn_module.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
        hooks.append(h)

    # Run the forward pass — this triggers all hooks
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return outputs.past_key_values, query_vectors


def main():
    parser = argparse.ArgumentParser(description="Extract KV cache for notebook")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name")
    parser.add_argument("--device", default=None, help="Device (mps/cuda/cpu). Auto-detected if omitted.")
    parser.add_argument("--output", default=None, help="Output .pt file path")
    args = parser.parse_args()

    # Auto-detect device
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

    # ── Step 1: Load model ────────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()
    # Load to CPU first, then move to target device.
    # This avoids issues with device_map=None on MPS.
    model, tokenizer = load_model_and_tokenizer(args.model, device="cpu")
    if args.device != "cpu":
        print(f"  Moving model to {args.device}...")
        model = model.to(args.device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Format and tokenize the article ──────────────────────────
    print("\nFormatting article...")
    from evaluation.utils import format_context, compute_article_indices
    formatted_context = format_context(tokenizer, ARTICLE, model_name=args.model)
    inputs = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(args.device)
    seq_len = input_ids.shape[1]
    article_indices = compute_article_indices(tokenizer, formatted_context, ARTICLE)
    print(f"  Total tokens: {seq_len}")
    print(f"  Article tokens: {len(article_indices)} (positions {article_indices.start}..{article_indices.stop})")

    # ── Step 3: Extract KV cache + query vectors ─────────────────────────
    print("\nRunning forward pass with query extraction hooks...")
    t0 = time.time()
    past_key_values, query_vectors = extract_queries_via_hooks(model, input_ids)
    dt = time.time() - t0
    print(f"  Forward pass: {dt:.1f}s")

    # Print shapes for verification
    n_layers = len(past_key_values)
    k0 = past_key_values[0][0]  # (batch, num_kv_heads, seq_len, head_dim)
    q0 = query_vectors[0]       # (batch, num_q_heads, seq_len, head_dim)
    print(f"  Layers: {n_layers}")
    print(f"  K shape per layer: {tuple(k0.shape)} (batch, kv_heads, seq_len, head_dim)")
    print(f"  Q shape per layer: {tuple(q0.shape)} (batch, q_heads, seq_len, head_dim)")

    # ── Step 4: Package and save ─────────────────────────────────────────
    # Move everything to CPU and squeeze batch dimension
    save_data = {
        "model_name": args.model,
        "article_text": ARTICLE,
        "formatted_context": formatted_context,
        "seq_len": seq_len,
        "article_indices_start": article_indices.start,
        "article_indices_stop": article_indices.stop,
        "n_layers": n_layers,
        "n_kv_heads": k0.shape[1],
        "n_q_heads": q0.shape[1],
        "head_dim": k0.shape[3],
    }

    # Save K, V, Q per layer (squeeze batch dim: (1, H, T, D) -> (H, T, D))
    for layer_idx in range(n_layers):
        k_l = past_key_values[layer_idx][0].detach().cpu().squeeze(0)  # (kv_heads, T, D)
        v_l = past_key_values[layer_idx][1].detach().cpu().squeeze(0)  # (kv_heads, T, D)
        q_l = query_vectors[layer_idx].squeeze(0)                      # (q_heads, T, D)

        save_data[f"K_{layer_idx}"] = k_l
        save_data[f"V_{layer_idx}"] = v_l
        save_data[f"Q_{layer_idx}"] = q_l

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        model_short = args.model.split("/")[-1]
        os.makedirs("data/cached_kv", exist_ok=True)
        output_path = f"data/cached_kv/{model_short}.pt"

    print(f"\nSaving to {output_path}...")
    torch.save(save_data, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.1f} MB")
    print("\nDone! The notebook can now load this file.")


if __name__ == "__main__":
    main()
