"""
Run NIAH / RULER-style long-context retrieval evals on a trained OLMo-core checkpoint.

Loads a checkpoint via :class:`TransformerGenerationModule.from_checkpoint`, sweeps
context lengths (and needle depths for NIAH), generates short answers greedily, and
scores by answer recall. Writes a results JSON consumable by ``plot_long_context.py``.

Single-GPU usage (local or inside Modal):

    python -m scripts.eval.long_context.run_long_context \
        --checkpoint /path/or/r2/uri/to/step1234 \
        --context-lengths 1024 2048 4096 8192 16384 \
        --depths 0.0 0.25 0.5 0.75 1.0 \
        --tasks niah_single niah_multikey niah_multivalue variable_tracking \
        --num-samples 20 \
        --output results_long_context.json

Notes:
  * Context lengths beyond the 8192 training sequence length test *extrapolation* and
    are where sliding-window / fixed-state linear attention are expected to degrade.
  * Greedy decoding (temperature 0) for determinism.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, List

import torch

import json as _json

from cached_path import cached_path
from olmo_core.data import TokenizerConfig
from olmo_core.generate import GenerationConfig, TransformerGenerationModule
from olmo_core.io import join_path, normalize_path
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import get_default_device

from .tasks import TASK_REGISTRY, Sample, score_sample


def build_tokenizer(identifier: str = "allenai/dolma2-tokenizer"):
    """Load the HF tokenizer matching the dolma2 training tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(identifier)
    return tok


def make_count_tokens(tok) -> Callable[[str], int]:
    def count(text: str) -> int:
        if not text:
            return 0
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return count


def left_pad_batch(token_lists: List[List[int]], pad_id: int, device):
    """Left-pad a batch of token id lists; return (input_ids, attention_mask)."""
    max_len = max(len(t) for t in token_lists)
    input_ids, mask = [], []
    for t in token_lists:
        pad = max_len - len(t)
        input_ids.append([pad_id] * pad + t)
        mask.append([0] * pad + [1] * len(t))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.long, device=device),
    )


def generate_samples(
    gen_module: TransformerGenerationModule,
    tok,
    samples: List[Sample],
    pad_id: int,
    batch_size: int,
    max_context_tokens: int,
) -> List[str]:
    """Tokenize, (truncate to max_context_tokens), generate, decode. Returns generations."""
    device = get_default_device()
    generations: List[str] = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        token_lists = []
        for s in chunk:
            ids = tok(s.prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > max_context_tokens:
                # Keep the *tail* so the question framing (at the end) survives;
                # the needle may be dropped at very small budgets -- that is the point
                # of the length sweep.
                ids = ids[-max_context_tokens:]
            token_lists.append(ids)
        input_ids, attn = left_pad_batch(token_lists, pad_id, device)
        with torch.no_grad():
            gen_ids, _, _ = gen_module.generate_batch(
                input_ids,
                attention_mask=attn,
                completions_only=True,
                log_timing=False,
            )
        for row in gen_ids:
            generations.append(tok.decode(row.tolist(), skip_special_tokens=True))
    return generations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Checkpoint dir (local path or R2/S3 URI).")
    ap.add_argument("--context-lengths", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--tasks", nargs="+", default=list(TASK_REGISTRY.keys()))
    ap.add_argument("--num-samples", type=int, default=20, help="Samples per (task, length[, depth]).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--use-cache", action="store_true",
                    help="Enable KV cache (only valid for pure-softmax models with a flash "
                         "backend; breaks on linear/hybrid models). Default off for robustness.")
    ap.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    ap.add_argument("--attention-backend", default=None,
                    help="Override attention backend, e.g. flash_2 / flash_3.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--output", default="results_long_context.json")
    ap.add_argument("--label", default=None, help="Optional run label (e.g. arch/size) for the JSON.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-generations", type=int, default=0,
                    help="Store the first N (expected, generation, score) per (task, length) "
                         "in the output JSON for debugging/inspection.")
    args = ap.parse_args()

    tcfg = TokenizerConfig.dolma2()
    tok = build_tokenizer(args.tokenizer)
    count_tokens = make_count_tokens(tok)

    from olmo_core.config import DType

    # use_cache=False is deliberate and required for cross-architecture robustness:
    #   (1) the checkpoints' saved attention backend (TorchAttentionBackend / SDPA) does not
    #       support KV caching; and
    #   (2) prepare_inference_cache() asserts every block.attention is an `Attention` module,
    #       which is false for the linear / hybrid GatedDeltaNet models (their mixer is not
    #       Attention). Disabling the cache skips that path and recomputes the forward each
    #       step, which is correct for all four families. Answers are short, so we keep
    #       max_new_tokens small to bound the recompute cost.
    gen_cfg = GenerationConfig(
        pad_token_id=tcfg.pad_token_id,
        eos_token_id=tcfg.eos_token_id,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
        use_cache=args.use_cache,
    )
    backend = AttentionBackendName(args.attention_backend) if args.attention_backend else None

    # Build the transformer config ourselves from config.json["model"]. The WSDS attention-ladder
    # checkpoints store `data_loader`/`instance_sources` rather than a top-level `dataset` key, so
    # from_checkpoint's default config parse (which expects config["dataset"]["tokenizer"]) raises
    # KeyError 'dataset'. Passing transformer_config explicitly bypasses that path; we already pass
    # generation_config, so the checkpoint's tokenizer config is not needed.
    ckpt = normalize_path(args.checkpoint)
    with cached_path(join_path(ckpt, "config.json")).open() as f:
        cfg_dict = _json.load(f)
    transformer_config = TransformerConfig.from_dict(cfg_dict["model"])

    print(f"[long-context] loading checkpoint: {args.checkpoint}", file=sys.stderr)
    gen_module = TransformerGenerationModule.from_checkpoint(
        args.checkpoint,
        transformer_config=transformer_config,
        generation_config=gen_cfg,
        dtype=DType(args.dtype),
        attention_backend=backend,
    )

    results = {
        "checkpoint": args.checkpoint,
        "label": args.label,
        "config": {
            "context_lengths": args.context_lengths,
            "depths": args.depths,
            "tasks": args.tasks,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
        },
        "scores": [],  # list of {task, context_length, depth, accuracy, n}
        "samples": [],  # debug dump (only populated if --dump-generations > 0)
    }

    for ctx_len in args.context_lengths:
        # Leave headroom for the framing + answer tokens.
        haystack_budget = max(64, ctx_len - 128)
        for task in args.tasks:
            gen_fn = TASK_REGISTRY[task]
            if task == "niah_single":
                samples = gen_fn(approx_context_tokens=haystack_budget, count_tokens=count_tokens,
                                 num_samples=args.num_samples, depths=args.depths, seed=args.seed)
            else:
                samples = gen_fn(approx_context_tokens=haystack_budget, count_tokens=count_tokens,
                                 num_samples=args.num_samples, seed=args.seed)
            gens = generate_samples(gen_module, tok, samples, tcfg.pad_token_id,
                                    args.batch_size, max_context_tokens=ctx_len)
            if args.dump_generations > 0:
                for s, g in zip(samples[: args.dump_generations], gens[: args.dump_generations]):
                    rec = {"task": task, "context_length": ctx_len,
                           "expected": s.answers, "generation": g,
                           "score": score_sample(s, g),
                           "prompt_tail": s.prompt[-300:]}
                    results["samples"].append(rec)
                    print(f"  [dump] {task} L={ctx_len} expected={s.answers} "
                          f"gen={g!r} score={rec['score']:.2f}", file=sys.stderr)
            # Aggregate. NIAH single is reported per-depth (for the heatmap).
            if task == "niah_single":
                by_depth = {d: [] for d in args.depths}
                for s, g in zip(samples, gens):
                    by_depth[s.depth_frac].append(score_sample(s, g))
                for d, sc in by_depth.items():
                    acc = sum(sc) / max(1, len(sc))
                    results["scores"].append({"task": task, "context_length": ctx_len,
                                              "depth": d, "accuracy": acc, "n": len(sc)})
                    print(f"  {task:18s} L={ctx_len:6d} depth={d:.2f} acc={acc:.3f}", file=sys.stderr)
            else:
                sc = [score_sample(s, g) for s, g in zip(samples, gens)]
                acc = sum(sc) / max(1, len(sc))
                results["scores"].append({"task": task, "context_length": ctx_len,
                                          "depth": None, "accuracy": acc, "n": len(sc)})
                print(f"  {task:18s} L={ctx_len:6d} acc={acc:.3f}", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[long-context] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
