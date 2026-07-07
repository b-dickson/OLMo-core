# Long-context evaluation: NIAH + RULER

Zero-shot long-context retrieval evals for trained OLMo-core checkpoints, used for the
attention-scaling paper's long-context section. Tests whether the loss-equivalence
between attention mechanisms (full / sliding-window / linear) survives as context
length grows past the 8192 training length.

## What's here

| File | Purpose |
|------|---------|
| `tasks.py` | Synthetic NIAH + RULER task generators (no external data needed). |
| `run_long_context.py` | Single-GPU runner: load checkpoint → generate → score → JSON. |
| `plot_long_context.py` | NIAH depth×length heatmap + accuracy-vs-length curves. |
| `../../modal/long_context_modal.py` | Modal launcher (one GPU container per checkpoint). |

## Tasks (adapted from RULER, Hsieh et al. 2024 + Kamradt NIAH)

- `niah_single` — one magic number for one key; **depth-swept** for the heatmap.
- `niah_multikey` — several keyed numbers (distractors); retrieve the queried one.
- `niah_multivalue` — one key, several numbers; retrieve all.
- `variable_tracking` — multi-hop assignment chain; find all variables equal to a value.

Haystack filler is synthetic neutral sentences (RULER's noise-filler variant), so runs
are fully reproducible from `--seed`. Scoring is answer **recall** (substring match);
greedy decoding (temperature 0).

## Local / single-GPU

```bash
cd OLMo-core/src
python -m scripts.eval.long_context.run_long_context \
    --checkpoint /path/to/stepXXXX \
    --context-lengths 1024 2048 4096 8192 16384 32768 \
    --depths 0.0 0.25 0.5 0.75 1.0 \
    --tasks niah_single niah_multikey niah_multivalue variable_tracking \
    --num-samples 20 \
    --output results_hgdn_370m.json
```

Context lengths **beyond 8192** test extrapolation — where sliding-window and
fixed-state linear attention are expected to degrade first. Sliding-window models may
need `--attention-backend flash_2` depending on the build.

## Modal (one container per checkpoint, pulls from R2)

Requires the `r2-creds` Modal secret (same one the training launcher uses) so
`from_checkpoint` can read `r2://…` URIs. Checkpoints live under
`r2://llm-data/checkpoints/model-ladders/<run>/<size>/stepN`.

```bash
cd OLMo-core/src/scripts/modal
modal run long_context_modal.py \
    --checkpoint-uri "r2://llm-data/checkpoints/model-ladders/vanilla_gated-4.0x-370M_seq8192_4/370M/step26000,r2://llm-data/checkpoints/model-ladders/sliding_gated-4.0x-370M_seq8192_.../370M/stepN" \
    --labels "Full-370M,Sliding-370M" \
    --context-lengths "1024,2048,4096,8192,16384,32768" \
    --gpu-type H100
# → ./long_context_results/long_context_<label>.json
```

## Plot for the paper

```bash
python -m scripts.eval.long_context.plot_long_context \
    long_context_results/*.json --outdir ../../../../papers/scaling/figures/
# → fig_niah_heatmap.png, fig_longctx_acc_vs_length.png
```

## Notes / extension points

- `variable_tracking` currently inserts the assignment chain as one block; full RULER
  scatters the hops — scatter them if you want a harder multi-hop test.
- To add real-essay haystacks (Paul Graham), swap the filler pool in `tasks.py`.
- Answer matching is recall-based; for `niah_multivalue`/`variable_tracking` it is the
  fraction of gold items recalled.
