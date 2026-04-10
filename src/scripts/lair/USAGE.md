## Attention-Scaling WSDS Ladders

This folder contains the WSDS (4x-Chinchilla) attention-ladder configs and Slurm launchers.
The 6 variants are laid out as a 2×3 grid (3 attention types × {no HC, with Identity Hyper-Connections}):

| variant | config file | slurm launcher (lair, H100) |
|---|---|---|
| `vanilla_gated` | `wsds_vanilla_gated.txt` | `slurm_wsds_vanilla_gated_h100.sh` |
| `vanilla_gated` + HC=4 | `wsds_vanilla_gated_hc.txt` | `slurm_wsds_vanilla_gated_hc_h100.sh` |
| `sliding_gated` | `wsds_sliding_gated.txt` | `slurm_wsds_sliding_gated_h100.sh` |
| `sliding_gated` + HC=4 | `wsds_sliding_gated_hc.txt` | `slurm_wsds_sliding_gated_hc_h100.sh` |
| `hybrid_gated_deltanet` | `wsds_hybrid_gated_deltanet.txt` | `slurm_wsds_hybrid_gated_deltanet_h100.sh` |
| `hybrid_gated_deltanet` + HC=4 | `wsds_hybrid_gated_deltanet_hc.txt` | `slurm_wsds_hybrid_gated_deltanet_hc_h100.sh` |

All 6 variants use the same standardized parameters:

- Sizes: `60M / 100M / 190M / 370M` (4 rows per file)
- Sequence length: `8192`
- Chinchilla multiple: `4.0`
- Sliding-window size: `4096` (used by `sliding_gated` only; full-attention layers see the full 8192-token sequence)
- Hyper-connection streams: `4` (HC variants only)

Config columns are `attention_type size chinchilla_multiple sequence_length`. The HC dimension is not encoded in the config row — it's set by the launcher (the `*_hc_*` slurm scripts pass `--hyper-connections=4`).

Slurm GRES names on this cluster are case-sensitive: `H100`. The current launchers all target `gpu:H100:2`. If you need an L40S launcher, copy one of the H100 scripts and adjust the `--gres` line.

## Launching on lair

Each launcher reads its own variant's config file by default — no env-var dance required.

Launch a full sweep (all 4 sizes for one variant):

```bash
sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh
sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_hc_h100.sh
sbatch --array=1-4 src/scripts/lair/slurm_wsds_hybrid_gated_deltanet_h100.sh
sbatch --array=1-4 src/scripts/lair/slurm_wsds_hybrid_gated_deltanet_hc_h100.sh
sbatch --array=1-4 src/scripts/lair/slurm_wsds_vanilla_gated_h100.sh
sbatch --array=1-4 src/scripts/lair/slurm_wsds_vanilla_gated_hc_h100.sh
```

Launch a single size (example: row 3 = `190M`):

```bash
sbatch --array=3 src/scripts/lair/slurm_wsds_hybrid_gated_deltanet_h100.sh
```

Row mapping (same for all 6 config files):

| array index | size |
|---|---|
| 1 | 60M |
| 2 | 100M |
| 3 | 190M |
| 4 | 370M |

## Defaults baked into the launchers

- `WANDB_PROJECT=attn-scaling-ladder`
- `WANDB_ENTITY=iu-cogai`
- `RUN_OPTIMIZER=muon`
- `SEQUENCE_LENGTH=8192` (also explicit in the config rows)
- `SLIDING_WINDOW_SIZE=4096`
- `BATCH_SIZE_MULTIPLIER=1.0`
- `MICROBATCH_DISCOUNT=1.0`
- `HC_STREAMS=4` (HC launchers only)
- Train data: `/data/user/dicksonb/data/nanochat/tokenized/*.npy`
- Eval data root: `/data/user/dicksonb/data`
- `LADDER_ROOT_DIR=/data/user/dicksonb/checkpoints`
- 2x H100 (`--gres=gpu:H100:2`)
- Wall time: 48h

## Common env-var overrides

All defaults are overridable at submit time:

```bash
RUN_OPTIMIZER=skipstep_adamw sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh
MICROBATCH_DISCOUNT=1.5      sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh
LADDER_PROJECT=my_project    sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh
LADDER_WANDB_ENTITY=my_team  sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh
HC_STREAMS=8                 sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_hc_h100.sh
SLIDING_WINDOW_SIZE=2048     sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh
```

You can also point any launcher at an alternate config file:

```bash
CONFIG_FILE=path/to/custom_config.txt sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh
```

## Manual local launch (no Slurm)

Useful when iterating on a single config:

```bash
torchrun --standalone --nproc-per-node=2 src/scripts/train/ladder/wsds_attention_ladder.py \
  run \
  --name=my-run \
  --size=190M \
  --attention-type=sliding_gated \
  --chinchilla-multiple=4.0 \
  --sequence-length=8192 \
  --sliding-window-size=4096 \
  --microbatch-discount=1.0 \
  --train-data='/data/user/dicksonb/data/nanochat/tokenized/*.npy' \
  --eval-data-dir=/data/user/dicksonb/data \
  --max-gpus=2 \
  --wandb-entity=iu-cogai \
  --project=attn-scaling-ladder \
  --optimizer=muon
```

For HC variants, add `--hyper-connections=4`. For other attention types, change `--attention-type=` to `vanilla_gated` or `hybrid_gated_deltanet`.

The valid `--attention-type` choices on the inner script are: `sliding_gated`, `hybrid_gated_deltanet`, `vanilla_gated`. (The plain `vanilla` choice was removed.)

## Layer-pattern reference

| variant | layer pattern |
|---|---|
| `vanilla_gated` | every layer is full softmax (over the 8192-token sequence) + headwise output gate |
| `sliding_gated` | per 4-layer block: 3× sliding-window (window=4096) + 1× full softmax (8192). Last layer forced to full. All attention layers have headwise gate. |
| `hybrid_gated_deltanet` | per 4-layer block: 3× GatedDeltaNet (linear attention) + 1× full softmax (8192). The full layers have headwise gate; the FLA layers have FLA's internal `use_gate=True`. |

Adding `+ HC=4` wraps each block's residual streams in `HyperConnectionStream` with 4 streams (Identity init).

## W&B requirement

All Slurm launchers require `WANDB_API_KEY` in the environment. They exit early with an error if it's missing.

```bash
export WANDB_API_KEY=...
sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh
```

## See also

- `src/scripts/modal/USAGE.md` — same 6 variants on Modal instead of Slurm.
- `src/scripts/train/ladder/wsds_attention_ladder.py` — the inner training script.
- `../../olmo_core/model_ladder/wsds_chinchilla_run_configurator.py` — the WSDS schedule + run configurator.
