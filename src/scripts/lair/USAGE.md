## Attention-Scaling Ladders

This folder has two Slurm-driven ladders:

- WSDS 4x-Chinchilla ladder:
  - Script: `src/scripts/train/ladder/wsds_attention_ladder.py`
  - Config: `src/scripts/lair/attention_scaling_config.txt`
  - Config columns: `attention_type size chinchilla_multiple`
- IsoFLOPs fixed-budget ladder:
  - Script: `src/scripts/train/ladder/isoflops_attention_ladder.py`
  - Config: `src/scripts/lair/isoflops_attention_scaling_config.txt`
  - Config columns: `attention_type size target_flops`

On this cluster, Slurm GRES names are case-sensitive: `H100`, `L40S`.

## WSDS (4x Chinchilla)

Launch full array:

```bash
sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_h100.sh
sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_l40s.sh
```

Launch a single row (example: row 7 = `hybrid_gated_deltanet 60M 4.0`):

```bash
sbatch --array=7 src/scripts/lair/slurm_attention_scaling_h100.sh
```

Defaults:

- W&B project: `attn-scaling-ladder`
- W&B entity: `iu-cogai`
- Optimizer: `muon` (override with `RUN_OPTIMIZER=skipstep_adamw`)
- Train data: `/data/user/dicksonb/data/nanochat/tokenized/*.npy`
- Eval data root: `/data/user/dicksonb/data`
- Ladder root: `/data/user/dicksonb/checkpoints`
- H100 launcher: `2x H100`, `MICROBATCH_DISCOUNT=1.0`
- L40S launcher: `8x L40S`, `MICROBATCH_DISCOUNT=2.0`

Common overrides:

```bash
RUN_OPTIMIZER=skipstep_adamw sbatch --array=7 src/scripts/lair/slurm_attention_scaling_h100.sh
MICROBATCH_DISCOUNT=1.5 sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_h100.sh
MICROBATCH_DISCOUNT=2.0 sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_l40s.sh
LADDER_PROJECT=my_project sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_h100.sh
LADDER_WANDB_ENTITY=my_entity sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_h100.sh
```

Generic launcher (you provide GPU flags at submit time):

```bash
sbatch --array=1-12 --gres=gpu:H100:2 src/scripts/lair/slurm_attention_scaling.sh
sbatch --array=1-12 --gres=gpu:L40S:8 src/scripts/lair/slurm_attention_scaling.sh
```

Manual local launch example:

```bash
torchrun --standalone --nproc-per-node=2 src/scripts/train/ladder/wsds_attention_ladder.py \
  run \
  --name=my-run \
  --size=190M \
  --attention-type=sliding_gated \
  --chinchilla-multiple=4.0 \
  --microbatch-discount=1.0 \
  --train-data=/data/user/dicksonb/data/nanochat/tokenized/*.npy \
  --eval-data-dir=/data/user/dicksonb/data \
  --sequence-length=2048 \
  --optimizer=muon
```

## IsoFLOPs (fixed budget)

Launch full array:

```bash
sbatch --array=1-48 src/scripts/lair/slurm_isoflops_attention_scaling_h100.sh
sbatch --array=1-48 src/scripts/lair/slurm_isoflops_attention_scaling_l40s.sh
```

Defaults:

- W&B project: `attn-scaling-isoflops`
- W&B entity: `iu-cogai`
- Train data: `/data/user/dicksonb/data/nanochat/tokenized/*.npy`
- Eval data root: `/data/user/dicksonb/data`
- Ladder root: `/data/user/dicksonb/checkpoints`
- H100 launcher: `2x H100`, `MICROBATCH_DISCOUNT=1.0`
- L40S launcher: `8x L40S`, `MICROBATCH_DISCOUNT=1.5`

Common overrides:

```bash
MICROBATCH_DISCOUNT=2.0 sbatch --array=1-48 src/scripts/lair/slurm_isoflops_attention_scaling_l40s.sh
LADDER_PROJECT=my_isoflops_project sbatch --array=1-48 src/scripts/lair/slurm_isoflops_attention_scaling_h100.sh
LADDER_WANDB_ENTITY=my_entity sbatch --array=1-48 src/scripts/lair/slurm_isoflops_attention_scaling_h100.sh
```

Manual local launch example:

```bash
torchrun --standalone --nproc-per-node=2 src/scripts/train/ladder/isoflops_attention_ladder.py \
  run \
  --name=my-isoflops-run \
  --size=190M \
  --attention-type=hybrid_gated_deltanet \
  --target-flops=4.64e18 \
  --microbatch-discount=1.0 \
  --train-data=/data/user/dicksonb/data/nanochat/tokenized/*.npy \
  --eval-data-dir=/data/user/dicksonb/data \
  --sequence-length=2048
```

## W&B requirement

All Slurm launchers require `WANDB_API_KEY` in the environment.
