## Attention-scaling ladder (current)

Run attention scaling experiments at fixed 4x Chinchilla compute using
`wsds_attention_ladder.py`.

### One-time config

Config:
  `src/scripts/lair/attention_scaling_config.txt`
`attention_type  size  4.0  [batch-size-multiplier]`

Each line can optionally include an extra `batch-size-multiplier` value.

### Launcher

Examples:
```bash
sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_h100.sh
sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_l40s.sh
MICROBATCH_DISCOUNT=2.0 sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_l40s.sh
MICROBATCH_DISCOUNT=1.5 sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_h100.sh
LADDER_PROJECT=my_project sbatch --array=1-12 src/scripts/lair/slurm_attention_scaling_h100.sh
```

Default W&B project is `attention_scaling_wsds` (override with `LADDER_PROJECT`).
On this cluster, Slurm GRES names are case-sensitive (`H100`, `L40S`).

### Microbatch discount policy

Recommended defaults:
- H100 (2x80GB): `MICROBATCH_DISCOUNT=1.0`
- L40 (8x40GB): `MICROBATCH_DISCOUNT=1.5`

If a specific L40 run still OOMs, retry that run with:
- `MICROBATCH_DISCOUNT=2.0`

### One-off manual launch

```bash
  torchrun --standalone --nproc-per-node=2 src/scripts/train/ladder/wsds_attention_ladder.py \
    run \
    --name=my-run \
    --size=190M \
    --attention-type=sliding_gated \
    --chinchilla-multiple=4.0 \
    --batch-size-multiplier=1.0 \
    --microbatch-discount=1.0 \
    --train-data=/data/user/dicksonb/data/nanochat/tokenized/*.npy \
    --eval-data-dir=/data/user/dicksonb/data \
    --sequence-length=2048
```
