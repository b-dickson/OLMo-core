## Modal attention-ladder launchers

This directory contains Modal equivalents for the `lair` array launchers.

Both launchers invoke `modal run`; `--dry-run` is handled inside `attention_ladder_modal.py` and exits before submission.

- `launch_wsds_attention_scaling.sh` -> WSDS (4x-Chinchilla) ladder
- `launch_isoflops_attention_scaling.sh` -> IsoFLOPs fixed-budget ladder
- `attention_ladder_modal.py` -> shared launcher implementation

Defaults in `attention_ladder_modal.py`:

- WSDS config: `src/scripts/lair/attention_scaling_config.txt`
- IsoFLOPs config: `src/scripts/lair/isoflops_attention_scaling_config.txt`
- Default task range: all rows in the selected config
- Modal GPU default: `--gpus` (also controlled by `args.gpus`; default 8)
- W&B default project:
  - WSDS: `attn-scaling-ladder`
  - IsoFLOPs: `attn-scaling-isoflops`
- Default `WANDB_ENTITY`: `iu-cogai`
- Supports AI2-style `r2://` data paths.
- Uses Modal secret: `r2-creds` (required for `R2_*` and AWS credentials unless you only use profile-based auth).

## Dry run for one WSDS row (single task)

```bash
src/scripts/modal/launch_wsds_attention_scaling.sh \
  --array=7 \
  --dry-run \
  --train-data r2://llm-data/nanochat/tokenized/*.npy \
  --eval-data-dir r2://llm-data/eval-data \
  --project attn-scaling-ladder \
  --wandb-entity your-entity
```
With `--dry-run`, no remote task is launched; command lines are printed for inspection.

## Dry run for one IsoFLOPs row

```bash
src/scripts/modal/launch_isoflops_attention_scaling.sh \
  --array=9 \
  --dry-run \
  --train-data r2://llm-data/nanochat/tokenized/*.npy \
  --eval-data-dir r2://llm-data/eval-data
```

## Launch WSDS rows with 8x H100

```bash
src/scripts/modal/launch_wsds_attention_scaling.sh \
  --array=1-12 \
  --gpus 8 \
  --train-data r2://llm-data/nanochat/tokenized/*.npy \
  --eval-data-dir r2://llm-data/eval-data \
  --project attn-scaling-ladder \
  --wandb-entity your-entity \
  --ladder-root-dir r2://llm-data/checkpoints/attn-scaling-ladder
```

## Docker image

The launcher pulls a pre-built image from `ghcr.io/allenai/olmo-core:latest` by default.
This image includes all heavy dependencies (PyTorch, flash-attn, FLA, etc.) so there is no compilation at launch time.

To use a custom image, set the `OLMO_DOCKER_IMAGE` environment variable:

```bash
OLMO_DOCKER_IMAGE=ghcr.io/your-org/olmo-core:custom \
  src/scripts/modal/launch_wsds_attention_scaling.sh --array=7 --gpus 8
```

Your local source code is always mounted into the container via `.add_local_dir()`, so code changes do not require rebuilding the image.

## Notes

- `--array` supports values like `5`, `1,3,7`, and `1-12`.
- Modal GPU count is controlled by `--gpus` (defaults to 8) and is reused for both `torchrun` and the `gpu` resource request.
- For R2, set:
  - `R2_ENDPOINT_URL=https://8fc7fa235e208d33c0d94a6130384f81.r2.cloudflarestorage.com`
  - `R2_PROFILE` to the AWS CLI profile name containing R2 credentials
  - Ensure your profile can read the `llm-data` bucket.
- Make sure W&B and AWS credentials are available to your Modal environment (env passthrough keys include:
  `WANDB_MODE`, and optional `AWS_SHARED_CREDENTIALS_FILE`, `AWS_CONFIG_FILE`).

### Modal secret setup (recommended)

Create once from your local env or from an exported script:

```bash
source ~/.config/modal/r2_local_env.sh
modal secret create r2-creds \\
  R2_ENDPOINT_URL="$R2_ENDPOINT_URL" \\
  R2_PROFILE="$R2_PROFILE" \\
  AWS_PROFILE="$AWS_PROFILE" \\
  AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \\
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \\
  AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \\
  WANDB_API_KEY="$WANDB_API_KEY" \\
  HF_TOKEN="$HF_TOKEN"
```

The launcher uses `secrets=[modal.Secret.from_name("r2-creds")]`, so you do not need to pass these credentials through local environment at runtime.
