## Modal launcher for the WSDS attention ladder

This directory contains the Modal-based launcher for the WSDS (4x-Chinchilla) attention ladder. It mirrors the Slurm launchers in `src/scripts/lair/` but runs the same training jobs as Modal containers instead of as Slurm allocations on lair.

Files:

| file | role |
|---|---|
| `attention_ladder_modal.py` | The Modal launcher. Reads a `wsds_*.txt` config file and submits one Modal task per row. |
| `launch_wsds_attention_scaling.sh` | Thin shell wrapper around `modal run`. Pass-through; no flags hardcoded. |
| `smoke_test.py` | Standalone Modal credential check (GPU visibility, R2 access, W&B/HF auth, eval-data mount). |
| `r2_local_env.sh` | Helper for sourcing R2 credentials into your local env before creating the Modal secret. |

The launcher only supports WSDS — iso-flops mode was removed. There is no `--mode` flag.

## Quick reference: launching the 6 variants

You can use `modal run` directly or the shell wrapper — both are equivalent. Substitute the variant config you want.

```bash
# Non-HC variants (no --hyper-connections needed)
modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_vanilla_gated.txt          --array=1-4

modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_sliding_gated.txt          --array=1-4

modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_hybrid_gated_deltanet.txt  --array=1-4

# HC variants (must pass --hyper-connections=4)
modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_vanilla_gated_hc.txt          --array=1-4 --hyper-connections=4

modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_sliding_gated_hc.txt          --array=1-4 --hyper-connections=4

modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_hybrid_gated_deltanet_hc.txt  --array=1-4 --hyper-connections=4
```

Or via the shell wrapper:

```bash
bash src/scripts/modal/launch_wsds_attention_scaling.sh \
  --config-file=src/scripts/lair/wsds_sliding_gated.txt --array=1-4
```

## Concurrency model

`--array=1-4` fires off **4 simultaneous Modal containers**, not 4 sequential ones:

1. The launcher loops over the array indices and calls `run_training_command.spawn(...)` for each. `.spawn()` is non-blocking — Modal allocates a fresh container immediately and returns a `FunctionCall` handle.
2. After all submissions, the launcher iterates the handles in order and calls `.get()` on each. `.get()` blocks until that specific call finishes.
3. Each container receives its own `--gpu-type:--gpus` allocation. With `--gpu-type=H100 --gpus=8 --array=1-4` you get **4 × 8 = 32 H100s** running concurrently.

Failure semantics: if task 2 fails, the launcher hits `.get()` for task 2 first (in submission order), re-raises, and exits. **The other in-flight tasks on Modal are NOT cancelled by the local exit** — they keep running and burning credits until they finish on their own. Use `modal app stop` to kill them if needed.

Modal account quotas can serialize parts of an array if you exceed concurrent-resource caps; the excess `.spawn()` calls queue and start as containers free up.

## Defaults baked into the launcher

| flag | default | notes |
|---|---|---|
| `--config-file` | `src/scripts/lair/wsds_sliding_gated.txt` | Pick one of the 6 `wsds_*.txt` files. |
| `--gpus` | `8` | GPUs per Modal container. |
| `--gpu-type` | `H100` | Modal GPU type. Other valid choices: `B200`, `L4`, `T4`, etc. |
| `--train-data` | `r2://llm-data/nanochat/tokenized/*.npy` | Cloudflare R2 path; expanded via boto3 inside the container. |
| `--eval-data-dir` | `/data` | Bundled eval data is mounted at `/data/eval-data` (parent is `/data`). |
| `--project` | `attn-scaling-ladder` | W&B project. |
| `--wandb-entity` | `iu-cogai` | W&B entity. |
| `--sequence-length` | (from config row) | Override the row's seq length if needed. |
| `--sliding-window-size` | `4096` | Used by `sliding_gated` only. |
| `--hyper-connections` | `0` | Set to `4` for HC variants. `0` = disabled. |
| `--optimizer` | `muon` | Pass `skipstep_adamw` to switch. |
| `--ladder-root-dir` | `r2://llm-data/checkpoints` | Maps to `LADDER_ROOT_DIR` env var. |

## B200 with no gradient accumulation

The `--no-grad-accum` flag sets `rank_microbatch_size = global_batch_size / dp_world_size`, so the full batch is processed in one step with no gradient accumulation. The number of GPUs needed depends on model size:

```bash
# 60M vanilla gated on 1x B200, no grad accum
modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_vanilla_gated.txt \
  --array=1 --gpus=1 --gpu-type=B200 --no-grad-accum

# 100M vanilla gated on 2x B200, no grad accum
modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_vanilla_gated.txt \
  --array=2 --gpus=2 --gpu-type=B200 --no-grad-accum
```

The Chinchilla-optimal batch size for 60M is ~220k tokens (~28 sequences of 8192), which fits on a single B200. The 100M model's batch (~327k tokens, 40 sequences) requires 2x B200.

For larger models (190M+), you will likely need more GPUs or gradient accumulation to hit the optimal batch size.

## Dry-run modes

The launcher supports two distinct dry-run modes:

- `--dry-run` — **purely local**. Builds the torchrun command line for each task, prints it, and exits without contacting Modal at all. Cost: $0. Useful for verifying the launcher's argument plumbing is what you expect.
- `--remote-dry-run` — **submits to Modal but tells the inner script to dry-run**. Pulls the docker image, mounts the repo, materializes the `r2-creds` secret, runs `uv pip install -e .[all]`, builds the model config (which exercises R2 reads to expand the data glob), generates the LR schedule plot, and exits without actually training. Useful for end-to-end credential and config-build validation. Costs a few cents per cheap-GPU container.

Example remote dry-run on a cheap GPU:

```bash
modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_sliding_gated.txt \
  --array=1 --remote-dry-run --gpu-type=L4 --gpus=1
```

(Use L4 or A10G for remote dry-runs — T4 lacks `sm_80+` and will fail to import flash-attn during the runtime install.)

## R2, W&B, and HF credentials

The launcher uses a Modal secret named `r2-creds` to provide all credentials inside the container. Create it once:

```bash
source ~/.config/modal/r2_local_env.sh   # or source src/scripts/modal/r2_local_env.sh
modal secret create r2-creds \
  R2_ENDPOINT_URL="$R2_ENDPOINT_URL" \
  R2_PROFILE="$R2_PROFILE" \
  AWS_PROFILE="$AWS_PROFILE" \
  AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
  WANDB_API_KEY="$WANDB_API_KEY" \
  HF_TOKEN="$HF_TOKEN"
```

Inside the container the launcher writes a minimal `~/.aws/config` and `~/.aws/credentials` for the R2 profile, so boto3 can resolve credentials via `Session(profile_name="r2")` (which is how the codebase reads R2 paths).

For R2 specifically: set `R2_ENDPOINT_URL` to your Cloudflare R2 endpoint (e.g. `https://<account>.r2.cloudflarestorage.com`) and make sure the profile can read the `llm-data` bucket.

To verify everything is wired up without launching a real training run, use the smoke test:

```bash
modal run src/scripts/modal/smoke_test.py
```

It runs on a T4:1 (the cheapest GPU tier), checks GPU visibility, secret env-vars, R2 listing via boto3, eval-data mount at `/data/eval-data`, W&B `wandb.Api().viewer`, and HF `whoami`. All in under a minute, costs sub-cent. You can also use `--remote-dry-run` against any variant on a cheap GPU for a fuller end-to-end check (see the "Dry-run modes" section above).

## Eval data mount

`attention_ladder_modal.py` mounts `<repo>/../datasets/eval-data` (resolved relative to `REPO_ROOT.parent`) into the container at `/data/eval-data`. If the local directory doesn't exist on the launching machine, the mount is silently skipped with a warning — `--remote-dry-run` will still work, but **real training runs will fail** because the inner script reads eval data from `/data/eval-data`.

To populate the local dir, rsync from lair (the perplexity files are ~26MB):

```bash
rsync -avz -e 'ssh -J bigred' lair:/data/user/dicksonb/data/eval-data ../datasets/
```

## Docker image

By default the launcher pulls `ghcr.io/allenai/olmo-core:latest`, which has all heavy dependencies (PyTorch, flash-attn, FLA, etc.) pre-installed. The local repo is mounted via `add_local_dir()` so code edits don't require image rebuilds. To use a custom image, set `OLMO_DOCKER_IMAGE`:

```bash
OLMO_DOCKER_IMAGE=ghcr.io/your-org/olmo-core:custom \
  modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_sliding_gated.txt --array=1-4
```

At container startup the launcher runs `uv pip install -e .[all] --system` from the mounted repo to pick up any local source edits.

### B200 image requirement

The upstream `ghcr.io/allenai/olmo-core:latest` image may not support B200 (Blackwell, sm_100). If you get errors like `NVIDIA B200 with CUDA capability sm_100 is not compatible` or `failed to open libnvrtc-builtins.so`, you need an image built from the current Dockerfile (PyTorch 2.10.0 + CUDA 12.8 + sm_100 support).

Build and push to your own registry:

```bash
sudo make docker-image
IMAGE_TAG=$(date "+tch2100cu128-%Y-%m-%d")
docker tag olmo-core:$IMAGE_TAG ghcr.io/b-dickson/olmo-core:latest
docker push ghcr.io/b-dickson/olmo-core:latest
```

Then use it:

```bash
OLMO_DOCKER_IMAGE=ghcr.io/b-dickson/olmo-core:latest \
  uv run modal run src/scripts/modal/attention_ladder_modal.py -- \
  --config-file=src/scripts/lair/wsds_vanilla_gated.txt \
  --array=1,2 --gpus=1 --gpu-type=B200 --no-grad-accum
```

If the image is in a private GHCR registry, create a Modal secret with your credentials:

```bash
modal secret create ghcr-creds \
  REGISTRY_USERNAME=<github-username> \
  REGISTRY_PASSWORD=<github-pat-with-read:packages>
```

The launcher automatically uses the `ghcr-creds` secret when pulling from `ghcr.io/b-dickson`.

## Notes

- `--array` accepts `5`, `1,3,5`, or `1-4`.
- Each variant config has 4 rows (60M / 100M / 190M / 370M), so the maximum useful array is `--array=1-4`.
- The HC vs non-HC distinction is set by the launcher's `--hyper-connections` flag, **not** by the config file content. The HC and non-HC config files for the same attention type have identical row contents.
- Run names include an `hc{N}-` prefix when `--hyper-connections=N>0`, so HC and non-HC W&B runs are easy to distinguish.
- Iso-flops mode was removed from this launcher entirely (no `--mode`, no `_build_isoflops_command`, no `IsoFlopsConfigRow`). If you need iso-flops back, the old configs and slurm scripts are in `../../../../archive/lair-2026-04-10/`.

## See also

- `src/scripts/lair/USAGE.md` — Slurm-on-lair launchers for the same 6 variants.
- `src/scripts/train/ladder/wsds_attention_ladder.py` — the inner training script.
- `src/scripts/modal/smoke_test.py` — credential validation without launching a real run.
