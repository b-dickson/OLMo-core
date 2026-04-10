#!/bin/bash

#SBATCH --account=cogneuroai
#SBATCH -J attn-sliding-gated-h100
#SBATCH --gres=gpu:H100:2
#SBATCH --mem=256G
#SBATCH --partition general
#SBATCH --output=/data/user/dicksonb/logs/attn-ladder-sliding-gated/%A_%a.txt
#SBATCH --error=/data/user/dicksonb/logs/attn-ladder-sliding-gated/%A_%a.err
#SBATCH --mail-type=None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
# Submit with: sbatch --array=1-4 src/scripts/lair/slurm_wsds_sliding_gated_h100.sh

CONFIG_FILE="${CONFIG_FILE:-src/scripts/lair/wsds_sliding_gated.txt}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CONFIG_LINE=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$CONFIG_LINE" ]; then
    echo "Error: No configuration found for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi
read -r ATTENTION_TYPE SIZE CHINCHILLA_MULTIPLE CONFIG_SEQUENCE_LENGTH <<< "$CONFIG_LINE"

if [ "$CHINCHILLA_MULTIPLE" != "4.0" ] && [ "$CHINCHILLA_MULTIPLE" != "4" ]; then
    echo "Error: config must specify chinchilla multiple=4 or 4.0. Found '${CHINCHILLA_MULTIPLE}'"
    exit 1
fi
CHINCHILLA_MULTIPLE="4.0"

DATA_DIR="${DATA_DIR:-/data/user/dicksonb/data/nanochat/tokenized/*.npy}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/data/user/dicksonb/data}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-${CONFIG_SEQUENCE_LENGTH:-8192}}"
BATCH_SIZE_MULTIPLIER="${BATCH_SIZE_MULTIPLIER:-1.0}"
MICROBATCH_DISCOUNT="${MICROBATCH_DISCOUNT:-1.0}"
SLIDING_WINDOW_SIZE="${SLIDING_WINDOW_SIZE:-4096}"
LADDER_PROJECT="${LADDER_PROJECT:-attn-scaling-ladder}"
LADDER_WANDB_ENTITY="${LADDER_WANDB_ENTITY:-iu-cogai}"
RUN_OPTIMIZER="${RUN_OPTIMIZER:-muon}"
export LADDER_ROOT_DIR="${LADDER_ROOT_DIR:-/data/user/dicksonb/checkpoints}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Error: WANDB_API_KEY is not set."
  exit 1
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${LADDER_WANDB_ENTITY}"
export WANDB_PROJECT="${LADDER_PROJECT}"

echo "Job started at $(date)"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Configuration: ${CONFIG_LINE}"
echo "Node: $(hostname)"

num_gpus=$(${PYTHON_BIN} -c "import torch; print(torch.cuda.device_count())")
[ -n "$num_gpus" ] || num_gpus=0
echo "Number of GPUs: $num_gpus"
if [ "$num_gpus" -ne 2 ]; then
    echo "Warning: expected 2 GPUs for H100 launcher, found ${num_gpus}."
fi
FORCE_MIN_WORLD_SIZE="${FORCE_MIN_WORLD_SIZE:-$num_gpus}"

RUN_NAME="${RUN_NAME:-${ATTENTION_TYPE}-${CHINCHILLA_MULTIPLE}x-${SIZE}_seq${SEQUENCE_LENGTH}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_OPTIMIZER}}"

echo "Variant: sliding_gated (no HC); 3:1 sliding(${SLIDING_WINDOW_SIZE}):full(${SEQUENCE_LENGTH}) per 4-layer block"
echo "Attention type: ${ATTENTION_TYPE}"
echo "Size: ${SIZE}"
echo "Chinchilla multiple: ${CHINCHILLA_MULTIPLE}"
echo "Sequence length: ${SEQUENCE_LENGTH}"
echo "Sliding window size: ${SLIDING_WINDOW_SIZE}"
echo "Optimizer: ${RUN_OPTIMIZER}"
echo "Run name: ${RUN_NAME}"
mkdir -p "/data/user/dicksonb/logs/attn-ladder-sliding-gated"

FORCE_MIN_WORLD_SIZE="${FORCE_MIN_WORLD_SIZE}" \
${PYTHON_BIN} -m torch.distributed.run --standalone --nproc-per-node=$num_gpus \
    src/scripts/train/ladder/wsds_attention_ladder.py \
    run \
    --name="${RUN_NAME}" \
    --size=${SIZE} \
    --attention-type="${ATTENTION_TYPE}" \
    --chinchilla-multiple=${CHINCHILLA_MULTIPLE} \
    --sequence-length=${SEQUENCE_LENGTH} \
    --batch-size-multiplier=${BATCH_SIZE_MULTIPLIER} \
    --microbatch-discount=${MICROBATCH_DISCOUNT} \
    --sliding-window-size=${SLIDING_WINDOW_SIZE} \
    --train-data="${DATA_DIR}" \
    --eval-data-dir="${EVAL_DATA_DIR}" \
    --max-gpus=${num_gpus} \
    --wandb-entity="${LADDER_WANDB_ENTITY}" \
    --project="${LADDER_PROJECT}" \
    --optimizer="${RUN_OPTIMIZER}"

echo "Job finished at $(date)"
