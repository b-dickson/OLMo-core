#!/bin/bash

#SBATCH --account=cogneuroai
#SBATCH -J attn-isoflops-h100
#SBATCH --gres=gpu:H100:2
#SBATCH --mem=256G
#SBATCH --partition=general
#SBATCH --output=/data/user/dicksonb/logs/attention-scaling-isoflops/%A_%a.txt
#SBATCH --error=/data/user/dicksonb/logs/attention-scaling-isoflops/%A_%a.err
#SBATCH --mail-type=None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
# Set array range when submitting: sbatch --array=1-48 src/scripts/lair/slurm_isoflops_attention_scaling_h100.sh
# Optional override: MICROBATCH_DISCOUNT=<float> (default 1.0)

CONFIG_FILE="${CONFIG_FILE:-src/scripts/lair/isoflops_attention_scaling_config.txt}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

# Read configuration for this array task (skip comments and blank lines)
CONFIG_LINE=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$CONFIG_LINE" ]; then
    echo "Error: No configuration found for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

ATTENTION_TYPE=$(echo "$CONFIG_LINE" | awk '{print $1}')
SIZE=$(echo "$CONFIG_LINE" | awk '{print $2}')
TARGET_FLOPS=$(echo "$CONFIG_LINE" | awk '{print $3}')
CONFIG_MICROBATCH_DISCOUNT=$(echo "$CONFIG_LINE" | awk '{print $4}')

if [ -z "${TARGET_FLOPS}" ]; then
    echo "Error: missing target FLOPs in config line: ${CONFIG_LINE}"
    exit 1
fi

DATA_DIR="${DATA_DIR:-/data/user/dicksonb/data/nanochat/tokenized/*.npy}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/data/user/dicksonb/data}"
MICROBATCH_DISCOUNT="${MICROBATCH_DISCOUNT:-1.0}"
if [ -z "${MICROBATCH_DISCOUNT+x}" ] && [ -n "${CONFIG_MICROBATCH_DISCOUNT}" ]; then
    MICROBATCH_DISCOUNT="${CONFIG_MICROBATCH_DISCOUNT}"
fi
MICROBATCH_DISCOUNT="${MICROBATCH_DISCOUNT:-1.0}"
LADDER_PROJECT="${LADDER_PROJECT:-attn-scaling-isoflops}"
LADDER_WANDB_ENTITY="${LADDER_WANDB_ENTITY:-iu-cogai}"
export LADDER_ROOT_DIR="${LADDER_ROOT_DIR:-/data/user/dicksonb/checkpoints}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Error: WANDB_API_KEY is not set. Set this environment variable before launching W&B runs."
  exit 1
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${LADDER_WANDB_ENTITY}"
export WANDB_PROJECT="${LADDER_PROJECT}"

echo "Job started at $(date)"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Configuration: ${CONFIG_LINE}"
echo "Python: $(which python)"
echo "Node: $(hostname)"

${PYTHON_BIN} -c "print('*'*50)"

num_gpus=$(${PYTHON_BIN} -c "import torch; print(torch.cuda.device_count())")
[ -n "$num_gpus" ] || num_gpus=0
echo "Number of GPUs: $num_gpus"
if [ "$num_gpus" -ne 2 ]; then
    echo "Warning: expected 2 GPUs for H100 launcher, found ${num_gpus}."
fi
FORCE_MIN_WORLD_SIZE="${FORCE_MIN_WORLD_SIZE:-$num_gpus}"

RUN_NAME="${ATTENTION_TYPE}-${SIZE}-${TARGET_FLOPS}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Attention type: ${ATTENTION_TYPE}"
echo "Size: ${SIZE}"
echo "Target FLOPs: ${TARGET_FLOPS}"
echo "Microbatch discount: ${MICROBATCH_DISCOUNT}"
echo "Forced min world size: ${FORCE_MIN_WORLD_SIZE}"
echo "Train data: ${DATA_DIR}"
echo "Eval data root: ${EVAL_DATA_DIR}"
echo "W&B project: ${LADDER_PROJECT}"
echo "W&B entity: ${LADDER_WANDB_ENTITY}"
echo "Ladder root: ${LADDER_ROOT_DIR}"
echo "Run name: ${RUN_NAME}"
mkdir -p "/data/user/dicksonb/logs/attention-scaling-isoflops"

FORCE_MIN_WORLD_SIZE="${FORCE_MIN_WORLD_SIZE}" \
${PYTHON_BIN} -m torch.distributed.run --standalone --nproc-per-node=$num_gpus \
    src/scripts/train/ladder/isoflops_attention_ladder.py \
    run \
    --name="${RUN_NAME}" \
    --size=${SIZE} \
    --attention-type="${ATTENTION_TYPE}" \
    --target-flops=${TARGET_FLOPS} \
    --microbatch-discount=${MICROBATCH_DISCOUNT} \
    --train-data="${DATA_DIR}" \
    --eval-data-dir="${EVAL_DATA_DIR}" \
    --sequence-length=2048 \
    --max-gpus=${num_gpus} \
    --wandb-entity="${LADDER_WANDB_ENTITY}" \
    --project="${LADDER_PROJECT}"

${PYTHON_BIN} -c "print('*'*50)"
echo "Job finished at $(date)"
