#!/bin/bash

#SBATCH --account=cogneuroai
#SBATCH -J attn-ladder
#SBATCH --partition general
#SBATCH --output=/data/user/dicksonb/logs/attention-scaling-ladder/%A_%a.txt
#SBATCH --error=/data/user/dicksonb/logs/attention-scaling-ladder/%A_%a.err
#SBATCH --mail-type=None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
# Set array range when submitting: sbatch --array=1-12 [resource flags] src/scripts/lair/slurm_attention_scaling.sh

CONFIG_FILE="${CONFIG_FILE:-src/scripts/lair/attention_scaling_config.txt}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

# Read configuration for this array task (skip comments and blank lines)
CONFIG_LINE=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$CONFIG_LINE" ]; then
    echo "Error: No configuration found for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

read -r ATTENTION_TYPE SIZE CHINCHILLA_MULTIPLE CONFIG_SEQUENCE_LENGTH CONFIG_MICROBATCH_DISCOUNT <<< "$CONFIG_LINE"
BATCH_SIZE_MULTIPLIER="${BATCH_SIZE_MULTIPLIER:-1.0}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"

# If the 4th token is numeric, treat it as sequence length; otherwise treat it as legacy
# microbatch discount override.
if [[ -n "${CONFIG_SEQUENCE_LENGTH}" && "${CONFIG_SEQUENCE_LENGTH}" =~ ^[0-9]+$ ]]; then
    SEQUENCE_LENGTH="${CONFIG_SEQUENCE_LENGTH}"
elif [ -n "${CONFIG_SEQUENCE_LENGTH}" ] && [ -z "${CONFIG_MICROBATCH_DISCOUNT}" ]; then
    CONFIG_MICROBATCH_DISCOUNT="${CONFIG_SEQUENCE_LENGTH}"
fi

if [ -z "${MICROBATCH_DISCOUNT+x}" ] && [ -n "${CONFIG_MICROBATCH_DISCOUNT}" ]; then
    MICROBATCH_DISCOUNT="${CONFIG_MICROBATCH_DISCOUNT}"
fi
MICROBATCH_DISCOUNT="${MICROBATCH_DISCOUNT:-1.0}"

if [ "$CHINCHILLA_MULTIPLE" != "4.0" ] && [ "$CHINCHILLA_MULTIPLE" != "4" ]; then
    echo "Error: config must specify chinchilla multiple=4 or 4.0. Found '${CHINCHILLA_MULTIPLE}'"
    exit 1
fi
CHINCHILLA_MULTIPLE="4.0"

DATA_DIR="${DATA_DIR:-/data/user/dicksonb/data/nanochat/tokenized/*.npy}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/data/user/dicksonb/data}"
LADDER_PROJECT="${LADDER_PROJECT:-attn-scaling-ladder}"
LADDER_WANDB_ENTITY="${LADDER_WANDB_ENTITY:-iu-cogai}"
RUN_OPTIMIZER="${RUN_OPTIMIZER:-muon}"
export LADDER_ROOT_DIR="/data/user/dicksonb/checkpoints"

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
if [ "$num_gpus" -eq 0 ]; then
    echo "Error: No GPUs detected. Launch with a GPU allocation."
    echo "Examples:"
    echo "  sbatch --array=1-12 --gres=gpu:h100:2 src/scripts/lair/slurm_attention_scaling.sh"
    echo "  sbatch --array=1-12 --gres=gpu:l40s:8 src/scripts/lair/slurm_attention_scaling.sh"
    exit 1
fi

RUN_NAME="${ATTENTION_TYPE}-${CHINCHILLA_MULTIPLE}x-${SIZE}_seq${SEQUENCE_LENGTH}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_OPTIMIZER}"

echo "Attention type: ${ATTENTION_TYPE}"
echo "Size: ${SIZE}"
echo "Chinchilla multiple: ${CHINCHILLA_MULTIPLE}"
echo "Batch size multiplier: ${BATCH_SIZE_MULTIPLIER}"
echo "Microbatch discount: ${MICROBATCH_DISCOUNT}"
echo "Train data: ${DATA_DIR}"
echo "Eval data root: ${EVAL_DATA_DIR}"
echo "Optimizer: ${RUN_OPTIMIZER}"
echo "W&B project: ${LADDER_PROJECT}"
echo "W&B entity: ${LADDER_WANDB_ENTITY}"
echo "Ladder root: ${LADDER_ROOT_DIR}"
echo "Run name: ${RUN_NAME}"
# Create directories
mkdir -p "/data/user/dicksonb/logs/attention-scaling-ladder"

${PYTHON_BIN} -m torch.distributed.run --standalone --nproc-per-node=$num_gpus \
    src/scripts/train/ladder/wsds_attention_ladder.py \
    run \
    --name="${RUN_NAME}" \
    --size=${SIZE} \
    --attention-type="${ATTENTION_TYPE}" \
    --chinchilla-multiple=${CHINCHILLA_MULTIPLE} \
    --batch-size-multiplier=${BATCH_SIZE_MULTIPLIER} \
    --microbatch-discount=${MICROBATCH_DISCOUNT} \
    --train-data="${DATA_DIR}" \
    --eval-data-dir="${EVAL_DATA_DIR}" \
    --sequence-length=${SEQUENCE_LENGTH} \
    --max-gpus=${num_gpus} \
    --wandb-entity="${LADDER_WANDB_ENTITY}" \
    --project="${LADDER_PROJECT}" \
    --optimizer="${RUN_OPTIMIZER}"

${PYTHON_BIN} -c "print('*'*50)"
echo "Job finished at $(date)"
