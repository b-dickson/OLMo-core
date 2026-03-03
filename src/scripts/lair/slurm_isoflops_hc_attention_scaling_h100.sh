#!/bin/bash

#SBATCH --account=cogneuroai
#SBATCH -J attn-isoflops-hc-h100
#SBATCH --gres=gpu:H100:2
#SBATCH --mem=256G
#SBATCH --partition=general
#SBATCH --output=/data/user/dicksonb/logs/attention-scaling-isoflops-hc/%A_%a.txt
#SBATCH --error=/data/user/dicksonb/logs/attention-scaling-isoflops-hc/%A_%a.err
#SBATCH --mail-type=None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
# Set array range when submitting: sbatch --array=1-16 src/scripts/lair/slurm_isoflops_hc_attention_scaling_h100.sh

CONFIG_FILE="${CONFIG_FILE:-src/scripts/lair/isoflops_hc_attention_scaling_config.txt}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HC_STREAMS="${HC_STREAMS:-4}"

# Read configuration for this array task (skip comments and blank lines)
CONFIG_LINE=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$CONFIG_LINE" ]; then
    echo "Error: No configuration found for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

read -r ATTENTION_TYPE SIZE TARGET_FLOPS CONFIG_SEQUENCE_LENGTH CONFIG_MICROBATCH_DISCOUNT <<< "$CONFIG_LINE"

if [ -z "${TARGET_FLOPS}" ]; then
    echo "Error: missing target FLOPs in config line: ${CONFIG_LINE}"
    exit 1
fi

DATA_DIR="${DATA_DIR:-/data/user/dicksonb/data/nanochat/tokenized/*.npy}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/data/user/dicksonb/data}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
SLIDING_WINDOW_SIZE="${SLIDING_WINDOW_SIZE:-1024}"

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
LADDER_PROJECT="${LADDER_PROJECT:-attn-scaling-isoflops-hc}"
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

RUN_NAME="hc${HC_STREAMS}-${ATTENTION_TYPE}-${SIZE}-${TARGET_FLOPS}_seq${SEQUENCE_LENGTH}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Attention type: ${ATTENTION_TYPE}"
echo "Size: ${SIZE}"
echo "Target FLOPs: ${TARGET_FLOPS}"
echo "Hyper connections: ${HC_STREAMS} streams"
echo "Microbatch discount: ${MICROBATCH_DISCOUNT}"
echo "Sliding window size: ${SLIDING_WINDOW_SIZE}"
echo "Sequence length: ${SEQUENCE_LENGTH}"
echo "Forced min world size: ${FORCE_MIN_WORLD_SIZE}"
echo "Train data: ${DATA_DIR}"
echo "Eval data root: ${EVAL_DATA_DIR}"
echo "W&B project: ${LADDER_PROJECT}"
echo "W&B entity: ${LADDER_WANDB_ENTITY}"
echo "Ladder root: ${LADDER_ROOT_DIR}"
echo "Run name: ${RUN_NAME}"
mkdir -p "/data/user/dicksonb/logs/attention-scaling-isoflops-hc"

FORCE_MIN_WORLD_SIZE="${FORCE_MIN_WORLD_SIZE}" \
${PYTHON_BIN} -m torch.distributed.run --standalone --nproc-per-node=$num_gpus \
    src/scripts/train/ladder/isoflops_attention_ladder.py \
    run \
    --name="${RUN_NAME}" \
    --size=${SIZE} \
    --attention-type="${ATTENTION_TYPE}" \
    --target-flops=${TARGET_FLOPS} \
    --hyper-connections=${HC_STREAMS} \
    --microbatch-discount=${MICROBATCH_DISCOUNT} \
    --sequence-length=${SEQUENCE_LENGTH} \
    --sliding-window-size=${SLIDING_WINDOW_SIZE} \
    --train-data="${DATA_DIR}" \
    --eval-data-dir="${EVAL_DATA_DIR}" \
    --max-gpus=${num_gpus} \
    --wandb-entity="${LADDER_WANDB_ENTITY}" \
    --project="${LADDER_PROJECT}"

${PYTHON_BIN} -c "print('*'*50)"
echo "Job finished at $(date)"
