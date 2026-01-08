#!/bin/bash

#SBATCH --account=cogneuroai
#SBATCH -J mla-l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition general
#SBATCH --output=./logs/mla/%j.txt
#SBATCH --error=./logs/mla/%j.err
#SBATCH --mail-type=None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00

echo "Job started at $(date)"
echo "Python: $(which python)"
echo "Node: $(hostname)"

python -c "print('*'*50)"

num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Number of GPUs: $num_gpus"

RUN_NAME="mla-l40s-${SLURM_JOB_ID}"
SAVE_FOLDER="./checkpoints/${RUN_NAME}"
DATA_PATH="../olmo-data/wiki/part-0-00000.npy"

torchrun --standalone --nproc_per_node=$num_gpus \
    src/scripts/train/mla-small.py train_single "$RUN_NAME" local \
    --trainer.save_folder="$SAVE_FOLDER" \
    --dataset.paths="[\"$DATA_PATH\"]" \
    --dataset.mix=null \
    --trainer.callbacks.wandb.enabled=true

python -c "print('*'*50)"
echo "Job finished at $(date)"
