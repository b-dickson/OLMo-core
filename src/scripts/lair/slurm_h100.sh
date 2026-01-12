#!/bin/bash

#SBATCH --account=cogneuroai
#SBATCH -J mla-h100
#SBATCH --gres=gpu:H100
#SBATCH --partition general
#SBATCH --output=../logs/mla/%j.txt
#SBATCH --error=../logs/mla/%j.err
#SBATCH --mail-type=None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

echo "Job started at $(date)"
echo "Python: $(which python)"
echo "Node: $(hostname)"

python -c "print('*'*50)"

num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Number of GPUs: $num_gpus"

RUN_NAME="mla-h100-${SLURM_JOB_ID}"

torchrun --standalone --nproc-per-node=$num_gpus \
  src/examples/mla/train.py "$RUN_NAME"
    trainer.save_folder=../OLMo-checkpoints/$RUN_NAME

python -c "print('*'*50)"
echo "Job finished at $(date)"
