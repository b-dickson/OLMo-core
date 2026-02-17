#!/bin/bash

#SBATCH --account=cogneuroai
#SBATCH --job-name=tokenize-fineweb
#SBATCH --partition=general
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/data/user/dicksonb/logs/tokenize_%j.log
#SBATCH --error=/data/user/dicksonb/logs/tokenize_%j.err

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

cd /data/project/cogneuroai/dicksonb/ai2/OLMo-core

# Activate venv
source .venv/bin/activate

# Input/output paths
INPUT_DIR="/data/user/dicksonb/data/nanochat/jsonl"
OUTPUT_DIR="/data/user/dicksonb/data/nanochat/tokenized"

echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Count input files
NUM_FILES=$(ls -1 ${INPUT_DIR}/*.jsonl 2>/dev/null | wc -l)
echo "Found ${NUM_FILES} JSONL files to tokenize"
echo ""

# Run dolma tokenization
dolma tokens \
    --documents "${INPUT_DIR}/*.jsonl" \
    --destination "${OUTPUT_DIR}" \
    --tokenizer.name_or_path allenai/dolma2-tokenizer \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --dtype uint32 \
    --processes ${SLURM_CPUS_PER_TASK}

echo ""
echo "Job finished at $(date)"
