#!/bin/bash
#SBATCH --job-name=es-rlvr-eval
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH --partition=gpu

# ── Environment ──────────────────────────────────────────────────────────────
conda activate grpo

cd "$SLURM_SUBMIT_DIR"

# ── Parameters — edit as needed ───────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
VAL_PARQUET="data/val/math500.parquet"
OUTPUT_DIR="outputs/val_preds"
TP=4
DEVICES="0,1,2,3"
TEMPERATURE=0.0
MAX_TOKENS=3072
N_SAMPLES=1

# Set WEIGHTS_PTH to evaluate a trained checkpoint; leave empty for base model.
# WEIGHTS_PTH="checkpoints/final_model_iter_200_YYYYMMDD_HHMMSS/pytorch_model.pth"
WEIGHTS_PTH=""

# ── Launch ────────────────────────────────────────────────────────────────────
EXTRA_ARGS=""
if [ -n "$WEIGHTS_PTH" ]; then
    EXTRA_ARGS="--weights_pth $WEIGHTS_PTH"
fi

python -m eval.final_math500 \
    --model_path            "$MODEL" \
    --parquet_path          "$VAL_PARQUET" \
    --output_dir            "$OUTPUT_DIR" \
    --tensor_parallel_size  "$TP" \
    --cuda_devices          "$DEVICES" \
    --temperature           "$TEMPERATURE" \
    --max_tokens            "$MAX_TOKENS" \
    --n_samples             "$N_SAMPLES" \
    $EXTRA_ARGS
