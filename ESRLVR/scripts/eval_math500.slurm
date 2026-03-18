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

conda activate grpo
cd "$SLURM_SUBMIT_DIR"

# Set WEIGHTS_PTH to evaluate a trained checkpoint; leave empty for base model.
# WEIGHTS_PTH="checkpoints/final_iter200_YYYYMMDD_HHMMSS/pytorch_model.pth"
WEIGHTS_PTH=""

EXTRA=""
[ -n "$WEIGHTS_PTH" ] && EXTRA="--weights_pth $WEIGHTS_PTH"

python -m eval.final_math500 \
    --model_path           Qwen/Qwen2.5-Math-1.5B-Instruct \
    --parquet_path         data/val/math500.parquet \
    --output_dir           outputs/val_preds \
    --tensor_parallel_size 4 \
    --cuda_devices         0,1,2,3 \
    --temperature          0.0 \
    --max_tokens           3084 \
    $EXTRA
