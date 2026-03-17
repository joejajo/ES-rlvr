#!/bin/bash
#SBATCH --job-name=esrlvr42
#SBATCH --output=logs/esrlvr42_%j.out
#SBATCH --error=logs/esrlvr42_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grpo

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p logs

# ── Prevent Ray from inheriting stale cluster addresses ───────────────────────
unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_GCS_SERVER_ADDRESS

# ── Move to repo root ─────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# ── Run ───────────────────────────────────────────────────────────────────────
python esrlvr42.py \
    --model_name        Qwen/Qwen2.5-Math-1.5B-Instruct \
    --parquet_path      "Dataset parquet/pi1_r128.parquet" \
    --num_engines       4 \
    --cuda_devices      0,1,2,3 \
    --sigma             0.001 \
    --alpha             0.0005 \
    --population_size   20 \
    --num_iterations    200 \
    --experiment_dir    outputs/esrlvr42 \
    --output_every      1
