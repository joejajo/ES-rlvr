#!/bin/bash
#SBATCH --job-name=es-rlvr-train
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

conda activate grpo
cd "$SLURM_SUBMIT_DIR"

python -m train.es_rlvr_train \
    --model_name       Qwen/Qwen2.5-Math-1.5B-Instruct \
    --parquet_path     data/train/pi1_r128.parquet \
    --val_parquet_path data/val/math500.parquet \
    --sigma            0.001 \
    --alpha            0.0005 \
    --population_size  20 \
    --num_iterations   200 \
    --num_engines      4 \
    --cuda_devices     0,1,2,3 \
    --val_every        20 \
    --output_every     5
