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

MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
TRAIN_PARQUET="data/train/pi1_r128.parquet"
VAL_PARQUET="data/val/math500.parquet"
SIGMA=0.001
ALPHA=0.0005
POP=20
ITERS=200
ENGINES=4
DEVICES="0,1,2,3"
VAL_EVERY=20
OUTPUT_EVERY=5

python -m train.es_rlvr_train \
    --model_name       "$MODEL" \
    --parquet_path     "$TRAIN_PARQUET" \
    --val_parquet_path "$VAL_PARQUET" \
    --sigma            "$SIGMA" \
    --alpha            "$ALPHA" \
    --population_size  "$POP" \
    --num_iterations   "$ITERS" \
    --num_engines      "$ENGINES" \
    --cuda_devices     "$DEVICES" \
    --val_every        "$VAL_EVERY" \
    --output_every     "$OUTPUT_EVERY"
