#!/bin/bash -l
#SBATCH --job-name=es_smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/ESRLVRTHESIS/outputs/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/ESRLVRTHESIS/outputs/%x_%j.err
#SBATCH --export=NONE
# =============================================================================
#  ES-RLVR — Smoke test (1 GPU, 4-member population, 3 iterations, no val)
# =============================================================================
unset SLURM_EXPORT_ENV
set -eo pipefail
module purge
module load cuda/12.4.1
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate grpo
set -u
# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR="/home/woody/iwi7/iwi7107h/ESRLVRTHESIS"
MODEL_NAME="/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B"
TRAIN_PARQUET="${BASE_DIR}/data/train/pi1_r128.parquet"
VAL_PARQUET="${BASE_DIR}/data/val/math500.parquet"
cd "${BASE_DIR}"
# ── Environment variables ─────────────────────────────────────────────────────
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export VLLM_ENABLE_V1_MULTIPROCESSING=0
unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_GCS_SERVER_ADDRESS
# ── Run ──────────────────────────────────────────────────────────────────────
python "${BASE_DIR}/train/es_rlvr_train.py" \
  --model_name        "${MODEL_NAME}" \
  --parquet_path      "${TRAIN_PARQUET}" \
  --val_parquet_path  "${VAL_PARQUET}" \
  --num_engines       1 \
  --cuda_devices      0 \
  --population_size   4 \
  --num_iterations    3 \
  --val_every         0 \
  --output_every      1 \
  --verbose
echo "Smoke test complete."
