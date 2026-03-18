#!/bin/bash -l
#SBATCH --job-name=es_rlvr
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/ESRLVR/outputs/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/ESRLVR/outputs/%x_%j.err
#SBATCH --export=NONE
# =============================================================================
#  ES-RLVR — 4-GPU SLURM script
#
#  One-Shot-RLVR + vLLM + Ray + NCCL, 4 engines across 4 GPUs.
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
# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR="/home/woody/iwi7/iwi7107h/ESRLVR"
TRAIN_PARQUET="${BASE_DIR}/Dataset parquet/pi1_r128.parquet"
OUT_DIR="${BASE_DIR}/outputs/results_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"
# ── Environment variables ────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Prevent Ray from trying to connect to an existing cluster
unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_GCS_SERVER_ADDRESS
# ── Hyperparameters (override via --export=ALL,VAR=val) ─────────────────────
MODEL_NAME="${MODEL_NAME:-/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B}"
SIGMA="${SIGMA:-0.001}"
ALPHA="${ALPHA:-0.0005}"
POPULATION_SIZE="${POPULATION_SIZE:-30}"
NUM_ENGINES="${NUM_ENGINES:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-200}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"
OUTPUT_EVERY="${OUTPUT_EVERY:-1}"   # print prompt+response every N iters
SCRIPT="${BASE_DIR}/train/es_rlvr_train.py"
python "${SCRIPT}" \
  --model_name        "${MODEL_NAME}" \
  --parquet_path      "${TRAIN_PARQUET}" \
  --sigma             "${SIGMA}" \
  --alpha             "${ALPHA}" \
  --population_size   "${POPULATION_SIZE}" \
  --num_engines       "${NUM_ENGINES}" \
  --num_iterations    "${NUM_ITERATIONS}" \
  --cuda_devices      0,1,2,3 \
  --global_seed       "${GLOBAL_SEED}" \
  --experiment_dir    "${OUT_DIR}/es_experiment" \
  --output_every      "${OUTPUT_EVERY}" \
  --verbose
echo "ES experiment dir: ${OUT_DIR}/es_experiment"
echo "Job complete: ${SLURM_JOB_ID}"
