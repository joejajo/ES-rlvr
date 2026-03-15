#!/bin/bash -l
#SBATCH --job-name=es_pure_binary
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/Esrlvr/outputs/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/Esrlvr/outputs/%x_%j.err
#SBATCH --export=NONE
# =============================================================================
#  ES (Pure Binary) — 4-GPU SLURM script
#
#  No entropy bonus. Binary reward only.
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
BASE_DIR="/home/woody/iwi7/iwi7107h/Esrlvr"
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

unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_GCS_SERVER_ADDRESS

# ── Hyperparameters ─────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B}"
SIGMA="${SIGMA:-0.001}"
ALPHA="${ALPHA:-0.0005}"
POPULATION_SIZE="${POPULATION_SIZE:-30}"
NUM_ENGINES="${NUM_ENGINES:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1500}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"
OUTPUT_EVERY="${OUTPUT_EVERY:-1}"

SCRIPT="${BASE_DIR}/no_entropy_es/es_train_no_entropy.py"

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
  --experiment_dir    "${OUT_DIR}/es_pure_binary" \
  --output_every      "${OUTPUT_EVERY}" \
  --antithetic \
  --verbose

echo "ES pure binary dir: ${OUT_DIR}/es_pure_binary"
echo "Job complete: ${SLURM_JOB_ID}"
