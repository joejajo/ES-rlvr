#!/bin/bash
#SBATCH --job-name=es_rlvr
#SBATCH --output=logs/es_rlvr_%j.out
#SBATCH --error=logs/es_rlvr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# ---------------------------------------------------------------------------
# Adjust --partition to your cluster's GPU partition name.
# Common names: gpu, a100, h100, accelerated, gpu_v100, gpu_h100, etc.
# ---------------------------------------------------------------------------
#SBATCH --partition=gpu

# ── Optional: email notifications ──────────────────────────────────────────
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your@email.de

# ── Optional: project/account (required on some clusters) ──────────────────
# #SBATCH --account=<your_project_id>

# ---------------------------------------------------------------------------
# ES-RLVR training — One-Shot-RLVR + vLLM + Ray + NCCL
#
# Usage:
#   sbatch slurm_es_rlvr.sh                    # defaults
#   sbatch slurm_es_rlvr.sh --export=ALL,SIGMA=0.002,ALPHA=0.001
#
# Requires: conda env "grpo" at
#   /home/woody/iwi7/iwi7107h/conda_envs/grpo
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Environment ─────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grpo

# Visible GPUs — must match --gres=gpu:N and --num_engines
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Prevent Ray from trying to connect to an existing cluster
unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_GCS_SERVER_ADDRESS

# Prevent tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# vLLM: disable V1 multiprocessing (required for multi-engine Ray deployment)
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ── Paths ───────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

TRAIN_PARQUET="${REPO_DIR}/Dataset parquet/pi1_r128.parquet"
VAL_PARQUET="${REPO_DIR}/Dataset parquet/math500.parquet"
EXPERIMENT_DIR="${REPO_DIR}/outputs/es_rlvr_slurm"

mkdir -p logs "${EXPERIMENT_DIR}"

# ── Hyperparameters (override via --export=ALL,VAR=val) ─────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
SIGMA="${SIGMA:-0.001}"
ALPHA="${ALPHA:-0.0005}"
POPULATION_SIZE="${POPULATION_SIZE:-20}"
NUM_ENGINES="${NUM_ENGINES:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-200}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
VAL_EVERY="${VAL_EVERY:-10}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-50}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"

# ── Print job info ───────────────────────────────────────────────────────────
echo "========================================================"
echo "  ES-RLVR SLURM Job"
echo "========================================================"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Node           : $(hostname)"
echo "  GPUs           : ${CUDA_VISIBLE_DEVICES}"
echo "  Conda env      : $(conda info --envs | grep '*' | awk '{print $1}')"
echo "  Python         : $(python --version)"
echo "  PyTorch        : $(python -c 'import torch; print(torch.__version__)')"
echo "  vLLM           : $(python -c 'import vllm; print(vllm.__version__)')"
echo "  CUDA available : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count      : $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  Model          : ${MODEL_NAME}"
echo "  sigma          : ${SIGMA}"
echo "  alpha          : ${ALPHA}"
echo "  population     : ${POPULATION_SIZE}"
echo "  engines        : ${NUM_ENGINES}"
echo "  iterations     : ${NUM_ITERATIONS}"
echo "  entropy_coeff  : ${ENTROPY_COEFF}"
echo "  val_every      : ${VAL_EVERY}"
echo "  Experiment dir : ${EXPERIMENT_DIR}"
echo "========================================================"
echo ""

# ── Launch training ──────────────────────────────────────────────────────────
python "${REPO_DIR}/es_rlvr_train.py" \
    --model_name        "${MODEL_NAME}" \
    --parquet_path      "${TRAIN_PARQUET}" \
    --val_parquet_path  "${VAL_PARQUET}" \
    --sigma             "${SIGMA}" \
    --alpha             "${ALPHA}" \
    --population_size   "${POPULATION_SIZE}" \
    --num_engines       "${NUM_ENGINES}" \
    --num_iterations    "${NUM_ITERATIONS}" \
    --entropy_coeff     "${ENTROPY_COEFF}" \
    --val_every         "${VAL_EVERY}" \
    --val_batch_size    "${VAL_BATCH_SIZE}" \
    --cuda_devices      "${CUDA_VISIBLE_DEVICES}" \
    --global_seed       "${GLOBAL_SEED}" \
    --experiment_dir    "${EXPERIMENT_DIR}" \
    --antithetic \
    --verbose

echo ""
echo "Job ${SLURM_JOB_ID} finished: $(date)"
