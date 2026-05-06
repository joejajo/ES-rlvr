#!/bin/bash -l
#SBATCH --job-name=es_rlvr
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --signal=SIGUSR1@120
#SBATCH --output=/home/woody/iwi7/iwi7107h/ESRLVRTHESIS/outputs/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/ESRLVRTHESIS/outputs/%x_%j.err
#SBATCH --export=NONE
# =============================================================================
#  ES-RLVR — 2-GPU SLURM script
#
#  One-Shot-RLVR + vLLM + Ray + NCCL, 2 engines across 2 GPUs.
# =============================================================================
unset SLURM_EXPORT_ENV
set -eo pipefail
module purge
module load cuda/12.4.1
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
# Propagate CUDA lib path to all child processes (Ray workers, vLLM)
CUDA_HOME="${CUDA_HOME:-/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-13.3.0/cuda-12.4.1-cwwkrpu24rnxndj5ouklz26aiskqd4is}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
set +u
conda activate myenv
set -u
# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR="/home/woody/iwi7/iwi7107h/ESRLVRTHESIS"
TRAIN_PARQUET="${BASE_DIR}/data/train/pi1_r128.parquet"
VAL_PARQUET="${BASE_DIR}/data/val/math500.parquet"
cd "${BASE_DIR}"
# ── Environment variables ────────────────────────────────────────────────────
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
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
NUM_ENGINES="${NUM_ENGINES:-2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000}"
N_ROLLOUTS_PER_PROMPT="${N_ROLLOUTS_PER_PROMPT:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"   # prompts sampled per iteration (128 total)
GLOBAL_SEED="${GLOBAL_SEED:-}"
OUTPUT_EVERY="${OUTPUT_EVERY:-1}"       # print prompt+response every N iters
RESUME_FROM="${RESUME_FROM:-}"          # path to checkpoint dir; empty = auto-detect
# Auto-resume from checkpoints/latest if RESUME_FROM not explicitly set
if [[ -z "${RESUME_FROM}" && -L "${BASE_DIR}/checkpoints/latest" ]]; then
    RESUME_FROM="$(readlink -f "${BASE_DIR}/checkpoints/latest")"
    echo "[AUTO-RESUME] Found checkpoint: ${RESUME_FROM}"
fi
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.6}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
SAVE_EVERY="${SAVE_EVERY:-20}"
VAL_EVERY="${VAL_EVERY:-20}"
SCRIPT="${BASE_DIR}/train/es_rlvr_train.py"
SEED_ARG=""
[[ -n "${GLOBAL_SEED}" ]] && SEED_ARG="--global_seed ${GLOBAL_SEED}"
RESUME_ARG=""
[[ -n "${RESUME_FROM}" ]] && RESUME_ARG="--resume_from ${RESUME_FROM}"
python "${SCRIPT}" \
  --model_name        "${MODEL_NAME}" \
  --parquet_path      "${TRAIN_PARQUET}" \
  --val_parquet_path  "${VAL_PARQUET}" \
  --sigma             "${SIGMA}" \
  --alpha             "${ALPHA}" \
  --population_size   "${POPULATION_SIZE}" \
  --num_engines       "${NUM_ENGINES}" \
  --num_iterations    "${NUM_ITERATIONS}" \
  --cuda_devices      0,1 \
  ${SEED_ARG} \
  ${RESUME_ARG} \
  --n_rollouts_per_prompt "${N_ROLLOUTS_PER_PROMPT}" \
  --train_batch_size  "${TRAIN_BATCH_SIZE}" \
  --output_every      "${OUTPUT_EVERY}" \
  --train_temperature "${TRAIN_TEMPERATURE}" \
  --val_temperature   "${VAL_TEMPERATURE}" \
  --save_every        "${SAVE_EVERY}" \
  --val_every         "${VAL_EVERY}" \
  --verbose
echo "Job complete: ${SLURM_JOB_ID}"
