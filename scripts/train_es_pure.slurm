#!/bin/bash -l
#SBATCH --job-name=es_pure
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --signal=SIGUSR1@120
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=NONE
# =============================================================================
#  Pure ES — matches VsonicV/es-fine-tuning-paper exactly.
#  All output (stdout + stderr) goes to outputs/job_<SLURM_JOB_ID>/
#
#  Key differences from train_es.slurm:
#    - --no_logprobs : no logprob overhead, no entropy metrics
#    - n_rollouts=1  : T=0.6 sampling; 1 rollout keeps compute low
#    - train_batch=32, T=0.6 for both train and val
# =============================================================================
unset SLURM_EXPORT_ENV
set -eo pipefail

# ── Per-job output dir — redirect ALL stdout/stderr here ────────────────────
BASE_DIR="/home/woody/iwi7/iwi7107h/ESRLVRTHESIS"
RUN_DIR="${BASE_DIR}/outputs/job_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
exec > "${RUN_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${RUN_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

echo "[RUN] Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "[RUN] Output dir: ${RUN_DIR}"

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
N_ROLLOUTS_PER_PROMPT="${N_ROLLOUTS_PER_PROMPT:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
GLOBAL_SEED="${GLOBAL_SEED:-33}"
OUTPUT_EVERY="${OUTPUT_EVERY:-1}"
RESUME_FROM="${RESUME_FROM:-}"
# Auto-resume from checkpoints/latest if RESUME_FROM not explicitly set
if [[ -z "${RESUME_FROM}" && -L "${BASE_DIR}/checkpoints/latest" ]]; then
    RESUME_FROM="$(readlink -f "${BASE_DIR}/checkpoints/latest")"
    echo "[AUTO-RESUME] Found checkpoint: ${RESUME_FROM}"
fi
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.6}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
SAVE_EVERY="${SAVE_EVERY:-20}"
VAL_EVERY="${VAL_EVERY:-20}"
# ── Baseline evaluation (fresh runs only, skipped on resume) ────────────────
if [[ -z "${RESUME_FROM}" ]]; then
    echo "[BASELINE] Evaluating base model at T=0.6 ..."
    CUDA_VISIBLE_DEVICES=0 python -m eval.final_math500 \
        --model_path    "${MODEL_NAME}" \
        --parquet_path  "${VAL_PARQUET}" \
        --temperature   0.6 \
        --cuda_devices  0 \
        --output_dir    "${RUN_DIR}/baseline"
    echo "[BASELINE] Evaluating base model greedy (T=0.0) ..."
    CUDA_VISIBLE_DEVICES=0 python -m eval.final_math500 \
        --model_path    "${MODEL_NAME}" \
        --parquet_path  "${VAL_PARQUET}" \
        --temperature   0.0 \
        --cuda_devices  0 \
        --output_dir    "${RUN_DIR}/baseline"
fi
# ── Training ─────────────────────────────────────────────────────────────────
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
  --output_dir        "${RUN_DIR}" \
  --no_logprobs \
  --verbose
echo "Job complete: ${SLURM_JOB_ID} at $(date)"
