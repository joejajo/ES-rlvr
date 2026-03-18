#!/bin/bash -l
#SBATCH --job-name=eval_math500
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/Esrlvr/outputs/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/Esrlvr/outputs/%x_%j.err
#SBATCH --export=NONE
# =============================================================================
#  Math500 standalone evaluation — 4-GPU tensor-parallel vLLM
#
#  No Ray, no NCCL.  Single vLLM LLM instance with tensor_parallel_size=4.
#  All 4 GPUs cooperate on every forward pass (tensor parallelism), giving
#  the same GPU utilisation as the training setup without Ray actor overhead.
#
#  Override via --export=ALL,VAR=val or by editing the variables below.
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR="/home/woody/iwi7/iwi7107h/Esrlvr"
SCRIPT="${BASE_DIR}/eval_math500.py"
VAL_PARQUET="${BASE_DIR}/Dataset parquet/math500.parquet"
OUT_DIR="${BASE_DIR}/outputs/eval_${SLURM_JOB_ID}"

mkdir -p "${OUT_DIR}"

# ── Model / checkpoint ────────────────────────────────────────────────────────
# Base HF model (local path or HF hub name)
MODEL_PATH="${MODEL_PATH:-/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B}"

# Optional: path to a pytorch_model.pth saved by the training script.
# Leave empty to evaluate the base model without any checkpoint overlay.
WEIGHTS_PTH="${WEIGHTS_PTH:-}"

# ── Eval hyperparameters ──────────────────────────────────────────────────────
TEMPERATURE="${TEMPERATURE:-0.0}"     # 0.0 = greedy (matches One-Shot-RLVR)
MAX_TOKENS="${MAX_TOKENS:-3072}"
N_SAMPLES="${N_SAMPLES:-1}"           # set >1 for avg@k (requires temperature>0)
SHOW_N="${SHOW_N:-5}"                 # problems to print full response for

# ── Environment variables ─────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    python "${SCRIPT}"
    --model_path         "${MODEL_PATH}"
    --parquet_path       "${VAL_PARQUET}"
    --output_dir         "${OUT_DIR}"
    --temperature        "${TEMPERATURE}"
    --max_tokens         "${MAX_TOKENS}"
    --n_samples          "${N_SAMPLES}"
    --tensor_parallel_size 4
    --cuda_devices       0,1,2,3
    --show_n             "${SHOW_N}"
)

# Append --weights_pth only if set
if [[ -n "${WEIGHTS_PTH}" ]]; then
    CMD+=(--weights_pth "${WEIGHTS_PTH}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "Results dir: ${OUT_DIR}"
echo "Accuracy:    $(python -c "import json; d=json.load(open('${OUT_DIR}/summary.json')); print(f\"{d['accuracy']*100:.2f}% ({d['correct']}/{d['total']})\")")"
echo "Job complete: ${SLURM_JOB_ID}"
