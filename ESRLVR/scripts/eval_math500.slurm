#!/bin/bash -l
#SBATCH --job-name=es_rlvr_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/ESRLVRTHESIS/outputs/%x_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/ESRLVRTHESIS/outputs/%x_%j.err
#SBATCH --export=NONE
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
BASE_DIR="/home/woody/iwi7/iwi7107h/ESRLVRTHESIS"
VAL_PARQUET="${BASE_DIR}/data/val/math500.parquet"
MODEL_NAME="${MODEL_NAME:-/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B}"
# Set WEIGHTS_PTH to evaluate a trained checkpoint; leave empty for base model.
WEIGHTS_PTH="${WEIGHTS_PTH:-/home/woody/iwi7/iwi7107h/ESRLVRTHESIS/checkpoints/final_iter300_20260322_045258/pytorch_model.pth}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
EXTRA=""
[ -n "${WEIGHTS_PTH}" ] && EXTRA="--weights_pth ${WEIGHTS_PTH}"
python -m eval.final_math500 \
    --model_path           "${MODEL_NAME}" \
    --parquet_path         "${VAL_PARQUET}" \
    --output_dir           "${BASE_DIR}/outputs/val_preds" \
    --tensor_parallel_size 4 \
    --cuda_devices         0,1,2,3 \
    --temperature          0.0 \
    ${EXTRA}
echo "Job complete: ${SLURM_JOB_ID}"
