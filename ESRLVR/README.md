# ESRLVR

Evolution Strategies + One-Shot-RLVR with vLLM + Ray + NCCL.

## Structure

```
ESRLVR/
├── worker_extn.py            # vLLM WorkerExtension (perturb/restore/broadcast/save/load)
├── train/
│   └── es_rlvr_train.py      # Main ES training loop
├── reward/
│   ├── deepscaler.py         # Binary reward (compute_training_score / compute_score)
│   └── reward_utils.py       # Answer normalisation + sympy/mathd graders
├── eval/
│   ├── inline_val.py         # Lightweight val pass called during training
│   ├── final_math500.py      # Standalone Math500 evaluation script
│   └── export_checkpoint.py  # Merge ES .pth state dict → HF model directory
├── scripts/
│   ├── train_es.slurm        # SLURM job template: training
│   └── eval_math500.slurm   # SLURM job template: evaluation
├── data/
│   ├── train/                # Training parquets (e.g. pi1_r128.parquet)
│   └── val/                  # Validation parquets (e.g. math500.parquet)
├── checkpoints/              # Saved model weights (.pth)
└── outputs/
    ├── logs/                 # SLURM stdout/stderr
    ├── tb/                   # TensorBoard event files
    ├── train_preds/          # Per-iteration training JSONL outputs
    └── val_preds/            # Inline + final validation JSONL outputs
```

## Setup

```bash
conda activate grpo
cd ESRLVR/
```

## Training

```bash
python -m train.es_rlvr_train \
    --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
    --parquet_path data/train/pi1_r128.parquet \
    --val_parquet_path data/val/math500.parquet \
    --num_engines 4 \
    --cuda_devices 0,1,2,3

# or via SLURM:
sbatch scripts/train_es.slurm
```

## Evaluation

```bash
# Base model
python -m eval.final_math500 --cuda_devices 0,1,2,3 --tensor_parallel_size 4

# Trained checkpoint
python -m eval.final_math500 \
    --weights_pth checkpoints/final_model_iter_200_XXX/pytorch_model.pth \
    --tensor_parallel_size 4 --cuda_devices 0,1,2,3

# or via SLURM:
sbatch scripts/eval_math500.slurm
```

## Export checkpoint to HF model dir

```bash
python -m eval.export_checkpoint \
    --model_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --weights_pth checkpoints/final_model_iter_200_XXX/pytorch_model.pth \
    --output_dir checkpoints/exported_hf_model
```
