# ESRLVR

Evolution Strategies + One-Shot-RLVR with vLLM + Ray + NCCL.

## Structure

```
ESRLVR/
├── train/
│   └── es_rlvr_train.py      # Main ES training loop
├── reward/
│   ├── deepscaler.py         # Binary reward
│   └── reward_utils.py       # Answer normalisation + sympy/mathd graders
├── eval/
│   ├── inline_val.py         # Lightweight val pass during training
│   ├── final_math500.py      # Standalone Math500 evaluation
│   └── export_checkpoint.py  # Merge ES .pth → HF model directory
├── utils/
│   └── worker_extn.py        # vLLM WorkerExtension (perturb/restore/broadcast/save/load)
├── scripts/
│   ├── train_es.slurm
│   └── eval_math500.slurm
├── data/
│   ├── train/                # pi1_r128.parquet
│   └── val/                  # math500.parquet
├── checkpoints/
└── outputs/
    ├── logs/
    ├── tb/
    ├── train_preds/
    └── val_preds/
```

## Setup

```bash
conda activate grpo
cd /home/woody/iwi7/iwi7107h/ESRLVRTHESIS
```

## Training

```bash
python -m train.es_rlvr_train \
    --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
    --parquet_path data/train/pi1_r128.parquet \
    --val_parquet_path data/val/math500.parquet \
    --num_engines 4 --cuda_devices 0,1,2,3

# or: sbatch scripts/train_es.slurm
```

## Evaluation

```bash
python -m eval.final_math500 --cuda_devices 0,1,2,3 --tensor_parallel_size 4

# With checkpoint:
python -m eval.final_math500 \
    --weights_pth checkpoints/final_iter200_XXX/pytorch_model.pth \
    --tensor_parallel_size 4 --cuda_devices 0,1,2,3

# or: sbatch scripts/eval_math500.slurm
```
