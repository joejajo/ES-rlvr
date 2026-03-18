"""
export_checkpoint.py — Merge an ES .pth state dict onto a base HF model and
save a full HuggingFace model directory ready for vLLM or evaluation.

Usage:
    python -m eval.export_checkpoint \\
        --model_path Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --weights_pth checkpoints/final_model_iter_200_XXX/pytorch_model.pth \\
        --output_dir  checkpoints/exported_hf_model
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_checkpoint(model_path: str, weights_pth: str, output_dir: str) -> None:
    print(f"[EXPORT] Base model  : {model_path}")
    print(f"[EXPORT] State dict  : {weights_pth}")
    print(f"[EXPORT] Output dir  : {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )

    state_dict = torch.load(weights_pth, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[EXPORT] Missing    ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[EXPORT] Unexpected ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"[EXPORT] Done → {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Export ES checkpoint to HF model dir")
    p.add_argument("--model_path",  type=str, required=True)
    p.add_argument("--weights_pth", type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_checkpoint(args.model_path, args.weights_pth, args.output_dir)
