"""
Merge an ES .pth state dict onto a base HF model and save as HF model directory.

Usage:
    python -m eval.export_checkpoint \\
        --model_path Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --weights_pth checkpoints/final_iter200_XXX/pytorch_model.pth \\
        --output_dir  checkpoints/exported_hf_model
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.ckpt_utils import unfuse_vllm_state_dict


def export_checkpoint(model_path, weights_pth, output_dir):
    print(f"[EXPORT] Base   : {model_path}")
    print(f"[EXPORT] Weights: {weights_pth}")
    print(f"[EXPORT] Out    : {output_dir}")

    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    model      = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")
    state_dict = torch.load(weights_pth, map_location="cpu", weights_only=True)
    state_dict = unfuse_vllm_state_dict(state_dict, model.config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"[EXPORT] Missing    {len(missing)}: {missing[:3]}")
    if unexpected: print(f"[EXPORT] Unexpected {len(unexpected)}: {unexpected[:3]}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"[EXPORT] Done → {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  required=True)
    p.add_argument("--weights_pth", required=True)
    p.add_argument("--output_dir",  required=True)
    args = p.parse_args()
    export_checkpoint(args.model_path, args.weights_pth, args.output_dir)
