#!/usr/bin/env python3
"""
Verify a .pth checkpoint produced by save_self_weights_to_disk against the
HuggingFace model's state dict.

Usage (from ES-rlvr/ESRLVR/):
    conda activate grpo
    python scripts/check_checkpoint.py
    python scripts/check_checkpoint.py --pth checkpoints/iter42_<run_tag>/pytorch_model.pth
    python scripts/check_checkpoint.py --pth checkpoints/final_iter1000_<run_tag>/pytorch_model.pth

Outputs:
    - Matched / Missing / Unexpected key counts
    - Parameter counts and coverage %
"""

import argparse
import os
import sys


def find_latest_checkpoint(ckpt_root="checkpoints"):
    """Return the most recently modified pytorch_model.pth under ckpt_root."""
    candidates = []
    for dirpath, _dirnames, filenames in os.walk(ckpt_root):
        if "pytorch_model.pth" in filenames:
            full = os.path.join(dirpath, "pytorch_model.pth")
            candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def parse_args():
    p = argparse.ArgumentParser(description="Check .pth checkpoint vs HF model")
    p.add_argument(
        "--pth",
        type=str,
        default=None,
        help="Path to pytorch_model.pth (auto-discovers latest if omitted)",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B",
        help="HuggingFace model ID or local path",
    )
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    pth = args.pth
    if pth is None:
        pth = find_latest_checkpoint()
        if pth is None:
            print("No pytorch_model.pth found under checkpoints/. "
                  "Pass --pth <path> explicitly.", file=sys.stderr)
            sys.exit(1)
        print(f"[auto] Using latest checkpoint: {pth}")

    if not os.path.exists(pth):
        print(f"File not found: {pth}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading .pth  : {pth}")
    sd = torch.load(pth, map_location="cpu", weights_only=True)

    print(f"Loading model : {args.model_name}")
    hf = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="cpu"
    )

    hf_keys   = set(hf.state_dict().keys())
    pth_keys  = set(sd.keys())
    matched   = hf_keys & pth_keys
    missing   = hf_keys - pth_keys    # in HF but NOT in .pth → keeps base weights
    unexpected = pth_keys - hf_keys   # in .pth but NOT in HF → silently discarded

    print(f"\nMatched  : {len(matched)}")
    print(f"Missing  : {len(missing)}   {list(missing)[:5]}")
    print(f"Unexpected: {len(unexpected)}  {list(unexpected)[:5]}")

    pth_params    = sum(sd[k].numel() for k in pth_keys)
    hf_params     = sum(v.numel() for v in hf.state_dict().values())
    loaded_params = sum(sd[k].numel() for k in matched)

    print(f"\nParams in .pth        : {pth_params:,}")
    print(f"Params in HF model    : {hf_params:,}")
    print(
        f"Params actually loaded: {loaded_params:,}"
        f"  ({100 * loaded_params / hf_params:.1f}%)"
    )


if __name__ == "__main__":
    main()
