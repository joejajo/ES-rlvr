"""
final_math500.py — Standalone Math500 evaluation for ES-RLVR checkpoints.

Architecture:
    Single vLLM LLM with tensor_parallel_size=N (no Ray, no NCCL).

Checkpoint loading:
    Pass --weights_pth to overlay an ES state dict onto the base model,
    or omit to evaluate the base model directly.

Usage:
    # Base model, greedy, 1 GPU
    python -m eval.final_math500 --cuda_devices 0

    # Trained checkpoint, 4-GPU tensor-parallel
    python -m eval.final_math500 \\
        --weights_pth checkpoints/final_model_iter_200_XXX/pytorch_model.pth \\
        --tensor_parallel_size 4 --cuda_devices 0,1,2,3

    # avg@8 sampling
    python -m eval.final_math500 --n_samples 8 --temperature 0.7 --cuda_devices 0
"""

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from reward.deepscaler import SYSTEM_PROMPT, compute_training_score


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Math500 evaluation for ES-RLVR")
    p.add_argument("--model_path",  type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--weights_pth", type=str, default=None)
    p.add_argument("--parquet_path", type=str, default="data/val/math500.parquet")
    p.add_argument("--output_dir",  type=str, default="outputs/val_preds")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens",  type=int,   default=3072)
    p.add_argument("--n_samples",   type=int,   default=1)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--cuda_devices", type=str, default="0")
    p.add_argument("--show_n",      type=int,   default=3)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_val_task_datas(parquet_path: str, tokenizer) -> list:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        raise RuntimeError(f"Parquet has zero rows: {parquet_path}")

    task_datas = []
    for _, row in df.iterrows():
        chat = list(row["prompt"])
        reward_model = row["reward_model"]
        gt = (reward_model["ground_truth"]
              if isinstance(reward_model, dict) else str(reward_model))
        ds = str(row.get("data_source", "deepscaler"))
        chat_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + chat
        prompt_str = tokenizer.apply_chat_template(
            chat_with_system, tokenize=False, add_generation_prompt=True
        )
        question = ""
        for msg in chat:
            if isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content", "")
                break
        task_datas.append({
            "prompt_str":   prompt_str,
            "ground_truth": str(gt),
            "data_source":  ds,
            "question":     question,
        })
    return task_datas


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_after_first_boxed(text: str) -> str:
    s = text or ""
    start = s.find(r"\boxed")
    if start == -1:
        return s
    i = start + len(r"\boxed")
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s) or s[i] != "{":
        return s
    depth = 0
    end = None
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    return s[:end + 1] if end is not None else s


def prepare_model_dir(model_path: str, weights_pth: str | None, tmp_root: str):
    if weights_pth is None:
        return model_path, None

    print(f"[WEIGHTS] Base model : {model_path}")
    print(f"[WEIGHTS] State dict : {weights_pth}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    state_dict = torch.load(weights_pth, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WEIGHTS] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WEIGHTS] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    tmp_dir = tempfile.mkdtemp(dir=tmp_root, prefix="eval_model_")
    print(f"[WEIGHTS] Saving merged model → {tmp_dir}")
    model.save_pretrained(tmp_dir)
    del model
    return tmp_dir, tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    vllm_model_path, tmp_dir = prepare_model_dir(
        args.model_path, args.weights_pth, args.output_dir
    )
    if tmp_dir is not None:
        tokenizer.save_pretrained(tmp_dir)

    print(f"[INIT] vLLM  tensor_parallel_size={args.tensor_parallel_size}")
    llm = LLM(
        model=vllm_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        enable_prefix_caching=False,
        enforce_eager=False,
    )

    print(f"[DATA] {args.parquet_path}")
    task_datas = load_val_task_datas(args.parquet_path, tokenizer)
    print(f"[DATA] {len(task_datas)} problems.")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_samples,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    t0 = time.time()
    outputs = llm.generate([d["prompt_str"] for d in task_datas], sampling_params)
    elapsed = time.time() - t0
    print(f"[EVAL] Done in {elapsed:.1f}s")

    results = []
    total_correct = 0.0

    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        gt = data["ground_truth"]
        sample_scores = [
            compute_training_score(_truncate_after_first_boxed(o.text), gt)
            for o in output.outputs
        ]
        avg_score = sum(sample_scores) / len(sample_scores)
        total_correct += avg_score

        first = _truncate_after_first_boxed(output.outputs[0].text)
        m = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", first)
        extracted = m.group(1).strip() if m else None

        tag = "CORRECT" if avg_score >= 1.0 else ("PARTIAL" if avg_score > 0 else "WRONG")

        if idx < args.show_n:
            print(f"\n{'='*70}\n[{idx+1}/{len(task_datas)}] {tag}")
            print(f"Q: {data['question'][:200]}")
            print(f"GT: {gt}  |  Extracted: {extracted}")
            if args.n_samples > 1:
                print(f"Scores: {sample_scores} → avg={avg_score:.3f}")
            print(first[:600])
            print("=" * 70)
        else:
            print(f"[{idx+1:3d}/{len(task_datas)}] {tag}  gt={gt!r}  pred={extracted!r}"
                  + (f"  avg@{args.n_samples}={avg_score:.2f}" if args.n_samples > 1 else ""))

        results.append({
            "idx":              idx,
            "question":         data["question"],
            "ground_truth":     gt,
            "extracted_answer": extracted,
            "avg_score":        avg_score,
            "sample_scores":    sample_scores,
            "data_source":      data["data_source"],
        })

    accuracy = total_correct / len(task_datas) if task_datas else 0.0
    n_correct = int(round(total_correct))
    n_total   = len(task_datas)

    print(f"\n{'='*70}")
    print(f"  Math500 Accuracy: {accuracy*100:.2f}%  ({n_correct}/{n_total})")
    if args.n_samples > 1:
        print(f"  Metric: avg@{args.n_samples}")
    print(f"{'='*70}\n")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"results_{ts}.jsonl")
    summary_path = os.path.join(args.output_dir, f"summary_{ts}.json")

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    with open(summary_path, "w") as f:
        json.dump({
            "accuracy":             accuracy,
            "correct":              n_correct,
            "total":                n_total,
            "model_path":           args.model_path,
            "weights_pth":          args.weights_pth,
            "temperature":          args.temperature,
            "max_tokens":           args.max_tokens,
            "n_samples":            args.n_samples,
            "tensor_parallel_size": args.tensor_parallel_size,
            "generation_time_s":    elapsed,
            "timestamp":            datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)

    print(f"[OUT] {results_path}")
    print(f"[OUT] {summary_path}")

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
