"""
eval_math500.py — Standalone Math500 evaluation script for ES-rlvr checkpoints.

Architecture:
    Single vLLM LLM instance with tensor_parallel_size=N (no Ray, no NCCL).
    tensor_parallel_size=4 gives the same 4-GPU utilisation as the training
    multi-engine setup, but without Ray actor startup / NCCL weight-broadcast
    overhead.  For eval we have ONE model copy — tensor parallelism is correct.

Checkpoint format:
    Training scripts save raw PyTorch state dicts via
    WorkerExtension.save_self_weights_to_disk → pytorch_model.pth.
    This script loads the base HF model, overlays the state dict, saves a
    temporary HF model directory, then loads it with vLLM.

Grading:
    deepscaler.compute_training_score — identical dual-pass logic to the
    One-Shot-RLVR upstream grader (grade_answer_mathd OR grade_answer_sympy).

Usage examples:
    # Evaluate base model (no checkpoint), greedy, 1 GPU
    python eval_math500.py --cuda_devices 0

    # Evaluate a .pth checkpoint, 4-GPU tensor-parallel
    python eval_math500.py \\
        --weights_pth outputs/esrlvr42/run_XXX/model_saves/final_model_iter_200/pytorch_model.pth \\
        --tensor_parallel_size 4 --cuda_devices 0,1,2,3

    # avg@8 sampling (temperature > 0)
    python eval_math500.py --n_samples 8 --temperature 0.7 --cuda_devices 0
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from deepscaler import SYSTEM_PROMPT, compute_score_routed


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Math500 evaluation for ES-rlvr")
    p.add_argument("--model_path", type=str,
                   default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                   help="HF model name or directory (base weights)")
    p.add_argument("--weights_pth", type=str, default=None,
                   help="Optional .pth state dict to overlay onto base model")
    p.add_argument("--parquet_path", type=str,
                   default="Dataset parquet/math500.parquet",
                   help="Path to math500 parquet file")
    p.add_argument("--output_dir", type=str,
                   default="outputs/eval_math500",
                   help="Directory for results.jsonl and summary.json")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0.0 = greedy)")
    p.add_argument("--max_tokens", type=int, default=3072,
                   help="Max generation tokens per sample")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Samples per problem (1 = greedy; >1 = avg@k)")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="vLLM tensor parallelism — number of GPUs for one model")
    p.add_argument("--cuda_devices", type=str, default="0",
                   help="CUDA_VISIBLE_DEVICES (e.g. '0,1,2,3')")
    p.add_argument("--show_n", type=int, default=3,
                   help="Number of problems to print full reasoning chain for")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same schema as training scripts)
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
# Generation helper
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_after_first_boxed(text: str) -> str:
    """Trim response right after the first complete \\boxed{...} block."""
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
    if end is None:
        return s
    return s[:end + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def prepare_model_dir(model_path: str, weights_pth: str | None, tmp_root: str) -> str:
    """
    If weights_pth is provided:
        1. Load base model from model_path (CPU, float16).
        2. Overlay state dict from weights_pth.
        3. Save as HF model dir under tmp_root.
        4. Return path to that temp dir (caller must clean up).
    Otherwise return model_path unchanged.
    """
    if weights_pth is None:
        return model_path, None

    print(f"[WEIGHTS] Loading base model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    print(f"[WEIGHTS] Loading state dict from: {weights_pth}")
    state_dict = torch.load(weights_pth, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WEIGHTS] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WEIGHTS] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    tmp_dir = tempfile.mkdtemp(dir=tmp_root, prefix="eval_model_")
    print(f"[WEIGHTS] Saving merged model to temp dir: {tmp_dir}")
    model.save_pretrained(tmp_dir)
    del model
    return tmp_dir, tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # GPU visibility must be set before importing vLLM / torch CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    print(f"[INIT] Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ── Checkpoint handling ────────────────────────────────────────────────────
    vllm_model_path, tmp_dir = prepare_model_dir(
        args.model_path, args.weights_pth, args.output_dir
    )

    # Save tokenizer to tmp_dir if we created one (vLLM needs tokenizer there)
    if tmp_dir is not None:
        tokenizer.save_pretrained(tmp_dir)

    # ── vLLM engine ───────────────────────────────────────────────────────────
    # Single LLM, tensor_parallel_size shards across all GPUs.
    # No Ray, no NCCL weight-broadcast — those only exist for ES perturbation.
    print(f"[INIT] Launching vLLM (tensor_parallel_size={args.tensor_parallel_size})")
    llm = LLM(
        model=vllm_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        enable_prefix_caching=False,
        enforce_eager=False,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"[DATA] Loading math500 from: {args.parquet_path}")
    task_datas = load_val_task_datas(args.parquet_path, tokenizer)
    print(f"[DATA] {len(task_datas)} problems loaded.")

    # ── Sampling ──────────────────────────────────────────────────────────────
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_samples,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    print(f"[EVAL] Generating (temperature={args.temperature}, "
          f"max_tokens={args.max_tokens}, n_samples={args.n_samples}) ...")
    t0 = time.time()
    outputs = llm.generate([d["prompt_str"] for d in task_datas], sampling_params)
    elapsed = time.time() - t0
    print(f"[EVAL] Generation done in {elapsed:.1f}s")

    # ── Grading ───────────────────────────────────────────────────────────────
    results = []
    total_correct = 0.0

    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        gt = data["ground_truth"]
        sample_scores = []

        for req_output in output.outputs:
            completion = _truncate_after_first_boxed(req_output.text)
            score = compute_score_routed(data["data_source"], completion, gt)
            sample_scores.append(score)

        avg_score = sum(sample_scores) / len(sample_scores)
        total_correct += avg_score

        # Extract answer from first sample for display/logging
        first_completion = _truncate_after_first_boxed(output.outputs[0].text)
        extracted = None
        m = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", first_completion)
        if m:
            extracted = m.group(1).strip()

        tag = "✓ CORRECT" if avg_score >= 1.0 else ("~PARTIAL~" if avg_score > 0 else "✗ WRONG")

        if idx < args.show_n:
            print(f"\n{'='*70}")
            print(f"[{idx+1}/{len(task_datas)}] {tag}")
            print(f"Q: {data['question'][:200]}")
            print(f"GT: {gt}")
            print(f"Extracted: {extracted}")
            if args.n_samples > 1:
                print(f"Sample scores: {sample_scores} → avg={avg_score:.3f}")
            print(f"--- Response (first sample) ---")
            print(first_completion[:600])
            print(f"{'='*70}")
        else:
            # Compact per-problem line
            print(f"[{idx+1:3d}/{len(task_datas)}] {tag}  gt={gt!r}  pred={extracted!r}"
                  + (f"  avg@{args.n_samples}={avg_score:.2f}" if args.n_samples > 1 else ""))

        results.append({
            "idx":             idx,
            "question":        data["question"],
            "ground_truth":    gt,
            "extracted_answer": extracted,
            "avg_score":       avg_score,
            "sample_scores":   sample_scores,
            "data_source":     data["data_source"],
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    accuracy = total_correct / len(task_datas) if task_datas else 0.0
    n_correct = int(round(total_correct))
    n_total = len(task_datas)

    print(f"\n{'='*70}")
    print(f"  Math500 Accuracy: {accuracy*100:.2f}%  ({n_correct}/{n_total})")
    if args.n_samples > 1:
        print(f"  Metric: avg@{args.n_samples} (mean correctness per problem)")
    print(f"{'='*70}\n")

    # ── Write outputs ─────────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "accuracy":            accuracy,
        "correct":             n_correct,
        "total":               n_total,
        "model_path":          args.model_path,
        "weights_pth":         args.weights_pth,
        "temperature":         args.temperature,
        "max_tokens":          args.max_tokens,
        "n_samples":           args.n_samples,
        "tensor_parallel_size": args.tensor_parallel_size,
        "generation_time_s":   elapsed,
        "timestamp":           datetime.utcnow().isoformat() + "Z",
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OUT] results.jsonl → {results_path}")
    print(f"[OUT] summary.json  → {summary_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if tmp_dir is not None:
        print(f"[CLEANUP] Removing temp model dir: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
