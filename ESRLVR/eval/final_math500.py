"""
Standalone Math500 evaluation for ES-RLVR checkpoints.

Usage:
    # Base model, greedy, 1 GPU
    python -m eval.final_math500 --cuda_devices 0

    # Trained checkpoint, 4-GPU tensor-parallel
    python -m eval.final_math500 \\
        --weights_pth checkpoints/final_iter200_XXX/pytorch_model.pth \\
        --tensor_parallel_size 4 --cuda_devices 0,1,2,3

    # avg@8 sampling
    python -m eval.final_math500 --n_samples 8 --temperature 0.7 --cuda_devices 0
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from reward.deepscaler import SYSTEM_PROMPT, compute_training_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",           type=str,   default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--weights_pth",          type=str,   default=None)
    p.add_argument("--parquet_path",         type=str,   default="data/val/math500.parquet")
    p.add_argument("--output_dir",           type=str,   default="outputs/val_preds")
    p.add_argument("--temperature",          type=float, default=0.0)
    p.add_argument("--max_tokens",           type=int,   default=3072)
    p.add_argument("--n_samples",            type=int,   default=1)
    p.add_argument("--tensor_parallel_size", type=int,   default=1)
    p.add_argument("--cuda_devices",         type=str,   default="0")
    return p.parse_args()


def load_val_task_datas(parquet_path, tokenizer):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    task_datas = []
    for _, row in df.iterrows():
        chat = list(row["prompt"])
        rm   = row["reward_model"]
        gt   = rm["ground_truth"] if isinstance(rm, dict) else str(rm)
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}] + chat,
            tokenize=False, add_generation_prompt=True,
        )
        question = next(
            (m.get("content", "") for m in chat
             if isinstance(m, dict) and m.get("role") == "user"), ""
        )
        task_datas.append({
            "prompt_str":   prompt_str,
            "ground_truth": str(gt),
            "data_source":  str(row.get("data_source", "deepscaler")),
            "question":     question,
        })
    return task_datas


def _extract_last_boxed(text):
    """Return the content of the last \\boxed{} in text, or None if absent."""
    s = text or ""
    parts = s.split(r"\boxed")
    if len(parts) < 2:
        return None
    last = parts[-1]
    i = 0
    while i < len(last) and last[i].isspace():
        i += 1
    if i >= len(last) or last[i] != "{":
        return None
    depth, end = 0, None
    for j in range(i, len(last)):
        if last[j] == "{":
            depth += 1
        elif last[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    return last[i + 1:end] if end is not None else None


def prepare_model_dir(model_path, weights_pth, tmp_root):
    if weights_pth is None:
        return model_path, None
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")
    state_dict = torch.load(weights_pth, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"[WEIGHTS] Missing    {len(missing)}: {missing[:3]}")
    if unexpected: print(f"[WEIGHTS] Unexpected {len(unexpected)}: {unexpected[:3]}")
    tmp_dir = tempfile.mkdtemp(dir=tmp_root, prefix="eval_model_")
    model.save_pretrained(tmp_dir)
    del model
    return tmp_dir, tmp_dir


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    vllm_path, tmp_dir = prepare_model_dir(args.model_path, args.weights_pth, args.output_dir)
    if tmp_dir:
        tokenizer.save_pretrained(tmp_dir)

    llm = LLM(model=vllm_path, tensor_parallel_size=args.tensor_parallel_size,
              dtype="float16", enable_prefix_caching=False, enforce_eager=False)

    task_datas = load_val_task_datas(args.parquet_path, tokenizer)
    print(f"[DATA] {len(task_datas)} problems.")

    t0 = time.time()
    outputs = llm.generate(
        [d["prompt_str"] for d in task_datas],
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                       n=args.n_samples, seed=0,
                       stop=["</s>", "<|im_end|>", "<|endoftext|>"],
                       stop_token_ids=[151645, 151643]),
    )
    elapsed = time.time() - t0
    print(f"[EVAL] Done in {elapsed:.1f}s")

    results, total_correct = [], 0.0
    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        gt     = data["ground_truth"]
        scores = [compute_training_score(o.text, gt) for o in output.outputs]
        avg    = sum(scores) / len(scores)
        total_correct += avg

        extracted = _extract_last_boxed(output.outputs[0].text)
        tag = "CORRECT" if avg >= 1.0 else ("PARTIAL" if avg > 0 else "WRONG")

        print(f"\n{'='*70}\n[{idx+1}/{len(task_datas)}] {tag}")
        print(f"Q: {data['question'][:200]}\nGT: {gt}  |  Ans: {extracted}")
        print(output.outputs[0].text)
        if args.n_samples > 1:
            print(f"avg@{args.n_samples}={avg:.2f}")

        results.append({
            "idx": idx, "question": data["question"], "ground_truth": gt,
            "extracted_answer": extracted, "avg_score": avg,
            "sample_scores": scores, "data_source": data["data_source"],
        })

    accuracy  = total_correct / len(task_datas) if task_datas else 0.0
    n_correct = int(round(total_correct))
    print(f"\n{'='*70}\n  Math500: {accuracy*100:.2f}%  ({n_correct}/{len(task_datas)})\n{'='*70}\n")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rp = os.path.join(args.output_dir, f"results_{ts}.jsonl")
    sp = os.path.join(args.output_dir, f"summary_{ts}.json")

    with open(rp, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(sp, "w") as f:
        json.dump({"accuracy": accuracy, "correct": n_correct, "total": len(task_datas),
                   "model_path": args.model_path, "weights_pth": args.weights_pth,
                   "temperature": args.temperature, "max_tokens": args.max_tokens,
                   "n_samples": args.n_samples, "tensor_parallel_size": args.tensor_parallel_size,
                   "generation_time_s": elapsed, "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)

    print(f"[OUT] {rp}\n[OUT] {sp}")
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
