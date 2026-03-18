"""Lightweight validation pass called from train/es_rlvr_train.py when --val_every > 0."""

import json
import os
import time

import numpy as np
import ray
from vllm import SamplingParams

from reward.deepscaler import compute_training_score


def run_inline_val(engine, val_task_datas, out_dir, run_tag, iteration,
                   temperature=0.0, max_tokens=3072):
    prompts = [d["prompt_str"] for d in val_task_datas]
    t0 = time.time()
    outputs = ray.get(engine.generate.remote(
        prompts, SamplingParams(temperature=temperature, max_tokens=max_tokens), use_tqdm=False
    ))
    elapsed = time.time() - t0

    scores, records = [], []
    for idx, (output, data) in enumerate(zip(outputs, val_task_datas)):
        completion = output.outputs[0].text
        score = compute_training_score(completion, data["ground_truth"])
        scores.append(score)
        records.append({
            "iter": iteration, "idx": idx,
            "question":       data.get("question", ""),
            "ground_truth":   data["ground_truth"],
            "model_response": completion,
            "score":          score,
        })

    accuracy = float(np.mean(scores)) if scores else 0.0
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"val_{run_tag}_iter{iteration:04d}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[VAL] iter={iteration}  acc={accuracy*100:.2f}%  "
          f"({int(round(sum(scores)))}/{len(scores)})  {elapsed:.1f}s  → {out_path}")
    return accuracy
