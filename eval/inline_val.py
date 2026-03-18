"""
inline_val.py — Lightweight validation pass called from the training loop.

Runs greedy inference on a single vLLM engine (engine 0, already warmed up)
and reports math500 accuracy.  No Ray, no NCCL — just a direct .generate call
on the engine actor.

Called from train/es_rlvr_train.py when --val_every > 0.
"""

import json
import os
import time

import numpy as np
from vllm import SamplingParams

from reward.deepscaler import compute_training_score


def run_inline_val(
    engine,
    val_task_datas: list,
    out_dir: str,
    run_tag: str,
    iteration: int,
    temperature: float = 0.0,
    max_tokens: int = 3072,
) -> float:
    """
    Run greedy inference on val_task_datas using a live vLLM engine actor.

    Args:
        engine:          Ray actor handle for vLLM engine 0.
        val_task_datas:  List of dicts with keys prompt_str / ground_truth.
        out_dir:         Directory to write per-iteration JSONL results.
        run_tag:         Run timestamp string (for file naming).
        iteration:       Current training iteration index.
        temperature:     Sampling temperature (0.0 = greedy).
        max_tokens:      Max generation tokens per problem.

    Returns:
        Accuracy in [0, 1].
    """
    prompts = [d["prompt_str"] for d in val_task_datas]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    import ray
    t0 = time.time()
    outputs = ray.get(engine.generate.remote(prompts, sampling_params, use_tqdm=False))
    elapsed = time.time() - t0

    scores = []
    records = []
    for idx, (output, data) in enumerate(zip(outputs, val_task_datas)):
        completion = output.outputs[0].text
        score = compute_training_score(completion, data["ground_truth"])
        scores.append(score)
        records.append({
            "iter":             iteration,
            "idx":              idx,
            "question":         data.get("question", ""),
            "ground_truth":     data["ground_truth"],
            "model_response":   completion,
            "score":            score,
        })

    accuracy = float(np.mean(scores)) if scores else 0.0

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"val_{run_tag}_iter{iteration:04d}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_correct = int(round(sum(scores)))
    print(f"[VAL] iter={iteration}  acc={accuracy*100:.2f}%  "
          f"({n_correct}/{len(scores)})  time={elapsed:.1f}s  → {out_path}")
    return accuracy
