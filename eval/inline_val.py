"""Lightweight validation pass called from train/es_rlvr_train.py when --val_every > 0."""

import json
import os
import time
from typing import Optional

import numpy as np
import ray
from vllm import SamplingParams

from reward.deepscaler import compute_training_score
from reward.reward_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def _score_reason(completion: str, ground_truth: str) -> dict:
    """Diagnose why a sample got its score. Does not change official score logic."""
    extracted_answer = extract_answer(completion)
    boxed_found = "\\boxed" in completion

    gt = str(ground_truth)
    gt_for_compare = extract_answer(gt) if "\\boxed" in gt else gt
    if gt_for_compare is None:
        gt_for_compare = gt

    if extracted_answer is None:
        return {
            "boxed_found": boxed_found,
            "parse_ok": False,
            "extracted_answer": None,
            "score_reason": "no_box_or_parse_fail",
        }

    is_correct = bool(
        grade_answer_mathd(extracted_answer, gt_for_compare)
        or grade_answer_sympy(extracted_answer, gt_for_compare)
    )
    return {
        "boxed_found": boxed_found,
        "parse_ok": True,
        "extracted_answer": extracted_answer,
        "score_reason": "correct" if is_correct else "mismatch",
    }


def run_inline_val(engine, val_task_datas, out_dir, run_tag, iteration,
                   temperature=0.6, max_tokens=4096, sampling_seed: Optional[int] = None):
    prompts = [d["prompt_str"] for d in val_task_datas]

    sampling_kwargs = dict(temperature=temperature, max_tokens=max_tokens)
    if sampling_seed is not None:
        sampling_kwargs["seed"] = sampling_seed

    t0 = time.time()
    outputs = ray.get(engine.generate.remote(
        prompts, SamplingParams(**sampling_kwargs), use_tqdm=False
    ))
    elapsed = time.time() - t0

    scores, records = [], []
    for idx, (output, data) in enumerate(zip(outputs, val_task_datas)):
        completion  = output.outputs[0].text
        num_tokens  = len(output.outputs[0].token_ids)
        score       = compute_training_score(completion, data["ground_truth"])
        scores.append(score)
        diag = _score_reason(completion, data["ground_truth"])
        records.append({
            "iter": iteration, "idx": idx,
            "question":       data.get("question", ""),
            "ground_truth":   data["ground_truth"],
            "model_response": completion,
            "score":          score,
            "num_tokens":     num_tokens,
            "response_len":   len(completion),
            **diag,
        })

    accuracy = float(np.mean(scores)) if scores else 0.0
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"val_{run_tag}_iter{iteration:04d}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    num_correct   = int(round(sum(scores)))
    num_parse_ok  = sum(int(r["parse_ok"])    for r in records)
    num_boxed     = sum(int(r["boxed_found"]) for r in records)
    num_wrong     = len(records) - num_correct
    parse_ok_frac = num_parse_ok / len(records) if records else 0.0
    boxed_frac    = num_boxed    / len(records) if records else 0.0
    mean_resp_len = float(np.mean([r["response_len"] for r in records])) if records else 0.0
    mean_tokens   = float(np.mean([r["num_tokens"]   for r in records])) if records else 0.0
    min_tokens    = int(min(r["num_tokens"] for r in records)) if records else 0
    max_tokens    = int(max(r["num_tokens"] for r in records)) if records else 0

    print(f"\n{'='*60}")
    print(f"  [VAL] iter={iteration}  T={temperature}  elapsed={elapsed:.1f}s")
    print(f"  Accuracy   : {accuracy*100:.2f}%  ({num_correct} correct / {len(scores)} total / {num_wrong} wrong)")
    print(f"  Parse OK   : {num_parse_ok}/{len(records)}  ({parse_ok_frac*100:.1f}%)")
    print(f"  Boxed      : {num_boxed}/{len(records)}  ({boxed_frac*100:.1f}%)")
    print(f"  Tokens     : mean={mean_tokens:.1f}  min={min_tokens}  max={max_tokens}")
    print(f"  Char len   : mean={mean_resp_len:.0f}")
    print(f"  Saved      : {out_path}")
    print(f"{'='*60}\n")

    return {
        "accuracy":          accuracy,
        "parse_ok_frac":     parse_ok_frac,
        "boxed_frac":        boxed_frac,
        "mean_response_len": mean_resp_len,
        "mean_tokens":       mean_tokens,
        "elapsed":           elapsed,
    }
