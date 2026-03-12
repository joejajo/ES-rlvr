#!/usr/bin/env python3
"""
ES-RLVR: Evolution Strategies + One-Shot-RLVR with vLLM + Ray + NCCL
======================================================================
Architecture: adapted from VsonicV/es-fine-tuning-paper
Methodology:  adapted from ypwang61/One-Shot-RLVR (NeurIPS 2025)

One-Shot-RLVR design principles preserved
------------------------------------------
- Binary correctness reward ONLY (deepscaler.compute_score).
  No format reward — 0.0 if no \\boxed{} found, full stop.
- GRPO reward normalisation (z-score across population):
    Aᵢ = (rᵢ − mean(R)) / (std(R) + ε)
  This is mathematically identical to the ES z-score normalisation —
  both formulas are the same; GRPO normalises across group rollouts,
  ES normalises across population perturbations.
- Entropy bonus as a separate term (coeff=0.001):
    bonus = entropy_coeff × H_approx
  The paper (verl) computes exact Shannon entropy from full-vocabulary
  logits: H = logsumexp(X) − Σ_v p_v X_v.  vLLM exposes only top-k
  log-softmax values, not raw logits; a direct forward pass requires
  vLLM-internal attention metadata (FlashInferMetadata) that is
  version-tied and fragile to build externally.
  We use a tight lower-bound approximation with top-20 logprobs (vLLM max):
    H_approx = −Σ_{i=1}^{20} p_i log p_i  −  p_tail log p_tail
  where p_tail = 1 − Σ p_i.  The tail bucket makes this a lower bound
  on true Shannon entropy rather than ignoring missing mass entirely.
  The gradient direction is exact.
- KL penalty: omitted.  Computing low_var_kl requires a reference model
  forward pass.  In a pure vLLM inference architecture there is no
  autograd model resident in memory, so this term is not feasible
  without a separate HuggingFace inference pass per perturbation.
- Antithetic pairs (±ε) for variance reduction.
- On-the-go validation on math500 every val_every iterations.
- Model output display during training (every 10 iters by default).

ES weight update (antithetic)
------------------------------
For each seed sᵢ with normalised rewards A⁺ᵢ, A⁻ᵢ:
  Δθ += (α/N) × (A⁺ᵢ − A⁻ᵢ) × εᵢ
Applied via WorkerExtension.perturb_self_weights(seed, coeff) on engine 0,
then NCCL-broadcast to all engines.
"""

import argparse
from datetime import datetime
import gc
import os
import re
import random
import shutil
import signal
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
import pandas as pd
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
try:
    from vllm.utils import get_ip, get_open_port
except ImportError:
    import socket

    def get_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            s.close()

    def get_open_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

from deepscaler import compute_score, SYSTEM_PROMPT
from utils.os_parser import extract_answer as os_extract_answer, strip_string
from utils.os_grader import math_equal

# ── qwen25-math-cot val prompt (matches One-Shot-RLVR eval exactly) ───────────
# Template: system + all demos + actual question in one user turn.
# 1-shot demo for math500: the wind-pressure problem (same as pi1_r128 train Q).
_QWEN_COT_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
_QWEN_COT_1SHOT_Q = (
    "The pressure \\( P \\) exerted by wind on a sail varies jointly as the area"
    " \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When"
    " the velocity is \\( 8 \\) miles per hour, the pressure on a sail of"
    " \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the"
    " pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds."
    " Let's think step by step and output the final answer within \\boxed{}."
)
_QWEN_COT_1SHOT_A = (
    "We start by writing the mathematical relationship for the pressure \\( P \\):\n"
    "\\[ P = k \\cdot A \\cdot V^3 \\]\n"
    "where \\( k \\) is a constant. We need to find \\( k \\) using the given information:\n"
    "\\[ 4 = k \\cdot 2 \\cdot 8^3 \\]\n"
    "Solving for \\( k \\):\n"
    "\\[ 4 = k \\cdot 2 \\cdot 512 \\]\n"
    "\\[ 4 = 1024k \\]\n"
    "\\[ k = \\frac{4}{1024} \\]\n"
    "\\[ k = \\frac{1}{256} \\]\n"
    "Now we use this value of \\( k \\) to find the velocity \\( V \\) when the"
    " pressure \\( P \\) on 4 square feet of sail is 32 pounds:\n"
    "\\[ 32 = \\frac{1}{256} \\cdot 4 \\cdot V^3 \\]\n"
    "\\[ 32 = \\frac{V^3}{64} \\]\n"
    "\\[ 32 \\cdot 64 = V^3 \\]\n"
    "\\[ 2048 = V^3 \\]\n"
    "\\[ V = \\sqrt[3]{2048} \\]\n"
    "\\[ V = 12.8 \\]\n"
    "Thus, the wind velocity is \\( \\boxed{12.8} \\) miles per hour."
)

def _build_qwen_cot_prompt(question: str) -> str:
    """
    Builds a qwen25-math-cot prompt: system + 1-shot demo + question
    in a single user turn, exactly as One-Shot-RLVR eval does.
    """
    demo = f"{_QWEN_COT_1SHOT_Q}\n\n{_QWEN_COT_1SHOT_A}"
    user_content = f"{demo}\n\n{question}"
    return (
        f"<|im_start|>system\n{_QWEN_COT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

SIGMA            = 0.001          # perturbation scale σ
ALPHA            = 0.0005         # ES learning rate α
POPULATION_SIZE  = 20             # number of perturbations per iteration
NUM_ENGINES      = 4              # parallel vLLM engines (= GPUs)
NUM_ITERATIONS   = 200
ENTROPY_COEFF    = 0.001          # matches OneShot-RLVR actor.entropy_coeff
VAL_EVERY        = 10             # evaluate on math500 every N iterations
EXPERIMENT_DIR   = "outputs/es_rlvr_oneshot"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ES-RLVR: One-Shot-RLVR methodology in a vLLM+Ray+NCCL ES loop"
    )
    # Model
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-Math-1.5B")
    # Data
    parser.add_argument("--parquet_path", type=str,
                        default="Dataset parquet/pi1_r128.parquet",
                        help="Train parquet (verl schema). One-shot: pi1_r128.")
    parser.add_argument("--val_parquet_path", type=str,
                        default="Dataset parquet/math500.parquet",
                        help="Validation parquet (verl schema). math500.")
    parser.add_argument("--val_batch_size", type=int, default=500,
                        help="Number of val examples to evaluate each val step.")
    # ES
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--antithetic", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Use antithetic (±ε) pairs (default: False).")
    parser.add_argument("--iid_noise", action="store_true", default=False,
                        help="Independent noise per parameter (vs shared noise).")
    # Reward
    parser.add_argument("--entropy_coeff", type=float, default=ENTROPY_COEFF,
                        help="Entropy bonus coefficient (One-Shot-RLVR: 0.001).")
    # Training
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--val_every", type=int, default=VAL_EVERY)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--global_seed", type=int, default=None)
    parser.add_argument("--output_every", type=int, default=1,
                        help="Print prompt+response sample every N training iterations (default: 1 = every iter).")
    parser.add_argument("--val_show_n", type=int, default=3,
                        help="Number of Math500 examples to display in full at each val step.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args


# ─────────────────────────────────────────────────────────────────────────────
# vLLM engine setup
# ─────────────────────────────────────────────────────────────────────────────

class ESNcclLLM(LLM):
    """LLM subclass that disables CUDA_VISIBLE_DEVICES and V1 multiprocessing
    before initialising vLLM, required for multi-engine Ray deployment."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines: int, model_path: str):
    """
    Allocate one Ray placement group (1 GPU) per engine, launch ESNcclLLM
    actors with WorkerExtension for in-GPU ES perturbations.
    """
    pgs = [
        placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
        for _ in range(num_engines)
    ]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(
            num_cpus=0, num_gpus=0, scheduling_strategy=strategy
        )(ESNcclLLM).remote(
            model=model_path,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (verl parquet schema)
# ─────────────────────────────────────────────────────────────────────────────

def load_val_task_datas(parquet_path: str) -> list:
    """
    Load val parquet using the qwen25-math-cot prompt format (One-Shot-RLVR eval).
    Does NOT need the tokenizer — prompt is built directly as a raw string.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Parquet not found: {parquet_path}\n"
            "Pass the correct path via --val_parquet_path"
        )
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        raise RuntimeError(f"Parquet has zero rows: {parquet_path}")

    task_datas = []
    for _, row in df.iterrows():
        chat = list(row["prompt"])
        reward_model = row["reward_model"]
        gt = reward_model["ground_truth"] if isinstance(reward_model, dict) else str(reward_model)
        ds = str(row.get("data_source", "deepscaler"))
        # extract raw question from first user message
        question = ""
        for msg in chat:
            if isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content", "")
                break
        prompt_str = _build_qwen_cot_prompt(question)
        task_datas.append({
            "prompt_str":   prompt_str,
            "ground_truth": str(gt),
            "data_source":  ds,
            "question":     question,
        })
    return task_datas


def load_task_datas(parquet_path: str, tokenizer) -> list:
    """
    Load a verl-schema parquet file.

    Expected columns:
      prompt        — list of chat dicts [{role, content}, ...]
      reward_model  — dict with key "ground_truth"
      data_source   — string label

    Returns list of dicts with keys:
      prompt_str    — full formatted prompt (system + chat + generation header)
      ground_truth  — string answer for grading
      data_source   — dataset label
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Parquet not found: {parquet_path}\n"
            "Pass the correct path via --parquet_path / --val_parquet_path"
        )
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        raise RuntimeError(f"Parquet has zero rows: {parquet_path}")

    task_datas = []
    for _, row in df.iterrows():
        chat = list(row["prompt"])
        reward_model = row["reward_model"]
        gt = reward_model["ground_truth"] if isinstance(reward_model, dict) else str(reward_model)
        ds = str(row.get("data_source", "deepscaler"))
        # Prepend system prompt — matching es_fine_tuning_deepscaler_accl.py
        chat_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + chat
        prompt_str = tokenizer.apply_chat_template(
            chat_with_system, tokenize=False, add_generation_prompt=True
        )
        # Extract the raw user question for display (first user-role message)
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
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_handle(llm, task_datas: list, temperature: float = 0.7,
                    max_tokens: int = 3072):
    """
    Launch an async vLLM generation on llm.
    logprobs=1 → per-token logprob of the sampled token, used as entropy proxy.
    Returns (object_ref, start_timestamp).
    """
    prompts = [d["prompt_str"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=temperature,
        seed=42,
        max_tokens=max_tokens,
        logprobs=20,         # top-20 for Shannon entropy approximation (vLLM max)
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation  (One-Shot-RLVR faithful)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_boxed(text: str):
    """Extract last \\boxed{} content — used for display only."""
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text or "")
    return matches[-1].strip() if matches else None


def _compute_token_entropy(output_obj) -> tuple:
    """
    Average per-token Shannon entropy from vLLM top-20 logprobs.

    Matches the One-Shot-RLVR / verl formula:
        H = logsumexp(X) − Σ_v p_v X_v  =  −Σ_v p_v log p_v

    Approximated using top-k log-softmax values returned by vLLM:
        H_approx = −Σ_{i=1}^{k} p_i log p_i  −  p_tail log p_tail

    The tail bucket (p_tail = 1 − Σ p_i) gives a tight lower bound;
    ignoring it entirely would underestimate entropy more.

    Returns:
        (mean_entropy, mean_coverage) over response tokens.
        mean_coverage tracks what fraction of probability mass is in top-k
        — use this to verify top-20 is sufficient for your model.
    """
    completion = output_obj.outputs[0]
    if not getattr(completion, "logprobs", None):
        return 0.0, 0.0

    token_entropies = []
    coverages = []

    for lp_dict in completion.logprobs:
        if not lp_dict:
            continue
        log_probs = []
        for lp_val in lp_dict.values():
            lp = lp_val.logprob if hasattr(lp_val, "logprob") else float(lp_val)
            log_probs.append(lp)
        if not log_probs:
            continue

        log_probs_arr = np.array(log_probs, dtype=np.float64)
        probs = np.exp(log_probs_arr)

        covered = float(np.sum(probs))
        covered = min(max(covered, 0.0), 1.0)
        tail = max(0.0, 1.0 - covered)

        # entropy over returned top-k tokens
        h = float(-np.sum(probs * log_probs_arr))
        # add tail bucket contribution
        if tail > 1e-9:
            h += float(-tail * np.log(tail))

        token_entropies.append(h)
        coverages.append(covered)

    mean_h   = float(np.mean(token_entropies)) if token_entropies else 0.0
    mean_cov = float(np.mean(coverages))       if coverages       else 0.0
    return mean_h, mean_cov


def _postprocess_outputs(outputs, task_datas: list,
                         entropy_coeff: float = 0.0,
                         debug_print: bool = False) -> dict:
    """
    Compute per-sample total rewards.

    One-Shot-RLVR reward design:
      binary_reward = compute_score(...)              → 1.0 or 0.0
      entropy_bonus = entropy_coeff × entropy_proxy   → exploratory signal
      total         = binary_reward + entropy_bonus

    KL penalty is omitted (see module docstring).

    Returns:
      scores             — list of total rewards (used for GRPO/ES normalisation)
      correctness_scores — list of binary rewards (for logging)
      avg_reward         — mean total reward across batch
      avg_correctness    — mean binary correctness across batch
      avg_entropy        — mean per-token Shannon entropy (top-k + tail bucket)
      avg_coverage       — mean top-100 probability mass coverage (diagnostic)
    """
    correctness_scores = []
    total_scores = []
    entropy_vals = []
    coverage_vals = []

    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        completion = output.outputs[0].text
        binary_reward = float(compute_score(
            data_source=data.get("data_source", "deepscaler"),
            solution_str=completion,
            ground_truth=data["ground_truth"],
            extra_info=None,
            use_think=False,
        ))
        ent, cov = _compute_token_entropy(output)
        total = binary_reward + entropy_coeff * ent

        correctness_scores.append(binary_reward)
        entropy_vals.append(ent)
        coverage_vals.append(cov)
        total_scores.append(total)

        if debug_print and idx == 0:
            boxed = _extract_boxed(completion)
            correct_str = "CORRECT" if binary_reward == 1.0 else "WRONG"
            display_resp = completion if len(completion) <= 2000 else completion[:2000] + "\n  ... [truncated]"
            print("\n" + "=" * 70)
            print("[TRAIN] Iteration sample")
            print("─" * 70)
            print("[PROMPT / QUESTION]")
            print(f"  {data.get('question', '(question unavailable)')}")
            print("─" * 70)
            print("[MODEL RESPONSE]")
            for line in display_resp.splitlines():
                print(f"  {line}")
            print("─" * 70)
            print(f"  Ground Truth    : {data['ground_truth']}")
            print(f"  Extracted Answer: {boxed if boxed else '(none — no \\boxed{})'}")
            print(f"  Result          : {correct_str}  (binary={binary_reward:.1f})")
            print(f"  Entropy (top-100): {ent:.4f}  coverage={cov:.4f}")
            if entropy_coeff > 0.0:
                print(f"  Entropy Bonus   : {entropy_coeff * ent:.6f}")
                print(f"  Total Reward    : {total:.4f}")
            print("=" * 70 + "\n")

    return {
        "scores":             total_scores,
        "correctness_scores": correctness_scores,
        "avg_reward":         float(np.mean(total_scores))       if total_scores    else 0.0,
        "avg_correctness":    float(np.mean(correctness_scores)) if correctness_scores else 0.0,
        "avg_entropy":        float(np.mean(entropy_vals))       if entropy_vals    else 0.0,
        "avg_coverage":       float(np.mean(coverage_vals))      if coverage_vals   else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_val_set(engine, val_task_datas: list, val_batch_size: int,
                     iteration: int, writer, val_show_n: int = 3) -> float:
    """
    On-the-go greedy evaluation on math500 (or any val set).

    Uses engine 0, which holds the current best weights after each ES update
    and NCCL broadcast.  Greedy decoding (temperature=0) for deterministic
    accuracy.

    Shows the question, full reasoning chain, extracted answer and
    correct/wrong label for the first val_show_n examples.
    """
    batch = val_task_datas[:val_batch_size] if val_batch_size > 0 else val_task_datas
    prompts = [d["prompt_str"] for d in batch]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=3072)

    print(f"\n{'#' * 70}")
    print(f"  [VAL] Iter {iteration} — Math500 greedy eval on {len(batch)} examples")
    print(f"{'#' * 70}")
    outputs = ray.get(engine.generate.remote(prompts, sampling_params, use_tqdm=False))

    correct = 0.0
    correctness_list = []
    for output, data in zip(outputs, batch):
        pred = os_extract_answer(output.outputs[0].text, data_name="math500")
        gt   = strip_string(str(data["ground_truth"]))
        c    = 1.0 if (pred != "" and math_equal(pred, gt, timeout=False)) else 0.0
        correct += c
        correctness_list.append(c)

    accuracy = correct / len(batch)

    # Show all evaluated examples: compact header + full CoT
    for idx in range(len(batch)):
        data = batch[idx]
        completion = outputs[idx].outputs[0].text
        extracted = os_extract_answer(completion, data_name="math500")
        is_correct = correctness_list[idx] == 1.0
        result_tag = "✓ CORRECT" if is_correct else "✗ WRONG"
        # Full CoT for val_show_n examples; compact (500 chars) for the rest
        if idx < val_show_n:
            display_resp = completion if len(completion) <= 2000 else completion[:2000] + "\n  ... [truncated]"
        else:
            display_resp = completion[:500] + " ... [truncated]" if len(completion) > 500 else completion

        print(f"\n{'─' * 70}")
        print(f"[VAL {idx + 1}/{len(batch)}]  {result_tag}")
        print(f"  Q : {data.get('question', '')[:120]}")
        print(f"  GT: {data['ground_truth']}   PRED: {extracted if extracted else '(none)'}")
        if idx < val_show_n:
            print("─" * 70)
            print("[REASONING CHAIN]")
            for line in display_resp.splitlines():
                print(f"  {line}")
        else:
            print(f"  CoT: {display_resp[:200].replace(chr(10), ' ')}")
        print("─" * 70)

    print(f"\n[VAL] accuracy = {accuracy:.4f}  ({int(correct)}/{len(batch)})")
    print(f"{'#' * 70}\n")

    if writer is not None:
        writer.add_scalar("val/accuracy", accuracy, iteration)
    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(history: dict, output_dir: str):
    """
    Save thesis-quality training curves as a single PNG figure.

    Panels:
      (top-left)     Train correctness vs iteration
      (top-right)    Math500 val accuracy vs iteration  (with baseline marker)
      (bottom-left)  ES reward mean ± std band vs iteration
      (bottom-right) Response entropy proxy vs iteration
    """
    iters = history["iter"]
    if not iters:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "ES-RLVR  ·  One-Shot Training on Single Example → Math500 Generalisation",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Top-left: train correctness ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(iters, history["train_correctness"],
            color="tab:blue", linewidth=1.5, label="train correctness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Correctness")
    ax.set_title("Training Correctness (single example)")
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # ── Top-right: val accuracy ───────────────────────────────────────────────
    ax = axes[0, 1]
    if history["val_iter"]:
        ax.plot(history["val_iter"], history["val_accuracy"],
                color="tab:orange", marker="o", linewidth=2,
                markersize=5, label="Math500 accuracy")
        baseline = history["val_accuracy"][0]
        ax.axhline(baseline, linestyle="--", color="gray", alpha=0.7,
                   label=f"Baseline {baseline:.1%}")
        best = max(history["val_accuracy"])
        ax.axhline(best, linestyle=":", color="tab:green", alpha=0.7,
                   label=f"Best {best:.1%}")
        ax.legend(fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Math500 Validation Accuracy")
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: reward mean ± std ────────────────────────────────────────
    ax = axes[1, 0]
    means = np.array(history["reward_mean"])
    stds  = np.array(history["reward_std"])
    ax.plot(iters, means, color="tab:green", linewidth=1.5, label="mean reward")
    ax.fill_between(iters, means - stds, means + stds,
                    alpha=0.20, color="tab:green", label="±1 std")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Reward (binary + entropy bonus)")
    ax.set_title("ES Population Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: entropy proxy ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(iters, history["entropy"],
            color="tab:purple", linewidth=1.5, label="H (top-100 + tail)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Shannon entropy  H (nats/token)")
    ax.set_title("Response Token Entropy (top-100 approx)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # Clear any stale Ray environment variables
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_dir = f"{args.experiment_dir}/run_{run_tag}"
    writer = SummaryWriter(log_dir=logging_dir)
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    # ── Tokenizer + base model snapshot ──────────────────────────────────────
    # vLLM needs an HF checkpoint directory; we save the base model once and
    # point all engines at it.  The model stays on CPU during this step.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to("cpu")
    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[DATA] Loading train: {args.parquet_path}")
    train_task_datas = load_task_datas(args.parquet_path, tokenizer)
    print(f"[DATA] {len(train_task_datas)} train examples loaded.")

    val_task_datas = None
    if args.val_parquet_path and os.path.exists(args.val_parquet_path):
        print(f"[DATA] Loading val:   {args.val_parquet_path}")
        val_task_datas = load_val_task_datas(args.val_parquet_path)
        print(f"[DATA] {len(val_task_datas)} val examples loaded.")
    else:
        print(f"[DATA] Val path not found ({args.val_parquet_path}), "
              f"skipping validation.")

    # ── Launch engines ────────────────────────────────────────────────────────
    print(f"\n[ENGINES] Launching {args.num_engines} vLLM engines...")
    engines, pgs = launch_engines(args.num_engines, base_model_path)

    # ── NCCL inter-engine communicator ────────────────────────────────────────
    master_address = get_ip()
    master_port    = get_open_port()
    print(f"[NCCL] Init inter-engine group at {master_address}:{master_port}")
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group",
            args=(master_address, master_port, i, args.num_engines),
        )
        for i in range(args.num_engines)
    ])
    print("[NCCL] Ready.\n")

    # ── Signal handlers for clean shutdown ────────────────────────────────────
    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # ── Config summary ────────────────────────────────────────────────────────
    print("=" * 70)
    print("  ES-RLVR: One-Shot-RLVR + vLLM + Ray + NCCL")
    print("=" * 70)
    print(f"  Model          : {args.model_name}")
    print(f"  Population     : {args.population_size}"
          f"  ({'antithetic ±ε pairs' if args.antithetic else 'standard'})")
    print(f"  σ (sigma)      : {args.sigma}")
    print(f"  α (alpha)      : {args.alpha}")
    print(f"  Entropy coeff  : {args.entropy_coeff}")
    print(f"  Engines (GPUs) : {args.num_engines}")
    print(f"  Iterations     : {args.num_iterations}")
    print(f"  Train examples : {len(train_task_datas)}")
    print(f"  Val examples   : {len(val_task_datas) if val_task_datas else 'N/A'}")
    print(f"  Val every      : {args.val_every} iters")
    print(f"  Log dir        : {logging_dir}")
    print("=" * 70 + "\n")

    # ── Metrics history (for plots + CSV) ────────────────────────────────────
    history = {
        "iter":               [],
        "train_correctness":  [],
        "reward_mean":        [],
        "reward_std":         [],
        "entropy":            [],
        "entropy_coverage":   [],
        "val_iter":           [],
        "val_accuracy":       [],
    }
    csv_path = os.path.join(logging_dir, "metrics.csv")
    csv_header_written = False

    # ── Pre-training baseline validation ─────────────────────────────────────
    if val_task_datas:
        print("[BASELINE] Evaluating before any training...")
        baseline_acc = evaluate_val_set(
            engines[0], val_task_datas, args.val_batch_size,
            -1, writer, val_show_n=args.val_show_n,
        )
        history["val_iter"].append(-1)
        history["val_accuracy"].append(baseline_acc)
        with open(csv_path, "w", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=[
                "iter", "train_correctness", "reward_mean",
                "reward_std", "entropy", "entropy_coverage", "val_accuracy",
            ])
            w.writeheader()
            w.writerow({
                "iter": "baseline", "train_correctness": "",
                "reward_mean": "", "reward_std": "",
                "entropy": "", "entropy_coverage": "", "val_accuracy": baseline_acc,
            })
        csv_header_written = True

    # ──────────────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────────────
    for i in range(args.num_iterations):
        print(f"\n{'─' * 50}")
        print(f"  Generation {i} / {args.num_iterations - 1}")
        print(f"{'─' * 50}")
        total_iter_start = time.time()

        # ── Build seed list for this generation ───────────────────────────────
        # Antithetic pairs: N/2 seeds, each yielding (seed, False) and
        # (seed, True) for +ε and −ε perturbations respectively.
        if args.antithetic:
            n_seeds = args.population_size // 2
            base_seeds = [random.randint(0, 1_000_000) for _ in range(n_seeds)]
            seed_tasks = []
            for s in base_seeds:
                seed_tasks.append((s, False))   # +ε
                seed_tasks.append((s, True))    # −ε
        else:
            base_seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
            seed_tasks = [(s, False) for s in base_seeds]

        seeds_perf: dict = {}            # (seed, negate) → metrics
        debug_this_gen = (i % args.output_every == 0)
        debug_fired    = False

        seed_iter = iter(seed_tasks)
        inflight:  dict = {}             # object_ref → metadata
        results_this_gen = []

        # ── Prime all engines ─────────────────────────────────────────────────
        for eng_idx, llm in enumerate(engines):
            try:
                seed, negate = next(seed_iter)
            except StopIteration:
                break
            ray.get(llm.collective_rpc.remote(
                "perturb_self_weights",
                args=(seed, args.sigma, negate, args.iid_noise),
            ))
            handle, start_ts = evaluate_handle(llm, train_task_datas)
            inflight[handle] = {
                "engine":     llm,
                "engine_idx": eng_idx,
                "seed":       seed,
                "negate":     negate,
                "start_ts":   start_ts,
            }

        # ── Round-robin pipeline ──────────────────────────────────────────────
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h    = done[0]
            meta = inflight.pop(h)

            outputs     = ray.get(h)
            do_debug    = debug_this_gen and not debug_fired
            metrics     = _postprocess_outputs(
                outputs, train_task_datas,
                entropy_coeff=args.entropy_coeff,
                debug_print=do_debug,
            )
            if do_debug:
                debug_fired = True

            elapsed = time.time() - meta["start_ts"]
            key = (meta["seed"], meta["negate"])
            seeds_perf[key] = metrics
            results_this_gen.append({
                "seed":            meta["seed"],
                "negate":          meta["negate"],
                "avg_reward":      metrics["avg_reward"],
                "avg_correctness": metrics["avg_correctness"],
                "avg_entropy":     metrics["avg_entropy"],
                "avg_coverage":    metrics["avg_coverage"],
                "time":            elapsed,
            })

            # Restore engine weights (pass negate so -ε perturbations are correctly undone)
            llm = meta["engine"]
            ray.get(llm.collective_rpc.remote(
                "restore_self_weights",
                args=(meta["seed"], args.sigma, args.iid_noise, meta["negate"]),
            ))

            # Schedule next seed on this engine
            try:
                next_seed, next_negate = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote(
                "perturb_self_weights",
                args=(next_seed, args.sigma, next_negate, args.iid_noise),
            ))
            handle, start_ts = evaluate_handle(llm, train_task_datas)
            inflight[handle] = {
                "engine":     llm,
                "engine_idx": meta["engine_idx"],
                "seed":       next_seed,
                "negate":     next_negate,
                "start_ts":   start_ts,
            }
            if args.verbose:
                sign = "(-)" if next_negate else "(+)"
                print(f"  Scheduled seed {next_seed} {sign} → engine {meta['engine_idx']}")

        # ── GRPO / ES reward normalisation ────────────────────────────────────
        # Aᵢ = (rᵢ − mean) / (std + ε)
        # Identical to OneShot-RLVR compute_grpo_outcome_advantage.
        all_rewards     = [v["avg_reward"]      for v in seeds_perf.values()]
        all_correctness = [v["avg_correctness"] for v in seeds_perf.values()]
        all_entropy     = [v["avg_entropy"]     for v in seeds_perf.values()]
        all_coverage    = [v["avg_coverage"]    for v in seeds_perf.values()]

        mean_r   = float(np.mean(all_rewards))      if all_rewards     else 0.0
        std_r    = float(np.std(all_rewards))       if all_rewards     else 0.0
        mean_c   = float(np.mean(all_correctness))  if all_correctness else 0.0
        mean_e   = float(np.mean(all_entropy))      if all_entropy     else 0.0
        mean_cov = float(np.mean(all_coverage))     if all_coverage    else 0.0

        print(f"\n[REWARD]  mean={mean_r:.4f}  std={std_r:.4f}  "
              f"correctness={mean_c:.4f}  entropy={mean_e:.4f}  "
              f"coverage={mean_cov:.4f}")

        for key, v in seeds_perf.items():
            v["norm_reward"] = (v["avg_reward"] - mean_r) / (std_r + 1e-8)
            if args.verbose:
                s, neg = key
                print(f"    seed={s} {'(-)' if neg else '(+)'}: "
                      f"reward={v['avg_reward']:.4f}  "
                      f"norm={v['norm_reward']:.4f}  "
                      f"correct={v['avg_correctness']:.4f}  "
                      f"ent={v['avg_entropy']:.4f}  "
                      f"cov={v['avg_coverage']:.4f}")

        # Log to TensorBoard
        writer.add_scalar("reward/mean",              mean_r,   i)
        writer.add_scalar("reward/std",               std_r,    i)
        writer.add_scalar("reward/correctness",       mean_c,   i)
        writer.add_scalar("reward/entropy",           mean_e,   i)
        writer.add_scalar("reward/entropy_coverage",  mean_cov, i)
        if results_this_gen:
            writer.add_scalar(
                "reward/max_correctness",
                max(r["avg_correctness"] for r in results_this_gen), i,
            )

        # ── Update metrics history + CSV ──────────────────────────────────────
        history["iter"].append(i)
        history["train_correctness"].append(mean_c)
        history["reward_mean"].append(mean_r)
        history["reward_std"].append(std_r)
        history["entropy"].append(mean_e)
        history["entropy_coverage"].append(mean_cov)

        csv_row = {
            "iter":              i,
            "train_correctness": mean_c,
            "reward_mean":       mean_r,
            "reward_std":        std_r,
            "entropy":           mean_e,
            "entropy_coverage":  mean_cov,
            "val_accuracy":      "",   # filled in at val steps
        }
        write_mode = "w" if not csv_header_written else "a"
        with open(csv_path, write_mode, newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=list(csv_row.keys()))
            if not csv_header_written:
                w.writeheader()
                csv_header_written = True
            w.writerow(csv_row)

        # ── ES weight update on engine 0 ──────────────────────────────────────
        # Antithetic: Δθ += (α/N) × (A⁺ᵢ − A⁻ᵢ) × εᵢ
        # Standard:   Δθ += (α/N) × Aᵢ × εᵢ
        perturb_start = time.time()
        handles = []

        if args.antithetic:
            for s in base_seeds:
                norm_pos = seeds_perf.get((s, False), {}).get("norm_reward", 0.0)
                norm_neg = seeds_perf.get((s, True),  {}).get("norm_reward", 0.0)
                # Combined antithetic coefficient: (A⁺ − A⁻) / N × α
                coeff = (args.alpha / args.population_size) * (norm_pos - norm_neg)
                if coeff != 0.0:
                    handles.append(engines[0].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(s, coeff, False, args.iid_noise),
                    ))
        else:
            for s, neg in seed_tasks:
                norm  = seeds_perf.get((s, neg), {}).get("norm_reward", 0.0)
                coeff = (args.alpha / args.population_size) * norm
                if coeff != 0.0:
                    handles.append(engines[0].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(s, coeff, neg, args.iid_noise),
                    ))

        ray.get(handles)
        t_perturb = time.time() - perturb_start
        if args.verbose:
            print(f"  ES update applied in {t_perturb:.2f}s")
        writer.add_scalar("time/perturbation_application", t_perturb, i)

        # ── NCCL broadcast: engine 0 → all engines ────────────────────────────
        broadcast_start = time.time()
        ray.get([
            e.collective_rpc.remote("broadcast_all_weights", args=(0,))
            for e in engines
        ])
        t_broadcast = time.time() - broadcast_start
        if args.verbose:
            print(f"  Broadcast in {t_broadcast:.2f}s")
        writer.add_scalar("time/broadcast", t_broadcast, i)

        # ── Iteration timing ──────────────────────────────────────────────────
        t_iter = time.time() - total_iter_start
        writer.add_scalar("time/iteration", t_iter, i)
        print(f"[ITER] Wall clock: {t_iter:.1f}s")
        print(f"  Generation {i} done.\n")

        # ── On-the-go validation ──────────────────────────────────────────────
        if val_task_datas and (
            i % args.val_every == 0 or i == args.num_iterations - 1
        ):
            val_acc = evaluate_val_set(
                engines[0], val_task_datas, args.val_batch_size,
                i, writer, val_show_n=args.val_show_n,
            )
            history["val_iter"].append(i)
            history["val_accuracy"].append(val_acc)
            # Update CSV val_accuracy column for this iter (append a marker row)
            with open(csv_path, "a", newline="") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=[
                    "iter", "train_correctness", "reward_mean",
                    "reward_std", "entropy", "entropy_coverage", "val_accuracy",
                ])
                w.writerow({
                    "iter": f"val@{i}", "train_correctness": "",
                    "reward_mean": "", "reward_std": "",
                    "entropy": "", "entropy_coverage": "", "val_accuracy": val_acc,
                })
            save_plots(history, logging_dir)

    # ── Save final model weights ──────────────────────────────────────────────
    final_path = f"{model_saves_dir}/final_model_iter_{args.num_iterations}"
    os.makedirs(final_path, exist_ok=True)
    ray.get(
        engines[0].collective_rpc.remote(
            "save_self_weights_to_disk",
            args=(f"{final_path}/pytorch_model.pth",),
        )
    )
    print(f"\n[SAVE] Final weights saved to {final_path}/pytorch_model.pth")

    save_plots(history, logging_dir)
    print(f"[CSV]  Metrics log → {csv_path}")

    cleanup()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
