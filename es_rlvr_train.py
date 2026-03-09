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
    bonus = entropy_coeff × entropy_proxy
  True Shannon entropy requires full-vocabulary logprobs, which vLLM
  does not expose cheaply.  We use the tractable per-token proxy:
    entropy_proxy = −mean(log p_chosen_token)   (over response tokens)
  Higher proxy → model more uncertain / exploratory, same qualitative
  effect as the white-box entropy term in OneShot-RLVR.
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

import numpy as np
import pandas as pd
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from deepscaler import compute_score, SYSTEM_PROMPT


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
                        default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    # Data
    parser.add_argument("--parquet_path", type=str,
                        default="Dataset parquet/pi1_r128.parquet",
                        help="Train parquet (verl schema). One-shot: pi1_r128.")
    parser.add_argument("--val_parquet_path", type=str,
                        default="Dataset parquet/math500.parquet",
                        help="Validation parquet (verl schema). math500.")
    parser.add_argument("--val_batch_size", type=int, default=50,
                        help="Number of val examples to evaluate each val step.")
    # ES
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--antithetic", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use antithetic (±ε) pairs (default: True).")
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
        task_datas.append({
            "prompt_str":   prompt_str,
            "ground_truth": str(gt),
            "data_source":  ds,
        })
    return task_datas


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_handle(llm, task_datas: list, temperature: float = 0.7,
                    max_tokens: int = 2048):
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
        logprobs=1,          # entropy proxy: −mean(log p_chosen)
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


def _entropy_proxy(output_obj) -> float:
    """
    Per-token entropy proxy from vLLM logprobs=1.

    vLLM returns the logprob of the sampled token at each step.
    entropy_proxy = −mean(log p_chosen)  over response tokens.

    This equals the average per-token NLL (negative log-likelihood) under
    the sampling policy — a tractable proxy for true Shannon entropy.
    Higher value → model more uncertain / exploratory.

    Note: True H(π) = −Σ_v π(v) log π(v) requires full-vocabulary
    logprobs.  We use this proxy because vLLM exposes only top-k logprobs.
    The qualitative gradient direction is the same: maximising this proxy
    encourages higher-entropy (more diverse) responses.
    """
    completion = output_obj.outputs[0]
    if not completion.logprobs:
        return 0.0

    chosen_lps = []
    for lp_dict in completion.logprobs:
        if not lp_dict:
            continue
        lp_val = next(iter(lp_dict.values()))
        # vLLM Logprob object exposes .logprob; fall back to float() for
        # older versions that return a plain number.
        chosen_lps.append(
            lp_val.logprob if hasattr(lp_val, "logprob") else float(lp_val)
        )

    return float(-np.mean(chosen_lps)) if chosen_lps else 0.0


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
      scores          — list of total rewards (used for GRPO/ES normalisation)
      correctness_scores — list of binary rewards (for logging)
      avg_reward      — mean total reward across batch
      avg_correctness — mean binary correctness across batch
      avg_entropy     — mean entropy proxy across batch
    """
    correctness_scores = []
    total_scores = []
    entropy_vals = []

    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        completion = output.outputs[0].text
        binary_reward = float(compute_score(
            data_source=data.get("data_source", "deepscaler"),
            solution_str=completion,
            ground_truth=data["ground_truth"],
            extra_info=None,
            use_think=False,
        ))
        ent = _entropy_proxy(output)
        total = binary_reward + entropy_coeff * ent

        correctness_scores.append(binary_reward)
        entropy_vals.append(ent)
        total_scores.append(total)

        if debug_print and idx == 0:
            tail = completion[-600:] if len(completion) > 600 else completion
            boxed = _extract_boxed(completion)
            print("\n" + "=" * 70)
            print("[TRAIN OUTPUT] Sample response:")
            print(f"  Ground Truth    : {data['ground_truth']}")
            print(f"  Extracted Answer: {boxed if boxed else '(none — no \\boxed{})'}")
            print(f"  Correctness     : {binary_reward}")
            print(f"  Entropy Proxy   : {ent:.4f}")
            if entropy_coeff > 0.0:
                print(f"  Entropy Bonus   : {entropy_coeff * ent:.6f}")
                print(f"  Total Reward    : {total:.4f}")
            print(f"  Response (tail) :\n    ...{tail}")
            print("=" * 70 + "\n")

    return {
        "scores":             total_scores,
        "correctness_scores": correctness_scores,
        "avg_reward":         float(np.mean(total_scores))       if total_scores else 0.0,
        "avg_correctness":    float(np.mean(correctness_scores)) if correctness_scores else 0.0,
        "avg_entropy":        float(np.mean(entropy_vals))       if entropy_vals else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_val_set(engine, val_task_datas: list, val_batch_size: int,
                     iteration: int, writer, verbose: bool = False) -> float:
    """
    On-the-go greedy evaluation on math500 (or any val set).

    Uses engine 0, which holds the current best weights after each ES update
    and NCCL broadcast.  Greedy decoding (temperature=0) for deterministic
    accuracy.
    """
    batch = val_task_datas[:val_batch_size]
    prompts = [d["prompt_str"] for d in batch]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    print(f"\n[VAL] Iter {iteration}: evaluating {len(batch)} examples (greedy)...")
    outputs = ray.get(engine.generate.remote(prompts, sampling_params, use_tqdm=False))

    correct = 0.0
    for output, data in zip(outputs, batch):
        correct += float(compute_score(
            data_source=data.get("data_source", "deepscaler"),
            solution_str=output.outputs[0].text,
            ground_truth=data["ground_truth"],
            use_think=False,
        ))

    accuracy = correct / len(batch)
    print(f"[VAL] accuracy = {accuracy:.4f}  ({int(correct)}/{len(batch)})")

    if verbose and outputs:
        completion = outputs[0].outputs[0].text
        tail = completion[-400:] if len(completion) > 400 else completion
        boxed = _extract_boxed(completion)
        print(f"\n[VAL SAMPLE]  gt={batch[0]['ground_truth']}"
              f"  pred={boxed if boxed else '(none)'}")
        print(f"  ...{tail}\n")

    if writer is not None:
        writer.add_scalar("val/accuracy", accuracy, iteration)
    return accuracy


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
        val_task_datas = load_task_datas(args.val_parquet_path, tokenizer)
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
        debug_this_gen = (i % 10 == 0)  # print output sample every 10 iters
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
                "time":            elapsed,
            })

            # Restore engine weights
            llm = meta["engine"]
            ray.get(llm.collective_rpc.remote(
                "restore_self_weights",
                args=(meta["seed"], args.sigma, args.iid_noise),
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

        mean_r = float(np.mean(all_rewards))      if all_rewards else 0.0
        std_r  = float(np.std(all_rewards))       if all_rewards else 0.0
        mean_c = float(np.mean(all_correctness))  if all_correctness else 0.0
        mean_e = float(np.mean(all_entropy))      if all_entropy else 0.0

        print(f"\n[REWARD]  mean={mean_r:.4f}  std={std_r:.4f}  "
              f"correctness={mean_c:.4f}  entropy_proxy={mean_e:.4f}")

        for key, v in seeds_perf.items():
            v["norm_reward"] = (v["avg_reward"] - mean_r) / (std_r + 1e-8)
            if args.verbose:
                s, neg = key
                print(f"    seed={s} {'(-)' if neg else '(+)'}: "
                      f"reward={v['avg_reward']:.4f}  "
                      f"norm={v['norm_reward']:.4f}  "
                      f"correct={v['avg_correctness']:.4f}  "
                      f"ent={v['avg_entropy']:.4f}")

        # Log to TensorBoard
        writer.add_scalar("reward/mean",        mean_r, i)
        writer.add_scalar("reward/std",         std_r,  i)
        writer.add_scalar("reward/correctness", mean_c, i)
        writer.add_scalar("reward/entropy",     mean_e, i)
        if results_this_gen:
            writer.add_scalar(
                "reward/max_correctness",
                max(r["avg_correctness"] for r in results_this_gen), i,
            )

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
            evaluate_val_set(
                engines[0], val_task_datas, args.val_batch_size,
                i, writer, args.verbose,
            )

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

    cleanup()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
