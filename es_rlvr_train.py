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
- Entropy tracked as a diagnostic metric (top-20 logprobs approximation,
  lower-bound on Shannon entropy) but NOT added to the reward signal.
  Binary correctness is the sole reward: total = binary_reward.
- KL penalty: omitted.  Computing low_var_kl requires a reference model
  forward pass.  In a pure vLLM inference architecture there is no
  autograd model resident in memory, so this term is not feasible
  without a separate HuggingFace inference pass per perturbation.
- Model output display during training (every iter by default).

ES weight update (standard, matches VsonicV countdown_accl)
------------------------------------------------------------
For each seed sᵢ with normalised reward Aᵢ:
  Δθ += (α/N) × Aᵢ × εᵢ
Applied via WorkerExtension.perturb_self_weights(seed, coeff) on engine 0,
then NCCL-broadcast to all engines.
"""

import argparse

from datetime import datetime
import gc
import json
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

from deepscaler import compute_training_score


# ─────────────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

SIGMA            = 0.001          # perturbation scale σ
ALPHA            = 0.0005         # ES learning rate α
POPULATION_SIZE  = 20             # number of perturbations per iteration
NUM_ENGINES      = 4              # parallel vLLM engines (= GPUs)
NUM_ITERATIONS   = 200
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
    # ES
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    # Training
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--global_seed", type=int, default=1234)
    parser.add_argument("--output_every", type=int, default=1,
                        help="Print prompt+response sample every N training iterations (default: 1 = every iter).")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.output_every <= 0:
        raise ValueError("--output_every must be >= 1")

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
            "Pass the correct path via --parquet_path"
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
        prompt_str = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
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
                    max_tokens: int = 4032):
    """
    Launch an async vLLM generation on llm.
    logprobs=1 → per-token logprob of the sampled token, used as entropy proxy.
    Returns (object_ref, start_timestamp).
    """
    prompts = [d["prompt_str"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=temperature,
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
                         debug_print: bool = False) -> dict:
    """
    Compute per-sample rewards and collect all sample records for JSONL logging.

    Reward: binary correctness only.
      binary_reward = compute_score(...)  → 1.0 or 0.0

    Entropy is computed and returned as a diagnostic metric but does NOT
    contribute to the reward signal used for ES normalisation.

    Returns:
      scores             — list of binary rewards (used for GRPO/ES normalisation)
      correctness_scores — list of binary rewards (for logging)
      avg_reward         — mean binary reward across batch
      avg_correctness    — mean binary correctness across batch
      avg_entropy        — mean per-token Shannon entropy (diagnostic only)
      avg_coverage       — mean top-20 probability mass coverage (diagnostic)
      samples            — list of dicts, one per prompt/response pair (for JSONL)
    """
    correctness_scores = []
    total_scores = []
    entropy_vals = []
    coverage_vals = []
    samples = []

    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        completion = output.outputs[0].text
        binary_reward = compute_training_score(completion, data["ground_truth"])
        ent, cov = _compute_token_entropy(output)

        correctness_scores.append(binary_reward)
        entropy_vals.append(ent)
        coverage_vals.append(cov)
        total_scores.append(binary_reward)

        samples.append({
            "sample_idx":       idx,
            "prompt_str":       data.get("prompt_str", ""),
            "question":         data.get("question", ""),
            "model_response":   completion,
            "extracted_answer": _extract_boxed(completion) or "",
            "ground_truth":     data.get("ground_truth", ""),
            "binary_reward":    binary_reward,
            "entropy":          ent,
            "entropy_coverage": cov,
        })

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
            extracted = boxed if boxed else "(none — no boxed answer)"
            print(f"  Extracted Answer: {extracted}")
            print(f"  Result          : {correct_str}  (binary={binary_reward:.1f})")
            print(f"  Entropy (top-20) : {ent:.4f}  coverage={cov:.4f}")
            print("=" * 70 + "\n")

    return {
        "scores":             total_scores,
        "correctness_scores": correctness_scores,
        "avg_reward":         float(np.mean(total_scores))       if total_scores       else 0.0,
        "avg_correctness":    float(np.mean(correctness_scores)) if correctness_scores else 0.0,
        "avg_entropy":        float(np.mean(entropy_vals))       if entropy_vals       else 0.0,
        "avg_coverage":       float(np.mean(coverage_vals))      if coverage_vals      else 0.0,
        "samples":            samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


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
    print(f"  Population     : {args.population_size}  (standard ES)")
    print(f"  σ (sigma)      : {args.sigma}")
    print(f"  α (alpha)      : {args.alpha}")
    print(f"  Engines (GPUs) : {args.num_engines}")
    print(f"  Iterations     : {args.num_iterations}")
    print(f"  Train examples : {len(train_task_datas)}")
    print(f"  Log dir        : {logging_dir}")
    print("=" * 70 + "\n")

    train_outputs_jsonl = os.path.join(logging_dir, "train_outputs.jsonl")

    # ──────────────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────────────
    for i in range(args.num_iterations):
        print(f"\n{'─' * 50}")
        print(f"  Generation {i} / {args.num_iterations - 1}")
        print(f"{'─' * 50}")
        total_iter_start = time.time()

        # ── Build seed list for this generation ───────────────────────────────
        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]

        seeds_perf: dict = {}            # seed → metrics
        debug_this_gen = (i % args.output_every == 0)
        debug_fired    = False

        seed_iter = iter(seeds)
        inflight:  dict = {}             # object_ref → metadata
        results_this_gen = []

        # ── Prime all engines ─────────────────────────────────────────────────
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            ray.get(llm.collective_rpc.remote(
                "perturb_self_weights",
                args=(seed, args.sigma, False),
            ))
            handle, start_ts = evaluate_handle(llm, train_task_datas)
            inflight[handle] = {
                "engine":     llm,
                "engine_idx": eng_idx,
                "seed":       seed,
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
                debug_print=do_debug,
            )
            if do_debug:
                debug_fired = True

            elapsed = time.time() - meta["start_ts"]
            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append({
                "seed":            meta["seed"],
                "avg_reward":      metrics["avg_reward"],
                "avg_correctness": metrics["avg_correctness"],
                "avg_entropy":     metrics["avg_entropy"],
                "avg_coverage":    metrics["avg_coverage"],
                "time":            elapsed,
            })

            # Restore engine weights
            llm = meta["engine"]
            ray.get(llm.collective_rpc.remote(
                "restore_self_weights",
                args=(meta["seed"], args.sigma),
            ))

            # Schedule next seed on this engine
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote(
                "perturb_self_weights",
                args=(next_seed, args.sigma, False),
            ))
            handle, start_ts = evaluate_handle(llm, train_task_datas)
            inflight[handle] = {
                "engine":     llm,
                "engine_idx": meta["engine_idx"],
                "seed":       next_seed,
                "start_ts":   start_ts,
            }
            if args.verbose:
                print(f"  Scheduled seed {next_seed} → engine {meta['engine_idx']}")

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

        for s, v in seeds_perf.items():
            v["norm_reward"] = (v["avg_reward"] - mean_r) / (std_r + 1e-8)
            if args.verbose:
                print(f"    seed={s}: "
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

        # ── Write all train outputs to JSONL (all seeds, all samples) ───────────
        with open(train_outputs_jsonl, "a", encoding="utf-8") as _jf:
            for s, v in seeds_perf.items():
                for sample in v.get("samples", []):
                    record = {"iter": i, "seed": s}
                    record.update(sample)
                    _jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ── ES weight update on engine 0 ──────────────────────────────────────
        # Standard ES: Δθ += (α/N) × Aᵢ × εᵢ
        perturb_start = time.time()
        handles = []

        for s in seeds:
            norm  = seeds_perf.get(s, {}).get("norm_reward", 0.0)
            coeff = (args.alpha / args.population_size) * norm
            handles.append(engines[0].collective_rpc.remote(
                "perturb_self_weights",
                args=(s, coeff, False),
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
