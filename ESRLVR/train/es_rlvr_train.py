#!/usr/bin/env python3
"""
ES-RLVR: Evolution Strategies + One-Shot-RLVR with vLLM + Ray + NCCL
======================================================================
Architecture: adapted from VsonicV/es-fine-tuning-paper
Methodology:  adapted from ypwang61/One-Shot-RLVR (NeurIPS 2025)

One-Shot-RLVR design principles
---------------------------------
- Binary correctness reward ONLY (reward.deepscaler.compute_training_score).
  No format reward — 0.0 if no \\boxed{} found.
- GRPO reward normalisation (z-score across population):
    Aᵢ = (rᵢ − mean(R)) / (std(R) + ε)
  Identical to ES z-score normalisation across perturbations.
- Entropy tracked as a diagnostic metric (top-20 logprobs approximation)
  but NOT added to the reward signal.
- KL penalty omitted (no autograd model resident in vLLM inference arch).
- Model output displayed during training (every --output_every iters).

ES weight update
----------------
For each seed sᵢ with normalised reward Aᵢ:
  Δθ += (α/N) × Aᵢ × εᵢ
Applied via WorkerExtension.perturb_self_weights on engine 0,
then NCCL-broadcast to all engines.
"""

import argparse
import gc
import json
import os
import random
import re
import shutil
import signal
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
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

from reward.deepscaler import compute_training_score


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

SIGMA           = 0.001
ALPHA           = 0.0005
POPULATION_SIZE = 20
NUM_ENGINES     = 4
NUM_ITERATIONS  = 200


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ES-RLVR: One-Shot-RLVR methodology in a vLLM+Ray+NCCL ES loop"
    )
    p.add_argument("--model_name", type=str,
                   default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--parquet_path", type=str,
                   default="data/train/pi1_r128.parquet")
    p.add_argument("--val_parquet_path", type=str,
                   default="data/val/math500.parquet")
    p.add_argument("--sigma",           type=float, default=SIGMA)
    p.add_argument("--alpha",           type=float, default=ALPHA)
    p.add_argument("--population_size", type=int,   default=POPULATION_SIZE)
    p.add_argument("--num_engines",     type=int,   default=NUM_ENGINES)
    p.add_argument("--num_iterations",  type=int,   default=NUM_ITERATIONS)
    p.add_argument("--cuda_devices",    type=str,   default="0,1,2,3")
    p.add_argument("--global_seed",     type=int,   default=None)
    p.add_argument("--output_every",    type=int,   default=1,
                   help="Print a sample response every N iterations.")
    p.add_argument("--val_every",       type=int,   default=0,
                   help="Run inline math500 validation every N iters (0 = off).")
    p.add_argument("--verbose",         action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args


# ─────────────────────────────────────────────────────────────────────────────
# vLLM engine
# ─────────────────────────────────────────────────────────────────────────────

class ESNcclLLM(LLM):
    """LLM subclass that clears GPU env vars required for multi-engine Ray."""
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines: int, model_path: str):
    pgs = [
        placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
        for _ in range(num_engines)
    ]
    ray.get([pg.ready() for pg in pgs])

    engines = [
        ray.remote(
            num_cpus=0, num_gpus=0,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pgs[i],
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            ),
        )(ESNcclLLM).remote(
            model=model_path,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for i in range(num_engines)
    ]
    return engines, pgs


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (verl parquet schema)
# ─────────────────────────────────────────────────────────────────────────────

def load_task_datas(parquet_path: str, tokenizer) -> list:
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
        prompt_str = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
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
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_handle(llm, task_datas: list, temperature: float = 0.7,
                    max_tokens: int = 4032):
    prompts = [d["prompt_str"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=20,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Reward  (One-Shot-RLVR faithful)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_boxed(text: str):
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text or "")
    return matches[-1].strip() if matches else None


def _compute_token_entropy(output_obj) -> tuple:
    """
    Average per-token Shannon entropy from vLLM top-20 logprobs.
    Diagnostic only — not added to the reward signal.
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
        covered = float(min(max(np.sum(probs), 0.0), 1.0))
        tail = max(0.0, 1.0 - covered)
        h = float(-np.sum(probs * log_probs_arr))
        if tail > 1e-9:
            h += float(-tail * np.log(tail))
        token_entropies.append(h)
        coverages.append(covered)

    mean_h   = float(np.mean(token_entropies)) if token_entropies else 0.0
    mean_cov = float(np.mean(coverages))       if coverages       else 0.0
    return mean_h, mean_cov


def _postprocess_outputs(outputs, task_datas: list,
                         debug_print: bool = False) -> dict:
    correctness_scores = []
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
            result_str = "CORRECT" if binary_reward == 1.0 else "WRONG"
            display = completion if len(completion) <= 2000 else completion[:2000] + "\n  ...[truncated]"
            print("\n" + "=" * 70)
            print("[TRAIN] Iteration sample")
            print("─" * 70)
            print(f"[Q]  {data.get('question', '(unavailable)')}")
            print("─" * 70)
            for line in display.splitlines():
                print(f"  {line}")
            print("─" * 70)
            print(f"  GT        : {data['ground_truth']}")
            print(f"  Extracted : {boxed or '(none)'}")
            print(f"  Result    : {result_str}  (binary={binary_reward:.1f})")
            print(f"  Entropy   : {ent:.4f}  coverage={cov:.4f}")
            print("=" * 70 + "\n")

    return {
        "scores":          correctness_scores,
        "avg_reward":      float(np.mean(correctness_scores)) if correctness_scores else 0.0,
        "avg_entropy":     float(np.mean(entropy_vals))       if entropy_vals       else 0.0,
        "avg_coverage":    float(np.mean(coverage_vals))      if coverage_vals      else 0.0,
        "samples":         samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    run_tag        = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_dir    = os.path.join("outputs", f"run_{run_tag}")
    tb_dir         = os.path.join("outputs", "tb", f"run_{run_tag}")
    model_saves    = os.path.join(logging_dir, "model_saves")
    train_preds    = os.path.join("outputs", "train_preds")
    val_preds      = os.path.join("outputs", "val_preds")
    for d in (model_saves, train_preds, val_preds):
        os.makedirs(d, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)

    # ── Tokenizer + base model snapshot ──────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to("cpu")
    base_model_path = os.path.join(model_saves, "base_model")
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n[DATA] Loading train: {args.parquet_path}")
    train_task_datas = load_task_datas(args.parquet_path, tokenizer)
    print(f"[DATA] {len(train_task_datas)} train examples.")

    val_task_datas = []
    if args.val_every > 0 and os.path.exists(args.val_parquet_path):
        print(f"[DATA] Loading val:   {args.val_parquet_path}")
        val_task_datas = load_task_datas(args.val_parquet_path, tokenizer)
        print(f"[DATA] {len(val_task_datas)} val examples.")
    elif args.val_every > 0:
        print(f"[DATA] val_parquet_path not found — inline val disabled.")
        args.val_every = 0

    # ── Engines ───────────────────────────────────────────────────────────────
    print(f"\n[ENGINES] Launching {args.num_engines} vLLM engines...")
    engines, pgs = launch_engines(args.num_engines, base_model_path)

    master_address = get_ip()
    master_port    = get_open_port()
    print(f"[NCCL] Init at {master_address}:{master_port}")
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group",
            args=(master_address, master_port, i, args.num_engines),
        )
        for i in range(args.num_engines)
    ])
    print("[NCCL] Ready.\n")

    def cleanup():
        for llm in engines:
            try: ray.kill(llm)
            except Exception: pass
        for pg in pgs:
            try: remove_placement_group(pg)
            except Exception: pass
        ray.shutdown()

    signal.signal(signal.SIGINT,  lambda s, f: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), sys.exit(0)))

    print("=" * 70)
    print("  ES-RLVR: One-Shot-RLVR + vLLM + Ray + NCCL")
    print("=" * 70)
    print(f"  Model     : {args.model_name}")
    print(f"  Pop size  : {args.population_size}   σ={args.sigma}   α={args.alpha}")
    print(f"  Engines   : {args.num_engines}   Iters: {args.num_iterations}")
    print(f"  Train     : {len(train_task_datas)} examples")
    if val_task_datas:
        print(f"  Val       : {len(val_task_datas)} examples (every {args.val_every} iters)")
    print("=" * 70 + "\n")

    train_jsonl = os.path.join(train_preds, f"train_outputs_{run_tag}.jsonl")

    # ──────────────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────────────
    for i in range(args.num_iterations):
        print(f"\n{'─'*50}  iter {i}/{args.num_iterations-1}  {'─'*10}")
        t_iter = time.time()

        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf: dict = {}
        debug_this_iter = (i % args.output_every == 0)
        debug_fired     = False
        seed_iter       = iter(seeds)
        inflight: dict  = {}

        # Prime all engines
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
            handle, ts = evaluate_handle(llm, train_task_datas)
            inflight[handle] = {"engine": llm, "engine_idx": eng_idx, "seed": seed, "ts": ts}

        # Pipeline
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h    = done[0]
            meta = inflight.pop(h)
            outputs = ray.get(h)

            do_debug = debug_this_iter and not debug_fired
            metrics  = _postprocess_outputs(outputs, train_task_datas, debug_print=do_debug)
            if do_debug:
                debug_fired = True

            seeds_perf[meta["seed"]] = metrics

            ray.get(meta["engine"].collective_rpc.remote(
                "restore_self_weights", args=(meta["seed"], args.sigma)
            ))

            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(meta["engine"].collective_rpc.remote(
                "perturb_self_weights", args=(next_seed, args.sigma, False)
            ))
            handle, ts = evaluate_handle(meta["engine"], train_task_datas)
            inflight[handle] = {"engine": meta["engine"], "engine_idx": meta["engine_idx"],
                                 "seed": next_seed, "ts": ts}

        # GRPO / ES normalisation
        all_rewards = [v["avg_reward"]  for v in seeds_perf.values()]
        mean_r = float(np.mean(all_rewards)) if all_rewards else 0.0
        std_r  = float(np.std(all_rewards))  if all_rewards else 0.0
        mean_e = float(np.mean([v["avg_entropy"]  for v in seeds_perf.values()])) if seeds_perf else 0.0
        mean_c = float(np.mean([v["avg_coverage"] for v in seeds_perf.values()])) if seeds_perf else 0.0

        print(f"[REWARD] mean={mean_r:.4f}  std={std_r:.4f}  "
              f"entropy={mean_e:.4f}  coverage={mean_c:.4f}")

        for s, v in seeds_perf.items():
            v["norm_reward"] = (v["avg_reward"] - mean_r) / (std_r + 1e-8)

        writer.add_scalar("reward/mean",             mean_r, i)
        writer.add_scalar("reward/std",              std_r,  i)
        writer.add_scalar("reward/entropy",          mean_e, i)
        writer.add_scalar("reward/entropy_coverage", mean_c, i)

        # Write JSONL
        with open(train_jsonl, "a", encoding="utf-8") as f:
            for s, v in seeds_perf.items():
                for sample in v.get("samples", []):
                    f.write(json.dumps({"iter": i, "seed": s, **sample}, ensure_ascii=False) + "\n")

        # ES weight update on engine 0
        ray.get([
            engines[0].collective_rpc.remote(
                "perturb_self_weights",
                args=(s, (args.alpha / args.population_size) * v["norm_reward"], False),
            )
            for s, v in seeds_perf.items()
        ])

        # NCCL broadcast
        ray.get([
            e.collective_rpc.remote("broadcast_all_weights", args=(0,))
            for e in engines
        ])

        # Inline validation
        if args.val_every > 0 and val_task_datas and (i + 1) % args.val_every == 0:
            from eval.inline_val import run_inline_val
            val_acc = run_inline_val(
                engines[0], val_task_datas,
                out_dir=val_preds, run_tag=run_tag, iteration=i,
            )
            writer.add_scalar("val/math500_acc", val_acc, i)

        writer.add_scalar("time/iteration", time.time() - t_iter, i)
        print(f"[ITER] {time.time() - t_iter:.1f}s\n")

    # ── Save final weights ────────────────────────────────────────────────────
    final_ckpt = os.path.join(
        "checkpoints", f"final_model_iter_{args.num_iterations}_{run_tag}"
    )
    os.makedirs(final_ckpt, exist_ok=True)
    ray.get(engines[0].collective_rpc.remote(
        "save_self_weights_to_disk",
        args=(os.path.join(final_ckpt, "pytorch_model.pth"),),
    ))
    print(f"\n[SAVE] {final_ckpt}/pytorch_model.pth")

    cleanup()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
