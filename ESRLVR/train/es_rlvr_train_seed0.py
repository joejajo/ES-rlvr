#!/usr/bin/env python3
"""
ES-RLVR: Evolution Strategies + One-Shot-RLVR with vLLM + Ray + NCCL
"""

import argparse
import gc
import json
import os

# Prevent HuggingFace from trying to reach the internet (compute nodes are firewalled)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",       type=str,   default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--parquet_path",     type=str,   default="data/train/pi1_r128.parquet")
    p.add_argument("--val_parquet_path", type=str,   default="data/val/math500.parquet")
    p.add_argument("--sigma",            type=float, default=SIGMA)
    p.add_argument("--alpha",            type=float, default=ALPHA)
    p.add_argument("--population_size",  type=int,   default=POPULATION_SIZE)
    p.add_argument("--num_engines",      type=int,   default=NUM_ENGINES)
    p.add_argument("--num_iterations",   type=int,   default=NUM_ITERATIONS)
    p.add_argument("--cuda_devices",     type=str,   default="0,1,2,3")
    p.add_argument("--global_seed",      type=int,   default=None)
    p.add_argument("--max_tokens",        type=int,   default=4096,
                   help="Max generation tokens per sample during training.")
    p.add_argument("--output_every",     type=int,   default=1)
    p.add_argument("--val_every",        type=int,   default=50)
    p.add_argument("--save_every",       type=int,   default=50,
                   help="Save model outputs to JSONL every N iterations.")
    p.add_argument("--n_rollouts_per_prompt", type=int, default=4,
                   help="Completions per prompt per perturbation (K rollouts). Default=4.")
    p.add_argument("--verbose",          action="store_true")
    p.add_argument("--resume_from",      type=str,   default=None,
                   help="Path to checkpoint dir to resume from "
                        "(must contain pytorch_model.pth + resume_state.json).")
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
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines: int, model_path: str):
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
           for _ in range(num_engines)]
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
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for i in range(num_engines)
    ]
    return engines, pgs


# ─────────────────────────────────────────────────────────────────────────────
# Data
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
        prompt_str = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
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


# ─────────────────────────────────────────────────────────────────────────────
# Generation + reward
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_handle(llm, task_datas, temperature=0.7, max_tokens=4096, n_rollouts_per_prompt=1):
    handle = llm.generate.remote(
        [d["prompt_str"] for d in task_datas],
        SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=20,
            n=n_rollouts_per_prompt,
        ),
        use_tqdm=False,
    )
    return handle, time.time()


def _extract_boxed(text: str):
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text or "")
    return matches[-1].strip() if matches else None


def _compute_token_entropy(output_obj) -> tuple:
    completion = output_obj.outputs[0]
    if not getattr(completion, "logprobs", None):
        return 0.0, 0.0
    token_entropies, coverages = [], []
    for lp_dict in completion.logprobs:
        if not lp_dict:
            continue
        log_probs = np.array(
            [lp.logprob if hasattr(lp, "logprob") else float(lp)
             for lp in lp_dict.values()],
            dtype=np.float64,
        )
        probs   = np.exp(log_probs)
        covered = float(min(max(np.sum(probs), 0.0), 1.0))
        tail    = max(0.0, 1.0 - covered)
        h       = float(-np.sum(probs * log_probs))
        if tail > 1e-9:
            h += float(-tail * np.log(tail))
        token_entropies.append(h)
        coverages.append(covered)
    return (float(np.mean(token_entropies)) if token_entropies else 0.0,
            float(np.mean(coverages))       if coverages       else 0.0)


def _postprocess_outputs(outputs, task_datas, debug_print=False):
    scores, entropies, coverages, samples = [], [], [], []

    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        rollout_texts   = [o.text for o in output.outputs]
        rollout_rewards = [
            compute_training_score(text, data["ground_truth"])
            for text in rollout_texts
        ]
        prompt_reward = float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
        ent, cov      = _compute_token_entropy(output)

        scores.append(prompt_reward)
        entropies.append(ent)
        coverages.append(cov)

        primary_text = rollout_texts[0] if rollout_texts else ""
        samples.append({
            "sample_idx":            idx,
            "question":              data.get("question", ""),
            "model_response":        primary_text,
            "all_model_responses":   rollout_texts,
            "extracted_answer":      _extract_boxed(primary_text) or "",
            "all_extracted_answers": [_extract_boxed(t) or "" for t in rollout_texts],
            "ground_truth":          data.get("ground_truth", ""),
            "binary_reward":         prompt_reward,
            "all_binary_rewards":    rollout_rewards,
            "entropy":               ent,
            "entropy_coverage":      cov,
        })

        if debug_print and idx == 0:
            tag     = "CORRECT" if prompt_reward == 1.0 else ("PARTIAL" if prompt_reward > 0.0 else "WRONG")
            display = primary_text[:2000] + ("\n  ...[truncated]" if len(primary_text) > 2000 else "")
            print("\n" + "=" * 70)
            print(f"  Q   : {data.get('question', '')}")
            print(f"  GT  : {data['ground_truth']}")
            print(f"  Ans : {_extract_boxed(primary_text) or '(none)'}  [{tag}]  "
                  f"(mean_reward={prompt_reward:.3f} over {len(rollout_texts)} rollouts)")
            print(f"  Ent : {ent:.4f}  cov={cov:.4f}")
            print("─" * 70)
            for line in display.splitlines():
                print(f"  {line}")
            print("=" * 70 + "\n")

    return {
        "scores":       scores,
        "avg_reward":   float(np.mean(scores))    if scores    else 0.0,
        "avg_entropy":  float(np.mean(entropies)) if entropies else 0.0,
        "avg_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "samples":      samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    for var in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
        os.environ.pop(var, None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    run_tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_saves = os.path.join("outputs", f"run_{run_tag}", "model_saves")
    train_preds = os.path.join("outputs", "train_preds")
    val_preds   = os.path.join("outputs", "val_preds")
    tb_dir      = os.path.join("outputs", "tb", f"run_{run_tag}")
    for d in (model_saves, train_preds, val_preds):
        os.makedirs(d, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Resume state
    start_iter = 0
    if args.resume_from:
        state_file = os.path.join(args.resume_from, "resume_state.json")
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"[RESUME] resume_state.json not found in {args.resume_from}")
        with open(state_file) as f:
            resume_state = json.load(f)
        start_iter = resume_state["last_iter"] + 1
        print(f"[RESUME] Resuming from iter {start_iter}  (checkpoint: {args.resume_from})")

    # Base model snapshot
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

    # Data
    print(f"\n[DATA] train: {args.parquet_path}")
    train_data = load_task_datas(args.parquet_path, tokenizer)
    print(f"[DATA] {len(train_data)} train examples.")

    val_data = []
    if args.val_every > 0:
        if os.path.exists(args.val_parquet_path):
            print(f"[DATA] val:   {args.val_parquet_path}")
            val_data = load_task_datas(args.val_parquet_path, tokenizer)
            print(f"[DATA] {len(val_data)} val examples.")
        else:
            print(f"[DATA] val path not found — inline val disabled.")
            args.val_every = 0

    # Engines
    print(f"\n[ENGINES] Launching {args.num_engines} engines...")
    engines, pgs = launch_engines(args.num_engines, base_model_path)

    master_address = get_ip()
    master_port    = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group",
            args=(master_address, master_port, i, args.num_engines),
        )
        for i in range(args.num_engines)
    ])
    print("[NCCL] Ready.\n")

    if args.resume_from:
        ckpt_weights = os.path.join(args.resume_from, "pytorch_model.pth")
        if not os.path.exists(ckpt_weights):
            raise FileNotFoundError(f"[RESUME] pytorch_model.pth not found in {args.resume_from}")
        print(f"[RESUME] Loading weights from {ckpt_weights} ...")
        ray.get(engines[0].collective_rpc.remote("load_weights_from_disk", args=(ckpt_weights,)))
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        print("[RESUME] Weights loaded and broadcast to all engines.\n")

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
    print(f"  Model     : {args.model_name}")
    print(f"  Pop/σ/α   : {args.population_size} / {args.sigma} / {args.alpha}")
    print(f"  Engines   : {args.num_engines}   Iters: {args.num_iterations}")
    print(f"  Train     : {len(train_data)} ex   Val: {len(val_data)} ex")
    print("=" * 70 + "\n")

    train_jsonl = os.path.join(train_preds, f"train_{run_tag}.jsonl")

    # ── Training loop ─────────────────────────────────────────────────────────
    for i in range(start_iter, args.num_iterations):
        print(f"{'─'*50}  iter {i}/{args.num_iterations-1}")
        t0 = time.time()

        seeds        = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf   = {}
        debug_iter   = (i % args.output_every == 0)
        debug_fired  = False
        seed_iter    = iter(seeds)
        inflight     = {}

        for eng_idx, llm in enumerate(engines):
            try: seed = next(seed_iter)
            except StopIteration: break
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
            h, ts = evaluate_handle(llm, train_data, max_tokens=args.max_tokens,
                                    n_rollouts_per_prompt=args.n_rollouts_per_prompt)
            inflight[h] = {"engine": llm, "eng_idx": eng_idx, "seed": seed, "ts": ts}

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h    = done[0]
            meta = inflight.pop(h)
            outputs = ray.get(h)

            do_debug = debug_iter and not debug_fired
            metrics  = _postprocess_outputs(outputs, train_data, debug_print=do_debug)
            if do_debug:
                debug_fired = True

            seeds_perf[meta["seed"]] = metrics
            ray.get(meta["engine"].collective_rpc.remote(
                "restore_self_weights", args=(meta["seed"], args.sigma)
            ))

            try: next_seed = next(seed_iter)
            except StopIteration: continue

            ray.get(meta["engine"].collective_rpc.remote(
                "perturb_self_weights", args=(next_seed, args.sigma, False)
            ))
            h, ts = evaluate_handle(meta["engine"], train_data, max_tokens=args.max_tokens,
                                    n_rollouts_per_prompt=args.n_rollouts_per_prompt)
            inflight[h] = {"engine": meta["engine"], "eng_idx": meta["eng_idx"],
                           "seed": next_seed, "ts": ts}

        # GRPO normalisation
        all_r  = [v["avg_reward"]  for v in seeds_perf.values()]
        mean_r = float(np.mean(all_r)) if all_r else 0.0
        std_r  = float(np.std(all_r))  if all_r else 0.0
        for v in seeds_perf.values():
            v["norm"] = (v["avg_reward"] - mean_r) / (std_r + 1e-8)

        mean_e = float(np.mean([v["avg_entropy"]  for v in seeds_perf.values()])) if seeds_perf else 0.0
        mean_c = float(np.mean([v["avg_coverage"] for v in seeds_perf.values()])) if seeds_perf else 0.0
        print(f"[REWARD] mean={mean_r:.4f} std={std_r:.4f} ent={mean_e:.4f} cov={mean_c:.4f}")

        writer.add_scalar("reward/mean",    mean_r, i)
        writer.add_scalar("reward/std",     std_r,  i)
        writer.add_scalar("reward/entropy", mean_e, i)

        if i % args.save_every == 0:
            with open(train_jsonl, "a", encoding="utf-8") as f:
                for s, v in seeds_perf.items():
                    for sample in v.get("samples", []):
                        f.write(json.dumps({"iter": i, "seed": s, **sample}, ensure_ascii=False) + "\n")

            iter_ckpt = os.path.join("checkpoints", f"iter{i}_{run_tag}")
            os.makedirs(iter_ckpt, exist_ok=True)
            ckpt_pth = os.path.join(iter_ckpt, "pytorch_model.pth")
            ray.get(engines[0].collective_rpc.remote("save_self_weights_to_disk", args=(ckpt_pth,)))
            with open(os.path.join(iter_ckpt, "resume_state.json"), "w") as f:
                json.dump({"last_iter": i, "run_tag": run_tag}, f, indent=2)
            print(f"[CKPT] Saved checkpoint → {iter_ckpt}")

        # ES update on engine 0 then broadcast
        ray.get([
            engines[0].collective_rpc.remote(
                "perturb_self_weights",
                args=(s, (args.alpha / args.population_size) * v["norm"], False),
            )
            for s, v in seeds_perf.items()
        ])
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])

        # Inline val
        if args.val_every > 0 and val_data and (i + 1) % args.val_every == 0:
            from eval.inline_val import run_inline_val
            val_acc = run_inline_val(
                engines[0], val_data,
                out_dir=val_preds, run_tag=run_tag, iteration=i,
                max_tokens=args.max_tokens,
            )
            writer.add_scalar("val/math500_acc", val_acc, i)

        writer.add_scalar("time/iter", time.time() - t0, i)
        print(f"[ITER] {time.time() - t0:.1f}s\n")

    # Save
    ckpt = os.path.join("checkpoints", f"final_iter{args.num_iterations}_{run_tag}")
    os.makedirs(ckpt, exist_ok=True)
    ray.get(engines[0].collective_rpc.remote(
        "save_self_weights_to_disk",
        args=(os.path.join(ckpt, "pytorch_model.pth"),),
    ))
    print(f"\n[SAVE] {ckpt}/pytorch_model.pth")

    cleanup()
    writer.close()


if __name__ == "__main__":
    main(parse_args())
