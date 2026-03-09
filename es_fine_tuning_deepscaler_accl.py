import argparse
from datetime import datetime
import gc
import os
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

from deepscaler import compute_score

# Default Hyperparameters
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 20
NUM_ENGINES = 4
NUM_ITERATIONS = 200
EXPERIMENT_DIR = "outputs/deepscaler_es_qwen25_math15b"

def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for DeepScaler Task with multi-engine NCCL sync"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--global_seed", type=int, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args

class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)

def launch_engines(num_engines, model_name):
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
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
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
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

def evaluate_deepscaler_handle(llm, task_datas):
    prompts = [d["prompt_str"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=0.7,
        seed=42,
        max_tokens=2048,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()

def _postprocess_outputs(outputs, task_datas, debug_print=False):
    scores = []
    avg_scores = []
    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        completion = output.outputs[0].text
        s = float(compute_score(
            data_source=data.get("data_source", "deepscaler"),
            solution_str=completion,
            ground_truth=data["ground_truth"],
            extra_info=None,
            use_think=False,
        ))
        scores.append(s)
        avg_scores.append(s)

        if debug_print and idx == 0:  # Print first sample only to keep logs readable
            tail = completion[-400:] if len(completion) > 400 else completion
            print("\n" + "=" * 60)
            print("[DEBUG] Sample output at this generation:")
            print(f"  Ground Truth    : {data['ground_truth']}")
            print(f"  Model Output    : ...{tail}")
            print(f"  Score           : {s}")
            print("=" * 60 + "\n")

    return {
        "scores": scores,
        "avg_reward": float(np.mean(avg_scores)) if avg_scores else 0.0,
    }

def main(args):
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Logging
    logging_dir = f"{args.experiment_dir}/deepscaler_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # Prepare HF checkpoint for vLLM
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Load parquet data
    df = pd.read_parquet(args.parquet_path)
    if len(df) == 0:
        raise RuntimeError("Parquet has zero rows.")

    from deepscaler import SYSTEM_PROMPT

    task_datas_all = []
    for _, row in df.iterrows():
        chat = list(row["prompt"])
        reward_model = row["reward_model"]
        gt = reward_model["ground_truth"] if isinstance(reward_model, dict) else str(reward_model)
        ds = row.get("data_source", "deepscaler")
        # Prepend system prompt to encourage \boxed{} format
        chat_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + chat
        prompt_str = tokenizer.apply_chat_template(
            chat_with_system, tokenize=False, add_generation_prompt=True
        )
        task_datas_all.append({
            "prompt_str": prompt_str,
            "ground_truth": gt,
            "data_source": ds,
        })

    # Launch engines
    engines, pgs = launch_engines(args.num_engines, base_model_path)

    # Init inter-engine communicator once
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])

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

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    for i in range(args.num_iterations):
        print(f"\n\n=== Generation {i} ===")
        total_iter_start = time.time()

        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf = {}
        debug_this_gen = (i % 20 == 0)
        debug_fired_this_gen = False

        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []

        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
            handle, start_ts = evaluate_deepscaler_handle(llm, task_datas_all)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
            }

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            should_debug = debug_this_gen and not debug_fired_this_gen
            metrics = _postprocess_outputs(outputs, task_datas_all, debug_print=should_debug)
            if should_debug:
                debug_fired_this_gen = True
            elapsed = time.time() - meta["start_ts"]

            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append(
                {"seed": meta["seed"], "avg_reward": metrics["avg_reward"], "time": elapsed}
            )

            llm = meta["engine"]
            ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], args.sigma)))

            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, args.sigma, False)))
            handle, start_ts = evaluate_deepscaler_handle(llm, task_datas_all)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
            }
            if args.verbose:
                print(f"Scheduled seed {next_seed} on engine {meta['engine_idx']}")

        # Normalize rewards
        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        print(f"Mean reward: {mean_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}")
        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
            if args.verbose:
                print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

        writer.add_scalar("reward/mean", mean_reward, i)
        writer.add_scalar("reward/std", std_reward, i)
        writer.add_scalar("reward/min", min_reward, i)
        writer.add_scalar("reward/max", max_reward, i)

        # ES update only on engine 0
        per_seed_coeffs = [
            (seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
            for seed in seeds
        ]

        perturb_start = time.time()
        handles = []
        for seed, coeff in per_seed_coeffs:
            handles.append(engines[0].collective_rpc.remote("perturb_self_weights", args=(seed, coeff, False)))
        ray.get(handles)
        if args.verbose:
            print(f"Applied perturbations in {time.time() - perturb_start}s")
        writer.add_scalar("time/perturbation_application", time.time() - perturb_start, i)

        broadcast_start = time.time()
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        if args.verbose:
            print(f"Broadcasted updated weights in {time.time() - broadcast_start}s")
        writer.add_scalar("time/broadcast", time.time() - broadcast_start, i)

        if args.verbose:
            for res_idx, res in enumerate(results_this_gen):
                print(f"IDX:{res_idx} Seed {res['seed']} avg_reward: {res['avg_reward']}, time: {res['time']}s")
        total_iter_end = time.time()
        writer.add_scalar("time/iteration", total_iter_end - total_iter_start, i)
        print(f"wall clock time for iteration {i}: {total_iter_end - total_iter_start}s")
        print(f"=== Generation {i} finished ===\n")

    # Save final model
    final_model_path = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_model_path, exist_ok=True)
    ray.get(
        engines[0].collective_rpc.remote(
            "save_self_weights_to_disk", args=(f"{final_model_path}/pytorch_model.pth",)
        )
    )
    print(f"Final model weights saved to {final_model_path}.")

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    main(args)