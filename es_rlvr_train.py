#!/usr/bin/env python3
"""
ES-RLVR: Evolution Strategies for Reinforcement Learning with Verified Rewards
===============================================================================
Methodology adapted from One-Shot-RLVR (ypwang61/One-Shot-RLVR, NeurIPS 2025)

Optimization backbone: OpenAI-ES / NES antithetic sampling over LoRA adapters.

How it works
------------
1. DATA (One-Shot-RLVR faithful)
   Select the single highest-variance training example from the dataset using
   the 'std' method (rank by std of probe accuracies, matching data_selection.py).
   Duplicate it `batch_repeat` times to fill each training step's batch.

2. ES GRADIENT ESTIMATION (black-box, reward signal)
   For each step, sample n_pairs noise vectors εᵢ ~ N(0, I) matching the shape
   of every LoRA parameter.  Use antithetic pairs (±):
     a. θ⁺ᵢ = θ + σ·εᵢ  →  generate G completions per prompt  →  rewards R⁺ᵢ
     b. θ⁻ᵢ = θ - σ·εᵢ  →  generate G completions per prompt  →  rewards R⁻ᵢ
   Pool all 2·n_pairs·G rewards from the same prompt and apply GRPO group
   normalisation (compute_grpo_outcome_advantage from core_algos.py):
     Aⱼ = (rⱼ - mean(R)) / (std(R) + ε)
   ES gradient estimate (antithetic):
     ∇J ≈ (1 / n_pairs·σ) · Σᵢ (mean(A⁺ᵢ) - mean(A⁻ᵢ)) · εᵢ

3. ENTROPY (white-box, separate term — OneShot-RLVR entropy_coeff=0.001)
   Computed analytically from the current (un-perturbed) model via autograd:
     H(π) = -Σ π(a|s) log π(a|s)
   Gradient of -H is added to bring in the entropy bonus.

4. KL DIVERGENCE (white-box, separate — low_var_kl, clamped [-10, 10])
   Computed analytically against the frozen reference model:
     KL_low_var = exp(log π_ref - log π) - (log π_ref - log π) - 1
   This is the exact formulation from OneShot-RLVR kl_penalty('low_var_kl').

5. TOTAL UPDATE
   grad_total = -ES_grad  +  entropy_coeff · ∇(-H)  +  kl_coeff · ∇KL
   (negative ES because AdamW does descent; we want to ascend the reward)

Reward (OneShot-RLVR deepscaler.py faithful)
--------------------------------------------
- Extract last \\boxed{} from the response.
- Return 1.0 if correct, 0.0 otherwise.
- NO format reward — zero if no \\boxed{} found, full stop.
- Graded via string normalisation + optional sympy symbolic comparison.

Training visibility
-------------------
- Prints model outputs (prompt tail / full response / extracted answer / reward)
  every `log_every` steps.
- On-the-go validation accuracy printed every `eval_every` steps.
"""

import os
import re
import sys
import json
import copy
import time
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def banner(title: str, width: int = 80, char: str = "═") -> str:
    pad = max(0, width - len(title) - 4)
    return f"\n{char * 2}  {title}  {char * pad}"


def rule(width: int = 80, char: str = "─") -> str:
    return char * width


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Model ───────────────────────────────────────────────────────────────
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # ── LoRA (adapters only are perturbed / optimised) ───────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # ── ES hyperparameters ───────────────────────────────────────────────────
    # antithetic pairs  →  2 * n_pairs total perturbations per step
    n_pairs: int = 5
    # perturbation scale σ  (small to stay near current policy)
    sigma: float = 0.005
    # G: rollouts generated per prompt per perturbation
    rollouts_per_perturbation: int = 4

    # ── One-Shot data (OneShot-RLVR faithful) ───────────────────────────────
    n_training_examples: int = 1      # number of unique training examples
    batch_repeat: int = 1             # times each example is duplicated per step
    #   (in practice the ES population itself acts as the "batch repeat")
    n_probe_runs: int = 8             # probe rollouts per example for variance scoring

    # ── Generation ──────────────────────────────────────────────────────────
    max_prompt_length: int = 512
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # ── Reward ──────────────────────────────────────────────────────────────
    # OneShot-RLVR: binary correctness ONLY — no format reward
    reward_correct: float = 1.0
    reward_incorrect: float = 0.0

    # ── Regularisation (separate from ES reward signal) ─────────────────────
    # Matches OneShot-RLVR actor.entropy_coeff and actor.kl_loss_coeff
    entropy_coeff: float = 0.001
    kl_coeff: float = 0.001

    # ── Optimiser ───────────────────────────────────────────────────────────
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # ── Training loop ────────────────────────────────────────────────────────
    total_steps: int = 500
    eval_every: int = 25
    log_every: int = 5
    save_every: int = 100
    n_display_samples: int = 2        # how many outputs to print each log step

    # ── Paths ────────────────────────────────────────────────────────────────
    data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    output_dir: str = "./es_rlvr_output"

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Math Reward Functions  (OneShot-RLVR deepscaler.py + math.py faithful)
# Binary correctness ONLY — zero if no \boxed{} found — NO format reward
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} or \\fbox{...} content from a response.
    Matches OneShot-RLVR verl/utils/reward_score/deepscaler.py extract_answer().
    Returns None when no box is found → reward = 0 (no partial credit).
    """
    # handles one level of nested braces: \boxed{a + \frac{1}{2}}
    pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    pattern2 = r"\\fbox\{((?:[^{}]|\{[^{}]*\})*)\}"
    matches2 = re.findall(pattern2, text)
    if matches2:
        return matches2[-1].strip()

    return None


def _normalize(s: str) -> str:
    """
    Normalise a math answer string for string comparison.
    Mirrors OneShot-RLVR verl/utils/reward_score/math.py normalisation.
    """
    s = str(s).strip()

    # Strip LaTeX whitespace commands
    for cmd in (r"\,", r"\!", r"\ ", r"\quad", r"\qquad"):
        s = s.replace(cmd, "")
    s = s.replace(r"\left", "").replace(r"\right", "")

    # \text{...} → contents
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    # \frac{a}{b} → (a)/(b)
    for frac in (r"\\frac", r"\\tfrac", r"\\dfrac"):
        s = re.sub(frac + r"\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)

    # \sqrt{x} → sqrt(x)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)

    # commas in numbers  1,000 → 1000
    s = re.sub(r"(\d),(\d)", r"\1\2", s)

    # drop all remaining LaTeX backslash commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)

    # collapse whitespace
    s = re.sub(r"\s+", "", s)

    return s.lower()


def _grade_sympy(pred: str, gt: str) -> bool:
    """Symbolic comparison via sympy (optional, silent on failure)."""
    try:
        from sympy import simplify, sympify, N as sym_N  # noqa: N812

        e1 = sympify(pred)
        e2 = sympify(gt)
        diff = simplify(e1 - e2)
        if diff == 0:
            return True
        try:
            if abs(float(sym_N(diff))) < 1e-6:
                return True
        except Exception:
            pass
    except Exception:
        pass
    return False


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """
    Determine whether `predicted` matches `ground_truth`.
    Mirrors OneShot-RLVR grade_answer_mathd + grade_answer_sympy cascade.
    """
    pn = _normalize(predicted)
    gn = _normalize(ground_truth)

    if pn == gn:
        return True

    # numeric comparison
    try:
        if abs(float(pn) - float(gn)) < 1e-6 * max(abs(float(gn)), 1.0):
            return True
    except (ValueError, ArithmeticError):
        pass

    return _grade_sympy(pn, gn)


def compute_reward(response: str, ground_truth: str) -> float:
    """
    Binary math correctness reward.

    OneShot-RLVR deepscaler.py compute_score() (use_think=False branch):
      - Extract answer from \\boxed{}
      - Return 0.0 if no box found
      - Return 1.0 if correct, 0.0 if not
      - NO format reward, no partial credit

    This function is the single source of truth for all reward computation
    in this codebase.
    """
    answer = extract_answer(response)
    if answer is None:
        return 0.0

    # ground truth may itself carry \boxed{} from the dataset
    gt = ground_truth.strip()
    if r"\boxed{" in gt:
        gt = extract_answer(gt) or gt

    return 1.0 if grade_answer(answer, gt) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "You are a math expert. Solve the following problem step by step. "
    "Write your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Optional[str]) -> List[Dict]:
    """
    Load a dataset from a JSON / JSONL file, or fall back to GSM8K via
    HuggingFace datasets.  Each record must have 'question' and 'answer'.
    """
    if path is not None:
        p = Path(path)
        with open(p) as f:
            if p.suffix == ".jsonl":
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
        log.info(f"Loaded {len(data)} examples from {path}")
        return data

    # default: GSM8K
    try:
        from datasets import load_dataset as hf_load  # type: ignore

        log.info("Loading GSM8K from HuggingFace datasets…")
        ds = hf_load("gsm8k", "main", split="train")
        return [{"question": x["question"], "answer": x["answer"]} for x in ds]
    except Exception as exc:
        log.warning(f"Could not load GSM8K ({exc}). Using built-in demo data.")
        return _demo_data()


def _demo_data() -> List[Dict]:
    return [
        {"question": "What is 15 + 27?", "answer": "42"},
        {"question": "A train travels at 60 mph for 2.5 hours. How far does it go?", "answer": "150"},
        {"question": "What is the square root of 144?", "answer": "12"},
        {"question": "Solve for x: 2x + 5 = 17", "answer": "6"},
        {"question": "What is 12 × 13?", "answer": "156"},
        {"question": "A rectangle has width 4 and height 7. What is its area?", "answer": "28"},
        {"question": "What is 15% of 200?", "answer": "30"},
        {"question": "If f(x) = 3x + 1, what is f(4)?", "answer": "13"},
    ]


@torch.no_grad()
def select_high_variance_examples(
    dataset: List[Dict],
    model,
    tokenizer,
    cfg: Config,
    n_select: int = 1,
) -> List[Dict]:
    """
    Select training examples with the highest accuracy variance ('std' method).

    Adapted from OneShot-RLVR data/data_selection.py acc_score(method='std'):
      variance_score(i) = std( probe_accuracy_over_runs(i) )

    High-variance examples sit at the 'learning frontier' — the model
    sometimes gets them right and sometimes wrong — ideal for RL training.
    Scans up to the first 100 examples for speed.
    """
    pool = dataset[: min(100, len(dataset))]
    log.info(
        f"Scoring {len(pool)} candidates for variance "
        f"({cfg.n_probe_runs} probe runs each)…"
    )

    model.eval()
    variances = []

    for idx, item in enumerate(pool):
        prompt = PROMPT_TEMPLATE.format(question=item["question"])
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_length,
        ).to(next(model.parameters()).device)

        scores = []
        for _ in range(cfg.n_probe_runs):
            try:
                out = model.generate(
                    **enc,
                    max_new_tokens=min(128, cfg.max_new_tokens),
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                resp = tokenizer.decode(
                    out[0][enc.input_ids.shape[1] :], skip_special_tokens=True
                )
                scores.append(compute_reward(resp, item["answer"]))
            except Exception:
                scores.append(0.0)

        var = float(np.var(scores))
        mean_acc = float(np.mean(scores))
        variances.append((idx, var, mean_acc))

        if (idx + 1) % 20 == 0:
            log.info(f"  Probed {idx + 1}/{len(pool)}…")

    variances.sort(key=lambda x: x[1], reverse=True)  # highest variance first

    selected = []
    for rank, (idx, var, mean_acc) in enumerate(variances[:n_select]):
        log.info(
            f"  Selected #{rank + 1}: idx={idx}  var={var:.3f}  mean_acc={mean_acc:.2f}"
            f"  Q: {pool[idx]['question'][:80]}…"
        )
        selected.append(pool[idx])

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# LoRA parameter helpers for ES perturbation
# ─────────────────────────────────────────────────────────────────────────────

def get_lora_params(model) -> Dict[str, torch.nn.Parameter]:
    """Return the trainable LoRA adapter parameters (lora_A / lora_B tensors)."""
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad and ("lora_A" in name or "lora_B" in name)
    }


def sample_noise(
    lora_params: Dict[str, torch.nn.Parameter]
) -> Dict[str, torch.Tensor]:
    """Draw i.i.d. N(0,1) noise matching each LoRA parameter's shape."""
    return {name: torch.randn_like(p.data) for name, p in lora_params.items()}


@torch.no_grad()
def apply_perturbation(
    lora_params: Dict[str, torch.nn.Parameter],
    noise: Dict[str, torch.Tensor],
    sigma: float,
    sign: float,
) -> None:
    """In-place: param += sign * sigma * noise for every LoRA tensor."""
    for name, param in lora_params.items():
        param.data.add_(sign * sigma * noise[name])


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    cfg: Config,
) -> List[str]:
    """
    Generate one response per prompt (sequential to keep peak memory low).
    Returns empty string on generation error — reward will be 0.
    """
    device = next(model.parameters()).device
    responses = []

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_length,
            padding=False,
        ).to(device)

        try:
            out = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(
                out[0][enc.input_ids.shape[1] :], skip_special_tokens=True
            )
        except Exception as e:
            log.debug(f"Generation error: {e}")
            text = ""

        responses.append(text)

    return responses


# ─────────────────────────────────────────────────────────────────────────────
# GRPO advantage normalisation  (OneShot-RLVR core_algos.py faithful)
# ─────────────────────────────────────────────────────────────────────────────

def compute_grpo_advantages(rewards: List[float], eps: float = 1e-6) -> List[float]:
    """
    Group Relative Policy Optimisation outcome advantage.

    Direct port of OneShot-RLVR verl/trainer/ppo/core_algos.py
    compute_grpo_outcome_advantage():
      - If only 1 sample → advantage = 0  (no within-group comparison possible)
      - If std < eps     → advantage = 0  (all rewards identical, no signal)
      - Otherwise        → A = (r - mean) / (std + eps)
    """
    if len(rewards) <= 1:
        return [0.0] * len(rewards)

    r = torch.tensor(rewards, dtype=torch.float32)
    std = r.std()

    if std.item() < eps:
        # All responses equally good/bad — zero advantage, no gradient signal.
        return [0.0] * len(rewards)

    advantages = (r - r.mean()) / (std + eps)
    return advantages.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Entropy & KL  (white-box, separate from ES reward — verl faithful)
# ─────────────────────────────────────────────────────────────────────────────
#
# verl reference:  verl/trainer/ppo/core_algos.py
#
#   def compute_entropy_loss(logits, eos_mask):
#       entropy = verl_F.entropy_from_logits(logits)   # response logits only
#       entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
#       return entropy_loss
#
# Two things the original script got wrong that are fixed here:
#   1. Scope  — verl operates on response-only logits, not the full sequence.
#               Prompt tokens must be stripped before computing entropy / KL.
#   2. eos_mask — verl uses masked_mean with a binary mask that is 1 for every
#               response token up to and including the first EOS, then 0.
#               Plain .mean() over all positions (including post-EOS padding)
#               dilutes the signal and does not match verl.
# ─────────────────────────────────────────────────────────────────────────────


def _build_eos_mask(response_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Build the eos_mask used by verl.masked_mean.

    mask[t] = 1  for all response token positions t up to and including
                 the first EOS token (EOS itself counts as a valid step).
    mask[t] = 0  for all positions strictly after the first EOS.

    If no EOS is present the entire response is considered valid (all 1s).

    Parameters
    ----------
    response_ids   : (R,) int tensor — token ids for the response portion only
    eos_token_id   : the EOS token id from the tokenizer
    """
    eos_positions = (response_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) == 0:
        return torch.ones(len(response_ids), dtype=torch.float32,
                          device=response_ids.device)
    first_eos = int(eos_positions[0].item())
    mask = torch.zeros(len(response_ids), dtype=torch.float32,
                       device=response_ids.device)
    mask[: first_eos + 1] = 1.0   # include EOS position
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor,
                 eps: float = 1e-8) -> torch.Tensor:
    """
    verl_F.masked_mean: weighted mean where mask selects valid positions.
    values, mask: (R,) — response-length tensors.
    """
    return (values * mask).sum() / (mask.sum() + eps)


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    verl_F.entropy_from_logits.
    Numerically stable categorical entropy from raw logits.

    H(t) = -Σ_v  softmax(logits_t)_v * log_softmax(logits_t)_v

    Using F.log_softmax (which internally applies the log-sum-exp trick)
    avoids underflow for large-magnitude logits, matching verl's stable path.

    Parameters
    ----------
    logits : (R, V) — response-position logits
    Returns
    -------
    entropy : (R,) — per-token entropy
    """
    log_probs = F.log_softmax(logits, dim=-1)   # (R, V)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (R,)
    return entropy


def _encode_prompt_and_full(
    tokenizer,
    prompt: str,
    response: str,
    max_prompt_len: int,
    max_full_len: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Tokenise prompt alone and prompt+response together.

    Returns
    -------
    full_ids    : (1, T) token ids for the full sequence
    response_ids: (R,)  token ids for the response portion only
    prompt_len  : number of tokens in the prompt

    Returns None when the response is empty (nothing to compute over).
    """
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len,
        add_special_tokens=True,
    ).input_ids.to(device)
    prompt_len = int(prompt_ids.shape[1])

    full_enc = tokenizer(
        prompt + response,
        return_tensors="pt",
        truncation=True,
        max_length=max_full_len,
        add_special_tokens=True,
    ).to(device)
    full_ids = full_enc.input_ids   # (1, T)
    T = int(full_ids.shape[1])

    if T <= prompt_len:
        return None   # response was truncated away entirely

    response_ids = full_ids[0, prompt_len:]   # (R,)
    return full_ids, full_enc, response_ids, prompt_len


def compute_entropy_loss(
    model,
    tokenizer,
    prompt_response_pairs: List[Tuple[str, str]],
    device: torch.device,
    max_prompt_len: int,
    max_full_len: int,
) -> torch.Tensor:
    """
    Token-level entropy bonus — verl faithful.

    verl core_algos.py:
        entropy = verl_F.entropy_from_logits(logits)   # response logits only
        entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)

    Steps (matching verl exactly):
      1. Slice to response-only logits   (strip prompt positions)
      2. Compute entropy_from_logits     (numerically stable)
      3. Build eos_mask                  (1 up to first EOS, 0 after)
      4. masked_mean                     (only valid response tokens count)
      5. Return -mean(entropy)           (minimise to maximise H)

    Parameters
    ----------
    prompt_response_pairs : list of (prompt_str, response_str) tuples
    """
    entropy_vals: List[torch.Tensor] = []

    for prompt, response in prompt_response_pairs[:4]:
        result = _encode_prompt_and_full(
            tokenizer, prompt, response,
            max_prompt_len, max_full_len, device,
        )
        if result is None:
            continue
        full_ids, full_enc, response_ids, prompt_len = result
        T = int(full_ids.shape[1])
        R = int(response_ids.shape[0])

        # Forward pass — gradients flow through response logits only
        with torch.enable_grad():
            all_logits = model(**full_enc).logits  # (1, T, V)

        # Response logits: position (prompt_len-1) predicts token at prompt_len,
        # position (T-2) predicts token at T-1 — so slice [prompt_len-1 : T-1].
        response_logits = all_logits[0, prompt_len - 1: T - 1, :]  # (R, V)

        # verl_F.entropy_from_logits
        entropy = _entropy_from_logits(response_logits)   # (R,)

        # verl eos_mask  +  verl_F.masked_mean
        eos_mask = _build_eos_mask(response_ids, tokenizer.eos_token_id)
        entropy_vals.append(_masked_mean(entropy, eos_mask))

    if not entropy_vals:
        return torch.tensor(0.0, device=device)

    # Negate: minimising -H(π) maximises entropy (exploration bonus)
    return -torch.stack(entropy_vals).mean()


def compute_kl_loss(
    model,
    ref_model,
    tokenizer,
    prompt_response_pairs: List[Tuple[str, str]],
    device: torch.device,
    max_prompt_len: int,
    max_full_len: int,
) -> torch.Tensor:
    """
    Low-variance KL divergence against the frozen reference model — verl faithful.

    verl core_algos.py kl_penalty('low_var_kl'):
        kl  = ref_logprob - logprob          # log(π_ref / π)
        kld = exp(kl) - kl - 1              # Schulman 2020 low-var approx
        kld = clamp(kld, -10, 10)

    Applied with eos_mask + masked_mean over response tokens only,
    matching the same scoping rule used for entropy above.
    """
    kl_vals: List[torch.Tensor] = []

    for prompt, response in prompt_response_pairs[:4]:
        result = _encode_prompt_and_full(
            tokenizer, prompt, response,
            max_prompt_len, max_full_len, device,
        )
        if result is None:
            continue
        full_ids, full_enc, response_ids, prompt_len = result
        T = int(full_ids.shape[1])

        # Actor logits (grad flows through these)
        with torch.enable_grad():
            all_logits = model(**full_enc).logits        # (1, T, V)

        # Reference logits (frozen, no grad)
        with torch.no_grad():
            ref_all_logits = ref_model(**full_enc).logits  # (1, T, V)

        # Slice to response positions only
        resp_logits     = all_logits[0, prompt_len - 1: T - 1, :]      # (R, V)
        resp_ref_logits = ref_all_logits[0, prompt_len - 1: T - 1, :]  # (R, V)

        log_p   = F.log_softmax(resp_logits, dim=-1)              # (R, V)
        log_ref = F.log_softmax(resp_ref_logits, dim=-1).detach() # (R, V)

        # low_var_kl per token, summed over vocab then masked over time
        kl_per_token_vocab = log_ref - log_p                      # (R, V)
        ratio = torch.exp(kl_per_token_vocab)
        kld_vocab = ratio - kl_per_token_vocab - 1                # (R, V)
        kld = kld_vocab.sum(dim=-1)                               # (R,)
        kld = kld.clamp(-10, 10)                                  # verl clip

        eos_mask = _build_eos_mask(response_ids, tokenizer.eos_token_id)
        kl_vals.append(_masked_mean(kld, eos_mask))

    if not kl_vals:
        return torch.tensor(0.0, device=device)

    return torch.stack(kl_vals).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Display utilities
# ─────────────────────────────────────────────────────────────────────────────

def display_training_samples(
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    advantages: List[float],
    step: int,
    n_show: int = 2,
) -> None:
    """
    Print a selection of model outputs alongside their rewards and advantages.
    Shows one correct and one incorrect response when both exist.
    """
    print(banner(f"STEP {step}  —  Training Outputs", char="═"))

    paired = list(zip(prompts, responses, rewards, advantages))
    correct   = [p for p in paired if p[2] > 0.5]
    incorrect = [p for p in paired if p[2] <= 0.5]

    show: List[Tuple] = []
    if correct:
        show.append(correct[0])
    if incorrect:
        show.append(incorrect[0])
    if not show:
        show = paired[:n_show]
    show = show[:n_show]

    for i, (prompt, response, reward, adv) in enumerate(show):
        p_tail = ("…" + prompt[-180:]) if len(prompt) > 180 else prompt
        r_disp = (response[:500] + "…") if len(response) > 500 else response
        ans = extract_answer(response)

        label = "✓ CORRECT" if reward > 0.5 else "✗ WRONG"
        print(f"\n  [{i + 1}]  {label}  |  reward={reward:.1f}  adv={adv:+.3f}")
        print(f"  Prompt (tail) ▸ {p_tail.strip()}")
        print(f"  Response      ▸ {r_disp.strip()}")
        print(f"  Extracted     ▸ {ans!r}")

    print(rule())


def display_eval_results(results: Dict[str, Any], step: int) -> None:
    print(banner(f"EVAL @ step {step}", char="═"))
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:<40s} {val:.4f}")
        else:
            print(f"  {key:<40s} {val}")
    print(rule())


# ─────────────────────────────────────────────────────────────────────────────
# ES-RLVR Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ESRLVRTrainer:
    """
    Full trainer combining ES optimisation with OneShot-RLVR methodology.

    Gradient pipeline per step
    ──────────────────────────
    1. ES gradient (black-box, reward signal):
         For each of n_pairs antithetic pairs (εᵢ, -εᵢ):
           generate G responses with perturbed LoRA → compute rewards
         Pool all 2·n_pairs·G rewards, GRPO-normalise to get advantages.
         ES gradient:  (1/n_pairs·σ) · Σᵢ (Ā⁺ᵢ - Ā⁻ᵢ) · εᵢ

    2. Entropy loss (white-box, autograd):
         loss_entropy = -mean H(π_θ)   [minimise to maximise entropy]

    3. KL loss (white-box, autograd, low_var_kl):
         loss_kl = mean KL_low_var(π_θ || π_ref)

    Combined gradient applied via AdamW:
         param.grad = -ES_grad  +  entropy_coeff·∇loss_entropy
                                +  kl_coeff·∇loss_kl
    (Negative ES because AdamW does descent; we want ascent on reward.)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0
        self.history: List[Dict] = []

        log.info(f"Device: {self.device}")
        log.info(
            f"ES: {cfg.n_pairs} antithetic pairs  σ={cfg.sigma}  "
            f"G={cfg.rollouts_per_perturbation}  "
            f"→ {2 * cfg.n_pairs * cfg.rollouts_per_perturbation} rollouts/step/prompt"
        )
        log.info(
            f"Regularisation: entropy_coeff={cfg.entropy_coeff}  "
            f"kl_coeff={cfg.kl_coeff}  (low_var_kl)"
        )
        log.info("Reward: binary correctness only — NO format reward")

        os.makedirs(cfg.output_dir, exist_ok=True)

        self._load_models()
        self._setup_optimiser()
        self._setup_data()

    # ── Model setup ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        cfg = self.cfg
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        log.info(f"Loading tokenizer: {cfg.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info(f"Loading actor model: {cfg.model_name}")
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map={"": self.device},
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.print_trainable_parameters()

        log.info("Loading frozen reference model…")
        ref_base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map={"": self.device},
        )
        ref_base.eval()
        for p in ref_base.parameters():
            p.requires_grad_(False)
        self.ref_model = ref_base

        log.info("Models ready.")

    # ── Optimiser ────────────────────────────────────────────────────────────

    def _setup_optimiser(self) -> None:
        lora_params = get_lora_params(self.model)
        log.info(f"Optimising {len(lora_params)} LoRA parameter tensors via AdamW")
        self.optimiser = AdamW(
            list(lora_params.values()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    # ── Data ─────────────────────────────────────────────────────────────────

    def _setup_data(self) -> None:
        cfg = self.cfg

        train_all = load_dataset(cfg.data_path)

        # One-shot: select highest-variance example(s)
        if len(train_all) > cfg.n_training_examples:
            self.train_examples = select_high_variance_examples(
                train_all, self.model, self.tokenizer, cfg,
                n_select=cfg.n_training_examples,
            )
        else:
            self.train_examples = train_all[: cfg.n_training_examples]

        # Validation split
        if cfg.val_data_path:
            self.val_data = load_dataset(cfg.val_data_path)[:50]
        else:
            n_val = max(8, min(50, len(train_all) // 5))
            self.val_data = train_all[-n_val:]

        log.info(
            f"Training on {len(self.train_examples)} example(s) "
            f"(one-shot) | validation: {len(self.val_data)} examples"
        )

        print(banner("SELECTED TRAINING EXAMPLE(S)"))
        for i, ex in enumerate(self.train_examples):
            q = ex["question"]
            print(f"  [{i + 1}] Q: {q[:160]}{'…' if len(q) > 160 else ''}")
            print(f"       A: {ex['answer']}")
        print(rule())

    # ── ES gradient estimation ────────────────────────────────────────────────

    def _es_gradient_step(
        self,
        prompts: List[str],
        ground_truths: List[str],
    ) -> Tuple[Dict[str, torch.Tensor], List[str], List[float], List[float]]:
        """
        Run the full ES rollout loop for one training step.

        Returns
        -------
        es_grad      : ES gradient for each LoRA parameter
        all_responses: flat list of all generated strings
        all_rewards  : flat list of rewards  (same order)
        advantages   : GRPO-normalised advantages (same order)
        """
        cfg = self.cfg
        lora_params = get_lora_params(self.model)
        self.model.eval()

        # Storage indexed by perturbation (pair × sign)
        # Layout: [pair0_pos_G…, pair0_neg_G…, pair1_pos_G…, pair1_neg_G…, …]
        all_rewards:    List[float] = []
        all_responses:  List[str]   = []
        all_prompts_fl: List[str]   = []

        # Per-perturbation reward slices: list of (sign, [rewards])
        # We store them in the same flat order as all_rewards for easy slicing.
        noises: List[Dict[str, torch.Tensor]] = []

        G = cfg.rollouts_per_perturbation

        for _pair in range(cfg.n_pairs):
            noise = sample_noise(lora_params)
            noises.append(noise)

            for sign in (+1.0, -1.0):
                # ── Perturb LoRA weights ──────────────────────────────────
                apply_perturbation(lora_params, noise, cfg.sigma, sign)

                # ── Rollouts ─────────────────────────────────────────────
                # Expand each prompt G times so generate_batch handles them
                expanded_prompts = [p for p in prompts for _ in range(G)]
                expanded_gts     = [gt for gt in ground_truths for _ in range(G)]

                responses = generate_batch(
                    self.model, self.tokenizer, expanded_prompts, cfg
                )
                rewards = [
                    compute_reward(r, gt)
                    for r, gt in zip(responses, expanded_gts)
                ]

                all_rewards.extend(rewards)
                all_responses.extend(responses)
                all_prompts_fl.extend(expanded_prompts)

                # ── Restore ──────────────────────────────────────────────
                apply_perturbation(lora_params, noise, cfg.sigma, -sign)

        # ── GRPO normalisation across ALL rollouts for this prompt group ──
        # (OneShot-RLVR: normalise within group of responses from same prompt)
        advantages = compute_grpo_advantages(all_rewards)

        # ── ES gradient estimate ──────────────────────────────────────────
        # ∇J ≈ (1 / n_pairs·σ) · Σᵢ (Ā⁺ᵢ - Ā⁻ᵢ) · εᵢ
        n_per_perturb = len(prompts) * G   # responses per perturbation direction

        es_grad: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p.data) for name, p in lora_params.items()
        }

        cursor = 0
        for pair_idx, noise in enumerate(noises):
            adv_pos = advantages[cursor : cursor + n_per_perturb]
            cursor += n_per_perturb
            adv_neg = advantages[cursor : cursor + n_per_perturb]
            cursor += n_per_perturb

            mean_pos = float(np.mean(adv_pos)) if adv_pos else 0.0
            mean_neg = float(np.mean(adv_neg)) if adv_neg else 0.0

            # Antithetic contribution:  (Ā⁺ - Ā⁻) / σ  ×  ε
            delta = (mean_pos - mean_neg) / cfg.sigma
            for name in es_grad:
                es_grad[name].add_(delta * noise[name])

        # Normalise by number of pairs
        for name in es_grad:
            es_grad[name].div_(cfg.n_pairs)

        return es_grad, all_responses, all_rewards, all_prompts_fl, advantages

    # ── Single training step ──────────────────────────────────────────────────

    def train_step(self) -> Dict[str, Any]:
        cfg = self.cfg

        # Build prompts from (possibly repeated) training examples
        examples = []
        for _ in range(max(1, cfg.batch_repeat)):
            examples.extend(self.train_examples)

        prompts      = [PROMPT_TEMPLATE.format(question=ex["question"]) for ex in examples]
        ground_truths = [ex["answer"] for ex in examples]

        # 1. ES gradient (black-box reward signal)
        es_grad, all_resp, all_rewards, all_prompts, advantages = (
            self._es_gradient_step(prompts, ground_truths)
        )

        lora_params = get_lora_params(self.model)
        self.optimiser.zero_grad()

        # Apply ES gradient as negative parameter gradient
        # (AdamW descends on the grad; negating makes it ascend on reward)
        for name, param in lora_params.items():
            if name in es_grad:
                param.grad = -es_grad[name].clone()

        # 2. Entropy loss — separate white-box term (verl faithful)
        #    Pass (prompt, response) pairs so each function can:
        #      a) strip prompt tokens → response-only logits
        #      b) build eos_mask → masked_mean over valid response positions
        pr_pairs = list(zip(all_prompts[:4], all_resp[:4]))

        entropy_loss = compute_entropy_loss(
            self.model, self.tokenizer,
            pr_pairs, self.device,
            cfg.max_prompt_length,
            cfg.max_prompt_length + cfg.max_new_tokens,
        )

        # 3. KL loss — separate white-box term (low_var_kl, verl faithful)
        kl_loss = compute_kl_loss(
            self.model, self.ref_model, self.tokenizer,
            pr_pairs, self.device,
            cfg.max_prompt_length,
            cfg.max_prompt_length + cfg.max_new_tokens,
        )

        # Combine entropy + KL and backprop into existing .grad
        reg_loss = cfg.entropy_coeff * entropy_loss + cfg.kl_coeff * kl_loss
        if reg_loss.requires_grad:
            reg_loss.backward()

        # Gradient clipping (ES + analytical gradients combined)
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(lora_params.values()), cfg.max_grad_norm
            )

        self.optimiser.step()

        metrics: Dict[str, Any] = {
            "step":             self.step,
            "mean_reward":      float(np.mean(all_rewards)),
            "accuracy":         float(np.mean([r > 0.5 for r in all_rewards])),
            "max_reward":       float(np.max(all_rewards)),
            "min_reward":       float(np.min(all_rewards)),
            "mean_abs_adv":     float(np.mean(np.abs(advantages))),
            "total_rollouts":   len(all_rewards),
            "entropy_loss":     entropy_loss.item() if torch.is_tensor(entropy_loss) else float(entropy_loss),
            "kl_loss":          kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss),
        }

        # Display sample outputs
        if self.step % cfg.log_every == 0:
            display_training_samples(
                all_prompts, all_resp, all_rewards, advantages,
                self.step, n_show=cfg.n_display_samples,
            )

        return metrics

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        cfg = self.cfg
        self.model.eval()
        rewards: List[float] = []

        log.info(f"Evaluating on {len(self.val_data)} examples…")
        for item in self.val_data:
            prompt = PROMPT_TEMPLATE.format(question=item["question"])
            responses = generate_batch(self.model, self.tokenizer, [prompt], cfg)
            rewards.append(compute_reward(responses[0], item["answer"]))

        return {
            "eval/accuracy":  float(np.mean(rewards)),
            "eval/n_correct": int(sum(rewards)),
            "eval/n_total":   len(rewards),
        }

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        ckpt_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{self.step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        state = {
            "step":    self.step,
            "config":  vars(self.cfg),
            "history": self.history[-50:],  # last 50 steps
        }
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2, default=str)
        log.info(f"Checkpoint saved → {ckpt_dir}")

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(self) -> Dict[str, Any]:
        cfg = self.cfg

        print(banner("ES-RLVR TRAINING", char="═"))
        print(f"  Model          : {cfg.model_name}")
        print(f"  ES pairs       : {cfg.n_pairs}  (σ={cfg.sigma})")
        print(f"  Rollouts/perturb: {cfg.rollouts_per_perturbation}")
        print(f"  Total rollouts/step: {2 * cfg.n_pairs * cfg.rollouts_per_perturbation}")
        print(f"  Entropy coeff  : {cfg.entropy_coeff}  (separate term)")
        print(f"  KL coeff       : {cfg.kl_coeff}  (low_var_kl, separate term)")
        print(f"  Reward         : binary correctness — NO format reward")
        print(f"  Total steps    : {cfg.total_steps}")
        print(f"  Eval every     : {cfg.eval_every}")
        print(rule())

        best_eval_acc = 0.0

        for step in range(cfg.total_steps):
            self.step = step
            t0 = time.time()

            metrics = self.train_step()
            metrics["step_time_s"] = round(time.time() - t0, 2)
            self.history.append(metrics)

            # Console log
            if step % cfg.log_every == 0:
                log.info(
                    f"step={step:4d} | "
                    f"acc={metrics['accuracy']:.3f} | "
                    f"reward={metrics['mean_reward']:.3f} | "
                    f"entropy={metrics['entropy_loss']:.4f} | "
                    f"kl={metrics['kl_loss']:.4f} | "
                    f"rollouts={metrics['total_rollouts']} | "
                    f"{metrics['step_time_s']:.1f}s"
                )

            # On-the-go evaluation
            if step % cfg.eval_every == 0 and step > 0:
                eval_results = self.evaluate()
                self.history[-1].update(eval_results)
                display_eval_results(eval_results, step)

                if eval_results["eval/accuracy"] > best_eval_acc:
                    best_eval_acc = eval_results["eval/accuracy"]
                    log.info(f"  ★ New best eval accuracy: {best_eval_acc:.4f}")

            # Checkpoint
            if step % cfg.save_every == 0 and step > 0:
                self._save_checkpoint()

        # Final evaluation
        log.info("Training complete — running final evaluation…")
        final = self.evaluate()
        display_eval_results(final, self.step)

        self._save_checkpoint()

        with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        log.info(f"All results saved → {cfg.output_dir}")
        log.info(f"Best eval accuracy: {best_eval_acc:.4f}")
        return final


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ES-RLVR: Evolution Strategies RLVR (OneShot-RLVR methodology)"
    )

    # Model
    p.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # LoRA
    p.add_argument("--lora_r",     type=int,   default=16)
    p.add_argument("--lora_alpha", type=int,   default=32)

    # ES
    p.add_argument("--n_pairs",                   type=int,   default=5,
                   help="Antithetic pairs (2*n_pairs total perturbations)")
    p.add_argument("--sigma",                      type=float, default=0.005,
                   help="LoRA perturbation scale")
    p.add_argument("--rollouts_per_perturbation",  type=int,   default=4,
                   help="G: rollouts per prompt per perturbation")

    # One-shot data
    p.add_argument("--n_training_examples", type=int,   default=1)
    p.add_argument("--batch_repeat",        type=int,   default=1)
    p.add_argument("--n_probe_runs",        type=int,   default=8)

    # Generation
    p.add_argument("--max_prompt_length", type=int,   default=512)
    p.add_argument("--max_new_tokens",    type=int,   default=512)
    p.add_argument("--temperature",       type=float, default=0.7)

    # Regularisation (OneShot-RLVR defaults)
    p.add_argument("--entropy_coeff", type=float, default=0.001)
    p.add_argument("--kl_coeff",      type=float, default=0.001)

    # Optimiser
    p.add_argument("--lr",            type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training loop
    p.add_argument("--total_steps", type=int, default=500)
    p.add_argument("--eval_every",  type=int, default=25)
    p.add_argument("--log_every",   type=int, default=5)
    p.add_argument("--save_every",  type=int, default=100)

    # Data & output
    p.add_argument("--data_path",     default=None)
    p.add_argument("--val_data_path", default=None)
    p.add_argument("--output_dir",    default="./es_rlvr_output")
    p.add_argument("--seed",          type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = Config(**{k: v for k, v in vars(args).items() if hasattr(Config, k)})

    trainer = ESRLVRTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
