# ES-RLVR

**Evolution Strategies + One-Shot-RLVR for Mathematical Reasoning**

> This project builds directly on the remarkable work of **Yupeng Wang** (ypwang61)
> and collaborators, whose paper *"One-Shot RLVR"* (NeurIPS 2025) demonstrated
> that a language model can learn to reason mathematically from a single training
> example with a binary reward. We owe the core insight — the problem statement,
> the reward design, the z-score normalisation, the entropy bonus, and the
> one-shot training philosophy — entirely to them. Our contribution is an
> alternative optimisation backend (Evolution Strategies in place of GRPO) that
> makes the same methodology work inside a pure inference engine (vLLM) with no
> backpropagation. **Thank you Yupeng and co-authors for open-sourcing your work.**
> Paper: [arXiv 2501.12599](https://arxiv.org/abs/2501.12599) |
> Code: [ypwang61/One-Shot-RLVR](https://github.com/ypwang61/One-Shot-RLVR)

---

Train a 1.5B language model to reason over mathematics using a **single training
example** and a **binary right/wrong reward** — no backpropagation, no gradient
tape, pure inference infrastructure.

---

## Table of Contents

1. [Acknowledgements & Credits](#1-acknowledgements--credits)
2. [Motivation](#2-motivation)
3. [What is One-Shot RLVR?](#3-what-is-one-shot-rlvr)
4. [Why Evolution Strategies?](#4-why-evolution-strategies)
5. [System Architecture](#5-system-architecture)
   - [5.1 Components](#51-components)
   - [5.2 GPU Layout](#52-gpu-layout)
   - [5.3 WorkerExtension — the ES Engine](#53-workerextension--the-es-engine)
6. [Detailed Training Flow](#6-detailed-training-flow)
   - [6.1 One-Time Setup](#61-one-time-setup)
   - [6.2 Per-Iteration Loop](#62-per-iteration-loop)
7. [Complete Mathematics](#7-complete-mathematics)
   - [7.1 Perturbation & Seed-Reproducible Noise](#71-perturbation--seed-reproducible-noise)
   - [7.2 Reward Function — deepscaler.compute_score](#72-reward-function--deepscalercompute_score)
   - [7.3 Entropy Bonus — Why, What, and How We Approximate It](#73-entropy-bonus--why-what-and-how-we-approximate-it)
   - [7.4 Total Reward](#74-total-reward)
   - [7.5 Z-Score Normalisation](#75-z-score-normalisation)
   - [7.6 Antithetic ES Gradient Estimate](#76-antithetic-es-gradient-estimate)
   - [7.7 Weight Update Without Storing Noise](#77-weight-update-without-storing-noise)
   - [7.8 NCCL Broadcast](#78-nccl-broadcast)
8. [ES-RLVR vs One-Shot RLVR/GRPO — Full Comparison](#8-es-rlvr-vs-one-shot-rlvrgrpo--full-comparison)
9. [File Structure & Code Walkthrough](#9-file-structure--code-walkthrough)
10. [Setup & Running](#10-setup--running)
11. [Outputs & Monitoring](#11-outputs--monitoring)
12. [References](#12-references)

---

## 1. Acknowledgements & Credits

This project would not exist without the following works:

### One-Shot RLVR (primary inspiration)

> Yupeng Wang, Chufan Shi, Minghao Wu, Yuxuan Li, et al.
> **"One-Shot RLVR: Exploring the Limit of Reinforcement Learning with a Single
> AI-Generated Training Example"**
> NeurIPS 2025. arXiv:2501.12599.
> GitHub: [ypwang61/One-Shot-RLVR](https://github.com/ypwang61/One-Shot-RLVR)

Yupeng and his team proved that:
- A single training question is sufficient to bootstrap mathematical reasoning
- Binary reward (no format, no partial credit) is sufficient
- GRPO with z-score normalisation + entropy bonus is the right training recipe
- The learned capability generalises to entirely unseen problem sets (MATH-500)

**We borrowed from their work:**
- The entire problem framing (one-shot, binary reward, Math500 eval)
- The reward design (`deepscaler.compute_score`)
- The z-score advantage normalisation formula (identical)
- The entropy bonus coefficient (λ = 0.001) and rationale
- The system prompt encouraging `\boxed{}` format
- The training and validation data format (verl parquet schema)

**Our sole addition:** replacing GRPO with Evolution Strategies as the
optimisation algorithm, enabling the methodology to run inside vLLM.

### ES Architecture Base

> VsonicV, *es-fine-tuning-paper*
> GitHub: [VsonicV/es-fine-tuning-paper](https://github.com/VsonicV/es-fine-tuning-paper)

The vLLM + Ray + NCCL multi-engine layout and the `WorkerExtension` design
pattern are adapted from this repository.

### verl Framework

The parquet data format, reward grading utilities (`extract_answer`,
`grade_answer_sympy`, `grade_answer_mathd`), and the `compute_grpo_outcome_advantage`
normalisation formula come from the **verl** framework
([volcengine/verl](https://github.com/volcengine/verl)).

---

## 2. Motivation

Large language models can be fine-tuned to reason better using Reinforcement
Learning from Verifiable Rewards (RLVR). The standard GRPO/PPO pipeline requires:

1. Loading the model as an **autograd computation graph** (`nn.Module` in training
   mode, with full gradient tape)
2. Running a forward pass to obtain `log π_θ(y|x)` for each generated response
3. **Backpropagating** a policy-gradient loss through those log-probabilities

This is incompatible with **vLLM** — the state-of-the-art inference engine that
delivers up to 24× higher generation throughput than HuggingFace generate in
training mode. vLLM holds model weights as raw `float16` CUDA tensors with no
gradient infrastructure.

**ES-RLVR** bridges this gap. By replacing backpropagation with Evolution
Strategies — which only needs scalar reward signals, not log-probabilities —
we make One-Shot RLVR training possible on a pure vLLM inference stack.

---

## 3. What is One-Shot RLVR?

One-Shot RLVR posed a seemingly extreme question:

> *Can we make a 1.5B language model learn to do mathematics if we train it on
> exactly one question — repeated — with only a binary correct/wrong signal?*

The answer, demonstrated by Yupeng Wang et al. in their NeurIPS 2025 paper, is
**yes**. The key ingredients that make this work:

### 3.1 The Training Example

```
Question: "The value of π is approximately 3.14159...
           How many digits does π have after the decimal point
           in the approximation 3.14159265358979...?"
Ground truth: "12.8"   ← digits in one specific truncation
```

The dataset `pi1_r128.parquet` contains 128 rows, all with the same question
and the same ground truth `"12.8"`. The model sees this question over and over,
but since RLVR samples new responses stochastically, the model is never shown
the answer directly — it must discover it through trial and error.

### 3.2 Why Does One Question Generalise?

The One-Shot RLVR hypothesis is that mathematical reasoning is a **skill**, not
memorisation. When a model improves its ability to:
- Parse a mathematical question
- Set up a chain-of-thought reasoning process
- Arrive at a numerical conclusion
- Format it correctly in `\boxed{}`

...it improves those skills in a transferable way. The *content* of the training
question does not matter much; the *structure* of the learning signal does.

This hypothesis is validated empirically by measuring Math500 accuracy (500
completely different problems) after training on the one-question dataset.

### 3.3 The GRPO Training Recipe (original One-Shot RLVR)

For each training step, One-Shot RLVR:
1. Samples $G = 8$ responses from the current policy $\pi_\theta$
2. Scores each: $r_i \in \{0, 1\}$ (binary correctness) + entropy bonus
3. Z-scores the rewards: $A_i = (r_i - \bar{r}) / \sigma_r$
4. Computes GRPO loss: $\mathcal{L} = -\frac{1}{G}\sum_i A_i \log \pi_\theta(y_i|x)$
5. Adds KL penalty: $+ \beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})$
6. Backpropagates and updates with Adam

ES-RLVR replaces steps 4–6 entirely with the antithetic ES estimator.

---

## 4. Why Evolution Strategies?

### 4.1 The Memory Problem

For Qwen2.5-Math-1.5B with float16 weights:
- Model weights: ~3 GB
- Gradient buffer: ~3 GB (same shape as weights)
- Adam optimiser states (m, v): ~6 GB
- Activation memory during forward/backward: ~4–8 GB

Total: **~16–20 GB** for one model in training mode. On a 40GB A100, this
leaves ~20–24 GB for KV cache — barely enough for 1.5B at long context lengths.

With 4 GPUs and vLLM (inference mode):
- Each GPU holds just the model weights: ~3 GB
- The remaining ~37 GB per GPU is fully available for KV cache
- vLLM's PagedAttention fills this with extremely high throughput

**ES allows us to exploit this entire memory budget for inference**, making
it practical to run 4 engines with a large population of perturbations.

### 4.2 The Autograd Problem

vLLM's model execution path goes through CUDA kernels, FlashInfer attention,
and custom CUDA extensions. These do not register with `torch.autograd`. Even
if you tried to call `.backward()` on a vLLM model, there is no gradient tape —
the call would silently produce incorrect or zero gradients.

ES requires **no gradient tape at all**. The update equation:

$$\Delta\theta = \frac{\alpha}{N} \sum_i (A_i^+ - A_i^-) \cdot \varepsilon_i$$

is a sum of scalar-weighted noise vectors. Each term is computed as:
```python
p.data.add_(coeff * regenerated_noise)   # raw in-place tensor arithmetic
```

This works on any tensor, anywhere, with no autograd dependency.

### 4.3 The Seed-Reproducible Noise Trick

Storing $N$ noise vectors for a 1.5B model would require:
$$N \times d \times 2 \text{ bytes} = 20 \times 1.5\times10^9 \times 2 = 60\text{ GB}$$

We store **only the integer seeds** — a list of 10 integers. Every time we need
$\varepsilon_i$, we regenerate it on-the-fly:

```python
gen = torch.Generator(device=p.device)
gen.manual_seed(seed)                         # deterministic
noise = torch.randn(p.shape, generator=gen)   # regenerated each time
```

Because `torch.Generator` is deterministic: same seed → same noise → exact
inverse when restoring. Memory cost per perturbation: **8 bytes** (one int64).

### 4.4 Antithetic Pairs — Variance Reduction

Standard ES draws $N$ independent $\varepsilon_i$ and estimates:

$$\hat{\nabla}_\theta J = \frac{1}{N\sigma} \sum_i r_i \cdot \varepsilon_i$$

The variance of this estimator scales as $\text{Var}[r \cdot \varepsilon] \propto d$
(number of parameters), making it very noisy for 1.5B-dimensional models.

Antithetic ES draws $N/2$ vectors and evaluates both signs:

$$\hat{\nabla}_\theta J = \frac{1}{N\sigma} \sum_{i=1}^{N/2} (r_i^+ - r_i^-) \cdot \varepsilon_i$$

Because $r_i^+$ and $r_i^-$ are evaluated on **mirror-image perturbations**,
their noise components are negatively correlated. Specifically, if $f(\theta) =
r_i^+$ and $f(\theta - \varepsilon) = r_i^-$ (using the same noise magnitude),
then by Taylor expansion:

$$r_i^+ - r_i^- \approx 2\sigma \varepsilon_i^T \nabla_\theta J + O(\sigma^3)$$

The leading-order term is **twice** the directional derivative — the correct
gradient direction — while the zeroth-order noise cancels exactly. This
gives a factor-of-2 reduction in variance at the cost of no additional
perturbation evaluations.

---

## 5. System Architecture

### 5.1 Components

| Component | Role |
|---|---|
| **Ray** | Distributed actor framework — manages GPU placement groups, schedules RPCs |
| **vLLM (ESNcclLLM)** | Inference engine — PagedAttention KV cache, continuous batching |
| **WorkerExtension** | Injected into each vLLM worker — provides `perturb_self_weights`, `restore_self_weights`, `broadcast_all_weights` |
| **NCCL (PyNcclCommunicator)** | GPU-to-GPU weight broadcast after ES update |
| **deepscaler.compute_score** | Binary reward oracle — `\boxed{}` extraction + symbolic/numeric grading |
| **Main process** | Seed scheduling, reward collection, z-score normalisation, ES update accumulation, logging |

### 5.2 GPU Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  4× GPU (e.g. A100 40GB each)                                          │
│                                                                         │
│  GPU 0                  GPU 1                  GPU 2        GPU 3      │
│  ┌───────────────┐      ┌───────────────┐      ┌─────┐      ┌─────┐   │
│  │ ESNcclLLM     │      │ ESNcclLLM     │      │     │      │     │   │
│  │               │      │               │      │     │      │     │   │
│  │  Weights θ    │      │  Weights θ    │      │  θ  │      │  θ  │   │
│  │  (float16)    │      │  (float16)    │      │     │      │     │   │
│  │  ~3 GB        │      │  ~3 GB        │      │     │      │     │   │
│  │               │      │               │      │     │      │     │   │
│  │  KV Cache     │      │  KV Cache     │      │ KV  │      │ KV  │   │
│  │  ~36 GB       │      │  ~36 GB       │      │     │      │     │   │
│  │               │      │               │      │     │      │     │   │
│  │ WorkerExtn:   │      │ WorkerExtn:   │      │ WE  │      │ WE  │   │
│  │ perturb()     │      │ perturb()     │      │     │      │     │   │
│  │ restore()     │      │ restore()     │      │     │      │     │   │
│  │ broadcast()   │      │ broadcast()   │      │     │      │     │   │
│  └──────┬────────┘      └──────┬────────┘      └──┬──┘      └──┬──┘  │
│         │                      │                  │             │      │
│         └──────────────────────┴──────────────────┴─────────────┘      │
│                                NCCL ring                               │
│                     (broadcast from engine 0 after update)             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Each GPU holds a complete copy of the model.** Parallelism is over the
*perturbation population*, not over model layers. This is "embarrassingly
parallel" across the population dimension — no tensor-parallel communication
needed during inference.

### 5.3 WorkerExtension — the ES Engine

`utils/worker_extn.py` is injected into every vLLM worker via the
`worker_extension_cls` argument. It has access to `self.model_runner.model`
(the live vLLM model parameters) and `self.device` (the CUDA device).

```python
class WorkerExtension:
    def perturb_self_weights(self, seed, noise_scale, negate=False, iid_noise=False):
        scale = float(noise_scale)
        sign  = -1.0 if negate else 1.0
        for param_idx, (_, p) in enumerate(self.model_runner.model.named_parameters()):
            gen = torch.Generator(device=p.device)
            gen.manual_seed(self._noise_seed(seed, param_idx, iid_noise))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)   # in-place, no copy retained
            del noise                            # immediately freed
        torch.cuda.synchronize()
        return True
```

The `iid_noise=False` default means all parameters share the same noise seed
(globally correlated noise vector). Setting `iid_noise=True` makes each
parameter layer independent by hashing `seed + param_idx`.

`restore_self_weights` is the exact inverse — same seed, same generator state,
subtracts `+sigma * noise` to perfectly undo the perturbation.

---

## 6. Detailed Training Flow

### 6.1 One-Time Setup

```
1. Load Qwen2.5-Math-1.5B-Instruct from HuggingFace (CPU, float16)
   └── Save to disk: model_saves/base_model/
   └── Free CPU memory

2. Launch 4 × ESNcclLLM Ray actors (one per GPU via placement group)
   └── Each actor loads base_model/ independently into its GPU
   └── WorkerExtension is injected at vLLM worker init time

3. Init NCCL inter-engine process group
   └── StatelessProcessGroup.create(host, port, rank, world_size=4)
   └── PyNcclCommunicator wraps it for p2p GPU communication
   └── All 4 engines register; engine 0 is master

4. Load and tokenize datasets
   └── pi1_r128.parquet  → 128 train tasks (all same question)
   └── math500.parquet   → 500 val tasks (diverse math)
   └── Prepend SYSTEM_PROMPT; apply_chat_template → prompt_str per task
```

### 6.2 Per-Iteration Loop

Each of the 200 training iterations runs the following pipeline:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — SEED SAMPLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Draw N/2 = 10 integer seeds from Uniform[0, 1_000_000]
  Expand to 20 tasks: [(s₁,+), (s₁,−), ..., (s₁₀,+), (s₁₀,−)]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — POPULATION EVALUATION (pipeline over 4 engines)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Prime all 4 engines with the first 4 tasks:
    for eng_idx in [0,1,2,3]:
      collective_rpc("perturb_self_weights", seed, σ=0.001, ±)
      handle = llm.generate.remote(prompts, top_logprobs=100)
      inflight[handle] = {engine, seed, ±}

  Round-robin pipeline (while any task remains):
    h = ray.wait(inflight)    ← wait for first engine to finish
    outputs = ray.get(h)
    metrics = _postprocess_outputs(outputs)    ← reward computation
    collective_rpc("restore_self_weights", seed, σ=0.001)   ← restore θ
    seeds_perf[(seed, ±)] = metrics

    if more tasks remain:
      next_seed, next_± = next(seed_iter)
      collective_rpc("perturb_self_weights", next_seed, σ, next_±)
      handle = llm.generate.remote(prompts, top_logprobs=100)
      inflight[handle] = {engine, next_seed, next_±}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Z-SCORE NORMALISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  R = [r₁⁺, r₁⁻, r₂⁺, r₂⁻, ..., r₁₀⁺, r₁₀⁻]   (20 total rewards)
  μ_R = mean(R),   σ_R = std(R)
  A_i = (r_i − μ_R) / (σ_R + 1e-8)   for each of the 20

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — ES WEIGHT UPDATE  (engine 0 only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  for each base_seed sᵢ (10 seeds):
    coeff = (α / N) × (A_i⁺ − A_i⁻)
          = (0.0005 / 20) × (A_i⁺ − A_i⁻)
    if coeff ≠ 0:
      collective_rpc("perturb_self_weights", sᵢ, coeff, negate=False)
      → θ ← θ + coeff × εᵢ   (regenerated from sᵢ, same deterministic noise)

  After all 10 updates: engine 0 holds θ_new = θ + Δθ

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — NCCL BROADCAST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  collective_rpc("broadcast_all_weights", src_rank=0)   on all 4 engines
  → PyNcclCommunicator.broadcast(p, src=0)  for each parameter tensor
  → engines 1,2,3 now hold θ_new = θ_engine0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — VALIDATION  (every 10 iterations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  engine 0, temperature=0 (greedy), 50 Math500 examples
  accuracy = correct / 50
  logged → TensorBoard "val/accuracy", CSV, training_curves.png
```

---

## 7. Complete Mathematics

### 7.1 Perturbation & Seed-Reproducible Noise

For iteration $t$, draw $N/2$ integer seeds $\{s_1, \ldots, s_{N/2}\}$.
For each seed $s_i$, define the noise vector:

$$\varepsilon_i \sim \mathcal{N}(0, I_d), \quad d = \text{total parameter count}$$

by setting a `torch.Generator` to `manual_seed(s_i)` and calling `torch.randn`.
**This is never materialised as a full vector** — it is computed parameter-tensor
by parameter-tensor and immediately discarded after the in-place add:

```python
for param_idx, (_, p) in enumerate(named_parameters()):
    gen = torch.Generator(device=p.device)
    gen.manual_seed(seed)          # ← same seed → same sequence of values
    noise = torch.randn(p.shape, generator=gen)
    p.data.add_(sign * scale * noise)
    del noise                       # ← freed immediately, never accumulated
```

Two perturbed variants per seed:

$$\theta_i^+ = \theta + \sigma \varepsilon_i \qquad (\texttt{negate=False})$$
$$\theta_i^- = \theta - \sigma \varepsilon_i \qquad (\texttt{negate=True})$$

Restoration is exact because the generator is deterministic:
$$\texttt{restore}(s_i, \sigma): \quad p \leftarrow p - \sigma \cdot \varepsilon_i$$
$$= (\theta + \sigma \varepsilon_i) - \sigma \varepsilon_i = \theta \quad \checkmark$$

Perturbation scale $\sigma = 0.001$ is chosen to be small relative to the
typical L2 norm of weight tensors (~1–10 for 1.5B models), ensuring perturbed
models remain in the same behavioural neighbourhood as $\theta$.

---

### 7.2 Reward Function — deepscaler.compute_score

For a generated response $y$ to training question $x$ with ground truth $y^*$:

**Step 1 — Extract answer:**
```
model_answer = extract_answer(y)
             = last \boxed{...} content in y, or None
```

If `model_answer is None`, return `0.0` immediately. The model **must** produce
a boxed answer — there is no partial credit and no fallback.

**Step 2 — Check correctness with three graders in sequence:**

$$r_\text{binary}(y, y^*) = \begin{cases}
1.0 & \text{if } \texttt{grade\_answer\_mathd}(a, y^*) = \text{True} \\
1.0 & \text{elif } \texttt{grade\_answer\_sympy}(a, y^*) = \text{True} \\
1.0 & \text{elif } |f(a) - f(y^*)| < 10^{-6} \\
0.0 & \text{otherwise}
\end{cases}$$

where $a = \texttt{model\_answer}$, `grade_answer_mathd` uses the math\_verify
library, `grade_answer_sympy` uses symbolic equality in SymPy, and the third
check is a raw numeric tolerance fallback.

**Why this strict design?** The One-Shot RLVR paper found that adding format
rewards or partial credit rewards caused the model to optimise for answer
presentation rather than reasoning. Binary reward forces the model to actually
*solve* the problem.

---

### 7.3 Entropy Bonus — Why, What, and How We Approximate It

#### Why entropy?

Consider what happens without the entropy bonus when the model begins getting
the training question right consistently. All 20 perturbed models in the
population score `r_binary = 1.0`. The z-scored advantages $A_i = 0$ for all
$i$ (mean subtraction makes them all zero). **The ES gradient estimate becomes
exactly zero.** Training stalls.

Entropy prevents this by ensuring that even when binary rewards are equal
across the population, reward variation still exists through the entropy term —
as long as different perturbations produce responses with different uncertainty
levels. The entropy term keeps the gradient signal alive throughout training.

Additionally, entropy regularises the response distribution. Without it, the
model tends to produce repetitive, low-diversity responses — essentially
memorising one response template. This collapses the generation distribution
to a near-delta function, which makes future exploration (and ES perturbation
evaluation) uninformative.

#### Exact entropy (One-Shot RLVR / verl formulation)

The Shannon entropy of the token distribution at position $t$ is:

$$H_t = -\sum_{v \in \mathcal{V}} p_v \log p_v$$

In verl, this is computed equivalently as:

$$H_t = \log \sum_{v} e^{X_v} - \sum_v p_v X_v$$

where $X_v$ are the raw pre-softmax logits. The identity holds because:
$$\log \sum_v e^{X_v} = \log Z = \text{(log partition function)}$$
$$\sum_v p_v X_v = \mathbb{E}_{v \sim p}[X_v]$$
$$H_t = \log Z - \mathbb{E}[X_v] = -\sum_v p_v (\log p_v)$$

This requires **raw logits $X_v$ over the entire vocabulary** ($|\mathcal{V}|
\approx 150{,}000$ for Qwen2.5-Math). Obtaining them requires a full autograd
forward pass through the language model head — which vLLM does not expose.

#### The vLLM constraint

vLLM's `SamplingParams(logprobs=k)` returns the top-$k$ **log-softmax**
values (i.e., $\log p_v$ for the $k$ most probable tokens), not raw logits.
Specifically:
- `logprobs` are already log-probabilities: $\ell_v = \log p_v = X_v - \log Z$
- Only the top-$k$ entries are returned; the rest are discarded
- $\log Z$ is not exposed separately
- There is no API to retrieve full-vocabulary logits without modifying vLLM internals

Trying to reconstruct $X_v = \ell_v + \log Z$ would require $\log Z$, which
vLLM does not provide at the Python API level. Even if we patched vLLM
internals (which would be version-tied and fragile), the FlashInfer attention
backend returns activations in a batched format that does not trivially expose
per-token logit vectors without reconstructing from the LM head weights.

#### Our approximation: top-100 + tail bucket

We request `logprobs=100` from vLLM. For each generated token at position $t$,
we receive $\{(v_1, \log p_{v_1}), \ldots, (v_{100}, \log p_{v_{100}})\}$
sorted by probability descending.

Define:
$$P_k = \sum_{i=1}^{k} p_{v_i} \qquad \text{(covered mass, top-100)}$$
$$p_\text{tail} = \max(0,\ 1 - P_k) \qquad \text{(residual mass)}$$

Our per-token entropy estimate:

$$H_\text{approx} = -\sum_{i=1}^{k} p_{v_i} \log p_{v_i} \;-\; p_\text{tail} \log p_\text{tail}$$

**Why is this a tight lower bound?**

By the concavity of Shannon entropy (which follows from $-x \log x$ being
concave), distributing the tail mass $p_\text{tail}$ over more tokens can only
*increase* entropy relative to collapsing it into one bin:

$$H_\text{approx} = -\sum_{i=1}^{k} p_i \log p_i - p_\text{tail} \log p_\text{tail}
\;\leq\; -\sum_{i=1}^{k} p_i \log p_i - \sum_{j=k+1}^{|V|} p_j \log p_j
= H_\text{true}$$

Because $p_\text{tail}$ collapses the tail into one bin (underestimates the
tail's contribution to entropy), $H_\text{approx} \leq H_\text{true}$.

**Why is the approximation tight?**

For Qwen2.5-Math-1.5B generating mathematical text, the top-100 tokens
typically capture $>99.9\%$ of the probability mass: $P_{100} > 0.999$.
Therefore $p_\text{tail} < 0.001$, and the tail bucket contributes at most:

$$|{-p_\text{tail} \log p_\text{tail}}| \leq p_\text{tail} \cdot |\log p_\text{tail}|
\approx 0.001 \times 6.9 = 0.007 \text{ nats/token}$$

compared to typical token entropies of $H \approx 1$–$3$ nats/token for
chain-of-thought mathematical responses. The error is $<1\%$.

We log `entropy_coverage` = $P_{100}$ every iteration. If this falls below
0.99, the approximation degrades and increasing `logprobs` would be warranted.

**Implementation:**

```python
def _compute_token_entropy(output_obj) -> tuple[float, float]:
    completion = output_obj.outputs[0]
    token_entropies, coverages = [], []

    for lp_dict in completion.logprobs:         # one dict per generated token
        log_probs = [lp.logprob for lp in lp_dict.values()]
        probs     = np.exp(np.array(log_probs, dtype=np.float64))

        covered   = float(np.clip(np.sum(probs), 0.0, 1.0))
        tail      = max(0.0, 1.0 - covered)

        h = float(-np.sum(probs * log_probs))   # top-k entropy contribution
        if tail > 1e-9:
            h += float(-tail * np.log(tail))    # tail bucket contribution

        token_entropies.append(h)
        coverages.append(covered)

    mean_H   = float(np.mean(token_entropies))
    mean_cov = float(np.mean(coverages))
    return mean_H, mean_cov
```

The function returns:
- `mean_H`: average per-token entropy in nats — this enters the reward
- `mean_cov`: average top-100 coverage — diagnostic only

---

### 7.4 Total Reward

Combining binary correctness and entropy bonus:

$$\tilde{r}_i = r_\text{binary}(y_i, y^*) + \lambda \cdot H_\text{approx}(y_i, \theta_i^\pm)$$

with $\lambda = 0.001$ matching One-Shot RLVR exactly.

The entropy term is response-specific and perturbation-specific: two
perturbations that both get binary reward 1.0 can still have different total
rewards if their response token distributions differ in entropy. This is exactly
what keeps the ES gradient from collapsing to zero when the model solves the
training question.

---

### 7.5 Z-Score Normalisation

After evaluating all $N = 20$ perturbations, collect the total rewards:

$$\mathbf{R} = [\tilde{r}_1^+,\ \tilde{r}_1^-,\ \tilde{r}_2^+,\ \tilde{r}_2^-,\
\ldots,\ \tilde{r}_{10}^+,\ \tilde{r}_{10}^-] \in \mathbb{R}^{20}$$

Compute:

$$\mu_R = \frac{1}{N}\sum_{i=1}^{N} \tilde{r}_i, \qquad
\sigma_R = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\tilde{r}_i - \mu_R)^2}$$

Normalise each reward:

$$A_i = \frac{\tilde{r}_i - \mu_R}{\sigma_R + \varepsilon}, \quad \varepsilon = 10^{-8}$$

**Properties of this normalisation:**
- $\{A_i\}$ is zero-mean and approximately unit-variance
- Perturbations that got higher-than-average reward have $A_i > 0$ — their
  direction $\varepsilon_i$ is reinforced
- Perturbations that got lower-than-average reward have $A_i < 0$ — their
  direction $\varepsilon_i$ is suppressed
- The $\varepsilon = 10^{-8}$ guard prevents division by zero when all rewards
  are identical (e.g., all correct or all wrong)

This is **identical in formula** to One-Shot RLVR's
`compute_grpo_outcome_advantage` from verl. The conceptual difference:

- **GRPO:** population is $G$ responses from the same $\theta$ at one step
- **ES:** population is $N$ evaluations of $N$ different $\theta_i^\pm$

---

### 7.6 Antithetic ES Gradient Estimate

The antithetic evolution strategies gradient estimator (Salimans et al., 2017)
for objective $J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[\tilde{r}(y)]$:

$$\hat{\nabla}_\theta J(\theta) = \frac{1}{N\sigma} \sum_{i=1}^{N/2} (A_i^+ - A_i^-)\, \varepsilon_i$$

**Derivation sketch:**

The score function gradient estimator is:
$$\nabla_\theta J = \mathbb{E}_{y \sim \pi_\theta}[\tilde{r}(y) \nabla_\theta \log \pi_\theta(y)]$$

ES approximates $\nabla_\theta \log \pi_\theta$ implicitly by treating the
reward difference as a directional derivative:

$$\frac{d}{d\alpha} J(\theta + \alpha\varepsilon_i)\big|_{\alpha=0}
= \varepsilon_i^T \nabla_\theta J(\theta) \approx \frac{J(\theta + \sigma\varepsilon_i) - J(\theta - \sigma\varepsilon_i)}{2\sigma}$$

Substituting $J(\theta + \sigma\varepsilon_i) \approx \tilde{r}_i^+$ and
$J(\theta - \sigma\varepsilon_i) \approx \tilde{r}_i^-$ (each estimated via
one stochastic rollout), and replacing raw rewards with z-scored advantages
$A_i^\pm$ for variance reduction:

$$\hat{\nabla}_\theta J(\theta) = \frac{1}{N/2} \sum_{i=1}^{N/2} \frac{A_i^+ - A_i^-}{2\sigma} \varepsilon_i
= \frac{1}{N\sigma} \sum_{i=1}^{N/2} (A_i^+ - A_i^-)\, \varepsilon_i$$

(using $N = 2 \times (N/2)$ in the denominator for the unbiased Monte Carlo
normalisation).

The gradient ascent step with learning rate $\alpha = 0.0005$:

$$\theta \leftarrow \theta + \alpha \cdot \hat{\nabla}_\theta J(\theta)
= \theta + \frac{\alpha}{N} \sum_{i=1}^{N/2} (A_i^+ - A_i^-)\, \varepsilon_i$$

---

### 7.7 Weight Update Without Storing Noise

In implementation, we never compute $\sum_i (A_i^+ - A_i^-)\, \varepsilon_i$
as a full $d$-dimensional vector. Instead, we make $N/2$ sequential calls to
`perturb_self_weights`, each applying one term of the sum:

```python
for s_i in base_seeds:                                    # N/2 = 10 seeds
    coeff_i = (alpha / N) * (A_i_pos - A_i_neg)          # scalar
    if coeff_i != 0.0:
        engines[0].collective_rpc(
            "perturb_self_weights",
            args=(s_i, coeff_i, False, iid_noise)
        )
        # → θ ← θ + coeff_i * ε_i   (parameter-by-parameter, from seed s_i)
```

The final state of engine 0's weights is:

$$\theta_\text{engine0} = \theta_\text{old} + \sum_{i=1}^{N/2} \text{coeff}_i \cdot \varepsilon_i
= \theta_\text{old} + \frac{\alpha}{N} \sum_{i=1}^{N/2} (A_i^+ - A_i^-)\, \varepsilon_i
= \theta_\text{old} + \alpha \cdot \hat{\nabla}_\theta J$$

This is exact, requires no extra GPU memory, and takes ~$N/2 = 10$ sequential
CUDA kernel calls per iteration.

---

### 7.8 NCCL Broadcast

After engine 0 holds $\theta_\text{new}$, synchronise all engines:

$$\theta_k \leftarrow \theta_0 \quad \forall k \in \{1, 2, 3\}$$

```python
# Issued simultaneously to all 4 engines:
engines[k].collective_rpc("broadcast_all_weights", args=(0,))

# Inside WorkerExtension.broadcast_all_weights:
for _, p in self.model_runner.model.named_parameters():
    self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
torch.cuda.synchronize()
```

`PyNcclCommunicator.broadcast` is the same NCCL primitive used by PyTorch DDP
and DeepSpeed for weight synchronisation. The `StatelessProcessGroup` allows
it to run without a full `torch.distributed.init_process_group` context.

After the broadcast, all 4 engines hold identical $\theta_\text{new}$, ready
for the next iteration's perturbation evaluation.

---

## 8. ES-RLVR vs One-Shot RLVR/GRPO — Full Comparison

### 8.1 Algorithmic Structure Side-by-Side

```
ONE-SHOT RLVR with GRPO (Wang et al., 2025)
═══════════════════════════════════════════════════════════════════
  Given θ (current policy), training question x, ground truth y*:

  1. ROLLOUT: sample G=8 responses {y₁,...,y₈} from π_θ(·|x)
     (HF model in training mode, stochastic temperature sampling)

  2. REWARD: for each yᵢ:
     rᵢ = compute_score(yᵢ, y*) + λ·H(π_θ(·|yᵢ))  ∈ ℝ

  3. NORMALISE:
     Aᵢ = (rᵢ − mean(r)) / (std(r) + ε)

  4. GRPO LOSS (clip ratio):
     ρᵢ(θ) = π_θ(yᵢ|x) / π_θ_old(yᵢ|x)
     L = −(1/G) Σᵢ min(ρᵢ·Aᵢ,  clip(ρᵢ, 1±ε_clip)·Aᵢ)
         + β · KL(π_θ ‖ π_ref)

  5. UPDATE: Adam step on ∇_θ L  (requires .backward())

  θ ← θ − η ∇_θ L


ES-RLVR (this work)
═══════════════════════════════════════════════════════════════════
  Given θ (current base weights), training question x, ground truth y*:

  1. PERTURB: for i=1..N/2, draw sᵢ ∈ Uniform[0,1M]:
     θᵢ⁺ = θ + σ·εᵢ  (εᵢ ~ N(0,I), reproduced from seed sᵢ)
     θᵢ⁻ = θ − σ·εᵢ

  2. ROLLOUT (parallel): for each perturbed θᵢ±:
     yᵢ± ~ π_{θᵢ±}(·|x)   via vLLM inference

  3. REWARD:
     r̃ᵢ± = compute_score(yᵢ±, y*) + λ·H_approx(yᵢ±)  ∈ ℝ

  4. NORMALISE (same formula as GRPO):
     Aᵢ± = (r̃ᵢ± − mean(R)) / (std(R) + ε)

  5. ES GRADIENT ESTIMATE (no .backward() required):
     Δθ = (α/N) Σᵢ (Aᵢ⁺ − Aᵢ⁻)·εᵢ

  6. UPDATE: θ ← θ + Δθ   (raw in-place tensor arithmetic)

  7. BROADCAST: NCCL θ_engine0 → all engines
```

### 8.2 Detailed Component Comparison

| Component | One-Shot RLVR / GRPO | ES-RLVR (this work) | Impact |
|---|---|---|---|
| **Optimisation algorithm** | GRPO (policy gradient with clipping) | Antithetic ES (finite difference) | ES has higher gradient variance; GRPO is more sample-efficient |
| **Gradient computation** | `∇_θ log π_θ(y\|x)` via `loss.backward()` | `(A⁺ - A⁻) · ε` via reward differences | ES needs no autograd at all |
| **Model in memory as** | `nn.Module`, training mode, float32/bf16 | vLLM raw `float16` CUDA tensors | ES is ~4× more memory-efficient per GPU |
| **Inference engine** | HuggingFace generate (training mode) | vLLM (PagedAttention, continuous batching) | vLLM has significantly higher throughput |
| **Parallelism axis** | Batch: $G$ rollouts from **one** $\theta$ | Population: $N$ evaluations of $N$ **different** $\theta_i^\pm$ | Different trade-off: GRPO shares θ, ES samples around θ |
| **Training data** | 1 question × 128 rows | 1 question × 128 rows | Identical |
| **Reward type** | Binary `{0, 1}` | Binary `{0, 1}` | Identical |
| **Advantage formula** | $A_i = (r_i - \bar{r}) / \sigma_r$ | $A_i = (r_i - \bar{r}) / \sigma_r$ | **Identical formula** — different population |
| **Entropy bonus** | $\lambda \cdot H_\text{exact}$ (full vocab logits from autograd pass) | $\lambda \cdot H_\text{approx}$ (top-100 + tail bucket) | <1% error on Qwen2.5-Math |
| **Entropy coefficient** | $\lambda = 0.001$ | $\lambda = 0.001$ | Identical |
| **KL penalty** | $\beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})$, $\beta=0.04$ | **Omitted** | No reference model pass feasible in vLLM |
| **Reference model** | Separate frozen $\pi_\text{ref}$ in memory | Not needed | ES saves ~3 GB GPU memory per engine |
| **Optimiser states** | Adam: $m$ and $v$ buffers (~2× model size) | None — raw in-place add | ES saves ~6 GB GPU memory per engine |
| **GPU memory (1.5B)** | ~20 GB (weights + grads + Adam + activations) | ~3 GB (weights only) | ES fits 4 engines on 4× 40GB GPUs with KV cache headroom |
| **Gradient noise** | Low — exact log-prob gradient | High — estimated from $N$ scalar rewards | GRPO converges faster per parameter update |
| **Convergence speed** | Faster per iteration | Slower — needs larger $N$ for signal | Trade-off against GPU throughput |
| **Noise storage** | N/A | **Zero** — seed-reproducible, never stored | 60 GB avoided for $N=20$, 1.5B model |
| **Implementation** | Standard RL libs (verl, trl, transformers) | Custom `WorkerExtension` + Ray orchestration | ES requires more infrastructure |

### 8.3 The Normalisation Identity — Same Formula, Different Meaning

```
GRPO (One-Shot RLVR):

  θ fixed → generate 8 responses → score them
  R = [r₁, r₂, ..., r₈]   ← rewards from 8 stochastic samples of π_θ
  Aᵢ = (rᵢ - mean(R)) / std(R)
  Meaning: "how good is THIS RESPONSE relative to OTHER RESPONSES from same θ?"

──────────────────────────────────────────────────────────────────

ES-RLVR (this work):

  θ fixed → perturb N times → evaluate each perturbation → score
  R = [r₁⁺, r₁⁻, ..., r₁₀⁺, r₁₀⁻]   ← rewards from 20 different θᵢ±
  Aᵢ = (rᵢ - mean(R)) / std(R)
  Meaning: "how good is THIS PERTURBATION relative to OTHER PERTURBATIONS?"
```

The formula is byte-for-byte identical. The semantic is fundamentally different:
GRPO uses response diversity under the same policy; ES uses parameter-space
perturbation diversity around the same base weights.

### 8.4 What We Lose vs GRPO

1. **KL regularisation.** GRPO adds `β · KL(π_θ ‖ π_ref)` to prevent the
   policy from drifting too far from the pretrained baseline. Without it, ES
   updates could in principle move weights in a direction that destroys
   pre-trained capabilities. In practice, the small `σ = 0.001` perturbation
   scale and `α = 0.0005` learning rate mean total weight change per iteration
   is tiny (on the order of $10^{-6}$ of parameter magnitude), providing an
   implicit constraint.

2. **Exact entropy.** Our top-100 approximation introduces a small downward
   bias in the entropy signal. This is acceptable given the approximation
   error is <1% of the true entropy value for this model on this domain.

3. **Sample efficiency.** GRPO gets $G=8$ gradient signal contributions from
   one $\theta$; ES gets $N=20$ scalar rewards from 20 different $\theta_i^\pm$.
   The ES gradient estimate is higher-variance per update, requiring more
   iterations for the same learning progress.

### 8.5 What We Gain vs GRPO

1. **No backpropagation.** Enables training inside vLLM with no modification
   to the inference engine's compute path.

2. **Memory efficiency.** 3 GB per GPU instead of 20 GB — enabling 4 parallel
   engines on 4× 40GB A100s with large KV cache headroom.

3. **Infrastructure simplicity.** The weight update is `p.data.add_(coeff * noise)`
   — fewer moving parts than an Adam optimiser with gradient accumulation.

4. **Exact weight control.** Because we never call an optimiser, we have
   deterministic, exact control over what happens to each weight tensor at each
   step. There are no momentum buffers, no learning rate schedules to worry
   about during debugging.

---

## 9. File Structure & Code Walkthrough

```
ES-rlvr/
├── README.md                          ← this file
├── CLAUDE.md                          ← AI assistant guidelines
│
├── es_rlvr_train.py                   ← main training script
│   │
│   ├── class ESNcclLLM(LLM)
│   │     LLM subclass that pops CUDA_VISIBLE_DEVICES and disables V1
│   │     multiprocessing before vLLM init. Required for multi-engine
│   │     Ray deployment where each actor manages its own GPU.
│   │
│   ├── launch_engines(num_engines, model_path)
│   │     Allocates one Ray PlacementGroup per engine (1 GPU each),
│   │     then launches ESNcclLLM actors with WorkerExtension injected
│   │     via worker_extension_cls="utils.worker_extn.WorkerExtension".
│   │     Returns (engines, placement_groups).
│   │
│   ├── load_task_datas(parquet_path, tokenizer)
│   │     Reads a verl-schema parquet file (columns: prompt, reward_model,
│   │     data_source). Prepends SYSTEM_PROMPT to the chat, applies the
│   │     tokenizer's chat template with add_generation_prompt=True.
│   │     Returns list of {prompt_str, ground_truth, data_source, question}.
│   │
│   ├── evaluate_handle(llm, task_datas)
│   │     Fires off an async vLLM generation on llm (non-blocking Ray remote
│   │     call). Returns (object_ref, start_timestamp). logprobs=100 is
│   │     requested for entropy computation.
│   │
│   ├── _compute_token_entropy(output_obj)
│   │     Extracts per-token top-100 logprobs from the vLLM output object.
│   │     Computes H_approx = -Σ pᵢ log pᵢ - p_tail log p_tail per token.
│   │     Returns (mean_entropy_nats, mean_coverage).
│   │     See Section 7.3 for full mathematical derivation.
│   │
│   ├── _postprocess_outputs(outputs, task_datas, entropy_coeff)
│   │     For each (output, task) pair:
│   │       binary = compute_score(output.text, task.ground_truth)
│   │       H, cov = _compute_token_entropy(output)
│   │       total  = binary + entropy_coeff * H
│   │     Returns dict of per-batch aggregated statistics.
│   │     Optionally prints question + full response + scores to console.
│   │
│   ├── evaluate_val_set(engine, val_task_datas, val_batch_size, ...)
│   │     Greedy decoding (temperature=0) on up to val_batch_size Math500
│   │     examples using engine 0 (which holds θ_new after each update).
│   │     Prints first val_show_n examples with full reasoning chains.
│   │     Logs accuracy to TensorBoard "val/accuracy".
│   │
│   ├── save_plots(history, output_dir)
│   │     Saves a 2×2 matplotlib figure:
│   │       [train correctness]  [Math500 val accuracy + baseline/best lines]
│   │       [reward mean ± std]  [token entropy proxy]
│   │
│   └── main(args)
│         Full training loop: setup → iteration loop → save final weights.
│         See Section 6 for detailed flow.
│
├── deepscaler.py                      ← reward oracle
│   │
│   ├── SYSTEM_PROMPT
│   │     "You are a mathematical reasoning assistant. Always solve step
│   │      by step and put your final answer inside \boxed{}..."
│   │     Injected into every query via prepend to chat template.
│   │
│   └── compute_score(data_source, solution_str, ground_truth, ...)
│         Binary {0, 1} reward.
│         Step 1: extract_answer(solution_str) → last \boxed{} content
│         Step 2: grade_answer_mathd OR grade_answer_sympy OR numeric tol
│         Returns 1.0 or 0.0. No partial credit.
│
└── utils/
    └── worker_extn.py                 ← injected into every vLLM worker
        │
        ├── _noise_seed(base_seed, param_idx, iid_noise)
        │     Returns base_seed + param_idx if iid_noise else base_seed.
        │     Controls whether all params share one noise stream or each
        │     has an independent one.
        │
        ├── perturb_self_weights(seed, noise_scale, negate, iid_noise)
        │     The core ES operation. For each named parameter tensor:
        │       gen.manual_seed(noise_seed)
        │       ε = torch.randn(p.shape, generator=gen)
        │       p.data.add_(±noise_scale * ε)
        │       del ε   ← freed immediately
        │     CUDA synchronise. Returns True.
        │
        ├── restore_self_weights(seed, sigma, iid_noise)
        │     Exact inverse of perturb: subtracts +sigma*ε (same seed).
        │     Because torch.Generator is deterministic: same seed → same ε.
        │
        ├── init_inter_engine_group(master_address, port, rank, world_size)
        │     Creates a StatelessProcessGroup and wraps it in
        │     PyNcclCommunicator. Stored as self.inter_pg.
        │     Called once at startup for all 4 engines.
        │
        ├── broadcast_all_weights(src_rank)
        │     For each parameter tensor:
        │       self.inter_pg.broadcast(p, src=src_rank, stream=current_stream)
        │     CUDA synchronise. Syncs engine src_rank → all others.
        │
        ├── save_self_weights_to_disk(filepath)
        │     Saves {name: p.detach().cpu()} as torch.save → .pth file.
        │
        └── load_weights_from_disk(filepath)
              Loads a .pth and copies matched tensors into the live model.
              Returns number of matched parameter tensors.
```

---

## 10. Setup & Running

### Environment

```bash
conda activate grpo
# Python 3.12, torch 2.10.0, vllm 0.17.0, ray 2.54.0, transformers 4.57.0
```

### Quick start

```bash
conda activate grpo
cd /path/to/ES-rlvr

python es_rlvr_train.py \
  --model_name         Qwen/Qwen2.5-Math-1.5B-Instruct \
  --parquet_path       "Dataset parquet/pi1_r128.parquet" \
  --val_parquet_path   "Dataset parquet/math500.parquet" \
  --num_engines        4 \
  --cuda_devices       0,1,2,3
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | HuggingFace model ID or local path |
| `--parquet_path` | `Dataset parquet/pi1_r128.parquet` | Training dataset (verl schema) |
| `--val_parquet_path` | `Dataset parquet/math500.parquet` | Validation dataset |
| `--val_batch_size` | `50` | Number of Math500 examples per val step |
| `--sigma` | `0.001` | ES perturbation scale σ |
| `--alpha` | `0.0005` | ES learning rate α |
| `--population_size` | `20` | Total perturbations per iteration (N) |
| `--antithetic` | `True` | Use ±ε pairs; `--no-antithetic` for one-sided |
| `--iid_noise` | `False` | Independent noise per parameter layer |
| `--entropy_coeff` | `0.001` | Entropy bonus coefficient λ |
| `--num_engines` | `4` | Number of GPUs / vLLM engines |
| `--num_iterations` | `200` | Training iterations |
| `--val_every` | `10` | Math500 validation frequency (iterations) |
| `--output_every` | `1` | Print full training sample every N iters |
| `--val_show_n` | `3` | Math500 examples to display in full at val |
| `--global_seed` | `None` | Fix RNG seed for reproducibility |
| `--verbose` | `False` | Print per-seed reward breakdown |
| `--experiment_dir` | `outputs/es_rlvr_oneshot` | Root output directory |
| `--cuda_devices` | `0,1,2,3` | Comma-separated CUDA device indices |

---

## 11. Outputs & Monitoring

All outputs are written to:
```
outputs/es_rlvr_oneshot/run_<YYYYMMDD_HHMMSS>/
├── events.out.tfevents.*      ← TensorBoard event file
├── metrics.csv                ← per-iteration and per-val-step CSV log
├── training_curves.png        ← 2×2 training figure (updated at each val step)
└── model_saves/
    ├── base_model/            ← HF checkpoint saved before training
    └── final_model_iter_N/
        └── pytorch_model.pth  ← state_dict of final ES-updated weights
```

### TensorBoard

```bash
tensorboard --logdir outputs/es_rlvr_oneshot
```

| Tag | Description |
|---|---|
| `reward/mean` | Mean total reward ($\tilde{r}$) across population |
| `reward/std` | Std of total reward — indicates population diversity |
| `reward/correctness` | Mean binary correctness across population |
| `reward/max_correctness` | Best binary correctness in the population this iter |
| `reward/entropy` | Mean per-token Shannon entropy estimate (nats/token) |
| `reward/entropy_coverage` | Mean top-100 probability coverage — should stay >0.999 |
| `val/accuracy` | Math500 greedy accuracy at each validation step |
| `time/iteration` | Wall-clock seconds per full iteration |
| `time/broadcast` | NCCL broadcast wall time |
| `time/perturbation_application` | ES update accumulation wall time |

### Training curves figure (4 panels)

1. **Train correctness** — binary correctness of the training question over
   iterations. As training progresses, this should rise and stabilise.
2. **Math500 val accuracy** — out-of-distribution generalisation. Baseline
   (iteration 0) and best-so-far are marked with dashed lines.
3. **ES reward mean ± std** — the total reward (binary + entropy) with ±1
   standard deviation band. Narrowing std indicates population convergence.
4. **Token entropy** — nats per generated token. If this falls sharply,
   the model is collapsing to repetitive responses; the entropy bonus is the
   safeguard against this.

---

## 12. References

### Primary (the work we build on)

1. **One-Shot RLVR** — the paper whose methodology we implement with ES:
   > Yupeng Wang, Chufan Shi, Minghao Wu, Yuxuan Li, Lifan Yuan, Ganqu Cui,
   > Weinan E, et al. *"One-Shot RLVR: Exploring the Limit of Reinforcement
   > Learning with a Single AI-Generated Training Example."*
   > NeurIPS 2025. [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)
   > Code: [github.com/ypwang61/One-Shot-RLVR](https://github.com/ypwang61/One-Shot-RLVR)

2. **ES architecture base** — the vLLM+Ray+NCCL multi-engine layout:
   > VsonicV. *es-fine-tuning-paper.*
   > [github.com/VsonicV/es-fine-tuning-paper](https://github.com/VsonicV/es-fine-tuning-paper)

### Foundational methods

3. **Evolution Strategies as RL** — antithetic ES gradient estimator:
   > Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever.
   > *"Evolution Strategies as a Scalable Alternative to Reinforcement Learning."*
   > arXiv:1703.03864, 2017.

4. **GRPO** — the policy gradient method we replace:
   > Zhihong Shao, Peiyi Wang, Qihao Zhu, et al.
   > *"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
   > Language Models."* arXiv:2402.03300, 2024.

5. **vLLM / PagedAttention** — our inference engine:
   > Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.
   > *"Efficient Memory Management for Large Language Model Serving with
   > PagedAttention."* SOSP 2023.

6. **verl framework** — parquet data format, reward grading utilities, GRPO
   normalisation reference:
   > Guangming Sheng, Chi Zhang, Zilingfeng Ye, et al.
   > *"HybridFlow: Flexible and Efficient RLHF for Large Language Models."*
   > arXiv:2409.19256, 2024.
   > [github.com/volcengine/verl](https://github.com/volcengine/verl)

7. **Qwen2.5-Math** — the base model:
   > Qwen Team. *"Qwen2.5-Math Technical Report."* arXiv:2409.12122, 2024.
