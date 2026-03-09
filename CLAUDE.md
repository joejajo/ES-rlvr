# CLAUDE.md — ES-rlvr

This file provides guidance for AI assistants (e.g. Claude Code) working in this repository.

---

## Repository Overview

**Name:** ES-rlvr
**Owner:** joejajo
**Remote:** `http://local_proxy@127.0.0.1:41211/git/joejajo/ES-rlvr`

This repository appears to be in its initial state (no committed source code yet). Update this file once the project structure is established.

---

## Git Workflow

### Branching Convention

- Feature branches: `feature/<short-description>`
- Bug fixes: `fix/<short-description>`
- AI/automated work: `claude/<task-id>` (e.g. `claude/claude-md-mmjmsyal1uqqzz0l-l304z`)

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

Examples:
```
feat(training): add PPO rollout buffer
fix(env): correct reward normalization bug
docs: update CLAUDE.md with project structure
```

### Push Instructions

Always use:
```bash
git push -u origin <branch-name>
```

Branch names for AI-generated work must start with `claude/` and end with the session ID.

---

## Development Setup

### Conda Environment

**Activate before running anything:**
```bash
conda activate grpo
# env path: /home/woody/iwi7/iwi7107h/conda_envs/grpo
```

Key package versions in `grpo` env:
| Package | Version |
|---|---|
| Python | 3.12.11 |
| torch | 2.10.0 |
| vllm | 0.17.0 |
| ray | 2.54.0 |
| transformers | 4.57.0 |
| peft | 0.17.1 |
| trl | 0.23.1 |
| accelerate | 1.10.1 |
| datasets | 4.2.0 |
| pandas | 2.3.3 |
| pyarrow | 21.0.0 |
| numpy | 2.2.6 |
| tensorboard | 2.20.0 |
| flashinfer-python | 0.6.4 |
| triton | 3.6.0 |
| CUDA runtime | 12.4 (pytorch-cuda 12.4) |
| nvidia-nccl-cu12 | 2.27.5 |
| sympy | 1.14.0 |

Full env snapshot: `conda list` in the `grpo` environment (recorded 2026-03-09).

### Running the training script

```bash
conda activate grpo
cd /path/to/ES-rlvr
python es_rlvr_train.py \
  --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --parquet_path "Dataset parquet/pi1_r128.parquet" \
  --val_parquet_path "Dataset parquet/math500.parquet" \
  --num_engines 4 \
  --cuda_devices 0,1,2,3
```

---

## Project Structure

```
ES-rlvr/
├── CLAUDE.md                          # This file
├── es_rlvr_train.py                   # Main training script (vLLM+Ray+NCCL ES loop)
├── es_fine_tuning_deepscaler_accl.py  # Reference: VsonicV/es-fine-tuning-paper base
├── deepscaler.py                      # Reward: binary correctness via \boxed{} extraction
├── utils/
│   └── worker_extn.py                 # vLLM WorkerExtension (perturb/restore/broadcast/save/load)
└── Dataset parquet/
    ├── pi1_r128.parquet               # Train: 128 rows, 1 unique question, ground_truth="12.8"
    └── math500.parquet                # Val:   500 rows, simplerl/math500 source
```

### Key files

- **`es_rlvr_train.py`**: ES-RLVR training loop combining:
  - vLLM + Ray + NCCL architecture (from `es_fine_tuning_deepscaler_accl.py`)
  - One-Shot-RLVR methodology (binary reward, GRPO z-score, entropy bonus, antithetic pairs)
  - On-the-go validation on math500, model output display during training

- **`deepscaler.py`**: Authoritative reward module. Returns 1.0 (correct) / 0.0 (wrong).
  No format reward. Uses `extract_answer` → `grade_answer_mathd` / `grade_answer_sympy`.

- **`utils/worker_extn.py`**: vLLM `WorkerExtension` injected via `worker_extension_cls`.
  Methods: `perturb_self_weights`, `restore_self_weights`, `init_inter_engine_group`,
  `broadcast_all_weights`, `save_self_weights_to_disk`, `load_weights_from_disk`.

---

## Testing

> Update once tests are established.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run a specific test file
pytest tests/test_trainer.py
```

Conventions:
- Tests live in `tests/` mirroring `src/` structure.
- Use `pytest` as the test runner.
- All new code should include corresponding tests.

---

## Code Style

> Update based on project linting configuration.

- Python: follow [PEP 8](https://peps.python.org/pep-0008/) with `ruff` or `black` for formatting.
- Imports: stdlib → third-party → local, separated by blank lines.
- Type hints encouraged for all public functions.
- Docstrings: NumPy or Google style.

Lint / format commands (update as configured):
```bash
ruff check .
ruff format .
mypy src/
```

---

## AI Assistant Guidelines

When working in this repository:

1. **Read before writing.** Always read existing files before modifying them.
2. **Minimal changes.** Only change what is necessary for the task at hand.
3. **No speculative abstractions.** Don't add helpers or utilities beyond what is immediately needed.
4. **No unused imports or variables.**
5. **No security vulnerabilities.** Avoid command injection, SQL injection, unsafe deserialization, etc.
6. **Commit incrementally.** Prefer small, focused commits over large monolithic ones.
7. **Update this file.** When the project structure changes significantly, keep CLAUDE.md current.
8. **Do not push to `main` or `master` directly.** Always use a feature branch and PR.

---

## Key Contacts / References

- Repository owner: `joejajo`
- Update this section with links to design docs, issue trackers, or runbooks as they are created.
