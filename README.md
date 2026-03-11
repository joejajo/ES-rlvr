# ES-RLVR

**Evolution Strategies + One-Shot-RLVR for Mathematical Reasoning**

Train a language model to solve math problems using a **single training example** and a **binary right/wrong reward** — no backpropagation, no gradient tape, pure inference infrastructure via vLLM + Ray.

---

## What this does

ES-RLVR replaces the GRPO gradient update from One-Shot-RLVR with **Evolution Strategies (ES)**:

- Perturb model weights with Gaussian noise across multiple vLLM engines
- Roll out generations and score them with a binary reward (`deepscaler.py`)
- Compute a z-score normalised ES gradient and update the base weights
- Entropy bonus discourages mode collapse
- Antithetic pairs (mirrored noise) reduce variance
- Validation on MATH-500 runs on the fly during training

No optimizer state, no autograd — weight updates happen directly inside the inference engine via a custom `WorkerExtension`.

---

## Credits

**One-Shot-RLVR** — Yupeng Wang et al.
The core insight (single-example training, binary reward, z-score normalisation, entropy bonus) comes entirely from their work.
- Paper: [arXiv:2504.20571](https://arxiv.org/pdf/2504.20571)
- Code: [ypwang61/One-Shot-RLVR](https://github.com/ypwang61/One-Shot-RLVR)

**ES Fine-Tuning (VsonicV)** — the vLLM + Ray + NCCL architecture and `WorkerExtension` pattern used here is based on:
- Code: [VsonicV/es-fine-tuning-paper](https://github.com/VsonicV/es-fine-tuning-paper)
