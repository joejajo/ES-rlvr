#!/usr/bin/env python3
"""
Thesis proof figures:
  1. Training reward mean + std over iterations (shows saturation at iter 134)
  2. Validation accuracy over iterations (shows collapse after saturation)

Usage:
  python analysis/proof_saturation.py --log_file <path_to_.out_file> --out_dir analysis/figures
"""
import argparse
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Parse args ────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--log_file",  required=True, help="Path to the SLURM .out log file")
p.add_argument("--out_dir",   default="analysis/figures")
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Parse log ─────────────────────────────────────────────────────────────────
reward_iters, reward_mean, reward_std = [], [], []
val_iters, val_acc = [], []

reward_pat = re.compile(
    r"\[REWARD\]\s+mean=([\d.]+)\s+std=([\d.]+).*nonzero=([\d.]+)"
)
iter_pat   = re.compile(r"─+\s+iter\s+(\d+)/")
val_pat    = re.compile(r"\[VAL\].*?acc=([\d.]+)%.*?iter[=\s](\d+)", re.IGNORECASE)
# also match inline_val style: "accuracy": 0.344
val_pat2   = re.compile(r"accuracy[_\s]*[=:]\s*([\d.]+)[%]?\s.*?iter[_\s]*[=:]?\s*(\d+)", re.IGNORECASE)

current_iter = None
with open(args.log_file) as f:
    for line in f:
        m = iter_pat.search(line)
        if m:
            current_iter = int(m.group(1))
            continue
        m = reward_pat.search(line)
        if m and current_iter is not None:
            reward_iters.append(current_iter)
            reward_mean.append(float(m.group(1)))
            reward_std.append(float(m.group(2)))
            continue
        # val lines like: [VAL] iter=40 accuracy=34.40% ...
        m = re.search(r"iter[=\s]+(\d+).*?accuracy[=\s]+([\d.]+)%", line, re.IGNORECASE)
        if m:
            val_iters.append(int(m.group(1)))
            val_acc.append(float(m.group(2)))

if not reward_iters:
    print("[WARN] No reward lines found — check log format.")
if not val_iters:
    print("[WARN] No validation lines found — check log format.")

print(f"Parsed {len(reward_iters)} reward rows, {len(val_iters)} val rows.")

# ── Figure 1: Training reward mean + std ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ri = np.array(reward_iters)
rm = np.array(reward_mean)
rs = np.array(reward_std)

ax.plot(ri, rm, color="#2196F3", linewidth=1.2, label="mean reward")
ax.fill_between(ri, np.clip(rm - rs, 0, 1), np.clip(rm + rs, 0, 1),
                alpha=0.25, color="#2196F3", label="±1 std")

# Mark saturation point
sat_iters = ri[rs == 0.0]
if len(sat_iters):
    sat = sat_iters[0]
    ax.axvline(sat, color="red", linestyle="--", linewidth=1.2,
               label=f"std=0 (saturation at iter {sat})")
    ax.annotate(f"reward/std = 0\nfrom iter {sat}",
                xy=(sat, 0.5), xytext=(sat + 30, 0.4),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red")

ax.set_xlabel("Iteration")
ax.set_ylabel("Training Reward")
ax.set_title("ES Training Reward — Saturation on Single Repeated Problem")
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
out1 = os.path.join(args.out_dir, "proof_reward_saturation.png")
fig.savefig(out1, dpi=150)
print(f"Saved → {out1}")
plt.close()

# ── Figure 2: Validation accuracy ────────────────────────────────────────────
if val_iters:
    fig, ax = plt.subplots(figsize=(10, 4))
    vi = np.array(val_iters)
    va = np.array(val_acc)

    ax.plot(vi, va, color="#4CAF50", linewidth=1.4, marker="o",
            markersize=3, label="val accuracy (MATH500)")
    ax.axhline(va[0], color="gray", linestyle=":", linewidth=1,
               label=f"baseline @ iter {vi[0]}: {va[0]:.1f}%")

    best_idx = np.argmax(va)
    ax.scatter(vi[best_idx], va[best_idx], color="gold", s=80, zorder=5,
               label=f"best: {va[best_idx]:.1f}% @ iter {vi[best_idx]}")

    if len(sat_iters):
        ax.axvline(sat, color="red", linestyle="--", linewidth=1.2,
                   label=f"reward saturates (iter {sat})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("MATH500 Accuracy (%)")
    ax.set_title("ES Validation Accuracy — Degrades After Reward Saturation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, "proof_val_degradation.png")
    fig.savefig(out2, dpi=150)
    print(f"Saved → {out2}")
    plt.close()

# ── Figure 3: Combined two-panel ─────────────────────────────────────────────
if val_iters:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(ri, rm, color="#2196F3", linewidth=1.2)
    ax1.fill_between(ri, np.clip(rm - rs, 0, 1), np.clip(rm + rs, 0, 1),
                     alpha=0.2, color="#2196F3")
    if len(sat_iters):
        ax1.axvline(sat, color="red", linestyle="--", linewidth=1.2)
    ax1.set_ylabel("Training Reward (mean ± std)")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_title("ES on One-Shot Training Data (pi1_r128: 1 problem × 128 copies)")
    ax1.grid(alpha=0.3)

    dead_patch = mpatches.Patch(color="red", alpha=0.08, label="dead zone: std=0, no gradient")
    if len(sat_iters):
        ax1.axvspan(sat, ri[-1], alpha=0.06, color="red")
        ax1.legend(handles=[dead_patch], fontsize=8, loc="lower right")

    ax2.plot(vi, va, color="#4CAF50", linewidth=1.4, marker="o", markersize=3)
    ax2.axhline(va[0], color="gray", linestyle=":", linewidth=1,
                label=f"base: {va[0]:.1f}%")
    ax2.scatter(vi[best_idx], va[best_idx], color="gold", s=80, zorder=5,
                label=f"best: {va[best_idx]:.1f}%")
    if len(sat_iters):
        ax2.axvline(sat, color="red", linestyle="--", linewidth=1.2,
                    label=f"saturation (iter {sat})")
        ax2.axvspan(sat, vi[-1], alpha=0.06, color="red")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MATH500 Accuracy (%)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out3 = os.path.join(args.out_dir, "proof_combined.png")
    fig.savefig(out3, dpi=150)
    print(f"Saved → {out3}")
    plt.close()

print("\nDone.")
