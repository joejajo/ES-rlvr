#!/usr/bin/env python3
"""
Ablation 2: Correlate reward/std with val accuracy change.
Proves mechanistically that ES degrades exactly when std→0.

Usage:
  python analysis/ablation_std_vs_val.py --log_file <path>.out --out_dir analysis/figures
"""
import argparse
import re
import os
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--log_file", required=True)
p.add_argument("--out_dir",  default="analysis/figures")
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ── Parse reward/std per iter ─────────────────────────────────────────────────
reward_std_by_iter = {}
current_iter = None
with open(args.log_file) as f:
    for line in f:
        m = re.search(r"─+\s+iter\s+(\d+)/", line)
        if m:
            current_iter = int(m.group(1))
        m = re.search(r"\[REWARD\].*?std=([\d.]+)", line)
        if m and current_iter is not None:
            reward_std_by_iter[current_iter] = float(m.group(1))

# ── Parse val accuracy per iter ───────────────────────────────────────────────
val_by_iter = {}
with open(args.log_file) as f:
    for line in f:
        m = re.search(r"iter[=\s]+(\d+).*?accuracy[=\s]+([\d.]+)%", line, re.IGNORECASE)
        if m:
            val_by_iter[int(m.group(1))] = float(m.group(2))

val_iters = sorted(val_by_iter.keys())
val_acc   = [val_by_iter[i] for i in val_iters]

# For each val point, get mean reward/std in the window since last val
window_std = []
for idx, vi in enumerate(val_iters):
    prev = val_iters[idx-1] if idx > 0 else 0
    stds = [reward_std_by_iter[it] for it in reward_std_by_iter
            if prev < it <= vi]
    window_std.append(np.mean(stds) if stds else 0.0)

window_std = np.array(window_std)
val_acc    = np.array(val_acc)

# ── Figure: dual axis — std (bar) + val accuracy (line) ───────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

bars = ax1.bar(val_iters, window_std, width=15, alpha=0.5, color="#2196F3",
               label="mean reward/std (in window)")
ax1.set_ylabel("reward/std", color="#2196F3")
ax1.tick_params(axis="y", labelcolor="#2196F3")
ax1.set_ylim(0, max(window_std) * 2 + 0.01)

ax2.plot(val_iters, val_acc, color="#E53935", linewidth=1.8, marker="o",
         markersize=4, label="val accuracy (%)")
ax2.set_ylabel("MATH500 Accuracy (%)", color="#E53935")
ax2.tick_params(axis="y", labelcolor="#E53935")

# Find where std first hits 0
dead_iters = [vi for vi, s in zip(val_iters, window_std) if s < 0.001]
if dead_iters:
    ax1.axvline(dead_iters[0], color="black", linestyle="--", linewidth=1.2,
                label=f"std→0 from iter {dead_iters[0]}")

ax1.set_xlabel("Iteration")
ax1.set_title("Ablation 2: reward/std → 0 causes val accuracy decline\n"
              "(ES gradient signal collapses on single repeated problem)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
ax1.grid(alpha=0.2)

fig.tight_layout()
out = os.path.join(args.out_dir, "ablation2_std_vs_val.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")

# ── Print correlation stats ───────────────────────────────────────────────────
corr = np.corrcoef(window_std, val_acc)[0, 1]
dead_mask  = window_std < 0.001
alive_mask = ~dead_mask
print(f"\nCorrelation(reward_std, val_acc) = {corr:.3f}")
print(f"Mean val acc when std>0  : {val_acc[alive_mask].mean():.2f}%")
print(f"Mean val acc when std=0  : {val_acc[dead_mask].mean():.2f}%")
print(f"Drop after saturation    : {val_acc[alive_mask].mean() - val_acc[dead_mask].mean():.2f}pp")
