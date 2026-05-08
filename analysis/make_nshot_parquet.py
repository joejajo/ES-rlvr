#!/usr/bin/env python3
"""
Ablation 3: Create N-shot training parquets for ES diversity ablation.

Takes first N unique problems from a source parquet (e.g. DSR-sub or full
MATH train), repeats each to fill a batch of 128, saves as piN_r<128//N>.parquet.

Usage:
  python analysis/make_nshot_parquet.py \
      --source_parquet data/train/dsr_sub.parquet \
      --output_dir     data/train/ablation \
      --n_values       1 2 4 8 16 32
"""
import argparse
import os
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--source_parquet", required=True,
               help="Source parquet with diverse unique problems")
p.add_argument("--output_dir",     default="data/train/ablation")
p.add_argument("--n_values",       type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
p.add_argument("--batch_size",     type=int, default=128,
               help="Total rows per parquet (repeats to fill)")
args = p.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

df = pd.read_parquet(args.source_parquet)

# Extract unique questions to deduplicate
questions = []
for _, row in df.iterrows():
    chat = list(row["prompt"])
    q = next((m.get("content", "") for m in chat
               if isinstance(m, dict) and m.get("role") == "user"), "")
    questions.append(q)
df["_question"] = questions
df_unique = df.drop_duplicates(subset="_question").drop(columns="_question").reset_index(drop=True)

print(f"Source: {len(df)} rows, {len(df_unique)} unique problems")

for n in args.n_values:
    if n > len(df_unique):
        print(f"[SKIP] n={n} > unique problems ({len(df_unique)})")
        continue

    subset   = df_unique.iloc[:n]
    repeats  = args.batch_size // n
    leftover = args.batch_size  % n

    rows = []
    for _ in range(repeats):
        rows.append(subset)
    if leftover:
        rows.append(subset.iloc[:leftover])
    out_df = pd.concat(rows, ignore_index=True)

    out_path = os.path.join(args.output_dir, f"pi{n}_r{args.batch_size}.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"[SAVED] n={n:3d} unique × {repeats} repeats → {len(out_df)} rows → {out_path}")
