#!/usr/bin/env python3
"""
Thesis proof: show pi1_r128.parquet contains 1 unique problem repeated 128 times.
"""
import pandas as pd
import json

df = pd.read_parquet("data/train/pi1_r128.parquet")

print("=" * 60)
print(f"Total rows       : {len(df)}")

# Extract questions
questions = []
for _, row in df.iterrows():
    chat = list(row["prompt"])
    q = next((m.get("content", "") for m in chat
               if isinstance(m, dict) and m.get("role") == "user"), "")
    questions.append(q)

unique_questions = set(questions)
print(f"Unique questions : {len(unique_questions)}")
print(f"Repeated         : {len(df)} x the same problem")

# Ground truth
gts = set()
for _, row in df.iterrows():
    rm = row["reward_model"]
    gt = rm["ground_truth"] if isinstance(rm, dict) else str(rm)
    gts.add(gt)
print(f"Unique answers   : {gts}")
print()
print("Problem text (first 300 chars):")
print(list(unique_questions)[0][:300])
print("=" * 60)
