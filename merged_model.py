import pandas as pd
import numpy as np

df = pd.read_csv("final1_filled.csv")

# ── CPU TIER based on b_singleScore + b_multiScore ──────────────
# Uses percentiles from the actual data so tiers are balanced
cpu_combined = (df["b_singleScore"] * 0.4) + (df["b_multiScore"] * 0.6)

cpu_bins  = [0,
             cpu_combined.quantile(0.25),
             cpu_combined.quantile(0.50),
             cpu_combined.quantile(0.75),
             float("inf")]
cpu_labels = [1, 2, 3, 4]   # 1=basic, 2=mid, 3=high, 4=top

df["cpu_tier"] = pd.cut(cpu_combined, bins=cpu_bins,
                         labels=cpu_labels, include_lowest=True).astype(int)

# ── GPU TIER based on b_G3Dmark ──────────────────────────────────
# G3Dmark is the standard 3D benchmark — best single signal for GPU
gpu_bins  = [0,
             df["b_G3Dmark"].quantile(0.25),
             df["b_G3Dmark"].quantile(0.50),
             df["b_G3Dmark"].quantile(0.75),
             float("inf")]
gpu_labels = [0, 1, 2, 3]   # 0=integrated, 1=entry, 2=mid, 3=high

df["gpu_tier"] = pd.cut(df["b_G3Dmark"], bins=gpu_bins,
                         labels=gpu_labels, include_lowest=True).astype(int)

# ── REPORT ───────────────────────────────────────────────────────
print("CPU Tier distribution:")
print(df["cpu_tier"].value_counts().sort_index())
print(f"\nCPU tier cutoffs (combined score):")
for i, (lo, hi) in enumerate(zip(cpu_bins, cpu_bins[1:]), 1):
    count = ((cpu_combined >= lo) & (cpu_combined < hi)).sum()
    print(f"  Tier {i}: {lo:8.1f} – {hi:8.1f}  ({count} laptops)")

print("\nGPU Tier distribution:")
print(df["gpu_tier"].value_counts().sort_index())
print(f"\nGPU tier cutoffs (G3Dmark):")
tier_names = ["0 - Integrated/Basic", "1 - Entry", "2 - Mid", "3 - High"]
for i, (lo, hi, name) in enumerate(zip(gpu_bins, gpu_bins[1:], tier_names)):
    count = ((df["b_G3Dmark"] >= lo) & (df["b_G3Dmark"] < hi)).sum()
    print(f"  Tier {name}: {lo:8.1f} – {hi:8.1f}  ({count} laptops)")

print("\nSample rows:")
print(df[["Processor_full", "b_singleScore", "b_multiScore",
          "cpu_tier", "Graphics_name", "b_G3Dmark", "gpu_tier"]].head(10).to_string())

df.to_csv("final1_tiered.csv", index=False)
print(f"\nSaved → final1_tiered.csv  ({len(df)} rows, {len(df.columns)} columns)")