"""
01_load_data.py
Load analytical_flat.csv, extract period 12 snapshot.

Output: period_12_poc.csv
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("DECIDR SYSTEM 2 — PoC")
print("01  Load data")
print("=" * 70)

df = pd.read_csv("analytical_flat.csv")

print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Unique goals  : {df['goal_id'].nunique()}")
print(f"Unique periods: {df['period_id'].nunique()}")

print(f"\nTarget — probability_of_hitting_target:")
print(f"  Range : {df['probability_of_hitting_target'].min():.2f} – "
      f"{df['probability_of_hitting_target'].max():.2f}")
print(f"  Mean  : {df['probability_of_hitting_target'].mean():.2f}")

# Period 12 snapshot (midpoint, most representative)
period_12 = df[df['period_id'] == 12].copy()
print(f"\nPeriod 12 rows: {len(period_12)}  (1 per goal)")

# Keep binary column for reference, but regression target is the continuous prob
threshold = 0.5
period_12['achieved'] = (period_12['probability_of_hitting_target'] >= threshold).astype(int)
print(f"\nBinary split at threshold={threshold}:")
print(f"  Achieved (≥{threshold}) : {period_12['achieved'].sum()} goals")
print(f"  Not achieved (<{threshold}): {(1 - period_12['achieved']).sum()} goals")
print(f"  NOTE: System 2 uses continuous target, not binary — see 04.")

period_12.to_csv("period_12_poc.csv", index=False)
print("\n✓ Saved period_12_poc.csv")
print("=" * 70)
