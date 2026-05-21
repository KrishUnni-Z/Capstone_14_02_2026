"""
load_data.py  v5
Validates all required source files, loads full 840-row dataset,
extracts p12 AND p18 snapshots.

Required source files:
  analytical_flat.csv, buckets.csv, goals.csv, allocations.csv,
  outputs.csv, metrics.csv, derived_fields.csv, periods.csv

Outputs:
  analytical_full.csv   - all 840 rows for GP training
  period_12_poc.csv     - 35 goals at period 12 (kept for backward compat)
  period_18_poc.csv     - 35 goals at period 18 (the anchor going forward)
"""

import pandas as pd
import numpy as np
import os
import sys


print("=" * 70)
print("DECIDR COHERENCE ENGINE")
print("01  Load and validate data")
print("=" * 70)

REQUIRED_FILES = {
    "analytical_flat.csv": "Main 840-row analytical dataset",
    "buckets.csv"        : "Bucket hierarchy (L1/L2/L3)",
    "goals.csv"          : "Goal definitions and allocation bands",
    "allocations.csv"    : "Budget allocations per goal per period",
    "outputs.csv"        : "Delivered outputs per goal per period",
    "metrics.csv"        : "Observed metric values per goal per period",
    "derived_fields.csv" : "Derived scoring fields (status, fitness, etc.)",
    "periods.csv"        : "Period definitions and dates",
}

print("\nChecking required files...")
missing = []
for fname, description in REQUIRED_FILES.items():
    exists = os.path.exists(fname)
    size   = f"{os.path.getsize(fname)/1024:.1f}KB" if exists else ""
    status = "OK    " if exists else "MISSING"
    print(f"  {status}  {fname:<30} {description}  {size}")
    if not exists:
        missing.append(fname)

if missing:
    print(f"\nERROR: {len(missing)} required file(s) missing:")
    for f in missing:
        print(f"  {f}")
    print("\nAll source files must be in the working directory.")
    sys.exit(1)

print(f"\nAll {len(REQUIRED_FILES)} files present.")

print("\nLoading analytical_flat.csv...")
df = pd.read_csv("analytical_flat.csv")

n_goals   = df['goal_id'].nunique()
n_periods = df['period_id'].nunique()
expected  = n_goals * n_periods

print(f"  Rows     : {len(df)}  (expected {expected} = {n_goals} goals x {n_periods} periods)")
print(f"  Columns  : {len(df.columns)}")
print(f"  Goals    : {n_goals}")
print(f"  Periods  : {n_periods}")

if len(df) != expected:
    print(f"  WARNING: row count mismatch  expected {expected}, got {len(df)}")
else:
    print(f"  Shape    : OK")

print("\nValidating cross-file consistency...")

buckets = pd.read_csv("buckets.csv")
goals   = pd.read_csv("goals.csv")
allocs  = pd.read_csv("allocations.csv")
outputs = pd.read_csv("outputs.csv")
metrics = pd.read_csv("metrics.csv")
derived = pd.read_csv("derived_fields.csv")
periods = pd.read_csv("periods.csv")

checks = {
    "bucket levels present (1,2,3)"   : set(buckets['bucket_level'].unique()) == {1, 2, 3},
    "goals match flat file"            : goals['goal_id'].nunique() == n_goals,
    "allocations cover all periods"    : allocs['period_id'].nunique() == n_periods,
    "outputs cover all periods"        : outputs['period_id'].nunique() == n_periods,
    "metrics cover all goals/periods"  : len(metrics) == n_goals * n_periods,
    "derived cover all goals/periods"  : len(derived) == n_goals * n_periods,
    "periods count matches"            : len(periods) == n_periods,
    "L3 buckets match goal count"      : buckets[buckets['bucket_level'] == 3].shape[0] == n_goals,
    "weighted_goal_status_score exists": 'weighted_goal_status_score' in derived.columns,
    "allocation_fitness_score exists"  : 'allocation_fitness_score' in derived.columns,
    "time_to_green_estimate exists"    : 'time_to_green_estimate' in derived.columns,
}

all_ok = True
for check, result in checks.items():
    status = "OK" if result else "FAIL"
    print(f"  {status:<6} {check}")
    if not result:
        all_ok = False

if not all_ok:
    print("\nWARNING: Some validation checks failed. Results may be unreliable.")
else:
    print("\nAll validation checks passed.")

print("\nGround truth  probability_of_hitting_target:")
print(f"  Range  : {df['probability_of_hitting_target'].min():.3f} - {df['probability_of_hitting_target'].max():.3f}")
print(f"  Mean   : {df['probability_of_hitting_target'].mean():.3f}")
print(f"  Std    : {df['probability_of_hitting_target'].std():.3f}")


def _print_dist(df, period):
    subset = df[df['period_id'] == period]['probability_of_hitting_target']
    print(f"\n  Period {period} distribution:")
    for v, c in subset.value_counts().sort_index().items():
        bar = "#" * c
        pct = c / len(subset) * 100
        print(f"    {v:.1f}: {c:>3} goals ({pct:.0f}%)  {bar}")
    if subset.nunique() <= 3:
        print(f"    NOTE: Only {subset.nunique()} unique values at p{period}.")


_print_dist(df, 12)
_print_dist(df, 18)

df.to_csv("analytical_full.csv", index=False)
print(f"\nOK  Saved analytical_full.csv  ({len(df)} rows)")

threshold = 0.5
for period, out_file in [(12, "period_12_poc.csv"), (18, "period_18_poc.csv")]:
    snap = df[df['period_id'] == period].copy().reset_index(drop=True)
    snap['achieved'] = (snap['probability_of_hitting_target'] >= threshold).astype(int)
    snap.to_csv(out_file, index=False)
    on_track = snap['achieved'].sum()
    at_risk  = (~snap['achieved'].astype(bool)).sum()
    print(f"OK  Saved {out_file}  ({len(snap)} rows  on-track={on_track}  at-risk={at_risk})")

print("=" * 70)
