"""
data_loader.py

System 3 — Load and inspect raw data from analytical_flat.csv.

Purpose:
- Read the raw flat table into Python
- Inspect its basic structure
- Check core identifiers and column availability
- Save a simple snapshot for quick inspection if needed

Outputs:
- period_12_snapshot.csv (optional inspection snapshot)
"""

import pandas as pd

print("=" * 70)
print("DECIDR SYSTEM 3")
print("01  Load and inspect raw data")
print("=" * 70)

INPUT_FILE = "analytical_flat.csv"
SNAPSHOT_FILE = "period_12_snapshot.csv"

# -------------------------------------------------------------------
# Load raw data
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded file          : {INPUT_FILE}")
print(f"Total rows           : {len(df)}")
print(f"Total columns        : {len(df.columns)}")

# -------------------------------------------------------------------
# Basic schema / ID inspection
# -------------------------------------------------------------------
print("\nCore ID checks")
if "goal_id" in df.columns:
    print(f"Unique goals         : {df['goal_id'].nunique()}")
else:
    print("goal_id column       : MISSING")

if "period_id" in df.columns:
    print(f"Unique periods       : {df['period_id'].nunique()}")
else:
    print("period_id column     : MISSING")

if "bucket_id" in df.columns:
    print(f"Unique buckets       : {df['bucket_id'].nunique()}")
else:
    print("bucket_id column     : not present")

# -------------------------------------------------------------------
# Missing values overview
# -------------------------------------------------------------------
missing_summary = df.isna().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

print("\nMissing values summary")
if missing_summary.empty:
    print("No missing values found.")
else:
    print(missing_summary.head(15))

# -------------------------------------------------------------------
# Duplicate row check
# -------------------------------------------------------------------
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate full rows  : {duplicate_count}")

# -------------------------------------------------------------------
# Column preview
# -------------------------------------------------------------------
print("\nFirst 20 columns")
print(list(df.columns[:20]))

# -------------------------------------------------------------------
# Optional inspection snapshot
# This is just for checking one representative period quickly
# -------------------------------------------------------------------
if "period_id" in df.columns:
    period_12 = df[df["period_id"] == 12].copy()
    print(f"\nPeriod 12 rows       : {len(period_12)}")

    if len(period_12) > 0:
        period_12.to_csv(SNAPSHOT_FILE, index=False)
        print(f"Saved snapshot       : {SNAPSHOT_FILE}")
    else:
        print("No rows found for period 12.")
else:
    print("\nCould not create snapshot because period_id is missing.")

print("\n Load + inspection complete")
print("=" * 70)
