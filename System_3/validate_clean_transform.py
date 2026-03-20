"""
validate_clean_transform.py

System 3 — Validate, clean, structure, and export JSON files for System 1.

Purpose:
- Read analytical_flat.csv
- Validate required columns
- Clean duplicates, nulls, and data types
- Split data into:
    1. output.json        -> period-level dynamic records
    2. goals_config.json  -> goal-level static configuration

Outputs:
- output.json
- goals_config.json
- cleaned_analytical_flat.csv
"""

import json
import pandas as pd
import numpy as np

print("=" * 70)
print("DECIDR SYSTEM 3")
print("02  Validate, clean, transform, export")
print("=" * 70)

INPUT_FILE = "analytical_flat.csv"
CLEANED_FILE = "cleaned_analytical_flat.csv"
OUTPUT_JSON = "output.json"
GOALS_CONFIG_JSON = "goals_config.json"

# -------------------------------------------------------------------
# Load
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
print(f"\nLoaded file          : {INPUT_FILE}")
print(f"Raw shape            : {df.shape}")

# -------------------------------------------------------------------
# Required columns for System 3
# -------------------------------------------------------------------
required_cols = ["goal_id", "period_id"]
missing_required = [c for c in required_cols if c not in df.columns]

if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

print(f"Required columns OK  : {required_cols}")

# -------------------------------------------------------------------
# Standardize text columns
# -------------------------------------------------------------------
text_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({"nan": None, "": None})

# -------------------------------------------------------------------
# Convert likely numeric columns where present
# -------------------------------------------------------------------
numeric_candidates = [
    "period_id",
    "goal_id",
    "bucket_id",
    "range_position_score",
    "allocated_amount",
    "allocated_time_hours",
    "allocation_percentage_of_parent",
    "allocation_percentage_of_total_bucket",
    "optimal_allocation_min",
    "optimal_allocation_max",
    "target_value",
    "target_value_final_period",
    "observed_value",
    "expected_value",
    "probability_of_hitting_target",
    "allocation_efficiency_ratio",
    "delivered_output_quality_score",
    "delivered_output_quantity",
    "trailing_6_period_slope",
    "variance_from_target",
    "volatility_measure",
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------------------------------------------
# Convert boolean-like flags
# -------------------------------------------------------------------
flag_cols = ["underfunded_flag", "overfunded_flag"]

def to_bool(val):
    if pd.isna(val):
        return False
    if isinstance(val, str):
        val = val.strip().lower()
        if val in {"true", "1", "yes", "y"}:
            return True
        if val in {"false", "0", "no", "n"}:
            return False
    return bool(val)

for col in flag_cols:
    if col in df.columns:
        df[col] = df[col].apply(to_bool)

# -------------------------------------------------------------------
# Validation checks
# -------------------------------------------------------------------
print("\nValidation checks")

dup_count = df.duplicated().sum()
print(f"Duplicate full rows  : {dup_count}")

missing_goal_id = df["goal_id"].isna().sum()
missing_period_id = df["period_id"].isna().sum()
print(f"Missing goal_id      : {missing_goal_id}")
print(f"Missing period_id    : {missing_period_id}")

if missing_goal_id > 0 or missing_period_id > 0:
    print("WARNING: Some core IDs are missing.")

# -------------------------------------------------------------------
# Cleaning
# -------------------------------------------------------------------
before_rows = len(df)
df = df.drop_duplicates()
after_rows = len(df)
print(f"\nRemoved duplicates   : {before_rows - after_rows}")

# Fill missing values in important numeric columns using median
important_numeric = [
    "allocated_amount",
    "observed_value",
    "expected_value",
    "probability_of_hitting_target",
    "allocation_efficiency_ratio",
    "delivered_output_quality_score",
]

for col in important_numeric:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Standardize status_band if present
if "status_band" in df.columns:
    df["status_band"] = (
        df["status_band"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"nan": None})
    )

# Drop rows missing core keys after cleaning
df = df.dropna(subset=["goal_id", "period_id"])

# -------------------------------------------------------------------
# Save cleaned CSV for audit/demo
# -------------------------------------------------------------------
df.to_csv(CLEANED_FILE, index=False)
print(f"Saved cleaned file   : {CLEANED_FILE}")

# -------------------------------------------------------------------
# Build output.json (period-level dynamic records)
# -------------------------------------------------------------------
period_record_cols = [
    "goal_id",
    "period_id",
    "goal_name",
    "bucket_id",
    "status_band",
    "range_position_score",
    "underfunded_flag",
    "overfunded_flag",
    "allocated_amount",
    "allocated_time_hours",
    "allocation_percentage_of_parent",
    "allocation_percentage_of_total_bucket",
    "observed_value",
    "expected_value",
    "probability_of_hitting_target",
    "allocation_efficiency_ratio",
    "delivered_output_quality_score",
    "delivered_output_quantity",
    "trailing_6_period_slope",
    "variance_from_target",
    "volatility_measure",
]

period_record_cols = [c for c in period_record_cols if c in df.columns]
output_df = df[period_record_cols].copy()

# -------------------------------------------------------------------
# Build goals_config.json (goal-level static configuration)
# -------------------------------------------------------------------
goal_config_cols = [
    "goal_id",
    "goal_name",
    "bucket_id",
    "target_value",
    "target_value_final_period",
    "optimal_allocation_min",
    "optimal_allocation_max",
]

goal_config_cols = [c for c in goal_config_cols if c in df.columns]

goals_config_df = (
    df[goal_config_cols]
    .drop_duplicates(subset=["goal_id"])
    .copy()
)

# -------------------------------------------------------------------
# Helper for JSON-safe conversion
# -------------------------------------------------------------------
def convert_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def records_for_json(frame):
    records = frame.to_dict(orient="records")
    cleaned_records = []
    for row in records:
        cleaned_records.append({k: convert_value(v) for k, v in row.items()})
    return cleaned_records

output_records = records_for_json(output_df)
goals_config_records = records_for_json(goals_config_df)

# -------------------------------------------------------------------
# Export JSON
# -------------------------------------------------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_records, f, indent=2)

with open(GOALS_CONFIG_JSON, "w", encoding="utf-8") as f:
    json.dump(goals_config_records, f, indent=2)

print(f"\nPeriod records       : {len(output_records)}")
print(f"Goal config records  : {len(goals_config_records)}")

print(f"\n Saved {OUTPUT_JSON}")
print(f" Saved {GOALS_CONFIG_JSON}")
print("\nThis is aligned with System 3:")
print("- load raw data")
print("- validate and clean it")
print("- structure it for downstream systems")
print("- export JSON for System 1")

print("=" * 70)
