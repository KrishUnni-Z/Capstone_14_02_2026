"""
data_loader.py

I use this file for the loading part of System 3.

What I do here:
- I check that all 8 required CSV files exist.
- I load all CSVs safely, even if encoding is not UTF-8.
- I validate basic consistency.
- I save:
    analytical_full.csv
    period_12_poc.csv
    period_18_poc.csv
"""

from pathlib import Path
from typing import Dict

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

REQUIRED_FILES = {
    "analytical_flat.csv": "Main analytical dataset",
    "buckets.csv": "Bucket hierarchy",
    "goals.csv": "Goal definitions",
    "allocations.csv": "Budget allocations",
    "outputs.csv": "Delivered outputs",
    "metrics.csv": "Observed metrics",
    "derived_fields.csv": "Derived scoring fields",
    "periods.csv": "Period definitions",
}


def resolve_file(filename: str) -> Path:
    """
    I look for the file in the project root.
    """
    return BASE_DIR / filename


def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    I try different encodings because some CSV files may not be UTF-8.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


def check_required_files(verbose: bool = True) -> Dict[str, Path]:
    """
    I check whether all required source files exist.
    """
    paths = {}
    missing = []

    if verbose:
        print("=" * 70)
        print("SYSTEM 3 — LOAD DATA")
        print("=" * 70)

    for filename, desc in REQUIRED_FILES.items():
        path = resolve_file(filename)

        if path.exists():
            paths[filename] = path
            status = "OK"
            size = f"{path.stat().st_size / 1024:.1f}KB"
        else:
            status = "MISSING"
            size = ""
            missing.append(filename)

        if verbose:
            print(f"{status:<8} {filename:<25} {desc:<30} {size}")

    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    return paths


def load_source_tables(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    I load all source CSVs into dataframes.
    """
    paths = check_required_files(verbose=verbose)

    tables = {
        "analytical_flat": read_csv_safely(paths["analytical_flat.csv"]),
        "buckets": read_csv_safely(paths["buckets.csv"]),
        "goals": read_csv_safely(paths["goals.csv"]),
        "allocations": read_csv_safely(paths["allocations.csv"]),
        "outputs": read_csv_safely(paths["outputs.csv"]),
        "metrics": read_csv_safely(paths["metrics.csv"]),
        "derived_fields": read_csv_safely(paths["derived_fields.csv"]),
        "periods": read_csv_safely(paths["periods.csv"]),
    }

    if verbose:
        print("\nLoaded tables:")
        for name, df in tables.items():
            print(f"  {name:<18} {df.shape}")

    return tables


def validate_source_tables(tables: Dict[str, pd.DataFrame], verbose: bool = True) -> Dict[str, bool]:
    """
    I validate whether the tables are consistent with each other.
    """
    flat = tables["analytical_flat"]
    buckets = tables["buckets"]
    goals = tables["goals"]
    allocations = tables["allocations"]
    outputs = tables["outputs"]
    metrics = tables["metrics"]
    derived = tables["derived_fields"]
    periods = tables["periods"]

    required_flat_cols = ["goal_id", "period_id", "probability_of_hitting_target"]
    missing_flat_cols = [col for col in required_flat_cols if col not in flat.columns]

    if missing_flat_cols:
        raise ValueError(
            f"analytical_flat.csv is missing required columns: {missing_flat_cols}. "
            "This may mean the wrong file was copied into the project folder."
        )

    n_goals = flat["goal_id"].nunique()
    n_periods = flat["period_id"].nunique()
    expected_rows = n_goals * n_periods

    checks = {
        "flat_has_expected_rows": len(flat) == expected_rows,
        "bucket_levels_are_1_2_3": "bucket_level" in buckets.columns and set(buckets["bucket_level"].unique()) == {1, 2, 3},
        "goals_match_flat": "goal_id" in goals.columns and goals["goal_id"].nunique() == n_goals,
        "allocations_cover_all_periods": "period_id" in allocations.columns and allocations["period_id"].nunique() == n_periods,
        "outputs_cover_all_periods": "period_id" in outputs.columns and outputs["period_id"].nunique() == n_periods,
        "metrics_cover_goal_periods": len(metrics) == expected_rows,
        "derived_cover_goal_periods": len(derived) == expected_rows,
        "period_count_matches": len(periods) == n_periods,
        "l3_buckets_match_goals": "bucket_level" in buckets.columns and buckets[buckets["bucket_level"] == 3].shape[0] == n_goals,
        "weighted_goal_status_score_exists": "weighted_goal_status_score" in derived.columns,
        "allocation_fitness_score_exists": "allocation_fitness_score" in derived.columns,
        "time_to_green_estimate_exists": "time_to_green_estimate" in derived.columns,
    }

    if verbose:
        print("\nValidation checks:")
        for name, ok in checks.items():
            print(f"  {'OK' if ok else 'FAIL':<6} {name}")

    return checks


def save_base_outputs(tables: Dict[str, pd.DataFrame], verbose: bool = True) -> None:
    """
    I save base files required by the brain pipeline.
    """
    flat = tables["analytical_flat"].copy()

    flat.to_csv(BASE_DIR / "analytical_full.csv", index=False)

    for period in [12, 18]:
        snap = flat[flat["period_id"] == period].copy().reset_index(drop=True)

        if "probability_of_hitting_target" in snap.columns:
            snap["achieved"] = (snap["probability_of_hitting_target"] >= 0.5).astype(int)

        snap.to_csv(BASE_DIR / f"period_{period}_poc.csv", index=False)

        if verbose:
            print(f"OK  Saved period_{period}_poc.csv ({len(snap)} rows)")

    if verbose:
        print(f"OK  Saved analytical_full.csv ({len(flat)} rows)")


def run_data_loader(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    I run the full loading stage and return loaded tables.
    """
    tables = load_source_tables(verbose=verbose)
    validate_source_tables(tables, verbose=verbose)
    save_base_outputs(tables, verbose=verbose)
    return tables


if __name__ == "__main__":
    run_data_loader(verbose=True)
