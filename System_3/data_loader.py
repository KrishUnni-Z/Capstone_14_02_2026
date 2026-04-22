"""
data_loader.py

System 3 — Data loading and validation layer.

What I do in this file:
- I check that all required source CSVs exist
- I load them into pandas dataframes
- I run a few basic cross-file validation checks
- I save two useful base outputs:
    1) analytical_full.csv
    2) period_12_poc.csv

This file aligns System 3 with the new Data Layer expectation from the team,
while still fitting our current System_3 folder structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import os
import sys
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

REQUIRED_FILES = {
    "analytical_flat.csv": "Main 840-row analytical dataset",
    "buckets.csv": "Bucket hierarchy (L1/L2/L3)",
    "goals.csv": "Goal definitions and allocation bands",
    "allocations.csv": "Budget allocations per goal per period",
    "outputs.csv": "Delivered outputs per goal per period",
    "metrics.csv": "Observed metric values per goal per period",
    "derived_fields.csv": "Derived scoring fields (status, fitness, etc.)",
    "periods.csv": "Period definitions and dates",
}


def _resolve_file_path(filename: str) -> Path:
    """
    I try a few common locations so the code works even if files are placed
    either in the project root or inside data folders.
    """
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / "raw" / filename,
        BASE_DIR / "data" / "processed" / filename,
        BASE_DIR / "project" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    return BASE_DIR / filename


def check_required_files(verbose: bool = True) -> Dict[str, Path]:
    """
    I make sure all required source CSVs exist before the rest of the pipeline runs.
    """
    resolved_paths: Dict[str, Path] = {}
    missing = []

    if verbose:
        print("=" * 70)
        print("DECIDR COHERENCE ENGINE")
        print("SYSTEM 3 — Load and validate data")
        print("=" * 70)
        print("\nChecking required files...")

    for filename, description in REQUIRED_FILES.items():
        path = _resolve_file_path(filename)
        exists = path.exists()
        size = f"{path.stat().st_size / 1024:.1f}KB" if exists else ""
        status = "OK    " if exists else "MISSING"

        if verbose:
            print(f"  {status}  {filename:<20} {description}  {size}")

        if exists:
            resolved_paths[filename] = path
        else:
            missing.append(filename)

    if missing:
        if verbose:
            print(f"\nERROR: {len(missing)} required file(s) missing:")
            for file_name in missing:
                print(f"  - {file_name}")
        raise FileNotFoundError(
            "Some required source CSV files are missing. "
            "Please place them in the project root or data folders."
        )

    if verbose:
        print(f"\nAll {len(REQUIRED_FILES)} required files are present.")

    return resolved_paths


def load_source_tables(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    I load all required source CSVs and return them as a dictionary of dataframes.
    """
    file_paths = check_required_files(verbose=verbose)

    tables = {
        "analytical_flat": pd.read_csv(file_paths["analytical_flat.csv"]),
        "buckets": pd.read_csv(file_paths["buckets.csv"]),
        "goals": pd.read_csv(file_paths["goals.csv"]),
        "allocations": pd.read_csv(file_paths["allocations.csv"]),
        "outputs": pd.read_csv(file_paths["outputs.csv"]),
        "metrics": pd.read_csv(file_paths["metrics.csv"]),
        "derived_fields": pd.read_csv(file_paths["derived_fields.csv"]),
        "periods": pd.read_csv(file_paths["periods.csv"]),
    }

    if verbose:
        flat = tables["analytical_flat"]
        print("\nLoaded source tables:")
        for name, df in tables.items():
            print(f"  {name:<15} {df.shape}")
        print(f"\nMain analytical table: {flat.shape}")

    return tables


def validate_source_tables(tables: Dict[str, pd.DataFrame], verbose: bool = True) -> Dict[str, bool]:
    """
    I run a few cross-file consistency checks so I know the source data is reliable enough.
    """
    flat = tables["analytical_flat"]
    buckets = tables["buckets"]
    goals = tables["goals"]
    allocations = tables["allocations"]
    outputs = tables["outputs"]
    metrics = tables["metrics"]
    derived = tables["derived_fields"]
    periods = tables["periods"]

    n_goals = flat["goal_id"].nunique() if "goal_id" in flat.columns else 0
    n_periods = flat["period_id"].nunique() if "period_id" in flat.columns else 0
    expected_rows = n_goals * n_periods

    checks = {
        "flat_has_expected_shape": len(flat) == expected_rows,
        "bucket_levels_present": set(buckets["bucket_level"].unique()) == {1, 2, 3},
        "goals_match_flat": goals["goal_id"].nunique() == n_goals,
        "allocations_cover_all_periods": allocations["period_id"].nunique() == n_periods,
        "outputs_cover_all_periods": outputs["period_id"].nunique() == n_periods,
        "metrics_cover_all_goal_periods": len(metrics) == expected_rows,
        "derived_cover_all_goal_periods": len(derived) == expected_rows,
        "period_count_matches": len(periods) == n_periods,
        "l3_bucket_count_matches_goal_count": buckets[buckets["bucket_level"] == 3].shape[0] == n_goals,
        "weighted_goal_status_score_exists": "weighted_goal_status_score" in derived.columns,
        "allocation_fitness_score_exists": "allocation_fitness_score" in derived.columns,
        "time_to_green_estimate_exists": "time_to_green_estimate" in derived.columns,
    }

    if verbose:
        print("\nValidation checks:")
        for check_name, result in checks.items():
            status = "OK" if result else "FAIL"
            print(f"  {status:<5} {check_name}")

    return checks


def save_base_outputs(tables: Dict[str, pd.DataFrame], output_dir: Path | None = None, verbose: bool = True) -> Dict[str, Path]:
    """
    I save the two base outputs that the team currently expects:
    - analytical_full.csv
    - period_12_poc.csv
    """
    if output_dir is None:
        output_dir = BASE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    flat = tables["analytical_flat"].copy()

    analytical_full_path = output_dir / "analytical_full.csv"
    flat.to_csv(analytical_full_path, index=False)

    period_12 = flat[flat["period_id"] == 12].copy().reset_index(drop=True)
    if "probability_of_hitting_target" in period_12.columns:
        period_12["achieved"] = (period_12["probability_of_hitting_target"] >= 0.5).astype(int)

    period_12_path = output_dir / "period_12_poc.csv"
    period_12.to_csv(period_12_path, index=False)

    if verbose:
        print(f"\n✓ Saved analytical_full.csv  ({len(flat)} rows)")
        print(f"✓ Saved period_12_poc.csv    ({len(period_12)} rows)")

    return {
        "analytical_full": analytical_full_path,
        "period_12_poc": period_12_path,
    }


def run_data_loading_pipeline(output_dir: Path | None = None, verbose: bool = True) -> Dict[str, object]:
    """
    I run the complete source loading + validation + base output pipeline.
    """
    tables = load_source_tables(verbose=verbose)
    checks = validate_source_tables(tables, verbose=verbose)
    outputs = save_base_outputs(tables, output_dir=output_dir, verbose=verbose)

    return {
        "tables": tables,
        "checks": checks,
        "outputs": outputs,
    }


if __name__ == "__main__":
    run_data_loading_pipeline()
