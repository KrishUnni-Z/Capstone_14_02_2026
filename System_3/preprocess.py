"""
preprocess.py

System 3 preprocessing helpers.

What I do in this file:
- I locate analytical_flat.csv
- I load it safely
- I standardize a few obvious column/data issues
- I give the rest of System 3 a clean dataframe to work with
"""

from pathlib import Path
from typing import Dict

import pandas as pd


def _candidate_paths() -> list[Path]:
    """
    I define a few possible places where analytical_flat.csv may exist.
    I do this because folder layouts can differ across branches/machines.
    """
    base_dir = Path(__file__).resolve().parent
    repo_dir = base_dir.parent

    return [
        repo_dir / "data" / "processed" / "analytical_flat.csv",
        repo_dir / "data" / "raw" / "analytical_flat.csv",
        repo_dir / "analytical_flat.csv",
        base_dir / "analytical_flat.csv",
        repo_dir / "project" / "analytical_flat.csv",
    ]


def find_analytical_flat_path() -> Path:
    """
    I search for the analytical_flat.csv file and return the first valid path.
    """
    for path in _candidate_paths():
        if path.exists():
            return path

    raise FileNotFoundError(
        "I could not find analytical_flat.csv. "
        "Please place it in data/processed, data/raw, project/, repo root, or System_3/."
    )


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    I do light cleaning so the downstream pipeline is more stable.
    """
    df = df.copy()

    # I remove duplicate full rows so each record is unique.
    df = df.drop_duplicates()

    # I strip spaces from text columns.
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "": None})

    # I coerce common numeric columns if they exist.
    numeric_candidates = [
        "goal_id",
        "period_id",
        "bucket_id",
        "parent_bucket_id",
        "projection_id",
        "observed_value",
        "expected_value",
        "variance_from_target",
        "trailing_6_period_slope",
        "volatility_measure",
        "allocated_amount",
        "allocated_time_hours",
        "allocation_percentage_of_total",
        "allocation_percentage_of_parent",
        "delivered_output_quantity",
        "delivered_output_quality_score",
        "output_cost_per_unit",
        "total_cost",
        "range_position_score",
        "allocation_efficiency_ratio",
        "probability_of_hitting_target",
        "time_to_green_estimate",
        "minimum_viable_allocation",
        "optimal_allocation_min",
        "optimal_allocation_max",
        "target_value_final_period",
        "initial_value",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # I normalize boolean-like columns.
    for col in ["underfunded_flag", "overfunded_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # I remove rows missing the two most important keys.
    required_cols = [c for c in ["goal_id", "period_id"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)

    return df.reset_index(drop=True)


def preprocess_all() -> Dict[str, pd.DataFrame]:
    """
    I load and return the main cleaned dataset inside a dictionary.

    I keep this function name because your ETL file already expects it.
    """
    analytical_flat_path = find_analytical_flat_path()
    analytical_flat = pd.read_csv(analytical_flat_path)
    analytical_flat = _standardize_dataframe(analytical_flat)

    return {
        "analytical_flat": analytical_flat
    }


def get_analytical_flat(cleaned_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    I return the cleaned analytical flat table.
    """
    if "analytical_flat" not in cleaned_tables:
        raise KeyError("I expected 'analytical_flat' inside cleaned_tables.")
    return cleaned_tables["analytical_flat"].copy()


def get_period_snapshot(df: pd.DataFrame, period_id: int) -> pd.DataFrame:
    """
    I return only the rows for the requested period.
    """
    if "period_id" not in df.columns:
        raise KeyError("I expected a 'period_id' column in the dataframe.")

    snapshot = df[df["period_id"] == period_id].copy()
    return snapshot.reset_index(drop=True)
