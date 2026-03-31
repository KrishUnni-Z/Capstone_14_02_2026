# System_3/preprocess.py

from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def _standardize_columns(df: pd.DataFrame):
    """Make column names lowercase and clean."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def _parse_dates(df: pd.DataFrame):
    """Parse date columns if they exist."""
    df = df.copy()
    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def _convert_types(df):
    df = df.copy()

    text_cols = {
        "name",
        "goal_name",
        "bucket_name",
        "parent_bucket_name",
        "metric_name",
        "metric_unit",
        "scenario_story",
        "status_band"
    }

    bool_cols = {
        "underfunded_flag",
        "overfunded_flag",
        "is_leaf"
    }

    id_cols = {
        "period_id", "projection_id", "bucket_id", "goal_id",
        "allocation_id", "output_id", "metric_id", "derived_id",
        "parent_bucket_id", "start_period", "end_period"
    }

    date_cols = {"start_date", "end_date"}

    for col in df.columns:

        col_lower = col.lower()

        if col_lower in text_cols:
            df[col] = df[col].astype("string")

        elif col_lower in bool_cols:
            df[col] = df[col].astype("boolean")

        elif col_lower in id_cols or col_lower.endswith("_id"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        elif col_lower in date_cols:
            pass
    
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _drop_duplicates(df: pd.DataFrame):
    """Remove duplicate rows."""
    return df.drop_duplicates().copy()


def _basic_missing_value_handling(df: pd.DataFrame):
    """
    Very safe missing-value handling.
    Just fix obvious issues.
    """
    df = df.copy()

    # fill missing text fields with NA placeholder only if needed
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].replace("", np.nan)

    return df


def clean_dataframe(df: pd.DataFrame):
    """Apply the common cleaning pipeline."""
    df = _standardize_columns(df)
    df = _parse_dates(df)
    df = _convert_types(df)
    df = _drop_duplicates(df)
    df = _basic_missing_value_handling(df)
    return df


def load_raw_tables(raw_dir: Path = RAW_DIR):
    """Load all CSV files from data/raw."""
    tables = {}

    for csv_file in raw_dir.glob("*.csv"):
        table_name = csv_file.stem.lower()
        df = pd.read_csv(csv_file)
        tables[table_name] = df

    return tables


def preprocess_all(raw_dir: Path = RAW_DIR):
    """Load and clean all raw tables."""
    raw_tables = load_raw_tables(raw_dir)
    cleaned_tables = {}

    for name, df in raw_tables.items():
        cleaned_tables[name] = clean_dataframe(df)

    return cleaned_tables


def save_processed_tables(
    cleaned_tables: dict[str, pd.DataFrame],
    processed_dir: Path = PROCESSED_DIR
):
    """Save cleaned tables to data/processed."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, df in cleaned_tables.items():
        output_path = processed_dir / f"{name}_clean.csv"
        df.to_csv(output_path, index=False)


def get_analytical_flat(cleaned_tables: dict[str, pd.DataFrame]):
    """Return cleaned analytical_flat table."""
    if "analytical_flat" not in cleaned_tables:
        raise ValueError("analytical_flat.csv not found in data/raw")
    return cleaned_tables["analytical_flat"].copy()


def get_period_snapshot(df: pd.DataFrame, period_id: int):
    """Extract one period snapshot, e.g. period 12."""
    if "period_id" not in df.columns:
        raise ValueError("period_id column not found in dataframe")
    return df[df["period_id"] == period_id].copy()


if __name__ == "__main__":
    cleaned = preprocess_all()
    save_processed_tables(cleaned)

    analytical_flat_df = get_analytical_flat(cleaned)
    period_12_df = get_period_snapshot(analytical_flat_df, period_id=12)

    print("Cleaned tables:", list(cleaned.keys()))
    print("analytical_flat shape:", analytical_flat_df.shape)
    print("period 12 shape:", period_12_df.shape)

