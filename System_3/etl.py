from pathlib import Path
import json
from typing import Optional

import pandas as pd

from .preprocess import preprocess_all, get_analytical_flat, get_period_snapshot
from .schema import GoalRecord, System2InputPayload, System2OutputPayload


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
JSON_DIR = PROCESSED_DIR / "json"


def dataframe_to_goal_records(df: pd.DataFrame) -> list[GoalRecord]:
    """
    Convert a dataframe into validated GoalRecord objects.
    """
    records: list[GoalRecord] = []

    for row in df.to_dict(orient="records"):
        clean_row = {
            key: value
            for key, value in row.items()
            if pd.notna(value)
        }
        record = GoalRecord(**clean_row)
        records.append(record)

    return records


def build_system2_input_payload(
    df: pd.DataFrame,
    period_id: int
) -> System2InputPayload:
    """
    Build validated System 2 payload from a period snapshot dataframe.
    """
    goal_records = dataframe_to_goal_records(df)

    payload = System2InputPayload(
        period_id=period_id,
        total_goals=len(goal_records),
        goals=goal_records
    )
    return payload


def run_etl_for_period(period_id: int = 12) -> tuple[pd.DataFrame, System2InputPayload]:
    """
    Full ETL flow:
    - preprocess raw files
    - get analytical_flat
    - extract one period snapshot
    - build System 2 payload
    """
    cleaned_tables = preprocess_all()
    analytical_flat_df = get_analytical_flat(cleaned_tables)
    period_df = get_period_snapshot(analytical_flat_df, period_id=period_id)

    payload = build_system2_input_payload(period_df, period_id=period_id)
    return period_df, payload


def save_payload_to_json(payload: System2InputPayload, filename: str) -> Path:
    """
    Save a System2InputPayload as JSON.
    """
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    output_path = JSON_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload.model_dump(mode="json"), f, indent=2, default=str)

    return output_path


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """
    Save dataframe into processed directory.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def load_system2_output_json(path: Path) -> System2OutputPayload:
    """
    Load scored JSON returned by System 2 and validate it.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return System2OutputPayload(**data)


if __name__ == "__main__":
    period_id = 12

    period_df, payload = run_etl_for_period(period_id=period_id)

    csv_path = save_dataframe(period_df, f"period_{period_id}_snapshot.csv")
    json_path = save_payload_to_json(payload, f"system2_input_period_{period_id}.json")

    print(f"Period snapshot shape: {period_df.shape}")
    print(f"Saved period snapshot to: {csv_path}")
    print(f"Saved System 2 input payload to: {json_path}")
