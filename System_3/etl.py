from pathlib import Path
import json
from typing import Optional, Tuple

import pandas as pd

from .preprocess import preprocess_all, get_analytical_flat, get_period_snapshot
from .schema import (
    ParsedRequest,
    System1ToSystem3Payload,
    GoalRecord,
    System2InputPayload,
    System2OutputPayload,
    DashboardGoalResult,
    System3ToSystem1Payload,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
JSON_DIR = PROCESSED_DIR / "json"


# -----------------------------
# Helpers
# -----------------------------

def ensure_output_dirs() -> None:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    (JSON_DIR / "system1_requests").mkdir(parents=True, exist_ok=True)
    (JSON_DIR / "system2_inputs").mkdir(parents=True, exist_ok=True)
    (JSON_DIR / "system2_outputs").mkdir(parents=True, exist_ok=True)
    (JSON_DIR / "dashboard_outputs").mkdir(parents=True, exist_ok=True)


def dataframe_to_goal_records(df: pd.DataFrame) -> list[GoalRecord]:
    """
    Convert dataframe rows into validated GoalRecord objects.
    """
    records: list[GoalRecord] = []

    for row in df.to_dict(orient="records"):
        clean_row = {k: v for k, v in row.items() if pd.notna(v)}
        records.append(GoalRecord(**clean_row))

    return records


def filter_goals(df: pd.DataFrame, goal_ids: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Filter by goal_ids if provided.
    """
    if goal_ids:
        return df[df["goal_id"].isin(goal_ids)].copy()
    return df.copy()


def sort_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a stable ordering for downstream systems / dashboard.
    """
    sort_cols = [col for col in ["goal_id", "period_id"] if col in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)
    return df.reset_index(drop=True)


# -----------------------------
# System 1 -> System 3
# -----------------------------

def validate_system1_payload(payload_dict: dict) -> System1ToSystem3Payload:
    """
    Validate payload arriving from System 1.
    """
    return System1ToSystem3Payload(**payload_dict)


def save_system1_request(payload: System1ToSystem3Payload, filename: str) -> Path:
    """
    Save validated request from System 1.
    """
    ensure_output_dirs()
    output_path = JSON_DIR / "system1_requests" / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload.model_dump(mode="json"), f, indent=2, default=str)

    return output_path


# -----------------------------
# Build System 2 input
# -----------------------------

def build_system2_input_payload(
    df: pd.DataFrame,
    period_id: int
) -> System2InputPayload:
    """
    Build validated System 2 scoring payload.
    """
    goal_records = dataframe_to_goal_records(df)

    return System2InputPayload(
        period_id=period_id,
        total_goals=len(goal_records),
        goals=goal_records
    )


def run_etl_from_parsed_request(
    parsed_request: ParsedRequest
) -> Tuple[pd.DataFrame, System2InputPayload]:
    """
    Main ETL flow for System 3 backend.

    Steps:
    1. preprocess raw files
    2. get analytical_flat
    3. extract requested period
    4. optionally filter specific goals
    5. build validated payload for System 2
    """
    cleaned_tables = preprocess_all()
    analytical_flat_df = get_analytical_flat(cleaned_tables)

    period_df = get_period_snapshot(analytical_flat_df, parsed_request.period_id)
    filtered_df = filter_goals(period_df, parsed_request.goal_ids)
    filtered_df = sort_goals(filtered_df)

    payload = build_system2_input_payload(
        filtered_df,
        period_id=parsed_request.period_id
    )

    return filtered_df, payload


def save_system2_input_payload(payload: System2InputPayload, filename: str) -> Path:
    """
    Save payload that will be sent to System 2.
    """
    ensure_output_dirs()
    output_path = JSON_DIR / "system2_inputs" / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload.model_dump(mode="json"), f, indent=2, default=str)

    return output_path


# -----------------------------
# Load System 2 output
# -----------------------------

def load_system2_output_json(path: Path) -> System2OutputPayload:
    """
    Load and validate scored output from System 2.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return System2OutputPayload(**data)


def save_system2_output_payload(payload: System2OutputPayload, filename: str) -> Path:
    """
    Save validated System 2 output payload.
    """
    ensure_output_dirs()
    output_path = JSON_DIR / "system2_outputs" / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload.model_dump(mode="json"), f, indent=2, default=str)

    return output_path


# -----------------------------
# Build dashboard output for System 1
# -----------------------------

def build_dashboard_payload(
    original_df: pd.DataFrame,
    scored_payload: System2OutputPayload
) -> System3ToSystem1Payload:
    """
    Merge original goal/context data with scored results from System 2
    and prepare clean dashboard-ready output for System 1.
    """
    scores_df = pd.DataFrame([score.model_dump() for score in scored_payload.scores])

    merged_df = original_df.merge(
        scores_df,
        on=["goal_id", "period_id"],
        how="left"
    )

    dashboard_records: list[DashboardGoalResult] = []
    for row in merged_df.to_dict(orient="records"):
        clean_row = {k: v for k, v in row.items() if pd.notna(v)}
        dashboard_records.append(DashboardGoalResult(**clean_row))

    return System3ToSystem1Payload(
        period_id=scored_payload.period_id,
        total_goals=len(dashboard_records),
        dashboard_data=dashboard_records,
        metadata={
            "source": "System_3",
            "status": "ready_for_dashboard"
        }
    )


def save_dashboard_payload(payload: System3ToSystem1Payload, filename: str) -> Path:
    """
    Save dashboard-ready payload for System 1.
    """
    ensure_output_dirs()
    output_path = JSON_DIR / "dashboard_outputs" / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload.model_dump(mode="json"), f, indent=2, default=str)

    return output_path


# -----------------------------
# Optional dataframe save helper
# -----------------------------

def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """
    Save dataframe to processed folder.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


# -----------------------------
# Example run
# -----------------------------

if __name__ == "__main__":
    example_payload = {
        "raw_user_input": "Show me goal scores for period 12",
        "parsed_request": {
            "request_type": "score_goals",
            "period_id": 12,
            "goal_ids": None,
            "top_k": None,
            "user_query": "Show me goal scores for period 12"
        }
    }

    system1_payload = validate_system1_payload(example_payload)
    save_system1_request(system1_payload, "example_system1_request.json")

    filtered_df, system2_payload = run_etl_from_parsed_request(
        system1_payload.parsed_request
    )

    save_dataframe(filtered_df, "period_12_filtered_snapshot.csv")
    save_system2_input_payload(system2_payload, "example_system2_input.json")

    print(f"Filtered dataframe shape: {filtered_df.shape}")
    print(f"System 2 payload built for {system2_payload.total_goals} goals.")
