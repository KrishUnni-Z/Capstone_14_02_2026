"""
pipeline.py

System 3 — Main orchestration pipeline.

What I do in this file:
- I support two modes:
    1) batch data-layer generation for System 2
    2) online request preparation for System 1 -> System 3 -> System 2 flow
- I keep our existing System_3 structure
- I align the logic with the newer teammate requirements
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .schema import (
    System1ToSystem3Payload,
    System2OutputPayload,
    System3ToSystem1Payload,
)
from .etl import (
    validate_system1_payload,
    run_etl_from_parsed_request,
    save_system1_request,
    save_system2_input_payload,
    save_system2_output_payload,
    save_dashboard_payload,
    build_dashboard_payload,
    save_dataframe,
)
from .data_loader import run_data_loading_pipeline, load_source_tables
from .features import save_feature_outputs, build_full_feature_dataframe
from .embeddings import add_retrieval_text, build_text_embeddings
from .faiss_index import build_faiss_index, search_similar_records


BASE_DIR = Path(__file__).resolve().parent.parent


def run_batch_data_layer(output_dir: Path | None = None, verbose: bool = True) -> Dict[str, object]:
    """
    I run the batch data-layer pipeline that generates the CSV files System 2 expects.
    """
    if output_dir is None:
        output_dir = BASE_DIR

    load_result = run_data_loading_pipeline(output_dir=output_dir, verbose=verbose)
    feature_outputs = save_feature_outputs(load_result["tables"], output_dir=output_dir, verbose=verbose)

    return {
        "load_result": load_result,
        "feature_outputs": feature_outputs,
    }


def prepare_system2_payload(system1_payload_dict: dict):
    """
    I run the online request-time preparation flow.

    This is used when System 1 sends a parsed request and System 3 needs to:
    - validate the request
    - filter the correct rows
    - build feature and rule snapshots for the requested period
    """
    system1_payload: System1ToSystem3Payload = validate_system1_payload(system1_payload_dict)
    save_system1_request(system1_payload, "latest_system1_request.json")

    # I use the existing ETL flow for request filtering.
    filtered_df, system2_payload = run_etl_from_parsed_request(system1_payload.parsed_request)

    # I also load all source tables so I can generate the complete engineered dataframe.
    source_tables = load_source_tables(verbose=False)
    full_feature_df = build_full_feature_dataframe(source_tables)

    # I filter the engineered dataframe down to the requested period and optional goal ids.
    period_id = system1_payload.parsed_request.period_id
    feature_df = full_feature_df[full_feature_df["period_id"] == period_id].copy()

    if system1_payload.parsed_request.goal_ids:
        feature_df = feature_df[feature_df["goal_id"].isin(system1_payload.parsed_request.goal_ids)].copy()

    feature_df = feature_df.sort_values(["goal_id", "period_id"]).reset_index(drop=True)

    # I create rule rows for this specific period using the same logic as batch generation.
    from .features import _compute_rule_scores_for_snapshot  # local import to avoid circular complexity
    rule_df = _compute_rule_scores_for_snapshot(feature_df).copy()

    # I save snapshots so I can inspect what was prepared.
    save_dataframe(filtered_df, "latest_filtered_snapshot.csv")
    save_dataframe(feature_df, "latest_feature_snapshot.csv")
    save_dataframe(rule_df, "latest_rule_snapshot.csv")
    save_system2_input_payload(system2_payload, "latest_system2_input.json")

    return {
        "system1_payload": system1_payload,
        "filtered_df": filtered_df,
        "feature_df": feature_df,
        "rule_df": rule_df,
        "system2_payload": system2_payload,
    }


def prepare_retrieval_artifacts(df: pd.DataFrame):
    """
    I prepare retrieval text, embeddings, and FAISS index for optional semantic search.
    """
    retrieval_df = add_retrieval_text(df)
    embeddings = build_text_embeddings(retrieval_df["retrieval_text"].tolist())
    index = build_faiss_index(embeddings)

    return {
        "retrieval_df": retrieval_df,
        "embeddings": embeddings,
        "faiss_index": index,
    }


def run_similarity_search(df: pd.DataFrame, query: str, top_k: int = 5) -> pd.DataFrame:
    """
    I run an optional semantic search over prepared goal rows.
    """
    artifacts = prepare_retrieval_artifacts(df)
    results = search_similar_records(
        query=query,
        df=artifacts["retrieval_df"],
        index=artifacts["faiss_index"],
        top_k=top_k,
    )
    return results


def finalize_dashboard_output(
    original_df: pd.DataFrame,
    system2_output_payload: System2OutputPayload
) -> System3ToSystem1Payload:
    """
    I merge System 2 output with original goal context and produce dashboard-ready output.
    """
    save_system2_output_payload(system2_output_payload, "latest_system2_output.json")
    dashboard_payload = build_dashboard_payload(original_df, system2_output_payload)
    save_dashboard_payload(dashboard_payload, "latest_dashboard_output.json")
    return dashboard_payload


if __name__ == "__main__":
    print("=" * 70)
    print("SYSTEM 3 PIPELINE")
    print("=" * 70)

    print("\n[1] Running batch data-layer generation...")
    batch_outputs = run_batch_data_layer(verbose=True)

    print("\n[2] Running request-time payload preparation...")
    example_request = {
        "raw_user_input": "Show me goal scores for period 12",
        "parsed_request": {
            "request_type": "score_goals",
            "period_id": 12,
            "goal_ids": None,
            "top_k": None,
            "user_query": "Show me goal scores for period 12",
        },
    }

    outputs = prepare_system2_payload(example_request)

    print("\nPrepared online payload:")
    print(f"  Filtered dataframe shape : {outputs['filtered_df'].shape}")
    print(f"  Feature dataframe shape  : {outputs['feature_df'].shape}")
    print(f"  Rule dataframe shape     : {outputs['rule_df'].shape}")
    print(f"  Total goals              : {outputs['system2_payload'].total_goals}")
