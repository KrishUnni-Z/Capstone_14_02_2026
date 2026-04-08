from pathlib import Path
from typing import Optional

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
from .features import build_feature_dataframe
from .embeddings import add_retrieval_text, build_text_embeddings
from .faiss_index import build_faiss_index, search_similar_records


def prepare_system2_payload(system1_payload_dict: dict):
    """
    Main backend preparation flow:
    System 1 request -> System 3 ETL -> feature prep -> System 2 payload
    """
    system1_payload: System1ToSystem3Payload = validate_system1_payload(system1_payload_dict)

    save_system1_request(system1_payload, "latest_system1_request.json")

    filtered_df, system2_payload = run_etl_from_parsed_request(system1_payload.parsed_request)

    feature_df = build_feature_dataframe(filtered_df)
    save_dataframe(filtered_df, "latest_filtered_snapshot.csv")
    save_dataframe(feature_df, "latest_feature_snapshot.csv")

    save_system2_input_payload(system2_payload, "latest_system2_input.json")

    return {
        "system1_payload": system1_payload,
        "filtered_df": filtered_df,
        "feature_df": feature_df,
        "system2_payload": system2_payload,
    }


def prepare_retrieval_artifacts(df: pd.DataFrame):
    """
    Optional retrieval prep for System 3:
    builds text representations, embeddings, and FAISS index.
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
    Optional helper for semantic search over goals.
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
    Merge System 2 output with original goal context and produce dashboard payload.
    """
    save_system2_output_payload(system2_output_payload, "latest_system2_output.json")
    dashboard_payload = build_dashboard_payload(original_df, system2_output_payload)
    save_dashboard_payload(dashboard_payload, "latest_dashboard_output.json")
    return dashboard_payload


if __name__ == "__main__":
    example_request = {
        "raw_user_input": "Show me goal scores for period 12",
        "parsed_request": {
            "request_type": "score_goals",
            "period_id": 12,
            "goal_ids": None,
            "top_k": None,
            "user_query": "Show me goal scores for period 12"
        }
    }

    outputs = prepare_system2_payload(example_request)

    print("Filtered dataframe shape:", outputs["filtered_df"].shape)
    print("Feature dataframe shape:", outputs["feature_df"].shape)
    print("System 2 payload goals:", outputs["system2_payload"].total_goals)
