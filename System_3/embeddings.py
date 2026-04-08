from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def add_retrieval_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a text representation for each goal row so System 3 can do semantic search later.
    """
    df = df.copy()

    def build_text(row) -> str:
        parts = [
            f"Goal ID {row['goal_id']}" if pd.notna(row.get("goal_id")) else None,
            f"Goal {row['goal_name']}" if pd.notna(row.get("goal_name")) else None,
            f"Metric {row['metric_name']}" if pd.notna(row.get("metric_name")) else None,
            f"Bucket {row['bucket_name']}" if pd.notna(row.get("bucket_name")) else None,
            f"Parent bucket {row['parent_bucket_name']}" if pd.notna(row.get("parent_bucket_name")) else None,
            f"Status band {row['status_band']}" if pd.notna(row.get("status_band")) else None,
            f"Scenario {row['scenario_story']}" if pd.notna(row.get("scenario_story")) else None,
            f"Observed value {row['observed_value']}" if pd.notna(row.get("observed_value")) else None,
            f"Target value {row['target_value_final_period']}" if pd.notna(row.get("target_value_final_period")) else None,
            f"Allocation amount {row['allocated_amount']}" if pd.notna(row.get("allocated_amount")) else None,
            f"Probability of hitting target {row['probability_of_hitting_target']}" if pd.notna(row.get("probability_of_hitting_target")) else None,
        ]
        return ". ".join([p for p in parts if p])

    df["retrieval_text"] = df.apply(build_text, axis=1)
    return df


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load sentence-transformer model.
    """
    return SentenceTransformer(model_name)


def build_text_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """
    Convert a list of texts into embeddings.
    """
    model = load_embedding_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def embed_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """
    Embed a single search query.
    """
    model = load_embedding_model(model_name)
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    return query_embedding
