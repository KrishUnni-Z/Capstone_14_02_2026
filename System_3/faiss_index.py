import numpy as np
import pandas as pd
import faiss

from .embeddings import embed_query


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS L2 index from embeddings.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    top_k: int = 5
):
    """
    Search FAISS index using a query embedding.
    """
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)

    distances, indices = index.search(query_embedding, top_k)
    return distances, indices


def search_similar_records(
    query: str,
    df: pd.DataFrame,
    index: faiss.Index,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Search for most similar rows in df using FAISS.
    Requires df to already align with the embeddings used to build the index.
    """
    query_embedding = embed_query(query)
    distances, indices = search_index(index, query_embedding, top_k=top_k)

    result_df = df.iloc[indices[0]].copy()
    result_df["faiss_distance"] = distances[0]
    return result_df.reset_index(drop=True)
