"""
embeddings.py

I use this file for retrieval and dependency-support logic.

Two responsibilities live here:
1. Semantic text embedding helpers for optional FAISS search.
2. Dependency inference helpers adapted from the brain team's infer_dependencies.py.

The dependency output is:
    goal_dependencies.csv

This file uses Bedrock only when infer_goal_dependencies() is called.
"""

from typing import List
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

# sentence_transformers is optional — only needed for semantic search / FAISS.
# Dependency inference (infer_goal_dependencies) uses Bedrock only and works
# without it. The package is loaded lazily inside the functions that need it.
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ANCHOR_PERIOD = 18
MAX_DEPS_PER_GOAL = 3


def add_retrieval_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    I build one text string per goal row for semantic search.
    """
    df = df.copy()

    def build_text(row) -> str:
        parts = [
            f"Goal ID {row.get('goal_id')}",
            f"Goal {row.get('goal_name')}",
            f"Metric {row.get('metric_name')}",
            f"Bucket {row.get('bucket_name')}",
            f"Status band {row.get('status_band')}",
            f"Scenario {row.get('scenario_story')}",
            f"Observed value {row.get('observed_value')}",
            f"Target value {row.get('target_value_final_period')}",
        ]
        return ". ".join([p for p in parts if p and "None" not in p])

    df["retrieval_text"] = df.apply(build_text, axis=1)
    return df


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    I load the embedding model. Requires sentence_transformers.
    """
    if not _ST_AVAILABLE:
        raise ImportError("sentence_transformers not installed. pip install sentence-transformers")
    return _SentenceTransformer(model_name)


def build_text_embeddings(texts: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """
    I convert text into embeddings.
    """
    model = load_embedding_model(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_query(query: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """
    I embed one user query.
    """
    model = load_embedding_model(model_name)
    return model.encode([query], convert_to_numpy=True)


def build_dependency_context(goals: pd.DataFrame, buckets: pd.DataFrame, derived: pd.DataFrame):
    """
    I prepare goal + hierarchy + p18 attainability context for dependency inference.
    """
    l1 = buckets[buckets["bucket_level"] == 1][["bucket_id", "bucket_name"]].rename(
        columns={"bucket_id": "l1_id", "bucket_name": "l1_name"}
    )
    l2 = buckets[buckets["bucket_level"] == 2][["bucket_id", "bucket_name", "parent_bucket_id"]].rename(
        columns={"bucket_id": "l2_id", "bucket_name": "l2_name", "parent_bucket_id": "l1_id"}
    )
    l3 = buckets[buckets["bucket_level"] == 3][["bucket_id", "bucket_name", "parent_bucket_id"]].rename(
        columns={"bucket_id": "l3_id", "bucket_name": "l3_name", "parent_bucket_id": "l2_id"}
    )

    hierarchy = l3.merge(l2, on="l2_id").merge(l1, on="l1_id")

    goals_context = goals.merge(
        hierarchy[["l3_id", "l3_name", "l2_name", "l1_name"]],
        left_on="bucket_id",
        right_on="l3_id",
        how="left",
    )

    attainability = derived[derived["period_id"] == ANCHOR_PERIOD][
        ["goal_id", "probability_of_hitting_target"]
    ].copy()

    if len(attainability) == 0:
        attainability = derived[derived["period_id"] == 12][
            ["goal_id", "probability_of_hitting_target"]
        ].copy()

    return goals_context, attainability


def build_dependency_prompt(goals_context: pd.DataFrame) -> str:
    """
    I build the prompt used to infer causal dependencies between goals.
    """
    lines = []

    for _, g in goals_context.iterrows():
        lines.append(
            f"Goal {int(g['goal_id'])}: {g.get('metric_name')} ({g.get('metric_unit')})\n"
            f"  Bucket path: {g.get('l1_name')} > {g.get('l2_name')} > {g.get('l3_name')}\n"
            f"  Trajectory: {g.get('initial_value')} -> {g.get('target_value_final_period')} by period 24\n"
            f"  Scenario: {g.get('scenario_story')}"
        )

    goal_block = "\n\n".join(lines)

    return f"""
You are mapping causal dependencies across 35 organisational goals.

A goal depends on another goal if its success requires the other goal to be on track first.

For every goal, return 0 to {MAX_DEPS_PER_GOAL} upstream goal dependencies.
Use integer goal_id values only.

Return ONLY valid JSON in this exact shape:

{{
  "dependencies": [
    {{"goal_id": 1, "depends_on_ids": [7, 12], "rationale": "under 25 words"}},
    {{"goal_id": 2, "depends_on_ids": [], "rationale": "under 25 words"}}
  ]
}}

Goals:

{goal_block}
""".strip()


def validate_dependency_graph(parsed: dict, valid_goal_ids: set[int]):
    """
    I clean the LLM dependency output so it is safe to use.
    """
    clean = []
    seen = set()

    for entry in parsed.get("dependencies", []):
        try:
            gid = int(entry["goal_id"])
        except Exception:
            continue

        if gid not in valid_goal_ids or gid in seen:
            continue

        seen.add(gid)

        deps = []
        for d in entry.get("depends_on_ids", []) or []:
            try:
                d = int(d)
            except Exception:
                continue

            if d != gid and d in valid_goal_ids and d not in deps:
                deps.append(d)

        deps = deps[:MAX_DEPS_PER_GOAL]

        clean.append({
            "goal_id": gid,
            "depends_on_ids": deps,
            "rationale": str(entry.get("rationale", ""))[:240],
        })

    for gid in sorted(valid_goal_ids):
        if gid not in seen:
            clean.append({
                "goal_id": gid,
                "depends_on_ids": [],
                "rationale": "defaulted because LLM did not return this goal",
            })

    return clean


def derive_dependency_columns(graph_rows: list[dict], attainability_df: pd.DataFrame) -> pd.DataFrame:
    """
    I convert dependency lists into columns consumed by feature engineering.
    """
    inverse = {row["goal_id"]: [] for row in graph_rows}

    for row in graph_rows:
        for upstream in row["depends_on_ids"]:
            if upstream in inverse:
                inverse[upstream].append(row["goal_id"])

    attain_lookup = dict(
        zip(attainability_df["goal_id"], attainability_df["probability_of_hitting_target"])
    )

    output_rows = []

    for row in graph_rows:
        gid = row["goal_id"]
        deps = row["depends_on_ids"]
        dependents = sorted(inverse.get(gid, []))

        n_deps = len(deps)

        if n_deps >= 3:
            risk = "high"
        elif n_deps == 2:
            risk = "medium"
        elif n_deps == 1:
            risk = "low"
        else:
            risk = "none"

        if deps:
            vals = [attain_lookup.get(d, np.nan) for d in deps]
            vals = [v for v in vals if pd.notna(v)]
            dep_avg_attain = float(np.mean(vals)) if vals else -1.0
        else:
            dep_avg_attain = -1.0

        output_rows.append({
            "goal_id": gid,
            "n_dependencies": n_deps,
            "n_dependents": len(dependents),
            "dependency_risk": risk,
            "depends_on_ids": json.dumps(deps),
            "depended_on_by_ids": json.dumps(dependents),
            "dep_avg_attain": round(dep_avg_attain, 4),
            "rationale": row["rationale"],
        })

    return pd.DataFrame(output_rows).sort_values("goal_id").reset_index(drop=True)


def infer_goal_dependencies(
    goals: pd.DataFrame,
    buckets: pd.DataFrame,
    derived: pd.DataFrame,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    I infer goal dependencies using Bedrock.

    If goal_dependencies.csv already exists and force=False,
    I simply load and return it.
    """
    output_path = BASE_DIR / "goal_dependencies.csv"

    if output_path.exists() and not force:
        if verbose:
            print("OK  goal_dependencies.csv already exists, so I am reusing it.")
        return pd.read_csv(output_path)

    try:
        from bedrock_client import call_model, extract_json
    except ImportError as e:
        raise ImportError(
            "bedrock_client.py must be available in the project root or Python path "
            "to infer dependencies."
        ) from e

    goals_context, attainability = build_dependency_context(goals, buckets, derived)
    prompt = build_dependency_prompt(goals_context)
    valid_ids = set(goals_context["goal_id"].astype(int))

    if verbose:
        print("\nInferring dependencies using Bedrock...")
        print(f"Prompt characters: {len(prompt)}")

    raw = call_model(
        "dependency_inferrer",
        prompt,
        max_tokens=6000,
        temperature=0.2,
    )

    parsed = extract_json(raw, required_keys=["dependencies"])
    graph_rows = validate_dependency_graph(parsed, valid_ids)
    dep_df = derive_dependency_columns(graph_rows, attainability)

    dep_df.to_csv(output_path, index=False)

    if verbose:
        print(f"OK  Saved goal_dependencies.csv ({len(dep_df)} rows)")

    return dep_df
