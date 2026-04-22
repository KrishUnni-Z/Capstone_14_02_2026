"""
features.py

System 3 — Feature engineering layer aligned with System 2 expectations.

What I do in this file:
- I build engineered features from the full source tables
- I generate full 840-row features for training/inference support
- I generate snapshot raw features for periods 6, 12, 18, and 24
- I generate rule score CSVs for those same periods
- I save feature scaler and feature name outputs for compatibility

This file keeps our current System_3 structure but aligns the logic with the
brain teammate's feature engineering flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_PERIODS = [6, 12, 18, 24]

SCENARIO_MAP = {
    "underfunded": 0.0,
    "dynamic": 0.33,
    "optimal": 0.67,
    "overfunded": 1.0,
}

STATUS_BAND_MAP = {
    "red_low": 0,
    "orange_low": 1,
    "green": 2,
    "orange_high": 3,
    "red_high": 4,
}


FEATURE_COLUMNS = [
    # Attainability
    "trailing_6_period_slope",
    "variance_from_target",
    "volatility_measure",
    "time_to_green_estimate",
    # Relevance
    "allocation_percentage_of_parent",
    "optimal_band_distance",
    "sibling_rank_pct",
    "scenario_encoded",
    "allocation_fitness_score",
    # Coherence
    "l3_share_of_l2",
    "l3_share_of_l1",
    "alloc_drift_std",
    "weighted_goal_status_score",
    "status_band_unique",
    # Integrity
    "delivered_output_quality_score",
    "delivered_output_quantity",
    "allocation_efficiency_ratio",
    "needle_move_ratio",
    "output_cost_per_unit",
    # Dependency
    "n_dependencies",
    "n_dependents",
    "dependency_risk_encoded",
    "dep_avg_attain",
    # Additional
    "observed_value",
    "allocated_amount",
    "allocated_time_hours",
]


def _safe_read_dependency_file() -> pd.DataFrame | None:
    """
    I load goal_dependencies.csv if it exists. If not, I continue without it.
    """
    path = BASE_DIR / "goal_dependencies.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _build_hierarchy_features(buckets: pd.DataFrame) -> pd.DataFrame:
    """
    I build hierarchy mappings from L3 to L2 and L1 plus sibling rank features.
    """
    l1 = buckets[buckets["bucket_level"] == 1][["bucket_id", "allocation_percentage_of_total"]].copy()
    l1.columns = ["l1_id", "l1_alloc_pct"]

    l2 = buckets[buckets["bucket_level"] == 2][["bucket_id", "parent_bucket_id", "allocation_percentage_of_total"]].copy()
    l2.columns = ["l2_id", "l1_id", "l2_alloc_pct"]

    l3 = buckets[buckets["bucket_level"] == 3][["bucket_id", "parent_bucket_id", "allocation_percentage_of_total", "is_leaf"]].copy()
    l3.columns = ["l3_id", "l2_id", "l3_alloc_pct", "is_leaf"]

    hier = l3.merge(l2, on="l2_id").merge(l1, on="l1_id")
    hier["l3_share_of_l2"] = hier["l3_alloc_pct"] / hier["l2_alloc_pct"].clip(lower=1e-9)
    hier["l3_share_of_l1"] = hier["l3_alloc_pct"] / hier["l1_alloc_pct"].clip(lower=1e-9)

    hier["sibling_count"] = hier.groupby("l2_id")["l3_id"].transform("count")
    hier["sibling_rank"] = hier.groupby("l2_id")["l3_alloc_pct"].rank(ascending=False, method="first")
    denom = (hier["sibling_count"] - 1).clip(lower=1)
    hier["sibling_rank_pct"] = (hier["sibling_rank"] - 1) / denom

    return hier


def _build_goal_static_features(goals: pd.DataFrame, hier: pd.DataFrame) -> pd.DataFrame:
    """
    I create goal-level static features that do not change by period.
    """
    goals = goals.copy()
    goals["scenario_encoded"] = goals["scenario_story"].map(SCENARIO_MAP).fillna(0.33)

    goal_static = goals[
        [
            "goal_id",
            "bucket_id",
            "scenario_encoded",
            "minimum_viable_allocation",
            "optimal_allocation_min",
            "optimal_allocation_max",
        ]
    ].merge(
        hier[["l3_id", "l2_id", "l1_id", "l3_share_of_l2", "l3_share_of_l1", "sibling_rank_pct"]],
        left_on="bucket_id",
        right_on="l3_id",
        how="left",
    )

    return goal_static


def _build_temporal_features(allocations: pd.DataFrame, flat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    I build temporal consistency features:
    - allocation drift across periods
    - unique status bands across periods
    """
    alloc_drift = allocations.groupby("bucket_id").agg(
        alloc_drift_std=("allocation_percentage_of_parent", "std"),
        alloc_mean=("allocation_percentage_of_parent", "mean"),
    ).reset_index()

    status_consistency = flat.groupby("goal_id").agg(
        status_band_unique=("status_band", "nunique"),
    ).reset_index()

    return alloc_drift, status_consistency


def _build_metrics_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    I compute needle_move_ratio from observed and expected values.
    """
    metrics = metrics.copy()
    metrics["needle_move_ratio"] = (
        metrics["observed_value"] / metrics["expected_value"].clip(lower=0.001)
    ).clip(0, 2)
    return metrics


def _merge_dependency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I merge dependency signals if goal_dependencies.csv exists.
    """
    dep_df = _safe_read_dependency_file()

    if dep_df is None:
        df["n_dependencies"] = 0.0
        df["n_dependents"] = 0.0
        df["dependency_risk_encoded"] = 0.0
        df["dep_avg_attain"] = -1.0
        return df

    dep_merge = dep_df[
        ["goal_id", "n_dependencies", "n_dependents", "dependency_risk", "dep_avg_attain"]
    ].copy()

    dep_merge["dependency_risk_encoded"] = dep_merge["dependency_risk"].map(
        {"none": 0.0, "low": 0.33, "medium": 0.67, "high": 1.0}
    ).fillna(0.0)

    df = df.merge(
        dep_merge[["goal_id", "n_dependencies", "n_dependents", "dependency_risk_encoded", "dep_avg_attain"]],
        on="goal_id",
        how="left",
    )

    df["n_dependencies"] = df["n_dependencies"].fillna(0)
    df["n_dependents"] = df["n_dependents"].fillna(0)
    df["dependency_risk_encoded"] = df["dependency_risk_encoded"].fillna(0.0)
    df["dep_avg_attain"] = df["dep_avg_attain"].fillna(-1.0)

    return df


def _add_optimal_band_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    I compute how far current allocation is from the center of the optimal band.
    """
    df = df.copy()

    alloc_col = "allocation_percentage_of_total_bucket"
    if alloc_col not in df.columns and "allocation_percentage_of_total" in df.columns:
        alloc_col = "allocation_percentage_of_total"

    opt_centre = (df["optimal_allocation_min"] + df["optimal_allocation_max"]) / 2.0
    opt_range = (df["optimal_allocation_max"] - df["optimal_allocation_min"]).clip(lower=1e-9)

    df["optimal_band_distance"] = (
        (df[alloc_col] - opt_centre).abs() / opt_range
    ).clip(0, 1)

    return df


def build_full_feature_dataframe(source_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    I build the full 840-row engineered feature dataframe from all source tables.
    """
    flat = source_tables["analytical_flat"].copy()
    buckets = source_tables["buckets"].copy()
    goals = source_tables["goals"].copy()
    allocations = source_tables["allocations"].copy()
    metrics = source_tables["metrics"].copy()
    derived = source_tables["derived_fields"].copy()

    hier = _build_hierarchy_features(buckets)
    goal_static = _build_goal_static_features(goals, hier)
    alloc_drift, status_consistency = _build_temporal_features(allocations, flat)
    metrics = _build_metrics_features(metrics)

    df = flat.copy()

    # Merge derived columns
    derived_needed = ["goal_id", "period_id", "weighted_goal_status_score", "allocation_fitness_score", "time_to_green_estimate"]
    derived_existing = [c for c in derived_needed if c in derived.columns]
    df = df.merge(
        derived[derived_existing],
        on=["goal_id", "period_id"],
        how="left",
    )

    # Handle duplicate time_to_green_estimate if merge created x/y columns
    if "time_to_green_estimate_y" in df.columns:
        df["time_to_green_estimate"] = df["time_to_green_estimate_y"]
    elif "time_to_green_estimate_x" in df.columns:
        df["time_to_green_estimate"] = df["time_to_green_estimate_x"]

    # Merge needle_move_ratio from metrics
    metrics_needed = ["goal_id", "period_id", "needle_move_ratio"]
    df = df.merge(metrics[metrics_needed], on=["goal_id", "period_id"], how="left")

    # Merge goal-level static features
    static_needed = ["goal_id", "scenario_encoded", "l3_share_of_l2", "l3_share_of_l1", "sibling_rank_pct",
                     "minimum_viable_allocation", "optimal_allocation_min", "optimal_allocation_max"]
    existing_static = [c for c in static_needed if c in goal_static.columns]
    df = df.merge(goal_static[existing_static], on="goal_id", how="left")

    # Merge allocation drift
    df = df.merge(alloc_drift[["bucket_id", "alloc_drift_std"]], on="bucket_id", how="left")

    # Merge status consistency
    df = df.merge(status_consistency, on="goal_id", how="left")

    # Merge dependencies if present
    df = _merge_dependency_features(df)

    # Add optimal band distance
    df = _add_optimal_band_distance(df)

    # Add encoded status band
    if "status_band" in df.columns:
        df["status_band_encoded"] = df["status_band"].map(STATUS_BAND_MAP).fillna(2)

    return df


def _prepare_feature_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[str], StandardScaler]:
    """
    I create raw and normalized feature matrices from the engineered dataframe.
    """
    available = [f for f in FEATURE_COLUMNS if f in df.columns]
    X_raw = df[available].copy()
    X_raw = X_raw.fillna(X_raw.mean(numeric_only=True))

    if "status_band_encoded" in df.columns:
        X_raw["status_band_encoded"] = df["status_band_encoded"].fillna(2)
        available = available + ["status_band_encoded"]

    scaler = StandardScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns, index=df.index)

    return X_raw, X_norm, available, scaler


def _compute_rule_scores_for_snapshot(df_snapshot: pd.DataFrame, hier: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    I compute rule-based R/C/I anchors for one snapshot period.
    """
    df = df_snapshot.copy()

    # Relevance
    rel_fitness = df["allocation_fitness_score"].fillna(0)
    rel_band = (1 - df["optimal_band_distance"].fillna(0.5)).clip(0, 1)
    rel_sibling = (1 - df["sibling_rank_pct"].fillna(0.5)).clip(0, 1)
    under_flags = df["underfunded_flag"] if "underfunded_flag" in df.columns else pd.Series([False] * len(df))
    over_flags = df["overfunded_flag"] if "overfunded_flag" in df.columns else pd.Series([False] * len(df))
    flag_penalty = np.where(under_flags | over_flags, 0.85, 1.0)
    relevance_rule = ((0.4 * rel_band) + (0.3 * rel_sibling) + (0.3 * rel_fitness)) * flag_penalty
    relevance_rule = relevance_rule.clip(0, 1)

    # Coherence
    drift_vals = df["alloc_drift_std"].fillna(df["alloc_drift_std"].mean())
    max_drift = max(float(drift_vals.max()), 1e-9)
    drift_score = 1 - (drift_vals / max_drift).clip(0, 1)

    wgs = df["weighted_goal_status_score"].fillna(0.5)
    status_unique = df["status_band_unique"].fillna(3)
    status_score = (1 - ((status_unique - 1) / 4)).clip(0, 1)

    if "l3_share_of_l2" in df.columns:
        hier_score = df["l3_share_of_l2"].fillna(df["l3_share_of_l2"].mean()).clip(0, 1)
    else:
        hier_score = pd.Series([0.5] * len(df))

    coherence_rule = (0.35 * hier_score + 0.25 * drift_score + 0.25 * wgs + 0.15 * status_score).clip(0, 1)

    # Integrity
    step1 = df["allocation_efficiency_ratio"].fillna(0).clip(0, 1)
    step2 = (
        0.5 * df["delivered_output_quality_score"].fillna(0).clip(0, 1) +
        0.5 * df["needle_move_ratio"].fillna(0).clip(0, 1)
    )
    obs_exp = (
        df["observed_value"] / df["expected_value"].clip(lower=0.001)
    ).clip(0, 1)
    integrity_rule = (0.3 * step1 + 0.35 * step2 + 0.35 * obs_exp).clip(0, 1)

    attainability_label = df["probability_of_hitting_target"].clip(0, 1)

    out = pd.DataFrame({
        "goal_idx": range(len(df)),
        "goal_id": df["goal_id"].values,
        "bucket_id": df["bucket_id"].values,
        "relevance_rule": relevance_rule.values,
        "coherence_rule": coherence_rule.values,
        "integrity_rule": integrity_rule.values,
        "attainability_label": attainability_label.values,
    })

    return out


def save_feature_outputs(source_tables: Dict[str, pd.DataFrame], output_dir: Path | None = None, verbose: bool = True) -> Dict[str, Path]:
    """
    I generate and save all feature engineering outputs that System 2 expects.
    """
    if output_dir is None:
        output_dir = BASE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = build_full_feature_dataframe(source_tables)
    X_raw, X_norm, available, scaler = _prepare_feature_matrices(full_df)

    # Save full feature outputs
    X_full_norm = X_norm.copy()
    X_full_norm["goal_id"] = full_df["goal_id"].values
    X_full_norm["period_id"] = full_df["period_id"].values
    X_full_norm["y_attain"] = full_df["probability_of_hitting_target"].values

    X_full_raw = X_raw.copy()
    X_full_raw["goal_id"] = full_df["goal_id"].values
    X_full_raw["period_id"] = full_df["period_id"].values

    outputs: Dict[str, Path] = {}

    outputs["features_full_normalized"] = output_dir / "features_full_normalized.csv"
    outputs["features_full_raw"] = output_dir / "features_full_raw.csv"
    X_full_norm.to_csv(outputs["features_full_normalized"], index=False)
    X_full_raw.to_csv(outputs["features_full_raw"], index=False)

    # Save POC period 12 outputs
    p12_mask = full_df["period_id"] == 12
    X_p12_norm = X_norm[p12_mask].copy().reset_index(drop=True)
    X_p12_raw = X_raw[p12_mask].copy().reset_index(drop=True)
    df_p12 = full_df[p12_mask].copy().reset_index(drop=True)

    outputs["features_normalized_poc"] = output_dir / "features_normalized_poc.csv"
    outputs["features_raw_poc"] = output_dir / "features_raw_poc.csv"
    outputs["period_12_poc"] = output_dir / "period_12_poc.csv"

    X_p12_norm.to_csv(outputs["features_normalized_poc"], index=False)
    X_p12_raw.to_csv(outputs["features_raw_poc"], index=False)
    df_p12.to_csv(outputs["period_12_poc"], index=False)

    # Save snapshot raw features + rule scores
    for period_id in SNAPSHOT_PERIODS:
        mask = full_df["period_id"] == period_id
        df_snapshot = full_df[mask].copy().reset_index(drop=True)
        raw_snapshot = X_raw[mask].copy().reset_index(drop=True)

        raw_path = output_dir / f"features_raw_p{period_id}.csv"
        rule_path = output_dir / f"rule_scores_p{period_id}.csv"

        raw_snapshot.to_csv(raw_path, index=False)
        outputs[f"features_raw_p{period_id}"] = raw_path

        rule_df = _compute_rule_scores_for_snapshot(df_snapshot)
        rule_df.to_csv(rule_path, index=False)
        outputs[f"rule_scores_p{period_id}"] = rule_path

        if period_id == 12:
            poc_rule_path = output_dir / "rule_scores_poc.csv"
            rule_df.to_csv(poc_rule_path, index=False)
            outputs["rule_scores_poc"] = poc_rule_path

    # Save scaler + feature names
    scaler_path = output_dir / "feature_scaler_poc.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    outputs["feature_scaler_poc"] = scaler_path

    names_path = output_dir / "feature_names_poc.txt"
    with open(names_path, "w", encoding="utf-8") as f:
        for name in available:
            f.write(f"{name}\n")
    outputs["feature_names_poc"] = names_path

    if verbose:
        print("\nFeature engineering outputs saved:")
        for name, path in outputs.items():
            print(f"  ✓ {name:<25} {path.name}")

    return outputs


if __name__ == "__main__":
    from .data_loader import load_source_tables

    source_tables = load_source_tables(verbose=True)
    save_feature_outputs(source_tables, verbose=True)
