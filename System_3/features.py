"""
features.py

System 3 feature engineering layer.

This file ncludes the rule-score helper from rules.py, so rules.py can be
removed if nothing else imports it.

What happens here:
- Build full 840-row engineered features.
- Build raw feature snapshots for p6, p12, p18, p24.
- Build rule score files for p6, p12, p18, p24.
- Save p12 aliases:
    features_raw_poc.csv
    features_normalized_poc.csv
    rule_scores_poc.csv
- Save p18 anchor snapshot:
    period_18_poc.csv
- Save:
    features_full_normalized.csv
    features_full_raw.csv
    feature_scaler_poc.pkl
    feature_names_poc.txt

This keeps the original feature-engineering behaviour and keeps p18 as the
anchor period, while p12 aliases remain for backward compatibility.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

ANCHOR_PERIOD = 18
SNAPSHOT_PERIODS = [6, 12, 18, 24]
FINAL_PERIOD = 24

SCENARIO_MAP = {
    "underfunded": 0.0,
    "dynamic": 0.33,
    "optimal": 0.67,
    "overfunded": 1.0,
}

STATUS_MAP = {
    "red_low": 0,
    "orange_low": 1,
    "green": 2,
    "orange_high": 3,
    "red_high": 4,
}

STAT_FEATURES = [
    "trailing_6_period_slope",
    "variance_from_target",
    "volatility_measure",
    "time_to_green_estimate",
    "allocation_percentage_of_parent",
    "optimal_band_distance",
    "sibling_rank_pct",
    "scenario_encoded",
    "allocation_fitness_score",
    "l3_share_of_l2",
    "l3_share_of_l1",
    "alloc_drift_std",
    "weighted_goal_status_score",
    "status_band_unique",
    "delivered_output_quality_score",
    "delivered_output_quantity",
    "allocation_efficiency_ratio",
    "needle_move_ratio",
    "output_cost_per_unit",
    "n_dependencies",
    "n_dependents",
    "dependency_risk_encoded",
    "dep_avg_attain",
    "observed_value",
    "allocated_amount",
    "allocated_time_hours",
    "budget_shock_exposure",
    "shock_alloc_impact",
    "recovery_period_estimate",
    "recovery_window_remaining",
    "market_shock_vulnerable",
    "market_shock_forward_risk",
    "period_id_scaled",
    "market_shock_period",
]


def build_hierarchy(buckets: pd.DataFrame) -> pd.DataFrame:
    """
    Build the L1/L2/L3 hierarchy and sibling ranking features.
    """
    l1 = buckets[buckets["bucket_level"] == 1][
        ["bucket_id", "allocation_percentage_of_total"]
    ].copy()
    l1.columns = ["l1_id", "l1_alloc_pct"]

    l2 = buckets[buckets["bucket_level"] == 2][
        ["bucket_id", "parent_bucket_id", "allocation_percentage_of_total"]
    ].copy()
    l2.columns = ["l2_id", "l1_id", "l2_alloc_pct"]

    l3 = buckets[buckets["bucket_level"] == 3][
        ["bucket_id", "parent_bucket_id", "allocation_percentage_of_total", "is_leaf"]
    ].copy()
    l3.columns = ["l3_id", "l2_id", "l3_alloc_pct", "is_leaf"]

    hier = l3.merge(l2, on="l2_id").merge(l1, on="l1_id")

    hier["l3_share_of_l2"] = hier["l3_alloc_pct"] / hier["l2_alloc_pct"].clip(lower=1e-9)
    hier["l3_share_of_l1"] = hier["l3_alloc_pct"] / hier["l1_alloc_pct"].clip(lower=1e-9)

    hier["sibling_count"] = hier.groupby("l2_id")["l3_id"].transform("count")
    hier["sibling_rank"] = hier.groupby("l2_id")["l3_alloc_pct"].rank(
        ascending=False,
        method="first",
    )
    hier["sibling_rank_pct"] = (
        (hier["sibling_rank"] - 1) / (hier["sibling_count"] - 1).clip(lower=1)
    )

    return hier


def add_dependency_features(df: pd.DataFrame, output_dir: Path = BASE_DIR) -> pd.DataFrame:
    """
    Merge dependency features if goal_dependencies.csv exists.
    If it does not exist, set safe default values.
    """
    path = output_dir / "goal_dependencies.csv"

    if not path.exists():
        df["n_dependencies"] = 0.0
        df["n_dependents"] = 0.0
        df["dependency_risk_encoded"] = 0.0
        df["dep_avg_attain"] = -1.0
        return df

    dep = pd.read_csv(path)
    dep = dep[
        ["goal_id", "n_dependencies", "n_dependents", "dependency_risk", "dep_avg_attain"]
    ].copy()

    dep["dependency_risk_encoded"] = dep["dependency_risk"].map(
        {"none": 0.0, "low": 0.33, "medium": 0.67, "high": 1.0}
    ).fillna(0.0)

    df = df.merge(
        dep[["goal_id", "n_dependencies", "n_dependents", "dependency_risk_encoded", "dep_avg_attain"]],
        on="goal_id",
        how="left",
    )

    df["n_dependencies"] = df["n_dependencies"].fillna(0.0)
    df["n_dependents"] = df["n_dependents"].fillna(0.0)
    df["dependency_risk_encoded"] = df["dependency_risk_encoded"].fillna(0.0)
    df["dep_avg_attain"] = df["dep_avg_attain"].fillna(-1.0)

    return df


def add_shock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the shock features expected by the brain/System 2 pipeline.
    """
    df = df.copy()

    budget_shock_periods = {10, 11, 12}
    market_shock_goals = {4, 7, 9, 10, 11}
    shock_end_period = 12

    df["budget_shock_exposure"] = df["period_id"].apply(
        lambda p: 1.0 if int(p) in budget_shock_periods else 0.0
    )

    baseline_alloc = (
        df[df["period_id"].isin([7, 8, 9])]
        .groupby("goal_id")["allocated_amount"]
        .mean()
        .rename("baseline_alloc")
    )

    df = df.merge(baseline_alloc, on="goal_id", how="left")

    df["shock_alloc_impact"] = (
        ((df["allocated_amount"] - df["baseline_alloc"]) / df["baseline_alloc"].clip(lower=1e-9))
        .clip(-1.0, 0.0)
        .abs()
        .fillna(0.0)
    )

    df = df.drop(columns=["baseline_alloc"], errors="ignore")

    df["recovery_period_estimate"] = df["period_id"].apply(
        lambda p: float(max(0, int(p) - shock_end_period)) if int(p) > shock_end_period else 0.0
    )

    df["recovery_window_remaining"] = df["period_id"].apply(
        lambda p: float(max(0, FINAL_PERIOD - max(int(p), shock_end_period)))
    )

    df["market_shock_vulnerable"] = df["goal_id"].apply(
        lambda g: 1.0 if int(g) in market_shock_goals else 0.0
    )

    p13 = df[df["period_id"] == 13].set_index("goal_id")["observed_value"]
    p14 = df[df["period_id"] == 14].set_index("goal_id")["observed_value"]
    market_drop = ((p14 - p13) / p13.clip(lower=1e-9)).clip(-1.0, 0.0).abs()
    df["market_shock_forward_risk"] = df["goal_id"].map(market_drop.to_dict()).fillna(0.0)

    return df


def build_full_feature_dataframe(source_tables: Dict[str, pd.DataFrame], output_dir: Path = BASE_DIR) -> pd.DataFrame:
    """
    Build the full engineered dataframe before selecting model features.
    """
    flat = source_tables["analytical_flat"].copy()
    buckets = source_tables["buckets"].copy()
    goals = source_tables["goals"].copy()
    allocations = source_tables["allocations"].copy()
    metrics = source_tables["metrics"].copy()
    derived = source_tables["derived_fields"].copy()

    hier = build_hierarchy(buckets)

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

    alloc_drift = allocations.groupby("bucket_id").agg(
        alloc_drift_std=("allocation_percentage_of_parent", "std"),
        alloc_mean=("allocation_percentage_of_parent", "mean"),
    ).reset_index()

    status_consistency = flat.groupby("goal_id").agg(
        status_band_unique=("status_band", "nunique"),
    ).reset_index()

    metrics = metrics.copy()
    metrics["needle_move_ratio"] = (
        metrics["observed_value"] / metrics["expected_value"].clip(lower=0.001)
    ).clip(0, 2)

    df = flat.copy()

    df = df.merge(
        derived[
            [
                "goal_id",
                "period_id",
                "weighted_goal_status_score",
                "allocation_fitness_score",
                "time_to_green_estimate",
            ]
        ],
        on=["goal_id", "period_id"],
        how="left",
    )

    if "time_to_green_estimate_y" in df.columns:
        df["time_to_green_estimate"] = df["time_to_green_estimate_y"]
    elif "time_to_green_estimate_x" in df.columns:
        df["time_to_green_estimate"] = df["time_to_green_estimate_x"]

    df = df.merge(
        metrics[["goal_id", "period_id", "needle_move_ratio"]],
        on=["goal_id", "period_id"],
        how="left",
    )

    df = df.merge(
        goal_static[
            [
                "goal_id",
                "scenario_encoded",
                "l3_share_of_l2",
                "l3_share_of_l1",
                "sibling_rank_pct",
            ]
        ],
        on="goal_id",
        how="left",
    )

    df = df.merge(
        alloc_drift[["bucket_id", "alloc_drift_std"]],
        on="bucket_id",
        how="left",
    )

    df = df.merge(status_consistency, on="goal_id", how="left")

    df = add_dependency_features(df, output_dir=output_dir)

    opt_center = (df["optimal_allocation_min"] + df["optimal_allocation_max"]) / 2
    opt_range = (df["optimal_allocation_max"] - df["optimal_allocation_min"]).clip(lower=1e-9)

    allocation_col = (
        "allocation_percentage_of_total_bucket"
        if "allocation_percentage_of_total_bucket" in df.columns
        else "allocation_percentage_of_total"
    )

    df["optimal_band_distance"] = (
        (df[allocation_col] - opt_center).abs() / opt_range
    ).clip(0, 1)

    df = add_shock_features(df)

    # Temporal GP features — must be present for meta_learner.py SlicedProduct kernel
    df["period_id_scaled"]   = df["period_id"] / 24.0
    df["market_shock_period"] = df["period_id"].isin([14, 15, 16, 17]).astype(float)

    if "status_band" in df.columns:
        df["status_band_encoded"] = df["status_band"].map(STATUS_MAP).fillna(2)

    return df


def prepare_feature_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], StandardScaler]:
    """
    Create raw and normalized feature matrices.
    """
    available = [col for col in STAT_FEATURES if col in df.columns]

    X_raw = df[available].copy()
    X_raw = X_raw.fillna(X_raw.mean(numeric_only=True))

    if "status_band_encoded" in df.columns:
        X_raw["status_band_encoded"] = df["status_band_encoded"].fillna(2)
        available.append("status_band_encoded")

    scaler = StandardScaler()
    X_norm = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=X_raw.columns,
        index=df.index,
    )

    return X_raw, X_norm, available, scaler



# rules.py merged here
def _clip01(series_or_value):
    """
    Clip values into the 0-1 range.
    """
    return np.clip(series_or_value, 0, 1)


def build_rule_score_dataframe(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Original rules.py helper, kept here so rules.py can be deleted.

    Note: save_feature_outputs() uses compute_rich_rule_scores() for p18 and
    compute_simple_rule_scores() for p6/p12/p24 to match the brain code.
    This helper is kept only for compatibility if another script imports it
    from features.py later.
    """
    df = feature_df.copy()

    band_score = 1 - df["optimal_band_distance"].fillna(0.5)
    sibling_score = 1 - df["sibling_rank_pct"].fillna(0.5)
    fitness = df["allocation_fitness_score"].fillna(0.0)

    flag_penalty = np.where(
        df.get("underfunded_flag", False) | df.get("overfunded_flag", False),
        0.85,
        1.0,
    )

    relevance_rule = _clip01(
        (0.40 * band_score + 0.35 * sibling_score + 0.25 * fitness) * flag_penalty
    )

    drift = df["alloc_drift_std"].fillna(0.0)
    max_drift = drift.max() if len(drift) and drift.max() > 0 else 1.0
    drift_score = 1 - _clip01(drift / max_drift)

    hier_score = df["weighted_goal_status_score"].fillna(0.5)
    status_band_unique = df["status_band_unique"].fillna(1.0)
    status_score = 1 - _clip01((status_band_unique - 1) / 4.0)

    coherence_rule = _clip01(
        0.40 * drift_score + 0.35 * hier_score + 0.25 * status_score
    )

    step1 = _clip01(df["allocation_efficiency_ratio"].fillna(0.0))
    step2 = _clip01(df["needle_move_ratio"].fillna(0.0))
    expected = df["expected_value"].replace(0, np.nan)
    step3 = _clip01((df["observed_value"] / expected).fillna(0.0))

    integrity_rule = _clip01(
        0.40 * step1 + 0.35 * step2 + 0.25 * step3
    )

    rule_df = pd.DataFrame({
        "goal_id": df["goal_id"].values,
        "period_id": df["period_id"].values,
        "relevance_rule": relevance_rule,
        "coherence_rule": coherence_rule,
        "integrity_rule": integrity_rule,
    })

    return rule_df.reset_index(drop=True)


def compute_rich_rule_scores(df_sp: pd.DataFrame, hier: pd.DataFrame):
    """
    Rich p18 anchor rule formula, kept from the brain feature_engineering.py.
    """
    rel_fitness = df_sp["allocation_fitness_score"].fillna(0)
    rel_band = (1 - df_sp["optimal_band_distance"]).clip(0, 1)
    rel_sibling = (1 - df_sp["sibling_rank_pct"]).clip(0, 1)

    flag_penalty = np.where(
        df_sp.get("underfunded_flag", pd.Series([False] * len(df_sp)))
        | df_sp.get("overfunded_flag", pd.Series([False] * len(df_sp))),
        0.85,
        1.0,
    )

    relevance = ((0.4 * rel_band + 0.3 * rel_sibling + 0.3 * rel_fitness) * flag_penalty).clip(0, 1)

    expected_share = 1.0 / hier.set_index("l3_id")["sibling_count"].reindex(df_sp["bucket_id"]).values
    actual_share = hier.set_index("l3_id")["l3_share_of_l2"].reindex(df_sp["bucket_id"]).values

    hier_gap = np.abs(actual_share - expected_share) / np.clip(expected_share, 0.001, None)
    hier_score = 1 - np.clip(hier_gap, 0, 1)

    drift = df_sp["alloc_drift_std"].fillna(df_sp["alloc_drift_std"].mean())
    drift_score = 1 - (drift / max(drift.max(), 1e-9)).clip(0, 1)

    wgs = df_sp["weighted_goal_status_score"].fillna(0.5)
    status_u = df_sp["status_band_unique"].fillna(3)
    status_score = (1 - (status_u - 1) / 4).clip(0, 1)

    coherence = (0.35 * hier_score + 0.25 * drift_score + 0.25 * wgs + 0.15 * status_score).clip(0, 1)

    step1 = df_sp["allocation_efficiency_ratio"].clip(0, 1)
    step2 = (
        0.5 * df_sp["delivered_output_quality_score"].clip(0, 1)
        + 0.5 * df_sp["needle_move_ratio"].clip(0, 1).fillna(0)
    )
    obs_exp = (df_sp["observed_value"] / df_sp["expected_value"].clip(lower=0.001)).clip(0, 1)

    integrity = (0.3 * step1 + 0.35 * step2 + 0.35 * obs_exp).clip(0, 1)

    return relevance, coherence, integrity


def compute_simple_rule_scores(df_sp: pd.DataFrame):
    """
    Simpler rule formula for non-anchor periods p6, p12, and p24.
    """
    rel_fitness = df_sp["allocation_fitness_score"].fillna(0)
    rel_band = (1 - df_sp["optimal_band_distance"].fillna(0.5)).clip(0, 1)
    rel_sibling = (1 - df_sp["sibling_rank_pct"].fillna(0.5)).clip(0, 1)

    flag_penalty = np.where(
        df_sp.get("underfunded_flag", pd.Series([False] * len(df_sp)))
        | df_sp.get("overfunded_flag", pd.Series([False] * len(df_sp))),
        0.85,
        1.0,
    )

    relevance = ((0.4 * rel_band + 0.3 * rel_sibling + 0.3 * rel_fitness) * flag_penalty).clip(0, 1)

    drift = df_sp["alloc_drift_std"].fillna(df_sp["alloc_drift_std"].mean())
    drift_score = 1 - (drift / max(drift.max(), 1e-9)).clip(0, 1)
    wgs = df_sp["weighted_goal_status_score"].fillna(0.5)
    coherence = (0.5 * drift_score + 0.5 * wgs).clip(0, 1)

    eff = df_sp["allocation_efficiency_ratio"].fillna(0).clip(0, 1)
    needle = df_sp["needle_move_ratio"].fillna(0).clip(0, 1)
    quality = df_sp["delivered_output_quality_score"].fillna(0).clip(0, 1)
    integrity = (0.4 * eff + 0.3 * needle + 0.3 * quality).clip(0, 1)

    return relevance, coherence, integrity


def save_feature_outputs(source_tables: Dict[str, pd.DataFrame], output_dir: Path | None = None, verbose: bool = True):
    """
    Generate all feature and rule files that System 2 / the brain expects.
    """
    if output_dir is None:
        output_dir = BASE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = build_full_feature_dataframe(source_tables, output_dir=output_dir)
    buckets = source_tables["buckets"]
    hier = build_hierarchy(buckets)

    X_raw, X_norm, available, scaler = prepare_feature_matrices(full_df)

    X_full_norm = X_norm.copy()
    X_full_norm["goal_id"] = full_df["goal_id"].values
    X_full_norm["period_id"] = full_df["period_id"].values
    X_full_norm["y_attain"] = full_df["probability_of_hitting_target"].values

    X_full_raw = X_raw.copy()
    X_full_raw["goal_id"] = full_df["goal_id"].values
    X_full_raw["period_id"] = full_df["period_id"].values

    X_full_norm.to_csv(output_dir / "features_full_normalized.csv", index=False)
    X_full_raw.to_csv(output_dir / "features_full_raw.csv", index=False)

    snapshot_dfs = {}
    snapshot_raws = {}

    for period in SNAPSHOT_PERIODS:
        mask = full_df["period_id"] == period
        df_sp = full_df[mask].copy().reset_index(drop=True)
        raw_sp = X_raw[mask].copy().reset_index(drop=True)

        raw_sp["goal_id"] = df_sp["goal_id"].values
        raw_sp["bucket_id"] = df_sp["bucket_id"].values
        raw_sp["target_value_final_period"] = df_sp["target_value_final_period"].values

        snapshot_dfs[period] = df_sp
        snapshot_raws[period] = raw_sp

        raw_sp.to_csv(output_dir / f"features_raw_p{period}.csv", index=False)

        if period == ANCHOR_PERIOD:
            rel, coh, integ = compute_rich_rule_scores(df_sp, hier)
        else:
            rel, coh, integ = compute_simple_rule_scores(df_sp)

        rule_df = pd.DataFrame({
            "goal_idx": range(len(df_sp)),
            "goal_id": df_sp["goal_id"].values,
            "bucket_id": df_sp["bucket_id"].values,
            "relevance_rule": rel.values if hasattr(rel, "values") else rel,
            "coherence_rule": coh.values if hasattr(coh, "values") else coh,
            "integrity_rule": integ.values if hasattr(integ, "values") else integ,
            "attainability_label": df_sp["probability_of_hitting_target"].clip(0, 1).values,
        })

        rule_df.to_csv(output_dir / f"rule_scores_p{period}.csv", index=False)

        if verbose:
            print(f"OK  features_raw_p{period}.csv  {raw_sp.shape}")
            print(f"OK  rule_scores_p{period}.csv")

    # Backward-compatible p12 aliases expected by some brain scripts.
    snapshot_raws[12].to_csv(output_dir / "features_raw_poc.csv", index=False)
    X_norm[full_df["period_id"] == 12].copy().reset_index(drop=True).to_csv(
        output_dir / "features_normalized_poc.csv",
        index=False,
    )

    pd.read_csv(output_dir / "rule_scores_p12.csv").to_csv(
        output_dir / "rule_scores_poc.csv",
        index=False,
    )

    # Anchor snapshot.
    snapshot_dfs[ANCHOR_PERIOD].to_csv(output_dir / f"period_{ANCHOR_PERIOD}_poc.csv", index=False)

    with open(output_dir / "feature_names_poc.txt", "w", encoding="utf-8") as f:
        for name in available:
            f.write(f"{name}\n")

    with open(output_dir / "feature_scaler_poc.pkl", "wb") as f:
        pickle.dump(scaler, f)

    if verbose:
        print(f"\nOK  features_full_normalized.csv  {X_full_norm.shape}")
        print("OK  features_full_raw.csv")
        print("OK  features_raw_poc.csv  alias for p12")
        print("OK  rule_scores_poc.csv   alias for p12")
        print("OK  period_18_poc.csv     anchor snapshot")
        print("OK  feature_scaler_poc.pkl")
        print("OK  feature_names_poc.txt")


if __name__ == "__main__":
    try:
        from .data_loader import load_source_tables
    except ImportError:
        from data_loader import load_source_tables

    tables = load_source_tables(verbose=True)
    save_feature_outputs(tables, verbose=True)
