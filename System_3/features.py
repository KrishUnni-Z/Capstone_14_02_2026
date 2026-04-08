import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "goal_id",
    "period_id",
    "observed_value",
    "expected_value",
    "variance_from_target",
    "trailing_3_period_slope",
    "trailing_6_period_slope",
    "volatility_measure",
    "allocated_amount",
    "allocated_time_hours",
    "allocation_percentage_of_total",
    "allocation_percentage_of_parent",
    "delivered_output_quantity",
    "delivered_output_quality_score",
    "output_cost_per_unit",
    "total_cost",
    "range_position_score",
    "allocation_efficiency_ratio",
    "probability_of_hitting_target",
    "time_to_green_estimate",
    "minimum_viable_allocation",
    "optimal_allocation_min",
    "optimal_allocation_max",
    "target_value_final_period",
    "initial_value",
]


def add_safe_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few simple and interpretable derived features.
    """
    df = df.copy()

    if {"target_value_final_period", "observed_value"}.issubset(df.columns):
        df["performance_gap"] = df["target_value_final_period"] - df["observed_value"]

    if {"observed_value", "target_value_final_period"}.issubset(df.columns):
        denominator = df["target_value_final_period"].replace(0, np.nan)
        df["progress_ratio"] = df["observed_value"] / denominator

    if {"allocated_amount", "delivered_output_quantity"}.issubset(df.columns):
        denominator = df["delivered_output_quantity"].replace(0, np.nan)
        df["cost_per_delivered_unit"] = df["allocated_amount"] / denominator

    if "probability_of_hitting_target" in df.columns:
        df["risk_flag"] = df["probability_of_hitting_target"] < 0.5

    if "status_band" in df.columns:
        status_map = {
            "red_low": 0,
            "orange_low": 1,
            "green": 2,
            "orange_high": 3,
            "red_high": 4,
        }
        df["status_band_encoded"] = df["status_band"].map(status_map)

    return df


def select_system2_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns relevant for System 2 statistical input.
    Keeps only columns that actually exist.
    """
    df = df.copy()

    extra_columns = [
        "performance_gap",
        "progress_ratio",
        "cost_per_delivered_unit",
        "risk_flag",
        "status_band_encoded",
    ]

    selected_cols = [col for col in FEATURE_COLUMNS + extra_columns if col in df.columns]
    return df[selected_cols].copy()


def fill_feature_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe missing-value handling for model-ready features.
    Numeric -> median
    Boolean -> False
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].fillna(False)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

    return df


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-preparation flow for System 2.
    """
    df = add_safe_derived_features(df)
    df = select_system2_feature_columns(df)
    df = fill_feature_missing_values(df)
    return df
