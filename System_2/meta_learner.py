import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
04_meta_learner_poc.py — v9 Coherence Engine
GP + Isotonic Calibration + Engineered Scores + Dynamic Ensemble

UPDATED:
  - Evaluation/calibration subset now uses BOTH period 12 and period 18
  - Safer alignment by goal_id + period_id
  - Keeps full 840-row GP training
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


GP_UNCERTAINTY_SCALE = 50.0
GP_UNCERTAINTY_FLAG = 0.15
MODELS = None
ATTAIN_MODELS = None
DIMS_BLEND = ["relevance", "coherence", "integrity"]

EVAL_PERIODS = [12, 18]

print("=" * 70)
print("META-LEARNER v9 — Coherence Engine (GP + Engineered + LLM Reasoning)")
print("=" * 70)


# ── Helpers ───────────────────────────────────────────────────────────────────
def clip01(x):
    return np.clip(x, 0, 1)


def load_snapshot_raw(period_id):
    if period_id == 12:
        fname = "features_raw_poc.csv"
    else:
        fname = f"features_raw_p{period_id}.csv"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing required raw snapshot file: {fname}")
    df = pd.read_csv(fname).copy()
    df["period_id"] = period_id
    return df


def load_snapshot_rule(period_id):
    if period_id == 12:
        fname = "rule_scores_poc.csv"
    else:
        fname = f"rule_scores_p{period_id}.csv"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing required rule score file: {fname}")
    df = pd.read_csv(fname).copy()
    df["period_id"] = period_id
    return df


def compute_baseline_arr(df):
    obs = df["observed_value"].clip(lower=0)
    slope = df["trailing_6_period_slope"]
    target = df["target_value_final_period"].clip(lower=1e-9)
    remaining = 24 - df["period_id"]
    return ((obs + slope * remaining) / target).clip(0, 1).values


# ── Load data ─────────────────────────────────────────────────────────────────
features_full = pd.read_csv("features_full_normalized.csv")
df_full = pd.read_csv("analytical_flat.csv")
llm_preds = pd.read_csv("llm_predictions_poc.csv")

if "period_id" in llm_preds.columns and llm_preds["period_id"].nunique() > 1:
    print(f"  Multi-period LLM predictions detected: {sorted(llm_preds['period_id'].unique())}")
else:
    llm_preds["period_id"] = 12
    print("  Single-period LLM predictions (period 12 only)")

# Auto-detect models from llm_predictions_poc.csv
MODELS = [
    col.replace("_attainability", "")
    for col in llm_preds.columns
    if col.endswith("_attainability")
    and col.replace("_attainability", "_success") in llm_preds.columns
]
ATTAIN_MODELS = MODELS.copy()
print(f"  Auto-detected models: {MODELS}")

# Core GP training set
goal_ids = features_full["goal_id"].values
y_full = features_full["y_attain"].values
feat_cols = [c for c in features_full.columns if c not in ["goal_id", "period_id", "y_attain"]]
X_full = features_full[feat_cols].values

print(f"\nGP training rows  : {len(X_full)}")
print(f"Features          : {len(feat_cols)}")

# ── Load raw/rule snapshot tables for BOTH p12 and p18 ───────────────────────
raw_eval = pd.concat([load_snapshot_raw(p) for p in EVAL_PERIODS], ignore_index=True)
rule_eval = pd.concat([load_snapshot_rule(p) for p in EVAL_PERIODS], ignore_index=True)

# Ensure goal_id exists in snapshot raw/rule data
if "goal_id" not in raw_eval.columns:
    raise ValueError("goal_id missing from raw snapshot files.")
if "goal_id" not in rule_eval.columns:
    raise ValueError("goal_id missing from rule score files.")

# ── Ground truth table for p12 + p18 ─────────────────────────────────────────
# We also pull target_value_final_period (and observed_value / slope as a
# safety net) from df_full here because compute_baseline_arr reads them from
# eval_df; features_raw_pNN.csv never carried the target column.
gt_cols = ["goal_id", "period_id", "probability_of_hitting_target"]
for extra in ("target_value_final_period", "observed_value", "trailing_6_period_slope"):
    if extra in df_full.columns and extra not in gt_cols:
        gt_cols.append(extra)
df_gt = df_full[gt_cols].copy()
df_gt = df_gt.rename(columns={"probability_of_hitting_target": "y_true"})
df_gt_eval = df_gt[df_gt["period_id"].isin(EVAL_PERIODS)].copy()

# Add goal_id to llm_preds if missing — map from goal_idx via p12 ordering if needed
if "goal_id" not in llm_preds.columns and "goal_idx" in llm_preds.columns:
    p12_ref = pd.read_csv("analytical_flat.csv")
    p12_ref = p12_ref[p12_ref["period_id"] == 12].reset_index(drop=True)
    goal_id_map = dict(enumerate(p12_ref["goal_id"].values))
    llm_preds["goal_id"] = llm_preds["goal_idx"].map(goal_id_map)
    print("  Mapped goal_idx → goal_id in llm_preds")

if "goal_id" not in llm_preds.columns:
    raise ValueError("llm_predictions_poc.csv must include goal_id or goal_idx.")

# ── Evaluation subset = BOTH p12 and p18 ─────────────────────────────────────
llm_eval = llm_preds[llm_preds["period_id"].isin(EVAL_PERIODS)].copy().reset_index(drop=True)
valid_mask = llm_eval["success"].astype(bool)
llm_valid = llm_eval[valid_mask].copy().reset_index(drop=True)
n_valid = len(llm_valid)

print(f"Evaluation periods : {EVAL_PERIODS}")
print(f"Valid LLM rows      : {n_valid}")

# Join to ground truth
eval_df = llm_valid.merge(df_gt_eval, on=["goal_id", "period_id"], how="inner")
if len(eval_df) == 0:
    raise ValueError("No matching ground-truth rows found for p12/p18 evaluation subset.")

# Join raw engineered features
eval_df = eval_df.merge(
    raw_eval,
    on=["goal_id", "period_id"],
    how="inner",
    suffixes=("", "_raw")
)

# Join rule scores
eval_df = eval_df.merge(
    rule_eval,
    on=["goal_id", "period_id"],
    how="left",
    suffixes=("", "_rule")
)

if len(eval_df) == 0:
    raise ValueError("Failed to align LLM predictions with raw/rule tables for p12/p18.")

# Snapshot the RAW signal columns compute_baseline_arr needs. The next merge
# brings in normalized feature values under the same column names and drops
# the raw duplicates; without this snapshot we would silently pass scaled
# values into the baseline formula.
_baseline_raw_cols = ["goal_id", "period_id", "observed_value",
                       "trailing_6_period_slope", "target_value_final_period"]
_baseline_raw_cols = [c for c in _baseline_raw_cols if c in eval_df.columns]
eval_raw_for_baseline = eval_df[_baseline_raw_cols].copy()

# Join normalized features for GP input.
# raw_eval above already brought the same-named engineered columns in. To keep
# feat_cols (which expect CLEAN names from the normalized table) usable on the
# following line, we suffix the raw duplicates and drop them.
features_eval = features_full[features_full["period_id"].isin(EVAL_PERIODS)].copy()
eval_df = eval_df.merge(
    features_eval[["goal_id", "period_id"] + feat_cols],
    on=["goal_id", "period_id"],
    how="inner",
    suffixes=("_rawdup", ""),
)
eval_df = eval_df.drop(columns=[c for c in eval_df.columns if c.endswith("_rawdup")])

if len(eval_df) == 0:
    raise ValueError("Failed to align evaluation subset with features_full_normalized.")

# Final aligned eval structures
y_eval = eval_df["y_true"].values
X_eval = eval_df[feat_cols].values
feat_subset = eval_df.copy()
rule_subset = eval_df.copy()

print(f"\nEval rows after alignment: {len(eval_df)}")
print(
    f"Ground truth: mean={y_eval.mean():.3f}  std={y_eval.std():.3f}  "
    f"range=[{y_eval.min():.3f}, {y_eval.max():.3f}]"
)

print(f"\nModel availability:")
for m in MODELS:
    col = f"{m}_success"
    n_eval = int(eval_df[col].sum()) if col in eval_df.columns else 0
    n_all = int(llm_preds[col].sum()) if col in llm_preds.columns else 0
    print(f"  {m:<10}: {n_eval}/{len(eval_df)} at p12+p18  |  {n_all}/{len(llm_preds)} across all periods")

# Multi-period calibration dataframe
llm_calibration_df = llm_preds.merge(df_gt, on=["goal_id", "period_id"], how="inner")
print(f"\nCalibration rows   : {len(llm_calibration_df)} across {llm_preds['period_id'].nunique()} periods")

# GP input for all 840 rows
X_all_periods = features_full[feat_cols].values

# Residual baseline
baseline_full = compute_baseline_arr(df_full)
y_residual = y_full - baseline_full

baseline_eval = compute_baseline_arr(eval_raw_for_baseline)
y_resid_eval = y_eval - baseline_eval

print(f"\nResidual target: mean={y_residual.mean():.3f}  std={y_residual.std():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN PROCESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("GAUSSIAN PROCESS — 840 rows, residual target")
print("─" * 60)

# VERSION 1 — Temporal GP: Matern(features) × RBF(period)
# sklearn kernels don't support active_dims — we split the feature matrix
# so the temporal feature is appended last and handled separately.
PERIOD_FEAT = "period_id_scaled"
if PERIOD_FEAT in feat_cols:
    period_idx  = feat_cols.index(PERIOD_FEAT)
    feature_idx = [i for i in range(len(feat_cols)) if i != period_idx]
    print(f"  Temporal GP V1: period_id_scaled at index {period_idx}")

    # Wrap kernels to operate on sliced columns via a custom kernel
    from sklearn.gaussian_process.kernels import Kernel

    class SlicedProduct(Kernel):
        """Matern(X[:, feature_idx]) × RBF(X[:, [period_idx]])"""
        def __init__(self, k_feat, k_time, feat_idx, time_idx):
            self.k_feat  = k_feat
            self.k_time  = k_time
            self.feat_idx = feat_idx
            self.time_idx = time_idx

        @property
        def theta(self):
            return np.r_[self.k_feat.theta, self.k_time.theta]

        @theta.setter
        def theta(self, value):
            split = len(self.k_feat.theta)
            self.k_feat.theta = value[:split]
            self.k_time.theta = value[split:]

        @property
        def bounds(self):
            return np.r_[self.k_feat.bounds, self.k_time.bounds]

        def __call__(self, X, Y=None, eval_gradient=False):
            Xf = X[:, self.feat_idx]
            Xt = X[:, [self.time_idx]]
            Yf = Y[:, self.feat_idx] if Y is not None else None
            Yt = Y[:, [self.time_idx]] if Y is not None else None
            if eval_gradient:
                Kf, dKf = self.k_feat(Xf, Yf, eval_gradient=True)
                Kt, dKt = self.k_time(Xt, Yt, eval_gradient=True)
                K = Kf * Kt
                dK = np.dstack([
                    dKf[:, :, i] * Kt for i in range(dKf.shape[2])
                ] + [
                    dKt[:, :, j] * Kf for j in range(dKt.shape[2])
                ])
                return K, dK
            return self.k_feat(Xf, Yf) * self.k_time(Xt, Yt)

        def diag(self, X):
            return self.k_feat.diag(X[:, self.feat_idx]) * self.k_time.diag(X[:, [self.time_idx]])

        def is_stationary(self):
            return self.k_feat.is_stationary() and self.k_time.is_stationary()

        def get_params(self, deep=True):
            return {"k_feat": self.k_feat, "k_time": self.k_time,
                    "feat_idx": self.feat_idx, "time_idx": self.time_idx}

        def __repr__(self):
            return f"SlicedProduct({self.k_feat} × {self.k_time})"

    k_feat = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
        length_scale=1.0, length_scale_bounds=(0.5, 20.0), nu=1.5
    )
    k_time = RBF(
        length_scale=6.0 / 24.0,
        length_scale_bounds=(1.0 / 24.0, 12.0 / 24.0),
    )
    kernel = SlicedProduct(k_feat, k_time, feature_idx, period_idx) + WhiteKernel(
        noise_level=0.05, noise_level_bounds=(1e-5, 0.5)
    )
else:
    print("  WARNING: period_id_scaled not found — fallback to base Matern kernel")
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0))
        * Matern(length_scale=1.0, length_scale_bounds=(0.5, 20.0), nu=1.5)
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 0.5))
    )

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=42,
)

print("Step 1: Fitting GP on 840 rows...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp.fit(X_full, y_residual)

print(f"Fitted kernel: {gp.kernel_}")

print(f"\nStep 2: LOO-CV on {len(y_eval)} eval rows (p12 + p18)...")
loo_resid_preds = np.zeros(len(y_eval))

for i in range(len(y_eval)):
    train_mask = np.ones(len(y_eval), dtype=bool)
    train_mask[i] = False

    gp_loo = GaussianProcessRegressor(
        kernel=gp.kernel_,
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,
        random_state=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_loo.fit(X_eval[train_mask], y_resid_eval[train_mask])
        loo_resid_preds[i] = gp_loo.predict(X_eval[[i]])[0]

loo_final = np.clip(baseline_eval + loo_resid_preds, 0, 1)
gkf_mae = mean_absolute_error(y_eval, loo_final)
gkf_rmse = np.sqrt(mean_squared_error(y_eval, loo_final))
gkf_r2 = r2_score(y_eval, loo_final)
baseline_mae = mean_absolute_error(y_eval, np.clip(baseline_eval, 0, 1))

print(f"\nLOO-CV ({len(y_eval)} rows, p12+p18):")
print(f"  Baseline MAE: {baseline_mae:.4f}")
print(f"  GP+Baseline : {gkf_mae:.4f}  ({'better' if gkf_mae < baseline_mae else 'worse'})")
print(f"  RMSE        : {gkf_rmse:.4f}")
print(f"  R2          : {gkf_r2:.4f}  {'OK' if gkf_r2 > 0 else 'NEGATIVE'}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp_resid_pred, gp_std = gp.predict(X_eval, return_std=True)

gp_mean = np.clip(baseline_eval + gp_resid_pred, 0, 1)

print(
    f"\nGP eval subset: mean={gp_mean.mean():.3f}  "
    f"std=[{gp_std.min():.4f},{gp_std.max():.4f}]  "
    f"uncertain={(gp_std > GP_UNCERTAINTY_FLAG).sum()}/{len(gp_std)}"
)

# GP predictions for all 840 rows
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp_mean_all_raw, gp_std_all = gp.predict(X_all_periods, return_std=True)

baseline_all = compute_baseline_arr(df_full)
gp_mean_all = np.clip(baseline_all + gp_mean_all_raw, 0, 1)
gp_std_all = np.clip(gp_std_all, 0, 1)

print(
    f"GP all 840 rows: mean={gp_mean_all.mean():.3f}  "
    f"uncertain={(gp_std_all > GP_UNCERTAINTY_FLAG).sum()}/840"
)

# ══════════════════════════════════════════════════════════════════════════════
# ISOTONIC CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("ISOTONIC CALIBRATION — continuous ground truth")
print("─" * 60)

iso_scalers = {}
calibrated_llm = {}

for m in ATTAIN_MODELS:
    col = f"{m}_attainability"
    ok_col = f"{m}_success"

    if col not in llm_calibration_df.columns:
        continue

    ok_mask_cal = (
        llm_calibration_df[ok_col].values.astype(bool)
        if ok_col in llm_calibration_df.columns
        else np.ones(len(llm_calibration_df), dtype=bool)
    )

    X_cal_full = llm_calibration_df.loc[ok_mask_cal, col].values
    y_cal_full = llm_calibration_df.loc[ok_mask_cal, "y_true"].values
    n_cal = len(X_cal_full)

    if n_cal < 5:
        if col in eval_df.columns:
            calibrated_llm[m] = eval_df[col].values
        continue

    sort_idx = np.argsort(X_cal_full)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X_cal_full[sort_idx], y_cal_full[sort_idx])
    iso_scalers[m] = iso

    if col in eval_df.columns:
        raw_vals_eval = eval_df[col].values
        ok_mask_eval = (
            eval_df[ok_col].values.astype(bool)
            if ok_col in eval_df.columns
            else np.ones(len(raw_vals_eval), dtype=bool)
        )

        cal_vals = np.full(len(raw_vals_eval), np.nan)
        cal_vals[ok_mask_eval] = iso.predict(raw_vals_eval[ok_mask_eval])
        cal_vals[~ok_mask_eval] = gp_mean[~ok_mask_eval]
        calibrated_llm[m] = cal_vals

        cal_mae = mean_absolute_error(y_cal_full, iso.predict(X_cal_full))
        print(
            f"  {m:<10}: raw={X_cal_full.mean():.3f}  "
            f"calibrated={cal_vals[ok_mask_eval].mean():.3f}  "
            f"gt={y_cal_full.mean():.3f}  MAE={cal_mae:.4f}  "
            f"(calibrated on {n_cal} rows, {llm_preds['period_id'].nunique()} periods)"
        )

# ══════════════════════════════════════════════════════════════════════════════
# ATTAINABILITY BLEND
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("ATTAINABILITY BLEND — GP × confidence + LLM × (1-confidence)")
print("─" * 60)

attainability_final = np.zeros(len(eval_df))
blend_meta = []

for g in range(len(eval_df)):
    gp_m = float(gp_mean[g])
    gp_s = float(gp_std[g])
    gp_conf = 1.0 / (1.0 + gp_s * GP_UNCERTAINTY_SCALE)

    cal_scores = [
        float(calibrated_llm[m][g])
        for m in ATTAIN_MODELS
        if m in calibrated_llm and pd.notna(calibrated_llm[m][g])
    ]

    if not cal_scores:
        attainability_final[g] = float(np.clip(gp_m, 0, 1))
        blend_meta.append({
            "gp_mean": round(gp_m, 4),
            "gp_std": round(gp_s, 4),
            "gp_weight": 1.0,
            "llm_weight": 0.0,
            "llm_mean": None,
            "uncertain": gp_s > GP_UNCERTAINTY_FLAG,
            "fallback": "no_llm",
        })
        continue

    llm_mean = float(np.mean(cal_scores))
    llm_w = 1.0 - gp_conf
    final = float(np.clip(gp_conf * gp_m + llm_w * llm_mean, 0, 1))
    attainability_final[g] = final

    blend_meta.append({
        "gp_mean": round(gp_m, 4),
        "gp_std": round(gp_s, 4),
        "gp_weight": round(gp_conf, 3),
        "llm_weight": round(llm_w, 3),
        "llm_mean": round(llm_mean, 4),
        "uncertain": bool(gp_s > GP_UNCERTAINTY_FLAG),
        "fallback": False,
    })

uncertain_n = sum(1 for m in blend_meta if m["uncertain"])
eval_mae = mean_absolute_error(y_eval, attainability_final)

print(
    f"\n  Final mean    : {attainability_final.mean():.3f}  "
    f"range=[{attainability_final.min():.3f},{attainability_final.max():.3f}]"
)
print(f"  Eval MAE      : {eval_mae:.4f}")
print(f"  Avg GP weight : {np.mean([m['gp_weight'] for m in blend_meta]):.3f}")
print(f"  Avg LLM weight: {np.mean([m['llm_weight'] for m in blend_meta]):.3f}")
print(f"  Uncertain     : {uncertain_n}/{len(eval_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# ENGINEERED SCORES — R / C / I
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("ENGINEERED SCORES — reading from p12 + p18 raw snapshot files")
print("─" * 60)


def engineered_relevance(df):
    fitness = clip01(df["allocation_fitness_score"].fillna(0.0).values)
    sibling = 1.0 - clip01(df["sibling_rank_pct"].fillna(0.5).values)
    band_dist = clip01(df["optimal_band_distance"].fillna(0.5).values)
    band_score = 1.0 - band_dist
    return clip01(0.40 * band_score + 0.35 * sibling + 0.25 * fitness)


def engineered_coherence(df):
    drift_vals = clip01(df["alloc_drift_std"].fillna(0.0).values)
    max_drift = drift_vals.max() if drift_vals.max() > 0 else 1.0
    drift_score = 1.0 - drift_vals / max_drift

    hier_score = clip01(df["weighted_goal_status_score"].fillna(0.5).values)

    status_u = df["status_band_unique"].fillna(3).values
    status_s = 1.0 - clip01((status_u - 1) / 4.0)

    return clip01(0.40 * drift_score + 0.35 * hier_score + 0.25 * status_s)


def engineered_integrity(df):
    efficiency = clip01(df["allocation_efficiency_ratio"].fillna(0.0).values)
    needle = clip01(df["needle_move_ratio"].fillna(0.0).values)
    quality = clip01(df["delivered_output_quality_score"].fillna(0.0).values)
    return clip01(0.40 * efficiency + 0.35 * needle + 0.25 * quality)


for dim_name, cols_needed in [
    ("Relevance", ["allocation_fitness_score", "sibling_rank_pct", "optimal_band_distance"]),
    ("Coherence", ["alloc_drift_std", "weighted_goal_status_score", "status_band_unique"]),
    ("Integrity", ["allocation_efficiency_ratio", "needle_move_ratio", "delivered_output_quality_score"]),
]:
    missing = [c for c in cols_needed if c not in feat_subset.columns]
    print(f"\n  {dim_name} columns: {'OK' if not missing else 'MISSING: ' + str(missing)}")

rel_eng = engineered_relevance(feat_subset)
coh_eng = engineered_coherence(feat_subset)
int_eng = engineered_integrity(feat_subset)

print(f"\n  Relevance engineered: mean={rel_eng.mean():.3f}  range=[{rel_eng.min():.3f},{rel_eng.max():.3f}]")
print(f"  Coherence engineered: mean={coh_eng.mean():.3f}  range=[{coh_eng.min():.3f},{coh_eng.max():.3f}]")
print(f"  Integrity engineered: mean={int_eng.mean():.3f}  range=[{int_eng.min():.3f},{int_eng.max():.3f}]")

# ── ENGINEERED + LLM BLEND ───────────────────────────────────────────────────
print("\n" + "─" * 60)
print("ENGINEERED + LLM BLEND — 50/50 prior, confidence-adjusted")
print("─" * 60)


def engineered_plus_llm(dim, eng_vals):
    final_scores = np.zeros(len(eng_vals))
    metadata = []

    for g in range(len(eng_vals)):
        eng = float(eng_vals[g])
        ok_scores = []

        for m in MODELS:
            s_col = f"{m}_success"
            d_col = f"{m}_{dim}"

            if s_col not in eval_df.columns or d_col not in eval_df.columns:
                continue

            if eval_df.iloc[g][s_col] and pd.notna(eval_df.iloc[g][d_col]):
                ok_scores.append(float(eval_df.iloc[g][d_col]))

        if len(ok_scores) == 0:
            final_scores[g] = eng
            metadata.append({
                "engineered_weight": 1.0,
                "llm_weight": 0.0,
                "variance": 0.0,
                "n_models": 0,
                "fallback": True,
            })
            continue

        arr = np.array(ok_scores)
        llm_mean = float(arr.mean())
        variance = float(arr.var())

        llm_conf = 1.0 / (1.0 + variance * 10.0)
        agreement = 1.0 / (1.0 + abs(llm_mean - eng) * 5.0)

        w_eng = 0.5 * agreement
        w_llm = 0.5 * llm_conf
        tot = w_eng + w_llm
        w_eng /= tot
        w_llm /= tot

        final_scores[g] = float(np.clip(w_eng * eng + w_llm * llm_mean, 0, 1))
        metadata.append({
            "engineered_weight": round(w_eng, 3),
            "llm_weight": round(w_llm, 3),
            "variance": round(variance, 4),
            "llm_mean": round(llm_mean, 3),
            "n_models": len(ok_scores),
            "fallback": False,
        })

    vars_ = [m["variance"] for m in metadata if not m["fallback"]]
    avg_ew = np.mean([m["engineered_weight"] for m in metadata])
    agree = (
        "good" if vars_ and np.mean(vars_) < 0.02 else
        "moderate" if vars_ and np.mean(vars_) < 0.05 else
        "low"
    )

    print(f"\n  {dim.upper()}")
    print(
        f"    Final mean           : {final_scores.mean():.3f}  "
        f"range=[{final_scores.min():.3f},{final_scores.max():.3f}]"
    )
    print(f"    Avg engineered weight: {avg_ew:.3f}  LLM weight: {1 - avg_ew:.3f}")
    if vars_:
        print(f"    LLM variance         : {np.mean(vars_):.4f}  ({agree} agreement)")

    return final_scores, metadata


relevance_scores, relevance_meta = engineered_plus_llm("relevance", rel_eng)
coherence_scores, coherence_meta = engineered_plus_llm("coherence", coh_eng)
integrity_scores, integrity_meta = engineered_plus_llm("integrity", int_eng)

coherence_composite = (
    0.25 * coherence_scores
    + 0.25 * attainability_final
    + 0.25 * relevance_scores
    + 0.25 * integrity_scores
)

print(f"\n{'─' * 60}")
print("COMPOSITE COHERENCE SCORE")
print("  Weights: Coherence=0.25  Attainability=0.25  Relevance=0.25  Integrity=0.25")
print(
    f"  Mean={coherence_composite.mean():.3f}  "
    f"range=[{coherence_composite.min():.3f},{coherence_composite.max():.3f}]"
)

# ── Save eval outputs ─────────────────────────────────────────────────────────
ensemble_meta_out = [
    {
        "attainability": blend_meta[g],
        "relevance": relevance_meta[g],
        "coherence": coherence_meta[g],
        "integrity": integrity_meta[g],
    }
    for g in range(len(eval_df))
]

pd.DataFrame([{
    "model": "CoherenceEngine_v9",
    "gkf_mae": round(gkf_mae, 4),
    "gkf_rmse": round(gkf_rmse, 4),
    "gkf_r2": round(gkf_r2, 4),
    "eval_mae": round(eval_mae, 4),
    "n_train": len(X_full),
    "n_eval": len(eval_df),
    "eval_periods": ",".join(map(str, EVAL_PERIODS)),
    "n_features": len(feat_cols),
    "uncertain_goals": uncertain_n,
    "baseline_mae": round(baseline_mae, 4),
}]).to_csv("meta_learner_results_poc.csv", index=False)

out = pd.DataFrame({
    "goal_id": eval_df["goal_id"].values,
    "period_id": eval_df["period_id"].values,
    "goal_idx": eval_df["goal_idx"].values if "goal_idx" in eval_df.columns else np.arange(len(eval_df)),
    "attainability": attainability_final,
    "relevance": relevance_scores,
    "coherence": coherence_scores,
    "integrity": integrity_scores,
    "overall": coherence_composite,
    "coherence_composite": coherence_composite,
    "y_actual_attain": y_eval,
    "residual_attain": y_eval - attainability_final,
    "gp_mean": gp_mean,
    "gp_std": gp_std,
    "uncertain": gp_std > GP_UNCERTAINTY_FLAG,
    "ensemble_meta": [json.dumps(m) for m in ensemble_meta_out],
})

for m in ATTAIN_MODELS:
    col = f"{m}_attainability"
    if col in eval_df.columns:
        out[f"llm_{m}_raw"] = eval_df[col].values
        out[f"llm_{m}_calibrated"] = calibrated_llm.get(m, [None] * len(eval_df))

out.to_csv("meta_learner_predictions_poc.csv", index=False)

# ── Save full 840-row predictions for dashboard ──────────────────────────────
print("\nBuilding full 24-period prediction table...")
full_preds = features_full[["goal_id", "period_id"] + feat_cols].copy()
full_preds["gp_mean"] = gp_mean_all
full_preds["gp_std"] = gp_std_all
full_preds["uncertain"] = gp_std_all > GP_UNCERTAINTY_FLAG

for m in ATTAIN_MODELS:
    col = f"{m}_attainability"
    if col in llm_preds.columns and "goal_id" in llm_preds.columns:
        llm_by_period = llm_preds[["goal_id", "period_id", col]].copy()
        llm_by_period.columns = ["goal_id", "period_id", f"llm_{m}"]
        full_preds = full_preds.merge(llm_by_period, on=["goal_id", "period_id"], how="left")

full_preds["attainability"] = np.clip(full_preds["gp_mean"], 0, 1)

# Blend attainability with LLM at snapshot periods
for period in [6, 12, 18]:
    period_out = out[out["period_id"] == period]
    for _, row in period_out.iterrows():
        mask = (full_preds["goal_id"] == row["goal_id"]) & (full_preds["period_id"] == period)
        if mask.any():
            full_preds.loc[mask, "attainability"] = float(row["attainability"])

# Per-period R/C/I for all 840 rows — THE KEY FIX for flat trajectories.
# Previous code copied snapshot scores to all periods → zero temporal variation.
# Now: engineered scoring runs on each row's actual features, then overridden
# at p6/p12/p18 with the richer LLM-blended scores from eval subset.
print("  Computing per-period R/C/I for all 840 rows from actual features...")
full_rel_eng = engineered_relevance(features_full)
full_coh_eng = engineered_coherence(features_full)
full_int_eng = engineered_integrity(features_full)

full_preds["relevance"] = np.clip(full_rel_eng, 0, 1)
full_preds["coherence"] = np.clip(full_coh_eng, 0, 1)
full_preds["integrity"] = np.clip(full_int_eng, 0, 1)

# Override at snapshot periods with LLM-blended scores
for period in [6, 12, 18]:
    period_out = out[out["period_id"] == period]
    for _, row in period_out.iterrows():
        mask = (full_preds["goal_id"] == row["goal_id"]) & (full_preds["period_id"] == period)
        if mask.any():
            full_preds.loc[mask, "relevance"] = float(row["relevance"])
            full_preds.loc[mask, "coherence"] = float(row["coherence"])
            full_preds.loc[mask, "integrity"] = float(row["integrity"])

print("  Per-goal std across 24 periods (should be > 0.01 for real temporal variation):")
for dim in ["attainability", "coherence", "relevance", "integrity"]:
    std = full_preds.groupby("goal_id")[dim].std().mean()
    print(f"    {dim:<16}: {std:.4f}")

full_preds["overall"] = (
    full_preds["coherence"] * 0.25
    + full_preds["attainability"] * 0.25
    + full_preds["relevance"] * 0.25
    + full_preds["integrity"] * 0.25
)

full_preds.to_csv("meta_learner_predictions_full.csv", index=False)
print(f"✓ meta_learner_predictions_full.csv  ({full_preds.shape})")

with open("gp_poc.pkl", "wb") as f:
    pickle.dump(gp, f)

with open("platt_scalers_poc.pkl", "wb") as f:
    pickle.dump(iso_scalers, f)

with open("gp_config_poc.json", "w") as f:
    json.dump({
        "method": "CoherenceEngine_v9",
        "kernel": str(gp.kernel_),
        "gp_uncertainty_scale": GP_UNCERTAINTY_SCALE,
        "gp_uncertainty_flag": GP_UNCERTAINTY_FLAG,
        "models": MODELS,
        "attain_models": ATTAIN_MODELS,
        "dims_blend": DIMS_BLEND,
        "dim_gp": "attainability",
        "feature_names": feat_cols,
        "eval_periods": EVAL_PERIODS,
        "gkf_r2": round(gkf_r2, 4),
        "gkf_mae": round(gkf_mae, 4),
    }, f, indent=2)

print(f"\n✓ meta_learner_results_poc.csv")
print(f"✓ meta_learner_predictions_poc.csv")
print(f"✓ gp_poc.pkl")
print(f"✓ platt_scalers_poc.pkl  (isotonic)")
print(f"✓ gp_config_poc.json")
print("=" * 70)