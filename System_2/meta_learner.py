"""
04_meta_learner_poc.py — v9 Coherence Engine
GP + Isotonic Calibration + Engineered Scores + Dynamic Ensemble

Architecture:
  ATTAINABILITY  → GP residual (840 rows) + Isotonic LLM calibration + uncertainty blend
  RELEVANCE      → Engineered score (optimal band + sibling rank + fitness) + LLM blend
  COHERENCE      → Engineered score (hierarchy gap + temporal drift + weighted status) + LLM blend
  INTEGRITY      → Engineered score (efficiency + needle move + quality) + LLM blend
  COMPOSITE      → Weighted mean (Coherence 35%, Attainability 25%, Relevance 20%, Integrity 20%)
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

GP_UNCERTAINTY_SCALE = 50.0
GP_UNCERTAINTY_FLAG  = 0.02
# Auto-detect models from llm_predictions_poc.csv — works with both 2 and 3 model runs
# Override here if needed: MODELS = ["llama3", "gemma3"]
MODELS        = None   # set after loading llm_preds
ATTAIN_MODELS = None   # set after loading llm_preds
DIMS_BLEND    = ["relevance", "coherence", "integrity"]

print("=" * 70)
print("META-LEARNER v9 — Coherence Engine (GP + Engineered + LLM Reasoning)")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
features_full = pd.read_csv("features_full_normalized.csv")
features_raw  = pd.read_csv("features_raw_poc.csv")   # has all engineered cols
period_12     = pd.read_csv("period_12_poc.csv")
llm_preds     = pd.read_csv("llm_predictions_poc.csv")
# Check if multi-period predictions exist
import os as _os
has_multi_period = "period_id" in llm_preds.columns and llm_preds["period_id"].nunique() > 1
if has_multi_period:
    print(f"  Multi-period LLM predictions detected: {sorted(llm_preds['period_id'].unique())}")
else:
    llm_preds["period_id"] = 12
    print("  Single-period LLM predictions (period 12)")
rule_scores   = pd.read_csv("rule_scores_poc.csv")

# Auto-detect which models are in this run
MODELS = [
    col.replace("_attainability", "")
    for col in llm_preds.columns
    if col.endswith("_attainability")
    and col.replace("_attainability", "_success") in llm_preds.columns
]
ATTAIN_MODELS = MODELS.copy()
print(f"  Auto-detected models: {MODELS}")

goal_ids  = features_full["goal_id"].values
y_full    = features_full["y_attain"].values
feat_cols = [c for c in features_full.columns
             if c not in ["goal_id", "period_id", "y_attain"]]
X_full    = features_full[feat_cols].values

# Use goal_id merge for safe indexing (not positional iloc)
# For GP training alignment, use period 12 predictions only
llm_p12     = llm_preds[llm_preds["period_id"] == 12].reset_index(drop=True)               if "period_id" in llm_preds.columns else llm_preds
valid_mask  = llm_p12["success"].astype(bool)
n_valid     = valid_mask.sum()
llm_valid   = llm_p12[valid_mask].reset_index(drop=True)
valid_goal_ids = llm_valid["goal_idx"].astype(int).values   # 0-based row indices

# Align all per-goal tables by positional index (0-34, matches period_12 row order)
valid_idx   = valid_goal_ids
rule_subset = rule_scores.iloc[valid_idx].copy().reset_index(drop=True)
feat_subset = features_raw.iloc[valid_idx].copy().reset_index(drop=True)  # FIX: engineered cols
p12_valid   = period_12.iloc[valid_idx].copy().reset_index(drop=True)
y_p12       = period_12["probability_of_hitting_target"].values[valid_idx]

print(f"\nGP training rows  : {len(X_full)}")
print(f"Features          : {len(feat_cols)}")
print(f"Period 12 goals   : {n_valid}")
print(f"\nGround truth: mean={y_p12.mean():.3f}  std={y_p12.std():.3f}  "
      f"range=[{y_p12.min():.3f}, {y_p12.max():.3f}]")
print(f"\nModel availability:")
for m in MODELS:
    col = f"{m}_success"
    n_p12 = int(llm_valid[col].sum()) if col in llm_valid.columns else 0
    n_all = int(llm_preds[col].sum()) if col in llm_preds.columns else 0
    print(f"  {m:<10}: {n_p12}/{n_valid} at p12  |  {n_all}/{len(llm_preds)} across all periods")

# Multi-period calibration data — use all 4 snapshot periods
df_gt = pd.read_csv("analytical_flat.csv")[["goal_id","period_id","probability_of_hitting_target"]].copy()
df_gt.columns = ["goal_id","period_id","y_true"]

# Add goal_id to llm_preds if missing — map from goal_idx via period_12
if "goal_id" not in llm_preds.columns and "goal_idx" in llm_preds.columns:
    goal_id_map = dict(enumerate(period_12["goal_id"].values))
    llm_preds["goal_id"] = llm_preds["goal_idx"].map(goal_id_map)
    print(f"  Mapped goal_idx → goal_id in llm_preds")

if "goal_id" in llm_preds.columns:
    llm_calibration_df = llm_preds.merge(df_gt, on=["goal_id","period_id"], how="inner")
    print(f"  Multi-period calibration: {len(llm_calibration_df)} rows across {llm_preds['period_id'].nunique()} periods")
else:
    llm_calibration_df = llm_valid.copy()
    llm_calibration_df["y_true"] = y_p12
    print(f"  Fallback: period 12 calibration only ({len(llm_calibration_df)} rows)")

# ── All-period feature matrix ────────────────────────────────────────────────
X_all_periods = features_full[feat_cols].values   # 840 rows
X_p12_all     = features_full[features_full["period_id"] == 12][feat_cols].values
X_p12_valid   = X_p12_all[valid_idx]

# ── Residual baseline ─────────────────────────────────────────────────────────
def compute_baseline_arr(df):
    obs       = df["observed_value"].clip(lower=0)
    slope     = df["trailing_6_period_slope"]
    target    = df["target_value_final_period"].clip(lower=1e-9)
    remaining = 24 - df["period_id"]
    return ((obs + slope * remaining) / target).clip(0, 1).values

df_full        = pd.read_csv("analytical_flat.csv")
baseline_full  = compute_baseline_arr(df_full)
y_residual     = y_full - baseline_full

baseline_p12   = ((p12_valid["observed_value"] + p12_valid["trailing_6_period_slope"] * 12) /
                  p12_valid["target_value_final_period"].clip(lower=1e-9)).clip(0, 1).values

print(f"\nResidual target: mean={y_residual.mean():.3f}  std={y_residual.std():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN PROCESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("GAUSSIAN PROCESS — 840 rows, residual target")
print("─" * 60)

kernel = (
    ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0)) *
    Matern(length_scale=1.0, length_scale_bounds=(0.5, 20.0), nu=1.5) +
    WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 0.5))
)
gp = GaussianProcessRegressor(
    kernel=kernel, alpha=1e-6, normalize_y=True,
    n_restarts_optimizer=10, random_state=42,
)

print("Step 1: Fitting GP on 840 rows...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp.fit(X_full, y_residual)
print(f"Fitted kernel: {gp.kernel_}")

print("\nStep 2: LOO-CV on 35 period-12 goals...")
y_resid_p12     = y_p12 - baseline_p12
loo_resid_preds = np.zeros(len(y_p12))

for i in range(len(y_p12)):
    train_mask = np.ones(len(y_p12), dtype=bool)
    train_mask[i] = False
    gp_loo = GaussianProcessRegressor(
        kernel=gp.kernel_, alpha=1e-6, normalize_y=True,
        optimizer=None, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_loo.fit(X_p12_valid[train_mask], y_resid_p12[train_mask])
        loo_resid_preds[i] = gp_loo.predict(X_p12_valid[[i]])[0]

loo_final    = np.clip(baseline_p12 + loo_resid_preds, 0, 1)
gkf_mae      = mean_absolute_error(y_p12, loo_final)
gkf_rmse     = np.sqrt(mean_squared_error(y_p12, loo_final))
gkf_r2       = r2_score(y_p12, loo_final)
baseline_mae = mean_absolute_error(y_p12, np.clip(baseline_p12, 0, 1))

print(f"\nLOO-CV (35 goals, period 12):")
print(f"  Baseline MAE: {baseline_mae:.4f}")
print(f"  GP+Baseline : {gkf_mae:.4f}  ({'better' if gkf_mae < baseline_mae else 'worse'})")
print(f"  RMSE        : {gkf_rmse:.4f}")
print(f"  R2          : {gkf_r2:.4f}  {'OK' if gkf_r2 > 0 else 'NEGATIVE'}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp_resid_pred, gp_std = gp.predict(X_p12_valid, return_std=True)

gp_mean = np.clip(baseline_p12 + gp_resid_pred, 0, 1)
print(f"\nGP period 12: mean={gp_mean.mean():.3f}  "
      f"std=[{gp_std.min():.4f},{gp_std.max():.4f}]  "
      f"uncertain={( gp_std > GP_UNCERTAINTY_FLAG).sum()}/{len(gp_std)}")

# GP predictions for all 840 rows (now that gp is trained)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp_mean_all_raw, gp_std_all = gp.predict(X_all_periods, return_std=True)
baseline_all = compute_baseline_arr(df_full)
gp_mean_all  = np.clip(baseline_all + gp_mean_all_raw, 0, 1)
gp_std_all   = np.clip(gp_std_all, 0, 1)
print(f"GP all 840 rows: mean={gp_mean_all.mean():.3f}  "
      f"uncertain={(gp_std_all > GP_UNCERTAINTY_FLAG).sum()}/840")

# ══════════════════════════════════════════════════════════════════════════════
# ISOTONIC CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("ISOTONIC CALIBRATION — continuous ground truth")
print("─" * 60)

iso_scalers    = {}
calibrated_llm = {}

for m in ATTAIN_MODELS:
    col    = f"{m}_attainability"
    ok_col = f"{m}_success"

    # Use multi-period calibration data if available
    if col in llm_calibration_df.columns:
        ok_mask_cal = llm_calibration_df[ok_col].values.astype(bool)                       if ok_col in llm_calibration_df.columns                       else np.ones(len(llm_calibration_df), dtype=bool)
        X_cal_full  = llm_calibration_df[col].values[ok_mask_cal]
        y_cal_full  = llm_calibration_df["y_true"].values[ok_mask_cal]
        n_cal       = ok_mask_cal.sum()
    elif col in llm_valid.columns:
        ok_mask_cal = llm_valid[ok_col].values.astype(bool)                       if ok_col in llm_valid.columns                       else np.ones(len(llm_valid), dtype=bool)
        X_cal_full  = llm_valid[col].values[ok_mask_cal]
        y_cal_full  = y_p12[ok_mask_cal]
        n_cal       = ok_mask_cal.sum()
    else:
        continue

    if n_cal < 5:
        calibrated_llm[m] = llm_valid[col].values if col in llm_valid.columns else None
        continue

    sort_idx = np.argsort(X_cal_full)
    iso      = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X_cal_full[sort_idx], y_cal_full[sort_idx])
    iso_scalers[m] = iso

    # Apply calibration to period 12 values for blend
    if col in llm_valid.columns:
        raw_vals_p12       = llm_valid[col].values
        ok_mask_p12        = llm_valid[ok_col].values.astype(bool)                              if ok_col in llm_valid.columns                              else np.ones(len(raw_vals_p12), dtype=bool)
        cal_vals           = np.full(len(raw_vals_p12), np.nan)
        cal_vals[ok_mask_p12]  = iso.predict(raw_vals_p12[ok_mask_p12])
        cal_vals[~ok_mask_p12] = gp_mean[~ok_mask_p12]
        calibrated_llm[m]      = cal_vals
        cal_mae = mean_absolute_error(y_cal_full, iso.predict(X_cal_full))
        print(f"  {m:<10}: raw={X_cal_full.mean():.3f}  "
              f"calibrated={cal_vals[ok_mask_p12].mean():.3f}  "
              f"gt={y_cal_full.mean():.3f}  MAE={cal_mae:.4f}  "
              f"(calibrated on {n_cal} rows, {llm_preds['period_id'].nunique()} periods)")

# ══════════════════════════════════════════════════════════════════════════════
# ATTAINABILITY BLEND
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("ATTAINABILITY BLEND — GP × confidence + LLM × (1-confidence)")
print("─" * 60)

attainability_final = np.zeros(len(valid_idx))
blend_meta          = []

for g in range(len(valid_idx)):
    gp_m    = float(gp_mean[g])
    gp_s    = float(gp_std[g])
    gp_conf = 1.0 / (1.0 + gp_s * GP_UNCERTAINTY_SCALE)
    cal_scores = [float(calibrated_llm[m][g]) for m in ATTAIN_MODELS
                  if m in calibrated_llm and pd.notna(calibrated_llm[m][g])]

    if not cal_scores:
        attainability_final[g] = float(np.clip(gp_m, 0, 1))
        blend_meta.append({"gp_mean":round(gp_m,4),"gp_std":round(gp_s,4),
                           "gp_weight":1.0,"llm_weight":0.0,"llm_mean":None,
                           "uncertain":gp_s>GP_UNCERTAINTY_FLAG,"fallback":"no_llm"})
        continue

    llm_mean = float(np.mean(cal_scores))
    llm_w    = 1.0 - gp_conf
    final    = float(np.clip(gp_conf * gp_m + llm_w * llm_mean, 0, 1))
    attainability_final[g] = final
    blend_meta.append({"gp_mean":round(gp_m,4),"gp_std":round(gp_s,4),
                       "gp_weight":round(gp_conf,3),"llm_weight":round(llm_w,3),
                       "llm_mean":round(llm_mean,4),
                       "uncertain":bool(gp_s>GP_UNCERTAINTY_FLAG),"fallback":False})

uncertain_n = sum(1 for m in blend_meta if m["uncertain"])
p12_mae     = mean_absolute_error(y_p12, attainability_final)
print(f"\n  Final mean    : {attainability_final.mean():.3f}  "
      f"range=[{attainability_final.min():.3f},{attainability_final.max():.3f}]")
print(f"  Period 12 MAE : {p12_mae:.4f}")
print(f"  Avg GP weight : {np.mean([m['gp_weight'] for m in blend_meta]):.3f}")
print(f"  Avg LLM weight: {np.mean([m['llm_weight'] for m in blend_meta]):.3f}")
print(f"  Uncertain     : {uncertain_n}/{len(valid_idx)}")

# ══════════════════════════════════════════════════════════════════════════════
# ENGINEERED SCORES — R / C / I
# FIX: read from features_raw_poc.csv which has all computed columns
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("ENGINEERED SCORES — reading from features_raw_poc.csv")
print("─" * 60)

def clip01(x):
    return np.clip(x, 0, 1)

# ── RELEVANCE ─────────────────────────────────────────────────────────────────
# FIX: use optimal_band_distance instead of alloc_pct/0.25
# optimal_band_distance = |actual - optimal_centre| / optimal_range  (0=perfect, 1=worst)
def engineered_relevance(df):
    fitness   = clip01(df["allocation_fitness_score"].fillna(0.0).values)
    sibling   = 1.0 - clip01(df["sibling_rank_pct"].fillna(0.5).values)   # 0=worst, 1=best
    band_dist = clip01(df["optimal_band_distance"].fillna(0.5).values)
    band_score= 1.0 - band_dist                                            # 1=in band, 0=far out
    return clip01(0.40 * band_score + 0.35 * sibling + 0.25 * fitness)

# ── COHERENCE ─────────────────────────────────────────────────────────────────
# FIX: use sibling_count-aware expected share instead of fixed 0.40
def engineered_coherence(df):
    # Drift: lower std = more coherent (normalise by max drift observed)
    drift_vals = clip01(df["alloc_drift_std"].fillna(0.0).values)
    max_drift  = drift_vals.max() if drift_vals.max() > 0 else 1.0
    drift_score= 1.0 - drift_vals / max_drift

    # Hierarchy: how close is L3 share to expected sibling share (1/n_siblings)
    # sibling_rank_pct already reflects relative position within group
    # l3_share_of_l2 is the actual share; we need to compare to expected
    # expected = 1/sibling_count — not stored directly, but sibling_rank_pct
    # encodes relative rank. Use weighted_goal_status_score as hierarchy proxy
    hier_score = clip01(df["weighted_goal_status_score"].fillna(0.5).values)

    # Status consistency (fewer band changes = more coherent)
    status_u   = df["status_band_unique"].fillna(3).values
    status_s   = 1.0 - clip01((status_u - 1) / 4.0)

    return clip01(0.40 * drift_score + 0.35 * hier_score + 0.25 * status_s)

# ── INTEGRITY ─────────────────────────────────────────────────────────────────
def engineered_integrity(df):
    efficiency = clip01(df["allocation_efficiency_ratio"].fillna(0.0).values)
    needle     = clip01(df["needle_move_ratio"].fillna(0.0).values)
    quality    = clip01(df["delivered_output_quality_score"].fillna(0.0).values)
    return clip01(0.40 * efficiency + 0.35 * needle + 0.25 * quality)

# Check which columns are available in feat_subset
for dim_name, cols_needed in [
    ("Relevance", ["allocation_fitness_score","sibling_rank_pct","optimal_band_distance"]),
    ("Coherence", ["alloc_drift_std","weighted_goal_status_score","status_band_unique"]),
    ("Integrity", ["allocation_efficiency_ratio","needle_move_ratio","delivered_output_quality_score"]),
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
# FIX: 50/50 prior — let confidence signals decide, not a hard 75/25
print("\n" + "─" * 60)
print("ENGINEERED + LLM BLEND — 50/50 prior, confidence-adjusted")
print("─" * 60)

def engineered_plus_llm(dim, eng_vals):
    final_scores = np.zeros(len(eng_vals))
    metadata     = []

    for g in range(len(eng_vals)):
        eng = float(eng_vals[g])
        ok_scores = []
        for m in MODELS:
            s_col = f"{m}_success"
            d_col = f"{m}_{dim}"
            if s_col not in llm_valid.columns or d_col not in llm_valid.columns: continue
            if llm_valid.iloc[g][s_col] and pd.notna(llm_valid.iloc[g][d_col]):
                ok_scores.append(float(llm_valid.iloc[g][d_col]))

        if len(ok_scores) == 0:
            final_scores[g] = eng
            metadata.append({"engineered_weight":1.0,"llm_weight":0.0,
                             "variance":0.0,"n_models":0,"fallback":True})
            continue

        arr      = np.array(ok_scores)
        llm_mean = float(arr.mean())
        variance = float(arr.var())

        # 50/50 prior — confidence signals adjust from there
        llm_conf  = 1.0 / (1.0 + variance * 10.0)
        agreement = 1.0 / (1.0 + abs(llm_mean - eng) * 5.0)

        w_eng = 0.5 * agreement    # FIX: was 0.75
        w_llm = 0.5 * llm_conf     # FIX: was 0.25
        tot   = w_eng + w_llm
        w_eng /= tot
        w_llm /= tot

        final_scores[g] = float(np.clip(w_eng * eng + w_llm * llm_mean, 0, 1))
        metadata.append({"engineered_weight":round(w_eng,3),"llm_weight":round(w_llm,3),
                         "variance":round(variance,4),"llm_mean":round(llm_mean,3),
                         "n_models":len(ok_scores),"fallback":False})

    vars_  = [m["variance"] for m in metadata if not m["fallback"]]
    avg_ew = np.mean([m["engineered_weight"] for m in metadata])
    agree  = "good" if vars_ and np.mean(vars_) < 0.02 else \
             "moderate" if vars_ and np.mean(vars_) < 0.05 else "low"
    print(f"\n  {dim.upper()}")
    print(f"    Final mean           : {final_scores.mean():.3f}  "
          f"range=[{final_scores.min():.3f},{final_scores.max():.3f}]")
    print(f"    Avg engineered weight: {avg_ew:.3f}  LLM weight: {1-avg_ew:.3f}")
    if vars_: print(f"    LLM variance         : {np.mean(vars_):.4f}  ({agree} agreement)")
    return final_scores, metadata

relevance_scores, relevance_meta = engineered_plus_llm("relevance", rel_eng)
coherence_scores, coherence_meta = engineered_plus_llm("coherence", coh_eng)
integrity_scores, integrity_meta = engineered_plus_llm("integrity", int_eng)

# Composite (Coherence engine — weighted)
coherence_composite = (
    0.35 * coherence_scores +
    0.25 * attainability_final +
    0.20 * relevance_scores +
    0.20 * integrity_scores
)

print(f"\n{'─'*60}")
print(f"COMPOSITE COHERENCE SCORE")
print(f"  Weights: Coherence=0.35  Attainability=0.25  Relevance=0.20  Integrity=0.20")
print(f"  Mean={coherence_composite.mean():.3f}  "
      f"range=[{coherence_composite.min():.3f},{coherence_composite.max():.3f}]")

# ── Save ──────────────────────────────────────────────────────────────────────
ensemble_meta_out = [
    {"attainability":blend_meta[g],"relevance":relevance_meta[g],
     "coherence":coherence_meta[g],"integrity":integrity_meta[g]}
    for g in range(len(valid_idx))
]

pd.DataFrame([{
    "model"          : "CoherenceEngine_v9",
    "gkf_mae"        : round(gkf_mae,4),
    "gkf_rmse"       : round(gkf_rmse,4),
    "gkf_r2"         : round(gkf_r2,4),
    "p12_mae"        : round(p12_mae,4),
    "n_train"        : len(X_full),
    "n_p12"          : n_valid,
    "n_features"     : len(feat_cols),
    "uncertain_goals": uncertain_n,
    "baseline_mae"   : round(baseline_mae,4),
}]).to_csv("meta_learner_results_poc.csv", index=False)

out = pd.DataFrame({
    "goal_idx"          : valid_idx,
    "goal_id"           : rule_subset["goal_id"].values,
    "attainability"     : attainability_final,
    "relevance"         : relevance_scores,
    "coherence"         : coherence_scores,
    "integrity"         : integrity_scores,
    "overall"           : coherence_composite,
    "coherence_composite": coherence_composite,
    "y_actual_attain"   : y_p12,
    "residual_attain"   : y_p12 - attainability_final,
    "gp_mean"           : gp_mean,
    "gp_std"            : gp_std,
    "uncertain"         : gp_std > GP_UNCERTAINTY_FLAG,
    "ensemble_meta"     : [json.dumps(m) for m in ensemble_meta_out],
})
for m in ATTAIN_MODELS:
    col = f"{m}_attainability"
    if col in llm_valid.columns:
        out[f"llm_{m}_raw"]        = llm_valid[col].values
        out[f"llm_{m}_calibrated"] = calibrated_llm.get(m, [None]*len(valid_idx))

out.to_csv("meta_learner_predictions_poc.csv", index=False)

# ── Save full 840-row predictions for dashboard ───────────────────────────────
print("\nBuilding full 24-period prediction table...")
full_preds = features_full[["goal_id","period_id"] + feat_cols].copy()
full_preds["gp_mean"] = gp_mean_all
full_preds["gp_std"]  = gp_std_all
full_preds["uncertain"] = gp_std_all > GP_UNCERTAINTY_FLAG

# Join LLM scores where available (at snapshot periods)
for m in ATTAIN_MODELS:
    col = f"{m}_attainability"
    if col in llm_preds.columns and "goal_id" in llm_preds.columns:
        llm_by_period = llm_preds[["goal_id","period_id",col]].copy()
        llm_by_period.columns = ["goal_id","period_id",f"llm_{m}"]
        # Merge on goal_id + period_id — clean, no cartesian product
        full_preds = full_preds.merge(
            llm_by_period, on=["goal_id","period_id"], how="left"
        )

# Attainability: GP mean for all periods, blended with LLM at snapshot periods
full_preds["attainability"] = np.clip(full_preds["gp_mean"], 0, 1)

# For R/C/I use period-specific engineered scores from features_full
for dim in ["relevance","coherence","integrity"]:
    if dim in features_full.columns:
        full_preds[dim] = features_full[dim].values
    else:
        # broadcast from p12 scores
        p12_vals = dict(zip(out["goal_id"], out[dim]))
        full_preds[dim] = full_preds["goal_id"].map(p12_vals)

full_preds["overall"] = (
    full_preds["coherence"]     * 0.35 +
    full_preds["attainability"] * 0.25 +
    full_preds["relevance"]     * 0.20 +
    full_preds["integrity"]     * 0.20
)

full_preds.to_csv("meta_learner_predictions_full.csv", index=False)
print(f"✓ meta_learner_predictions_full.csv  ({full_preds.shape})")

with open("gp_poc.pkl","wb")            as f: pickle.dump(gp, f)
with open("platt_scalers_poc.pkl","wb") as f: pickle.dump(iso_scalers, f)
with open("gp_config_poc.json","w")     as f:
    json.dump({
        "method"              : "CoherenceEngine_v9",
        "kernel"              : str(gp.kernel_),
        "gp_uncertainty_scale": GP_UNCERTAINTY_SCALE,
        "gp_uncertainty_flag" : GP_UNCERTAINTY_FLAG,
        "models"              : MODELS,
        "attain_models"       : ATTAIN_MODELS,
        "dims_blend"          : DIMS_BLEND,
        "dim_gp"              : "attainability",
        "feature_names"       : feat_cols,
        "gkf_r2"              : round(gkf_r2,4),
        "gkf_mae"             : round(gkf_mae,4),
    }, f, indent=2)

print(f"\n✓ meta_learner_results_poc.csv")
print(f"✓ meta_learner_predictions_poc.csv")
print(f"✓ gp_poc.pkl")
print(f"✓ platt_scalers_poc.pkl  (isotonic)")
print(f"✓ gp_config_poc.json")
print("=" * 70)
