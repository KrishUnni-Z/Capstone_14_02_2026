"""
04_meta_learner_poc.py — ElasticNet, all 4 dimensions
Architecture:
  Attainability → ElasticNet meta-learner (ML, has ground truth)
  Relevance     → 0.6 * rule_score + 0.4 * llm_score  (no ground truth)
  Coherence     → 0.6 * rule_score + 0.4 * llm_score
  Integrity     → 0.6 * rule_score + 0.4 * llm_score
  Overall       → equal-weighted mean of all 4
"""

import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

MODEL_TYPE = "ElasticNet"

print("=" * 70)
print(f"META-LEARNER ({MODEL_TYPE}) — all 4 dimensions")
print("=" * 70)

X_normalized    = pd.read_csv("features_normalized_poc.csv")
period_12       = pd.read_csv("period_12_poc.csv")
llm_predictions = pd.read_csv("llm_predictions_poc.csv")
rule_scores     = pd.read_csv("rule_scores_poc.csv")

valid_mask = llm_predictions["success"].astype(bool)
n_valid    = valid_mask.sum()
n_failed   = (~valid_mask).sum()

print(f"\nLLM predictions : {n_valid} valid / {n_failed} excluded")
if n_valid < 5:
    print("ERROR: Fewer than 5 valid LLM predictions.")
    raise SystemExit(1)

valid_idx   = llm_predictions[valid_mask]["goal_idx"].astype(int).values
X_subset    = X_normalized.iloc[valid_idx].copy().reset_index(drop=True)
rule_subset = rule_scores.iloc[valid_idx].copy().reset_index(drop=True)
llm_valid   = llm_predictions[valid_mask].reset_index(drop=True)

y_attain = period_12["probability_of_hitting_target"].values[valid_idx]

# ── ATTAINABILITY — ElasticNet meta-learner ───────────────────────────────────
llm_attain = llm_valid["llm_attainability"].values
llm_scaler = StandardScaler()
llm_scaled = llm_scaler.fit_transform(llm_attain.reshape(-1, 1)).flatten()

X_meta = X_subset.copy()
X_meta["llm_attainability"] = llm_scaled

model = ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=5000, random_state=42)

print("\n--- Attainability (ElasticNet) ---")
loo = LeaveOneOut()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    loo_mae = cross_val_score(model, X_meta, y_attain, cv=loo,
                              scoring="neg_mean_absolute_error")
    loo_mse = cross_val_score(model, X_meta, y_attain, cv=loo,
                              scoring="neg_mean_squared_error")
    loo_r2  = cross_val_score(model, X_meta, y_attain, cv=loo, scoring="r2")
    r2_mean = float(np.nanmean(loo_r2))

print(f"  LOO MAE  : {(-loo_mae).mean():.4f}")
print(f"  LOO RMSE : {np.sqrt((-loo_mse).mean()):.4f}")
print(f"  LOO R²   : {'N/A' if np.isnan(r2_mean) else f'{r2_mean:.4f}'}")

model.fit(X_meta, y_attain)
attainability_pred = np.clip(model.predict(X_meta), 0, 1)

train_r2  = r2_score(y_attain, attainability_pred)
train_mae = mean_absolute_error(y_attain, attainability_pred)
print(f"  Train R² : {train_r2:.4f}  Train MAE: {train_mae:.4f}")

importance = pd.DataFrame({
    "feature"    : X_meta.columns,
    "coefficient": model.coef_,
    "abs_coef"   : np.abs(model.coef_),
}).sort_values("abs_coef", ascending=False).reset_index(drop=True)

llm_rank = importance[importance["feature"] == "llm_attainability"].index
llm_rank = llm_rank[0] + 1 if len(llm_rank) else "zeroed"
print(f"  LLM rank : #{llm_rank} of {len(X_meta.columns)} features")

# ── OTHER 3 — rule + LLM blend (no ground truth → no ML) ─────────────────────
BLEND_RULE = 0.6
BLEND_LLM  = 0.4

def blend_score(rule_col, llm_col, name):
    rule = rule_subset[rule_col].values
    llm  = llm_valid[llm_col].values
    score = np.clip(BLEND_RULE * rule + BLEND_LLM * llm, 0, 1)
    print(f"\n--- {name} (rule×{BLEND_RULE} + LLM×{BLEND_LLM}) ---")
    print(f"  Rule:  {rule.min():.3f} - {rule.max():.3f}  mean={rule.mean():.3f}")
    print(f"  LLM:   {llm.min():.3f}  - {llm.max():.3f}   mean={llm.mean():.3f}")
    print(f"  Final: {score.min():.3f} - {score.max():.3f}  mean={score.mean():.3f}")
    return score

relevance_score  = blend_score("relevance_rule",  "llm_relevance",  "Relevance")
coherence_score  = blend_score("coherence_rule",  "llm_coherence",  "Coherence")
integrity_score  = blend_score("integrity_rule",  "llm_integrity",  "Integrity")

# ── Overall score ─────────────────────────────────────────────────────────────
overall_score = np.mean([
    attainability_pred,
    relevance_score,
    coherence_score,
    integrity_score,
], axis=0)

print(f"\n--- Overall (equal-weighted mean of 4) ---")
print(f"  Range: {overall_score.min():.3f} - {overall_score.max():.3f}  mean={overall_score.mean():.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame([{
    "model"     : MODEL_TYPE,
    "loo_mae"   : round(float((-loo_mae).mean()), 4),
    "loo_rmse"  : round(float(np.sqrt((-loo_mse).mean())), 4),
    "loo_r2"    : round(r2_mean, 4) if not np.isnan(r2_mean) else None,
    "train_r2"  : round(train_r2, 4),
    "train_mae" : round(train_mae, 4),
    "n_samples" : n_valid,
    "n_features": X_meta.shape[1],
}]).to_csv("meta_learner_results_poc.csv", index=False)

pd.DataFrame({
    "goal_idx"         : valid_idx,
    "goal_id"          : rule_subset["goal_id"].values,
    "attainability"    : attainability_pred,
    "relevance"        : relevance_score,
    "coherence"        : coherence_score,
    "integrity"        : integrity_score,
    "overall"          : overall_score,
    "y_actual_attain"  : y_attain,
    "residual_attain"  : y_attain - attainability_pred,
    "llm_attainability": llm_attain,
}).to_csv("meta_learner_predictions_poc.csv", index=False)

importance.to_csv("feature_importance_poc.csv", index=False)

with open("llm_scaler_poc.pkl", "wb") as f:
    pickle.dump(llm_scaler, f)

print("\n✓ Saved meta_learner_results_poc.csv")
print("✓ Saved meta_learner_predictions_poc.csv")
print("✓ Saved feature_importance_poc.csv")
print("=" * 70)
