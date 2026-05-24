"""
explanations.py  v7
GP uncertainty analysis + per-goal ensemble transparency + score plots.
Perturbation importance computed at the anchor period (p18).
"""

import os
import json
import pickle

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANCHOR_PERIOD = 18

print("=" * 70)
print("EXPLANATIONS + VISUALISATIONS")
print("=" * 70)

preds   = pd.read_csv("meta_learner_predictions_poc.csv")
results = pd.read_csv("meta_learner_results_poc.csv")

with open("gp_config_poc.json") as f:
    cfg = json.load(f)

MODELS = cfg["models"]
DIMS   = ["attainability", "relevance", "coherence", "integrity"]

print(f"\nModel        : {cfg['method']}")
print(f"GP kernel    : {cfg.get('kernels', {}).get('attainability', cfg.get('kernel', 'N/A'))}")
print(f"GroupKFold R2: {cfg['gkf_r2']:.4f}  MAE: {cfg['gkf_mae']:.4f}")
print(f"Uncertain flag threshold: {cfg['gp_uncertainty_flag']}")

# Filter to anchor period for display (preds may contain p12 + p18)
if "period_id" in preds.columns and preds["period_id"].nunique() > 1:
    anchor_preds = preds[preds["period_id"] == ANCHOR_PERIOD].copy().reset_index(drop=True)
    print(f"\nFiltering to period {ANCHOR_PERIOD}: {len(anchor_preds)} goals")
else:
    anchor_preds = preds.copy().reset_index(drop=True)

print(f"Goals scored : {len(anchor_preds)}")
print(f"Uncertain    : {anchor_preds['uncertain'].sum()} goals  "
      f"(gp_std > {cfg['gp_uncertainty_flag']})")

print(f"\nGP std distribution (period {ANCHOR_PERIOD}):")
print(f"  Mean : {anchor_preds['gp_std'].mean():.4f}")
print(f"  Min  : {anchor_preds['gp_std'].min():.4f}")
print(f"  Max  : {anchor_preds['gp_std'].max():.4f}")

# Per-goal ensemble weight analysis
print(f"\nPer-goal ensemble weights (Attainability at p{ANCHOR_PERIOD}):")
print(f"  {'Goal':<6} {'GP mean':<10} {'GP std':<10} "
      f"{'GP weight':<12} {'LLM weight':<12} {'Final':<8} {'Uncertain'}")
for _, row in anchor_preds.iterrows():
    try:
        meta = json.loads(row["ensemble_meta"])["attainability"]
        unc  = "  *** HIGH STD ***" if meta["uncertain"] else ""
        print(f"  {int(row['goal_idx']):<6} "
              f"{meta['gp_mean']:<10.3f} "
              f"{meta['gp_std']:<10.4f} "
              f"{meta['gp_weight']:<12.3f} "
              f"{meta['llm_weight']:<12.3f} "
              f"{row['attainability']:<8.3f}"
              f"{unc}")
    except (json.JSONDecodeError, KeyError, TypeError):
        continue

# Plot 1: GP predictions with uncertainty
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(anchor_preds))
ax.scatter(x, anchor_preds["y_actual_attain"], color="#2E4057", s=50, zorder=5, label="Actual", alpha=0.9)
ax.plot(x, anchor_preds["gp_mean"], color="#4C72B0", lw=1.5, label="GP mean")
ax.fill_between(x,
                np.clip(anchor_preds["gp_mean"] - 2 * anchor_preds["gp_std"], 0, 1),
                np.clip(anchor_preds["gp_mean"] + 2 * anchor_preds["gp_std"], 0, 1),
                alpha=0.2, color="#4C72B0", label="GP +/-2 sigma")
unc_mask = anchor_preds["uncertain"].values
ax.scatter(x[unc_mask], anchor_preds["attainability"].values[unc_mask],
           color="#C44E52", s=80, zorder=6, marker="^",
           label=f"Uncertain (std>{cfg['gp_uncertainty_flag']})")
ax.set_xticks(x)
ax.set_xticklabels([f"G{int(i)}" for i in anchor_preds["goal_idx"]], fontsize=7)
ax.set_ylabel("Attainability score")
ax.set_ylim(0, 1.05)
ax.set_title(f"GP Attainability at p{ANCHOR_PERIOD}  predictions vs actual", fontsize=10)
ax.legend(fontsize=8)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("gp_uncertainty_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK  Saved gp_uncertainty_poc.png")

# Plot 2: All 4 scores
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(anchor_preds))
w = 0.18
cols = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
for i, (dim, color) in enumerate(zip(DIMS, cols)):
    ax.bar(x + i * w, anchor_preds[dim], w, label=dim.capitalize(), color=color, alpha=0.85)

for xi in x[unc_mask]:
    ax.axvline(xi + w * 1.5, color="#C44E52", lw=0.8, ls=":", alpha=0.6)

ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.4)
ax.set_xticks(x + w * 1.5)
ax.set_xticklabels([f"G{int(i)}" for i in anchor_preds["goal_idx"]], fontsize=7)
ax.set_ylabel("Score (0-1)")
ax.set_ylim(0, 1.05)
ax.set_title(f"All 4 dimension scores per goal at p{ANCHOR_PERIOD}", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("all_scores_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK  Saved all_scores_poc.png")

# Plot 3: Calibration comparison
llm_raw_cols = [c for c in anchor_preds.columns if c.endswith("_raw")]
llm_cal_cols = [c for c in anchor_preds.columns if c.endswith("_calibrated")]

if llm_raw_cols and llm_cal_cols:
    fig, axes = plt.subplots(1, len(llm_raw_cols), figsize=(5 * len(llm_raw_cols), 4))
    if len(llm_raw_cols) == 1:
        axes = [axes]
    for ax, raw_col, cal_col in zip(axes, llm_raw_cols, llm_cal_cols):
        model_label = raw_col.replace("_raw", "").replace("llm_", "")
        raw = anchor_preds[raw_col].dropna()
        cal = anchor_preds[cal_col].dropna()
        gt  = anchor_preds["y_actual_attain"][:len(raw)]
        min_len = min(len(cal), len(gt))
        cal = cal[:min_len]
        gt  = gt[:min_len]
        ax.scatter(raw, gt, alpha=0.7, label="Raw LLM", color="#4C72B0", s=50)
        ax.scatter(cal, gt, alpha=0.7, label="Isotonic calibrated", color="#55A868", marker="^", s=50)
        lims = [0, 1]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("LLM prediction")
        ax.set_ylabel("Ground truth")
        ax.set_title(f"{model_label} calibration", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("llm_calibration_poc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("OK  Saved llm_calibration_poc.png")

# GP perturbation importance at anchor period
try:
    with open("gp_poc.pkl", "rb") as f:
        _gp_data = pickle.load(f)
    gp = _gp_data.get("attainability") if isinstance(_gp_data, dict) else _gp_data

    feat_names = cfg["feature_names"]
    X_full = pd.read_csv("features_full_normalized.csv")
    X_sp   = X_full[X_full["period_id"] == ANCHOR_PERIOD][feat_names].values

    base_preds, _ = gp.predict(X_sp, return_std=True)
    importances   = []

    for j, feat in enumerate(feat_names):
        X_pert = X_sp.copy()
        X_pert[:, j] = 0
        pert_preds, _ = gp.predict(X_pert, return_std=True)
        importance = np.abs(base_preds - pert_preds).mean()
        importances.append({"feature": feat, "importance": round(float(importance), 5)})

    imp_df = pd.DataFrame(importances).sort_values("importance", ascending=False)

    print(f"\nGP feature importance (perturbation at p{ANCHOR_PERIOD}):")
    for _, row in imp_df.head(7).iterrows():
        bar = "#" * int(row["importance"] * 200)
        print(f"  {row['feature']:<40} {row['importance']:.5f}  {bar}")

    imp_df.to_csv("shap_importance_poc.csv", index=False)
    print("OK  Saved shap_importance_poc.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    top = imp_df.head(10)
    ax.barh(range(len(top)), top["importance"], color="#4C72B0", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |prediction change| when feature zeroed")
    ax.set_title(f"GP feature importance  perturbation at p{ANCHOR_PERIOD}", fontsize=10)
    plt.tight_layout()
    plt.savefig("shap_summary_poc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("OK  Saved shap_summary_poc.png")

except Exception as e:
    print(f"  GP importance skipped: {e}")

print("=" * 70)
