"""
05_explanations_poc.py — v6
GP uncertainty analysis + per-goal ensemble transparency + score plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

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
print(f"GP kernel    : {cfg['kernel']}")
print(f"GroupKFold R2: {cfg['gkf_r2']:.4f}  MAE: {cfg['gkf_mae']:.4f}")
print(f"Uncertain flag threshold: {cfg['gp_uncertainty_flag']}")
print(f"\nGoals scored : {len(preds)}")
print(f"Uncertain    : {preds['uncertain'].sum()} goals  "
      f"(gp_std > {cfg['gp_uncertainty_flag']})")

# ── GP uncertainty summary ────────────────────────────────────────────────────
print(f"\nGP std distribution:")
print(f"  Mean : {preds['gp_std'].mean():.4f}")
print(f"  Min  : {preds['gp_std'].min():.4f}")
print(f"  Max  : {preds['gp_std'].max():.4f}")

# ── Ensemble weight analysis ──────────────────────────────────────────────────
print(f"\nPer-goal ensemble weights (Attainability):")
print(f"  {'Goal':<6} {'GP mean':<10} {'GP std':<10} "
      f"{'GP weight':<12} {'LLM weight':<12} {'Final':<8} {'Uncertain'}")
for _, row in preds.iterrows():
    meta = json.loads(row["ensemble_meta"])["attainability"]
    unc  = "  *** HIGH STD ***" if meta["uncertain"] else ""
    print(f"  {int(row['goal_idx']):<6} "
          f"{meta['gp_mean']:<10.3f} "
          f"{meta['gp_std']:<10.4f} "
          f"{meta['gp_weight']:<12.3f} "
          f"{meta['llm_weight']:<12.3f} "
          f"{row['attainability']:<8.3f}"
          f"{unc}")

# ── Plot 1: GP prediction vs actual with uncertainty bands ───────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(preds))
ax.scatter(x, preds["y_actual_attain"], color="#2E4057", s=50, zorder=5,
           label="Actual", alpha=0.9)
ax.plot(x, preds["gp_mean"], color="#4C72B0", lw=1.5, label="GP mean")
ax.fill_between(x,
                np.clip(preds["gp_mean"] - 2*preds["gp_std"], 0, 1),
                np.clip(preds["gp_mean"] + 2*preds["gp_std"], 0, 1),
                alpha=0.2, color="#4C72B0", label="GP ±2σ")
ax.scatter(x[preds["uncertain"].values],
           preds["attainability"].values[preds["uncertain"].values],
           color="#C44E52", s=80, zorder=6, marker="^",
           label=f"Uncertain (std>{cfg['gp_uncertainty_flag']})")
ax.set_xticks(x)
ax.set_xticklabels([f"G{int(i)}" for i in preds["goal_idx"]], fontsize=7)
ax.set_ylabel("Attainability score")
ax.set_ylim(0, 1.05)
ax.set_title("GP Attainability — predictions vs actual with uncertainty bands", fontsize=10)
ax.legend(fontsize=8)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("gp_uncertainty_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved gp_uncertainty_poc.png")

# ── Plot 2: All 4 dimension scores ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x     = np.arange(len(preds))
w     = 0.18
cols  = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
for i, (dim, color) in enumerate(zip(DIMS, cols)):
    ax.bar(x + i*w, preds[dim], w, label=dim.capitalize(),
           color=color, alpha=0.85)

# Mark uncertain goals
unc_x = x[preds["uncertain"].values]
for xi in unc_x:
    ax.axvline(xi + w*1.5, color="#C44E52", lw=0.8, ls=":", alpha=0.6)

ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.4)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels([f"G{int(i)}" for i in preds["goal_idx"]], fontsize=7)
ax.set_ylabel("Score (0-1)")
ax.set_ylim(0, 1.05)
ax.set_title("All 4 dimension scores per goal  (dotted = GP uncertain)", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("all_scores_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved all_scores_poc.png")

# ── Plot 3: Calibration comparison per LLM ───────────────────────────────────
llm_raw_cols = [c for c in preds.columns if c.endswith("_raw")]
llm_cal_cols = [c for c in preds.columns if c.endswith("_calibrated")]

if llm_raw_cols and llm_cal_cols:
    fig, axes = plt.subplots(1, len(llm_raw_cols), figsize=(5*len(llm_raw_cols), 4))
    if len(llm_raw_cols) == 1:
        axes = [axes]
    for ax, raw_col, cal_col in zip(axes, llm_raw_cols, llm_cal_cols):
        model_label = raw_col.replace("_raw", "").replace("llm_", "")
        raw = preds[raw_col].dropna()
        cal = preds[cal_col].dropna()
        gt  = preds["y_actual_attain"][:len(raw)]
        min_len = min(len(cal), len(gt))
        cal = cal[:min_len]
        gt = gt[:min_len]
        ax.scatter(raw, gt, alpha=0.7, label="Raw LLM", color="#4C72B0", s=50)
        ax.scatter(cal, gt, alpha=0.7, label="Platt calibrated", color="#55A868",
                   marker="^", s=50)
        lims = [0, 1]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("LLM prediction")
        ax.set_ylabel("Ground truth")
        ax.set_title(f"{model_label} calibration", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("llm_calibration_poc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved llm_calibration_poc.png")

# ── SHAP-style: GP feature importance via input perturbation ──────────────────
try:
    import pickle
    with open("gp_poc.pkl", "rb") as f:
        gp = pickle.load(f)

    feat_names = cfg["feature_names"]
    X_p12_full = pd.read_csv("features_full_normalized.csv")
    X_p12      = X_p12_full[X_p12_full["period_id"] == 12][feat_names].values

    base_preds, _ = gp.predict(X_p12, return_std=True)
    importances   = []

    for j, feat in enumerate(feat_names):
        X_pert    = X_p12.copy()
        X_pert[:, j] = 0  # zero out feature
        pert_preds, _ = gp.predict(X_pert, return_std=True)
        importance = np.abs(base_preds - pert_preds).mean()
        importances.append({"feature": feat, "importance": round(float(importance), 5)})

    imp_df = pd.DataFrame(importances).sort_values("importance", ascending=False)

    print(f"\nGP feature importance (perturbation — period 12):")
    for _, row in imp_df.head(7).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<40} {row['importance']:.5f}  {bar}")

    imp_df.to_csv("shap_importance_poc.csv", index=False)
    print("✓ Saved shap_importance_poc.csv")

    # Plot importance
    fig, ax = plt.subplots(figsize=(9, 5))
    top = imp_df.head(10)
    ax.barh(range(len(top)), top["importance"], color="#4C72B0", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |prediction change| when feature zeroed")
    ax.set_title("GP feature importance — perturbation analysis (period 12)", fontsize=10)
    plt.tight_layout()
    plt.savefig("shap_summary_poc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved shap_summary_poc.png")

except Exception as e:
    print(f"  GP importance skipped: {e}")

print("=" * 70)
