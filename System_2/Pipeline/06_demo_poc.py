"""
06_demo_poc.py — Demo dashboard, all 4 dimensions
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

print("=" * 70)
print("DECIDR SYSTEM 2 — PoC DEMO (all 4 dimensions)")
print("=" * 70)

period_12    = pd.read_csv("period_12_poc.csv")
features_raw = pd.read_csv("features_raw_poc.csv")
llm_preds    = pd.read_csv("llm_predictions_poc.csv")
meta_results = pd.read_csv("meta_learner_results_poc.csv")
preds_df     = pd.read_csv("meta_learner_predictions_poc.csv")
feat_imp     = pd.read_csv("shap_importance_poc.csv")
rule_scores  = pd.read_csv("rule_scores_poc.csv")

model_name = meta_results["model"].iloc[0]

print(f"\n[1/4] SYSTEM PERFORMANCE ({model_name})")
print("-" * 70)
print(f"  Goals scored         : {len(preds_df)} / 35")
print(f"  LLM success rate     : {int(llm_preds['success'].sum())}/{len(llm_preds)}")
print(f"  Attainability LOO MAE: {meta_results['loo_mae'].iloc[0]:.4f}")
print(f"  Attainability LOO RMSE:{meta_results['loo_rmse'].iloc[0]:.4f}")
r2_val = meta_results['loo_r2'].iloc[0]
print(f"  Attainability LOO R² : {r2_val:.4f}" if pd.notna(r2_val) else
      f"  Attainability LOO R² : N/A (n too small)")
print(f"  Training R²          : {meta_results['train_r2'].iloc[0]:.4f}")

print(f"\n[2/4] MEAN SCORES ACROSS ALL 4 DIMENSIONS")
print("-" * 70)
for dim in ["attainability", "relevance", "coherence", "integrity", "overall"]:
    vals = preds_df[dim]
    print(f"  {dim.capitalize():<16}: mean={vals.mean():.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}")

print(f"\n[3/4] TOP FEATURES (SHAP — Attainability)")
print("-" * 70)
for i, row in feat_imp.head(5).iterrows():
    print(f"  {i+1}. {row['feature']:<40} {row['mean_abs_shap']:.4f}")

print(f"\n[4/4] EXAMPLE PREDICTIONS (first 3 goals)")
print("-" * 70)
for i, pred_row in preds_df.head(3).iterrows():
    goal_idx = int(pred_row["goal_idx"])
    goal     = features_raw.iloc[goal_idx]
    print(f"\n  Goal {goal_idx}  (ID: {int(pred_row['goal_id'])})")
    print(f"    Progress      : {goal['observed_value']:.2f} / {goal['target_value_final_period']:.2f}")
    print(f"    Gap           : {goal['variance_from_target']:.2f}")
    print(f"    Trend         : {goal['trailing_6_period_slope']:.4f}/period")
    print(f"    ┌─ Attainability : {pred_row['attainability']:.3f}  (actual={pred_row['y_actual_attain']:.2f})")
    print(f"    ├─ Relevance     : {pred_row['relevance']:.3f}")
    print(f"    ├─ Coherence     : {pred_row['coherence']:.3f}")
    print(f"    ├─ Integrity     : {pred_row['integrity']:.3f}")
    print(f"    └─ OVERALL       : {pred_row['overall']:.3f}")

# ── Dashboard ─────────────────────────────────────────────────────────────────
print("\n[5/4] GENERATING DASHBOARD ...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle(f"Decidr System 2 — All 4 Dimensions ({model_name})", fontsize=14, fontweight="bold")

dims   = ["attainability", "relevance", "coherence", "integrity"]
colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
labels = ["Attainability", "Relevance", "Coherence", "Integrity"]

# Plot 1-4: Distribution of each score
for idx, (dim, color, label) in enumerate(zip(dims, colors, labels)):
    ax = fig.add_subplot(3, 4, idx + 1)
    ax.hist(preds_df[dim], bins=6, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(preds_df[dim].mean(), color="black", lw=1.5, ls="--",
               label=f"mean={preds_df[dim].mean():.2f}")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7)

# Plot 5: Overall score per goal
ax5 = fig.add_subplot(3, 2, 3)
x   = np.arange(len(preds_df))
w   = 0.18
for i, (dim, color) in enumerate(zip(dims, colors)):
    ax5.bar(x + i * w, preds_df[dim], w, color=color, alpha=0.8, label=dim.capitalize())
ax5.bar(x + 4 * w, preds_df["overall"], w, color="black", alpha=0.6, label="Overall")
ax5.set_xticks(x + w * 2)
ax5.set_xticklabels([f"G{int(i)}" for i in preds_df["goal_idx"]], fontsize=7)
ax5.set_ylabel("Score")
ax5.set_ylim(0, 1.05)
ax5.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
ax5.set_title("All 4 scores per goal", fontsize=10)
ax5.legend(fontsize=7, ncol=3)

# Plot 6: Attainability predicted vs actual
ax6 = fig.add_subplot(3, 2, 4)
ax6.scatter(preds_df["y_actual_attain"], preds_df["attainability"],
            color="#4C72B0", s=60, alpha=0.8, zorder=3)
lo = min(preds_df["y_actual_attain"].min(), preds_df["attainability"].min()) - 0.05
hi = max(preds_df["y_actual_attain"].max(), preds_df["attainability"].max()) + 0.05
ax6.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4)
r2  = r2_score(preds_df["y_actual_attain"], preds_df["attainability"])
mae = mean_absolute_error(preds_df["y_actual_attain"], preds_df["attainability"])
ax6.text(0.05, 0.92, f"R²={r2:.3f}  MAE={mae:.3f}",
         transform=ax6.transAxes, fontsize=9)
ax6.set_title("Attainability: predicted vs actual", fontsize=10)
ax6.set_xlabel("Actual")
ax6.set_ylabel("Predicted")

# Plot 7: SHAP feature importance
ax7 = fig.add_subplot(3, 2, 5)
top8 = feat_imp.head(8)
ax7.barh(range(len(top8)), top8["mean_abs_shap"], color="#4C72B0", alpha=0.8)
ax7.set_yticks(range(len(top8)))
ax7.set_yticklabels(top8["feature"], fontsize=8)
ax7.invert_yaxis()
ax7.set_title("Feature importance (SHAP — attainability)", fontsize=10)
ax7.set_xlabel("mean |SHAP|")

# Plot 8: Radar / spider of mean scores
ax8 = fig.add_subplot(3, 2, 6, polar=True)
means  = [preds_df[d].mean() for d in dims]
angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
means  += means[:1]
angles += angles[:1]
ax8.plot(angles, means, "o-", lw=2, color="#4C72B0")
ax8.fill(angles, means, alpha=0.25, color="#4C72B0")
ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(labels, fontsize=9)
ax8.set_ylim(0, 1)
ax8.set_title("Mean scores — all 4 dimensions", fontsize=10, pad=15)

plt.tight_layout()
plt.savefig("demo_dashboard_poc.png", dpi=150, bbox_inches="tight")
print("✓ Saved demo_dashboard_poc.png")

print("\n" + "=" * 70)
print("PoC COMPLETE — ALL 4 DIMENSIONS SCORED")
print("=" * 70)
