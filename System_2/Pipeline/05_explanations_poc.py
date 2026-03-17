"""
05_explanations_poc.py — SHAP for attainability + all 4 score summary
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("SHAP EXPLANATIONS — all 4 dimensions")
print("=" * 70)

X_normalized    = pd.read_csv("features_normalized_poc.csv")
period_12       = pd.read_csv("period_12_poc.csv")
llm_predictions = pd.read_csv("llm_predictions_poc.csv")
preds_df        = pd.read_csv("meta_learner_predictions_poc.csv")

valid_mask  = llm_predictions["success"].astype(bool)
valid_idx   = preds_df["goal_idx"].astype(int).values
y_attain    = period_12["probability_of_hitting_target"].values[valid_idx]
llm_valid   = llm_predictions[valid_mask].reset_index(drop=True)

X_subset    = X_normalized.iloc[valid_idx].copy().reset_index(drop=True)
llm_attain  = llm_valid["llm_attainability"].values
llm_scaler  = StandardScaler()
llm_scaled  = llm_scaler.fit_transform(llm_attain.reshape(-1, 1)).flatten()
X_meta      = X_subset.copy()
X_meta["llm_attainability"] = llm_scaled

use_ridge  = os.environ.get("POC_MODEL", "elastic").lower() == "ridge"
model_name = "Ridge" if use_ridge else "ElasticNet"
model      = Ridge(alpha=1.0) if use_ridge else ElasticNet(alpha=0.005, l1_ratio=0.5,
                                                            max_iter=5000, random_state=42)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_meta, y_attain)

print(f"\nRefitted {model_name} on {len(X_meta)} samples, {X_meta.shape[1]} features")

# ── Manual linear SHAP ────────────────────────────────────────────────────────
X_arr       = X_meta.values
X_mean      = X_arr.mean(axis=0)
coef        = model.coef_
shap_values = (X_arr - X_mean) * coef
base_value  = float(model.predict(X_mean.reshape(1, -1))[0])

max_err = np.abs((base_value + shap_values.sum(axis=1)) -
                  np.clip(model.predict(X_meta), 0, 1)).max()
print(f"SHAP recon error    : {max_err:.2e}  {'✓' if max_err < 1e-6 else '⚠'}")

shap_importance = pd.DataFrame({
    "feature"       : list(X_meta.columns),
    "mean_abs_shap" : np.abs(shap_values).mean(axis=0),
    "coef"          : coef,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

llm_row  = shap_importance[shap_importance["feature"] == "llm_attainability"]
llm_rank = llm_row.index[0] + 1 if len(llm_row) else "N/A"
llm_shap = llm_row["mean_abs_shap"].iloc[0] if len(llm_row) else 0
print(f"LLM rank: #{llm_rank}  |SHAP|={llm_shap:.6f}  "
      f"{'ACTIVE' if llm_shap > 1e-6 else 'zeroed'}")

# ── Plot 1: SHAP beeswarm ─────────────────────────────────────────────────────
sorted_idx = shap_importance.index.tolist()
fig, ax    = plt.subplots(figsize=(9, 5))
for rank, feat_i in enumerate(sorted_idx):
    sv     = shap_values[:, feat_i]
    fv     = X_arr[:, feat_i]
    rng    = fv.max() - fv.min()
    norm   = (fv - fv.min()) / (rng if rng > 0 else 1)
    ax.scatter(sv, [rank] * len(sv), c=plt.cm.RdBu_r(norm), s=60, alpha=0.8, zorder=3)

ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_yticks(range(len(X_meta.columns)))
ax.set_yticklabels([shap_importance.loc[i, "feature"]
                    for i in range(len(X_meta.columns))], fontsize=9)
ax.set_xlabel("SHAP value (impact on attainability)")
ax.set_title(f"Feature impact — {model_name} attainability\n(red=high, blue=low)", fontsize=10)
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02).set_label("Feature value", fontsize=8)
plt.tight_layout()
plt.savefig("shap_summary_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved shap_summary_poc.png")

# ── Plot 2: Waterfall Goal 0 ──────────────────────────────────────────────────
sv0    = shap_values[0]
order  = np.argsort(np.abs(sv0))[::-1][:10]
labels = [list(X_meta.columns)[i] for i in order]
vals   = sv0[order]
running = base_value
lefts   = []
for v in vals:
    lefts.append(running if v >= 0 else running + v)
    running += v

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(len(vals)), np.abs(vals), left=lefts,
        color=["#C44E52" if v >= 0 else "#4C72B0" for v in vals], alpha=0.8)
ax.axvline(base_value, color="gray", lw=1, ls="--", label=f"Base: {base_value:.3f}")
ax.axvline(preds_df["attainability"].iloc[0], color="black", lw=1.5,
           label=f"Predicted: {preds_df['attainability'].iloc[0]:.3f}")
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Cumulative SHAP")
ax.set_title(f"Goal {valid_idx[0]} — {model_name} waterfall\n"
             f"Actual: {preds_df['y_actual_attain'].iloc[0]:.2f}  "
             f"Predicted: {preds_df['attainability'].iloc[0]:.3f}", fontsize=10)
ax.legend(fontsize=8)
for i, (v, left) in enumerate(zip(vals, lefts)):
    ax.text(left + abs(v)/2, i, f"{'+'if v>=0 else ''}{v:.3f}",
            va="center", ha="center", fontsize=7, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_waterfall_goal0_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved shap_waterfall_goal0_poc.png")

# ── Plot 3: All 4 scores per goal ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x       = np.arange(len(preds_df))
width   = 0.2
dims    = ["attainability", "relevance", "coherence", "integrity"]
colors  = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

for i, (dim, color) in enumerate(zip(dims, colors)):
    ax.bar(x + i * width, preds_df[dim], width, label=dim.capitalize(), color=color, alpha=0.8)

ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="0.5 threshold")
ax.set_xlabel("Goal index")
ax.set_ylabel("Score (0-1)")
ax.set_title(f"All 4 dimension scores per goal — {model_name}", fontsize=10)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"G{int(i)}" for i in preds_df["goal_idx"]], fontsize=8)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("all_scores_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved all_scores_poc.png")

shap_importance.to_csv("shap_importance_poc.csv", index=False)
print("✓ Saved shap_importance_poc.csv")
print("=" * 70)
