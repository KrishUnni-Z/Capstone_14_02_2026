"""
dashboard.py  v7  Demo dashboard
Reads meta_learner_predictions_poc.csv (filters to p18 anchor), plots summary.

Fixes a longstanding bug: column names were n_p12/p12_mae, meta_learner
actually writes n_eval/eval_mae.
"""

import os
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANCHOR_PERIOD = 18

print("=" * 70)
print("DECIDR SYSTEM 2  Demo Dashboard")
print("=" * 70)

preds   = pd.read_csv("meta_learner_predictions_poc.csv")
results = pd.read_csv("meta_learner_results_poc.csv")

composite_scores = pd.read_csv("composite_scores_poc.csv") if os.path.exists("composite_scores_poc.csv") else None
coherence_ts     = pd.read_csv("coherence_timeseries_poc.csv") if os.path.exists("coherence_timeseries_poc.csv") else None
forward_proj     = pd.read_csv("forward_projection_poc.csv") if os.path.exists("forward_projection_poc.csv") else None

with open("gp_config_poc.json") as f:
    cfg = json.load(f)

DIMS = ["attainability", "relevance", "coherence", "integrity"]

# Filter to anchor period for display
if "period_id" in preds.columns and preds["period_id"].nunique() > 1:
    preds_display = preds[preds["period_id"] == ANCHOR_PERIOD].copy().reset_index(drop=True)
else:
    preds_display = preds.copy().reset_index(drop=True)

# ── Composite summary ───────────────────────────────────────────────────────
if composite_scores is not None:
    print(f"\n[0/4] COMPOSITE COHERENCE SUMMARY (p{ANCHOR_PERIOD})")
    print("-" * 70)
    if "period_id" in composite_scores.columns:
        p_anchor = composite_scores[composite_scores["period_id"] == ANCHOR_PERIOD]
    else:
        p_anchor = composite_scores
    print(f"  Goals scored     : {len(p_anchor)}")
    print(f"  At risk          : {p_anchor['at_risk'].sum()}/{len(p_anchor)}")
    print(f"  Critical         : {p_anchor['critical'].sum()}/{len(p_anchor)}")
    print(f"  Avg composite    : {p_anchor['composite'].mean():.3f}")
    if 'composite_raw' in p_anchor.columns:
        print(f"  Avg composite raw: {p_anchor['composite_raw'].mean():.3f}")

    if coherence_ts is not None:
        cols = [c for c in ['period_id', 'avg_composite', 'at_risk_count',
                            'avg_coherence', 'shock_label'] if c in coherence_ts.columns]
        print(f"\n  Coherence over time (period 1 -> 24):")
        print(coherence_ts[cols].to_string(index=False))

# ── Performance ──────────────────────────────────────────────────────────────
print(f"\n[1/4] SYSTEM PERFORMANCE")
print("-" * 70)
print(f"  Method           : {cfg['method']}")
print(f"  Training rows    : {results['n_train'].iloc[0]}")

# Use the actual column names from meta_learner
n_eval_col = "n_eval" if "n_eval" in results.columns else "n_p12"
eval_mae_col = "eval_mae" if "eval_mae" in results.columns else "p12_mae"

if n_eval_col in results.columns:
    print(f"  Eval rows        : {results[n_eval_col].iloc[0]}")
print(f"  GroupKFold MAE   : {results['gkf_mae'].iloc[0]:.4f}")
print(f"  GroupKFold RMSE  : {results['gkf_rmse'].iloc[0]:.4f}")
print(f"  GroupKFold R2    : {results['gkf_r2'].iloc[0]:.4f}")
if eval_mae_col in results.columns:
    print(f"  Eval MAE         : {results[eval_mae_col].iloc[0]:.4f}")
if "eval_periods" in results.columns:
    print(f"  Eval periods     : {results['eval_periods'].iloc[0]}")
print(f"  Uncertain goals  : {results['uncertain_goals'].iloc[0]}")

# ── Mean scores ──────────────────────────────────────────────────────────────
print(f"\n[2/4] MEAN SCORES ACROSS ALL 4 DIMENSIONS (p{ANCHOR_PERIOD})")
print("-" * 70)
for dim in DIMS + ["overall"]:
    if dim in preds_display.columns:
        vals = preds_display[dim]
        print(f"  {dim.capitalize():<16}: mean={vals.mean():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

# ── Uncertain goals ──────────────────────────────────────────────────────────
print(f"\n[3/4] UNCERTAIN GOALS (GP std > {cfg['gp_uncertainty_flag']})")
print("-" * 70)
unc = preds_display[preds_display["uncertain"]]
if len(unc):
    for _, row in unc.iterrows():
        try:
            meta = json.loads(row["ensemble_meta"])["attainability"]
            print(f"  Goal {int(row['goal_idx']):<4} "
                  f"gp_std={meta['gp_std']:.4f}  "
                  f"gp_weight={meta['gp_weight']:.3f}  "
                  f"llm_weight={meta['llm_weight']:.3f}  "
                  f"final={row['attainability']:.3f}  "
                  f"actual={row['y_actual_attain']:.3f}")
        except Exception:
            continue
else:
    print("  No uncertain goals")

# ── Sample predictions ──────────────────────────────────────────────────────
print(f"\n[4/4] SAMPLE PREDICTIONS (first 3 goals at p{ANCHOR_PERIOD})")
print("-" * 70)
for _, row in preds_display.head(3).iterrows():
    try:
        meta = json.loads(row["ensemble_meta"])
        am = meta["attainability"]
        print(f"\n  Goal {int(row['goal_idx'])}  (ID: {int(row['goal_id'])})")
        print(f"    Attainability  : {row['attainability']:.3f}  "
              f"(actual={row['y_actual_attain']:.2f}  "
              f"gp={am['gp_mean']:.3f}+/-{am['gp_std']:.4f}  "
              f"gp_w={am['gp_weight']:.2f}  llm_w={am['llm_weight']:.2f})"
              f"{'  UNCERTAIN' if am['uncertain'] else ''}")
        for dim in ["relevance", "coherence", "integrity"]:
            dm = meta[dim]
            src = "fallback->rules" if dm["fallback"] else \
                  f"rule_w={dm.get('engineered_weight', dm.get('rule_weight', 0)):.2f} " \
                  f"llm_w={dm['llm_weight']:.2f} var={dm['variance']:.4f}"
            print(f"    {dim.capitalize():<14}: {row[dim]:.3f}  ({src})")
        print(f"    Overall        : {row['overall']:.3f}")
    except Exception as e:
        print(f"  (sample skipped: {e})")

# ── Dashboard plot ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle(f"Decidr System 2  {cfg['method']}  (anchor p{ANCHOR_PERIOD})", fontsize=13, fontweight="bold")

colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

for idx, (dim, color) in enumerate(zip(DIMS, colors)):
    ax = fig.add_subplot(3, 4, idx + 1)
    ax.hist(preds_display[dim], bins=8, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(preds_display[dim].mean(), color="black", lw=1.5, ls="--",
               label=f"mean={preds_display[dim].mean():.2f}")
    ax.set_title(dim.capitalize(), fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7)

ax5 = fig.add_subplot(3, 2, 3)
ax5.scatter(preds_display["y_actual_attain"], preds_display["attainability"],
            c=preds_display["gp_std"], cmap="RdYlGn_r", s=60, alpha=0.85, zorder=3)
ax5.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
ax5.set_xlabel("Actual Attainability")
ax5.set_ylabel("Predicted")
ax5.set_title("Attainability: predicted vs actual (colour = GP uncertainty)", fontsize=9)
sc = plt.cm.ScalarMappable(
    cmap="RdYlGn_r",
    norm=plt.Normalize(preds_display["gp_std"].min(), preds_display["gp_std"].max())
)
sc.set_array([])
plt.colorbar(sc, ax=ax5, fraction=0.046, pad=0.04).set_label("GP std", fontsize=8)

ax6 = fig.add_subplot(3, 2, 4)
x = np.arange(len(preds_display))
w = 0.18
for i, (dim, color) in enumerate(zip(DIMS, colors)):
    ax6.bar(x + i * w, preds_display[dim], w, color=color, alpha=0.8, label=dim.capitalize())
ax6.bar(x + 4 * w, preds_display["overall"], w, color="black", alpha=0.5, label="Overall")
ax6.errorbar(x + 0 * w + w / 2, preds_display["attainability"],
             yerr=2 * preds_display["gp_std"], fmt="none",
             ecolor="#2E4057", elinewidth=1, capsize=2, alpha=0.6)
ax6.set_xticks(x + w * 2)
ax6.set_xticklabels([f"G{int(i)}" for i in preds_display["goal_idx"]], fontsize=6)
ax6.set_ylim(0, 1.15)
ax6.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.4)
ax6.set_title("All 4 scores per goal  (error bars = GP +/-2 sigma)", fontsize=9)
ax6.legend(fontsize=7, ncol=3)

ax7 = fig.add_subplot(3, 2, 5, polar=True)
means  = [preds_display[d].mean() for d in DIMS]
labels = [d.capitalize() for d in DIMS]
angles = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False).tolist()
means  += means[:1]
angles += angles[:1]
ax7.plot(angles, means, "o-", lw=2, color="#4C72B0")
ax7.fill(angles, means, alpha=0.25, color="#4C72B0")
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(labels, fontsize=9)
ax7.set_ylim(0, 1)
ax7.set_title(f"Mean scores  all 4 dimensions (p{ANCHOR_PERIOD})", fontsize=10, pad=15)

ax8 = fig.add_subplot(3, 2, 6)
bar_colors = ["#C44E52" if u else "#4C72B0" for u in preds_display["uncertain"].values]
ax8.bar(x, preds_display["gp_std"], color=bar_colors, alpha=0.8)
ax8.axhline(cfg["gp_uncertainty_flag"], color="#C44E52", lw=1.5, ls="--",
            label=f"Threshold ({cfg['gp_uncertainty_flag']})")
ax8.set_xticks(x)
ax8.set_xticklabels([f"G{int(i)}" for i in preds_display["goal_idx"]], fontsize=6)
ax8.set_ylabel("GP std")
ax8.set_title("GP uncertainty per goal  (red = flagged)", fontsize=9)
ax8.legend(fontsize=8)

plt.tight_layout()
plt.savefig("demo_dashboard_poc.png", dpi=150, bbox_inches="tight")
print("\nOK  Saved demo_dashboard_poc.png")
print("=" * 70)
print("PoC COMPLETE")
print("=" * 70)
