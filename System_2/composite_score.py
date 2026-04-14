"""
08_composite_score.py — Composite Coherence Scoring + Portfolio Analysis

Reads meta_learner_predictions_poc.csv and produces:
  1. Per-goal composite coherence score (weighted, confidence-adjusted)
  2. Risk flags per goal and per dimension
  3. Portfolio summary per parent bucket (L2)
  4. Forward projection: 6 and 12 period horizon per goal

Outputs:
  composite_scores_poc.csv     — per goal final scores + flags
  portfolio_summary_poc.csv    — per L2 parent bucket aggregation
  forward_projection_poc.csv   — trajectory at +6 and +12 periods
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
from verify_goal import verify_scores
import matplotlib.pyplot as plt

print("=" * 70)
print("DECIDR COHERENCE ENGINE — Composite Scoring + Portfolio Analysis")
print("=" * 70)

# ── Config ────────────────────────────────────────────────────────────────────
# Dimension weights for composite score
# Equal by default — adjust based on Decidr's priorities
WEIGHTS = {
    "coherence"    : 0.35,   # primary — coherence engine
    "attainability": 0.25,   # has ground truth, GP-trained
    "relevance"    : 0.20,   # allocation justified
    "integrity"    : 0.20,   # outcomes match intent
}

# Risk thresholds
RISK_THRESHOLD    = 0.35   # below this = at risk
CRITICAL_THRESHOLD= 0.20   # below this = critical
UNCERTAINTY_PENALTYF = 0.10  # reduce composite by this if GP uncertain

DIMS = ["attainability", "relevance", "coherence", "integrity"]

# ── Load data ─────────────────────────────────────────────────────────────────
import os as _os
# Use full 840-row predictions if available, fallback to p12
_full_pred_file = "meta_learner_predictions_full.csv"
_p12_pred_file  = "meta_learner_predictions_poc.csv"
if _os.path.exists(_full_pred_file):
    preds = pd.read_csv(_full_pred_file)
    print(f"  Using full predictions: {preds.shape} ({preds['period_id'].nunique()} periods)")
else:
    preds = pd.read_csv(_p12_pred_file)
    preds["period_id"] = 12
    print(f"  Using period 12 predictions only: {preds.shape}")
# 'overall' and 'coherence_composite' both present from 04
rules   = pd.read_csv("rule_scores_poc.csv")
period12= pd.read_csv("period_12_poc.csv").reset_index(drop=True)
buckets = pd.read_csv("buckets.csv")
goals   = pd.read_csv("goals.csv")

print(f"\nGoals scored     : {len(preds)}")
print(f"Dimension weights: {WEIGHTS}")
print(f"Risk threshold   : <{RISK_THRESHOLD}  Critical: <{CRITICAL_THRESHOLD}")

# ── Hierarchy lookup ──────────────────────────────────────────────────────────
l2 = buckets[buckets['bucket_level']==2][['bucket_id','bucket_name','parent_bucket_id']].copy()
l2.columns = ['l2_id','l2_name','l1_id']
l3 = buckets[buckets['bucket_level']==3][['bucket_id','parent_bucket_id']].copy()
l3.columns = ['l3_id','l2_id']
l1 = buckets[buckets['bucket_level']==1][['bucket_id','bucket_name']].copy()
l1.columns = ['l1_id','l1_name']

hier = l3.merge(l2, on='l2_id').merge(l1, on='l1_id')
goal_hier = goals[['goal_id','bucket_id']].merge(
    hier, left_on='bucket_id', right_on='l3_id', how='left'
)

preds = preds.merge(goal_hier[['goal_id','l2_id','l2_name','l1_name']], on='goal_id', how='left')

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — PER-GOAL COMPOSITE SCORE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("PART 1 — Per-goal composite coherence score")
print("─" * 60)

# Weighted composite — works for both single period and multi-period
preds['composite'] = sum(
    preds[dim] * w for dim, w in WEIGHTS.items()
)

# Add period_id if missing
if 'period_id' not in preds.columns:
    preds['period_id'] = 12

# Confidence adjustment: penalise uncertain goals slightly
preds['composite_adjusted'] = np.where(
    preds['uncertain'],
    (preds['composite'] - UNCERTAINTY_PENALTYF).clip(0, 1),
    preds['composite']
)

# Risk flags
preds['at_risk']  = preds['composite_adjusted'] < RISK_THRESHOLD
preds['critical'] = preds['composite_adjusted'] < CRITICAL_THRESHOLD

# Per-dimension risk flags
for dim in DIMS:
    preds[f"{dim}_at_risk"] = preds[dim] < RISK_THRESHOLD

# Weakest dimension per goal
preds['weakest_dim'] = preds[DIMS].idxmin(axis=1)
preds['weakest_score'] = preds[DIMS].min(axis=1)

# Summary
print(f"\n  Composite score (weighted):")
print(f"    Mean  : {preds['composite'].mean():.3f}")
print(f"    Range : [{preds['composite'].min():.3f}, {preds['composite'].max():.3f}]")
print(f"\n  Confidence-adjusted composite:")
print(f"    Mean  : {preds['composite_adjusted'].mean():.3f}")
print(f"\n  Risk breakdown:")
print(f"    At risk (composite < {RISK_THRESHOLD})  : {preds['at_risk'].sum()} / {len(preds)} goals")
print(f"    Critical (composite < {CRITICAL_THRESHOLD}): {preds['critical'].sum()} / {len(preds)} goals")
print(f"    Uncertain (GP flagged)            : {preds['uncertain'].sum()} / {len(preds)} goals")

print(f"\n  Weakest dimension distribution:")
for dim in DIMS:
    count = (preds['weakest_dim'] == dim).sum()
    print(f"    {dim:<16}: {count} goals weakest")

# Show period 18 snapshot (latest trained state) not period 12
if 'period_id' in preds.columns and preds['period_id'].nunique() > 1:
    preds_display = preds[preds['period_id'] == 18].copy()
    print(f"\n  Goal scores at period 18 (latest state — ranked by composite):")
else:
    preds_display = preds.copy()
    print(f"\n  Goal scores (ranked by composite):")
display_cols = ['goal_id','attainability','relevance','coherence','integrity',
                'composite_adjusted','uncertain','weakest_dim','at_risk']
ranked = preds_display[display_cols].sort_values('composite_adjusted')
print(ranked.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — PORTFOLIO SUMMARY BY PARENT BUCKET (L2)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("PART 2 — Portfolio summary by parent bucket (L2)")
print("─" * 60)

# Use period 18 as the primary portfolio view
preds_p18_port = preds[preds['period_id'] == 18].copy()     if 'period_id' in preds.columns and preds['period_id'].nunique() > 1     else preds.copy()
portfolio = preds_p18_port.groupby(['l2_name','l1_name']).agg(
    n_goals          = ('goal_id','count'),
    avg_composite    = ('composite_adjusted','mean'),
    min_composite    = ('composite_adjusted','min'),
    avg_attainability= ('attainability','mean'),
    avg_relevance    = ('relevance','mean'),
    avg_coherence    = ('coherence','mean'),
    avg_integrity    = ('integrity','mean'),
    at_risk_count    = ('at_risk','sum'),
    critical_count   = ('critical','sum'),
    uncertain_count  = ('uncertain','sum'),
).round(3).reset_index()

portfolio['risk_pct'] = (portfolio['at_risk_count'] / portfolio['n_goals'] * 100).round(1)
portfolio = portfolio.sort_values('avg_composite')

print(f"\n{'L2 Bucket':<30} {'L1':<12} {'Goals':>6} {'Composite':>10} {'At Risk':>8} {'Critical':>9}")
print("-" * 80)
for _, row in portfolio.iterrows():
    flag = "  CRITICAL" if row['critical_count'] > 0 else ("  AT RISK" if row['at_risk_count'] > 0 else "")
    print(f"  {row['l2_name']:<28} {row['l1_name']:<12} {row['n_goals']:>6} "
          f"{row['avg_composite']:>10.3f} {row['at_risk_count']:>8} {row['critical_count']:>9}{flag}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — FORWARD PROJECTION (+6 and +12 periods)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("PART 3 — Forward projection (+6 and +12 periods)")
print("─" * 60)
print("  Projects from period 24 (final known state) into future periods 30 and 36")
print("  R/C/I held at period 24 values — attainability projected via slope")

# Forward projection from period 18 — gives 6-period validation window (p19-p24)
# We predict what happens at p24, then compare against actual p24 values
PROJECTION_FROM = 18   # change to 24 to project beyond dataset

analytical = pd.read_csv("analytical_flat.csv")

if 'period_id' in preds.columns and preds['period_id'].nunique() > 1:
    preds_proj = preds[preds['period_id'] == PROJECTION_FROM].copy().reset_index(drop=True)
    print(f"  Forward projection from period {PROJECTION_FROM} — validating against actual period 24")
else:
    preds_proj = preds.copy()
    print("  Forward projection (single period mode)")

proj = preds_proj[['goal_id','attainability','composite_adjusted','gp_std','uncertain']].copy()

# Signals at projection period
period_proj = analytical[analytical['period_id'] == PROJECTION_FROM][
    ['goal_id','trailing_6_period_slope','volatility_measure',
     'observed_value','target_value_final_period','variance_from_target']
].copy()

if len(period_proj) == 0:
    print(f"  WARNING: period {PROJECTION_FROM} not found — falling back to period 12")
    period_proj = period12[['goal_id','trailing_6_period_slope','volatility_measure',
                             'observed_value','target_value_final_period','variance_from_target']].copy()

proj = proj.merge(period_proj, on='goal_id', how='left')

# Actual period 24 values for validation
actual_p24 = analytical[analytical['period_id'] == 24][
    ['goal_id','observed_value','probability_of_hitting_target']
].rename(columns={
    'observed_value'              : 'actual_observed_p24',
    'probability_of_hitting_target': 'actual_attain_p24'
})
proj = proj.merge(actual_p24, on='goal_id', how='left')

# Also join actual period 24 composite score for full comparison
if 'period_id' in preds.columns and preds['period_id'].nunique() > 1:
    actual_composite_p24 = preds[preds['period_id'] == 24][
        ['goal_id','composite_adjusted','coherence','relevance','integrity','attainability']
    ].rename(columns={
        'composite_adjusted': 'actual_composite_p24',
        'coherence'         : 'actual_coherence_p24',
        'relevance'         : 'actual_relevance_p24',
        'integrity'         : 'actual_integrity_p24',
        'attainability'     : 'actual_attainability_p24',
    })
    proj = proj.merge(actual_composite_p24, on='goal_id', how='left')
    print(f"  Joined actual period 24 composite scores for validation")

PROJECTION_FROM = 18   # must match value above

def project_attainability(observed, slope, target, periods_ahead, current_period=None):
    projected_value = observed + slope * periods_ahead
    return float(np.clip(projected_value / max(target, 1e-9), 0, 1))

def compute_dim_slope(full_preds_df, goal_id, dim, from_period, n_periods=6):
    """Compute trailing slope for a dimension over last n_periods before from_period."""
    goal_data = full_preds_df[
        (full_preds_df['goal_id'] == goal_id) &
        (full_preds_df['period_id'] <= from_period) &
        (full_preds_df['period_id'] > from_period - n_periods)
    ].sort_values('period_id')
    if len(goal_data) < 2 or dim not in goal_data.columns:
        return 0.0
    vals = goal_data[dim].values
    return float(np.polyfit(range(len(vals)), vals, 1)[0])

# Load full predictions for slope computation
full_preds_df = pd.read_csv("meta_learner_predictions_full.csv")     if _os.path.exists("meta_learner_predictions_full.csv") else None

proj['attain_p6']  = proj.apply(lambda r: project_attainability(
    r['observed_value'], r['trailing_6_period_slope'],
    r['target_value_final_period'], 6), axis=1)

proj['attain_p12'] = proj.apply(lambda r: project_attainability(
    r['observed_value'], r['trailing_6_period_slope'],
    r['target_value_final_period'], 12), axis=1)

# Project all 4 dimensions forward using trailing slopes
print("  Computing dimension slopes at period 18 for full 4-dimension projection...")

# Get p18 base values into proj cleanly (no merge inside loop)
dim_base = preds_proj[['goal_id','coherence','relevance','integrity']].copy()
dim_base.columns = ['goal_id','coh_base','rel_base','int_base']
proj = proj.merge(dim_base, on='goal_id', how='left')

for dim, base_col in [('coherence','coh_base'),('relevance','rel_base'),('integrity','int_base')]:
    proj[f'{dim}_slope'] = proj['goal_id'].apply(
        lambda gid: compute_dim_slope(full_preds_df, gid, dim, PROJECTION_FROM)
        if full_preds_df is not None else 0.0
    )
    proj[f'{dim}_p6']  = (proj[base_col] + proj[f'{dim}_slope'] * 6).clip(0, 1)
    proj[f'{dim}_p12'] = (proj[base_col] + proj[f'{dim}_slope'] * 12).clip(0, 1)

print(f"  Coherence slope mean: {proj['coherence_slope'].mean():.4f}")
print(f"  Relevance slope mean: {proj['relevance_slope'].mean():.4f}")
print(f"  Integrity slope mean: {proj['integrity_slope'].mean():.4f}")

# Composite projection: attainability changes, R/C/I held constant
# (R/C/I projection would require future allocation data)
# Full 4-dimension composite projection
proj['composite_p6']  = (proj['attain_p6']       * WEIGHTS['attainability'] +
                          proj['relevance_p6']     * WEIGHTS['relevance'] +
                          proj['coherence_p6']     * WEIGHTS['coherence'] +
                          proj['integrity_p6']     * WEIGHTS['integrity'])

proj['composite_p12'] = (proj['attain_p12']      * WEIGHTS['attainability'] +
                          proj['relevance_p12']    * WEIGHTS['relevance'] +
                          proj['coherence_p12']    * WEIGHTS['coherence'] +
                          proj['integrity_p12']    * WEIGHTS['integrity'])

proj['trajectory_p6']  = proj['composite_p6']  - proj['composite_adjusted']
proj['trajectory_p12'] = proj['composite_p12'] - proj['composite_adjusted']

# Validation: compare p18→p24 projection against actual p24
proj['projected_attain_p24']    = proj['attain_p6']   # +6 from p18 = p24
proj['attain_error_p24']        = (proj['projected_attain_p24'] - proj['actual_attain_p24']).abs()
# Composite validation
if 'actual_composite_p24' in proj.columns:
    proj['composite_error_p24'] = (proj['composite_p6'] - proj['actual_composite_p24']).abs()

proj['improving_p6']   = proj['trajectory_p6']  > 0.02
proj['improving_p12']  = proj['trajectory_p12'] > 0.02
proj['degrading_p6']   = proj['trajectory_p6']  < -0.02
proj['degrading_p12']  = proj['trajectory_p12'] < -0.02

print(f"\n  Trajectory summary:")
print(f"    Improving in +6 periods  : {proj['improving_p6'].sum()} goals")
print(f"    Degrading in +6 periods  : {proj['degrading_p6'].sum()} goals")
print(f"    Improving in +12 periods : {proj['improving_p12'].sum()} goals")
print(f"    Degrading in +12 periods : {proj['degrading_p12'].sum()} goals")

print(f"\n  Top 5 goals most at risk at projected p24 (composite_p6):")
worst_p12 = proj.nsmallest(5, 'composite_p6')[
    ['goal_id','coh_base','coherence_p6','rel_base','relevance_p6',
     'int_base','integrity_p6','attain_p6','composite_p6']]
worst_p12.columns = ['goal_id','coh_p18','coh_p24','rel_p18','rel_p24',
                     'int_p18','int_p24','attain_p24','composite_p24']
print(worst_p12.round(3).to_string(index=False))

print(f"\n  Validation — projected p24 vs actual p24 (all 4 dimensions):")
if 'actual_composite_p24' in proj.columns and proj['actual_composite_p24'].notna().any():
    # Compute MAE for all 4 dimensions + composite
    dim_map = {
        'coherence'    : ('coherence_p6',   'actual_coherence_p24'),
        'relevance'    : ('relevance_p6',   'actual_relevance_p24'),
        'integrity'    : ('integrity_p6',   'actual_integrity_p24'),
        'attainability': ('attain_p6',       'actual_attainability_p24'),
        'composite'    : ('composite_p6',   'actual_composite_p24'),
    }
    print(f"    {'Dimension':<16} {'Projected':>10} {'Actual':>10} {'MAE':>10}")
    print(f"    {'-'*50}")
    for dim, (proj_col, actual_col) in dim_map.items():
        if proj_col in proj.columns and actual_col in proj.columns:
            mae = (proj[proj_col] - proj[actual_col]).abs().dropna().mean()
            proj_mean   = proj[proj_col].mean()
            actual_mean = proj[actual_col].mean()
            print(f"    {dim:<16} {proj_mean:>10.3f} {actual_mean:>10.3f} {mae:>10.4f}")
            if dim != 'composite':
                proj[f'{dim}_error_p24'] = (proj[proj_col] - proj[actual_col]).abs()

    print(f"\n    Per-goal full dimension comparison at p24:")
    compare_cols = ['goal_id',
                    'coherence_p6','actual_coherence_p24',
                    'relevance_p6','actual_relevance_p24',
                    'integrity_p6','actual_integrity_p24',
                    'attain_p6','actual_attainability_p24',
                    'composite_p6','actual_composite_p24','composite_error_p24']
    available = [c for c in compare_cols if c in proj.columns]
    compare_out = proj[available].copy()
    compare_out = compare_out.sort_values('composite_error_p24') if 'composite_error_p24' in compare_out.columns else compare_out
    print(compare_out.round(3).to_string(index=False))
elif 'actual_attain_p24' in proj.columns and proj['actual_attain_p24'].notna().any():
    mae_val = proj['attain_error_p24'].dropna().mean()
    print(f"    Attainability MAE (p18→p24): {mae_val:.4f}")

print(f"\n  Top 5 goals most likely to improve by projected p24:")
best_p12 = proj.nlargest(5, 'trajectory_p6')[
    ['goal_id','coh_base','coherence_p6','composite_adjusted','composite_p6','trajectory_p6']]
best_p12.columns = ['goal_id','coh_p18','coh_p24','composite_p18','composite_p24','delta']
print(best_p12.round(3).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Decidr Coherence Engine — Composite Scoring & Portfolio Analysis",
             fontsize=13, fontweight="bold")

# Plot 1: Composite scores per goal
ax1 = axes[0, 0]
colors = ['#C44E52' if c else '#4C72B0' for c in preds['at_risk']]
bars = ax1.barh(range(len(preds)), preds.sort_values('composite_adjusted')['composite_adjusted'],
                color=[colors[i] for i in preds.sort_values('composite_adjusted').index])
ax1.axvline(RISK_THRESHOLD, color='#C44E52', lw=1.5, ls='--', label=f'Risk threshold ({RISK_THRESHOLD})')
ax1.axvline(CRITICAL_THRESHOLD, color='#8B0000', lw=1.5, ls=':', label=f'Critical ({CRITICAL_THRESHOLD})')
ax1.set_xlabel("Composite coherence score")
ax1.set_title("Goals ranked by composite score\n(red = at risk)", fontsize=10)
ax1.legend(fontsize=8)
ax1.set_xlim(0, 1)

# Plot 2: Dimension breakdown heatmap
ax2 = axes[0, 1]
dim_data = preds[DIMS].values
im = ax2.imshow(dim_data.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax2.set_yticks(range(4))
ax2.set_yticklabels([d.capitalize() for d in DIMS])
ax2.set_xlabel("Goal index")
ax2.set_title("Dimension scores heatmap\n(red=poor, green=good)", fontsize=10)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# Plot 3: Portfolio bar chart
ax3 = axes[1, 0]
port_sorted = portfolio.sort_values('avg_composite', ascending=True)
port_colors = ['#C44E52' if r > 0 else '#4C72B0' for r in port_sorted['at_risk_count']]
ax3.barh(range(len(port_sorted)), port_sorted['avg_composite'], color=port_colors, alpha=0.85)
ax3.set_yticks(range(len(port_sorted)))
ax3.set_yticklabels(port_sorted['l2_name'], fontsize=8)
ax3.axvline(RISK_THRESHOLD, color='#C44E52', lw=1.5, ls='--')
ax3.set_xlabel("Avg composite coherence score")
ax3.set_title("Portfolio coherence by parent bucket\n(red = has at-risk goals)", fontsize=10)
ax3.set_xlim(0, 1)

# Plot 4: Forward projection
ax4 = axes[1, 1]
x = np.arange(len(proj))
ax4.plot(x, proj['composite_adjusted'], 'o-', color='#4C72B0', label='Period 18 (projection start)', lw=2)
ax4.plot(x, proj['composite_p6'],  's--', color='#55A868', label='+6 periods → P24 (validation)', lw=1.5)
ax4.plot(x, proj['composite_p12'], '^:', color='#DD8452', label='+12 periods → P30 (future)', lw=1.5)
ax4.axhline(RISK_THRESHOLD, color='#C44E52', lw=1, ls='--', alpha=0.6)
ax4.fill_between(x, 0, RISK_THRESHOLD, alpha=0.05, color='#C44E52')
ax4.set_xlabel("Goal index")
ax4.set_ylabel("Composite coherence score")
ax4.set_title("Forward projection — composite trajectory", fontsize=10)
ax4.legend(fontsize=8)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("composite_dashboard_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved composite_dashboard_poc.png")

# ── Save ──────────────────────────────────────────────────────────────────────
# ── VERIFICATION PASS ────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("VERIFICATION — qwen3 reviewing composite scores")
print("─" * 60)

features_raw = pd.read_csv("features_raw_poc.csv")
period_12    = pd.read_csv("period_12_poc.csv").reset_index(drop=True)

# Merge extra signal cols into features_raw for verifier
extra_sig_cols = [
    "needle_move_ratio", "weighted_goal_status_score", "alloc_drift_std",
    "sibling_rank_pct", "allocation_fitness_score", "observed_value",
    "target_value_final_period", "trailing_6_period_slope",
    "allocation_efficiency_ratio", "delivered_output_quality_score",
]
for col in extra_sig_cols:
    if col not in features_raw.columns and col in period_12.columns:
        features_raw[col] = period_12[col].values

ver_results = []
for i, row in preds.iterrows():
    goal_idx = int(row["goal_idx"]) if "goal_idx" in row.index else i
    signals  = features_raw.iloc[goal_idx].to_dict() if goal_idx < len(features_raw) else {}

    scores = {
        "attainability": float(row["attainability"]),
        "relevance"    : float(row["relevance"]),
        "coherence"    : float(row["coherence"]),
        "integrity"    : float(row["integrity"]),
    }
    ensemble_meta = {}
    try:
        ensemble_meta = json.loads(row["ensemble_meta"]) if "ensemble_meta" in row.index else {}
    except Exception:
        pass

    print(f"  Goal {goal_idx}", end=" ", flush=True)
    vr = verify_scores(
        scores        = scores,
        signals       = signals,
        composite     = float(row["composite_adjusted"]),
        weights       = WEIGHTS,
        ensemble_meta = ensemble_meta,
        verbose       = True,
    )
    ver_results.append(vr)

# Apply verified scores back to preds
preds["verified_attainability"]  = [v["adjusted_attainability"]  for v in ver_results]
preds["verified_relevance"]      = [v["adjusted_relevance"]      for v in ver_results]
preds["verified_coherence"]      = [v["adjusted_coherence"]      for v in ver_results]
preds["verified_integrity"]      = [v["adjusted_integrity"]      for v in ver_results]
preds["verified_composite"]      = [v["adjusted_composite"]      for v in ver_results]
preds["verification_flags"]      = [json.dumps(v["flags"])       for v in ver_results]
preds["verification_narrative"]  = [v["narrative"]               for v in ver_results]
preds["verified"]                = [v["verified"]                for v in ver_results]
preds["verification_adjustments"]= [json.dumps(v["adjustments"]) for v in ver_results]

n_verified   = sum(1 for v in ver_results if v["verified"])
n_flagged    = sum(1 for v in ver_results if v["flags"])
n_adjusted   = sum(1 for v in ver_results if v["adjustments"])
print(f"\n  Verified (no flags)  : {n_verified}/{len(preds)}")
print(f"  Goals flagged        : {n_flagged}/{len(preds)}")
print(f"  Scores adjusted      : {n_adjusted}/{len(preds)}")


composite_out = preds[['goal_id','l2_name','l1_name',
                        'attainability','relevance','coherence','integrity',
                        'composite','composite_adjusted',
                        'gp_std','uncertain',
                        'at_risk','critical','weakest_dim','weakest_score',
                        'verified_attainability','verified_relevance','verified_coherence',
                        'verified_integrity','verified_composite','verified',
                        'verification_flags','verification_narrative','verification_adjustments'] +
                       [f"{d}_at_risk" for d in DIMS]].copy()
composite_out.to_csv("composite_scores_poc.csv", index=False)

portfolio.to_csv("portfolio_summary_poc.csv", index=False)

# Portfolio summary per period (for time series dashboard)
if preds['period_id'].nunique() > 1:
    portfolio_ts = preds.groupby(['period_id','l2_name','l1_name']).agg(
        n_goals          = ('goal_id','count'),
        avg_composite    = ('composite_adjusted','mean'),
        at_risk_count    = ('at_risk','sum'),
    ).round(3).reset_index()
    portfolio_ts['budget_shock'] = portfolio_ts['period_id'].isin([10,11,12]).astype(int)
    portfolio_ts['market_shock'] = portfolio_ts['period_id'].isin([14,15,16,17]).astype(int)
    portfolio_ts.to_csv("portfolio_timeseries_poc.csv", index=False)
    print("✓ Saved portfolio_timeseries_poc.csv")

    # Overall coherence per period
    coherence_ts = preds.groupby('period_id').agg(
        avg_composite    = ('composite_adjusted','mean'),
        at_risk_count    = ('at_risk','sum'),
        avg_coherence    = ('coherence','mean'),
        avg_attainability= ('attainability','mean'),
    ).round(3).reset_index()

    # Add shock period markers for dashboard visualisation
    BUDGET_SHOCK_PERIODS = [10, 11, 12]
    MARKET_SHOCK_PERIODS = [14, 15, 16, 17]
    coherence_ts['budget_shock'] = coherence_ts['period_id'].isin(BUDGET_SHOCK_PERIODS).astype(int)
    coherence_ts['market_shock'] = coherence_ts['period_id'].isin(MARKET_SHOCK_PERIODS).astype(int)
    coherence_ts['any_shock']    = (
        coherence_ts['budget_shock'] | coherence_ts['market_shock']
    ).astype(int)
    coherence_ts['shock_label']  = coherence_ts.apply(
        lambda r: 'Budget Shock' if r['budget_shock']
        else ('Market Shock' if r['market_shock'] else ''), axis=1
    )
    coherence_ts.to_csv("coherence_timeseries_poc.csv", index=False)
    print("✓ Saved coherence_timeseries_poc.csv")
    print(f"  Budget shock periods: {BUDGET_SHOCK_PERIODS}")
    print(f"  Market shock periods: {MARKET_SHOCK_PERIODS}")

save_cols = ['goal_id','composite_adjusted','composite_p6','composite_p12',
             'trajectory_p6','trajectory_p12',
             'attain_p6','attain_p12',
             'improving_p6','improving_p12',
             'degrading_p6','degrading_p12',
             'projected_attain_p24','actual_attain_p24','attain_error_p24']
for opt_col in ['actual_composite_p24','composite_error_p24',
                'actual_coherence_p24','coherence_error_p24',
                'actual_relevance_p24','relevance_error_p24',
                'actual_integrity_p24','integrity_error_p24',
                'actual_attainability_p24','attainability_error_p24',
                'coherence_p6','relevance_p6','integrity_p6',
                'coherence_p12','relevance_p12','integrity_p12']:
    if opt_col in proj.columns:
        save_cols.append(opt_col)
proj[save_cols].to_csv("forward_projection_poc.csv", index=False)

print("✓ Saved composite_scores_poc.csv")
print("✓ Saved portfolio_summary_poc.csv")
print("✓ Saved forward_projection_poc.csv")
print("=" * 70)
print("COMPOSITE SCORING COMPLETE")
print("=" * 70)