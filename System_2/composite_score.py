import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
composite_score.py  v5  Composite Coherence Scoring + Portfolio Analysis

Reads meta_learner_predictions_poc.csv and produces:
  1. Per-goal composite coherence score (weighted, confidence-adjusted)
  2. Risk flags per goal and per dimension
  3. Portfolio summary per parent bucket (L2)
  4. Forward projection to +6 and +12 periods from p18

Pass D change:
  SNAPSHOT_PERIODS is now [6, 12, 18]. p24 is held out, no LLM-scored p24
  rows exist to verify. p24 validation comes from forward projection only.

Outputs:
  composite_scores_poc.csv     per goal final scores + flags
  portfolio_summary_poc.csv    per L2 parent bucket aggregation
  forward_projection_poc.csv   trajectory at +6 and +12 periods
"""

import os as _os
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from System_2.verify_goal import verify_scores
except ImportError:
    from verify_goal import verify_scores


print("=" * 70)
print("DECIDR COHERENCE ENGINE  Composite Scoring + Portfolio Analysis")
print("=" * 70)

WEIGHTS = {
    "coherence"    : 0.25,
    "attainability": 0.25,
    "relevance"    : 0.25,
    "integrity"    : 0.25,
}

RISK_THRESHOLD       = 0.35
CRITICAL_THRESHOLD   = 0.20
UNCERTAINTY_PENALTYF = 0.10

DIMS              = ["attainability", "relevance", "coherence", "integrity"]
NORM_REF_PERIODS  = [12, 18]
SNAPSHOT_PERIODS  = [6, 12, 18]
ANCHOR_PERIOD     = 18
PROJECTION_FROM   = 18

# Period at which the rest of the pipeline considers "now" for projection.
# LLM snapshots never exceed ANCHOR_PERIOD; p24 is proven against, not scored.

# ── Load data ───────────────────────────────────────────────────────────────
_full_pred_file = "meta_learner_predictions_full.csv"
_main_pred_file = "meta_learner_predictions_poc.csv"

if _os.path.exists(_full_pred_file):
    preds = pd.read_csv(_full_pred_file)
    print(f"  Using full predictions: {preds.shape} ({preds['period_id'].nunique()} periods)")
else:
    preds = pd.read_csv(_main_pred_file)
    if "period_id" not in preds.columns:
        preds["period_id"] = ANCHOR_PERIOD
    print(f"  Using primary predictions: {preds.shape}")

rules   = pd.read_csv("rule_scores_poc.csv") if _os.path.exists("rule_scores_poc.csv") else None
buckets = pd.read_csv("buckets.csv")
goals   = pd.read_csv("goals.csv")

# Anchor reference for fallback merges
if _os.path.exists("period_18_poc.csv"):
    anchor_ref = pd.read_csv("period_18_poc.csv").reset_index(drop=True)
else:
    anchor_ref = pd.read_csv("period_12_poc.csv").reset_index(drop=True)

print(f"\nGoals scored     : {len(preds)}")
print(f"Dimension weights: {WEIGHTS}")
print(f"Snapshot periods : {SNAPSHOT_PERIODS}")
print(f"Anchor           : p{ANCHOR_PERIOD}")
print(f"Risk threshold   : <{RISK_THRESHOLD}  Critical: <{CRITICAL_THRESHOLD}")

# ── Hierarchy lookup ────────────────────────────────────────────────────────
l2 = buckets[buckets["bucket_level"] == 2][["bucket_id", "bucket_name", "parent_bucket_id"]].copy()
l2.columns = ["l2_id", "l2_name", "l1_id"]

l3 = buckets[buckets["bucket_level"] == 3][["bucket_id", "parent_bucket_id"]].copy()
l3.columns = ["l3_id", "l2_id"]

l1 = buckets[buckets["bucket_level"] == 1][["bucket_id", "bucket_name"]].copy()
l1.columns = ["l1_id", "l1_name"]

hier = l3.merge(l2, on="l2_id").merge(l1, on="l1_id")
goal_hier = goals[["goal_id", "bucket_id"]].merge(
    hier, left_on="bucket_id", right_on="l3_id", how="left"
)

preds = preds.merge(goal_hier[["goal_id", "l2_id", "l2_name", "l1_name"]], on="goal_id", how="left")

# ══════════════════════════════════════════════════════════════════════════
# COMPOSITE COMPUTATION  runs before verification
# ══════════════════════════════════════════════════════════════════════════
# Min-max normalization per dimension across all 840 rows.
# Low attainability (mean~0.11) is CORRECT and reflects the dataset reality:
# most goals are genuinely off-track at p18. The composite will be lower
# because of this, which is the honest signal the dashboard should surface.
print(f"\n  Per-dimension raw score ranges:")
for dim in DIMS:
    print(f"    {dim:<16}: mean={preds[dim].mean():.3f}  "
          f"min={preds[dim].min():.3f}  max={preds[dim].max():.3f}")

dim_min = {d: preds[d].min() for d in DIMS}
dim_max = {d: preds[d].max() for d in DIMS}

for dim in DIMS:
    span = dim_max[dim] - dim_min[dim]
    preds[f"{dim}_norm"] = (
        ((preds[dim] - dim_min[dim]) / span).clip(0, 1)
        if span > 0.01 else 0.5
    )

preds["composite_raw"]      = sum(preds[dim] * w for dim, w in WEIGHTS.items())
preds["composite"]          = preds["composite_raw"]   # one composite, no hidden normalization
preds["composite_adjusted"] = np.where(
    preds["uncertain"],
    (preds["composite"] - UNCERTAINTY_PENALTYF).clip(0, 1),
    preds["composite"],
)

preds["at_risk"]       = False
preds["critical"]      = False
preds["weakest_dim"]   = "attainability"
preds["weakest_score"] = 0.0

print(f"\n  Raw composite (true scale): mean={preds['composite_raw'].mean():.3f}")
print(f"  Normalized composite      : mean={preds['composite'].mean():.3f}")

# ── Verification pass ───────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("VERIFICATION  Bedrock reviewing composite scores")
print(f"Periods verified: {SNAPSHOT_PERIODS}  (p24 held out for projection)")
print("-" * 60)

EXTRA_SIG_COLS = [
    "needle_move_ratio", "weighted_goal_status_score", "alloc_drift_std",
    "sibling_rank_pct", "allocation_fitness_score", "observed_value",
    "target_value_final_period", "trailing_6_period_slope",
    "allocation_efficiency_ratio", "delivered_output_quality_score",
    "dependency_risk_encoded", "dep_avg_attain",
    # Shock signals now real values, not 0.0 defaults
    "budget_shock_exposure", "shock_alloc_impact",
    "recovery_period_estimate", "recovery_window_remaining",
    "market_shock_vulnerable", "market_shock_forward_risk",
]

all_ver_results = []

for VER_PERIOD in SNAPSHOT_PERIODS:
    _ver_feat_file = f"features_raw_p{VER_PERIOD}.csv"
    if _os.path.exists(_ver_feat_file):
        features_ver = pd.read_csv(_ver_feat_file)
    else:
        print(f"  Period {VER_PERIOD}: no feature file {_ver_feat_file}, skipping")
        continue

    for col in EXTRA_SIG_COLS:
        if col not in features_ver.columns and col in anchor_ref.columns:
            features_ver[col] = anchor_ref[col].values[:len(features_ver)]

    if preds["period_id"].nunique() > 1:
        preds_ver = preds[preds["period_id"] == VER_PERIOD].copy().reset_index(drop=True)
    else:
        preds_ver = preds.copy().reset_index(drop=True)

    if len(preds_ver) == 0:
        print(f"  Period {VER_PERIOD}: no rows, skipping")
        continue

    print(f"  Verifying {len(preds_ver)} goals at period {VER_PERIOD}...")
    period_ver_results = []

    for i, row in preds_ver.iterrows():
        goal_idx = int(row["goal_idx"]) if "goal_idx" in row.index else i
        if goal_idx < len(features_ver):
            signals = features_ver.iloc[goal_idx].to_dict()
        else:
            signals = {}

        # Pass current period so verifier's projection math is right
        signals["current_period"] = VER_PERIOD

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

        print(f"  [p{VER_PERIOD}] Goal {goal_idx}", flush=True)
        vr = verify_scores(
            scores=scores,
            signals=signals,
            composite=float(row["composite_adjusted"]),
            weights=WEIGHTS,
            ensemble_meta=ensemble_meta,
            verbose=True,
        )
        period_ver_results.append(vr)

    preds_ver["verified_attainability"]     = [v["adjusted_attainability"] for v in period_ver_results]
    preds_ver["verified_relevance"]         = [v["adjusted_relevance"]     for v in period_ver_results]
    preds_ver["verified_coherence"]         = [v["adjusted_coherence"]     for v in period_ver_results]
    preds_ver["verified_integrity"]         = [v["adjusted_integrity"]     for v in period_ver_results]
    preds_ver["verified_composite"]         = [v["adjusted_composite"]    for v in period_ver_results]
    preds_ver["verification_flags"]         = [json.dumps(v["flags"])      for v in period_ver_results]
    preds_ver["verification_narrative"]     = [v["narrative"]              for v in period_ver_results]
    preds_ver["verified"]                   = [v["verified"]               for v in period_ver_results]
    preds_ver["verification_adjustments"]   = [json.dumps(v["adjustments"]) for v in period_ver_results]
    preds_ver["verifier_model_used"]        = [v.get("verifier_model", "")  for v in period_ver_results]
    preds_ver["period_id_ver"]              = VER_PERIOD

    # shock_effect: sum of all shock adjustment deltas across 4 dims (always <= 0)
    # Captured directly from analyse_shock_effect() result stored in adjustments key
    try:
        from System_2.verify_goal import analyse_shock_effect
    except ImportError:
        from verify_goal import analyse_shock_effect

    _shock_effects = []
    for i, row in preds_ver.iterrows():
        goal_idx = int(row["goal_idx"]) if "goal_idx" in row.index else i
        if goal_idx < len(features_ver):
            _sigs = features_ver.iloc[goal_idx].to_dict()
        else:
            _sigs = {}
        _sadj, _ = analyse_shock_effect(_sigs)
        _shock_effects.append(round(sum(_sadj.values()), 4))
    preds_ver["shock_effect"] = _shock_effects

    n_ver = sum(1 for v in period_ver_results if v["verified"])
    n_flg = sum(1 for v in period_ver_results if v["flags"])
    print(f"    Period {VER_PERIOD}: {n_ver}/{len(preds_ver)} clean, {n_flg} flagged")

    all_ver_results.append(preds_ver)

ver_all = pd.concat(all_ver_results, ignore_index=True) if all_ver_results else pd.DataFrame()

ver_cols = [
    "goal_id", "period_id_ver",
    "verified_attainability", "verified_relevance",
    "verified_coherence", "verified_integrity",
    "verified_composite", "verification_flags",
    "verification_narrative", "verified",
    "verification_adjustments", "verifier_model_used",
    "shock_effect",
]
ver_cols = [c for c in ver_cols if c in ver_all.columns]

if len(ver_all) > 0:
    ver_merge = ver_all[ver_cols].rename(columns={"period_id_ver": "period_id"})
    preds = preds.merge(ver_merge, on=["goal_id", "period_id"], how="left")

n_verified = int(ver_all["verified"].sum()) if "verified" in ver_all.columns else 0

# ── Recompute all flags on VERIFIED scores ─────────────────────────────────
preds["final_composite"] = (
    preds["verified_composite"].where(
        preds["verified_composite"].notna(),
        preds["composite_adjusted"],
    )
    if "verified_composite" in preds.columns
    else preds["composite_adjusted"]
)

preds["at_risk"]  = preds["final_composite"] < RISK_THRESHOLD
preds["critical"] = preds["final_composite"] < CRITICAL_THRESHOLD

for dim in DIMS:
    vcol = f"verified_{dim}"
    if vcol in preds.columns:
        base = preds[vcol].where(preds[vcol].notna(), preds[dim])
    else:
        base = preds[dim]
    preds[f"{dim}_at_risk"] = base < RISK_THRESHOLD


def _get_verified_dim(df, dim):
    vcol = f"verified_{dim}"
    if vcol in df.columns:
        return df[vcol].where(df[vcol].notna(), df[dim])
    return df[dim]


dim_scores = pd.DataFrame({d: _get_verified_dim(preds, d) for d in DIMS})
preds["weakest_dim"]   = dim_scores.idxmin(axis=1)
preds["weakest_score"] = dim_scores.min(axis=1)

# ── goal_priority — P1/P2/P3/P4 label ────────────────────────────────────────
# Computed at every period but most meaningful at p18 (anchor) and p24 (projection).
# Rules:
#   P1 Critical  — composite < 0.20  OR  (at_risk AND shock_effect < -0.06)  OR  (uncertain AND at_risk)
#   P2 At Risk   — composite 0.20-0.35  OR  (uncertain AND composite < 0.45)
#   P3 Watch     — composite 0.35-0.50  OR  any single verified dimension below 0.35
#   P4 Healthy   — all four dimensions above 0.35 AND composite >= 0.50
_shock_col = preds["shock_effect"] if "shock_effect" in preds.columns else pd.Series(0.0, index=preds.index)

def _assign_priority(row):
    fc        = row["final_composite"]
    uncertain = bool(row.get("uncertain", False))
    at_risk   = bool(row.get("at_risk", False))
    shock_eff = float(row.get("shock_effect", 0.0)) if "shock_effect" in row.index else 0.0
    wk_score  = float(row.get("weakest_score", 1.0))

    if fc < CRITICAL_THRESHOLD:
        return "P1"
    if at_risk and shock_eff < -0.06:
        return "P1"
    if at_risk and uncertain:
        return "P1"
    if at_risk:
        return "P2"
    if uncertain and fc < 0.45:
        return "P2"
    if fc < 0.50 or wk_score < RISK_THRESHOLD:
        return "P3"
    return "P4"

preds["goal_priority"] = preds.apply(_assign_priority, axis=1)

_priority_counts = preds[preds["period_id"] == ANCHOR_PERIOD]["goal_priority"].value_counts().to_dict() \
                   if "period_id" in preds.columns else preds["goal_priority"].value_counts().to_dict()
print(f"\n  Goal priority at p{ANCHOR_PERIOD}:")
for _p in ["P1", "P2", "P3", "P4"]:
    print(f"    {_p}: {_priority_counts.get(_p, 0)} goals")

print(f"\n  Final scores (post-verification):")
print(f"    Composite mean : {preds['final_composite'].mean():.3f}")
print(f"    At risk        : {preds['at_risk'].sum()}/{len(preds)} rows")
print(f"    Critical       : {preds['critical'].sum()}/{len(preds)} rows")

print(f"\n  Weakest dimension distribution (verified):")
for dim in DIMS:
    count = (preds["weakest_dim"] == dim).sum()
    print(f"    {dim:<16}: {count} rows weakest")


def _count_flagged(df):
    count = 0
    for _, r in df.iterrows():
        try:
            flags_str = r.get("verification_flags")
            if flags_str and isinstance(flags_str, str) and json.loads(flags_str):
                count += 1
        except Exception:
            pass
    return count


n_flagged = _count_flagged(ver_all)
print(f"\n  Total verified (no flags) : {n_verified}/{len(ver_all)}")
print(f"  Total goals flagged       : {n_flagged}/{len(ver_all)}")
if "period_id_ver" in ver_all.columns:
    print(f"  Periods verified          : {sorted(ver_all['period_id_ver'].unique())}")

norm_cols = [f"{d}_norm" for d in DIMS if f"{d}_norm" in preds.columns]
opt_verify = [
    "verified_attainability", "verified_relevance", "verified_coherence",
    "verified_integrity", "verified_composite", "verified",
    "verification_flags", "verification_narrative", "verification_adjustments",
    "verifier_model_used",
]
opt_verify = [c for c in opt_verify if c in preds.columns]

# ══════════════════════════════════════════════════════════════════════════
# PART 1  Per-goal composite coherence score (verified)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  Composite score (weighted):")
print(f"    Mean  : {preds['composite'].mean():.3f}")
print(f"    Range : [{preds['composite'].min():.3f}, {preds['composite'].max():.3f}]")

print(f"\n  Confidence-adjusted composite:")
print(f"    Mean  : {preds['composite_adjusted'].mean():.3f}")

if preds["period_id"].nunique() > 1:
    preds_display = preds[preds["period_id"] == ANCHOR_PERIOD].copy()
    print(f"\n  Goal scores at period {ANCHOR_PERIOD} (anchor, ranked by composite):")
else:
    preds_display = preds.copy()
    print(f"\n  Goal scores (ranked by composite):")

display_cols = [
    "goal_id", "attainability", "relevance", "coherence", "integrity",
    "composite_adjusted", "final_composite",
    "uncertain", "weakest_dim", "at_risk",
]

ranked = preds_display[display_cols].sort_values("final_composite")
print(ranked.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# PART 2  Portfolio summary by parent bucket (L2)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print(f"PART 2  Portfolio summary by parent bucket (L2) at p{ANCHOR_PERIOD}")
print("-" * 60)

if preds["period_id"].nunique() > 1:
    preds_anchor_port = preds[preds["period_id"] == ANCHOR_PERIOD].copy()
else:
    preds_anchor_port = preds.copy()

portfolio = preds_anchor_port.groupby(["l2_name", "l1_name"]).agg(
    n_goals         = ("goal_id",         "count"),
    avg_composite   = ("final_composite", "mean"),
    min_composite   = ("final_composite", "min"),
    avg_attainability = ("attainability",    "mean"),
    avg_relevance     = ("relevance",        "mean"),
    avg_coherence     = ("coherence",        "mean"),
    avg_integrity     = ("integrity",        "mean"),
    at_risk_count     = ("at_risk",          "sum"),
    critical_count    = ("critical",         "sum"),
    uncertain_count   = ("uncertain",        "sum"),
).round(3).reset_index()

portfolio["risk_pct"] = (portfolio["at_risk_count"] / portfolio["n_goals"] * 100).round(1)
portfolio = portfolio.sort_values("avg_composite")

print(f"\n{'L2 Bucket':<30} {'L1':<12} {'Goals':>6} {'Composite':>10} {'At Risk':>8} {'Critical':>9}")
print("-" * 80)
for _, row in portfolio.iterrows():
    flag = "  CRITICAL" if row["critical_count"] > 0 else ("  AT RISK" if row["at_risk_count"] > 0 else "")
    print(
        f"  {row['l2_name']:<28} {row['l1_name']:<12} {row['n_goals']:>6} "
        f"{row['avg_composite']:>10.3f} {row['at_risk_count']:>8} {row['critical_count']:>9}{flag}"
    )

# ══════════════════════════════════════════════════════════════════════════
# PART 3  Forward projection (+6 and +12 periods from p18)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print(f"PART 3  Forward projection (+6 and +12 periods from p{PROJECTION_FROM})")
print("-" * 60)
print(f"  +6 periods = p24 (honest forward test, LLMs never saw this)")
print(f"  +12 periods = p30 (future, no validation)")
print(f"  R/C/I projected via linear slope from history, attainability via metric slope")

analytical = pd.read_csv("analytical_full.csv")

if preds["period_id"].nunique() > 1:
    preds_proj = preds[preds["period_id"] == PROJECTION_FROM].copy().reset_index(drop=True)
    print(f"  Forward projection from period {PROJECTION_FROM}, validating against actual period 24")
else:
    preds_proj = preds.copy()
    print("  Forward projection (single period mode)")

proj = preds_proj[["goal_id", "attainability", "final_composite", "gp_std", "uncertain"]].copy()
proj["composite_adjusted"] = proj["final_composite"]

period_proj = analytical[analytical["period_id"] == PROJECTION_FROM][
    ["goal_id", "trailing_6_period_slope", "volatility_measure",
     "observed_value", "target_value_final_period", "variance_from_target"]
].copy()

if len(period_proj) == 0:
    print(f"  WARNING: period {PROJECTION_FROM} not found in analytical_full, falling back to p12")
    period_proj = analytical[analytical["period_id"] == 12][
        ["goal_id", "trailing_6_period_slope", "volatility_measure",
         "observed_value", "target_value_final_period", "variance_from_target"]
    ].copy()

proj = proj.merge(period_proj, on="goal_id", how="left")

actual_p24 = analytical[analytical["period_id"] == 24][
    ["goal_id", "observed_value", "probability_of_hitting_target"]
].rename(columns={
    "observed_value"              : "actual_observed_p24",
    "probability_of_hitting_target": "actual_attain_p24",
})
proj = proj.merge(actual_p24, on="goal_id", how="left")


def project_attainability(observed, slope, target, periods_ahead):
    projected_value = observed + slope * periods_ahead
    return float(np.clip(projected_value / max(target, 1e-9), 0, 1))


def compute_dim_slope(full_df, goal_id, dim, from_period, n_periods=6):
    if full_df is None:
        return 0.0
    goal_data = full_df[
        (full_df["goal_id"] == goal_id) &
        (full_df["period_id"] <= from_period) &
        (full_df["period_id"] > from_period - n_periods)
    ].sort_values("period_id")
    if len(goal_data) < 2 or dim not in goal_data.columns:
        return 0.0
    vals = goal_data[dim].values
    return float(np.polyfit(range(len(vals)), vals, 1)[0])


full_preds_df = (
    pd.read_csv("meta_learner_predictions_full.csv")
    if _os.path.exists("meta_learner_predictions_full.csv") else None
)

# ── Forward projection: per-goal slopes for ALL 4 dimensions ─────────────
# Uses the full 24-period meta-learner predictions to compute a per-goal
# linear slope for each dimension (p1-p18 history only, never p19-p24).
# This gives genuine per-goal differentiation for p24 and p30 across all
# four dimensions — not just attainability. Attainability additionally uses
# the long-run observed_value metric slope as a second signal.
from scipy import stats as _scipy_stats

_anl = pd.read_csv("analytical_full.csv") if _os.path.exists("analytical_full.csv") \
       else pd.read_csv("analytical_flat.csv")

print(f"  Computing per-goal slopes (p1-p{PROJECTION_FROM}) for all 4 dimensions...")

_slope_rows = []
for gid in proj["goal_id"]:

    # ── Dimension slopes from meta-learner full predictions ───────────────
    if full_preds_df is not None:
        gdf = full_preds_df[
            (full_preds_df["goal_id"] == gid) &
            (full_preds_df["period_id"] <= PROJECTION_FROM)
        ].sort_values("period_id")
    else:
        gdf = pd.DataFrame()

    def _dim_slope(df, col):
        if len(df) < 3 or col not in df.columns:
            return 0.0
        sl, _, _, _, _ = _scipy_stats.linregress(df["period_id"], df[col])
        return float(sl)

    sl_coh = _dim_slope(gdf, "coherence")
    sl_rel = _dim_slope(gdf, "relevance")
    sl_int = _dim_slope(gdf, "integrity")
    sl_att = _dim_slope(gdf, "attainability")

    # p18 base values from meta-learner
    p18_row = gdf[gdf["period_id"] == PROJECTION_FROM]
    coh_p18 = float(p18_row["coherence"].values[0])    if len(p18_row) else 0.5
    rel_p18 = float(p18_row["relevance"].values[0])    if len(p18_row) else 0.5
    int_p18 = float(p18_row["integrity"].values[0])    if len(p18_row) else 0.5
    att_p18 = float(p18_row["attainability"].values[0]) if len(p18_row) else 0.1

    # ── Attainability: also compute metric slope as cross-check ──────────
    g_anl = _anl[
        (_anl["goal_id"] == gid) &
        (_anl["period_id"] <= PROJECTION_FROM)
    ].sort_values("period_id")
    tgt = g_anl["target_value_final_period"].values[-1] if len(g_anl) else 1.0
    p18v = float(g_anl[g_anl["period_id"] == PROJECTION_FROM]["observed_value"].values[0]) \
           if len(g_anl[g_anl["period_id"] == PROJECTION_FROM]) > 0 \
           else float(g_anl["observed_value"].values[-1]) if len(g_anl) else 0.0
    sl_metric = _scipy_stats.linregress(g_anl["period_id"], g_anl["observed_value"])[0] \
                if len(g_anl) >= 3 else 0.0

    # Blend: 50% meta-learner attainability slope + 50% metric slope
    attain_p24_meta   = float(np.clip(att_p18 + sl_att * 6, 0, 1))
    attain_p24_metric = float(np.clip((p18v + sl_metric * 6) / max(tgt, 1e-9), 0, 1))
    attain_p24 = float(np.clip(0.5 * attain_p24_meta + 0.5 * attain_p24_metric, 0, 1))

    attain_p30_meta   = float(np.clip(att_p18 + sl_att * 12, 0, 1))
    attain_p30_metric = float(np.clip((p18v + sl_metric * 12) / max(tgt, 1e-9), 0, 1))
    attain_p30 = float(np.clip(0.5 * attain_p30_meta + 0.5 * attain_p30_metric, 0, 1))

    _slope_rows.append({
        "goal_id"     : gid,
        # p24 (6 periods ahead)
        "attain_p6"   : attain_p24,
        "coherence_p6": float(np.clip(coh_p18 + sl_coh * 6, 0, 1)),
        "relevance_p6": float(np.clip(rel_p18 + sl_rel * 6, 0, 1)),
        "integrity_p6": float(np.clip(int_p18 + sl_int * 6, 0, 1)),
        # p30 (12 periods ahead)
        "attain_p12"   : attain_p30,
        "coherence_p12": float(np.clip(coh_p18 + sl_coh * 12, 0, 1)),
        "relevance_p12": float(np.clip(rel_p18 + sl_rel * 12, 0, 1)),
        "integrity_p12": float(np.clip(int_p18 + sl_int * 12, 0, 1)),
        # Store slopes for reporting
        "coherence_slope" : round(sl_coh, 6),
        "relevance_slope" : round(sl_rel, 6),
        "integrity_slope" : round(sl_int, 6),
        "attain_slope"    : round(sl_att, 6),
        # p18 bases for display
        "coh_base": coh_p18, "rel_base": rel_p18,
        "int_base": int_p18, "att_base": att_p18,
    })

_slope_df = pd.DataFrame(_slope_rows)
proj = proj.merge(_slope_df, on="goal_id", how="left")

for dim in ["coherence","relevance","integrity","attainability"]:
    scol = f"{dim}_slope" if dim != "attainability" else "attain_slope"
    if scol in proj.columns:
        pos = (proj[scol] > 0.001).sum()
        neg = (proj[scol] < -0.001).sum()
        print(f"    {dim:<16}: improving={pos}/35  declining={neg}/35  "
              f"mean slope={proj[scol].mean():.5f}")

print(f"  attain_p6  mean={proj['attain_p6'].mean():.3f}  "
      f"min={proj['attain_p6'].min():.3f}  max={proj['attain_p6'].max():.3f}")
print(f"  attain_p12 mean={proj['attain_p12'].mean():.3f}  "
      f"min={proj['attain_p12'].min():.3f}  max={proj['attain_p12'].max():.3f}")
print(f"  Coherence slope mean: {proj['coherence_slope'].mean():.4f}")
print(f"  Relevance slope mean: {proj['relevance_slope'].mean():.4f}")
print(f"  Integrity slope mean: {proj['integrity_slope'].mean():.4f}")

# dim_base already set in slope_df — ensure coh_base exists for downstream

proj["composite_p6"] = (
    proj["attain_p6"]     * WEIGHTS["attainability"] +
    proj["relevance_p6"]  * WEIGHTS["relevance"] +
    proj["coherence_p6"]  * WEIGHTS["coherence"] +
    proj["integrity_p6"]  * WEIGHTS["integrity"]
)
proj["composite_p12"] = (
    proj["attain_p12"]    * WEIGHTS["attainability"] +
    proj["relevance_p12"] * WEIGHTS["relevance"] +
    proj["coherence_p12"] * WEIGHTS["coherence"] +
    proj["integrity_p12"] * WEIGHTS["integrity"]
)

proj["trajectory_p6"]  = proj["composite_p6"]  - proj["composite_adjusted"]
proj["trajectory_p12"] = proj["composite_p12"] - proj["composite_adjusted"]

proj["projected_attain_p24"] = proj["attain_p6"]
proj["attain_error_p24"]     = (proj["projected_attain_p24"] - proj["actual_attain_p24"]).abs()

proj["improving_p6"]  = proj["trajectory_p6"]  > 0.02
proj["improving_p12"] = proj["trajectory_p12"] > 0.02
proj["degrading_p6"]  = proj["trajectory_p6"]  < -0.02
proj["degrading_p12"] = proj["trajectory_p12"] < -0.02

print(f"\n  Trajectory summary:")
print(f"    Improving in +6 periods  : {proj['improving_p6'].sum()} goals")
print(f"    Degrading in +6 periods  : {proj['degrading_p6'].sum()} goals")
print(f"    Improving in +12 periods : {proj['improving_p12'].sum()} goals")
print(f"    Degrading in +12 periods : {proj['degrading_p12'].sum()} goals")

print(f"\n  Top 5 goals most at risk at projected p24 (composite_p6):")
worst_p12 = proj.nsmallest(5, "composite_p6")[
    ["goal_id", "coh_base", "coherence_p6", "rel_base", "relevance_p6",
     "int_base", "integrity_p6", "attain_p6", "composite_p6"]
]
worst_p12.columns = [
    "goal_id", "coh_p18", "coh_p24", "rel_p18", "rel_p24",
    "int_p18", "int_p24", "attain_p24", "composite_p24",
]
print(worst_p12.round(3).to_string(index=False))

print(f"\n  Validation  projected p24 vs actual metric (attainability only):")
if "actual_attain_p24" in proj.columns and proj["actual_attain_p24"].notna().any():
    mae_val = proj["attain_error_p24"].dropna().mean()
    print(f"    Attainability MAE (p{PROJECTION_FROM} -> p24): {mae_val:.4f}")
    print(f"    Projected mean : {proj['projected_attain_p24'].mean():.3f}")
    print(f"    Actual mean    : {proj['actual_attain_p24'].mean():.3f}")

print(f"\n  Top 5 goals most likely to improve by projected p24:")
best_p12 = proj.nlargest(5, "trajectory_p6")[
    ["goal_id", "coh_base", "coherence_p6", "composite_adjusted", "composite_p6", "trajectory_p6"]
]
best_p12.columns = ["goal_id", "coh_p18", "coh_p24", "composite_p18", "composite_p24", "delta"]
print(best_p12.round(3).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# PART 4  Visualisations
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"Decidr Coherence Engine  Composite Scoring & Portfolio Analysis (anchor p{ANCHOR_PERIOD})",
    fontsize=13, fontweight="bold"
)

# Plot 1: Composite scores per goal (anchor period)
ax1 = axes[0, 0]
preds_plot = preds_anchor_port.sort_values("composite_adjusted").copy()
colors = ["#C44E52" if c else "#4C72B0" for c in preds_plot["at_risk"]]
ax1.barh(range(len(preds_plot)), preds_plot["composite_adjusted"], color=colors)
ax1.axvline(RISK_THRESHOLD,     color="#C44E52", lw=1.5, ls="--", label=f"Risk threshold ({RISK_THRESHOLD})")
ax1.axvline(CRITICAL_THRESHOLD, color="#8B0000", lw=1.5, ls=":",  label=f"Critical ({CRITICAL_THRESHOLD})")
ax1.set_xlabel("Composite coherence score")
ax1.set_title(f"Goals ranked by composite at p{ANCHOR_PERIOD}\n(red = at risk)", fontsize=10)
ax1.legend(fontsize=8)
ax1.set_xlim(0, 1)

# Plot 2: Dimension breakdown heatmap
ax2 = axes[0, 1]
dim_data = preds_anchor_port[DIMS].values
im = ax2.imshow(dim_data.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax2.set_yticks(range(4))
ax2.set_yticklabels([d.capitalize() for d in DIMS])
ax2.set_xlabel("Goal index")
ax2.set_title(f"Dimension scores heatmap at p{ANCHOR_PERIOD}\n(red=poor, green=good)", fontsize=10)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# Plot 3: Portfolio bar chart
ax3 = axes[1, 0]
port_sorted = portfolio.sort_values("avg_composite", ascending=True)
port_colors = ["#C44E52" if r > 0 else "#4C72B0" for r in port_sorted["at_risk_count"]]
ax3.barh(range(len(port_sorted)), port_sorted["avg_composite"], color=port_colors, alpha=0.85)
ax3.set_yticks(range(len(port_sorted)))
ax3.set_yticklabels(port_sorted["l2_name"], fontsize=8)
ax3.axvline(RISK_THRESHOLD, color="#C44E52", lw=1.5, ls="--")
ax3.set_xlabel("Avg composite coherence score")
ax3.set_title("Portfolio coherence by parent bucket\n(red = has at-risk goals)", fontsize=10)
ax3.set_xlim(0, 1)

# Plot 4: Forward projection
ax4 = axes[1, 1]
x = np.arange(len(proj))
ax4.plot(x, proj["composite_adjusted"], "o-", color="#4C72B0",
         label=f"Period {PROJECTION_FROM} (projection start)", lw=2)
ax4.plot(x, proj["composite_p6"],  "s--", color="#55A868",
         label=f"+6 periods to p24 (validation)", lw=1.5)
ax4.plot(x, proj["composite_p12"], "^:",  color="#DD8452",
         label=f"+12 periods to p30 (future)", lw=1.5)
ax4.axhline(RISK_THRESHOLD, color="#C44E52", lw=1, ls="--", alpha=0.6)
ax4.fill_between(x, 0, RISK_THRESHOLD, alpha=0.05, color="#C44E52")
ax4.set_xlabel("Goal index")
ax4.set_ylabel("Composite coherence score")
ax4.set_title("Forward projection  composite trajectory", fontsize=10)
ax4.legend(fontsize=8)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("composite_dashboard_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK  Saved composite_dashboard_poc.png")

# ── Additional plots ────────────────────────────────────────────────────────

# Plot 5: Shock timeline — coherence over 24 periods with shock shading
# Requires full 840-row predictions. Shows the dip and recovery story clearly.
if full_preds_df is not None and "coherence" in full_preds_df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    ts = full_preds_df.groupby("period_id").agg(
        avg_coherence     = ("coherence",        "mean"),
        avg_attainability = ("attainability",     "mean"),
        avg_relevance     = ("relevance",         "mean"),
        avg_integrity     = ("integrity",         "mean"),
        avg_composite     = ("overall",           "mean") if "overall" in full_preds_df.columns
                            else ("coherence",    "mean"),
    ).reset_index()

    ax.plot(ts["period_id"], ts["avg_coherence"],     label="Coherence",     color="#4C72B0", lw=2)
    ax.plot(ts["period_id"], ts["avg_relevance"],     label="Relevance",     color="#55A868", lw=1.5, ls="--")
    ax.plot(ts["period_id"], ts["avg_integrity"],     label="Integrity",     color="#DD8452", lw=1.5, ls="--")
    ax.plot(ts["period_id"], ts["avg_attainability"], label="Attainability", color="#C44E52", lw=1.5, ls=":")

    # Shade shock periods
    ax.axvspan(10, 12, alpha=0.12, color="#C44E52", label="Budget shock (p10-12)")
    ax.axvspan(14, 17, alpha=0.12, color="#DD8452", label="Market shock (p14-17)")
    ax.axvline(18, color="#1E2761", lw=1.5, ls="--", alpha=0.6, label="Anchor p18")
    ax.axvline(24, color="#64748B", lw=1,   ls=":",  alpha=0.6, label="p24 (held out)")

    ax.set_xlabel("Period")
    ax.set_ylabel("Mean score across all goals")
    ax.set_title("All four dimensions over 24 periods  with shock events marked", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(1, 24)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 25))
    plt.tight_layout()
    plt.savefig("shock_timeline_poc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("OK  Saved shock_timeline_poc.png")

# Plot 6: Dimension balance stacked bar at p18 — shows relative contribution
# of each dimension per goal side by side. Reveals which goals are balanced
# vs which lean heavily on one dimension.
fig, ax = plt.subplots(figsize=(14, 5))
p18_sorted = preds_anchor_port.sort_values("composite_adjusted").reset_index(drop=True)
x = np.arange(len(p18_sorted))
bar_w = 0.2
colors_dims = {"attainability": "#C44E52", "relevance": "#55A868",
               "coherence": "#4C72B0", "integrity": "#DD8452"}
for i, dim in enumerate(DIMS):
    norm_col = f"{dim}_norm" if f"{dim}_norm" in p18_sorted.columns else dim
    ax.bar(x + i * bar_w, p18_sorted[norm_col], bar_w,
           label=dim.capitalize(), color=colors_dims[dim], alpha=0.85)
ax.set_xticks(x + bar_w * 1.5)
ax.set_xticklabels([f"G{int(g)}" for g in p18_sorted["goal_id"]], fontsize=7, rotation=45)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.4)
ax.set_ylabel("Normalized score")
ax.set_ylim(0, 1.05)
ax.set_title(f"Dimension balance per goal at p{ANCHOR_PERIOD}  (sorted by composite)", fontsize=10)
ax.legend(fontsize=8, ncol=4)
plt.tight_layout()
plt.savefig("dimension_balance_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK  Saved dimension_balance_poc.png")

# Plot 7: Goal trajectory — composite score across p6, p12, p18 for each goal.
# Shows which goals improved, which degraded, which stayed flat over the
# three scored snapshots. Lines coloured by final composite at p18.
if full_preds_df is not None:
    fig, ax = plt.subplots(figsize=(12, 6))
    snap_preds = full_preds_df[full_preds_df["period_id"].isin([6, 12, 18])].copy()
    if "overall" in snap_preds.columns:
        score_col = "overall"
    else:
        score_col = "coherence"

    cmap = plt.cm.RdYlGn
    p18_scores = snap_preds[snap_preds["period_id"] == 18].set_index("goal_id")[score_col]

    for goal_id, group in snap_preds.groupby("goal_id"):
        group = group.sort_values("period_id")
        final = p18_scores.get(goal_id, 0.5)
        ax.plot(group["period_id"], group[score_col],
                color=cmap(final), lw=1.2, alpha=0.7, marker="o", ms=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02).set_label("p18 score", fontsize=8)

    ax.axvspan(10, 12, alpha=0.10, color="#C44E52")
    ax.axvspan(14, 17, alpha=0.10, color="#DD8452")
    ax.set_xticks([6, 12, 18])
    ax.set_xticklabels(["p6", "p12 (budget shock)", "p18 (anchor)"])
    ax.set_ylabel(f"{score_col.capitalize()} score")
    ax.set_title("Goal trajectories across scored snapshots  (green = high p18 score, red = low)", fontsize=10)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("goal_trajectories_poc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("OK  Saved goal_trajectories_poc.png")

# Plot 8: Weakest dimension per goal — pie + per-goal bar.
# Shows which dimension is most commonly the weakest link across the portfolio.
fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

weak_counts = preds_anchor_port["weakest_dim"].value_counts()
pie_colors = [colors_dims.get(d, "#999999") for d in weak_counts.index]
ax_pie.pie(weak_counts.values, labels=[d.capitalize() for d in weak_counts.index],
           colors=pie_colors, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
ax_pie.set_title(f"Most common weakest dimension at p{ANCHOR_PERIOD}", fontsize=10)

p18_sorted2 = preds_anchor_port.sort_values("composite_adjusted").reset_index(drop=True)
bar_colors2 = [colors_dims.get(d, "#999999") for d in p18_sorted2["weakest_dim"]]
ax_bar.bar(range(len(p18_sorted2)), p18_sorted2["weakest_score"], color=bar_colors2, alpha=0.85)
ax_bar.axhline(RISK_THRESHOLD, color="#C44E52", lw=1.2, ls="--", label=f"Risk threshold ({RISK_THRESHOLD})")
ax_bar.set_xticks(range(len(p18_sorted2)))
ax_bar.set_xticklabels([f"G{int(g)}" for g in p18_sorted2["goal_id"]], fontsize=7, rotation=45)
ax_bar.set_ylabel("Weakest dimension score")
ax_bar.set_title("Score of weakest dimension per goal  (colour = which dimension)", fontsize=10)
ax_bar.legend(fontsize=8)
handles = [plt.Rectangle((0,0),1,1, color=colors_dims[d]) for d in DIMS]
ax_bar.legend(handles, [d.capitalize() for d in DIMS], fontsize=8, title="Weakest dim")

plt.tight_layout()
plt.savefig("weakest_dimension_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK  Saved weakest_dimension_poc.png")
composite_out_cols = [
    "goal_id", "period_id", "l2_name", "l1_name",
    "attainability", "relevance", "coherence", "integrity",
    "composite_raw", "composite", "composite_adjusted", "final_composite",
    "gp_std", "uncertain",
    "at_risk", "critical", "weakest_dim", "weakest_score",
    "shock_effect", "goal_priority",
] + norm_cols + opt_verify + [f"{d}_at_risk" for d in DIMS]

composite_out_cols = [c for c in composite_out_cols if c in preds.columns]
preds[composite_out_cols].to_csv("composite_scores_poc.csv", index=False)

portfolio.to_csv("portfolio_summary_poc.csv", index=False)

# Portfolio timeseries (if multiple periods available)
if preds["period_id"].nunique() > 1:
    portfolio_ts = preds.groupby(["period_id", "l2_name", "l1_name"]).agg(
        n_goals            = ("goal_id",            "count"),
        avg_composite      = ("composite_adjusted", "mean"),
        avg_composite_raw  = ("composite_raw",      "mean"),
        at_risk_count      = ("at_risk",            "sum"),
        avg_coherence      = ("coherence",          "mean"),
        avg_coherence_norm = ("coherence_norm" if "coherence_norm" in preds.columns else "coherence", "mean"),
        avg_attainability  = ("attainability",      "mean"),
        avg_relevance      = ("relevance",          "mean"),
        avg_integrity      = ("integrity",          "mean"),
    ).round(3).reset_index()

    portfolio_ts["budget_shock"] = portfolio_ts["period_id"].isin([10, 11, 12]).astype(int)
    portfolio_ts["market_shock"] = portfolio_ts["period_id"].isin([14, 15, 16, 17]).astype(int)
    portfolio_ts.to_csv("portfolio_timeseries_poc.csv", index=False)
    print("OK  Saved portfolio_timeseries_poc.csv")

    coherence_ts = preds.groupby("period_id").agg(
        avg_composite_raw  = ("composite_raw",      "mean"),
        avg_composite      = ("composite_adjusted", "mean"),
        at_risk_count      = ("at_risk",            "sum"),
        avg_coherence      = ("coherence",          "mean"),
        avg_coherence_norm = ("coherence_norm" if "coherence_norm" in preds.columns else "coherence", "mean"),
        avg_attainability  = ("attainability",      "mean"),
        avg_relevance      = ("relevance",          "mean"),
        avg_integrity      = ("integrity",          "mean"),
    ).round(3).reset_index()

    BUDGET_SHOCK_PERIODS = [10, 11, 12]
    MARKET_SHOCK_PERIODS = [14, 15, 16, 17]

    coherence_ts["budget_shock"] = coherence_ts["period_id"].isin(BUDGET_SHOCK_PERIODS).astype(int)
    coherence_ts["market_shock"] = coherence_ts["period_id"].isin(MARKET_SHOCK_PERIODS).astype(int)
    coherence_ts["any_shock"]    = (coherence_ts["budget_shock"] | coherence_ts["market_shock"]).astype(int)
    coherence_ts["shock_label"]  = coherence_ts.apply(
        lambda r: "Budget Shock" if r["budget_shock"]
        else ("Market Shock" if r["market_shock"] else ""),
        axis=1,
    )
    coherence_ts.to_csv("coherence_timeseries_poc.csv", index=False)
    print("OK  Saved coherence_timeseries_poc.csv")
    print(f"  Budget shock periods: {BUDGET_SHOCK_PERIODS}")
    print(f"  Market shock periods: {MARKET_SHOCK_PERIODS}")

save_cols = [
    "goal_id", "composite_adjusted", "final_composite", "composite_p6", "composite_p12",
    "trajectory_p6", "trajectory_p12",
    "attain_p6", "attain_p12",
    "improving_p6", "improving_p12",
    "degrading_p6", "degrading_p12",
    "projected_attain_p24", "actual_attain_p24", "attain_error_p24",
    "coherence_p6", "relevance_p6", "integrity_p6",
    "coherence_p12", "relevance_p12", "integrity_p12",
    "coherence_slope", "relevance_slope", "integrity_slope",
]
save_cols = [c for c in save_cols if c in proj.columns]
proj[save_cols].to_csv("forward_projection_poc.csv", index=False)

print("OK  Saved composite_scores_poc.csv")
print("OK  Saved portfolio_summary_poc.csv")
print("OK  Saved forward_projection_poc.csv")
print("=" * 70)
print("COMPOSITE SCORING COMPLETE")