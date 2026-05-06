import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
feature_engineering.py  v5 (anchor = p18)

Builds features from all source files using the full bucket hierarchy.

The anchor period is now p18. The "rich" rule formula (hierarchy gap +
flag penalty) is applied to p18. Other snapshot periods (p6, p12, p24) get
a simpler formula; that's fine because they're training/validation signals,
not the anchor the rest of the pipeline builds on.

Sources:
  buckets.csv, goals.csv, allocations.csv, outputs.csv, metrics.csv,
  derived_fields.csv, analytical_flat.csv, goal_dependencies.csv (optional)

Outputs:
  features_full_normalized.csv   all 840 rows for GP training
  features_full_raw.csv
  features_raw_p{6,12,18,24}.csv  per-period raw feature snapshots
  rule_scores_p{6,12,18,24}.csv   per-period rule scores
  features_raw_poc.csv            alias for p18 (backward compat)
  features_normalized_poc.csv     p18 normalized slice
  rule_scores_poc.csv             alias for p18 rule scores (rich formula)
  period_18_poc.csv               anchor snapshot
  feature_scaler_poc.pkl
  feature_names_poc.txt
"""

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


ANCHOR_PERIOD = 18
SNAPSHOT_PERIODS = [6, 12, 18, 24]

print("=" * 70)
print(f"DECIDR SYSTEM 2  Feature Engineering v5 (anchor = p{ANCHOR_PERIOD})")
print("=" * 70)

# ── Load sources ─────────────────────────────────────────────────────────────
flat    = pd.read_csv("analytical_flat.csv")
buckets = pd.read_csv("buckets.csv")
goals   = pd.read_csv("goals.csv")
allocs  = pd.read_csv("allocations.csv")
outputs = pd.read_csv("outputs.csv")
metrics = pd.read_csv("metrics.csv")
derived = pd.read_csv("derived_fields.csv")

print(f"\nLoaded: flat={flat.shape}  buckets={buckets.shape}  goals={goals.shape}")
print(f"        allocs={allocs.shape}  outputs={outputs.shape}  metrics={metrics.shape}  derived={derived.shape}")

# ── Hierarchy ────────────────────────────────────────────────────────────────
l1 = buckets[buckets['bucket_level'] == 1][['bucket_id', 'allocation_percentage_of_total']].copy()
l1.columns = ['l1_id', 'l1_alloc_pct']
l2 = buckets[buckets['bucket_level'] == 2][['bucket_id', 'parent_bucket_id', 'allocation_percentage_of_total']].copy()
l2.columns = ['l2_id', 'l1_id', 'l2_alloc_pct']
l3 = buckets[buckets['bucket_level'] == 3][['bucket_id', 'parent_bucket_id', 'allocation_percentage_of_total', 'is_leaf']].copy()
l3.columns = ['l3_id', 'l2_id', 'l3_alloc_pct', 'is_leaf']

hier = l3.merge(l2, on='l2_id').merge(l1, on='l1_id')
hier['l3_share_of_l2'] = hier['l3_alloc_pct'] / hier['l2_alloc_pct']
hier['l3_share_of_l1'] = hier['l3_alloc_pct'] / hier['l1_alloc_pct']
hier['sibling_count']    = hier.groupby('l2_id')['l3_id'].transform('count')
hier['sibling_rank']     = hier.groupby('l2_id')['l3_alloc_pct'].rank(ascending=False)
hier['sibling_rank_pct'] = (hier['sibling_rank'] - 1) / (hier['sibling_count'] - 1).clip(lower=1)

print(f"\nHierarchy: {len(hier)} L3 goals mapped to L2 and L1 parents")

# ── Goal-level static features ──────────────────────────────────────────────
scenario_map = {'underfunded': 0.0, 'dynamic': 0.33, 'optimal': 0.67, 'overfunded': 1.0}
goals['scenario_encoded'] = goals['scenario_story'].map(scenario_map)

goal_static = goals[['goal_id', 'bucket_id', 'scenario_encoded',
                     'minimum_viable_allocation', 'optimal_allocation_min',
                     'optimal_allocation_max']].merge(
    hier[['l3_id', 'l2_id', 'l1_id', 'l3_share_of_l2', 'l3_share_of_l1', 'sibling_rank_pct']],
    left_on='bucket_id', right_on='l3_id', how='left'
)

alloc_drift = allocs.groupby('bucket_id').agg(
    alloc_drift_std=('allocation_percentage_of_parent', 'std'),
    alloc_mean=('allocation_percentage_of_parent', 'mean'),
).reset_index()

status_consist = flat.groupby('goal_id').agg(
    status_band_unique=('status_band', 'nunique'),
).reset_index()

metrics['needle_move_ratio'] = (
    metrics['observed_value'] / metrics['expected_value'].clip(lower=0.001)
).clip(0, 2)

print("\nBuilding per-period feature matrix (840 rows)...")

# ── Merge everything per (goal, period) ─────────────────────────────────────
df = flat.copy()

df = df.merge(
    derived[['goal_id', 'period_id', 'weighted_goal_status_score',
             'allocation_fitness_score', 'time_to_green_estimate']],
    on=['goal_id', 'period_id'], how='left'
)

if 'time_to_green_estimate_y' in df.columns:
    df['time_to_green_estimate'] = df['time_to_green_estimate_y']
elif 'time_to_green_estimate_x' in df.columns:
    df['time_to_green_estimate'] = df['time_to_green_estimate_x']

df = df.merge(
    metrics[['goal_id', 'period_id', 'needle_move_ratio']],
    on=['goal_id', 'period_id'], how='left'
)

df = df.merge(
    goal_static[['goal_id', 'scenario_encoded', 'l3_share_of_l2',
                 'l3_share_of_l1', 'sibling_rank_pct']],
    on='goal_id', how='left'
)

df = df.merge(alloc_drift[['bucket_id', 'alloc_drift_std']], on='bucket_id', how='left')
df = df.merge(status_consist, on='goal_id', how='left')

# Dependency features
if os.path.exists("goal_dependencies.csv"):
    dep_df = pd.read_csv("goal_dependencies.csv")
    dep_merge = dep_df[['goal_id', 'n_dependencies', 'n_dependents',
                         'dependency_risk', 'dep_avg_attain']].copy()
    dep_merge['dependency_risk_encoded'] = dep_merge['dependency_risk'].map(
        {'none': 0.0, 'low': 0.33, 'medium': 0.67, 'high': 1.0}).fillna(0.0)
    df = df.merge(
        dep_merge[['goal_id', 'n_dependencies', 'n_dependents',
                   'dependency_risk_encoded', 'dep_avg_attain']],
        on='goal_id', how='left'
    )
    df['n_dependencies']          = df['n_dependencies'].fillna(0)
    df['n_dependents']            = df['n_dependents'].fillna(0)
    df['dependency_risk_encoded'] = df['dependency_risk_encoded'].fillna(0.0)
    df['dep_avg_attain']          = df['dep_avg_attain'].fillna(-1.0)
    print(f"  Dependency features merged: n_dependencies mean={df['n_dependencies'].mean():.2f}")
else:
    df['n_dependencies']          = 0.0
    df['n_dependents']            = 0.0
    df['dependency_risk_encoded'] = 0.0
    df['dep_avg_attain']          = -1.0
    print("  goal_dependencies.csv not found  dependency features set to 0")

opt_centre = (df['optimal_allocation_min'] + df['optimal_allocation_max']) / 2
opt_range  = (df['optimal_allocation_max'] - df['optimal_allocation_min']).clip(lower=1e-9)
df['optimal_band_distance'] = ((df['allocation_percentage_of_total_bucket'] - opt_centre).abs() / opt_range).clip(0, 1)

# ── Shock features (engineered from raw data) ────────────────────────────────
# Budget shock: total_budget_available dropped 20% at periods 10-12.
# Market shock: growth metrics (visitors, count, dollars, attendees, leads)
#               dropped ~15% at period 14, recovered by period 17.
# These columns were referenced in LLM prompts with row.get() fallback of 0.0,
# meaning every prior call got 0.0. Now they're real.

BUDGET_SHOCK_PERIODS = {10, 11, 12}
MARKET_SHOCK_GOALS   = {4, 7, 9, 10, 11}   # confirmed from data: visitors, count, dollars, attendees, leads
SHOCK_END_PERIOD     = 12
FINAL_PERIOD         = 24

# 1. budget_shock_exposure: 1.0 during shock periods, 0.0 otherwise
df['budget_shock_exposure'] = df['period_id'].apply(
    lambda p: 1.0 if p in BUDGET_SHOCK_PERIODS else 0.0
)

# 2. shock_alloc_impact: actual drop vs pre-shock baseline (periods 7-9) per goal
baseline_alloc = (
    df[df['period_id'].isin([7, 8, 9])]
    .groupby('goal_id')['allocated_amount']
    .mean()
    .rename('baseline_alloc')
)
df = df.merge(baseline_alloc, on='goal_id', how='left')
df['shock_alloc_impact'] = (
    ((df['allocated_amount'] - df['baseline_alloc']) / df['baseline_alloc'].clip(lower=1e-9))
    .clip(-1.0, 0.0)
    .abs()
    .fillna(0.0)
)
df.drop(columns=['baseline_alloc'], inplace=True)

# 3. recovery_period_estimate: how many periods since shock ended (0 before/during shock)
df['recovery_period_estimate'] = df['period_id'].apply(
    lambda p: float(max(0, p - SHOCK_END_PERIOD)) if p > SHOCK_END_PERIOD else 0.0
)

# 4. recovery_window_remaining: periods available to recover before end of timeline
df['recovery_window_remaining'] = df['period_id'].apply(
    lambda p: float(max(0, FINAL_PERIOD - max(p, SHOCK_END_PERIOD)))
)

# 5. market_shock_vulnerable: 1.0 for goals whose metric type is affected by market shock
df['market_shock_vulnerable'] = df['goal_id'].apply(
    lambda g: 1.0 if g in MARKET_SHOCK_GOALS else 0.0
)

# 6. market_shock_forward_risk: magnitude of observed drop at p14 vs p13 per goal, 0 for unaffected
p13_obs = df[df['period_id'] == 13].set_index('goal_id')['observed_value']
p14_obs = df[df['period_id'] == 14].set_index('goal_id')['observed_value']
market_drop = ((p14_obs - p13_obs) / p13_obs.clip(lower=1e-9)).clip(-1.0, 0.0).abs()
market_drop_map = market_drop.to_dict()
df['market_shock_forward_risk'] = df['goal_id'].map(market_drop_map).fillna(0.0)

shock_cols = ['budget_shock_exposure','shock_alloc_impact','recovery_period_estimate',
              'recovery_window_remaining','market_shock_vulnerable','market_shock_forward_risk']
print(f"\n  Shock features engineered ({len(shock_cols)} cols):")
for c in shock_cols:
    print(f"    {c}: mean={df[c].mean():.3f}  max={df[c].max():.3f}")

print(f"  Merged shape: {df.shape}")

# ── Feature columns ─────────────────────────────────────────────────────────
STAT_FEATURES = [
    'trailing_6_period_slope', 'variance_from_target', 'volatility_measure', 'time_to_green_estimate',
    'allocation_percentage_of_parent', 'optimal_band_distance', 'sibling_rank_pct',
    'scenario_encoded', 'allocation_fitness_score',
    'l3_share_of_l2', 'l3_share_of_l1', 'alloc_drift_std',
    'weighted_goal_status_score', 'status_band_unique',
    'delivered_output_quality_score', 'delivered_output_quantity',
    'allocation_efficiency_ratio', 'needle_move_ratio', 'output_cost_per_unit',
    'n_dependencies', 'n_dependents', 'dependency_risk_encoded', 'dep_avg_attain',
    'observed_value', 'allocated_amount', 'allocated_time_hours',
    # Shock signals  were only in the LLM prompt before; now in the GP feature
    # matrix too so statistical scoring also learns from shock exposure and
    # recovery patterns
    'budget_shock_exposure', 'shock_alloc_impact',
    'recovery_period_estimate', 'recovery_window_remaining',
    'market_shock_vulnerable', 'market_shock_forward_risk',
    # Temporal features added for temporal GP versions
    'period_id_scaled',
    'market_shock_period',
]

# Compute temporal features before X_raw is built
df['period_id_scaled']   = df['period_id'] / 24.0
df['market_shock_period'] = df['period_id'].isin([14, 15, 16, 17]).astype(float)
print(f"  Temporal features: period_id_scaled mean={df['period_id_scaled'].mean():.3f}  market_shock_period n={df['market_shock_period'].sum():.0f}")

available = [f for f in STAT_FEATURES if f in df.columns]
missing   = [f for f in STAT_FEATURES if f not in df.columns]
if missing:
    print(f"\nWARNING: missing columns: {missing}")

X_raw = df[available].copy()
X_raw.fillna(X_raw.mean(), inplace=True)

if 'status_band' in df.columns:
    status_map = {'red_low': 0, 'orange_low': 1, 'green': 2, 'orange_high': 3, 'red_high': 4}
    X_raw['status_band_encoded'] = df['status_band'].map(status_map).fillna(2)
    available.append('status_band_encoded')

print(f"\nFeature matrix: {X_raw.shape[0]} rows x {len(available)} features")

scaler = StandardScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X_raw), columns=available, index=df.index)

X_full_norm = X_norm.copy()
X_full_norm['goal_id']   = df['goal_id'].values
X_full_norm['period_id'] = df['period_id'].values
X_full_norm['y_attain']  = df['probability_of_hitting_target'].values

X_full_raw = X_raw.copy()
X_full_raw['goal_id']   = df['goal_id'].values
X_full_raw['period_id'] = df['period_id'].values

print(f"\nExtracting snapshots at periods {SNAPSHOT_PERIODS}...")
snapshot_dfs  = {}
snapshot_raws = {}
for sp in SNAPSHOT_PERIODS:
    mask = df['period_id'] == sp
    snapshot_dfs[sp]  = df[mask].copy().reset_index(drop=True)
    snapshot_raws[sp] = X_raw[mask].copy().reset_index(drop=True)
    # Keep goal_id, bucket_id, and target_value_final_period in per-period raw files.
    # goal_id and bucket_id needed by llm_scoring merges.
    # target_value_final_period needed by composite_score verifier signals.
    snapshot_raws[sp]['goal_id']                  = df[mask]['goal_id'].values
    snapshot_raws[sp]['bucket_id']                = df[mask]['bucket_id'].values
    snapshot_raws[sp]['target_value_final_period'] = df[mask]['target_value_final_period'].values
    snapshot_raws[sp]['period_id_scaled']          = df[mask]['period_id_scaled'].values
    snapshot_raws[sp]['market_shock_period']       = df[mask]['market_shock_period'].values
    print(f"  Period {sp}: {mask.sum()} goals")

df_anchor = snapshot_dfs[ANCHOR_PERIOD]


# ── Rich rule score formula (applied to anchor period only) ─────────────────
def compute_rich_rule_scores(df_sp, hier_df):
    """
    Rich formula for Relevance, Coherence, Integrity:
      Relevance  : 0.4*band + 0.3*sibling + 0.3*fitness, with flag penalty
      Coherence  : 0.35*hier_gap + 0.25*drift + 0.25*wgs + 0.15*status_consistency
      Integrity  : 0.3*efficiency + 0.35*delivery + 0.35*obs/exp ratio
    """
    rel_fitness  = df_sp['allocation_fitness_score'].fillna(0)
    rel_band     = (1 - df_sp['optimal_band_distance']).clip(0, 1)
    rel_sibling  = (1 - df_sp['sibling_rank_pct']).clip(0, 1)
    flag_penalty = np.where(
        df_sp.get('underfunded_flag', pd.Series([False] * len(df_sp))) |
        df_sp.get('overfunded_flag',  pd.Series([False] * len(df_sp))), 0.85, 1.0
    )
    relevance = ((0.4 * rel_band + 0.3 * rel_sibling + 0.3 * rel_fitness) * flag_penalty).clip(0, 1)

    # Coherence: hierarchy gap (actual vs equal-share-within-L2)
    expected_share = 1.0 / hier_df.set_index('l3_id')['sibling_count'].reindex(df_sp['bucket_id']).values
    actual_share   = hier_df.set_index('l3_id')['l3_share_of_l2'].reindex(df_sp['bucket_id']).values
    hier_gap   = np.abs(actual_share - expected_share) / np.clip(expected_share, 0.001, None)
    hier_score = (1 - np.clip(hier_gap, 0, 1))

    drift_vals  = df_sp['alloc_drift_std'].fillna(df_sp['alloc_drift_std'].mean())
    drift_score = (1 - (drift_vals / max(drift_vals.max(), 1e-9)).clip(0, 1))
    wgs         = df_sp['weighted_goal_status_score'].fillna(0.5)
    status_u    = df_sp['status_band_unique'].fillna(3)
    status_s    = (1 - (status_u - 1) / 4).clip(0, 1)

    coherence = (0.35 * hier_score + 0.25 * drift_score + 0.25 * wgs + 0.15 * status_s).clip(0, 1)

    step1   = df_sp['allocation_efficiency_ratio'].clip(0, 1)
    step2   = (0.5 * df_sp['delivered_output_quality_score'].clip(0, 1) +
               0.5 * df_sp['needle_move_ratio'].clip(0, 1).fillna(0))
    obs_exp = (df_sp['observed_value'] / df_sp['expected_value'].clip(lower=0.001)).clip(0, 1)
    integrity = (0.3 * step1 + 0.35 * step2 + 0.35 * obs_exp).clip(0, 1)

    return relevance, coherence, integrity


# ── Simple rule score formula (applied to non-anchor periods) ───────────────
def compute_simple_rule_scores(df_sp):
    rel_fitness  = df_sp['allocation_fitness_score'].fillna(0)
    rel_band     = (1 - df_sp['optimal_band_distance'].fillna(0.5)).clip(0, 1)
    rel_sibling  = (1 - df_sp['sibling_rank_pct'].fillna(0.5)).clip(0, 1)
    flag_penalty = np.where(
        df_sp.get('underfunded_flag', pd.Series([False] * len(df_sp))) |
        df_sp.get('overfunded_flag',  pd.Series([False] * len(df_sp))), 0.85, 1.0
    )
    relevance = ((0.4 * rel_band + 0.3 * rel_sibling + 0.3 * rel_fitness) * flag_penalty).clip(0, 1)

    drift_sp   = df_sp['alloc_drift_std'].fillna(df_sp['alloc_drift_std'].mean())
    drift_s_sp = (1 - (drift_sp / max(drift_sp.max(), 1e-9)).clip(0, 1))
    wgs_sp     = df_sp['weighted_goal_status_score'].fillna(0.5)
    coherence  = (0.5 * drift_s_sp + 0.5 * wgs_sp).clip(0, 1)

    eff_sp    = df_sp['allocation_efficiency_ratio'].fillna(0).clip(0, 1)
    needle_sp = df_sp['needle_move_ratio'].fillna(0).clip(0, 1)
    qual_sp   = df_sp['delivered_output_quality_score'].fillna(0).clip(0, 1)
    integrity = (0.4 * eff_sp + 0.3 * needle_sp + 0.3 * qual_sp).clip(0, 1)

    return relevance, coherence, integrity


# ── Compute rule scores per snapshot ────────────────────────────────────────
print("\nComputing rule scores per snapshot...")
for sp in SNAPSHOT_PERIODS:
    df_sp = snapshot_dfs[sp]
    if len(df_sp) == 0:
        continue
    if sp == ANCHOR_PERIOD:
        rel, coh, intg = compute_rich_rule_scores(df_sp, hier)
        print(f"  p{sp} (anchor): rich formula")
    else:
        rel, coh, intg = compute_simple_rule_scores(df_sp)
        print(f"  p{sp}         : simple formula")

    att = df_sp['probability_of_hitting_target'].clip(0, 1)

    rule_df = pd.DataFrame({
        'goal_idx'           : range(len(df_sp)),
        'goal_id'            : df_sp['goal_id'].values,
        'bucket_id'          : df_sp['bucket_id'].values,
        'relevance_rule'     : rel.values if hasattr(rel, 'values') else rel,
        'coherence_rule'     : coh.values if hasattr(coh, 'values') else coh,
        'integrity_rule'     : intg.values if hasattr(intg, 'values') else intg,
        'attainability_label': att.values,
    })
    rule_df.to_csv(f"rule_scores_p{sp}.csv", index=False)
    print(f"    Relevance mean={rule_df['relevance_rule'].mean():.3f}  "
          f"Coherence mean={rule_df['coherence_rule'].mean():.3f}  "
          f"Integrity mean={rule_df['integrity_rule'].mean():.3f}")

# ── Save outputs ────────────────────────────────────────────────────────────
X_full_norm.to_csv("features_full_normalized.csv", index=False)
X_full_raw.to_csv("features_full_raw.csv", index=False)

for sp, snap_raw in snapshot_raws.items():
    snap_raw.to_csv(f"features_raw_p{sp}.csv", index=False)
    print(f"OK  features_raw_p{sp}.csv  ({snap_raw.shape})")

# _poc.csv aliases are p12 for backward compat with meta_learner.py
# (which reads features_raw_poc.csv when asked for period 12). The rich
# rule formula still lives in rule_scores_p18.csv.
p12_norm = X_norm[df['period_id'] == 12].copy().reset_index(drop=True)
p12_raw  = snapshot_raws[12]
p12_raw.to_csv("features_raw_poc.csv", index=False)
p12_norm.to_csv("features_normalized_poc.csv", index=False)

p12_rules_path = "rule_scores_p12.csv"
if os.path.exists(p12_rules_path):
    pd.read_csv(p12_rules_path).to_csv("rule_scores_poc.csv", index=False)

# Anchor period snapshot kept separate as period_18_poc.csv
df_anchor.to_csv(f"period_{ANCHOR_PERIOD}_poc.csv", index=False)

with open("feature_names_poc.txt", "w") as f:
    for name in available:
        f.write(f"{name}\n")

with open("feature_scaler_poc.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\nOK  features_full_normalized.csv  ({X_full_norm.shape})")
print(f"OK  features_raw_poc.csv (alias for p12, backward compat)")
print(f"OK  rule_scores_poc.csv  (alias for p12, backward compat)")
print(f"OK  rule_scores_p{ANCHOR_PERIOD}.csv    (anchor, rich formula)")
print(f"OK  period_{ANCHOR_PERIOD}_poc.csv        (anchor snapshot)")
print(f"OK  feature_scaler_poc.pkl")
print("=" * 70)