"""
02_feature_engineering_poc.py — v4 (Tom's signals)
Builds features from all source files using the full bucket hierarchy.

Sources:
  buckets.csv, goals.csv, allocations.csv, outputs.csv,
  metrics.csv, derived_fields.csv, analytical_flat.csv

Outputs:
  features_full_normalized.csv  — all 840 rows for GP training
  features_full_raw.csv
  features_normalized_poc.csv   — period 12 only (35 rows)
  features_raw_poc.csv
  rule_scores_poc.csv           — improved rule scores for R/C/I
  feature_scaler_poc.pkl
  feature_names_poc.txt
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("DECIDR SYSTEM 2 — Feature Engineering v4 (Tom's signals)")
print("=" * 70)

# ── Load all source files ─────────────────────────────────────────────────────
flat    = pd.read_csv("analytical_flat.csv")
buckets = pd.read_csv("buckets.csv")
goals   = pd.read_csv("goals.csv")
allocs  = pd.read_csv("allocations.csv")
outputs = pd.read_csv("outputs.csv")
metrics = pd.read_csv("metrics.csv")
derived = pd.read_csv("derived_fields.csv")

print(f"\nLoaded: flat={flat.shape}  buckets={buckets.shape}  goals={goals.shape}")
print(f"        allocs={allocs.shape}  outputs={outputs.shape}  metrics={metrics.shape}  derived={derived.shape}")

# ── Build bucket hierarchy ────────────────────────────────────────────────────
l1 = buckets[buckets['bucket_level']==1][['bucket_id','allocation_percentage_of_total']].copy()
l1.columns = ['l1_id','l1_alloc_pct']
l2 = buckets[buckets['bucket_level']==2][['bucket_id','parent_bucket_id','allocation_percentage_of_total']].copy()
l2.columns = ['l2_id','l1_id','l2_alloc_pct']
l3 = buckets[buckets['bucket_level']==3][['bucket_id','parent_bucket_id','allocation_percentage_of_total','is_leaf']].copy()
l3.columns = ['l3_id','l2_id','l3_alloc_pct','is_leaf']

hier = l3.merge(l2, on='l2_id').merge(l1, on='l1_id')
hier['l3_share_of_l2'] = hier['l3_alloc_pct'] / hier['l2_alloc_pct']
hier['l3_share_of_l1'] = hier['l3_alloc_pct'] / hier['l1_alloc_pct']

# Sibling rank within same L2 parent
hier['sibling_count']   = hier.groupby('l2_id')['l3_id'].transform('count')
hier['sibling_rank']    = hier.groupby('l2_id')['l3_alloc_pct'].rank(ascending=False)
hier['sibling_rank_pct']= (hier['sibling_rank'] - 1) / (hier['sibling_count'] - 1).clip(lower=1)

print(f"\nHierarchy: {len(hier)} L3 goals mapped to L2 and L1 parents")
print(f"  l3_share_of_l2: mean={hier['l3_share_of_l2'].mean():.3f}")
print(f"  sibling_rank_pct: 0=best funded, 1=worst funded in group")

# Goal-level static features
scenario_map = {'underfunded': 0.0, 'dynamic': 0.33, 'optimal': 0.67, 'overfunded': 1.0}
goals['scenario_encoded'] = goals['scenario_story'].map(scenario_map)

goal_static = goals[['goal_id','bucket_id','scenario_encoded',
                      'minimum_viable_allocation','optimal_allocation_min',
                      'optimal_allocation_max']].merge(
    hier[['l3_id','l2_id','l1_id','l3_share_of_l2','l3_share_of_l1','sibling_rank_pct']],
    left_on='bucket_id', right_on='l3_id', how='left'
)

# Temporal features: alloc drift per goal across all periods
alloc_drift = allocs.groupby('bucket_id').agg(
    alloc_drift_std=('allocation_percentage_of_parent','std'),
    alloc_mean=('allocation_percentage_of_parent','mean'),
).reset_index()

# Status band consistency across periods
status_consist = flat.groupby('goal_id').agg(
    status_band_unique=('status_band','nunique'),
).reset_index()

# Needle move ratio: observed / expected (period level)
metrics['needle_move_ratio'] = (
    metrics['observed_value'] /
    metrics['expected_value'].clip(lower=0.001)
).clip(0, 2)

print("\nBuilding per-period feature matrix (840 rows)...")

# ── Merge everything into one row per (goal, period) ─────────────────────────
df = flat.copy()

df = df.merge(
    derived[['goal_id','period_id','weighted_goal_status_score',
             'allocation_fitness_score','time_to_green_estimate']],
    on=['goal_id','period_id'],
    how='left'
)

if 'time_to_green_estimate_y' in df.columns:
    df['time_to_green_estimate'] = df['time_to_green_estimate_y']
elif 'time_to_green_estimate_x' in df.columns:
    df['time_to_green_estimate'] = df['time_to_green_estimate_x']

# Join needle_move_ratio from metrics
df = df.merge(
    metrics[['goal_id','period_id','needle_move_ratio']],
    on=['goal_id','period_id'], how='left'
)

# Join goal-level static features
df = df.merge(
    goal_static[['goal_id','scenario_encoded','l3_share_of_l2',
                 'l3_share_of_l1','sibling_rank_pct']],
    on='goal_id', how='left'
)

# Join temporal alloc drift (goal-level, static across periods)
df = df.merge(
    alloc_drift[['bucket_id','alloc_drift_std']],
    on='bucket_id', how='left'
)

# Join status consistency
df = df.merge(status_consist, on='goal_id', how='left')

# Dependency features from infer_dependencies.py
import os as _os
if _os.path.exists("goal_dependencies.csv"):
    dep_df = pd.read_csv("goal_dependencies.csv")
    dep_merge = dep_df[['goal_id','n_dependencies','n_dependents','dependency_risk','dep_avg_attain']].copy()
    dep_merge['dependency_risk_encoded'] = dep_merge['dependency_risk'].map(
        {'none': 0.0, 'low': 0.33, 'medium': 0.67, 'high': 1.0}).fillna(0.0)
    df = df.merge(dep_merge[['goal_id','n_dependencies','n_dependents',
                              'dependency_risk_encoded','dep_avg_attain']], on='goal_id', how='left')
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
    print("  goal_dependencies.csv not found — dependency features set to 0")

# Optimal band distance: how far is actual from optimal centre
opt_centre  = (df['optimal_allocation_min'] + df['optimal_allocation_max']) / 2
opt_range   = (df['optimal_allocation_max'] - df['optimal_allocation_min']).clip(lower=1e-9)
df['optimal_band_distance'] = ((df['allocation_percentage_of_total_bucket'] - opt_centre).abs() / opt_range).clip(0, 1)

print(f"  Merged shape: {df.shape}")

# ── Define feature columns ────────────────────────────────────────────────────
STAT_FEATURES = [
    # Attainability signals
    'trailing_6_period_slope',
    'variance_from_target',
    'volatility_measure',
    'time_to_green_estimate',
    # Relevance signals
    'allocation_percentage_of_parent',
    'optimal_band_distance',
    'sibling_rank_pct',
    'scenario_encoded',
    'allocation_fitness_score',
    # Coherence signals
    'l3_share_of_l2',
    'l3_share_of_l1',
    'alloc_drift_std',
    'weighted_goal_status_score',
    'status_band_unique',
    # Integrity signals
    'delivered_output_quality_score',
    'delivered_output_quantity',
    'allocation_efficiency_ratio',
    'needle_move_ratio',
    'output_cost_per_unit',
    # Dependency signals
    'n_dependencies',
    'n_dependents',
    'dependency_risk_encoded',
    'dep_avg_attain',
    # Additional
    'observed_value',
    'allocated_amount',
    'allocated_time_hours',
]

available = [f for f in STAT_FEATURES if f in df.columns]
missing   = [f for f in STAT_FEATURES if f not in df.columns]
if missing:
    print(f"\nWARNING: missing columns: {missing}")

X_raw = df[available].copy()
X_raw.fillna(X_raw.mean(), inplace=True)

# status_band encoded
if 'status_band' in df.columns:
    status_map = {'red_low':0,'orange_low':1,'green':2,'orange_high':3,'red_high':4}
    X_raw['status_band_encoded'] = df['status_band'].map(status_map).fillna(2)
    available.append('status_band_encoded')

print(f"\nFeature matrix: {X_raw.shape[0]} rows × {len(available)} features")
print(f"Features: {available}")

# ── Fit scaler on full 840 rows ───────────────────────────────────────────────
scaler   = StandardScaler()
X_norm   = pd.DataFrame(scaler.fit_transform(X_raw), columns=available, index=df.index)

# Add metadata columns
X_full_norm = X_norm.copy()
X_full_norm['goal_id']   = df['goal_id'].values
X_full_norm['period_id'] = df['period_id'].values
X_full_norm['y_attain']  = df['probability_of_hitting_target'].values

X_full_raw = X_raw.copy()
X_full_raw['goal_id']   = df['goal_id'].values
X_full_raw['period_id'] = df['period_id'].values

# ── Multi-period snapshots: 6, 12, 18, 24 ───────────────────────────────────
SNAPSHOT_PERIODS = [6, 12, 18, 24]
print(f"\nExtracting snapshots at periods {SNAPSHOT_PERIODS}...")

# Keep period 12 as primary for backward compatibility
p12_mask     = df['period_id'] == 12
X_p12_norm   = X_norm[p12_mask].copy().reset_index(drop=True)
X_p12_raw    = X_raw[p12_mask].copy().reset_index(drop=True)
df_p12       = df[p12_mask].copy().reset_index(drop=True)

# Build snapshot dataframes for all 4 periods
snapshot_dfs  = {}
snapshot_raws = {}
for sp in SNAPSHOT_PERIODS:
    mask = df['period_id'] == sp
    snapshot_dfs[sp]  = df[mask].copy().reset_index(drop=True)
    snapshot_raws[sp] = X_raw[mask].copy().reset_index(drop=True)
    print(f"  Period {sp}: {mask.sum()} goals")

print(f"\nPeriod 12 snapshot: {len(df_p12)} goals")

# ── Improved rule scores ──────────────────────────────────────────────────────
print("\nComputing improved rule scores...")

# RELEVANCE — Tom's signal: allocation fitness + band distance + sibling rank
rel_fitness  = df_p12['allocation_fitness_score'].fillna(0)
rel_band     = (1 - df_p12['optimal_band_distance']).clip(0, 1)
rel_sibling  = (1 - df_p12['sibling_rank_pct']).clip(0, 1)  # 0=worst, 1=best
# Penalise funding flags
flag_penalty = np.where(df_p12['underfunded_flag'] | df_p12['overfunded_flag'], 0.85, 1.0)
relevance_rule = (0.4 * rel_band + 0.3 * rel_sibling + 0.3 * rel_fitness) * flag_penalty
relevance_rule = relevance_rule.clip(0, 1)
print(f"  Relevance: mean={relevance_rule.mean():.3f}  range=[{relevance_rule.min():.3f}, {relevance_rule.max():.3f}]")

# COHERENCE — Tom's signal: hierarchy consistency + temporal drift + status consistency
# Hierarchy: how well does L3 share match expected (equal split within L2)
expected_share = 1.0 / hier.set_index('l3_id')['sibling_count'].reindex(df_p12['bucket_id']).values
actual_share   = hier.set_index('l3_id')['l3_share_of_l2'].reindex(df_p12['bucket_id']).values
hier_gap       = np.abs(actual_share - expected_share) / expected_share.clip(min=0.001)
hier_score     = (1 - hier_gap.clip(0, 1))

# Temporal drift: lower drift = more coherent
alloc_drift_vals = df_p12['alloc_drift_std'].fillna(df_p12['alloc_drift_std'].mean())
drift_score      = (1 - (alloc_drift_vals / alloc_drift_vals.max()).clip(0, 1))

# Status consistency: fewer band changes = more coherent
status_u     = df_p12['status_band_unique'].fillna(3)
status_score = (1 - (status_u - 1) / 4).clip(0, 1)

# Weighted goal status
wgs          = df_p12['weighted_goal_status_score'].fillna(0.5)

coherence_rule = (0.35 * hier_score + 0.25 * drift_score +
                  0.25 * wgs + 0.15 * status_score).clip(0, 1)
print(f"  Coherence: mean={coherence_rule.mean():.3f}  range=[{coherence_rule.min():.3f}, {coherence_rule.max():.3f}]")

# INTEGRITY — Tom's three-step pipeline
# Step 1: Allocation quality (efficiency)
step1 = df_p12['allocation_efficiency_ratio'].clip(0, 1)

# Step 2: Delivery quality
step2 = (0.5 * df_p12['delivered_output_quality_score'].clip(0, 1) +
         0.5 * df_p12['needle_move_ratio'].clip(0, 1).fillna(0))

# Step 3: Needle moved (observed vs expected)
obs_exp = (df_p12['observed_value'] / df_p12['expected_value'].clip(lower=0.001)).clip(0, 1)

integrity_rule = (0.3 * step1 + 0.35 * step2 + 0.35 * obs_exp).clip(0, 1)
print(f"  Integrity: mean={integrity_rule.mean():.3f}  range=[{integrity_rule.min():.3f}, {integrity_rule.max():.3f}]")

attainability_label = df_p12['probability_of_hitting_target'].clip(0, 1)
print(f"  Attainability GT: mean={attainability_label.mean():.3f}  range=[{attainability_label.min():.3f}, {attainability_label.max():.3f}]")

# ── Save ──────────────────────────────────────────────────────────────────────
X_full_norm.to_csv("features_full_normalized.csv", index=False)
X_full_raw.to_csv("features_full_raw.csv", index=False)
X_p12_norm.to_csv("features_normalized_poc.csv", index=False)
X_p12_raw.to_csv("features_raw_poc.csv", index=False)

# Save per-snapshot raw CSVs for multi-period LLM scoring
for sp, snap_raw in snapshot_raws.items():
    snap_raw.to_csv(f"features_raw_p{sp}.csv", index=False)
    print(f"✓ features_raw_p{sp}.csv  ({snap_raw.shape})")

pd.DataFrame({
    'goal_idx'           : range(len(df_p12)),
    'goal_id'            : df_p12['goal_id'].values,
    'bucket_id'          : df_p12['bucket_id'].values,
    'relevance_rule'     : relevance_rule.values,
    'coherence_rule'     : coherence_rule.values,
    'integrity_rule'     : integrity_rule.values,
    'attainability_label': attainability_label.values,
}).to_csv("rule_scores_poc.csv", index=False)

# Compute and save rule scores for each snapshot period
for sp in SNAPSHOT_PERIODS:
    if sp == 12:
        continue   # already done above
    df_sp = snapshot_dfs[sp]
    if len(df_sp) == 0:
        continue

    # Relevance
    rel_fitness_sp  = df_sp['allocation_fitness_score'].fillna(0)
    rel_band_sp     = (1 - df_sp['optimal_band_distance'].fillna(0.5)).clip(0, 1)
    rel_sibling_sp  = (1 - df_sp['sibling_rank_pct'].fillna(0.5)).clip(0, 1)
    flag_penalty_sp = np.where(
        df_sp.get('underfunded_flag', pd.Series([False]*len(df_sp))) |
        df_sp.get('overfunded_flag',  pd.Series([False]*len(df_sp))), 0.85, 1.0)
    rel_sp = ((0.4 * rel_band_sp + 0.3 * rel_sibling_sp + 0.3 * rel_fitness_sp) * flag_penalty_sp).clip(0,1)

    # Coherence
    drift_sp   = df_sp['alloc_drift_std'].fillna(df_sp['alloc_drift_std'].mean())
    drift_s_sp = (1 - (drift_sp / max(drift_sp.max(), 1e-9)).clip(0,1))
    wgs_sp     = df_sp['weighted_goal_status_score'].fillna(0.5)
    coh_sp     = (0.5 * drift_s_sp + 0.5 * wgs_sp).clip(0,1)

    # Integrity
    eff_sp    = df_sp['allocation_efficiency_ratio'].fillna(0).clip(0,1)
    needle_sp = df_sp['needle_move_ratio'].fillna(0).clip(0,1)
    qual_sp   = df_sp['delivered_output_quality_score'].fillna(0).clip(0,1)
    int_sp    = (0.4 * eff_sp + 0.3 * needle_sp + 0.3 * qual_sp).clip(0,1)

    att_sp = df_sp['probability_of_hitting_target'].clip(0,1)

    pd.DataFrame({
        'goal_idx'           : range(len(df_sp)),
        'goal_id'            : df_sp['goal_id'].values,
        'bucket_id'          : df_sp['bucket_id'].values,
        'relevance_rule'     : rel_sp.values,
        'coherence_rule'     : coh_sp.values,
        'integrity_rule'     : int_sp.values,
        'attainability_label': att_sp.values,
    }).to_csv(f"rule_scores_p{sp}.csv", index=False)
    print(f"✓ rule_scores_p{sp}.csv")

# Save period 12 snapshot
df_p12.to_csv("period_12_poc.csv", index=False)

with open("feature_names_poc.txt","w") as f:
    for name in available:
        f.write(f"{name}\n")

with open("feature_scaler_poc.pkl","wb") as f:
    pickle.dump(scaler, f)

print(f"\n✓ features_full_normalized.csv  ({X_full_norm.shape})")
print(f"✓ features_full_raw.csv")
print(f"✓ features_normalized_poc.csv   ({X_p12_norm.shape})")
print(f"✓ features_raw_poc.csv")
print(f"✓ rule_scores_poc.csv  (improved Relevance, Coherence, Integrity)")
print(f"✓ period_12_poc.csv")
print(f"✓ feature_scaler_poc.pkl")
print("=" * 70)
