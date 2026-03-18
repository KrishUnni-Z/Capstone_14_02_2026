"""
02_feature_engineering_poc.py — v2 (all 4 dimensions)
Computes rule-based scores for Relevance, Coherence, Integrity
alongside statistical features for Attainability.

Output:
  features_normalized_poc.csv  — z-scored features for meta-learner
  features_raw_poc.csv         — raw features
  rule_scores_poc.csv          — rule-based scores for all 4 dimensions
  feature_names_poc.txt
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

print("=" * 70)
print("DECIDR SYSTEM 2 — PoC")
print("02  Feature engineering (all 4 dimensions)")
print("=" * 70)

df = pd.read_csv("period_12_poc.csv")

# ── Statistical features for Attainability meta-learner ───────────────────────
NUMERIC_FEATURES = [
    'trailing_6_period_slope',
    'variance_from_target',
    'delivered_output_quality_score',
    'delivered_output_quantity',
    'allocation_percentage_of_parent',
    'volatility_measure',
    'target_value_final_period',
    'observed_value',
    'allocated_amount',
    'allocated_time_hours',
]

available = [f for f in NUMERIC_FEATURES if f in df.columns]
X_raw = df[available].copy()
X_raw.fillna(X_raw.mean(), inplace=True)

if 'status_band' in df.columns:
    status_map = {'red_low': 0, 'orange_low': 1, 'green': 2,
                  'orange_high': 3, 'red_high': 4}
    X_raw['status_band_encoded'] = df['status_band'].map(status_map).fillna(2)
    available.append('status_band_encoded')

scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X_raw), columns=available, index=df.index)

print(f"\nAttainability features : {len(available)}")

# ── Rule-based scores ─────────────────────────────────────────────────────────

# --- RELEVANCE: Is the allocation justified? ---------------------------------
# Symmetric band — both over and under allocation are penalised
# range_position_score: red_low=0.05, green=~0.53, red_high=0.95
# Triangle function peaked at 0.5 (green centre)
rps = df['range_position_score'].clip(0, 1)
relevance_band = (1 - 2 * (rps - 0.5).abs()).clip(0, 1)

# Penalise hard flags
flag_penalty = np.where(df['underfunded_flag'] | df['overfunded_flag'], 0.85, 1.0)
relevance_rule = (relevance_band * flag_penalty).clip(0, 1)

print(f"\nRelevance rule score   : {relevance_rule.min():.3f} - {relevance_rule.max():.3f}  mean={relevance_rule.mean():.3f}")

# --- COHERENCE: Is allocation internally consistent? -------------------------
# 1. Status band alignment score
band_score_map = {'green': 1.0, 'orange_low': 0.55, 'orange_high': 0.55,
                  'red_low': 0.15, 'red_high': 0.15}
band_score = df['status_band'].map(band_score_map).fillna(0.5)

# 2. Allocation within optimal band
alloc_pct  = df['allocation_percentage_of_total_bucket']
opt_min    = df['optimal_allocation_min']
opt_max    = df['optimal_allocation_max']
opt_centre = (opt_min + opt_max) / 2
opt_range  = (opt_max - opt_min).clip(lower=1e-9)

# Distance from optimal centre, normalised
alloc_dist  = (alloc_pct - opt_centre).abs() / opt_range
alloc_score = (1 - alloc_dist.clip(0, 1))

coherence_rule = (0.6 * band_score + 0.4 * alloc_score).clip(0, 1)

print(f"Coherence rule score   : {coherence_rule.min():.3f} - {coherence_rule.max():.3f}  mean={coherence_rule.mean():.3f}")

# --- INTEGRITY: Do outcomes match allocation? --------------------------------
# 1. Observed vs expected (clipped to 1.0 max — exceeding is fine)
obs_exp = (df['observed_value'] / df['expected_value'].clip(lower=1e-9)).clip(0, 1)

# 2. Allocation efficiency ratio (normalise 0-1)
eff_min = df['allocation_efficiency_ratio'].min()
eff_max = df['allocation_efficiency_ratio'].max()
eff_norm = ((df['allocation_efficiency_ratio'] - eff_min) / (eff_max - eff_min + 1e-9)).clip(0, 1)

# 3. Output quality (already 0-1)
quality = df['delivered_output_quality_score'].clip(0, 1)

integrity_rule = (0.4 * obs_exp + 0.3 * eff_norm + 0.3 * quality).clip(0, 1)

print(f"Integrity rule score   : {integrity_rule.min():.3f} - {integrity_rule.max():.3f}  mean={integrity_rule.mean():.3f}")

# --- ATTAINABILITY: existing ground truth label ------------------------------
attainability_label = df['probability_of_hitting_target'].clip(0, 1)
print(f"Attainability label    : {attainability_label.min():.3f} - {attainability_label.max():.3f}  mean={attainability_label.mean():.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
X_normalized.to_csv("features_normalized_poc.csv", index=False)
X_raw.to_csv("features_raw_poc.csv", index=False)

rule_scores = pd.DataFrame({
    'goal_idx'          : range(len(df)),
    'goal_id'           : df['goal_id'].values,
    'relevance_rule'    : relevance_rule.values,
    'coherence_rule'    : coherence_rule.values,
    'integrity_rule'    : integrity_rule.values,
    'attainability_label': attainability_label.values,
})
rule_scores.to_csv("rule_scores_poc.csv", index=False)

with open("feature_names_poc.txt", "w") as f:
    for name in available:
        f.write(f"{name}\n")

print(f"\n✓ Saved features_normalized_poc.csv  ({X_normalized.shape})")
print(f"✓ Saved features_raw_poc.csv")
print(f"✓ Saved rule_scores_poc.csv")
print(f"✓ Saved feature_names_poc.txt")
print("=" * 70)
