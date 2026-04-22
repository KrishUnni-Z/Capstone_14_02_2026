
# Decidr Coherence Engine — Data Layer
**Team 14-02 | iLab Capstone 36127 | UTS MDSI**

---

## Overview

The Data Layer is responsible for loading, validating, and transforming the
raw source data into engineered feature CSVs that System 2 (Intelligence Core)
consumes for scoring.

```
[Raw Source CSVs — 8 files]
  analytical_flat.csv, buckets.csv, goals.csv, allocations.csv
  outputs.csv, metrics.csv, derived_fields.csv, periods.csv
        ↓
[Data Layer — This Repo]
  load_data.py → feature_engineering.py → infer_dependencies.py
        ↓
[Output CSVs — consumed by System 2]
  features_raw_p6/12/18/24.csv
  rule_scores_p6/12/18/24.csv
  goal_dependencies.csv
  features_full_normalized.csv
```

---

## Files in This Repo

| File | Purpose |
|------|---------|
| `load_data.py` | Validates all 8 source CSVs, extracts period snapshots |
| `feature_engineering.py` | Builds 33 engineered features, computes rule scores at 4 periods |
| `infer_dependencies.py` | Infers cross-goal dependencies via llama3, saves dependency map |

---

## Features Produced (33 total)

**Attainability signals:**
`trailing_6_period_slope`, `variance_from_target`, `volatility_measure`,
`time_to_green_estimate`

**Relevance signals:**
`allocation_percentage_of_parent`, `optimal_band_distance`, `sibling_rank_pct`,
`scenario_encoded`, `allocation_fitness_score`

**Coherence signals:**
`l3_share_of_l2`, `l3_share_of_l1`, `alloc_drift_std`,
`weighted_goal_status_score`, `status_band_unique`

**Integrity signals:**
`delivered_output_quality_score`, `delivered_output_quantity`,
`allocation_efficiency_ratio`, `needle_move_ratio`, `output_cost_per_unit`

**Dependency signals:**
`n_dependencies`, `n_dependents`, `dependency_risk_encoded`, `dep_avg_attain`

**Shock signals:**
`budget_shock_exposure`, `shock_alloc_impact`, `recovery_period_estimate`,
`market_shock_vulnerable`, `market_shock_forward_risk`, `recovery_window_remaining`

**Additional:**
`observed_value`, `allocated_amount`, `allocated_time_hours`

---

## Shock Parameters (configurable)

In `feature_engineering.py`:
```python
BUDGET_SHOCK_PERIODS  = {10, 11, 12}   # 20% budget reduction
MARKET_SHOCK_PERIOD   = 14             # 15% growth metric reduction
MARKET_RECOVERY_END   = 17             # recovery complete by period 17
RECOVERY_SPEED_FACTOR = 1.0            # increase to assume faster recovery
```

---

## Setup

**Required source CSVs (place in same folder):**
```
analytical_flat.csv    buckets.csv      goals.csv
allocations.csv        outputs.csv      metrics.csv
derived_fields.csv     periods.csv
```

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn python-dotenv
```

**Ollama needed for infer_dependencies.py:**
```bash
ollama pull llama3:latest
```

---

## Running

```bash
python load_data.py           # step 1 — validate and load
python feature_engineering.py # step 2 — build features
python infer_dependencies.py  # step 3 — infer dependencies
```

Or via the full pipeline runner in System 2:
```bash
python run.py   # runs all steps in order
```

---

## Output Files

| File | Rows | Description |
|------|------|-------------|
| `features_raw_p6.csv` | 35 | Features at period 6 snapshot |
| `features_raw_p12.csv` | 35 | Features at period 12 snapshot |
| `features_raw_p18.csv` | 35 | Features at period 18 snapshot |
| `features_raw_p24.csv` | 35 | Features at period 24 snapshot |
| `features_raw_poc.csv` | 35 | Alias for period 12 (backward compat) |
| `rule_scores_p6/12/18/24.csv` | 35 each | Rule-based R/C/I anchors per period |
| `rule_scores_poc.csv` | 35 | Alias for period 12 rule scores |
| `features_full_normalized.csv` | 840 | All periods normalised (GP training) |
| `features_full_raw.csv` | 840 | All periods raw |
| `period_12_poc.csv` | 35 | Period 12 full row snapshot |
| `goal_dependencies.csv` | 35 | Cross-goal dependency map |
| `analytical_full.csv` | 840 | Validated flat table |

---

## Notes for System 2

System 2 reads these files directly. File names must match exactly.
If you add new features, also update `FEATURES` list in `feature_engineering.py`
so they are included in the normalised output for the GP.

The `rule_scores_poc.csv` file must contain at minimum:
`goal_id`, `relevance_rule`, `coherence_rule`, `integrity_rule`, `attainability_label`
