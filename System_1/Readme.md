# Decidr Coherence Engine — Presentation Layer
**Team 14-02 | iLab Capstone 36127 | UTS MDSI**

---

## Overview

The Presentation Layer reads the scored output CSVs produced by System 2
(Intelligence Core) and renders them as an interactive Streamlit dashboard.
No LLM calls, no scoring logic — purely visualisation and user interaction.

```
[System 2 Output CSVs]
  composite_scores_poc.csv
  coherence_timeseries_poc.csv
  portfolio_timeseries_poc.csv
  meta_learner_predictions_full.csv
  forward_projection_poc.csv
  portfolio_summary_poc.csv
        ↓
[Presentation Layer — This Repo]
  Streamlit Dashboard
        ↓
[User — Browser at localhost:8501]
```

---

## Files in This Repo

| File | Purpose |
|------|---------|
| `dashboard_app.py` | Main Streamlit app — all dashboard panels |
| `explanations.py` | GP feature importance and calibration charts (static PNGs) |
| `dashboard.py` | Legacy 6-panel static dashboard (PNG output) |

---

## Dashboard Panels

**Panel 1 — Portfolio Health Overview**
KPI cards: overall coherence score, at-risk goal count, goal retention rate
at period 24, weakest bucket. Snapshot at period 12.

**Panel 2 — Coherence Over Time**
Line chart across all 24 periods. Vertical markers at periods 10-12
(budget shock) and 14-17 (market shock). Shows coherence drop and recovery.

**Panel 3 — Goal Breakdown**
Heatmap of all 35 goals × 4 dimensions. Colour coded by score.
Filterable by bucket, scenario story, at-risk status.

**Panel 4 — Shock Analysis**
Pre-shock (periods 7-9) vs during-shock (10-12) vs post-shock (13-17)
coherence per bucket. Which buckets were most affected.

**Panel 5 — Forward Projection**
Per-goal trajectory at +6 and +12 periods. Improving vs degrading.

**Panel 6 — Reallocation Recommendations**
Under 20% budget cut scenario — which goals to protect and which to cut
based on composite score, scenario story, and recovery trajectory.

---

## Setup

**Install dependencies:**
```bash
pip install streamlit pandas numpy matplotlib plotly
```

**Required input files (produced by System 2):**
```
composite_scores_poc.csv
coherence_timeseries_poc.csv
portfolio_timeseries_poc.csv
meta_learner_predictions_full.csv
forward_projection_poc.csv
portfolio_summary_poc.csv
goal_dependencies.csv
goals.csv
buckets.csv
```

---

## Running

```bash
streamlit run dashboard_app.py
```

Opens at `http://localhost:8501`

---

## Input File Schemas

**composite_scores_poc.csv** — one row per goal at period 12
```
goal_id, l2_name, l1_name
attainability, relevance, coherence, integrity
composite, composite_adjusted
at_risk, critical, weakest_dim, weakest_score
verified_composite, verified, flags, narrative
```

**coherence_timeseries_poc.csv** — one row per period
```
period_id, avg_composite, at_risk_count,
avg_coherence, avg_attainability
```

**portfolio_timeseries_poc.csv** — one row per period per bucket
```
period_id, l2_name, l1_name,
n_goals, avg_composite, at_risk_count
```

**meta_learner_predictions_full.csv** — 840 rows (35 goals × 24 periods)
```
goal_id, period_id
attainability, relevance, coherence, integrity, overall
gp_mean, gp_std, uncertain
```

---

## Notes

Do not modify any of the input CSV schemas — these are produced by System 2
and any schema change needs to be coordinated with Team 14-02.

If System 2 adds new fields to the output (e.g. shock flags, dependency fields)
the dashboard can optionally display them but should not break if they are absent.
Use `.get()` or check column existence before reading any new field.

The `narrative` field in `composite_scores_poc.csv` contains a natural language
summary produced by the deepseek verifier. This can be surfaced directly to users
without any further processing.
