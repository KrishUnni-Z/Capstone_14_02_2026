# Decidr Coherence Engine — System 2 Core
**Team 14-02 | iLab Capstone 36127 | UTS MDSI**

---

## Overview

System 2 is the Intelligence Core of the Decidr Coherence Engine. It takes
engineered features from the data layer and produces coherence scores across
four dimensions for every organisational goal.

```
[Data Layer — Team 3]
  features_raw_p6/12/18/24.csv
  rule_scores_p6/12/18/24.csv
  goal_dependencies.csv
        ↓
[System 2 Core — This Repo]
  LLM Scoring → Meta Learner → Composite Score → Verifier
        ↓
  composite_scores_poc.csv
  coherence_timeseries_poc.csv
  meta_learner_predictions_full.csv
        ↓
[Presentation Layer — Team 1]
  Streamlit Dashboard
```

---

## Files in This Repo

| File | Purpose |
|------|---------|
| `llm_scoring.py` | Calls llama3 and gemma3 via Ollama at 4 snapshot periods |
| `meta_learner.py` | GP model for attainability, blends engineered + LLM scores |
| `composite_score.py` | Weighted composite across 4 dimensions, portfolio summary |
| `verify_goal.py` | deepseek-r1:8b post-composite verifier, 9 hard rules |
| `score_goal.py` | On-demand inference for System 3 — single goal scoring |
| `score_group.py` | On-demand inference for System 3 — bucket-level scoring |
| `run.py` | Full pipeline runner |

---

## Dimensions and Weights

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Coherence | 35% | Consistency of decisions across hierarchy, time, and goals |
| Attainability | 25% | Likelihood of reaching target by period 24 |
| Relevance | 20% | Whether allocation level is justified |
| Integrity | 20% | Whether spend produced expected outputs |

---

## Models Used

| Model | Role | Prompt style |
|-------|------|-------------|
| llama3:latest | Scorer | Narrative analyst brief with trajectory context |
| gemma3:4b | Scorer | Threshold-based rubric per dimension |
| deepseek-r1:8b | Verifier only | Post-composite consistency judge |

---

## Setup

**Install Ollama and pull models (once only):**
```bash
ollama pull llama3:latest
ollama pull gemma3:4b
ollama pull deepseek-r1:8b
```

**Install Python dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib scipy python-dotenv
```

**Required input files (provided by Team 3 — Data Layer):**
```
features_raw_p6.csv       features_raw_p12.csv
features_raw_p18.csv      features_raw_p24.csv
rule_scores_p6.csv        rule_scores_p12.csv (= rule_scores_poc.csv)
rule_scores_p18.csv       rule_scores_p24.csv
goal_dependencies.csv
features_full_normalized.csv
features_full_raw.csv
period_12_poc.csv
analytical_flat.csv
llm_predictions_poc.csv   (from previous run, for --skip03)
```

---

## Running

**Full pipeline (LLM scoring at 4 periods + GP + composite):**
```bash
python run.py
```

**Skip LLM scoring — reuse existing predictions:**
```bash
python run.py --skip03
```

**Test LLMs on one goal before full run:**
```bash
python llm_scoring.py --test
```

**Run LLM scoring for one period only:**
```bash
python llm_scoring.py --period 12
```

**On-demand inference (called by System 3):**
```bash
python score_goal.py --goal_id 0
python score_goal.py --goal_id 0 --output result.json
python score_group.py --bucket "Paid Acquisition"
```

---

## Output Files (consumed by Team 1 — Presentation Layer)

| File | Description |
|------|-------------|
| `composite_scores_poc.csv` | Per-goal scores at period 12 with flags |
| `meta_learner_predictions_full.csv` | 840 rows — all goals all periods |
| `coherence_timeseries_poc.csv` | Portfolio coherence per period (for time series chart) |
| `portfolio_timeseries_poc.csv` | Per-bucket coherence per period |
| `portfolio_summary_poc.csv` | Per-bucket aggregation at period 12 |
| `forward_projection_poc.csv` | +6 and +12 period trajectory per goal |

---

## Interface for System 3

System 3 calls `score_goal()` directly after performing feature engineering:

```python
from score_goal import score_goal

result = score_goal(
    goal_row  = goal_row,   # one row from features_raw_p12.csv
    rule_row  = rule_row,   # one row from rule_scores_poc.csv
    verbose   = False
)

# result contains:
# attainability, relevance, coherence, integrity, overall
# flags, narrative, verified, adjustments
# dependencies block
```

For bucket-level scoring:
```python
from score_group import score_group

result = score_group(
    goal_rows      = [row1, row2, row3],
    rule_rows      = [rule1, rule2, rule3],
    bucket_context = {
        "l2_name"     : "Paid Acquisition",
        "l1_name"     : "Marketing",
        "l2_alloc_pct": 0.14,
        "l1_alloc_pct": 0.40,
        "n_siblings"  : 3,
    }
)
```

---

## Architecture Notes

- LLM scoring runs at 4 snapshot periods (6, 12, 18, 24)
- Each period's prompt includes trajectory context from the prior checkpoint
- GP trained on all 840 rows (35 goals × 24 periods), predicts on all 840
- LLM scores blended with GP and engineered scores via isotonic calibration
- Verifier adjustments capped at ±0.10 per dimension
- All inference runs locally via Ollama — no internet needed after model pull
