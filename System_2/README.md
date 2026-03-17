# Decidr Coherence Engine — System 2
**Goal Scoring & Intelligence Layer**
UTS Innovation Lab | MDSI Capstone 2026

---

## Overview

System 2 is the scoring brain of the Decidr Coherence Engine. It receives structured goal data from System 1, scores each goal across four coherence dimensions, and returns a structured score payload for System 1 to interpret to the user.

```
System 1 (input)
      ↓  structured JSON
  System 2  ←── you are here
      ↓  score payload  
System 1 (output)
```

### Four Scoring Dimensions

| Dimension | Description | Method |
|---|---|---|
| **Attainability** | Probability of hitting target by period 24 | ElasticNet / Ridge meta-learner + LLM |
| **Relevance** | Is the allocation level justified given the goal? | Rule-based + LLM blend |
| **Coherence** | Is the allocation internally consistent with goals? | Rule-based + LLM blend |
| **Integrity** | Do outcomes match what the allocation should produce? | Rule-based + LLM blend |

All scores are continuous 0.0–1.0. An **Overall** score is the equal-weighted mean of all four.

---

## Repo Structure

```
system2/
│
├── data/
│   └── analytical_flat.csv           ← source data (not committed, add manually)
│
├── pipeline/
│   ├── 01_load_data.py               ← ingest + extract period 12 snapshot
│   ├── 02_feature_engineering_poc.py ← feature selection, normalisation, rule scores
│   ├── 03_llm_predictions_poc.py     ← LLM scoring via Ollama REST API
│   ├── 04_meta_learner_poc.py        ← ElasticNet meta-learner (attainability)
│   ├── 04_meta_learner_poc_ridge.py  ← Ridge variant for comparison
│   ├── 05_explanations_poc.py        ← SHAP explanations
│   └── 06_demo_poc.py                ← dashboard + demo output
│
├── outputs/                          ← generated files (gitignored)
│   ├── *.csv
│   └── *.png
│
├── run_poc.py                        ← master runner (entry point)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone and create environment
```bash
git clone <repo-url>
cd system2

python -m venv decidr_env

# Windows
decidr_env\Scripts\activate

# Mac / Linux
source decidr_env/bin/activate

pip install -r requirements.txt
```

### 2. Add source data
Place `analytical_flat.csv` inside the `data/` folder.

### 3. Install Ollama and pull model
```bash
# Mac
brew install ollama

# Windows — download from https://ollama.com/download

# Pull the model (one-time, ~4.7 GB)
ollama pull llama3
```

### 4. Verify setup
```bash
python -c "import pandas, numpy, sklearn, matplotlib; print('packages ok')"
ollama run llama3 "say ready"
```

---

## Running the Pipeline

Ollama runs as a background service on Windows — no need to start it manually.
On Mac/Linux, run `ollama serve` in a separate terminal first.

```bash
# Run both variants and compare
python run_poc.py

# Run a single variant
python run_poc.py --elastic
python run_poc.py --ridge
```

### What gets produced

| File | Description |
|---|---|
| `period_12_poc.csv` | Period 12 snapshot — 35 goals |
| `features_normalized_poc.csv` | Z-scored feature matrix |
| `features_raw_poc.csv` | Raw feature values |
| `rule_scores_poc.csv` | Rule-based scores for all 4 dimensions |
| `llm_predictions_poc.csv` | LLM scores for all 4 dimensions |
| `meta_learner_predictions_poc.csv` | Final scores — all 4 + overall |
| `shap_summary_poc.png` | Feature impact beeswarm |
| `all_scores_poc.png` | All 4 scores per goal |
| `demo_dashboard_poc.png` | Full 8-panel dashboard |

When running both variants, outputs are saved with `_elastic` / `_ridge` suffixes.

---

## Architecture

### Scoring Pipeline

```
analytical_flat.csv
        ↓
  01  Load → period_12_poc.csv (35 goals)
        ↓
  02  Feature engineering
      ├── 11 statistical features → z-scored
      └── Rule-based scores → relevance, coherence, integrity
        ↓
  03  LLM scoring (llama3 via Ollama REST API)
      └── Returns all 4 dimension scores per goal
        ↓
  04  Meta-learner
      ├── Attainability → ElasticNet/Ridge on [11 features + LLM score]
      └── Other 3      → 0.6 × rule + 0.4 × LLM
        ↓
  05  SHAP explanations
        ↓
  06  Demo dashboard
```

### ElasticNet vs Ridge

| | ElasticNet | Ridge |
|---|---|---|
| Regularisation | L1 + L2 | L2 only |
| LOO MAE | ~0.19 | ~0.31 |
| Overfitting | Lower | Higher at small n |
| LLM signal | Active (rank #2) | Active but weaker |
| **Recommended** | **✓ PoC default** | Comparison only |

### Key Design Decisions

**Regression not classification** — Only 1 of 35 goals has probability ≥ 0.5. Binary classification collapses to predicting "no" for everything.

**Z-score the LLM output** — Without scaling, ElasticNet zeros the LLM signal. After z-scoring, LLM ranks #2 by SHAP importance.

**Rule + LLM blend for 3 dimensions** — Only Attainability has a ground truth label. Relevance, Coherence, Integrity use 60% rule + 40% LLM until labelled data arrives in April.

**No PCA** — Blending LLM into components destroys its independent SHAP interpretability.

---

## PoC Results (10-sample subset)

```
Attainability LOO MAE  : ~0.19
Attainability Train R² : ~0.54
LLM SHAP rank          : #2 of 12 features
Relevance mean score   : ~0.27
Coherence mean score   : ~0.36
Integrity mean score   : ~0.49
```

---

## Roadmap

| Milestone | Date |
|---|---|
| PoC complete | Mar 17, 2026 |
| Full dataset handoff from System 3 | Apr 2, 2026 |
| Full build — 3 LLMs + ensemble | Apr–May, 2026 |
| Final demo | May 5, 2026 |

**Full build architecture (April):**
Three LLMs in parallel (DeepSeek-R1 14B, Qwen-QwQ 32B, Llama 3.3 70B via Groq API) feeding a stacked ElasticNet meta-learner across all 4 dimensions with proper ground truth labels.

---

## Troubleshooting

**`ollama serve` — address already in use**
→ Already running as background service on Windows. Skip it.

**`ollama` not found after install**
→ Restart VS Code. PATH needs a full restart on Windows.

**LLM scores all 0.05**
→ Check `Prob std` in 03 output — should be > 0.10. Set `temperature=0.3`.

**KeyError in compare()**
→ Make sure you have the latest `run_poc.py`.

---

## Team

| System | Role |
|---|---|
| System 1 | Input/output — validation, LLM explanation to user |
| **System 2** | **Scoring — this repo** |
| System 3 | Data pipeline — feature generation |
