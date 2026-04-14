# Decidr Coherence Engine
**Team 14-02 | iLab Capstone 36127 | UTS MDSI**
**Supervisor: Pouya Salehpour | Industry Partner: Decidr**

---

## What This Is

An AI-powered coherence scoring engine for organisational goal portfolios.
Given a set of goals with allocations, outputs, and metrics across 24 periods,
the engine scores each goal across four dimensions and produces a portfolio
health picture with shock analysis, forward projections, and reallocation
recommendations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     System 1 (Team 1)                   │
│          Streamlit UI — NL input + Score display        │
└─────────────────────┬───────────────────────────────────┘
                      │ goal payload
┌─────────────────────▼───────────────────────────────────┐
│                     System 3                            │
│          Orchestration — routing + feature lookup       │
└──────────┬──────────────────────────────────────────────┘
           │ feature rows
┌──────────▼──────────────────────────────────────────────┐
│              Data Layer (system3/ folder)               │
│   load_data → feature_engineering → infer_dependencies  │
└──────────┬──────────────────────────────────────────────┘
           │ feature CSVs
┌──────────▼──────────────────────────────────────────────┐
│            System 2 Core (system2/ folder)              │
│   llm_scoring → meta_learner → composite_score          │
│               → verify_goal                             │
│   score_goal / score_group  ← called by System 3        │
└──────────┬──────────────────────────────────────────────┘
           │ score CSVs
┌──────────▼──────────────────────────────────────────────┐
│          Presentation Layer (dashboard/ folder)         │
│          Streamlit dashboard + EDA visualisations       │
└─────────────────────────────────────────────────────────┘
```

---

## Repo Structure

```
decidr-coherence-engine/
│
├── README.md                  ← this file
├── run_all.py                 ← full pipeline runner
├── requirements.txt
│
├── system3/                   ← Data Layer
│   ├── README.md
│   ├── load_data.py
│   ├── feature_engineering.py
│   └── infer_dependencies.py
│
├── system2/                   ← Intelligence Core (this team)
│   ├── README.md
│   ├── llm_scoring.py
│   ├── meta_learner.py
│   ├── composite_score.py
│   ├── verify_goal.py
│   ├── score_goal.py
│   ├── score_group.py
│   └── run.py
│
└── dashboard/                 ← Presentation Layer
    ├── README.md
    ├── dashboard_app.py
    ├── explanations.py
    └── dashboard.py
```

---

## Four Scoring Dimensions

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Coherence | 35% | Consistency across hierarchy, time, and goals |
| Attainability | 25% | Likelihood of reaching target by period 24 |
| Relevance | 20% | Whether allocation level is justified |
| Integrity | 20% | Whether spend produced expected outputs |

---

## Quick Start

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Pull Ollama models (once only):**
```bash
ollama pull llama3:latest
ollama pull gemma3:4b
ollama pull deepseek-r1:8b
```

**3. Place source CSVs in `system3/` folder:**
```
analytical_flat.csv  buckets.csv  goals.csv  allocations.csv
outputs.csv  metrics.csv  derived_fields.csv  periods.csv
```

**4. Run the full pipeline:**
```bash
python run_all.py
```

**5. Launch the dashboard:**
```bash
streamlit run dashboard/dashboard_app.py
```

---

## Models

| Model | Role |
|-------|------|
| llama3:latest | Scorer — narrative analyst brief |
| gemma3:4b | Scorer — threshold rubric |
| deepseek-r1:8b | Verifier — post-composite consistency judge |

All models run locally via Ollama. No internet needed after initial pull.

---

## Dataset

35 SMARTeR goals across 24 monthly periods (Jan 2024 - Dec 2025).
3-level budget hierarchy: 4 L1 → 14 L2 → 35 L3 leaf buckets.

Two embedded shock events:
- Budget shock: 20% reduction in periods 10-12
- Market shock: 15% reduction to growth metrics in period 14-17

LLM scoring runs at 4 snapshot periods: 6, 12, 18, 24.
Each snapshot includes trajectory context from the prior checkpoint.

---

## Team

| Member | Role |
|--------|------|
| Krishnan Unni Prasad | Team Lead — System 2 Intelligence Core |
| Adrian Mato | System 2 — Model testing and pipeline validation |

Supervisor: Pouya Salehpour
Industry Partner: Tom, Decidr
