

# **FINAL README (System 3 — Data Layer)**

```markdown
# Decidr Coherence Engine — System 3 (Data Layer)
**Team 14-02 | iLab Capstone | UTS MDSI**

---

## Overview

System 3 is the **data layer (“heart”)** of the Decidr Coherence Engine.

Its role is to:
- Load and validate raw multi-table data
- Clean and standardize datasets
- Engineer meaningful features
- Generate rule-based scoring signals
- Produce ML-ready datasets for System 2 (Scoring Engine)

This system ensures that all downstream components receive **clean, structured, and reliable data**.

---

## Project Structure

```

Capstone/
│
├── analytical_flat.csv
├── buckets.csv
├── goals.csv
├── allocations.csv
├── outputs.csv
├── metrics.csv
├── derived_fields.csv
├── periods.csv
├── bedrock_client.py
│
└── System_3/
├── **init**.py
├── Readme.md
├── data_loader.py
├── embeddings.py
├── features.py
├── pipeline.py
├── system3_runner.py
├── etl.py
├── schema.py
├── faiss_index.py
├── validate_clean_transform.py
├── preprocess.py
├── rules.py
└── mock_system2.py

```

---

## System Flow

```

Raw CSVs
↓
data_loader.py (validation + loading)
↓
preprocess.py + validate_clean_transform.py (cleaning)
↓
etl.py (structured transformation)
↓
features.py (feature engineering)
↓
rules.py (rule score generation)
↓
embeddings.py + faiss_index.py (retrieval & optional dependencies)
↓
pipeline.py (orchestration)
↓
Outputs for System 2

```

---

## Key Modules

### 🔹 data_loader.py
- Loads all 8 raw datasets
- Performs validation checks:
  - Goal-period consistency
  - Bucket hierarchy validation
  - Data completeness
- Outputs:
  - `analytical_full.csv`
  - `period_12_poc.csv`
  - `period_18_poc.csv`

---

### 🔹 preprocess.py
- Cleans raw data:
  - Removes duplicates
  - Fixes data types
  - Handles missing values

---

### 🔹 validate_clean_transform.py
- Performs additional validation checks after preprocessing
- Ensures data integrity before feature generation

---

### 🔹 etl.py
- Transforms cleaned data into structured format
- Prepares data for feature engineering

---

### 🔹 features.py
- Builds engineered features used by System 2
- Examples:
  - Performance gap
  - Progress ratio
  - Cost per unit
  - Efficiency metrics
- Outputs:
  - `features_full_normalized.csv`
  - `features_full_raw.csv`
  - `features_raw_p6/p12/p18/p24.csv`

---

### 🔹 rules.py
- Generates rule-based scores:
  - Relevance
  - Coherence
  - Integrity

- These act as **interpretable anchors** for System 2

Outputs:
- `rule_scores_p6/p12/p18/p24.csv`
- `rule_scores_poc.csv`

---

### 🔹 embeddings.py
- Creates text representations of goals
- Supports:
  - Semantic retrieval
  - Dependency inference (LLM-based)

Dependency inference requires AWS Bedrock and is optional

---

### 🔹 faiss_index.py
- Builds FAISS index for similarity search
- Enables fast retrieval of similar goals

---

### 🔹 schema.py
- Defines structured payloads using Pydantic
- Ensures consistent data exchange between systems

---

### 🔹 mock_system2.py
- Simulates System 2 scoring
- Used for:
  - Testing pipeline
  - Debugging without ML models

---

### 🔹 pipeline.py
- Main orchestrator for System 3
- Runs:
  1. Data loading
  2. Validation
  3. Feature engineering
  4. Rule scoring

---

### 🔹 system3_runner.py
- Runs full System 3 pipeline
- Can integrate:
  - Mock System 2
  - Real System 2 (later stage)

---

## Outputs (System 2 Inputs)

| File | Description |
|------|------------|
| analytical_full.csv | Clean unified dataset |
| period_12_poc.csv | Snapshot at period 12 |
| period_18_poc.csv | Snapshot at period 18 |
| features_full_normalized.csv | ML-ready dataset |
| features_full_raw.csv | Raw feature dataset |
| features_raw_p12/p18/... | Snapshot features |
| rule_scores_poc.csv | Rule-based scores |
| feature_scaler_poc.pkl | Scaling logic |
| feature_names_poc.txt | Feature ordering |

---

## How to Run

### Step 1 — Install dependencies
```

pip install pandas numpy scikit-learn sentence-transformers faiss-cpu pydantic

```

---

### Step 2 — Run System 3 pipeline
```

python -m System_3.pipeline

```

---

### Step 3 — Run full pipeline with mock System 2
```

python -m System_3.system3_runner

```

---

## Notes

- Dependency inference is disabled for local testing
- Requires AWS Bedrock to enable LLM-based dependencies
- All outputs are aligned with System 2 input requirements

---

## Key Design Decisions

- Modular architecture for scalability
- Separation of concerns (data, features, scoring)
- Hybrid approach:
  - Rule-based logic (interpretability)
  - Feature-based ML input (predictive power)

---

## Summary

System 3 transforms raw organisational data into structured, validated, and feature-rich datasets.

It acts as the **foundation of the entire Coherence Engine**, enabling reliable scoring, explainability, and decision support.

---

## Contribution

- Designed and implemented full System 3 pipeline
- Built feature engineering and rule scoring logic
- Ensured compatibility with System 2
- Tested end-to-end pipeline locally
```



