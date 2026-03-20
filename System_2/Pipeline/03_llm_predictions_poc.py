"""
03_llm_predictions_poc.py — v4 (all 4 dimensions)
LLM scores all 4 dimensions per goal via Ollama REST API.
"""

import pandas as pd
import numpy as np
import json
import re
import urllib.request

MODEL_NAME  = "llama3"
OLLAMA_URL  = "http://localhost:11434/api/generate"
MAX_SAMPLES = 10
MAX_RETRIES = 3
TIMEOUT_SEC = 120


def build_prompt(row, rule_row):
    slope      = row["trailing_6_period_slope"]
    variance   = row["variance_from_target"]
    quality    = row["delivered_output_quality_score"]
    observed   = row["observed_value"]
    target     = row["target_value_final_period"]
    alloc_pct  = row["allocation_percentage_of_parent"]
    projected  = observed + slope * 12
    efficiency = row["allocation_efficiency_ratio"]
    quantity   = row["delivered_output_quantity"]

    return f"""Score this goal 0.0-1.0 on 4 dimensions.

DATA: current={observed:.2f} target={target:.2f} gap={variance:.2f} trend={slope:.4f} projected={projected:.2f} quality={quality:.2f} alloc={alloc_pct:.1%} efficiency={efficiency:.3f}
RULE SCORES: relevance={rule_row['relevance_rule']:.2f} coherence={rule_row['coherence_rule']:.2f} integrity={rule_row['integrity_rule']:.2f}

DIMENSIONS:
relevance: is allocation justified? (1=optimal band, 0=severely over/under)
coherence: allocation consistent with goals? (1=green stable, 0=misaligned)
integrity: outcomes match allocation? (1=full delivery, 0=poor vs spend)
attainability: will goal hit target by period 24? (1=certain, 0=very unlikely)

Reply ONLY with this JSON, one line:
{{"relevance": 0.XX, "coherence": 0.XX, "integrity": 0.XX, "attainability": 0.XX, "reasoning": "under 10 words"}}"""


def extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return json.loads(text)


def call_ollama(prompt):
    payload = json.dumps({
        "model"  : MODEL_NAME,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature": 0.3, "num_predict": 120, "num_ctx": 2048, "top_p": 0.9},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


def predict_single(row, rule_row, goal_idx):
    prompt     = build_prompt(row, rule_row)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw    = call_ollama(prompt)
            if not raw:
                raise ValueError("Empty response")
            parsed = extract_json(raw)

            # Validate all 4 keys present
            for key in ["relevance", "coherence", "integrity", "attainability"]:
                if key not in parsed:
                    raise ValueError(f"Missing key: {key}")

            return {
                "goal_idx"       : goal_idx,
                "llm_relevance"  : float(np.clip(float(parsed["relevance"]),    0.05, 0.95)),
                "llm_coherence"  : float(np.clip(float(parsed["coherence"]),    0.05, 0.95)),
                "llm_integrity"  : float(np.clip(float(parsed["integrity"]),    0.05, 0.95)),
                "llm_attainability": float(np.clip(float(parsed["attainability"]), 0.05, 0.95)),
                "reasoning"      : parsed.get("reasoning", ""),
                "attempts"       : attempt,
                "success"        : True,
            }
        except Exception as e:
            last_error = e
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {e}")

    return {
        "goal_idx"         : goal_idx,
        "llm_relevance"    : None,
        "llm_coherence"    : None,
        "llm_integrity"    : None,
        "llm_attainability": None,
        "reasoning"        : str(last_error),
        "attempts"         : MAX_RETRIES,
        "success"          : False,
    }


def predict_batch(df_raw, rule_scores, max_samples=MAX_SAMPLES):
    print("=" * 70)
    print("LLM PREDICTIONS — v4 (all 4 dimensions)")
    print("=" * 70)

    try:
        urllib.request.urlopen("http://localhost:11434", timeout=5)
        print("Ollama reachable  OK")
    except Exception:
        print("ERROR: Ollama not reachable at localhost:11434")
        raise SystemExit(1)

    n           = min(len(df_raw), max_samples)
    predictions = []

    for i in range(n):
        row      = df_raw.iloc[i]
        rule_row = rule_scores.iloc[i]
        print(f"\n[{i+1}/{n}] Goal {i} ...", flush=True)
        pred = predict_single(row, rule_row, i)
        predictions.append(pred)
        if pred["success"]:
            print(f"  R={pred['llm_relevance']:.2f}  "
                  f"C={pred['llm_coherence']:.2f}  "
                  f"I={pred['llm_integrity']:.2f}  "
                  f"A={pred['llm_attainability']:.2f}  "
                  f"\"{pred['reasoning'][:50]}\"")
        else:
            print(f"  -> FAILED")

    df        = pd.DataFrame(predictions)
    success_n = int(df["success"].sum())
    fail_n    = int((~df["success"]).sum())

    print(f"\n{'='*70}")
    print(f"Results : {success_n} succeeded / {fail_n} failed / {n} total")
    if fail_n:
        print(f"WARNING : {fail_n} goals excluded from meta-learner")
    print("=" * 70)
    return df


if __name__ == "__main__":
    df_raw      = pd.read_csv("features_raw_poc.csv")
    rule_scores = pd.read_csv("rule_scores_poc.csv")

    # Merge extra columns needed by prompt that are not in features_raw
    period_12 = pd.read_csv("period_12_poc.csv").reset_index(drop=True)
    for col in ["allocation_efficiency_ratio", "delivered_output_quantity"]:
        if col not in df_raw.columns and col in period_12.columns:
            df_raw[col] = period_12[col].values

    predictions = predict_batch(df_raw, rule_scores, max_samples=MAX_SAMPLES)
    predictions.to_csv("llm_predictions_poc.csv", index=False)
    print("\n✓ Saved llm_predictions_poc.csv")

    ok = predictions[predictions["success"]]
    if len(ok):
        for dim in ["llm_relevance","llm_coherence","llm_integrity","llm_attainability"]:
            vals = ok[dim].dropna()
            print(f"  {dim:<22}: {vals.min():.2f}-{vals.max():.2f}  std={vals.std():.3f}")
