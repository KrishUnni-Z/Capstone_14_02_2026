"""
llm_scoring.py  v6

Two testers scoring each (goal, period) via Bedrock Converse API.
  Llama 3.3 70B  + Mistral Large 2407

Outputs:
  llm_predictions_p{6,12,18}.csv  (per-period)
  llm_predictions_poc.csv         (combined)

Pass D: p24 is held out, no LLMs score p24. Forward projection produces p24.

Anchor period for prompt/projection math is each row's current_period, not
a hardcoded 12. The "halfway point" language is gone.
"""

import os
import time
import argparse

import pandas as pd
import numpy as np

from bedrock_client import call_model, extract_json, BedrockCallError


MAX_SAMPLES = 35
MAX_RETRIES = 3
FINAL_PERIOD = 24

DIMS = ["relevance", "coherence", "integrity", "attainability"]

# Two testers, referenced by role into bedrock_client.MODEL_IDS
MODELS = [
    {"role": "llama_tester",   "label": "llama"},
    {"role": "mistral_tester", "label": "mistral"},
]

# Snapshots scored by LLMs. p24 is excluded so predictions of p24 are honest
# forward projections, not descriptions of the held-out target.
SNAPSHOT_PERIODS = [6, 12, 18]

# Budget shock was in periods 10-12, market shock in 14-17. This maps which
# shocks occurred between two consecutive snapshot periods.
SHOCK_BETWEEN = {
    (0, 6)  : 0.0,
    (6, 12) : 1.0,   # budget shock
    (12, 18): 1.0,   # market shock
}

# Per-model generation settings. Mistral Large 2407 handles long context and
# structured output cleanly, so we use the same settings as Llama.
MODEL_OPTIONS = {
    "llama"  : {"temperature": 0.2, "max_tokens": 800},
    "mistral": {"temperature": 0.2, "max_tokens": 800},
}

# System prompts per model. Both Llama and Mistral follow instructions well
# enough without a system-level JSON guard.
SYSTEM_PROMPTS = {
    "llama"  : None,
    "mistral": None,
}

# Assistant-turn prefill per model. Neither Llama nor Mistral needs a prefill
# to produce clean JSON; extract_json already strips Mistral's ```json fences.
PREFILL = {
    "llama"  : None,
    "mistral": None,
}


def clean_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fallback_reason_for_dim(dim):
    fallback = {
        "attainability": f"The score reflects the likelihood of reaching the target by period {FINAL_PERIOD}.",
        "relevance"    : "The score reflects how well allocation appears aligned with the goal.",
        "coherence"    : "The score reflects consistency across levels, periods, and decisions.",
        "integrity"    : "The score reflects whether outputs and metric movement appear to match the resources used.",
    }
    return fallback[dim]


def _signals(row, rule_row):
    s = {}
    s["observed"]   = clean_float(row.get("observed_value", 0))
    s["target"]     = clean_float(row.get("target_value_final_period", 0))
    s["slope"]      = clean_float(row.get("trailing_6_period_slope", 0))
    s["variance"]   = clean_float(row.get("variance_from_target", 0))
    s["quality"]    = clean_float(row.get("delivered_output_quality_score", 0))
    s["quantity"]   = clean_float(row.get("delivered_output_quantity", 0))
    s["efficiency"] = clean_float(row.get("allocation_efficiency_ratio", 0))
    s["needle"]     = clean_float(row.get("needle_move_ratio", 0.5))
    s["alloc_pct"]  = clean_float(row.get("allocation_percentage_of_parent", 0))
    s["fitness"]    = clean_float(row.get("allocation_fitness_score", 0))
    s["sibling_r"]  = clean_float(row.get("sibling_rank_pct", 0.5))
    s["l3_l2"]      = clean_float(row.get("l3_share_of_l2", 0.4))
    s["drift"]      = clean_float(row.get("alloc_drift_std", 0))
    s["ttg"]        = clean_float(row.get("time_to_green_estimate", 0))
    s["wgs"]        = clean_float(row.get("weighted_goal_status_score", 0.5))
    s["r_rule"]     = clean_float(rule_row.get("relevance_rule", 0.5))
    s["c_rule"]     = clean_float(rule_row.get("coherence_rule", 0.5))
    s["i_rule"]     = clean_float(rule_row.get("integrity_rule", 0.5))

    s["current_period"]    = int(clean_float(row.get("current_period", 12)))
    s["periods_remaining"] = FINAL_PERIOD - s["current_period"]
    s["projected"]         = s["observed"] + s["slope"] * s["periods_remaining"]
    s["anchor"] = float(np.clip((s["projected"] / s["target"]) if s["target"] > 0 else 0.5, 0.05, 0.95))

    s["sibling_label"] = (
        "best funded among peers" if s["sibling_r"] < 0.3
        else "mid-ranked among peers" if s["sibling_r"] < 0.7
        else "worst funded among peers"
    )
    s["fitness_label"] = (
        "allocation sits within optimal funding band"
        if s["fitness"] == 1.0
        else "allocation falls outside optimal funding band"
    )
    s["pct_of_target"] = round(s["observed"] / s["target"] * 100, 1) if s["target"] > 0 else 0
    s["proj_pct"]      = round(s["projected"] / s["target"] * 100, 1) if s["target"] > 0 else 0

    s["prev_composite"]   = clean_float(row.get("prev_composite", -1.0))
    s["composite_delta"]  = clean_float(row.get("composite_delta", 0.0))
    s["prev_period"]      = int(clean_float(row.get("prev_period", 0)))
    s["shock_since_prev"] = clean_float(row.get("shock_since_prev", 0.0))
    has_prev = s["prev_composite"] > 0
    s["traj_label"] = (
        f"no prior checkpoint  period {s['current_period']} is first assessment"
        if not has_prev else
        f"period {s['prev_period']} composite was {s['prev_composite']:.2f}, "
        f"{'improved' if s['composite_delta'] >= 0 else 'degraded'} by {abs(s['composite_delta']):.3f}"
        + (f" (shock occurred between periods {s['prev_period']} and {s['current_period']})"
           if s["shock_since_prev"] > 0 else "")
    )

    s["budget_shock_exp"] = clean_float(row.get("budget_shock_exposure", 0.0))
    s["shock_impact"]     = clean_float(row.get("shock_alloc_impact", 0.0))
    s["recovery_est"]     = clean_float(row.get("recovery_period_estimate", 0.0))
    s["market_vuln"]      = clean_float(row.get("market_shock_vulnerable", 0.0))
    s["market_risk"]      = clean_float(row.get("market_shock_forward_risk", 0.0))
    s["recovery_window"]  = clean_float(row.get("recovery_window_remaining", 7.0))
    s["in_shock"] = s["budget_shock_exp"] >= 0.33
    s["shock_label"] = (
        "currently in budget shock period (20% budget cut)" if s["in_shock"]
        else "no active budget shock"
    )
    s["market_label"] = (
        f"metric is market-shock sensitive (forward risk {s['market_risk']:.2f})"
        if s["market_vuln"] > 0 else "metric not market-shock sensitive"
    )

    risk_map = {
        0.0 : "No dependency risk",
        0.33: "Low dependency risk",
        0.67: "Medium dependency risk",
        1.0 : "High dependency risk",
    }
    s["dep_risk_label"] = risk_map.get(
        clean_float(row.get("dependency_risk_encoded", 0.0)),
        "No dependency risk"
    )
    s["dep_attain"]   = clean_float(row.get("dep_avg_attain", -1.0))
    s["n_dependents"] = clean_float(row.get("n_dependents", 0))
    s["n_deps"]       = clean_float(row.get("n_dependencies", 0))
    s["dep_risk"]     = clean_float(row.get("dependency_risk_encoded", 0.0))

    return s


def prompt_narrative(s):
    """Long analyst-brief prompt. Used by Llama 3.3 70B which handles long
    context well and reasons naturally in prose."""
    return f"""You are a strategic analyst reviewing an organisational goal at period {s['current_period']} of {FINAL_PERIOD}.

GOAL SNAPSHOT
  Current value   : {s["observed"]:.3f}  (target by period {FINAL_PERIOD}: {s["target"]:.3f})
  Progress so far : {s["pct_of_target"]}% of target reached
  Trend           : {s["slope"]:+.4f} per period  (projected at period {FINAL_PERIOD}: {s["projected"]:.3f}, which is {s["proj_pct"]}% of target)
  Periods remain  : {s["periods_remaining"]} of {FINAL_PERIOD}
  Time to green   : {s["ttg"]:.0f} periods estimated

ALLOCATION PICTURE
  Share of parent bucket  : {s["alloc_pct"]:.1%}
  Sibling rank            : {s["sibling_label"]}  (rank score {s["sibling_r"]:.2f}, lower is better funded)
  Funding band status     : {s["fitness_label"]}
  Allocation drift (std)  : {s["drift"]:.4f} over 24 periods  (higher = more volatile)
  Budget share of L2      : {s["l3_l2"]:.2f}

DEPENDENCY CONTEXT
  {s["dep_risk_label"]}
  Goals depending on this one: {int(s["n_dependents"])}
  {f"Upstream avg attainability: {s['dep_attain']:.2f}" if s["dep_attain"] > 0 else ""}

DELIVERY PICTURE
  Output quality          : {s["quality"]:.2f} / 1.0
  Output quantity         : {s["quantity"]:.1f} units
  Efficiency ratio        : {s["efficiency"]:.3f}
  Needle move ratio       : {s["needle"]:.3f}
  Weighted status score   : {s["wgs"]:.3f}

SHOCK CONTEXT
  Budget shock    : {s["shock_label"]}
  Alloc impact    : {s["shock_impact"]:.4f} drop from pre-shock level
  Recovery est    : {s["recovery_est"]:.1f} periods to return to pre-shock allocation
  Market exposure : {s["market_label"]}
  Forward risk    : {s["market_risk"]:.2f}
  Recovery runway : {s["recovery_window"]:.0f} periods post-shock to hit target

TRAJECTORY CONTEXT (prior checkpoint)
  {s["traj_label"]}

COMPUTED ANCHOR SCORES
  Attainability anchor : {s["anchor"]:.2f}
  Relevance rule score : {s["r_rule"]:.2f}
  Coherence rule score : {s["c_rule"]:.2f}
  Integrity rule score : {s["i_rule"]:.2f}

SCORING TASK
Evaluate this goal on four dimensions. For each, reason from the evidence above and assign a score between 0.05 and 0.95.

  RELEVANCE     is the allocation level justified against stated goals?
  COHERENCE     are decisions consistent across the hierarchy, time, and goals?
  INTEGRITY     did the allocation produce the expected outputs and move the metric?
  ATTAINABILITY will this goal realistically reach its target by period {FINAL_PERIOD}?

Return ONLY valid JSON.
Do not include headings.
Do not include bullet points.
Do not explain before the JSON.
Do not explain after the JSON.
Start with {{ and end with }}.

{{
  "relevance": 0.XX,
  "coherence": 0.XX,
  "integrity": 0.XX,
  "attainability": 0.XX,
  "relevance_reason": "One specific sentence explaining the relevance score.",
  "coherence_reason": "One specific sentence explaining the coherence score.",
  "integrity_reason": "One specific sentence explaining the integrity score.",
  "attainability_reason": "One specific sentence explaining the attainability score."
}}""".strip()


def prompt_rubric(s):
    """Short rubric-based prompt. Used by anthropic which on Bedrock weakens
    on long context and hallucinates scenarios without tight constraints.
    Pair with the anthropic system prompt that enforces JSON-only output."""
    return f"""Score this goal using the rubric. Use each threshold to assign one score per dimension.

EVIDENCE (period {s['current_period']} of {FINAL_PERIOD}):
  observed={s["observed"]:.3f}  target={s["target"]:.3f}  progress={s["pct_of_target"]}%
  slope={s["slope"]:+.4f}/period  projected_at_p{FINAL_PERIOD}={s["projected"]:.3f} ({s["proj_pct"]}% of target)
  periods_remaining={s["periods_remaining"]}
  alloc_pct_of_parent={s["alloc_pct"]:.3f}  sibling_rank_pct={s["sibling_r"]:.2f}  fitness={s["fitness_label"]}
  efficiency={s["efficiency"]:.3f}  needle_move_ratio={s["needle"]:.3f}  quality={s["quality"]:.2f}
  alloc_drift_std={s["drift"]:.4f}  weighted_status={s["wgs"]:.3f}
  dep_risk={s["dep_risk_label"]}  n_upstream_deps={int(s["n_deps"])}
  budget_shock={s["shock_label"]}  recovery_est={s["recovery_est"]:.1f}

RUBRIC  each dimension uses its own thresholds:

ATTAINABILITY (projected_pct driven):
  projected >= 90%        -> 0.80-0.95
  projected 65-89%        -> 0.55-0.79
  projected 40-64%        -> 0.35-0.54
  projected 20-39%        -> 0.15-0.34
  projected < 20%         -> 0.05-0.14

RELEVANCE (sibling_rank + fitness):
  sibling_rank < 0.3 AND in optimal band  -> 0.80-0.95
  sibling_rank < 0.5 OR mostly in band    -> 0.55-0.79
  sibling_rank 0.5-0.7 OR outside band    -> 0.35-0.54
  sibling_rank > 0.7 AND outside band     -> 0.15-0.34
  sibling_rank > 0.9                      -> 0.05-0.14

COHERENCE (drift + status):
  drift < 0.03 AND status > 0.5   -> 0.80-0.95
  drift < 0.06 OR status > 0.3    -> 0.55-0.79
  drift 0.06-0.10                 -> 0.35-0.54
  drift > 0.10 OR status < 0.15   -> 0.15-0.34
  drift > 0.15 AND status < 0.10  -> 0.05-0.14

INTEGRITY (efficiency + needle_move):
  efficiency > 0.5 AND needle > 0.7  -> 0.80-0.95
  efficiency > 0.3 AND needle > 0.5  -> 0.55-0.79
  efficiency > 0.2 OR needle > 0.3   -> 0.35-0.54
  efficiency < 0.2 AND needle < 0.3  -> 0.15-0.34
  efficiency < 0.1 AND needle < 0.2  -> 0.05-0.14

Return this JSON exactly and nothing else:
{{"relevance":0.XX,"coherence":0.XX,"integrity":0.XX,"attainability":0.XX,"relevance_reason":"one sentence citing the signal","coherence_reason":"one sentence citing the signal","integrity_reason":"one sentence citing the signal","attainability_reason":"one sentence citing the signal"}}""".strip()


# Dispatcher. Both strong testers use the narrative prompt.
PROMPT_FN = {
    "llama"  : prompt_narrative,
    "mistral": prompt_narrative,
}


def build_prompt(row, rule_row, model_label="llama"):
    s = _signals(row, rule_row)
    fn = PROMPT_FN.get(model_label, prompt_narrative)
    return fn(s)


def fix_placeholder_reason(reason, fallback_text):
    if not reason:
        return fallback_text
    cleaned = reason.strip().lower()
    if cleaned in {"one short sentence", "short sentence", "placeholder", "one specific sentence"}:
        return fallback_text
    return reason


def predict_model(model, prompt):
    label = model["label"]
    role  = model["role"]
    opts  = MODEL_OPTIONS.get(label, {"temperature": 0.2, "max_tokens": 800})
    system = SYSTEM_PROMPTS.get(label)
    prefill = PREFILL.get(label)
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            raw = call_model(
                role,
                prompt,
                max_tokens=opts["max_tokens"],
                temperature=opts["temperature"],
                system=system,
                prefill=prefill,
            )
            parsed = extract_json(raw, required_keys=DIMS)
            result = {}
            for d in DIMS:
                result[d] = float(np.clip(float(parsed[d]), 0.05, 0.95))
                result[f"{d}_reason"] = fix_placeholder_reason(
                    str(parsed.get(f"{d}_reason", "")).strip(),
                    fallback_reason_for_dim(d),
                )
            result["success"]   = True
            result["elapsed_s"] = round(time.time() - t0, 1)
            return result

        except (BedrockCallError, ValueError) as e:
            last_err = e
            msg = str(e).lower()
            is_transient = any(tok in msg for tok in (
                "throttl", "timeout", "timed out", "read timed",
                "connection reset", "rate", "service unavailable", "503",
            ))
            if attempt < MAX_RETRIES and is_transient:
                wait = 5 * attempt
                print(f"      [{label}] transient error (attempt {attempt}/{MAX_RETRIES}): {e}")
                print(f"      [{label}] backing off {wait}s before retry")
                time.sleep(wait)
                continue
            print(f"      [{label}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2)
        except Exception as e:
            last_err = e
            print(f"      [{label}] attempt {attempt}/{MAX_RETRIES} unexpected: {type(e).__name__}: {e}")
            time.sleep(2)

    return (
        {d: None for d in DIMS}
        | {f"{d}_reason": "" for d in DIMS}
        | {"success": False, "elapsed_s": 0, "error": str(last_err)}
    )


def print_model_output(label, res):
    if res["success"]:
        print(
            f"      {label:<10}"
            f"A={res['attainability']:.2f}  "
            f"R={res['relevance']:.2f}  "
            f"C={res['coherence']:.2f}  "
            f"I={res['integrity']:.2f}  "
            f"({res['elapsed_s']:.1f}s)"
        )
        for d in DIMS:
            r = res.get(f"{d}_reason", "")
            if r:
                print(f"        {d:<14}: {r}")
    else:
        print(f"      {label:<10}FAILED ({res.get('elapsed_s', 0):.1f}s)")
        print(f"        error: {res.get('error', '')}")


def predict_goal(df_raw, rule_scores, goal_idx, models):
    row      = df_raw.iloc[goal_idx]
    rule_row = rule_scores.iloc[goal_idx]

    record = {"goal_idx": goal_idx}
    n_ok   = 0
    model_results = []

    for m in models:
        prompt = build_prompt(row, rule_row, m["label"])
        res    = predict_model(m, prompt)
        print_model_output(m["label"], res)

        if res["success"]:
            n_ok += 1
            model_results.append(res)

        for d in DIMS:
            record[f"{m['label']}_{d}"]        = res[d]
            record[f"{m['label']}_{d}_reason"] = res.get(f"{d}_reason", "")
        record[f"{m['label']}_success"]   = res["success"]
        record[f"{m['label']}_elapsed_s"] = res.get("elapsed_s", 0)
        record[f"{m['label']}_error"]     = res.get("error", "")

    if model_results:
        ens_a = round(float(np.mean([r["attainability"] for r in model_results])), 4)
        ens_r = round(float(np.mean([r["relevance"]   for r in model_results])), 4)
        ens_c = round(float(np.mean([r["coherence"]   for r in model_results])), 4)
        ens_i = round(float(np.mean([r["integrity"]   for r in model_results])), 4)
        print(f"      {'ensemble':<10}A={ens_a:.2f}  R={ens_r:.2f}  C={ens_c:.2f}  I={ens_i:.2f}")

    record["n_models_ok"] = n_ok
    record["success"]     = n_ok > 0
    return record


def merge_extra_columns(df_raw, period_ref):
    extra_cols = [
        "allocation_efficiency_ratio",
        "delivered_output_quantity",
        "allocation_percentage_of_parent",
        "target_value_final_period",
        "time_to_green_estimate",
        "weighted_goal_status_score",
        "sibling_rank_pct",
        "l3_share_of_l2",
        "alloc_drift_std",
        "needle_move_ratio",
        "allocation_fitness_score",
        "observed_value",
        "variance_from_target",
        "delivered_output_quality_score",
        "trailing_6_period_slope",
        # Shock signals engineered by feature_engineering.py
        "budget_shock_exposure",
        "shock_alloc_impact",
        "recovery_period_estimate",
        "recovery_window_remaining",
        "market_shock_vulnerable",
        "market_shock_forward_risk",
    ]
    for col in extra_cols:
        if col not in df_raw.columns and col in period_ref.columns:
            df_raw[col] = period_ref[col].values
    return df_raw


def _paths_for_period(sp):
    if sp == 12:
        return ("features_raw_poc.csv", "rule_scores_poc.csv", "llm_predictions_poc.csv")
    return (f"features_raw_p{sp}.csv", f"rule_scores_p{sp}.csv", f"llm_predictions_p{sp}.csv")


def _load_period_ref():
    if os.path.exists("period_18_poc.csv"):
        return pd.read_csv("period_18_poc.csv").reset_index(drop=True)
    return pd.read_csv("period_12_poc.csv").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--goal", type=int, default=None)
    parser.add_argument("--period", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("DECIDR COHERENCE ENGINE  LLM Predictions")
    print(f"Testers: Llama 3.3 70B + Mistral Large 2407")
    print(f"Snapshot periods: {SNAPSHOT_PERIODS}  (p24 held out)")
    print("=" * 70)

    if args.test or args.goal is not None:
        goal_idx = 0 if args.test else args.goal
        for sp in [12, 18]:
            raw_file, rule_file, _ = _paths_for_period(sp)
            if not os.path.exists(raw_file):
                print(f"\nSkipping period {sp}  {raw_file} not found")
                continue
            df_raw      = pd.read_csv(raw_file)
            rule_scores = pd.read_csv(rule_file)
            period_ref  = _load_period_ref()
            df_raw      = merge_extra_columns(df_raw, period_ref)

            df_raw["current_period"]   = sp
            prev_p = max([p for p in SNAPSHOT_PERIODS if p < sp], default=0)
            df_raw["prev_composite"]   = -1.0
            df_raw["composite_delta"]  = 0.0
            df_raw["prev_period"]      = prev_p
            df_raw["shock_since_prev"] = SHOCK_BETWEEN.get((prev_p, sp), 0.0)

            print(f"\n{'TEST' if args.test else 'SINGLE GOAL'} MODE  goal {goal_idx} at period {sp}")
            pred = predict_goal(df_raw, rule_scores, goal_idx, MODELS)
            print(f"=> {pred['n_models_ok']}/{len(MODELS)} models OK")
        raise SystemExit(0)

    periods_to_run = [args.period] if args.period else SNAPSHOT_PERIODS
    all_preds   = []
    prev_scores = {}

    t_total = time.time()
    for sp in periods_to_run:
        raw_file, rule_file, out_file = _paths_for_period(sp)

        if not os.path.exists(raw_file):
            print(f"\nSkipping period {sp}  {raw_file} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"PERIOD {sp}  LLM SCORING")
        print(f"{'=' * 70}")

        df_raw      = pd.read_csv(raw_file)
        rule_scores = pd.read_csv(rule_file)
        period_ref  = _load_period_ref()
        df_raw      = merge_extra_columns(df_raw, period_ref)

        df_raw["current_period"]   = sp
        prev_p = max([p for p in SNAPSHOT_PERIODS if p < sp], default=0)
        df_raw["prev_period"]      = prev_p
        df_raw["shock_since_prev"] = SHOCK_BETWEEN.get((prev_p, sp), 0.0)

        def get_prev(gid):
            return prev_scores.get((gid, prev_p), -1.0)

        if "goal_id" in df_raw.columns:
            df_raw["prev_composite"]  = df_raw["goal_id"].apply(get_prev)
            df_raw["composite_delta"] = df_raw.apply(
                lambda r: (
                    get_prev(r.get("goal_id", -1)) - r["prev_composite"]
                    if r["prev_composite"] > 0 else 0.0
                ),
                axis=1
            )
        else:
            df_raw["prev_composite"]  = -1.0
            df_raw["composite_delta"] = 0.0

        goals_to_run = list(range(min(len(df_raw), MAX_SAMPLES)))
        preds   = []
        t_start = time.time()
        n_total = len(goals_to_run)

        for i, goal_idx in enumerate(goals_to_run):
            elapsed = time.time() - t_start
            if i > 0:
                eta = (elapsed / i) * (n_total - i) / 60
                print(f"\n[p{sp}][{i+1}/{n_total}] Goal {goal_idx}  (elapsed: {elapsed/60:.1f}m  ETA: {eta:.1f}m)")
                time.sleep(1)   # brief pause between goals to avoid Mistral throttling
            else:
                print(f"\n[p{sp}][{i+1}/{n_total}] Goal {goal_idx}")

            pred = predict_goal(df_raw, rule_scores, goal_idx, MODELS)
            pred["period_id"] = sp
            preds.append(pred)
            print(f"  => {pred['n_models_ok']}/{len(MODELS)} models OK")

        period_min = (time.time() - t_start) / 60
        df_period  = pd.DataFrame(preds)
        df_period.to_csv(out_file, index=False)
        all_preds.append(df_period)
        print(f"\nOK  Period {sp} done in {period_min:.1f} min  saved {out_file}")

        for _, row in df_period.iterrows():
            gid = int(row.get("goal_idx", -1))
            scores = [
                row[f"{m['label']}_attainability"]
                for m in MODELS
                if f"{m['label']}_attainability" in row.index
                and pd.notna(row[f"{m['label']}_attainability"])
            ]
            if scores:
                prev_scores[(gid, sp)] = float(np.mean(scores))

    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        combined.to_csv("llm_predictions_poc.csv", index=False)
        total_min = (time.time() - t_total) / 60
        print(f"\n{'=' * 70}")
        print(f"ALL PERIODS COMPLETE in {total_min:.1f} minutes")
        print(f"Total predictions: {len(combined)} rows saved to llm_predictions_poc.csv")
        print("=" * 70)