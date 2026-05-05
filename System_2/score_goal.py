import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
score_goal.py  v7  Inference via Bedrock Converse

Usage:
    python score_goal.py --goal_id 0
    python score_goal.py --goal_id 7 --output result.json
    python score_goal.py --goal_id 0 --period 18

Called by System 3:
    from score_goal import score_goal
    result = score_goal(goal_row, rule_row)
"""

import os
import json
import re
import pickle
import argparse
import warnings
import time

import pandas as pd
import numpy as np

try:
    from System_2.bedrock_client import call_model, extract_json, BedrockCallError
except ImportError:
    from bedrock_client import call_model, extract_json, BedrockCallError
try:
    from System_2.verify_goal import verify_scores
except ImportError:
    from verify_goal import verify_scores


TIMEOUT_SEC = 300
MAX_RETRIES = 3
FINAL_PERIOD = 24

BEDROCK_MODELS = [
    {"role": "llama_tester",   "label": "llama"},
    {"role": "mistral_tester", "label": "mistral"},
]

DIMS = ["relevance", "coherence", "integrity", "attainability"]

dep_map = {}
if os.path.exists("goal_dependencies.csv"):
    dep_df_inf = pd.read_csv("goal_dependencies.csv")
    dep_map = (
        dep_df_inf
        .drop_duplicates(subset="goal_id", keep="last")
        .set_index("goal_id")
        .to_dict("index")
    )


def load_artefacts():
    _s2_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(_s2_root, "gp_poc.pkl"), "rb") as f:
        gp = pickle.load(f)
    with open(os.path.join(_s2_root, "platt_scalers_poc.pkl"), "rb") as f:
        iso = pickle.load(f)
    with open(os.path.join(_s2_root, "feature_scaler_poc.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(_s2_root, "gp_config_poc.json")) as f:
        cfg = json.load(f)
    return gp, iso, scaler, cfg


def compute_baseline(observed, slope, target, period=18, total=FINAL_PERIOD):
    remaining = total - period
    return float(np.clip((observed + slope * remaining) / max(target, 1e-9), 0, 1))


MODEL_OPTIONS = {
    "llama"  : {"temperature": 0.2, "max_tokens": 1200},
    "mistral": {"temperature": 0.2, "max_tokens": 1200},
}

SYSTEM_PROMPTS = {
    "llama"  : None,
    "mistral": None,
}

PREFILL = {
    "llama"  : None,
    "mistral": None,
}


def _signals_inline(row, rule_row, current_period):
    """Extract and label signals for both prompt variants."""
    s = {}
    s["slope"]      = float(row.get("trailing_6_period_slope", 0))
    s["variance"]   = float(row.get("variance_from_target", 0))
    s["quality"]    = float(row.get("delivered_output_quality_score", 0))
    s["observed"]   = float(row.get("observed_value", 0))
    s["target"]     = float(row.get("target_value_final_period", 0))
    s["alloc_pct"]  = float(row.get("allocation_percentage_of_parent", 0))
    s["efficiency"] = float(row.get("allocation_efficiency_ratio", 0))
    s["quantity"]   = float(row.get("delivered_output_quantity", 0))
    s["ttg"]        = float(row.get("time_to_green_estimate", 0))
    s["wgs"]        = float(row.get("weighted_goal_status_score", 0.5))
    s["sibling_r"]  = float(row.get("sibling_rank_pct", 0.5))
    s["l3_l2"]      = float(row.get("l3_share_of_l2", 0.4))
    s["drift"]      = float(row.get("alloc_drift_std", 0))
    s["needle"]     = float(row.get("needle_move_ratio", 0.5))
    s["fitness"]    = float(row.get("allocation_fitness_score", 0))
    s["r_rule"]     = float(rule_row.get("relevance_rule", 0.5))
    s["c_rule"]     = float(rule_row.get("coherence_rule", 0.5))
    s["i_rule"]     = float(rule_row.get("integrity_rule", 0.5))
    s["current_period"]    = int(current_period)
    s["periods_remaining"] = FINAL_PERIOD - int(current_period)
    s["projected"] = s["observed"] + s["slope"] * s["periods_remaining"]
    s["proj_pct"]  = round(s["projected"] / s["target"] * 100, 1) if s["target"] > 0 else 0
    s["pct_of_target"] = round(s["observed"] / s["target"] * 100, 1) if s["target"] > 0 else 0
    s["sibling_label"] = (
        "best funded" if s["sibling_r"] < 0.3
        else "mid-ranked" if s["sibling_r"] < 0.7
        else "worst funded"
    )
    s["fitness_label"] = "within optimal band" if s["fitness"] == 1.0 else "outside optimal band"
    return s


def prompt_narrative(s):
    """Long analyst brief for Llama 3.3 70B."""
    return f"""You are scoring an organisational goal at period {s['current_period']} of {FINAL_PERIOD} across 4 dimensions. Use the FULL 0.0-1.0 scale.

GOAL DATA:
  Current value      : {s['observed']:.3f}
  Target (period {FINAL_PERIOD}) : {s['target']:.3f}
  Gap remaining      : {s['variance']:.3f}
  Trend/period       : {s['slope']:.4f}
  Projected @ P{FINAL_PERIOD}    : {s['projected']:.3f}  ({s['periods_remaining']} periods remain)
  Output quality     : {s['quality']:.2f}
  Output quantity    : {s['quantity']:.1f}
  Needle move ratio  : {s['needle']:.3f}  (observed / expected)
  Allocation         : {s['alloc_pct']:.1%} of parent  ({s['fitness_label']})
  Efficiency ratio   : {s['efficiency']:.3f}
  Sibling rank       : {s['sibling_label']} ({s['sibling_r']:.2f})
  L3 share of L2     : {s['l3_l2']:.2f}
  Alloc drift (std)  : {s['drift']:.4f}
  Time to green      : {s['ttg']:.0f} periods
  Weighted status    : {s['wgs']:.3f}
  Rule-based scores  : relevance={s['r_rule']:.2f}  coherence={s['c_rule']:.2f}  integrity={s['i_rule']:.2f}

SCORING DEFINITIONS:
  RELEVANCE (0-1): Is the allocation justified against stated goals?
  COHERENCE (0-1): Are decisions consistent across levels, goals, and time?
  INTEGRITY (0-1): Are assumptions transparent and outcomes honest?
  ATTAINABILITY (0-1): Is the goal realistically achievable?

Scale: 0.05-0.15=very poor | 0.15-0.35=poor | 0.35-0.55=moderate | 0.55-0.75=good | 0.75-0.95=very good

Return ONLY valid JSON.
Do not include headings.
Do not include bullet points.
Do not explain before the JSON.
Do not explain after the JSON.

{{"relevance": 0.XX, "coherence": 0.XX, "integrity": 0.XX, "attainability": 0.XX, "relevance_reason": "scores X because ...", "coherence_reason": "scores X because ...", "integrity_reason": "scores X because ...", "attainability_reason": "scores X because ..."}}"""


def prompt_rubric(s):
    """Short rubric prompt for DeepSeek V3. Pair with JSON-only system message."""
    return f"""Score this goal using the rubric.

EVIDENCE (period {s['current_period']} of {FINAL_PERIOD}):
  observed={s['observed']:.3f}  target={s['target']:.3f}  progress={s['pct_of_target']}%
  slope={s['slope']:+.4f}/period  projected_at_p{FINAL_PERIOD}={s['projected']:.3f} ({s['proj_pct']}% of target)
  periods_remaining={s['periods_remaining']}
  alloc_pct={s['alloc_pct']:.3f}  sibling_rank_pct={s['sibling_r']:.2f}  fitness={s['fitness_label']}
  efficiency={s['efficiency']:.3f}  needle_move_ratio={s['needle']:.3f}  quality={s['quality']:.2f}
  alloc_drift_std={s['drift']:.4f}  weighted_status={s['wgs']:.3f}

RUBRIC:

ATTAINABILITY (projected_pct driven):
  >= 90%  -> 0.80-0.95
  65-89%  -> 0.55-0.79
  40-64%  -> 0.35-0.54
  20-39%  -> 0.15-0.34
  < 20%   -> 0.05-0.14

RELEVANCE (sibling + fitness):
  sibling < 0.3 AND in band   -> 0.80-0.95
  sibling < 0.5 OR mostly band -> 0.55-0.79
  sibling 0.5-0.7 OR out band  -> 0.35-0.54
  sibling > 0.7 AND out band   -> 0.15-0.34
  sibling > 0.9                -> 0.05-0.14

COHERENCE (drift + status):
  drift < 0.03 AND status > 0.5 -> 0.80-0.95
  drift < 0.06 OR status > 0.3  -> 0.55-0.79
  drift 0.06-0.10               -> 0.35-0.54
  drift > 0.10 OR status < 0.15 -> 0.15-0.34
  drift > 0.15 AND status < 0.10 -> 0.05-0.14

INTEGRITY (efficiency + needle):
  eff > 0.5 AND needle > 0.7  -> 0.80-0.95
  eff > 0.3 AND needle > 0.5  -> 0.55-0.79
  eff > 0.2 OR needle > 0.3   -> 0.35-0.54
  eff < 0.2 AND needle < 0.3  -> 0.15-0.34
  eff < 0.1 AND needle < 0.2  -> 0.05-0.14

Return this JSON exactly and nothing else:
{{"relevance":0.XX,"coherence":0.XX,"integrity":0.XX,"attainability":0.XX,"relevance_reason":"one sentence citing the signal","coherence_reason":"one sentence citing the signal","integrity_reason":"one sentence citing the signal","attainability_reason":"one sentence citing the signal"}}"""


PROMPT_FN = {
    "llama"  : prompt_narrative,
    "mistral": prompt_narrative,
}


def build_prompt(row, rule_row, current_period=18, model_label="llama"):
    s = _signals_inline(row, rule_row, current_period)
    fn = PROMPT_FN.get(model_label, prompt_narrative)
    return fn(s)


def predict_model(model, prompt, verbose=True):
    label = model["label"]
    role  = model["role"]
    opts  = MODEL_OPTIONS.get(label, {"temperature": 0.2, "max_tokens": 1200})
    system = SYSTEM_PROMPTS.get(label)
    prefill = PREFILL.get(label)
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0  = time.time()
            raw = call_model(
                role,
                prompt,
                max_tokens=opts["max_tokens"],
                temperature=opts["temperature"],
                system=system,
                prefill=prefill,
            )
            parsed = extract_json(raw, required_keys=DIMS)

            result = {d: float(np.clip(float(parsed[d]), 0.05, 0.95)) for d in DIMS}
            for d in DIMS:
                reason = re.sub(r"<think>.*?</think>", "", str(parsed.get(f"{d}_reason", "")), flags=re.DOTALL).strip()
                result[f"{d}_reason"] = reason

            result["success"]   = True
            result["elapsed_s"] = round(time.time() - t0, 1)

            if verbose:
                print(
                    f"  {label:<12} A={result['attainability']:.2f} "
                    f"R={result['relevance']:.2f} C={result['coherence']:.2f} "
                    f"I={result['integrity']:.2f}  ({result['elapsed_s']}s)"
                )
                for d in DIMS:
                    r = result.get(f"{d}_reason", "")
                    if r:
                        print(f"    {d}: {r}")
            return result

        except (BedrockCallError, ValueError) as e:
            last_err = e
            if verbose:
                print(f"  {label:<12} attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2)
        except Exception as e:
            last_err = e
            if verbose:
                print(f"  {label:<12} attempt {attempt}/{MAX_RETRIES} unexpected: {type(e).__name__}: {e}")
            time.sleep(2)

    return {d: None for d in DIMS} | {f"{d}_reason": "" for d in DIMS} | {
        "success": False, "elapsed_s": 0, "error": str(last_err)
    }


def score_goal(goal_row, rule_row, verbose=True, current_period=18):
    gp, iso_scalers, feat_scaler, cfg = load_artefacts()
    feat_names = cfg["feature_names"]

    row_vals = {k: float(goal_row.get(k, 0)) for k in feat_names}
    X_raw    = pd.DataFrame([row_vals])[feat_names]
    X_scaled = feat_scaler.transform(X_raw)

    baseline = compute_baseline(
        observed=float(goal_row.get("observed_value", 0)),
        slope=float(goal_row.get("trailing_6_period_slope", 0)),
        target=float(goal_row.get("target_value_final_period", 1)),
        period=current_period,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_resid, gp_std_arr = gp.predict(X_scaled, return_std=True)

    gp_mean   = float(np.clip(baseline + gp_resid[0], 0, 1))
    gp_std    = float(gp_std_arr[0])
    gp_conf   = 1.0 / (1.0 + gp_std * cfg["gp_uncertainty_scale"])
    uncertain = bool(gp_std > cfg["gp_uncertainty_flag"])

    if verbose:
        print(
            f"\nGP: baseline={baseline:.3f}  residual={gp_resid[0]:+.3f}  "
            f"final={gp_mean:.3f}  std={gp_std:.4f}  conf={gp_conf:.3f}"
            f"{'  UNCERTAIN' if uncertain else ''}"
        )
        print(f"\nCalling Bedrock models (p{current_period})...")

    llm_results = {}
    for m in BEDROCK_MODELS:
        prompt = build_prompt(goal_row, rule_row, current_period=current_period, model_label=m["label"])
        llm_results[m["label"]] = predict_model(m, prompt, verbose=verbose)

    n_ok = sum(1 for r in llm_results.values() if r["success"])
    if n_ok == 0:
        return {
            "status"    : "error",
            "message"   : "All LLM calls failed",
            "goal_id"   : int(rule_row.get("goal_id", -1)),
            "gp_mean"   : gp_mean,
            "gp_std"    : gp_std,
            "uncertain" : uncertain,
        }

    cal_attain = []
    for m in BEDROCK_MODELS:
        label = m["label"]
        res   = llm_results.get(label, {})
        if not res.get("success") or res.get("attainability") is None:
            continue
        raw = res["attainability"]
        cal = float(np.clip(iso_scalers[label].predict([raw])[0], 0, 1)) if label in iso_scalers else raw
        cal_attain.append(cal)

    if cal_attain:
        llm_mean = float(np.mean(cal_attain))
        llm_w    = 1.0 - gp_conf
        attainability = float(np.clip(gp_conf * gp_mean + llm_w * llm_mean, 0, 1))
    else:
        llm_mean = None
        llm_w    = 0.0
        attainability = float(np.clip(gp_mean, 0, 1))

    def blend_dim(dim, rule_val):
        ok = [
            float(llm_results[m["label"]][dim])
            for m in BEDROCK_MODELS
            if llm_results.get(m["label"], {}).get("success")
            and llm_results[m["label"]][dim] is not None
        ]
        if len(ok) < 2:
            return float(np.clip(rule_val, 0, 1)), {
                "rule_weight": 1.0,
                "llm_weight" : 0.0,
                "variance"   : 0.0,
                "fallback"   : True,
            }

        arr = np.array(ok)
        lm  = float(arr.mean())
        var = float(arr.var())
        lc  = 1.0 / (1.0 + var * 10.0)
        rc  = 1.0 / (1.0 + abs(lm - rule_val) * 5.0)
        wr  = 0.5 * rc
        wl  = 0.5 * lc
        tot = wr + wl
        wr /= tot
        wl /= tot
        return float(np.clip(wr * rule_val + wl * lm, 0, 1)), {
            "rule_weight": round(wr, 3),
            "llm_weight" : round(wl, 3),
            "variance"   : round(var, 4),
            "llm_mean"   : round(lm, 3),
            "fallback"   : False,
        }

    relevance, rel_m = blend_dim("relevance", float(rule_row.get("relevance_rule", 0.5)))
    coherence, coh_m = blend_dim("coherence", float(rule_row.get("coherence_rule", 0.5)))
    integrity, int_m = blend_dim("integrity", float(rule_row.get("integrity_rule", 0.5)))
    overall = float(np.mean([attainability, relevance, coherence, integrity]))

    reasoning = {}
    for d in DIMS:
        for m in BEDROCK_MODELS:
            r = llm_results.get(m["label"], {}).get(f"{d}_reason", "")
            if r:
                reasoning[d] = r
                break
        if d not in reasoning:
            reasoning[d] = ""

    dim_weights = {"coherence": 0.25, "attainability": 0.25, "relevance": 0.25, "integrity": 0.25}
    goal_composite = float(sum(
        {"attainability": attainability, "relevance": relevance, "coherence": coherence, "integrity": integrity}[d] * w
        for d, w in dim_weights.items()
    ))

    if verbose:
        print("\nRunning verifier...")
    signals = dict(goal_row) if hasattr(goal_row, "to_dict") else goal_row
    signals["current_period"] = current_period
    ver = verify_scores(
        scores={"attainability": attainability, "relevance": relevance, "coherence": coherence, "integrity": integrity},
        signals=signals,
        composite=goal_composite,
        weights=dim_weights,
        ensemble_meta={
            "attainability": {},
            "relevance": rel_m,
            "coherence": coh_m,
            "integrity": int_m,
        },
        verbose=verbose,
    )

    if verbose:
        print(
            f"\n  Attainability : {attainability:.3f}  "
            f"(gp={gp_mean:.3f}x{gp_conf:.2f} + llm={'N/A' if llm_mean is None else f'{llm_mean:.3f}'}x{llm_w:.2f})"
            f"{'  UNCERTAIN' if uncertain else ''}"
        )
        print(f"  Relevance     : {relevance:.3f}")
        print(f"  Coherence     : {coherence:.3f}")
        print(f"  Integrity     : {integrity:.3f}")
        print(f"  Overall       : {overall:.3f}")

    goal_id = int(rule_row.get("goal_id", -1))
    return {
        "goal_id"      : goal_id,
        "attainability": round(attainability, 4),
        "relevance"    : round(relevance, 4),
        "coherence"    : round(coherence, 4),
        "integrity"    : round(integrity, 4),
        "overall"      : round(overall, 4),
        "gp_mean"      : round(gp_mean, 4),
        "gp_std"       : round(gp_std, 4),
        "gp_weight"    : round(gp_conf, 3),
        "llm_weight"   : round(llm_w, 3),
        "baseline"     : round(baseline, 4),
        "uncertain"    : uncertain,
        "llm_scores"   : {
            m["label"]: {
                d: round(float(llm_results[m["label"]][d]), 4)
                if llm_results[m["label"]]["success"] and llm_results[m["label"]][d] is not None
                else None
                for d in DIMS
            } | {"success": llm_results[m["label"]]["success"]}
            for m in BEDROCK_MODELS
        },
        "reasoning"    : reasoning,
        "ensemble_meta": {
            "attainability": {
                "gp_mean"  : gp_mean,
                "gp_std"   : gp_std,
                "gp_weight": gp_conf,
                "llm_weight": llm_w,
                "llm_mean" : llm_mean,
                "baseline" : baseline,
                "uncertain": uncertain,
            },
            "relevance": rel_m,
            "coherence": coh_m,
            "integrity": int_m,
        },
        "n_llm_ok": n_ok,
        "status"  : "ok",
        "dependencies": {
            "depends_on"     : json.loads(dep_map.get(goal_id, {}).get("depends_on_ids", "[]")),
            "depended_on_by" : json.loads(dep_map.get(goal_id, {}).get("depended_on_by_ids", "[]")),
            "dependency_risk": dep_map.get(goal_id, {}).get("dependency_risk", "none"),
            "dep_avg_attain" : dep_map.get(goal_id, {}).get("dep_avg_attain", -1.0),
            "n_dependencies" : dep_map.get(goal_id, {}).get("n_dependencies", 0),
        },
        "verified_attainability": ver["adjusted_attainability"],
        "verified_relevance"    : ver["adjusted_relevance"],
        "verified_coherence"    : ver["adjusted_coherence"],
        "verified_integrity"    : ver["adjusted_integrity"],
        "verified_composite"    : ver["adjusted_composite"],
        "flags"                 : ver["flags"],
        "narrative"             : ver["narrative"],
        "verified"              : ver["verified"],
        "adjustments"           : ver["adjustments"],
        "verifier_model"        : ver.get("verifier_model"),
    }


def merge_extra_columns(features_df, period_df):
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
        # Shock signals engineered by feature_engineering.py
        "budget_shock_exposure",
        "shock_alloc_impact",
        "recovery_period_estimate",
        "recovery_window_remaining",
        "market_shock_vulnerable",
        "market_shock_forward_risk",
    ]
    out = features_df.copy()
    for col in extra_cols:
        if col not in out.columns and col in period_df.columns:
            out[col] = period_df[col].values
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal_id", type=int, default=0)
    parser.add_argument("--period", type=int, default=None, help="Optional single period: 6, 12, or 18")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f"DECIDR COHERENCE ENGINE  Inference (goal index {args.goal_id})")
    print("=" * 70)

    period_ref_candidate = "period_18_poc.csv" if os.path.exists("period_18_poc.csv") else "period_12_poc.csv"

    period_configs = {
        6 : {"features": "features_raw_p6.csv",  "rules": "rule_scores_p6.csv",  "period_ref": period_ref_candidate},
        12: {"features": "features_raw_poc.csv" if os.path.exists("features_raw_poc.csv") else "features_raw_p12.csv",
             "rules"   : "rule_scores_poc.csv" if os.path.exists("rule_scores_poc.csv") else "rule_scores_p12.csv",
             "period_ref": period_ref_candidate},
        18: {"features": "features_raw_p18.csv", "rules": "rule_scores_p18.csv", "period_ref": period_ref_candidate},
    }

    # p24 inference supported only if someone explicitly asks; not in default loop
    if args.period == 24:
        period_configs[24] = {
            "features": "features_raw_p24.csv",
            "rules"   : "rule_scores_p24.csv",
            "period_ref": period_ref_candidate,
        }

    periods_to_run = [args.period] if args.period else [6, 12, 18]
    results = []

    for period in periods_to_run:
        cfg = period_configs.get(period)
        if cfg is None:
            print(f"\nSkipping invalid period: {period}")
            continue
        if not os.path.exists(cfg["features"]):
            print(f"\nSkipping period {period}  missing {cfg['features']}")
            continue
        if not os.path.exists(cfg["rules"]):
            print(f"\nSkipping period {period}  missing {cfg['rules']}")
            continue

        print("\n" + "=" * 70)
        print(f"RUNNING PERIOD {period}")
        print("=" * 70)

        features_raw = pd.read_csv(cfg["features"])
        rule_scores  = pd.read_csv(cfg["rules"])
        period_ref   = pd.read_csv(cfg["period_ref"]).reset_index(drop=True)
        features_raw = merge_extra_columns(features_raw, period_ref)

        if args.goal_id >= len(features_raw):
            raise IndexError(f"goal_id {args.goal_id} out of range for period {period}. Max index is {len(features_raw)-1}")

        result = score_goal(
            features_raw.iloc[args.goal_id],
            rule_scores.iloc[args.goal_id],
            verbose=True,
            current_period=period,
        )
        result["period"] = period
        results.append(result)

    print("\n" + "=" * 70)
    print("SCORE PAYLOAD:")
    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
        payload_to_save = results[0]
    else:
        print(json.dumps(results, indent=2))
        payload_to_save = results

    if args.output:
        with open(args.output, "w") as f:
            json.dump(payload_to_save, f, indent=2)
        print(f"\nOK  Saved to {args.output}")

    print("=" * 70)