"""
07_score_goal.py — v6 Inference via Ollama local
Updated for v4 feature engineering and Tom's hierarchy signals.

Usage:
    python 07_score_goal.py --goal_id 0
    python 07_score_goal.py --goal_id 7 --output result.json

Called by System 3:
    from 07_score_goal import score_goal
    result = score_goal(goal_row, rule_row)
"""

import os, json, re, pickle, argparse, warnings, time
from verify_goal import verify_scores
import pandas as pd
import numpy as np
import urllib.request
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
TIMEOUT_SEC = 300
MAX_RETRIES = 3

OLLAMA_MODELS = [
    {"id": "llama3:latest", "label": "llama3"},
    {"id": "gemma3:4b",     "label": "gemma3"},
    {"id": "qwen3:4b",      "label": "qwen3"},
]

DIMS = ["relevance", "coherence", "integrity", "attainability"]


def load_artefacts():
    with open("gp_poc.pkl","rb")            as f: gp     = pickle.load(f)
    with open("platt_scalers_poc.pkl","rb") as f: iso    = pickle.load(f)
    with open("feature_scaler_poc.pkl","rb")as f: scaler = pickle.load(f)
    with open("gp_config_poc.json")         as f: cfg    = json.load(f)
    return gp, iso, scaler, cfg


def compute_baseline(observed, slope, target, period=12, total=24):
    remaining = total - period
    return float(np.clip((observed + slope * remaining) / max(target, 1e-9), 0, 1))


def build_prompt(row, rule_row):
    slope     = float(row.get("trailing_6_period_slope", 0))
    variance  = float(row.get("variance_from_target", 0))
    quality   = float(row.get("delivered_output_quality_score", 0))
    observed  = float(row.get("observed_value", 0))
    target    = float(row.get("target_value_final_period", 0))
    alloc_pct = float(row.get("allocation_percentage_of_parent", 0))
    projected = observed + slope * 12
    efficiency= float(row.get("allocation_efficiency_ratio", 0))
    quantity  = float(row.get("delivered_output_quantity", 0))
    ttg       = float(row.get("time_to_green_estimate", 0))
    wgs       = float(row.get("weighted_goal_status_score", 0.5))
    sibling_r = float(row.get("sibling_rank_pct", 0.5))
    l3_l2     = float(row.get("l3_share_of_l2", 0.4))
    drift     = float(row.get("alloc_drift_std", 0))
    needle    = float(row.get("needle_move_ratio", 0.5))
    fitness   = float(row.get("allocation_fitness_score", 0))
    r_rule    = float(rule_row.get("relevance_rule", 0.5))
    c_rule    = float(rule_row.get("coherence_rule", 0.5))
    i_rule    = float(rule_row.get("integrity_rule", 0.5))
    sibling_label = "best funded" if sibling_r < 0.3 else "mid-ranked" if sibling_r < 0.7 else "worst funded"
    fitness_label = "within optimal band" if fitness == 1.0 else "outside optimal band"

    return f"""You are scoring an organisational goal across 4 dimensions. Use the FULL 0.0-1.0 scale.

GOAL DATA (period 12 of 24):
  Current value      : {observed:.3f}
  Target (period 24) : {target:.3f}
  Gap remaining      : {variance:.3f}
  Trend/period       : {slope:.4f}
  Projected @ P24    : {projected:.3f}
  Output quality     : {quality:.2f}
  Output quantity    : {quantity:.1f}
  Needle move ratio  : {needle:.3f}  (observed / expected)
  Allocation         : {alloc_pct:.1%} of parent  ({fitness_label})
  Efficiency ratio   : {efficiency:.3f}
  Sibling rank       : {sibling_label} ({sibling_r:.2f})
  L3 share of L2     : {l3_l2:.2f}
  Alloc drift (std)  : {drift:.4f}
  Time to green      : {ttg:.0f} periods
  Weighted status    : {wgs:.3f}
  Rule-based scores  : relevance={r_rule:.2f}  coherence={c_rule:.2f}  integrity={i_rule:.2f}

SCORING DEFINITIONS:
  RELEVANCE (0-1): Is the allocation justified against stated goals?
  COHERENCE (0-1): Are decisions consistent across levels, goals, and time?
  INTEGRITY (0-1): Are assumptions transparent and outcomes honest?
  ATTAINABILITY (0-1): Is the goal realistically achievable?

Scale: 0.05-0.15=very poor | 0.15-0.35=poor | 0.35-0.55=moderate | 0.55-0.75=good | 0.75-0.95=very good

Return ONLY this JSON on one line:
{{"relevance": 0.XX, "coherence": 0.XX, "integrity": 0.XX, "attainability": 0.XX, "relevance_reason": "scores X because ...", "coherence_reason": "scores X because ...", "integrity_reason": "scores X because ...", "attainability_reason": "scores X because ..."}}"""


def build_prompt_nemotron(row, rule_row):
    slope     = float(row.get("trailing_6_period_slope", 0))
    variance  = float(row.get("variance_from_target", 0))
    quality   = float(row.get("delivered_output_quality_score", 0))
    observed  = float(row.get("observed_value", 0))
    target    = float(row.get("target_value_final_period", 0))
    alloc_pct = float(row.get("allocation_percentage_of_parent", 0))
    projected = observed + slope * 12
    efficiency= float(row.get("allocation_efficiency_ratio", 0))
    sibling_r = float(row.get("sibling_rank_pct", 0.5))
    l3_l2     = float(row.get("l3_share_of_l2", 0.4))
    drift     = float(row.get("alloc_drift_std", 0))
    needle    = float(row.get("needle_move_ratio", 0.5))
    r_rule    = float(rule_row.get("relevance_rule", 0.5))
    c_rule    = float(rule_row.get("coherence_rule", 0.5))
    i_rule    = float(rule_row.get("integrity_rule", 0.5))
    return f"""Score this goal on 4 dimensions using any decimal from 0.05 to 0.95.
Do not round to 0.1 or 0.5. Use precise decimals based on the data.

Data:
  current={observed:.3f}  target={target:.3f}  gap={variance:.3f}
  trend={slope:+.4f}/period  projected_p24={projected:.3f}
  quality={quality:.2f}  allocation={alloc_pct:.1%}  efficiency={efficiency:.3f}
  sibling_rank={sibling_r:.2f}  l3_share_of_l2={l3_l2:.2f}
  alloc_drift={drift:.4f}  needle_move_ratio={needle:.3f}
  rule_relevance={r_rule:.2f}  rule_coherence={c_rule:.2f}  rule_integrity={i_rule:.2f}

Return ONLY this JSON:
{{"relevance":0.XX,"coherence":0.XX,"integrity":0.XX,"attainability":0.XX,"relevance_reason":"brief","coherence_reason":"brief","integrity_reason":"brief","attainability_reason":"brief"}}"""


def call_ollama(model_id, prompt):
    use_json = any(m in model_id for m in ["qwen3","nemotron","deepseek"])
    payload  = {"model":model_id,"prompt":prompt,"stream":False,
                "options":{"temperature":0.2,"num_predict":400,"top_p":0.9,
                           "repeat_penalty":1.1,"stop":["}\n","\n\n"]}}
    if use_json: payload["format"] = "json"
    req = urllib.request.Request(OLLAMA_URL,
          data=json.dumps(payload).encode("utf-8"),
          headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        return json.loads(resp.read()).get("response","").strip()


def extract_json(text):
    text = re.sub(r"<think>.*?</think>","",text,flags=re.DOTALL)
    text = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>","",text,flags=re.DOTALL).strip()
    if not text: raise ValueError("Empty response")
    for pattern in [r"\{.*\}", r"\{[^{}]+\}"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    result = {}
    for d in DIMS:
        m2 = re.search(rf'"{d}"\s*:\s*([0-9.]+)', text)
        if m2: result[d] = float(m2.group(1))
    if len(result) == 4:
        for d in DIMS: result[f"{d}_reason"] = ""
        return result
    raise ValueError(f"No JSON in: {text[:120]}")


def predict_model(model, prompt, verbose=True):
    label = model["label"]
    for attempt in range(1, MAX_RETRIES+1):
        try:
            t0     = time.time()
            raw    = call_ollama(model["id"], prompt)
            parsed = extract_json(raw)
            for d in DIMS:
                if d not in parsed: raise ValueError(f"Missing: {d}")
            result = {d: float(np.clip(float(parsed[d]),0.05,0.95)) for d in DIMS}
            for d in DIMS:
                reason = re.sub(r"<think>.*?</think>","",str(parsed.get(f"{d}_reason","")),flags=re.DOTALL).strip()
                result[f"{d}_reason"] = reason
            result["success"]   = True
            result["elapsed_s"] = round(time.time()-t0, 1)
            if verbose:
                print(f"  {label:<12} A={result['attainability']:.2f} "
                      f"R={result['relevance']:.2f} C={result['coherence']:.2f} "
                      f"I={result['integrity']:.2f}  ({result['elapsed_s']}s)")
                for d in DIMS:
                    r = result.get(f"{d}_reason","")
                    if r: print(f"    {d}: {r}")
            return result
        except Exception as e:
            if verbose: print(f"  {label:<12} attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2)
    return {d:None for d in DIMS} | {f"{d}_reason":"" for d in DIMS} | {"success":False,"elapsed_s":0}


def score_goal(goal_row, rule_row, verbose=True):
    gp, iso_scalers, feat_scaler, cfg = load_artefacts()
    feat_names    = cfg["feature_names"]
    ATTAIN_MODELS = cfg.get("attain_models", cfg["models"])

    # GP prediction
    row_vals = {k: float(goal_row.get(k, 0)) for k in feat_names}
    X_raw    = pd.DataFrame([row_vals])[feat_names]
    X_scaled = feat_scaler.transform(X_raw)

    baseline = compute_baseline(
        observed = float(goal_row.get("observed_value", 0)),
        slope    = float(goal_row.get("trailing_6_period_slope", 0)),
        target   = float(goal_row.get("target_value_final_period", 1)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_resid, gp_std_arr = gp.predict(X_scaled, return_std=True)

    gp_mean   = float(np.clip(baseline + gp_resid[0], 0, 1))
    gp_std    = float(gp_std_arr[0])
    gp_conf   = 1.0 / (1.0 + gp_std * cfg["gp_uncertainty_scale"])
    uncertain = bool(gp_std > cfg["gp_uncertainty_flag"])

    if verbose:
        print(f"\nGP: baseline={baseline:.3f}  residual={gp_resid[0]:+.3f}  "
              f"final={gp_mean:.3f}  std={gp_std:.4f}  conf={gp_conf:.3f}"
              f"{'  UNCERTAIN' if uncertain else ''}")
        print("\nCalling Ollama models...")

    llm_results = {}
    for m in OLLAMA_MODELS:
        prompt = build_prompt_nemotron(goal_row, rule_row) if m["label"] == "nemotron" \
                 else build_prompt(goal_row, rule_row)
        llm_results[m["label"]] = predict_model(m, prompt, verbose=verbose)

    n_ok = sum(1 for r in llm_results.values() if r["success"])
    if n_ok == 0:
        return {"status":"error","message":"All LLM calls failed",
                "goal_id":int(rule_row.get("goal_id",-1)),
                "gp_mean":gp_mean,"gp_std":gp_std,"uncertain":uncertain}

    # Isotonic calibration
    cal_attain = []
    for label in ATTAIN_MODELS:
        res = llm_results.get(label,{})
        if not res.get("success") or res.get("attainability") is None: continue
        raw = res["attainability"]
        cal = float(np.clip(iso_scalers[label].predict([raw])[0],0,1)) \
              if label in iso_scalers else raw
        cal_attain.append(cal)

    if cal_attain:
        llm_mean      = float(np.mean(cal_attain))
        llm_w         = 1.0 - gp_conf
        attainability = float(np.clip(gp_conf * gp_mean + llm_w * llm_mean, 0, 1))
    else:
        llm_mean      = None
        llm_w         = 0.0
        attainability = float(np.clip(gp_mean, 0, 1))

    def blend_dim(dim, rule_val):
        ok = [float(llm_results[m["label"]][dim])
              for m in OLLAMA_MODELS
              if llm_results.get(m["label"],{}).get("success")
              and llm_results[m["label"]][dim] is not None]
        if len(ok) < 2:
            return float(np.clip(rule_val,0,1)), \
                   {"rule_weight":1.0,"llm_weight":0.0,"variance":0.0,"fallback":True}
        arr  = np.array(ok)
        lm   = float(arr.mean()); var = float(arr.var())
        lc   = 1.0/(1.0+var*10.0); rc = 1.0/(1.0+abs(lm-rule_val)*5.0)
        wr   = 0.5*rc; wl = 0.5*lc; tot = wr+wl; wr/=tot; wl/=tot
        return float(np.clip(wr*rule_val+wl*lm,0,1)), \
               {"rule_weight":round(wr,3),"llm_weight":round(wl,3),
                "variance":round(var,4),"llm_mean":round(lm,3),"fallback":False}

    relevance, rel_m = blend_dim("relevance", float(rule_row.get("relevance_rule",0.5)))
    coherence, coh_m = blend_dim("coherence", float(rule_row.get("coherence_rule",0.5)))
    integrity, int_m = blend_dim("integrity", float(rule_row.get("integrity_rule",0.5)))
    overall          = float(np.mean([attainability,relevance,coherence,integrity]))

    reasoning = {}
    for d in DIMS:
        for m in OLLAMA_MODELS:
            r = llm_results.get(m["label"],{}).get(f"{d}_reason","")
            if r: reasoning[d] = r; break
        if d not in reasoning: reasoning[d] = ""

    # ── Compute per-goal composite ───────────────────────────────────────────
    dim_weights = {"coherence": 0.35, "attainability": 0.25,
                   "relevance": 0.20, "integrity": 0.20}
    goal_composite = float(sum(
        {"attainability": attainability, "relevance": relevance,
         "coherence": coherence, "integrity": integrity}[d] * w
        for d, w in dim_weights.items()
    ))

    # ── Verification ─────────────────────────────────────────────────────────
    if verbose: print("\nRunning verifier (qwen3)...")
    ver = verify_scores(
        scores     = {"attainability": attainability, "relevance": relevance,
                      "coherence": coherence, "integrity": integrity},
        signals    = dict(goal_row) if hasattr(goal_row, "to_dict") else goal_row,
        composite  = goal_composite,
        weights    = dim_weights,
        ensemble_meta = {
            "attainability": blend_meta if "blend_meta" in dir() else {},
            "relevance": rel_m, "coherence": coh_m, "integrity": int_m,
        },
        verbose    = verbose,
    )

    if verbose:
        print(f"\n  Attainability : {attainability:.3f}  (gp={gp_mean:.3f}×{gp_conf:.2f} + llm={'N/A' if llm_mean is None else f'{llm_mean:.3f}'}×{llm_w:.2f}){'  UNCERTAIN' if uncertain else ''}")
        print(f"  Relevance     : {relevance:.3f}")
        print(f"  Coherence     : {coherence:.3f}")
        print(f"  Integrity     : {integrity:.3f}")
        print(f"  Overall       : {overall:.3f}")

    return {
        "goal_id"      : int(rule_row.get("goal_id",-1)),
        "attainability": round(attainability,4),
        "relevance"    : round(relevance,4),
        "coherence"    : round(coherence,4),
        "integrity"    : round(integrity,4),
        "overall"      : round(overall,4),
        "gp_mean"      : round(gp_mean,4),
        "gp_std"       : round(gp_std,4),
        "gp_weight"    : round(gp_conf,3),
        "llm_weight"   : round(llm_w,3),
        "baseline"     : round(baseline,4),
        "uncertain"    : uncertain,
        "llm_scores"   : {m["label"]:{d:round(float(llm_results[m["label"]][d]),4)
                          if llm_results[m["label"]]["success"]
                          and llm_results[m["label"]][d] is not None else None
                          for d in DIMS} | {"success":llm_results[m["label"]]["success"]}
                          for m in OLLAMA_MODELS},
        "reasoning"    : reasoning,
        "ensemble_meta": {"attainability":{"gp_mean":gp_mean,"gp_std":gp_std,
                          "gp_weight":gp_conf,"llm_weight":llm_w,
                          "llm_mean":llm_mean,"baseline":baseline,"uncertain":uncertain},
                          "relevance":rel_m,"coherence":coh_m,"integrity":int_m},
        "n_llm_ok"     : n_ok,
        "status"       : "ok",
        "dependencies" : {
            "depends_on"       : json.loads(dep_map.get(int(rule_row.get("goal_id",-1)),{}).get("depends_on_ids","[]")),
            "depended_on_by"   : json.loads(dep_map.get(int(rule_row.get("goal_id",-1)),{}).get("depended_on_by_ids","[]")),
            "dependency_risk"  : dep_map.get(int(rule_row.get("goal_id",-1)),{}).get("dependency_risk","none"),
            "dep_avg_attain"   : dep_map.get(int(rule_row.get("goal_id",-1)),{}).get("dep_avg_attain",-1.0),
            "n_dependencies"   : dep_map.get(int(rule_row.get("goal_id",-1)),{}).get("n_dependencies",0),
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

    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal_id", type=int, default=0)
    parser.add_argument("--output",  type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f"DECIDR COHERENCE ENGINE — Inference (goal index {args.goal_id})")
    print("=" * 70)

    # Load dependency map if available
    import os as _os
    dep_map = {}
    if _os.path.exists("goal_dependencies.csv"):
        dep_df_inf = pd.read_csv("goal_dependencies.csv")
        dep_map = dep_df_inf.set_index("goal_id").to_dict("index")

    features_raw = pd.read_csv("features_raw_poc.csv")
    rule_scores  = pd.read_csv("rule_scores_poc.csv")
    period_12    = pd.read_csv("period_12_poc.csv").reset_index(drop=True)

    extra_cols = ["allocation_efficiency_ratio","delivered_output_quantity",
                  "allocation_percentage_of_parent","target_value_final_period",
                  "time_to_green_estimate","weighted_goal_status_score",
                  "sibling_rank_pct","l3_share_of_l2","alloc_drift_std",
                  "needle_move_ratio","allocation_fitness_score",
                  "observed_value","variance_from_target"]
    for col in extra_cols:
        if col not in features_raw.columns and col in period_12.columns:
            features_raw[col] = period_12[col].values

    result = score_goal(features_raw.iloc[args.goal_id],
                        rule_scores.iloc[args.goal_id], verbose=True)

    print("\n" + "=" * 70)
    print("SCORE PAYLOAD:")
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output,"w") as f: json.dump(result, f, indent=2)
        print(f"\n✓ Saved to {args.output}")
    print("=" * 70)
