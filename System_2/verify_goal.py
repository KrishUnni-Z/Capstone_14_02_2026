"""
verify_goal.py — Post-composite verification module
Called by 07_score_goal.py (per-goal inference) and 08_composite_score.py (batch).

Uses qwen3:4b as a reasoning verifier. Sees blended scores + key signals,
checks for internal inconsistencies, can adjust scores by at most ±ADJUST_CAP
per dimension, and returns flags + narrative for System 1.

Usage:
    from verify_goal import verify_scores

    result = verify_scores(
        scores       = {"attainability": 0.14, "relevance": 0.58, "coherence": 0.42, "integrity": 0.70},
        signals      = {"needle_move_ratio": 0.12, "alloc_drift_std": 0.14, ...},
        composite    = 0.47,
        weights      = {"coherence": 0.35, "attainability": 0.25, "relevance": 0.20, "integrity": 0.20},
        ensemble_meta= {...},   # from meta_learner_predictions_poc.csv
        verbose      = True,
    )
"""

import json
import re
import time
import urllib.request
import numpy as np

OLLAMA_URL     = "http://localhost:11434/api/generate"
VERIFIER_MODEL = "deepseek-r1:8b"
ADJUST_CAP     = 0.10      # maximum adjustment per dimension (in either direction)
TIMEOUT_SEC    = 300
MAX_RETRIES    = 2

DIMS = ["attainability", "relevance", "coherence", "integrity"]

# Hard consistency rules — if violated, verifier MUST flag regardless of LLM output
# (rule, readable description)
HARD_RULES = [
    (lambda s, sc: sc["integrity"]     > 0.55 and s.get("needle_move_ratio", 1) < 0.25,
     "integrity too high given needle_move_ratio < 0.25"),
    (lambda s, sc: sc["integrity"]     > 0.60 and s.get("allocation_efficiency_ratio", 1) < 0.20,
     "integrity too high given allocation_efficiency_ratio < 0.20"),
    (lambda s, sc: sc["coherence"]     > 0.65 and s.get("alloc_drift_std", 0) > 0.10,
     "coherence too high given alloc_drift_std > 0.10"),
    (lambda s, sc: sc["coherence"]     > 0.65 and s.get("weighted_goal_status_score", 1) < 0.15,
     "coherence too high given weighted_status in deep red zone"),
    (lambda s, sc: sc["attainability"] > 0.50 and
     (s.get("projected_p24", 1) / max(s.get("target", 1), 1e-9)) < 0.45,
     "attainability too high given projected value under 45% of target"),
    (lambda s, sc: sc["relevance"]     > 0.65 and s.get("sibling_rank_pct", 0) > 0.80,
     "relevance too high given goal is worst funded among siblings (rank > 0.80)"),
    (lambda s, sc: sc["attainability"] > 0.45 and s.get("dependency_risk_encoded", 0) >= 0.67,
     "attainability may be optimistic given upstream dependencies are at high risk"),
    (lambda s, sc: sc["integrity"]     > 0.50 and s.get("dep_avg_attain", 1.0) != -1.0
     and s.get("dep_avg_attain", 1.0) < 0.15,
     "integrity too high given upstream dependencies have very low attainability"),
    # Shock-aware rule: do not flag coherence as too high if goal is in active budget shock
    # The drift and status signals are depressed by the shock, not by poor management
    (lambda s, sc: sc["coherence"] < 0.25
     and s.get("budget_shock_exposure", 0) >= 0.33
     and s.get("shock_alloc_impact", 0) > 0.01,
     "coherence score may be understated due to active budget shock period"),
]


def _build_verifier_prompt(scores, signals, composite, weights, ensemble_meta):
    a = scores["attainability"]
    r = scores["relevance"]
    c = scores["coherence"]
    i = scores["integrity"]

    # Pull key diagnostic signals
    needle    = signals.get("needle_move_ratio", "N/A")
    drift     = signals.get("alloc_drift_std", "N/A")
    efficiency= signals.get("allocation_efficiency_ratio", "N/A")
    wgs       = signals.get("weighted_goal_status_score", "N/A")
    sibling   = signals.get("sibling_rank_pct", "N/A")
    proj_pct  = signals.get("proj_pct_of_target", "N/A")
    quality   = signals.get("delivered_output_quality_score", "N/A")
    observed  = signals.get("observed_value", "N/A")
    target    = signals.get("target_value_final_period", "N/A")
    fitness   = signals.get("allocation_fitness_score", "N/A")

    # LLM agreement per dimension from ensemble_meta
    def get_variance(dim):
        try:
            return ensemble_meta.get(dim, {}).get("variance", "N/A")
        except Exception:
            return "N/A"

    return f"""Verify these four scores against the evidence. Adjust each by -0.10 to +0.10 if inconsistent. Use 0.0 if fine.

scores: attainability={a:.3f} relevance={r:.3f} coherence={c:.3f} integrity={i:.3f} composite={composite:.3f}

evidence:
  observed={observed} target={target} projected_pct={proj_pct}%
  needle_move={needle} efficiency={efficiency} quality={quality}
  drift={drift} status={wgs} sibling_rank={sibling} fitness={fitness}

rules:
  integrity>0.55 needs needle>=0.25 and efficiency>=0.20
  coherence>0.65 needs drift<=0.10 and status>=0.15
  attainability>0.50 needs projected_pct>=45
  relevance>0.65 needs sibling_rank<=0.80

Return JSON only:
{{"adjustments":{{"attainability":0.0,"relevance":0.0,"coherence":0.0,"integrity":0.0}},"flags":[],"narrative":"one paragraph for end users","verified":true}}""".strip()


def _call_verifier(prompt):
    payload = {
        "model"  : VERIFIER_MODEL,
        "prompt" : prompt,
        "stream" : False,
        "format" : "json",
        "options": {
            "temperature"   : 0.1,
            "num_predict"   : 1500,
            "top_p"         : 0.9,
            "repeat_penalty": 1.1,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        raw = json.loads(resp.read()).get("response", "").strip()
    # Strip thinking blocks and markdown fences
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```json", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON in verifier response: {raw[:200]}")


def verify_scores(scores, signals, composite, weights, ensemble_meta=None, verbose=True):
    """
    Parameters
    ----------
    scores       : dict  — {"attainability": float, "relevance": float,
                             "coherence": float, "integrity": float}
    signals      : dict  — key diagnostic signals from features_raw_poc.csv / period_12
    composite    : float — current composite score
    weights      : dict  — dimension weights for recomputing composite
    ensemble_meta: dict  — metadata from blend step (variance per dim, optional)
    verbose      : bool

    Returns
    -------
    dict with keys:
        adjusted_attainability, adjusted_relevance, adjusted_coherence, adjusted_integrity,
        adjusted_composite, original_composite, flags, narrative, verified,
        adjustments, hard_rule_flags, verifier_ok
    """
    if ensemble_meta is None:
        ensemble_meta = {}

    # ── Step 1: Run hard consistency rules regardless of LLM ──────────────────
    hard_flags = []
    for rule_fn, desc in HARD_RULES:
        try:
            if rule_fn(signals, scores):
                hard_flags.append(desc)
        except Exception:
            pass

    # ── Step 2: Call qwen3 verifier ───────────────────────────────────────────
    # Add proj_pct to signals for prompt
    obs    = signals.get("observed_value", 0)
    tgt    = signals.get("target_value_final_period", 1)
    proj   = obs + signals.get("trailing_6_period_slope", 0) * 12
    signals["proj_pct_of_target"] = round(proj / max(tgt, 1e-9) * 100, 1)

    verifier_ok  = False
    llm_result   = None
    last_err     = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt     = _build_verifier_prompt(scores, signals, composite, weights, ensemble_meta)
            t0         = time.time()
            llm_result = _call_verifier(prompt)
            elapsed    = round(time.time() - t0, 1)
            verifier_ok = True
            if verbose:
                print(f"  verifier (deepseek-r1) {elapsed}s  "
                      f"verified={llm_result.get('verified')}  "
                      f"flags={len(llm_result.get('flags', []))}")
            break
        except Exception as e:
            last_err = e
            if verbose:
                print(f"  verifier attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2)

    # ── Step 3: Apply adjustments (cap at ±ADJUST_CAP) ───────────────────────
    llm_adjustments = {}
    llm_flags       = []
    narrative       = ""

    if verifier_ok and llm_result:
        raw_adj   = llm_result.get("adjustments", {})
        llm_flags = llm_result.get("flags", [])
        # Filter out empty flag strings
        llm_flags = [f for f in llm_flags if f and f.strip()]
        narrative = llm_result.get("narrative", "")
        for dim in DIMS:
            delta = float(raw_adj.get(dim, 0))
            llm_adjustments[dim] = float(np.clip(delta, -ADJUST_CAP, ADJUST_CAP))

    # Combine hard rule flags and LLM flags (deduplicate)
    all_flags = list(dict.fromkeys(hard_flags + llm_flags))

    # Compute adjusted scores
    adjusted = {}
    for dim in DIMS:
        delta          = llm_adjustments.get(dim, 0.0)
        adjusted[dim]  = float(np.clip(scores[dim] + delta, 0.05, 0.95))

    # Recompute composite with adjusted scores
    adj_composite = float(sum(adjusted[dim] * weights.get(dim, 0.25) for dim in DIMS))
    adj_composite = float(np.clip(adj_composite, 0, 1))

    # verified = True only if verifier said so AND no hard rule flags AND no LLM flags
    verified = (
        verifier_ok
        and llm_result.get("verified", False)
        and len(all_flags) == 0
    )

    # What actually changed
    changes_made = {
        dim: round(adjusted[dim] - scores[dim], 4)
        for dim in DIMS
        if abs(adjusted[dim] - scores[dim]) > 0.001
    }

    if verbose and changes_made:
        print(f"  Adjustments applied: {changes_made}")
    if verbose and all_flags:
        for flag in all_flags:
            print(f"  FLAG: {flag}")

    return {
        # Adjusted scores
        "adjusted_attainability" : round(adjusted["attainability"], 4),
        "adjusted_relevance"     : round(adjusted["relevance"], 4),
        "adjusted_coherence"     : round(adjusted["coherence"], 4),
        "adjusted_integrity"     : round(adjusted["integrity"], 4),
        "adjusted_composite"     : round(adj_composite, 4),
        "original_composite"     : round(composite, 4),
        # Diagnostics
        "flags"                  : all_flags,
        "hard_rule_flags"        : hard_flags,
        "narrative"              : narrative,
        "verified"               : verified,
        "adjustments"            : changes_made,
        "verifier_ok"            : verifier_ok,
        "verifier_error"         : str(last_err) if not verifier_ok else None,
    }
