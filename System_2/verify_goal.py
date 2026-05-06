"""
verify_goal.py  post-composite verification module.

Verifier: Llama 3.1 70B (stable, fast, no throttle issues).

Applies hard rule checks plus an LLM verifier that returns per-dimension
adjustments capped at +- ADJUST_CAP. Clips final scores to [0.05, 0.95].
"""

import time

import numpy as np

try:
    from System_2.bedrock_client import call_model, extract_json, BedrockCallError
except ImportError:
    from bedrock_client import call_model, extract_json, BedrockCallError


ADJUST_CAP  = 0.10
MAX_RETRIES = 2

DIMS = ["attainability", "relevance", "coherence", "integrity"]


HARD_RULES = [
    (lambda s, sc: sc["integrity"] > 0.55 and s.get("needle_move_ratio", 1) < 0.25,
     "integrity too high given needle_move_ratio < 0.25"),
    (lambda s, sc: sc["integrity"] > 0.60 and s.get("allocation_efficiency_ratio", 1) < 0.20,
     "integrity too high given allocation_efficiency_ratio < 0.20"),
    (lambda s, sc: sc["coherence"] > 0.65 and s.get("alloc_drift_std", 0) > 0.10,
     "coherence too high given alloc_drift_std > 0.10"),
    (lambda s, sc: sc["coherence"] > 0.65 and s.get("weighted_goal_status_score", 1) < 0.15,
     "coherence too high given weighted_status in deep red zone"),
    (lambda s, sc: sc["attainability"] > 0.50 and
     (s.get("projected_p24", 1) / max(s.get("target", 1), 1e-9)) < 0.45,
     "attainability too high given projected value under 45% of target"),
    (lambda s, sc: sc["relevance"] > 0.65 and s.get("sibling_rank_pct", 0) > 0.80,
     "relevance too high given goal is worst funded among siblings (rank > 0.80)"),
    (lambda s, sc: sc["attainability"] > 0.45 and s.get("dependency_risk_encoded", 0) >= 0.67,
     "attainability may be optimistic given upstream dependencies are at high risk"),
    (lambda s, sc: sc["integrity"] > 0.50 and s.get("dep_avg_attain", 1.0) != -1.0
     and s.get("dep_avg_attain", 1.0) < 0.15,
     "integrity too high given upstream dependencies have very low attainability"),
    (lambda s, sc: sc["coherence"] < 0.25
     and s.get("budget_shock_exposure", 0) >= 0.33
     and s.get("shock_alloc_impact", 0) > 0.01,
     "coherence score may be understated due to active budget shock period"),
]

def apply_hard_rule_adjustments(scores, signals):
    """
    Deterministic verifier layer.
    These adjustments still work even if the LLM verifier fails.
    """
    adjustments = {dim: 0.0 for dim in DIMS}
    flags = []

    if scores["integrity"] > 0.55 and signals.get("needle_move_ratio", 1) < 0.25:
        adjustments["integrity"] -= 0.08
        flags.append("integrity reduced because needle_move_ratio is below 0.25")

    if scores["integrity"] > 0.60 and signals.get("allocation_efficiency_ratio", 1) < 0.20:
        adjustments["integrity"] -= 0.08
        flags.append("integrity reduced because allocation_efficiency_ratio is below 0.20")

    if scores["coherence"] > 0.65 and signals.get("alloc_drift_std", 0) > 0.10:
        adjustments["coherence"] -= 0.07
        flags.append("coherence reduced because alloc_drift_std is above 0.10")

    if scores["coherence"] > 0.65 and signals.get("weighted_goal_status_score", 1) < 0.15:
        adjustments["coherence"] -= 0.08
        flags.append("coherence reduced because weighted_goal_status_score is in the deep red zone")

    if scores["attainability"] > 0.50 and signals.get("proj_pct_of_target", 100) < 45:
        adjustments["attainability"] -= 0.10
        flags.append("attainability reduced because projected achievement is below 45% of target")

    if scores["relevance"] > 0.65 and signals.get("sibling_rank_pct", 0) > 0.80:
        adjustments["relevance"] -= 0.06
        flags.append("relevance reduced because goal is poorly funded among siblings")

    return adjustments, flags


def analyse_shock_effect(signals):
    """
    Shock-aware verifier layer.
    Checks whether budget or market shock should affect the goal scores.
    """
    adjustments = {dim: 0.0 for dim in DIMS}
    flags = []

    budget_shock = signals.get("budget_shock_exposure", 0) >= 0.33
    allocation_drop = signals.get("shock_alloc_impact", 0) > 0.01
    market_shock = signals.get("market_shock_vulnerable", 0) >= 0.50
    forward_market_risk = signals.get("market_shock_forward_risk", 0) >= 0.50

    if budget_shock and allocation_drop:
        adjustments["attainability"] -= 0.06
        adjustments["coherence"] -= 0.04
        flags.append("budget shock reduced allocation and may weaken attainability and coherence")

    if market_shock:
        adjustments["attainability"] -= 0.05
        flags.append("market shock vulnerability may reduce future attainability")

    if forward_market_risk:
        adjustments["coherence"] -= 0.04
        flags.append("forward market shock risk may weaken portfolio coherence")

    return adjustments, flags

def _build_verifier_prompt(scores, signals, composite, weights, ensemble_meta):
    a = scores["attainability"]
    r = scores["relevance"]
    c = scores["coherence"]
    i = scores["integrity"]

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

    return f"""Verify these four scores against the evidence. Adjust each by -0.10 to +0.10 if inconsistent. Use 0.0 if fine.

scores: attainability={a:.3f} relevance={r:.3f} coherence={c:.3f} integrity={i:.3f} composite={composite:.3f}

evidence:
  observed={observed} target={target} projected_pct={proj_pct}%
  needle_move={needle} efficiency={efficiency} quality={quality}
  drift={drift} status={wgs} sibling_rank={sibling} fitness={fitness}

rules:
  Only flag a score if it is too HIGH compared with weak evidence.
  Do not flag a score just because it is low.
  Low attainability may be correct when projected_pct is weak.
  Low coherence may be correct when drift or status evidence is weak.

  integrity>0.55 needs needle>=0.25 and efficiency>=0.20
  coherence>0.65 needs drift<=0.10 and status>=0.15
  attainability>0.50 needs projected_pct>=45
  relevance>0.65 needs sibling_rank<=0.80

  Important:
- Flags must describe contradictions only.
- Do not create flags such as "attainability is below 0.50" or "coherence is below 0.65".
- A low score is acceptable if the evidence is weak.
- Adjustments should only correct inconsistency, not punish already-low scores.
Return JSON only:
{{"adjustments":{{"attainability":0.0,"relevance":0.0,"coherence":0.0,"integrity":0.0}},"flags":[],"narrative":"one paragraph for end users","verified":true}}""".strip()


VERIFIER_SYSTEM = (
    "You are a verification engine. You return ONLY a single valid JSON object "
    "with keys: adjustments, flags, narrative, verified. Never include any other "
    "text, markdown, headers, code fences, or commentary. Your entire output "
    "starts with { and ends with }."
)


def _call_verifier(prompt):
    """
    Try primary (Anthropic) first, then fallback (Llama).
    """

    roles = ["verifier_primary", "verifier_fallback"]
    last_err = None

    for role in roles:
        for attempt in range(1, 4):
            try:
                text = call_model(
                    role,
                    prompt,
                    max_tokens=800,
                    temperature=0.1,
                    system=VERIFIER_SYSTEM,
                )
                return extract_json(text), role

            except (BedrockCallError, ValueError) as e:
                last_err = e
                msg = str(e).lower()
                is_throttle = any(tok in msg for tok in (
                    "throttl", "rate", "service unavailable", "503",
                ))

                if is_throttle and attempt < 3:
                    wait = 2 ** attempt
                    print(f"  ! {role} throttled (attempt {attempt}/3), waiting {wait}s...")
                    time.sleep(wait)
                    continue

                break

            except Exception as e:
                last_err = e
                break

        print(f"  ! {role} failed, trying next verifier...")

    raise RuntimeError(f"Verifier failed after primary and fallback: {last_err}")


def verify_scores(scores, signals, composite, weights, ensemble_meta=None, verbose=True):
    if ensemble_meta is None:
        ensemble_meta = {}

    hard_flags = []
    for rule_fn, desc in HARD_RULES:
        try:
            if rule_fn(signals, scores):
                hard_flags.append(desc)
        except Exception:
            pass

    obs = signals.get("observed_value", 0)
    tgt = signals.get("target_value_final_period", 1)
    current_period = signals.get("current_period", 18)
    periods_remaining = 24 - int(current_period)
    proj = obs + signals.get("trailing_6_period_slope", 0) * periods_remaining
    signals["proj_pct_of_target"] = round(proj / max(tgt, 1e-9) * 100, 1)
    signals["projected_p24"] = proj
    signals["target"] = tgt

    verifier_ok        = False
    llm_result         = None
    verifier_model_used = None
    last_err           = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt = _build_verifier_prompt(scores, signals, composite, weights, ensemble_meta)
            t0 = time.time()
            llm_result, verifier_model_used = _call_verifier(prompt)
            elapsed = round(time.time() - t0, 1)
            verifier_ok = True
            if verbose:
                print(
                    f"  verifier ({verifier_model_used}) {elapsed}s  "
                    f"verified={llm_result.get('verified')}  "
                    f"flags={len(llm_result.get('flags', []))}"
                )
            break
        except Exception as e:
            last_err = e
            if verbose:
                print(f"  verifier attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2)

    llm_adjustments = {}
    llm_flags       = []
    narrative       = ""

    if verifier_ok and llm_result:
        raw_adj   = llm_result.get("adjustments", {})
        llm_flags = [f for f in llm_result.get("flags", []) if f and f.strip()]
        narrative = llm_result.get("narrative", "")
        for dim in DIMS:
            delta = float(raw_adj.get(dim, 0))
            llm_adjustments[dim] = float(np.clip(delta, -ADJUST_CAP, ADJUST_CAP))

    hard_adjustments, hard_adjustment_flags = apply_hard_rule_adjustments(scores, signals)
    shock_adjustments, shock_flags = analyse_shock_effect(signals)

    all_flags = list(dict.fromkeys(
        hard_adjustment_flags + shock_flags + llm_flags
    ))

    adjusted = {}
    for dim in DIMS:
        total_delta = (
            hard_adjustments.get(dim, 0.0)
            + shock_adjustments.get(dim, 0.0)
            + llm_adjustments.get(dim, 0.0)
        )

        total_delta = float(np.clip(total_delta, -ADJUST_CAP, ADJUST_CAP))
        adjusted[dim] = float(np.clip(scores[dim] + total_delta, 0.05, 0.95))

    adj_composite = float(sum(adjusted[dim] * weights.get(dim, 0.25) for dim in DIMS))
    adj_composite = float(np.clip(adj_composite, 0, 1))

    verified = verifier_ok and llm_result.get("verified", False)

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
        if changes_made:
            rule_summary = "; ".join(
            [f"{dim} adjusted by {delta:+.2f}" for dim, delta in changes_made.items()]
        )

    if changes_made:
        rule_summary = "; ".join(
            [f"{dim} adjusted by {delta:+.2f}" for dim, delta in changes_made.items()]
        )

        if narrative:
            narrative = narrative + " Final verifier adjustments applied: " + rule_summary + "."
        else:
            narrative = "Final verifier adjustments applied: " + rule_summary + "."

    return {
        "adjusted_attainability": round(adjusted["attainability"], 4),
        "adjusted_relevance"    : round(adjusted["relevance"], 4),
        "adjusted_coherence"    : round(adjusted["coherence"], 4),
        "adjusted_integrity"    : round(adjusted["integrity"], 4),
        "adjusted_composite"    : round(adj_composite, 4),
        "original_composite"    : round(composite, 4),
        "flags"                 : all_flags,
        "hard_rule_flags"       : hard_adjustment_flags,
        "narrative"             : narrative,
        "verified"              : verified,
        "adjustments"           : changes_made,
        "verifier_ok"           : verifier_ok,
        "verifier_model"        : verifier_model_used,
        "verifier_error"        : str(last_err) if not verifier_ok else None,
    }


if __name__ == "__main__":
    import json as _json
    scores = {"attainability": 0.14, "relevance": 0.58, "coherence": 0.42, "integrity": 0.70}
    signals = {
        "needle_move_ratio": 0.12, "alloc_drift_std": 0.14,
        "allocation_efficiency_ratio": 0.18, "weighted_goal_status_score": 0.20,
        "sibling_rank_pct": 0.85, "observed_value": 20,
        "target_value_final_period": 100, "trailing_6_period_slope": 2,
        "allocation_fitness_score": 0.40, "current_period": 18,
    }
    weights = {"coherence": 0.35, "attainability": 0.25, "relevance": 0.20, "integrity": 0.20}
    result = verify_scores(scores, signals, 0.47, weights, ensemble_meta={}, verbose=True)
    print(_json.dumps(result, indent=2))