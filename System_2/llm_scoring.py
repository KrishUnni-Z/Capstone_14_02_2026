import os
import json
import re
import time
import argparse
import urllib.request
import urllib.error

import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# Architecture: 2 scorers (llama3 + gemma3) + qwen3 as verifier (in verify_goal.py)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MAX_SAMPLES = 35
MAX_RETRIES = 3
TIMEOUT_SEC = 900

DIMS = ["relevance", "coherence", "integrity", "attainability"]

MODELS = [
    {"id": "llama3:latest", "label": "llama3"},
    {"id": "gemma3:4b", "label": "gemma3"},
]

ALL_MODELS = [
    {"id": "llama3:latest",      "label": "llama3"},
    {"id": "gemma3:4b",          "label": "gemma3"},
    {"id": "nemotron-3-nano:4b", "label": "nemotron"},
    {"id": "qwen3:4b",           "label": "qwen3"},
    {"id": "deepseek-r1:8b",     "label": "deepseek"},
]


def check_ollama():
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=5)
        print("  Ollama reachable  OK")
        return True
    except Exception as e:
        print(f"  ERROR: Ollama not reachable — {e}")
        return False


def list_ollama_models():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def available_models(requested_models, installed_names):
    installed = set(installed_names)
    ok = []
    missing = []

    for m in requested_models:
        if m["id"] in installed:
            ok.append(m)
        else:
            missing.append(m["id"])

    return ok, missing


def clean_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fallback_reason_for_dim(dim):
    fallback = {
        "attainability": "The score reflects the likelihood of reaching the target by period 24.",
        "relevance": "The score reflects how well allocation appears aligned with the goal.",
        "coherence": "The score reflects consistency across levels, periods, and decisions.",
        "integrity": "The score reflects whether outputs and metric movement appear to match the resources used.",
    }
    return fallback[dim]


def _signals(row, rule_row):
    """Extract and pre-compute all signal values used across prompts."""
    s = {}
    s["observed"]   = clean_float(row.get("observed_value", 0))
    s["target"]     = clean_float(row.get("target_value_final_period", 0))
    s["slope"]      = clean_float(row.get("trailing_6_period_slope", 0))
    s["variance"]   = clean_float(row.get("variance_from_target", 0))
    s["projected"]  = s["observed"] + s["slope"] * 12
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
    s["anchor"]     = float(np.clip(
        (s["projected"] / s["target"]) if s["target"] > 0 else 0.5, 0.05, 0.95))
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
    # Trajectory context from previous checkpoint
    s["current_period"]   = int(clean_float(row.get("current_period", 12)))
    s["prev_composite"]   = clean_float(row.get("prev_composite", -1.0))
    s["composite_delta"]  = clean_float(row.get("composite_delta", 0.0))
    s["prev_period"]      = int(clean_float(row.get("prev_period", 0)))
    s["shock_since_prev"] = clean_float(row.get("shock_since_prev", 0.0))
    has_prev = s["prev_composite"] > 0
    s["traj_label"] = (
        f"no prior checkpoint — period {s['current_period']} is first assessment"
        if not has_prev else
        f"period {s['prev_period']} composite was {s['prev_composite']:.2f}, "
        f"{'improved' if s['composite_delta'] >= 0 else 'degraded'} by {abs(s['composite_delta']):.3f}"
        + (f" (shock occurred between periods {s['prev_period']} and {s['current_period']})"
           if s['shock_since_prev'] > 0 else "")
    )
    # Shock signals
    s["budget_shock_exp"]  = clean_float(row.get("budget_shock_exposure", 0.0))
    s["shock_impact"]      = clean_float(row.get("shock_alloc_impact", 0.0))
    s["recovery_est"]      = clean_float(row.get("recovery_period_estimate", 0.0))
    s["market_vuln"]       = clean_float(row.get("market_shock_vulnerable", 0.0))
    s["market_risk"]       = clean_float(row.get("market_shock_forward_risk", 0.0))
    s["recovery_window"]   = clean_float(row.get("recovery_window_remaining", 7.0))
    s["in_shock"]          = s["budget_shock_exp"] >= 0.33
    s["shock_label"]       = (
        "currently in budget shock (20% cut periods 10-12)"
        if s["in_shock"] else "no active budget shock"
    )
    s["market_label"]      = (
        f"metric is market-shock sensitive (forward risk {s['market_risk']:.2f})"
        if s["market_vuln"] > 0 else "metric not market-shock sensitive"
    )
    # Shock signals
    s["budget_shock_exp"]  = clean_float(row.get("budget_shock_exposure", 0.0))
    s["shock_impact"]      = clean_float(row.get("shock_alloc_impact", 0.0))
    s["recovery_est"]      = clean_float(row.get("recovery_period_estimate", 0.0))
    s["market_vuln"]       = clean_float(row.get("market_shock_vulnerable", 0.0))
    s["in_shock"]          = s["budget_shock_exp"] >= 0.33
    s["shock_label"]       = (
        "currently in budget shock period (20% budget cut)" if s["in_shock"]
        else "no active budget shock"
    )
    s["market_label"]      = (
        "metric type is market-shock sensitive" if s["market_vuln"] > 0
        else "metric type not market-shock sensitive"
    )

    risk_map = {
    0.0: "No dependency risk",
    0.33: "Low dependency risk",
    0.67: "Medium dependency risk",
    1.0: "High dependency risk"
}

    s["dep_risk_label"] = risk_map.get(
    clean_float(row.get("dependency_risk_encoded", 0.0)),
    "No dependency risk"
)

    s["dep_attain"] = clean_float(row.get("dep_avg_attain", -1.0))
    s["n_dependents"] = clean_float(row.get("n_dependents", 0))
    s["n_deps"] = clean_float(row.get("n_dependencies", 0))
    s["dep_risk"] = clean_float(row.get("dependency_risk_encoded", 0.0))
    
    return s


# ── LLAMA3 — Narrative analyst brief ──────────────────────────────────────────
# llama3 handles long context well and reasons naturally in prose.
# We brief it like a management consultant reviewing a goal file.
# Scores anchored to computed signals. Slightly higher temperature for richer reasons.
def prompt_llama3(s):
    return f"""You are a strategic analyst reviewing an organisational goal at the halfway point (period 12 of 24).

GOAL SNAPSHOT
  Current value   : {s["observed"]:.3f}  (target by period 24: {s["target"]:.3f})
  Progress so far : {s["pct_of_target"]}% of target reached
  Trend           : {s["slope"]:+.4f} per period  (projected at period 24: {s["projected"]:.3f}, which is {s["proj_pct"]}% of target)
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
  Efficiency ratio        : {s["efficiency"]:.3f}  (value delivered per unit of resource)
  Needle move ratio       : {s["needle"]:.3f}  (observed vs expected — 1.0 means on track)
  Weighted status score   : {s["wgs"]:.3f}  (0.05=red zone, 0.95=green zone)

SHOCK CONTEXT
  Budget shock status : {s["shock_label"]}
  Allocation impact   : {s["shock_impact"]:.4f} drop from pre-shock level
  Recovery estimate   : {s["recovery_est"]:.1f} periods to return to pre-shock allocation
  Market exposure     : {s["market_label"]}

SHOCK CONTEXT
  Budget shock    : {s["shock_label"]}
  Alloc impact    : {s["shock_impact"]:.4f} drop from pre-shock level
  Recovery est    : {s["recovery_est"]:.1f} periods to return to pre-shock allocation
  Market exposure : {s["market_label"]}
  Forward risk    : {s["market_risk"]:.2f}  (0=goal survives shock, 1=cannot recover)
  Recovery runway : {s["recovery_window"]:.0f} periods post-shock to hit target

TRAJECTORY CONTEXT (prior checkpoint)
  {s["traj_label"]}

COMPUTED ANCHOR SCORES (use these as your primary reference — adjust only if evidence clearly warrants it)
  Attainability anchor : {s["anchor"]:.2f}  (based on projected trajectory)
  Relevance rule score : {s["r_rule"]:.2f}  (based on allocation band and sibling rank)
  Coherence rule score : {s["c_rule"]:.2f}  (based on allocation drift and status consistency)
  Integrity rule score : {s["i_rule"]:.2f}  (based on efficiency, needle move, quality)

SCORING TASK
Evaluate this goal on four dimensions. For each, reason from the evidence above and assign a score between 0.05 and 0.95.

  RELEVANCE     — Is the allocation level justified against stated goals?
  COHERENCE     — Are decisions consistent across the hierarchy, time, and goals?
  INTEGRITY     — Did the allocation produce the expected outputs and move the metric?
  ATTAINABILITY — Will this goal realistically reach its target by period 24?

Return exactly this JSON and nothing else:
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


# ── GEMMA3 — Rubric-based scoring ─────────────────────────────────────────────
# gemma3 is clean, efficient, and follows explicit structure well.
# Give it a scoring rubric per dimension so it applies criteria rather than guesses.
def prompt_gemma3(s):
    return f"""Score this goal on four dimensions. Use the signal thresholds in the rubric to assign each score independently.

EVIDENCE (period 12 of 24):
  observed={s["observed"]:.3f} target={s["target"]:.3f} progress={s["pct_of_target"]}%
  slope={s["slope"]:+.4f}/period projected={s["projected"]:.3f} ({s["proj_pct"]}% of target)
  alloc_pct={s["alloc_pct"]:.3f} sibling_rank={s["sibling_r"]:.2f} fitness={s["fitness_label"]}
  efficiency={s["efficiency"]:.3f} needle_move={s["needle"]:.3f} quality={s["quality"]:.2f}
  drift={s["drift"]:.4f} status={s["wgs"]:.3f} ttg={s["ttg"]:.0f}
  shock={s["shock_label"]}  shock_impact={s["shock_impact"]:.4f}  recovery_est={s["recovery_est"]:.1f}

RUBRIC — score each dimension using the thresholds below:

ATTAINABILITY (use projected_pct and slope):
  projected >= 90% of target                → 0.80-0.95
  projected 65-89%                          → 0.55-0.79
  projected 40-64%                          → 0.35-0.54
  projected 20-39%                          → 0.15-0.34
  projected < 20%                           → 0.05-0.14

RELEVANCE (use sibling_rank and fitness):
  sibling_rank < 0.3 and in optimal band    → 0.80-0.95
  sibling_rank < 0.5 or mostly in band      → 0.55-0.79
  sibling_rank 0.5-0.7 or outside band      → 0.35-0.54
  sibling_rank > 0.7 and outside band       → 0.15-0.34
  sibling_rank > 0.9                        → 0.05-0.14

COHERENCE (use drift and status):
  drift < 0.03 and status > 0.5             → 0.80-0.95
  drift < 0.06 or status > 0.3             → 0.55-0.79
  drift 0.06-0.10 or status 0.15-0.3       → 0.35-0.54
  drift > 0.10 or status < 0.15            → 0.15-0.34
  drift > 0.15 and status < 0.10           → 0.05-0.14

INTEGRITY (use efficiency and needle_move):
  efficiency > 0.5 and needle_move > 0.7   → 0.80-0.95
  efficiency > 0.3 and needle_move > 0.5   → 0.55-0.79
  efficiency > 0.2 or needle_move > 0.3    → 0.35-0.54
  efficiency < 0.2 and needle_move < 0.3   → 0.15-0.34
  efficiency < 0.1 and needle_move < 0.2   → 0.05-0.14

Return only this JSON:
{{
  "relevance": 0.XX, "coherence": 0.XX, "integrity": 0.XX, "attainability": 0.XX,
  "relevance_reason": "One sentence citing the specific signal value that drove this score.",
  "coherence_reason": "One sentence citing the specific signal value that drove this score.",
  "integrity_reason": "One sentence citing the specific signal value that drove this score.",
  "attainability_reason": "One sentence citing the specific signal value that drove this score."
}}""".strip()


# ── QWEN3 — Computational reasoning ───────────────────────────────────────────
# qwen3 is a reasoning model — it thinks step by step.
# Give it raw signals and ask it to derive each score computationally.
# format:json is enforced. Keep prompt compact and mathematical.
# No example scores in the JSON template — it should compute, not copy.
def prompt_qwen3(s):
    return f"""You are a scoring engine. Compute scores for four dimensions from the signals below.

Signals:
  observed={s["observed"]:.4f}  target={s["target"]:.4f}  slope={s["slope"]:+.4f}
  projected_p24={s["projected"]:.4f}  gap={s["variance"]:.4f}
  alloc_pct={s["alloc_pct"]:.4f}  sibling_rank_pct={s["sibling_r"]:.4f}
  optimal_fitness={s["fitness"]:.1f}  l3_share_of_l2={s["l3_l2"]:.4f}
  alloc_drift_std={s["drift"]:.4f}  weighted_status={s["wgs"]:.4f}
  n_upstream_deps={int(s["n_deps"])}  dep_risk_encoded={s["dep_risk"]:.2f}  dep_avg_attain={s["dep_attain"]:.2f}
  efficiency={s["efficiency"]:.4f}  needle_move_ratio={s["needle"]:.4f}
  quality={s["quality"]:.4f}  time_to_green={s["ttg"]:.1f}

Pre-computed anchors (validate and refine, do not copy blindly):
  attainability_anchor={s["anchor"]:.4f}
  relevance_rule={s["r_rule"]:.4f}
  coherence_rule={s["c_rule"]:.4f}
  integrity_rule={s["i_rule"]:.4f}

Compute each score on 0.05-0.95. Definitions:
  relevance     = how well allocation level is justified (band position, sibling rank, fitness)
  coherence     = consistency across hierarchy and time (drift, status stability, hierarchy share)
  integrity     = degree to which spend produced expected outcomes (efficiency, needle, quality)
  attainability = probability of reaching target by period 24 (trajectory, gap, trend)

Output JSON only:
{{
  "relevance": <float>,
  "coherence": <float>,
  "integrity": <float>,
  "attainability": <float>,
  "relevance_reason": "<one sentence>",
  "coherence_reason": "<one sentence>",
  "integrity_reason": "<one sentence>",
  "attainability_reason": "<one sentence>"
}}""".strip()


# ── NEMOTRON — Minimal key=value (test only) ───────────────────────────────────
# nemotron-3-nano collapses to buckets with long prompts and scale labels.
# Minimal format, no anchors, force precise decimals via output format.
def prompt_nemotron(s):
    return f"""Evaluate one organisational goal on four dimensions.
Use any decimal between 0.05 and 0.95. Do not round to 0.1 or 0.5.

Evidence:
current={s["observed"]:.3f}  target={s["target"]:.3f}  gap={s["variance"]:.3f}
slope={s["slope"]:+.4f}  projected={s["projected"]:.3f}
quality={s["quality"]:.2f}  quantity={s["quantity"]:.1f}
needle_move={s["needle"]:.3f}  alloc_pct={s["alloc_pct"]:.3f}
efficiency={s["efficiency"]:.3f}  sibling_rank={s["sibling_r"]:.2f}
l3_share_of_l2={s["l3_l2"]:.2f}  drift={s["drift"]:.4f}
time_to_green={s["ttg"]:.0f}  status_score={s["wgs"]:.3f}

relevance   means: is allocation justified vs goals?
coherence   means: are decisions consistent across levels and time?
integrity   means: did outcomes match the allocation?
attainability means: will this goal reach target by period 24?

Return EXACTLY this format, no other text:
Values: relevance 0.XX, coherence 0.XX, integrity 0.XX, attainability 0.XX
Reasons:
attainability: one sentence
relevance: one sentence
coherence: one sentence
integrity: one sentence""".strip()


# ── DEEPSEEK — Chain of thought (test only) ────────────────────────────────────
# deepseek-r1:8b is a heavy reasoning model with thinking blocks.
# Ask it to reason explicitly through each dimension before scoring.
# format:json enforced. Give extra token budget for thinking.
def prompt_deepseek(s):
    return f"""Score an organisational goal on four dimensions. Think step by step for each.

Data (period 12 of 24):
  observed={s["observed"]:.4f}  target={s["target"]:.4f}
  slope={s["slope"]:+.4f}/period  projected_at_p24={s["projected"]:.4f}
  gap_to_target={s["variance"]:.4f}
  alloc_of_parent={s["alloc_pct"]:.4f}  sibling_rank_pct={s["sibling_r"]:.4f}  fitness={s["fitness"]:.1f}
  l3_share_of_l2={s["l3_l2"]:.4f}  alloc_drift_std={s["drift"]:.4f}
  efficiency={s["efficiency"]:.4f}  needle_move_ratio={s["needle"]:.4f}
  quality={s["quality"]:.4f}  weighted_status={s["wgs"]:.4f}
  time_to_green={s["ttg"]:.1f}
  budget_shock_exposure={s["budget_shock_exp"]:.2f}  shock_alloc_impact={s["shock_impact"]:.4f}
  recovery_est={s["recovery_est"]:.1f}  market_vulnerable={s["market_vuln"]:.0f}

Anchors: attain={s["anchor"]:.4f}  relevance={s["r_rule"]:.4f}  coherence={s["c_rule"]:.4f}  integrity={s["i_rule"]:.4f}

For each dimension: reason from the evidence, then give a score 0.05-0.95.

  RELEVANCE   — Is the allocation level justified by priority and band position?
  COHERENCE   — Are allocations consistent across the hierarchy, time, and status bands?
  INTEGRITY   — Does the spend match delivered outputs and metric movement?
  ATTAINABILITY — Will this goal reach its target by period 24?

Output JSON only:
{{
  "relevance": <float 0.05-0.95>,
  "coherence": <float 0.05-0.95>,
  "integrity": <float 0.05-0.95>,
  "attainability": <float 0.05-0.95>,
  "relevance_reason": "<specific one-sentence explanation>",
  "coherence_reason": "<specific one-sentence explanation>",
  "integrity_reason": "<specific one-sentence explanation>",
  "attainability_reason": "<specific one-sentence explanation>"
}}""".strip()


# ── Dispatcher ─────────────────────────────────────────────────────────────────
PROMPT_FN = {
    "llama3"   : prompt_llama3,
    "gemma3"   : prompt_gemma3,
    "qwen3"    : prompt_qwen3,
    "nemotron" : prompt_nemotron,
    "deepseek" : prompt_deepseek,
}

def build_prompt(row, rule_row, model_label):
    s  = _signals(row, rule_row)
    fn = PROMPT_FN.get(model_label, prompt_llama3)
    return fn(s)


# Per-model generation options
MODEL_OPTIONS = {
    "llama3"  : {"temperature": 0.2, "num_predict": 450, "top_p": 0.9, "repeat_penalty": 1.1},
    "gemma3"  : {"temperature": 0.1, "num_predict": 500, "top_p": 0.9, "repeat_penalty": 1.1},
    "qwen3"   : {"temperature": 0.1, "num_predict": 350, "top_p": 0.9, "repeat_penalty": 1.1},
    "nemotron": {"temperature": 0.1, "num_predict": 260, "top_p": 0.9, "repeat_penalty": 1.1},
    "deepseek": {"temperature": 0.1, "num_predict": 600, "top_p": 0.9, "repeat_penalty": 1.1},
}

# Models that need JSON format enforced
JSON_FORMAT_MODELS = {"qwen3", "nemotron", "deepseek"}

def call_ollama(model_id, prompt):
    label   = next((m["label"] for m in ALL_MODELS if m["id"] == model_id), "llama3")
    options = MODEL_OPTIONS.get(label, MODEL_OPTIONS["llama3"])
    payload = {
        "model"  : model_id,
        "prompt" : prompt,
        "stream" : False,
        "options": options,
    }
    if label in JSON_FORMAT_MODELS:
        payload["format"] = "json"

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        raw = resp.read()
        data = json.loads(raw)
        return data.get("response", "").strip()


def parse_nemotron_values_block(text):
    m = re.search(
        r"Values:\s*relevance\s*([0-9.]+)\s*,\s*coherence\s*([0-9.]+)\s*,\s*integrity\s*([0-9.]+)\s*,\s*attainability\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None

    result = {
        "relevance": float(np.clip(float(m.group(1)), 0.05, 0.95)),
        "coherence": float(np.clip(float(m.group(2)), 0.05, 0.95)),
        "integrity": float(np.clip(float(m.group(3)), 0.05, 0.95)),
        "attainability": float(np.clip(float(m.group(4)), 0.05, 0.95)),
    }

    for dim in DIMS:
        rm = re.search(rf"{dim}\s*:\s*(.+)", text, flags=re.IGNORECASE)
        result[f"{dim}_reason"] = rm.group(1).strip() if rm else fallback_reason_for_dim(dim)

    return result


def extract_score_from_text(text, key):
    patterns = [
        rf'"{key}"\s*:\s*([0-9.]+)',
        rf"{key}\s*[:=]\s*([0-9.]+)",
        rf"{key.capitalize()}\s*[:=]\s*([0-9.]+)",
        rf"\b{key}\b\s+([0-9.]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def fallback_parse_non_json(text):
    nemotron_style = parse_nemotron_values_block(text)
    if nemotron_style is not None:
        return nemotron_style

    result = {}
    for d in DIMS:
        val = extract_score_from_text(text, d)
        if val is None:
            return None
        result[d] = float(np.clip(val, 0.05, 0.95))
        result[f"{d}_reason"] = fallback_reason_for_dim(d)

    return result


def extract_json(text):
    if not text:
        raise ValueError("Empty response")

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_text = match.group(0)
        try:
            parsed = json.loads(json_text)
            result = {}
            for d in DIMS:
                if d not in parsed:
                    raise ValueError(f"Missing field: {d}")
                result[d] = float(np.clip(float(parsed[d]), 0.05, 0.95))
                result[f"{d}_reason"] = str(parsed.get(f"{d}_reason", "")).strip()
            return result
        except Exception:
            pass

    fallback = fallback_parse_non_json(text)
    if fallback is not None:
        return fallback

    raise ValueError(f"No JSON object found. Raw response: {text[:250]}")


def fix_placeholder_reason(reason, fallback_text):
    if not reason:
        return fallback_text
    reason_clean = reason.strip().lower()
    if reason_clean in {"one short sentence", "short sentence", "placeholder"}:
        return fallback_text
    return reason


def predict_model(model, prompt):
    label = model["label"]
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            raw = call_ollama(model["id"], prompt)
            parsed = extract_json(raw)

            parsed["attainability_reason"] = fix_placeholder_reason(
                parsed.get("attainability_reason", ""),
                fallback_reason_for_dim("attainability")
            )
            parsed["relevance_reason"] = fix_placeholder_reason(
                parsed.get("relevance_reason", ""),
                fallback_reason_for_dim("relevance")
            )
            parsed["coherence_reason"] = fix_placeholder_reason(
                parsed.get("coherence_reason", ""),
                fallback_reason_for_dim("coherence")
            )
            parsed["integrity_reason"] = fix_placeholder_reason(
                parsed.get("integrity_reason", ""),
                fallback_reason_for_dim("integrity")
            )

            parsed["success"] = True
            parsed["elapsed_s"] = round(time.time() - t0, 1)
            return parsed
        except Exception as e:
            last_err = e
            print(f"      [{label}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2)

    return (
        {d: None for d in DIMS}
        | {f"{d}_reason": "" for d in DIMS}
        | {"success": False, "elapsed_s": 0, "error": str(last_err)}
    )


def print_model_output(label, res):
    if res["success"]:
        print(
            f"{label:<10}"
            f"A={res['attainability']:.2f}  "
            f"R={res['relevance']:.2f}  "
            f"C={res['coherence']:.2f}  "
            f"I={res['integrity']:.2f}  "
            f"({res['elapsed_s']:.1f}s)"
        )
        print(f"      attainability: {res.get('attainability_reason', '')}")
        print(f"      relevance    : {res.get('relevance_reason', '')}")
        print(f"      coherence    : {res.get('coherence_reason', '')}")
        print(f"      integrity    : {res.get('integrity_reason', '')}")
    else:
        print(f"{label:<10}FAILED ({res.get('elapsed_s', 0):.1f}s)")
        print(f"      error: {res.get('error', '')}")


def predict_goal(df_raw, rule_scores, goal_idx, models):
    row = df_raw.iloc[goal_idx]
    rule_row = rule_scores.iloc[goal_idx]

    record = {"goal_idx": goal_idx}
    n_ok = 0
    model_results = []

    for m in models:
        prompt = build_prompt(row, rule_row, m["label"])
        res = predict_model(m, prompt)

        print_model_output(m["label"], res)

        if res["success"]:
            n_ok += 1
            model_results.append(res)

        for d in DIMS:
            record[f"{m['label']}_{d}"] = res[d]
            record[f"{m['label']}_{d}_reason"] = res.get(f"{d}_reason", "")

        record[f"{m['label']}_success"] = res["success"]
        record[f"{m['label']}_elapsed_s"] = res.get("elapsed_s", 0)
        record[f"{m['label']}_error"] = res.get("error", "")

    if model_results:
        ens_a = round(float(np.mean([r["attainability"] for r in model_results])), 4)
        ens_r = round(float(np.mean([r["relevance"]     for r in model_results])), 4)
        ens_c = round(float(np.mean([r["coherence"]     for r in model_results])), 4)
        ens_i = round(float(np.mean([r["integrity"]     for r in model_results])), 4)
        print(f"  {'ensemble':<10} A={ens_a:.2f}  R={ens_r:.2f}  C={ens_c:.2f}  I={ens_i:.2f}")

    record["n_models_ok"] = n_ok
    record["success"] = n_ok > 0
    return record


def merge_extra_columns(df_raw, period_12):
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
    ]

    for col in extra_cols:
        if col not in df_raw.columns and col in period_12.columns:
            df_raw[col] = period_12[col].values

    return df_raw


SNAPSHOT_PERIODS = [6, 12, 18, 24]
SHOCK_BETWEEN = {
    (6,  12): 1.0,   # budget shock in periods 10-12
    (12, 18): 1.0,   # market shock in period 14
    (18, 24): 0.0,   # recovery phase
    (0,   6): 0.0,   # pre-shock
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",   action="store_true", help="Run goal 0 only at period 12")
    parser.add_argument("--goal",   type=int, default=None, help="Run a single goal index")
    parser.add_argument("--period", type=int, default=None, help="Run specific period only (6/12/18/24)")
    args = parser.parse_args()

    print("=" * 70)
    print("DECIDR COHERENCE ENGINE — LLM Predictions")
    print("=" * 70)

    if not check_ollama():
        raise SystemExit(1)

    installed = list_ollama_models()
    print("\nInstalled Ollama models:")
    for name in installed:
        print(f"  - {name}")

    models_to_use, missing = available_models(MODELS, installed)

    if missing:
        print("\nSkipping missing models:")
        for m in missing:
            print(f"  - {m}")

    if not models_to_use:
        print("\nERROR: No matching installed models found.")
        raise SystemExit(1)

    print("\nModels to use:")
    for m in models_to_use:
        print(f"  - {m['id']}")

    import os

    if args.test:
        # Test mode — run goal 0 at period 12 only
        df_raw      = pd.read_csv("features_raw_poc.csv")
        rule_scores = pd.read_csv("rule_scores_poc.csv")
        period_12   = pd.read_csv("period_12_poc.csv").reset_index(drop=True)
        df_raw      = merge_extra_columns(df_raw, period_12)
        df_raw["current_period"] = 12
        df_raw["prev_composite"] = -1.0
        df_raw["composite_delta"] = 0.0
        df_raw["prev_period"] = 0
        df_raw["shock_since_prev"] = 0.0
        print("\nTEST MODE — goal 0 at period 12 only")
        pred = predict_goal(df_raw, rule_scores, 0, models_to_use)
        print(f"=> {pred['n_models_ok']}/{len(models_to_use)} models OK")
        raise SystemExit(0)

    if args.goal is not None:
        df_raw      = pd.read_csv("features_raw_poc.csv")
        rule_scores = pd.read_csv("rule_scores_poc.csv")
        period_12   = pd.read_csv("period_12_poc.csv").reset_index(drop=True)
        df_raw      = merge_extra_columns(df_raw, period_12)
        df_raw["current_period"] = 12
        df_raw["prev_composite"] = -1.0
        df_raw["composite_delta"] = 0.0
        df_raw["prev_period"] = 0
        df_raw["shock_since_prev"] = 0.0
        print(f"\nSINGLE GOAL MODE — goal {args.goal} at period 12")
        pred = predict_goal(df_raw, rule_scores, args.goal, models_to_use)
        print(f"=> {pred['n_models_ok']}/{len(models_to_use)} models OK")
        raise SystemExit(0)

    # ── FULL RUN — 4 snapshot periods ────────────────────────────────────────
    periods_to_run = [args.period] if args.period else SNAPSHOT_PERIODS
    all_preds      = []
    prev_scores    = {}   # goal_id -> composite score from previous period

    t_total = time.time()
    for sp in periods_to_run:
        raw_file  = f"features_raw_p{sp}.csv"
        rule_file = f"rule_scores_p{sp}.csv" if sp != 12 else "rule_scores_poc.csv"
        out_file  = f"llm_predictions_p{sp}.csv"

        if not os.path.exists(raw_file):
            print(f"\nSkipping period {sp} — {raw_file} not found")
            continue

        print(f"\n{'='*70}")
        print(f"PERIOD {sp} — LLM SCORING ({len(SNAPSHOT_PERIODS)} snapshots total)")
        print(f"{'='*70}")

        df_raw      = pd.read_csv(raw_file)
        rule_scores = pd.read_csv(rule_file)
        period_ref  = pd.read_csv("period_12_poc.csv").reset_index(drop=True)
        df_raw      = merge_extra_columns(df_raw, period_ref)

        # Add trajectory context columns
        df_raw["current_period"] = sp
        prev_p = max([p for p in SNAPSHOT_PERIODS if p < sp], default=0)
        df_raw["prev_period"]     = prev_p
        df_raw["shock_since_prev"]= SHOCK_BETWEEN.get((prev_p, sp), 0.0)

        def get_prev(gid):
            return prev_scores.get((gid, prev_p), -1.0)

        df_raw["prev_composite"]  = df_raw["goal_id"].apply(get_prev) if "goal_id" in df_raw.columns else -1.0
        df_raw["composite_delta"] = df_raw.apply(
            lambda r: r["prev_composite"] - get_prev(r.get("goal_id", -1))
            if get_prev(r.get("goal_id",-1)) > 0 else 0.0, axis=1
        ) if "goal_id" in df_raw.columns else 0.0

        goals_to_run = list(range(min(len(df_raw), MAX_SAMPLES)))
        preds        = []
        t_start      = time.time()
        n_total      = len(goals_to_run)

        for i, goal_idx in enumerate(goals_to_run):
            elapsed = time.time() - t_start
            if i > 0:
                eta = (elapsed / i) * (n_total - i) / 60
                print(f"\n[p{sp}][{i+1}/{n_total}] Goal {goal_idx}  "
                      f"(elapsed: {elapsed/60:.1f}m  ETA: {eta:.1f}m)")
            else:
                print(f"\n[p{sp}][{i+1}/{n_total}] Goal {goal_idx}")

            pred = predict_goal(df_raw, rule_scores, goal_idx, models_to_use)
            pred["period_id"] = sp
            preds.append(pred)
            print(f"  => {pred['n_models_ok']}/{len(models_to_use)} models OK")

        period_min = (time.time() - t_start) / 60
        df_period  = pd.DataFrame(preds)
        df_period.to_csv(out_file, index=False)
        all_preds.append(df_period)
        print(f"\n✓ Period {sp} done in {period_min:.1f} min — saved {out_file}")

        # Store composite for next period context
        # Use ensemble mean as proxy composite until meta_learner runs
        for _, row in df_period.iterrows():
            gid    = int(row.get("goal_idx", -1))
            scores = [row[f"{m['label']}_attainability"] for m in models_to_use
                      if f"{m['label']}_attainability" in row.index
                      and pd.notna(row[f"{m['label']}_attainability"])]
            if scores:
                prev_scores[(gid, sp)] = float(np.mean(scores))

    # Combine all periods into one file
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        combined.to_csv("llm_predictions_poc.csv", index=False)
        total_min = (time.time() - t_total) / 60
        print(f"\n{'='*70}")
        print(f"ALL PERIODS COMPLETE in {total_min:.1f} minutes")
        print(f"Total predictions: {len(combined)} rows saved to llm_predictions_poc.csv")
        print("=" * 70)