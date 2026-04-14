"""
score_group.py — Multi-goal group scoring
Called by System 3 when scope is l2_bucket or l1_bucket.

Takes a list of goals from the same parent bucket, scores each individually
via score_goal(), then runs one group verifier call to catch cross-goal
inconsistencies that per-goal scoring misses.

Usage:
    from score_group import score_group

    result = score_group(
        goal_rows      = [row1, row2, row3],   # list of dicts/Series
        rule_rows      = [rule1, rule2, rule3],
        bucket_context = {
            "l2_name"    : "Paid Acquisition",
            "l1_name"    : "Marketing",
            "l2_alloc_pct": 0.14,
            "l1_alloc_pct": 0.40,
            "n_siblings" : 3,
        },
        verbose = True,
    )
"""

import json
import re
import time
import urllib.request
import numpy as np
import pandas as pd

OLLAMA_URL     = "http://localhost:11434/api/generate"
TIMEOUT_SEC    = 300
DIMS           = ["attainability", "relevance", "coherence", "integrity"]
WEIGHTS        = {"coherence": 0.35, "attainability": 0.25,
                  "relevance": 0.20, "integrity": 0.20}

# Import score_goal from same folder
try:
    from score_goal import score_goal, load_artefacts
except ImportError:
    raise ImportError("score_group.py must be in the same folder as score_goal.py")


# ── Group verifier prompt ─────────────────────────────────────────────────────
def _build_group_prompt(goal_scores, bucket_context, individual_flags):
    bucket  = bucket_context.get("l2_name", "Unknown bucket")
    l1      = bucket_context.get("l1_name", "")
    l2_pct  = bucket_context.get("l2_alloc_pct", 0)
    l1_pct  = bucket_context.get("l1_alloc_pct", 0)

    lines = []
    for gs in goal_scores:
        gid   = gs.get("goal_id", "?")
        name  = gs.get("goal_name", f"goal_{gid}")
        a     = gs.get("attainability", 0)
        r     = gs.get("relevance", 0)
        c     = gs.get("coherence", 0)
        i     = gs.get("integrity", 0)
        comp  = gs.get("overall", 0)
        alloc = gs.get("alloc_pct_of_parent", "?")
        lines.append(
            f"  ID={gid}  {name:<35}  "
            f"A={a:.2f} R={r:.2f} C={c:.2f} I={i:.2f}  "
            f"composite={comp:.2f}  alloc={alloc}"
        )

    goal_block = "\n".join(lines)

    existing_flags = []
    for gs in goal_scores:
        gflags = individual_flags.get(gs.get("goal_id"), [])
        for f in gflags:
            existing_flags.append(f"[goal {gs.get('goal_id')}] {f}")
    flags_block = "\n".join(existing_flags) if existing_flags else "none"

    return f"""Review these {len(goal_scores)} goals from the same parent bucket for cross-goal coherence.

BUCKET: {bucket} (part of {l1})
  L2 allocation: {l2_pct:.1%} of total  |  L1 allocation: {l1_pct:.1%} of total

INDIVIDUAL SCORES:
{goal_block}

EXISTING FLAGS (from individual verification):
{flags_block}

CROSS-GOAL CHECKS:
1. Are allocation shares proportional to relative performance? (high alloc + low scores = problem)
2. Is coherence spread across siblings defensible, or are siblings too inconsistent?
3. Does the best-performing goal justify its allocation share vs weakest?
4. Any goals dragging down the bucket that need flagging?

Return JSON only:
{{"group_flags": ["<cross-goal issue if any>"], "narrative": "<one paragraph on this bucket for end users>", "bucket_verified": <true/false>, "weakest_goal_id": <int>}}""".strip()


def _call_group_verifier(prompt, model_id):
    payload = {
        "model"  : model_id,
        "prompt" : prompt,
        "stream" : False,
        "format" : "json",
        "options": {"temperature": 0.1, "num_predict": 600,
                    "top_p": 0.9, "repeat_penalty": 1.1},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        raw = json.loads(resp.read()).get("response", "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```json|```", "", raw).strip()
    m   = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON in group verifier: {raw[:200]}")


def score_group(goal_rows, rule_rows, bucket_context,
                group_verifier_model=None, verbose=True):
    """
    Parameters
    ----------
    goal_rows       : list of dict or pd.Series — one per goal
    rule_rows       : list of dict or pd.Series — matching rule scores
    bucket_context  : dict — l2_name, l1_name, l2_alloc_pct, l1_alloc_pct, n_siblings
    group_verifier_model : str — Ollama model ID for group check (None = skip)
    verbose         : bool

    Returns
    -------
    dict:
        goals          — list of individual score_goal payloads
        group_summary  — avg scores, weakest goal, flags, narrative, bucket_composite
    """
    if len(goal_rows) != len(rule_rows):
        raise ValueError(f"goal_rows ({len(goal_rows)}) and rule_rows ({len(rule_rows)}) must match")

    bucket = bucket_context.get("l2_name", "Unknown")

    if verbose:
        print("=" * 70)
        print(f"SCORE GROUP — {bucket}  ({len(goal_rows)} goals)")
        print("=" * 70)

    # ── Score each goal individually ──────────────────────────────────────────
    individual_results = []
    individual_flags   = {}

    for idx, (goal_row, rule_row) in enumerate(zip(goal_rows, rule_rows)):
        gid = int(rule_row.get("goal_id", idx)) if hasattr(rule_row, "get") \
              else int(rule_row["goal_id"]) if "goal_id" in rule_row else idx
        if verbose:
            print(f"\n  [{idx+1}/{len(goal_rows)}] Goal {gid}")

        result = score_goal(goal_row, rule_row, verbose=verbose)

        # Attach alloc_pct for group prompt
        alloc = float(goal_row.get("allocation_percentage_of_parent", 0)) \
                if hasattr(goal_row, "get") else 0
        result["alloc_pct_of_parent"] = round(alloc, 4)

        # Attach goal name if available
        bucket_name = goal_row.get("bucket_name", f"goal_{gid}") \
                      if hasattr(goal_row, "get") else f"goal_{gid}"
        result["goal_name"] = bucket_name

        individual_results.append(result)
        individual_flags[gid] = result.get("flags", [])

    # ── Aggregate stats ───────────────────────────────────────────────────────
    def safe_mean(key):
        vals = [r[key] for r in individual_results
                if r.get("status") == "ok" and key in r]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    avg_attainability = safe_mean("attainability")
    avg_relevance     = safe_mean("relevance")
    avg_coherence     = safe_mean("coherence")
    avg_integrity     = safe_mean("integrity")
    avg_composite     = safe_mean("overall")

    # Weighted bucket composite
    bucket_composite = round(
        avg_coherence * WEIGHTS["coherence"] +
        avg_attainability * WEIGHTS["attainability"] +
        avg_relevance * WEIGHTS["relevance"] +
        avg_integrity * WEIGHTS["integrity"], 4
    )

    # Find weakest goal by composite
    ok_results = [r for r in individual_results if r.get("status") == "ok"]
    weakest = min(ok_results, key=lambda x: x.get("overall", 1.0)) if ok_results else None
    weakest_goal_id = weakest.get("goal_id", -1) if weakest else -1
    weakest_dim = min(
        DIMS, key=lambda d: avg_attainability if d=="attainability"
        else avg_relevance if d=="relevance"
        else avg_coherence if d=="coherence"
        else avg_integrity
    )

    at_risk_count = sum(1 for r in ok_results if r.get("overall", 1.0) < 0.35)

    # ── Group verifier ────────────────────────────────────────────────────────
    group_flags  = []
    narrative    = ""
    bucket_verified = True
    verifier_ran = False

    if group_verifier_model and len(ok_results) > 1:
        if verbose:
            print(f"\n  Group verifier ({group_verifier_model})...")
        try:
            prompt  = _build_group_prompt(ok_results, bucket_context, individual_flags)
            t0      = time.time()
            vr      = _call_group_verifier(prompt, group_verifier_model)
            elapsed = round(time.time() - t0, 1)
            group_flags     = [f for f in vr.get("group_flags", []) if f and f.strip()]
            narrative       = vr.get("narrative", "")
            bucket_verified = vr.get("bucket_verified", True) and len(group_flags) == 0
            verifier_weakest= vr.get("weakest_goal_id", weakest_goal_id)
            if verifier_weakest: weakest_goal_id = verifier_weakest
            verifier_ran = True
            if verbose:
                print(f"  Verifier: {elapsed}s  flags={len(group_flags)}  "
                      f"verified={bucket_verified}")
                for flag in group_flags:
                    print(f"  GROUP FLAG: {flag}")
        except Exception as e:
            if verbose:
                print(f"  Group verifier failed: {e}")

    if verbose:
        print(f"\n  BUCKET SUMMARY — {bucket}")
        print(f"    avg_composite   : {bucket_composite:.3f}")
        print(f"    avg_coherence   : {avg_coherence:.3f}")
        print(f"    at_risk         : {at_risk_count}/{len(ok_results)}")
        print(f"    weakest_goal    : {weakest_goal_id}")
        print(f"    group_flags     : {len(group_flags)}")

    return {
        "scope"        : "group",
        "bucket"       : bucket,
        "l1"           : bucket_context.get("l1_name", ""),
        "n_goals"      : len(goal_rows),
        "goals"        : individual_results,
        "group_summary": {
            "bucket_composite"    : bucket_composite,
            "avg_attainability"   : avg_attainability,
            "avg_relevance"       : avg_relevance,
            "avg_coherence"       : avg_coherence,
            "avg_integrity"       : avg_integrity,
            "at_risk_count"       : at_risk_count,
            "weakest_goal_id"     : weakest_goal_id,
            "weakest_dimension"   : weakest_dim,
            "group_flags"         : group_flags,
            "narrative"           : narrative,
            "bucket_verified"     : bucket_verified,
            "verifier_ran"        : verifier_ran,
        },
        "status"       : "ok" if ok_results else "error",
    }


# ── CLI — test with real data ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, default="Paid Acquisition",
                        help="L2 bucket name to score as a group")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-verifier", action="store_true",
                        help="Skip group verifier call")
    args = parser.parse_args()

    print("=" * 70)
    print(f"SCORE GROUP — {args.bucket}")
    print("=" * 70)

    # Load data
    features_raw = pd.read_csv("features_raw_poc.csv")
    rule_scores  = pd.read_csv("rule_scores_poc.csv")
    period_12    = pd.read_csv("period_12_poc.csv").reset_index(drop=True)
    buckets_df   = pd.read_csv("buckets.csv")
    goals_df     = pd.read_csv("goals.csv")

    # Merge extra columns
    extra_cols = [
        "allocation_efficiency_ratio","delivered_output_quantity",
        "allocation_percentage_of_parent","target_value_final_period",
        "time_to_green_estimate","weighted_goal_status_score",
        "sibling_rank_pct","l3_share_of_l2","alloc_drift_std",
        "needle_move_ratio","allocation_fitness_score",
        "observed_value","variance_from_target",
        "delivered_output_quality_score","trailing_6_period_slope",
        "bucket_name", "parent_bucket_name",
    ]
    for col in extra_cols:
        if col not in features_raw.columns and col in period_12.columns:
            features_raw[col] = period_12[col].values

    # Find goals in this L2 bucket
    l3_in_bucket = buckets_df[
        (buckets_df['bucket_level'] == 3) &
        (buckets_df['parent_bucket_id'].isin(
            buckets_df[buckets_df['bucket_name'] == args.bucket]['bucket_id'].values
        ))
    ]['bucket_id'].values

    l2_row  = buckets_df[buckets_df['bucket_name'] == args.bucket]
    if l2_row.empty:
        print(f"ERROR: bucket '{args.bucket}' not found in buckets.csv")
        raise SystemExit(1)

    l2_id   = l2_row['bucket_id'].values[0]
    l1_id   = l2_row['parent_bucket_id'].values[0]
    l2_alloc= float(l2_row['allocation_percentage_of_total'].values[0])
    l1_alloc= float(buckets_df[buckets_df['bucket_id']==l1_id]['allocation_percentage_of_total'].values[0])
    l1_name = buckets_df[buckets_df['bucket_id']==l1_id]['bucket_name'].values[0]

    # Match to goal rows
    goal_ids_in_bucket = goals_df[goals_df['bucket_id'].isin(l3_in_bucket)]['goal_id'].values
    feat_rows  = [features_raw[features_raw['goal_id']==gid].iloc[0]
                  if gid in features_raw['goal_id'].values else features_raw.iloc[0]
                  for gid in goal_ids_in_bucket]
    rule_rows_list = [rule_scores[rule_scores['goal_id']==gid].iloc[0]
                      if gid in rule_scores['goal_id'].values else rule_scores.iloc[0]
                      for gid in goal_ids_in_bucket]

    if not feat_rows:
        print(f"ERROR: no goals found for bucket '{args.bucket}'")
        raise SystemExit(1)

    print(f"Found {len(feat_rows)} goals: {list(goal_ids_in_bucket)}")

    bucket_context = {
        "l2_name"     : args.bucket,
        "l1_name"     : l1_name,
        "l2_alloc_pct": l2_alloc,
        "l1_alloc_pct": l1_alloc,
        "n_siblings"  : len(feat_rows),
    }

    # Determine verifier model
    verifier_model = None
    if not args.no_verifier:
        import os
        # Check if verify_goal.py exists in this folder
        if os.path.exists("verify_goal.py"):
            import re as _re
            with open("verify_goal.py") as vf:
                vm = _re.search(r'VERIFIER_MODEL\s*=\s*"([^"]+)"', vf.read())
                verifier_model = vm.group(1) if vm else "deepseek-r1:8b"
            print(f"Group verifier: {verifier_model}")
        else:
            print("No verify_goal.py found — skipping group verifier")

    result = score_group(
        goal_rows      = feat_rows,
        rule_rows      = rule_rows_list,
        bucket_context = bucket_context,
        group_verifier_model = verifier_model,
        verbose        = True,
    )

    print("\n" + "=" * 70)
    print("GROUP RESULT:")
    print(json.dumps({
        "bucket"         : result["bucket"],
        "n_goals"        : result["n_goals"],
        "group_summary"  : result["group_summary"],
    }, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Saved to {args.output}")
    print("=" * 70)
