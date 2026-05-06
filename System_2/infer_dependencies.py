import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
infer_dependencies.py  v2 (single shot, Mistral Large 2407)

Reads goals.csv + buckets.csv + derived_fields.csv, makes ONE Bedrock call
to Mistral Large 2407 to extract a dependency graph across all 35 goals,
validates the graph, computes derived columns, writes goal_dependencies.csv.

Pipeline position: AFTER load_data, BEFORE feature_engineering.

Output columns (consumed by feature_engineering and score_goal):
  goal_id, n_dependencies, n_dependents, dependency_risk,
  depends_on_ids (JSON list), depended_on_by_ids (JSON list),
  dep_avg_attain (float), rationale (str)

Density target:
  Medium. Each goal has 0 to 3 upstream dependencies, average around 1 to 2.

If the old broken duplicate-scoring version of this script left a stale
goal_dependencies.csv behind, delete it first or pass --force to overwrite.
"""

import os
import sys
import json
import time
import argparse

import pandas as pd
import numpy as np

from bedrock_client import call_model, extract_json, BedrockCallError


ANCHOR_PERIOD       = 18
MAX_RETRIES         = 3
MAX_DEPS_PER_GOAL   = 3
OUTPUT_FILE         = "goal_dependencies.csv"
SOURCE_GOALS        = "goals.csv"
SOURCE_BUCKETS      = "buckets.csv"
SOURCE_DERIVED      = "derived_fields.csv"


def load_inputs():
    missing = [f for f in (SOURCE_GOALS, SOURCE_BUCKETS, SOURCE_DERIVED) if not os.path.exists(f)]
    if missing:
        print(f"  ERROR: missing required source file(s): {missing}")
        sys.exit(1)

    goals   = pd.read_csv(SOURCE_GOALS)
    buckets = pd.read_csv(SOURCE_BUCKETS)
    derived = pd.read_csv(SOURCE_DERIVED)

    l1 = buckets[buckets["bucket_level"] == 1][["bucket_id", "bucket_name"]].rename(
        columns={"bucket_id": "l1_id", "bucket_name": "l1_name"})
    l2 = buckets[buckets["bucket_level"] == 2][["bucket_id", "bucket_name", "parent_bucket_id"]].rename(
        columns={"bucket_id": "l2_id", "bucket_name": "l2_name", "parent_bucket_id": "l1_id"})
    l3 = buckets[buckets["bucket_level"] == 3][["bucket_id", "bucket_name", "parent_bucket_id"]].rename(
        columns={"bucket_id": "l3_id", "bucket_name": "l3_name", "parent_bucket_id": "l2_id"})
    hier = l3.merge(l2, on="l2_id").merge(l1, on="l1_id")

    goals_ctx = goals.merge(
        hier[["l3_id", "l3_name", "l2_name", "l1_name"]],
        left_on="bucket_id", right_on="l3_id", how="left"
    )

    attain = derived[derived["period_id"] == ANCHOR_PERIOD][
        ["goal_id", "probability_of_hitting_target"]
    ].copy()
    if len(attain) == 0:
        print(f"  WARNING: no derived rows at period {ANCHOR_PERIOD}, falling back to period 12")
        attain = derived[derived["period_id"] == 12][
            ["goal_id", "probability_of_hitting_target"]
        ].copy()

    return goals_ctx, attain


def build_prompt(goals_ctx):
    lines = []
    for _, g in goals_ctx.iterrows():
        lines.append(
            f"Goal {int(g['goal_id'])}: {g['metric_name']} ({g['metric_unit']})\n"
            f"  Bucket path : {g['l1_name']} > {g['l2_name']} > {g['l3_name']}\n"
            f"  Trajectory  : {g['initial_value']} -> {g['target_value_final_period']} by period 24\n"
            f"  Scenario    : {g['scenario_story']}"
        )
    goal_block = "\n\n".join(lines)

    return f"""You are mapping causal dependencies across 35 organisational goals.

DEFINITION
  Goal A "depends on" Goal B if A's success requires B to be on track first.
  Examples:
    Lead Generation depends on Website Traffic (no traffic, no leads)
    Customer Retention depends on Product Quality (bad product, churn rises)
    Sales Revenue depends on Lead Conversion (no conversion, no revenue)
  This is directed: A depends on B does NOT mean B depends on A.

THE 35 GOALS

{goal_block}

INSTRUCTIONS
  1. For each goal, identify 0 to {MAX_DEPS_PER_GOAL} OTHER goals it depends on.
     Aim for an average density of 1 to 2 dependencies per goal across the set.
  2. Only suggest a dependency when you can articulate a concrete causal mechanism.
     Thematic similarity (both in Marketing) is NOT enough.
  3. Cross-bucket edges (e.g. Marketing -> Sales, Operations -> Customer) are
     usually more meaningful than same-bucket edges.
  4. A goal cannot depend on itself.
  5. When unsure, prefer fewer dependencies over more.
  6. Use the integer goal_id values, not bucket ids or names.
  7. Include EVERY goal in the response. Use an empty depends_on_ids list if none.

Return ONLY valid JSON in this exact shape. No markdown, no preamble, no commentary.
Start with {{ and end with }}.

{{
  "dependencies": [
    {{"goal_id": 1, "depends_on_ids": [7, 12], "rationale": "under 25 words"}},
    {{"goal_id": 2, "depends_on_ids": [], "rationale": "under 25 words"}}
  ]
}}
""".strip()


def validate_graph(parsed, valid_goal_ids):
    if "dependencies" not in parsed:
        raise ValueError("Response missing 'dependencies' key")

    seen_goals = set()
    clean      = []
    issues     = []

    for entry in parsed["dependencies"]:
        try:
            gid = int(entry["goal_id"])
        except (KeyError, ValueError, TypeError):
            issues.append(f"  bad goal_id in entry: {entry}")
            continue

        if gid not in valid_goal_ids:
            issues.append(f"  unknown goal_id {gid}, skipped")
            continue

        if gid in seen_goals:
            issues.append(f"  duplicate goal_id {gid}, kept first occurrence")
            continue
        seen_goals.add(gid)

        raw_deps  = entry.get("depends_on_ids", []) or []
        rationale = str(entry.get("rationale", "")).strip()

        deps_clean = []
        for d in raw_deps:
            try:
                d_int = int(d)
            except (ValueError, TypeError):
                issues.append(f"  goal {gid}: non-integer dep {d}, skipped")
                continue
            if d_int == gid:
                issues.append(f"  goal {gid}: self-loop dropped")
                continue
            if d_int not in valid_goal_ids:
                issues.append(f"  goal {gid}: unknown upstream {d_int}, skipped")
                continue
            if d_int in deps_clean:
                continue
            deps_clean.append(d_int)

        if len(deps_clean) > MAX_DEPS_PER_GOAL:
            issues.append(f"  goal {gid}: capped from {len(deps_clean)} to {MAX_DEPS_PER_GOAL} deps")
            deps_clean = deps_clean[:MAX_DEPS_PER_GOAL]

        clean.append({
            "goal_id"       : gid,
            "depends_on_ids": deps_clean,
            "rationale"     : rationale[:240],
        })

    for gid in sorted(valid_goal_ids):
        if gid not in seen_goals:
            issues.append(f"  goal {gid}: missing from response, defaulted to empty")
            clean.append({
                "goal_id"       : gid,
                "depends_on_ids": [],
                "rationale"     : "missing from LLM response",
            })

    return clean, issues


def derive_columns(graph_rows, attain_df):
    by_id = {r["goal_id"]: r for r in graph_rows}

    inverse = {gid: [] for gid in by_id}
    for r in graph_rows:
        for upstream in r["depends_on_ids"]:
            if upstream in inverse:
                inverse[upstream].append(r["goal_id"])

    attain_lookup = dict(zip(attain_df["goal_id"], attain_df["probability_of_hitting_target"]))

    enriched = []
    for r in graph_rows:
        gid    = r["goal_id"]
        deps   = r["depends_on_ids"]
        depots = sorted(inverse.get(gid, []))
        n_deps = len(deps)
        n_dpts = len(depots)

        if   n_deps >= 3: risk = "high"
        elif n_deps == 2: risk = "medium"
        elif n_deps == 1: risk = "low"
        else:             risk = "none"

        if deps:
            upstream_attain = [attain_lookup.get(d, np.nan) for d in deps]
            valid           = [a for a in upstream_attain if pd.notna(a)]
            dep_avg_attain  = float(np.mean(valid)) if valid else -1.0
        else:
            dep_avg_attain = -1.0

        enriched.append({
            "goal_id"           : gid,
            "n_dependencies"    : n_deps,
            "n_dependents"      : n_dpts,
            "dependency_risk"   : risk,
            "depends_on_ids"    : json.dumps(deps),
            "depended_on_by_ids": json.dumps(depots),
            "dep_avg_attain"    : round(dep_avg_attain, 4),
            "rationale"         : r["rationale"],
        })

    return enriched


def print_summary(rows):
    n = len(rows)
    total_edges   = sum(r["n_dependencies"] for r in rows)
    avg_deps      = total_edges / n if n else 0
    risk_counts   = {k: sum(1 for r in rows if r["dependency_risk"] == k)
                     for k in ["none", "low", "medium", "high"]}
    isolated      = sum(1 for r in rows if r["n_dependencies"] == 0 and r["n_dependents"] == 0)
    most_depended = sorted(rows, key=lambda r: r["n_dependents"], reverse=True)[:5]

    print("\n  GRAPH SUMMARY")
    print(f"    Goals             : {n}")
    print(f"    Total edges       : {total_edges}")
    print(f"    Avg deps per goal : {avg_deps:.2f}")
    print(f"    Risk distribution : {risk_counts}")
    print(f"    Isolated goals    : {isolated}")
    print(f"\n  Top 5 most-depended-on goals:")
    for r in most_depended:
        if r["n_dependents"] == 0:
            break
        print(f"    goal {r['goal_id']:>3}  has {r['n_dependents']} dependent(s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompt, print it, do not call the LLM.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing goal_dependencies.csv.")
    args = parser.parse_args()

    print("=" * 70)
    print("DECIDR COHERENCE ENGINE  Dependency Inference (Claude 3 Haiku)")
    print("=" * 70)

    if os.path.exists(OUTPUT_FILE) and not args.force and not args.dry_run:
        print(f"\n  {OUTPUT_FILE} already exists.")
        print(f"  Use --force to overwrite, or pass --skip-deps to run.py to skip.")
        return 0

    print(f"\n  Model         : Claude 3 Haiku")
    print(f"  Anchor period : {ANCHOR_PERIOD} (for dep_avg_attain only)")
    print(f"  Density cap   : {MAX_DEPS_PER_GOAL} deps per goal")
    print(f"  Region        : {os.getenv('AWS_REGION', 'us-west-2')}")

    print("\n  Loading inputs...")
    goals_ctx, attain = load_inputs()
    valid_ids = set(goals_ctx["goal_id"].astype(int))
    print(f"    goals     : {len(goals_ctx)}")
    print(f"    p{ANCHOR_PERIOD} attain rows : {len(attain)}")

    prompt = build_prompt(goals_ctx)
    print(f"    prompt chars: {len(prompt)}")

    if args.dry_run:
        print("\n  DRY RUN  prompt below, no LLM call made:\n")
        print("-" * 70)
        print(prompt)
        print("-" * 70)
        return 0

    parsed   = None
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n  Calling Bedrock (attempt {attempt}/{MAX_RETRIES})...")
            t0  = time.time()
            raw = call_model(
                "dependency_inferrer",
                prompt,
                max_tokens=2000,
                temperature=0.2,
            )
            elapsed = time.time() - t0
            print(f"    response in {elapsed:.1f}s, {len(raw)} chars")

            parsed = extract_json(raw, required_keys=["dependencies"])
            n_entries = len(parsed.get("dependencies", []))
            print(f"    parsed {n_entries} entries from response")
            break
        except (BedrockCallError, ValueError, Exception) as e:
            last_err = e
            msg = str(e).lower()
            is_transient = any(tok in msg for tok in (
                "throttl", "timeout", "timed out", "read timed",
                "connection reset", "rate", "service unavailable", "503",
            ))
            if attempt < MAX_RETRIES and is_transient:
                wait = 5 * attempt
                print(f"    transient error (attempt {attempt}/{MAX_RETRIES}): {e}")
                print(f"    backing off {wait}s before retry")
                time.sleep(wait)
            else:
                print(f"    attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(2)

    if parsed is None:
        print(f"\n  ERROR: all {MAX_RETRIES} LLM attempts failed.")
        print(f"  Last error: {last_err}")
        print(f"  Writing empty fallback graph (all goals isolated).")
        graph_rows = [{"goal_id": gid, "depends_on_ids": [], "rationale": "LLM failed, fallback"}
                      for gid in sorted(valid_ids)]
        issues = [f"all {MAX_RETRIES} LLM attempts failed: {last_err}"]
    else:
        print("\n  Validating graph...")
        graph_rows, issues = validate_graph(parsed, valid_ids)

    if issues:
        print(f"\n  Validation flagged {len(issues)} issue(s):")
        for msg in issues[:20]:
            print(msg)
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")

    print("\n  Computing derived columns...")
    enriched = derive_columns(graph_rows, attain)
    print_summary(enriched)

    out_df = pd.DataFrame(enriched).sort_values("goal_id").reset_index(drop=True)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  OK  wrote {OUTPUT_FILE} ({len(out_df)} rows, {len(out_df.columns)} cols)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())