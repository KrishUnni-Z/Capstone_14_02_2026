"""
infer_dependencies.py — Cross-goal dependency inference
Runs once after feature_engineering.py. Calls one LLM with all 35 goals
and infers which goals depend on each other.

Output: goal_dependencies.csv
  goal_id | depends_on_ids | depended_on_by_ids | dependency_types | confidence_scores
"""

import json
import re
import time
import urllib.request
import pandas as pd
import numpy as np

OLLAMA_URL   = "http://localhost:11434/api/generate"
INFER_MODEL  = "llama3:latest"   # cheap, one-shot call
TIMEOUT_SEC  = 300
CONFIDENCE_THRESHOLD = 0.65      # ignore low-confidence dependencies

print("=" * 70)
print("INFER DEPENDENCIES — Cross-goal relationship mapping")
print("=" * 70)

# ── Load goals and hierarchy ──────────────────────────────────────────────────
goals   = pd.read_csv("goals.csv")
buckets = pd.read_csv("buckets.csv")

# Build full goal list with bucket path
l1 = buckets[buckets['bucket_level']==1][['bucket_id','bucket_name']].copy()
l1.columns = ['l1_id','l1_name']
l2 = buckets[buckets['bucket_level']==2][['bucket_id','bucket_name','parent_bucket_id']].copy()
l2.columns = ['l2_id','l2_name','l1_id']
l3 = buckets[buckets['bucket_level']==3][['bucket_id','bucket_name','parent_bucket_id']].copy()
l3.columns = ['l3_id','l3_name','l2_id']

hier = l3.merge(l2, on='l2_id').merge(l1, on='l1_id')
goal_info = goals[['goal_id','metric_name']].merge(
    hier, left_on=goals['goal_id'].apply(lambda x: buckets[buckets['bucket_id']==x+18]['bucket_id'].values[0] if x+18 <= 53 else x),
    right_on='l3_id', how='left'
)

# Simpler join via analytical flat
flat = pd.read_csv("analytical_flat.csv")
p12  = flat[flat['period_id']==12][['goal_id','bucket_id','bucket_name',
                                    'parent_bucket_name','probability_of_hitting_target']].copy()
p12  = p12.merge(goals[['goal_id','metric_name']], on='goal_id', how='left')

print(f"\nGoals: {len(p12)}")
print("Building dependency prompt...")

# ── Build prompt ──────────────────────────────────────────────────────────────
goal_lines = []
for _, row in p12.iterrows():
    goal_lines.append(
        f"  ID={int(row['goal_id']):>3}  [{row['parent_bucket_name']} → {row['bucket_name']}]  "
        f"attainability={row['probability_of_hitting_target']:.1f}"
    )

goal_list = "\n".join(goal_lines)

prompt = f"""You are analysing an organisational goal portfolio. Below are 35 goals with their bucket paths and current attainability scores.

Your task: identify which goals have DIRECT dependencies on other goals.
A dependency exists when Goal A's performance REQUIRES or is DIRECTLY AFFECTED by Goal B's performance.

Focus on clear operational dependencies:
  resource   = Goal A needs output/infrastructure that Goal B produces
  sequential = Goal A logically follows Goal B
  metric     = Goal A's metric is directly driven by Goal B's execution

GOALS:
{goal_list}

Return a JSON array. Only include relationships with confidence >= 0.65.
For each dependency, lower the goal_id that depends on the other.

Return ONLY this JSON array:
[
  {{"goal_id": <int>, "depends_on_id": <int>, "dependency_type": "resource|sequential|metric", "confidence": <0.65-0.99>, "reason": "one sentence"}},
  ...
]

If no clear dependencies exist for a goal, omit it. Return empty array [] if none found."""

# ── Call LLM ─────────────────────────────────────────────────────────────────
print(f"Calling {INFER_MODEL} for dependency inference...")

payload = {
    "model"  : INFER_MODEL,
    "prompt" : prompt,
    "stream" : False,
    "options": {"temperature": 0.1, "num_predict": 800, "top_p": 0.9},
}
req = urllib.request.Request(
    OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}, method="POST",
)

raw_deps = []
try:
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        raw = json.loads(resp.read()).get("response", "").strip()
    elapsed = round(time.time() - t0, 1)
    print(f"  LLM response: {elapsed}s")

    # Strip think blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    # Parse JSON array
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw_deps = json.loads(m.group(0))
        # Filter by confidence
        raw_deps = [d for d in raw_deps if d.get("confidence", 0) >= CONFIDENCE_THRESHOLD]
        print(f"  Dependencies found: {len(raw_deps)} (confidence >= {CONFIDENCE_THRESHOLD})")
    else:
        print("  No dependency array found — using empty set")
        raw_deps = []
except Exception as e:
    print(f"  LLM call failed: {e} — using empty dependencies")
    raw_deps = []

# ── Build per-goal dependency records ────────────────────────────────────────
goal_ids = p12['goal_id'].tolist()

records = []
for gid in goal_ids:
    depends_on     = [d for d in raw_deps if d['goal_id']     == gid]
    depended_on_by = [d for d in raw_deps if d['depends_on_id'] == gid]

    dep_ids   = [d['depends_on_id']  for d in depends_on]
    rdep_ids  = [d['goal_id']        for d in depended_on_by]
    dep_types = [d.get('dependency_type','resource') for d in depends_on]
    dep_conf  = [d.get('confidence', 0.7) for d in depends_on]
    dep_reasons = [d.get('reason','') for d in depends_on]

    # Dependency risk: high if any dependency has low attainability
    dep_attain = []
    for did in dep_ids:
        row = p12[p12['goal_id'] == did]
        if not row.empty:
            dep_attain.append(float(row['probability_of_hitting_target'].values[0]))

    dep_risk = "none"
    if dep_attain:
        min_dep = min(dep_attain)
        dep_risk = "high" if min_dep <= 0.1 else "medium" if min_dep <= 0.3 else "low"

    records.append({
        "goal_id"           : gid,
        "depends_on_ids"    : json.dumps(dep_ids),
        "depended_on_by_ids": json.dumps(rdep_ids),
        "dependency_types"  : json.dumps(dep_types),
        "confidence_scores" : json.dumps(dep_conf),
        "dependency_reasons": json.dumps(dep_reasons),
        "n_dependencies"    : len(dep_ids),
        "n_dependents"      : len(rdep_ids),
        "dependency_risk"   : dep_risk,
        "dep_avg_attain"    : round(float(np.mean(dep_attain)), 3) if dep_attain else -1.0,
    })

dep_df = pd.DataFrame(records)
dep_df.to_csv("goal_dependencies.csv", index=False)

print(f"\nDependency summary:")
print(f"  Goals with dependencies   : {(dep_df['n_dependencies'] > 0).sum()}")
print(f"  Goals depended on by others: {(dep_df['n_dependents'] > 0).sum()}")
print(f"  High risk (upstream at risk): {(dep_df['dependency_risk'] == 'high').sum()}")
print(f"\nDependency map:")
for _, row in dep_df[dep_df['n_dependencies'] > 0].iterrows():
    dep_ids = json.loads(row['depends_on_ids'])
    types   = json.loads(row['dependency_types'])
    reasons = json.loads(row['dependency_reasons'])
    goal_name = p12[p12['goal_id']==row['goal_id']]['bucket_name'].values[0]
    for did, dtype, reason in zip(dep_ids, types, reasons):
        dep_name = p12[p12['goal_id']==did]['bucket_name'].values[0] if did in p12['goal_id'].values else f"goal_{did}"
        print(f"  [{row['goal_id']}] {goal_name:<35} --{dtype}--> [{did}] {dep_name}")
        print(f"       {reason}")

print(f"\n✓ Saved goal_dependencies.csv")
print("=" * 70)
