"""
mock_system2.py

I use this file when I want to test only System 3 without running the real System 2.

What I do:
- accept one goal row and one rule row
- return fake but realistic score output
"""

import numpy as np


def mock_score_goal(goal_row, rule_row, verbose=True):
    """
    I create fake scores using a few simple signals from the goal row.
    This lets me test the full System 3 pipeline safely.
    """
    prob = float(goal_row.get("probability_of_hitting_target", 0.5))
    eff = float(goal_row.get("allocation_efficiency_ratio", 0.5))
    relevance_rule = float(rule_row.get("relevance_rule", 0.5))
    coherence_rule = float(rule_row.get("coherence_rule", 0.5))
    integrity_rule = float(rule_row.get("integrity_rule", 0.5))

    attainability = float(np.clip(prob, 0.05, 0.95))
    relevance = float(np.clip(relevance_rule, 0.05, 0.95))
    coherence = float(np.clip(coherence_rule, 0.05, 0.95))
    integrity = float(np.clip((integrity_rule + eff) / 2.0, 0.05, 0.95))

    overall = float(
        0.25 * attainability +
        0.20 * relevance +
        0.35 * coherence +
        0.20 * integrity
    )

    if verbose:
        print(
            f"I am mock-scoring goal_id={int(goal_row.get('goal_id', -1))} "
            f"-> overall={overall:.3f}"
        )

    return {
        "goal_id": int(goal_row.get("goal_id", -1)),
        "attainability": round(attainability, 4),
        "relevance": round(relevance, 4),
        "coherence": round(coherence, 4),
        "integrity": round(integrity, 4),
        "overall": round(overall, 4),
        "gp_mean": round(attainability, 4),
        "gp_std": 0.10,
        "gp_weight": 0.50,
        "llm_weight": 0.50,
        "baseline": round(attainability, 4),
        "uncertain": False,
        "reasoning": {
            "attainability": "I used probability_of_hitting_target as a mock signal.",
            "relevance": "I used relevance_rule as a mock signal.",
            "coherence": "I used coherence_rule as a mock signal.",
            "integrity": "I used integrity_rule and efficiency as a mock signal.",
        },
        "narrative": "This is a mock System 2 result used only to test the System 3 pipeline.",
        "verified_attainability": round(attainability, 4),
        "verified_relevance": round(relevance, 4),
        "verified_coherence": round(coherence, 4),
        "verified_integrity": round(integrity, 4),
        "verified_composite": round(overall, 4),
        "flags": [],
        "verified": True,
        "adjustments": {},
        "status": "ok",
    }
