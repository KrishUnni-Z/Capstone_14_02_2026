"""
run.py  Master runner v5

Pipeline:
  1. load_data.py           - validate inputs, extract p12 and p18 snapshots
  2. infer_dependencies.py  - Mistral Large one-shot, builds goal_dependencies.csv
  3. feature_engineering.py - builds features and rule scores, reads dependency graph
  4. llm_scoring.py         - Llama 3.3 + Mistral Large 2407 testers, periods [6, 12, 18]
  5. meta_learner.py        - GP + isotonic calibration + dynamic ensemble (p12+p18)
  6. composite_score.py     - weighted composite, Anthropic verifier, forward projection
  7. explanations.py        - GP uncertainty, dimension plots, perturbation importance
  8. dashboard.py           - summary dashboard

Flags:
  python run.py                         # full run
  python run.py --skip-llm              # skip llm_scoring, reuse predictions
  python run.py --skip-deps             # skip infer_dependencies, reuse graph
  python run.py --skip-llm --skip-deps  # skip both

Env:
  Requires .env with AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
  pip install boto3 python-dotenv scikit-learn pandas numpy matplotlib
"""

import subprocess
import sys
import os


SKIP_LLM  = "--skip-llm" in sys.argv or "--skip03" in sys.argv
SKIP_DEPS = "--skip-deps" in sys.argv

PIPELINE = [
    ("load_data.py",           False),
    ("infer_dependencies.py",  SKIP_DEPS),
    ("feature_engineering.py", False),
    ("llm_scoring.py",         SKIP_LLM),
    ("meta_learner.py",        False),
    ("composite_score.py",     False),
    ("explanations.py",        False),
    ("dashboard.py",           False),
]


def run(script):
    print(f"\nRunning {script} ...")
    result = subprocess.run([sys.executable, script], env=os.environ.copy())
    if result.returncode != 0:
        print(f"\n  FAILED: {script} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  {script}  OK")


def check_outputs():
    key_files = [
        "analytical_full.csv",
        "goal_dependencies.csv",
        "features_full_normalized.csv",
        "features_raw_p18.csv",
        "rule_scores_p18.csv",
        "llm_predictions_poc.csv",
        "meta_learner_predictions_poc.csv",
        "meta_learner_results_poc.csv",
        "gp_poc.pkl",
        "platt_scalers_poc.pkl",
        "gp_config_poc.json",
        "shap_importance_poc.csv",
        "shap_summary_poc.png",
        "gp_uncertainty_poc.png",
        "all_scores_poc.png",
        "demo_dashboard_poc.png",
        "composite_scores_poc.csv",
        "portfolio_summary_poc.csv",
        "forward_projection_poc.csv",
        "composite_dashboard_poc.png",
    ]
    print("\nOutput files:")
    for f in key_files:
        exists = os.path.exists(f)
        print(f"  {'OK     ' if exists else 'MISSING'} {f}")


if __name__ == "__main__":
    print("=" * 70)
    print("DECIDR COHERENCE ENGINE  Runner v5")
    print("Testers  : Llama 3.3 70B + Mistral Large 2407")
    print("Verifier : Verifier : Claude 3 Haiku (fallback: Llama 3.1 8B))")
    print("Deps     : Mistral Large 2407 (single shot)")
    print("Anchor   : period 18")
    print("=" * 70)

    flag_map = {
        "llm_scoring.py"        : "--skip-llm",
        "infer_dependencies.py" : "--skip-deps",
    }

    for script, skip in PIPELINE:
        if skip:
            flag = flag_map.get(script, "skip")
            print(f"\nSkipping {script}  ({flag} flag)")
        else:
            run(script)

    check_outputs()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("\nTo score a single goal (inference):")
    print("  python score_goal.py --goal_id 0")
    print("  python score_goal.py --goal_id 7 --output result.json")
    print("  python score_goal.py --goal_id 0 --period 18")
    print("\nTo re-run without calling LLMs again:")
    print("  python run.py --skip-llm                # skip LLM scoring")
    print("  python run.py --skip-deps               # skip dependency inference")
    print("  python run.py --skip-llm --skip-deps    # skip both")
    print("=" * 70)