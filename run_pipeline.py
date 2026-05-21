"""
run_pipeline.py — Decidr Coherence Engine
Full pipeline orchestrator. Run once before launching app.py.

Steps:
  1. System 3  — Load raw CSVs, validate, engineer features, infer dependencies
  2. System 2  — Train Temporal GP meta-learner, produce predictions
  3. System 2  — Compute composite scores and forward projection

Usage:
  python run_pipeline.py                  # full run
  python run_pipeline.py --skip-deps      # skip Bedrock dependency inference
  python run_pipeline.py --skip-meta      # only prep data (System 3 only)
  python run_pipeline.py --skip-verify    # skip LLM verification pass

Then run:
  streamlit run app.py
"""

import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
for sub in ("System_1", "System_2", "System_3"):
    sys.path.insert(0, os.path.join(ROOT, sub))

# Load .env into os.environ before anything else — same as run.py
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"), override=False)
except ImportError:
    pass


def run_script(script_path: str, extra_args: list = None):
    cmd = [sys.executable, script_path] + (extra_args or [])
    # Pass full os.environ (including .env vars) to subprocess — same as run.py
    result = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(f"{script_path} failed — check output above.")


def main():
    parser = argparse.ArgumentParser(description="Decidr Coherence Engine — full pipeline")
    parser.add_argument("--skip-deps",   action="store_true",
                        help="Skip Bedrock dependency inference (reuse existing goal_dependencies.csv)")
    parser.add_argument("--skip-meta",   action="store_true",
                        help="Run System 3 only — stop before meta-learner")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip LLM verification in composite_score.py")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("STEP 1 — SYSTEM 3: Data Loading + Feature Engineering")
    print("=" * 70)

    try:
        from System_3.pipeline import run_system3_data_pipeline
        # Run data loading and feature engineering only — no dependency inference
        run_system3_data_pipeline(
            infer_dependencies=False,
            force_dependencies=False,
            verbose=True,
        )
    except ImportError:
        print("System_3 not importable — running feature_engineering.py directly")
        run_script(os.path.join(ROOT, "System_2", "feature_engineering.py"))

    # Dependency inference always uses the original infer_dependencies.py
    # (same as run.py) — System_3/embeddings.py uses a different Bedrock
    # client initialisation that doesn't pick up .env credentials correctly.
    if not args.skip_deps:
        run_script(os.path.join(ROOT, "System_2", "infer_dependencies.py"))

    if args.skip_meta:
        print("\nStopping after System 3 (--skip-meta). Feature files ready. Run llm_scoring.py next.")
        return

    print("\n" + "=" * 70)
    print("STEP 2 — SYSTEM 2: LLM Scoring (Llama + Mistral via Bedrock)")
    print("=" * 70)
    run_script(os.path.join(ROOT, "System_2", "llm_scoring.py"))

    print("\n" + "=" * 70)
    print("STEP 3 — SYSTEM 2: Meta-Learner (Temporal GP)")
    print("=" * 70)
    run_script(os.path.join(ROOT, "System_2", "meta_learner.py"))

    print("\n" + "=" * 70)
    print("STEP 4 — SYSTEM 2: Composite Scoring + Forward Projection")
    print("=" * 70)
    extra = ["--skip-verify"] if args.skip_verify else []
    run_script(os.path.join(ROOT, "System_2", "composite_score.py"), extra)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — all CSV outputs ready")
    print("  Launch dashboard:  streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
