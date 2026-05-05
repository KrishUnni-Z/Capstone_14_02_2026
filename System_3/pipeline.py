"""
pipeline.py

I use this file as the main System 3 orchestrator.

This file splits the brain team's old pipeline into our System_3 structure:

1. data_loader.py
   I load and validate all raw CSV files.

2. embeddings.py
   I infer or reuse goal_dependencies.csv.

3. features.py
   I build engineered feature files and rule score files.

The final output is exactly what System 2 / the brain expects.
"""

from pathlib import Path

try:
    from .data_loader import run_data_loader
    from .features import save_feature_outputs
except ImportError:
    from data_loader import run_data_loader
    from features import save_feature_outputs

# embeddings (sentence_transformers + faiss) loaded lazily
# so missing the package does not break the whole import
def _get_infer_fn():
    try:
        from .embeddings import infer_goal_dependencies
    except ImportError:
        try:
            from embeddings import infer_goal_dependencies
        except ImportError as e:
            raise ImportError(
                "sentence_transformers is required for dependency inference. "
                "Install it with:  pip install sentence-transformers\n"
                "Or run with --skip-deps to skip this step."
            ) from e
    return infer_goal_dependencies


BASE_DIR = Path(__file__).resolve().parent.parent


def run_system3_data_pipeline(
    infer_dependencies: bool = True,
    force_dependencies: bool = False,
    verbose: bool = True,
):
    """
    I run the full System 3 data pipeline.

    This produces:
    - analytical_full.csv
    - period_12_poc.csv
    - period_18_poc.csv
    - goal_dependencies.csv
    - features_full_normalized.csv
    - features_full_raw.csv
    - features_raw_p6/p12/p18/p24.csv
    - rule_scores_p6/p12/p18/p24.csv
    - features_raw_poc.csv
    - rule_scores_poc.csv
    - feature_scaler_poc.pkl
    - feature_names_poc.txt
    """
    if verbose:
        print("=" * 70)
        print("SYSTEM 3 DATA PIPELINE")
        print("=" * 70)

    # Step 1: I load and validate all raw source CSV files.
    tables = run_data_loader(verbose=verbose)

    # Step 2: I infer dependencies before feature engineering.
    # Feature engineering needs goal_dependencies.csv if dependency features are required.
    if infer_dependencies:
        _infer_fn = _get_infer_fn()
        _infer_fn(
            goals=tables["goals"],
            buckets=tables["buckets"],
            derived=tables["derived_fields"],
            force=force_dependencies,
            verbose=verbose,
        )
    else:
        if verbose:
            print("Skipping dependency inference. Existing goal_dependencies.csv will be used if present.")

    # Step 3: I build all feature and rule-score files.
    save_feature_outputs(
        source_tables=tables,
        output_dir=BASE_DIR,
        verbose=verbose,
    )

    if verbose:
        print("=" * 70)
        print("SYSTEM 3 COMPLETE")
        print("All System 2 input files are ready.")
        print("=" * 70)

"""
if __name__ == "__main__":
    run_system3_data_pipeline(
        infer_dependencies=True,
        force_dependencies=False,
        verbose=True,
    )
    """
#skipping Bedrock dependency inference and letting System 3 finish locally.
if __name__ == "__main__":
    run_system3_data_pipeline(
        infer_dependencies=True,
        force_dependencies=False,
        verbose=True,
    )
