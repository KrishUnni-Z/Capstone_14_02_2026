"""
system3_runner.py

I keep this as a simple runner for System 3.

It calls pipeline.py, which runs:
- data loading
- dependency inference
- feature engineering
- rule score generation
"""

try:
    from .pipeline import run_system3_data_pipeline
except ImportError:
    from pipeline import run_system3_data_pipeline


def run_system3(verbose: bool = True):
    """
    I run the final System 3 pipeline.
    """
    return run_system3_data_pipeline(
        infer_dependencies=True,
        force_dependencies=False,
        verbose=verbose,
    )


if __name__ == "__main__":
    run_system3(verbose=True)
