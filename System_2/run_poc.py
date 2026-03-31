"""
run_poc.py — Master runner v2
Runs full PoC pipeline for ElasticNet and/or Ridge variants.

Usage:
    python run_poc.py            # both variants
    python run_poc.py --elastic  # ElasticNet only
    python run_poc.py --ridge    # Ridge only
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np

RUN_ELASTIC = "--ridge"   not in sys.argv
RUN_RIDGE   = "--elastic" not in sys.argv

VARIANTS = []
if RUN_ELASTIC:
    VARIANTS.append({
        "name" : "ElasticNet",
        "meta" : "Pipeline/04_meta_learner_poc.py",
        "shap" : "Pipeline/05_explanations_poc.py",
        "demo" : "Pipeline/06_demo_poc.py",
        "env"  : "elastic",
    })
if RUN_RIDGE:
    VARIANTS.append({
        "name" : "Ridge",
        "meta" : "Pipeline/04_meta_learner_poc_ridge.py",
        "shap" : "Pipeline/05_explanations_poc.py",
        "demo" : "Pipeline/06_demo_poc.py",
        "env"  : "ridge",
    })

SHARED = [
    "Pipeline/01_load_data.py",
    "Pipeline/02_feature_engineering_poc.py",
    "Pipeline/03_llm_predictions_poc.py",
]


def run(script, label="", model_env=None):
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Running {script} ...")
    env = os.environ.copy()
    if model_env:
        env["POC_MODEL"] = model_env
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        print(f"\n  FAILED: {script} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  {script}  OK")


def save_outputs(variant_name):
    suffix = "_elastic" if variant_name == "ElasticNet" else "_ridge"
    for f in [
        "meta_learner_results_poc.csv",
        "meta_learner_predictions_poc.csv",
        "feature_importance_poc.csv",
        "shap_importance_poc.csv",
        "demo_dashboard_poc.png",
        "shap_summary_poc.png",
        "shap_waterfall_goal0_poc.png",
    ]:
        if os.path.exists(f):
            base, ext = os.path.splitext(f)
            os.replace(f, f"{base}{suffix}{ext}")
    print(f"  Saved with suffix '{suffix}'")


def compare():
    print("\n" + "=" * 70)
    print("COMPARISON — ElasticNet vs Ridge")
    print("=" * 70)

    try:
        e  = pd.read_csv("meta_learner_results_poc_elastic.csv")
        r  = pd.read_csv("meta_learner_results_poc_ridge.csv")
        ep = pd.read_csv("meta_learner_predictions_poc_elastic.csv")
        rp = pd.read_csv("meta_learner_predictions_poc_ridge.csv")
        ei = pd.read_csv("shap_importance_poc_elastic.csv")
        ri = pd.read_csv("shap_importance_poc_ridge.csv")
    except FileNotFoundError as err:
        print(f"  Missing file: {err}")
        return

    def fmt(val):
        return f"{val:.4f}" if pd.notna(val) else "  N/A "

    print(f"\n{'Metric':<22} {'ElasticNet':>12} {'Ridge':>12}")
    print("-" * 48)
    print(f"{'LOO MAE':<22} {fmt(e['loo_mae'].iloc[0]):>12} {fmt(r['loo_mae'].iloc[0]):>12}")
    print(f"{'LOO RMSE':<22} {fmt(e['loo_rmse'].iloc[0]):>12} {fmt(r['loo_rmse'].iloc[0]):>12}")
    print(f"{'Train R²':<22} {fmt(e['train_r2'].iloc[0]):>12} {fmt(r['train_r2'].iloc[0]):>12}")
    print(f"{'Train MAE':<22} {fmt(e['train_mae'].iloc[0]):>12} {fmt(r['train_mae'].iloc[0]):>12}")
    print(f"{'Mean |residual|':<22} {ep['residual_attain'].abs().mean():>12.4f} {rp['residual_attain'].abs().mean():>12.4f}")

    def llm_shap(df):
        row = df[df["feature"] == "llm_probability"]["mean_abs_shap"]
        return row.iloc[0] if len(row) else 0.0

    e_llm = llm_shap(ei)
    r_llm = llm_shap(ri)
    e_llm_str = f"{e_llm:.6f}" if e_llm > 1e-8 else "zeroed"
    r_llm_str = f"{r_llm:.6f}" if r_llm > 1e-8 else "zeroed"
    print(f"{'LLM |SHAP|':<22} {e_llm_str:>12} {r_llm_str:>12}")

    print("\nTop 5 features:")
    print(f"  {'ElasticNet':<35}  {'Ridge':<35}")
    print(f"  {'-'*35}  {'-'*35}")
    for i in range(5):
        ef = ei.iloc[i]["feature"] if i < len(ei) else ""
        rf = ri.iloc[i]["feature"] if i < len(ri) else ""
        print(f"  {ef:<35}  {rf:<35}")

    print("\nVerdict:")
    e_res = ep["residual_attain"].abs().mean()
    r_res = rp["residual_attain"].abs().mean()
    winner = "Ridge" if r_res < e_res else "ElasticNet"
    print(f"  {winner} wins on residuals  (elastic={e_res:.4f}  ridge={r_res:.4f})")
    if e_llm > 1e-8 or r_llm > 1e-8:
        llm_winner = "Ridge" if r_llm > e_llm else "ElasticNet"
        print(f"  {llm_winner} has stronger LLM signal  (elastic={e_llm:.6f}  ridge={r_llm:.6f})")
    else:
        print("  LLM signal zeroed in both — increase LLM prediction variance")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("DECIDR SYSTEM 2 — PoC RUNNER v2")
    print(f"Variants: {' + '.join(v['name'] for v in VARIANTS)}")
    print("=" * 70)

    print("\n--- SHARED PIPELINE ---")
    for s in SHARED:
        run(s)

    for v in VARIANTS:
        print(f"\n--- VARIANT: {v['name']} ---")
        run(v["meta"], v["name"])
        run(v["shap"], v["name"], model_env=v["env"])
        run(v["demo"], v["name"])
        if len(VARIANTS) > 1:
            save_outputs(v["name"])

    if len(VARIANTS) == 2:
        compare()

    print("\nDone. Output files:")
    suffix = ["_elastic", "_ridge"] if len(VARIANTS) == 2 else [""]
    for s in suffix:
        print(f"  demo_dashboard_poc{s}.png")
        print(f"  shap_summary_poc{s}.png")