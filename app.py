"""
app.py — Decidr Coherence Engine  (System 1 — Streamlit UI)
Entry point for:  streamlit run app.py

Folder layout:
    System_1/   UI components (viz, schemas, goal_extractor, dashboard)
    System_2/   Intelligence core (meta_learner, score_goal, verify_goal, composite_score)
    System_3/   Data layer (data_loader, features, embeddings, pipeline)
"""

import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("System_1", "System_2", "System_3"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Credentials: .env locally → st.secrets on Streamlit Cloud ────────────────
# Locally:        create a .env file in the project root (never commit it)
# Streamlit Cloud: App Settings → Secrets → paste your key=value pairs
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)   # loads .env into os.environ if present
except ImportError:
    pass

try:
    import streamlit as _st_sec
    for _k, _v in _st_sec.secrets.items():
        if _k not in os.environ:
            os.environ[_k] = str(_v)
    del _st_sec, _k, _v
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

 
import os, json, time
from datetime import datetime
from pathlib import Path
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
 
from schemas import (
    GoalInput, System3Payload, BUCKET_HIERARCHY, METRIC_UNITS,
    SCENARIOS, SHOCK_TYPES, DIMS, WEIGHTS, get_l2_for_l1, get_l3_for_l2,
)
from goal_extractor import GoalExtractor
import viz
 
st.set_page_config(
    page_title="Decidr Coherence Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ═════════════════════════════════════════════════════════════════════════════
#  THEME — slate/blue base + input-half component extensions
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0b0f1e 0%, #0f1228 100%);
    color: #E5E7EB;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 1.2rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1228 0%, #1a0f1f 100%);
    border-right: 1px solid rgba(168, 85, 247, 0.12);
}
[data-testid="stSidebar"] * { color: #E5E7EB; }
 
/* ── Decidr brand accents (purple → orange gradient on hero & buttons) ──── */
.brand-bar {
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 50%, #f97316 100%);
    border-radius: 2px;
    margin-bottom: 0;
}
.decidr-mark {
    display: inline-block;
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #8b5cf6, #f97316);
    border-radius: 8px;
    vertical-align: middle;
    margin-right: 10px;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.35);
}
 
/* ── Hero panels — purple-tinted dark ────────────────────────────────────── */
.hero-panel {
    background: linear-gradient(135deg, rgba(45,30,80,0.55) 0%, rgba(15,18,40,0.95) 100%);
    border: 1px solid rgba(168, 85, 247, 0.18);
    border-radius: 20px;
    padding: 24px 26px 18px 26px;
    margin-bottom: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.32);
}
.hero-title {
    font-size: 2.8rem; font-weight: 800; color: #F8FAFC;
    margin-bottom: 6px; letter-spacing: -0.03em;
}
.hero-subtitle { color: #94A3B8; font-size: 1rem; margin-bottom: 0.2rem; }
 
/* ── KPI / metric cards ──────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(180deg, rgba(17,24,39,0.98) 0%, rgba(15,23,42,0.98) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.22);
    min-height: 128px;
}
.metric-label { font-size: 0.9rem; font-weight: 600; margin-bottom: 10px; }
.metric-value { font-size: 2rem; font-weight: 750; line-height: 1.05; color: #F9FAFB; margin-bottom: 8px; }
.metric-sub   { font-size: 0.82rem; color: #9CA3AF; }
 
/* ── Input-half components (badge, step labels, goal preview cards) ──────── */
.badge {
    display: inline-block; padding: 6px 14px; border-radius: 20px;
    background: rgba(76,114,176,0.14); border: 1px solid rgba(76,114,176,0.35);
    font-size: 11px; font-weight: 700; color: #93C5FD;
    letter-spacing: 0.08em; margin-bottom: 12px;
}
.step-label {
    font-size: 11px; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 22px 0 10px 0;
}
.goal-card {
    background: rgba(17,24,39,0.9); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 20px; margin-bottom: 12px;
}
.goal-card-glow {
    background: linear-gradient(135deg, rgba(76,114,176,0.08), rgba(17,24,39,0.95));
    border: 1px solid rgba(76,114,176,0.35);
    border-radius: 14px; padding: 20px; margin-bottom: 12px;
}
.goal-card-hl {
    background: linear-gradient(135deg, rgba(76,114,176,0.14), rgba(17,24,39,0.92));
    border: 2px solid rgba(76,114,176,0.5);
    border-radius: 14px; padding: 20px; margin-bottom: 12px;
}
 
/* ── Scope pills ─────────────────────────────────────────────────────────── */
.scope-badge {
    display: inline-block; padding: 3px 11px; border-radius: 12px;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em;
}
.scope-goal { background: rgba(59,130,246,0.15); color: #93C5FD; border: 1px solid rgba(59,130,246,0.3); }
.scope-l2   { background: rgba(221,132,82,0.15); color: #FCD34D; border: 1px solid rgba(221,132,82,0.3); }
.scope-l1   { background: rgba(244,114,182,0.15); color: #F9A8D4; border: 1px solid rgba(244,114,182,0.3); }
 
/* ── Dimension bars (used on output page) ────────────────────────────────── */
.bar-wrap { height: 6px; background: rgba(148,163,184,0.15); border-radius: 3px; margin: 8px 0; overflow: hidden; }
.bar-fill { height: 6px; border-radius: 3px; }
 
/* ── Word counter ────────────────────────────────────────────────────────── */
.wc { text-align: right; font-size: 11px; color: #64748B; margin-top: -6px; }
 
/* ── Processing dots ─────────────────────────────────────────────────────── */
@keyframes pulse { 0%,100% { opacity: 0.3 } 50% { opacity: 1 } }
.dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    background: #4C72B0; margin: 0 4px;
}
.dot:nth-child(1) { animation: pulse 1.4s infinite 0s; }
.dot:nth-child(2) { animation: pulse 1.4s infinite 0.2s; }
.dot:nth-child(3) { animation: pulse 1.4s infinite 0.4s; }
 
/* ── Streamlit widget overrides ──────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.96rem; font-weight: 600;
    border-radius: 12px 12px 0 0; padding: 10px 16px;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; overflow: hidden;
}
.stProgress > div > div > div { background: linear-gradient(90deg, #8b5cf6, #ec4899, #f97316) !important; }
button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 70%, #f97316 100%) !important;
    border: none !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 18px rgba(168, 85, 247, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar toggle ────────────────────────────────────────────────────────────
# ── Sidebar toggle ────────────────────────────────────────────────────────────
# Fixed floating ☰ button in top-left corner. Always visible even when the
# sidebar is hidden. Clicking it toggles session_state and reruns the page.



 
# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """Load pipeline CSVs. Tries canonical names, then flexible column detection."""
    data = {}
    mapping = {}     # slot → filename
    inventory = {}   # filename → list of columns (for diagnostic display)
 
    def _try_read(path):
        try: return pd.read_csv(path)
        except Exception: return None
 
    # Stage 0: collect every CSV and its columns (for diagnostic display)
    csv_files = []
    for folder in (".", "data"):
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.endswith(".csv"):
                    p = os.path.join(folder, f)
                    csv_files.append(p)
                    try:
                        peek = pd.read_csv(p, nrows=0)
                        inventory[f] = list(peek.columns)
                    except Exception:
                        inventory[f] = ["<could not read>"]
 
    # Stage 1: canonical filenames first
    canonical = {
        "composite"   : "composite_scores_poc.csv",
        "coherence_ts": "coherence_timeseries_poc.csv",
        "portfolio_ts": "portfolio_timeseries_poc.csv",
        "forward_proj": "forward_projection_poc.csv",
        "portfolio"   : "portfolio_summary_poc.csv",
        "goals"       : "goals.csv",
        "buckets"     : "buckets.csv",
    }
    for key, fname in canonical.items():
        for folder in (".", "data"):
            p = os.path.join(folder, fname)
            if os.path.exists(p):
                df = _try_read(p)
                if df is not None:
                    data[key] = df
                    mapping[key] = fname
                break
 
    # Helper: case-insensitive column presence check with aliases
    def _has(cols_lower, *names):
        return any(n.lower() in cols_lower for n in names)
 
    # Flexible signatures. Each returns True if df looks like the right kind.
    # Goal: catch common renamings — uppercase, _score suffix, "overall" for composite, etc.
    def _is_composite(df):
        cols = {c.lower() for c in df.columns}
        has_id  = _has(cols, "goal_id", "goalid", "id", "goal")
        has_dim = _has(cols, "coherence", "coherence_score", "coh", "coh_score",
                              "attainability", "attainability_score", "attain", "attain_score")
        has_score = _has(cols, "final_composite", "composite", "composite_score", "composite_adjusted",
                                "overall", "overall_score", "final_score", "score",
                                "meta_score", "prediction", "pred_score", "y_pred")
        # Must have an id + at least one dim OR a composite-like score
        return has_id and (has_dim or has_score)
 
    def _is_coherence_ts(df):
        cols = {c.lower() for c in df.columns}
        has_period = _has(cols, "period_id", "period", "t", "step")
        has_avg    = _has(cols, "avg_composite", "mean_composite", "composite",
                                "avg_score", "mean_score")
        # Period-level aggregate — should NOT have a per-bucket dimension
        not_bucket = not _has(cols, "l2_name", "l2", "bucket_name", "bucket")
        return has_period and has_avg and not_bucket
 
    def _is_portfolio_ts(df):
        cols = {c.lower() for c in df.columns}
        has_period = _has(cols, "period_id", "period")
        has_bucket = _has(cols, "l2_name", "l2", "bucket_name", "bucket")
        has_avg    = _has(cols, "avg_composite", "mean_composite", "composite")
        return has_period and has_bucket and has_avg
 
    def _is_forward_proj(df):
        cols = {c.lower() for c in df.columns}
        return _has(cols, "composite_p6", "composite_p12", "composite_adjusted",
                          "improving_p6", "degrading_p6", "projection", "projected",
                          "pred_p6", "forecast")
 
    def _is_portfolio_summary(df):
        cols = {c.lower() for c in df.columns}
        has_bucket = _has(cols, "l2_name", "l2", "bucket_name", "bucket")
        has_avg    = _has(cols, "avg_composite", "mean_composite", "avg_score")
        no_period  = not _has(cols, "period_id", "period")
        return has_bucket and has_avg and no_period
 
    checks = {
        "composite"   : _is_composite,
        "coherence_ts": _is_coherence_ts,
        "portfolio_ts": _is_portfolio_ts,
        "forward_proj": _is_forward_proj,
        "portfolio"   : _is_portfolio_summary,
    }
 
    # Stage 2: file-name-priority hints per slot. Filename tokens boost score so
    # "analytical_flat.csv" beats "features_raw_p6.csv" for composite scores.
    priority_tokens = {
        "composite"   : ["analytical", "meta_learner_results", "meta_learner_predictions",
                         "outputs", "composite_scores", "scores"],
        "coherence_ts": ["coherence_timeseries", "periods", "period_timeseries", "timeseries"],
        "portfolio_ts": ["portfolio_timeseries", "bucket_timeseries", "l2_timeseries"],
        "forward_proj": ["forward_projection", "projections", "projection", "forecast"],
        "portfolio"   : ["portfolio_summary", "bucket_summary", "portfolio"],
    }
 
    def _priority_score(slot, fname):
        """Higher score = more likely the right file for this slot."""
        f = fname.lower()
        for i, token in enumerate(priority_tokens.get(slot, [])):
            if token in f:
                return 1000 - i  # earlier tokens rank higher
        return 0
 
    # For each slot not yet filled, rank candidate files and pick the best
    for slot, check_fn in checks.items():
        if slot in data:
            continue
        candidates = []
        for path in csv_files:
            df_peek = _try_read(path)
            if df_peek is None:
                continue
            try:
                if check_fn(df_peek):
                    fname = os.path.basename(path)
                    candidates.append((_priority_score(slot, fname), path, df_peek))
            except Exception:
                continue
        if candidates:
            # Highest priority, then fullest (most rows) wins
            candidates.sort(key=lambda t: (-t[0], -len(t[2])))
            _, best_path, best_df = candidates[0]
            data[slot]    = best_df
            mapping[slot] = os.path.basename(best_path)
 
    # Normalise column names to what the dashboard expects (module-level helper)
    for slot in list(data.keys()):
        if slot in _RENAME_MAPS:
            data[slot] = _normalise_cols(data[slot], slot)
 
    data["_mapping"]   = mapping
    data["_inventory"] = inventory
    return data
 
 
# Column alias maps: "what we might see" → "what the charts expect".
# Hoisted to module scope so the manual-override path in the portfolio page
# can normalise user-picked files the same way the auto-detector does.
_RENAME_MAPS = {
    "composite": {
        "goalid": "goal_id", "goal": "goal_id", "id": "goal_id",
        "period": "period_id", "t": "period_id",
        "composite_score": "composite", "overall": "composite",
        "overall_score": "composite", "final_score": "composite",
        "meta_score": "composite", "prediction": "composite", "y_pred": "composite",
        "coherence_score": "coherence", "coh_score": "coherence", "coh": "coherence",
        "attainability_score": "attainability", "attain_score": "attainability", "attain": "attainability",
        "relevance_score": "relevance",
        "integrity_score": "integrity",
        "l1": "l1_name", "l2": "l2_name", "bucket_name": "l2_name",
    },
    "coherence_ts": {
        "period": "period_id", "t": "period_id",
        "composite": "avg_composite", "mean_composite": "avg_composite",
        "avg_score": "avg_composite", "mean_score": "avg_composite",
    },
    "portfolio_ts": {
        "period": "period_id",
        "l2": "l2_name", "bucket_name": "l2_name",
        "composite": "avg_composite", "mean_composite": "avg_composite",
    },
    "portfolio": {
        "l2": "l2_name", "bucket_name": "l2_name",
        "mean_composite": "avg_composite", "avg_score": "avg_composite",
    },
}
 
def _normalise_cols(df, slot):
    """Rename columns in `df` according to _RENAME_MAPS[slot]."""
    renames = _RENAME_MAPS.get(slot, {})
    if not renames or df is None:
        return df
    actual = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for alias, canon in renames.items():
        if alias in lower_cols and canon not in df.columns:
            actual[lower_cols[alias]] = canon
    return df.rename(columns=actual) if actual else df
 
DATA = load_data()
 
COLORS = {
    "coherence"    : "#4C72B0",
    "attainability": "#DD8452",
    "relevance"    : "#55A868",
    "integrity"    : "#C44E52",
    "composite"    : "#8172B2",
    "at_risk"      : "#C44E52",
    "safe"         : "#4C72B0",
    "budget_shock" : "#FF6B35",
    "market_shock" : "#9B2335",
}
 
# Demo score payload — replaced by System 3 once it's wired in.
DEMO_SCORE = {
    "goal_id": 1, "attainability": 0.6823, "relevance": 0.7145,
    "coherence": 0.5891, "integrity": 0.6234, "overall": 0.6523,
    "gp_mean": 0.7012, "gp_std": 0.0834, "gp_weight": 0.706, "llm_weight": 0.294,
    "baseline": 0.65, "uncertain": False, "n_llm_ok": 3, "status": "ok",
    "reasoning": {
        "attainability": "Trailing slope positive at 0.042/period. Projects to 89% of target.",
        "relevance":     "5.6% of parent allocation, within optimal band. Strong priority alignment.",
        "coherence":     "L3 proportional to L2 parent. Minor drift periods 8–10 but stabilised.",
        "integrity":     "Output quality 0.78 aligns with inputs. Needle move ratio 0.85.",
    },
    "ensemble_meta": {
        "attainability": {"gp_weight": 0.706, "llm_weight": 0.294, "gp_std": 0.0834},
        "relevance":     {"rule_weight": 0.528, "llm_weight": 0.472, "variance": 0.0021},
        "coherence":     {"rule_weight": 0.612, "llm_weight": 0.388, "variance": 0.0089},
        "integrity":     {"rule_weight": 0.491, "llm_weight": 0.509, "variance": 0.0015},
    },
    "forward": {
        "p6_projected" : 0.7012, "p6_lower" : 0.6178, "p6_upper" : 0.7846,
        "p12_projected": 0.7341, "p12_lower": 0.6234, "p12_upper": 0.8448,
        "p18_projected": 0.7589, "p18_lower": 0.6112, "p18_upper": 0.9067,
        "trajectory_class": "improving",   # improving / stable / degrading
        "confidence_p6": 0.83,
        "expected_target_attainment": 0.89,
    },
    "shock": {
        "pre_shock"    : 0.6823,
        "budget_shock" : 0.5142,
        "market_shock" : 0.4378,
        "post_shock"   : 0.6512,
        "recovery_periods": 4,
        "vulnerability_class": "moderate",   # resilient / moderate / vulnerable
        "vs_portfolio_avg_drop": "+8% better than portfolio mean",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
#  LIVE SCORING — wires submitted goal into score_goal.py
# ═════════════════════════════════════════════════════════════════════════════
def _live_score_goal(l2_bucket: str, current_period: int = 18) -> dict | None:
    """
    Find the closest matching goal row for the submitted L2 bucket and run
    score_goal(). Returns the score dict on success, None on any failure.
    The caller falls back to DEMO_SCORE if None is returned.
    """
    try:
        from score_goal import score_goal as _sg, merge_extra_columns
        import pandas as pd

        _root = os.path.dirname(os.path.abspath(__file__))
        goals_df = pd.read_csv(os.path.join(_root, "goals.csv"))
        buckets_df = pd.read_csv(os.path.join(_root, "buckets.csv"))

        # Find L2 bucket_id that matches the submitted bucket name
        l2_row = buckets_df[
            (buckets_df["bucket_level"] == 2) &
            (buckets_df["bucket_name"].str.lower() == l2_bucket.lower())
        ]
        if len(l2_row) == 0:
            # Fuzzy fallback — partial match
            l2_row = buckets_df[
                (buckets_df["bucket_level"] == 2) &
                (buckets_df["bucket_name"].str.lower().str.contains(l2_bucket.lower()[:8]))
            ]
        if len(l2_row) == 0:
            return None
        l2_id = l2_row.iloc[0]["bucket_id"]

        # Find L3 buckets under this L2
        l3_ids = buckets_df[
            (buckets_df["bucket_level"] == 3) &
            (buckets_df["parent_bucket_id"] == l2_id)
        ]["bucket_id"].tolist()

        # Find goals in those L3 buckets
        matching_goals = goals_df[goals_df["bucket_id"].isin(l3_ids)]
        if len(matching_goals) == 0:
            return None
        goal_row_idx = int(matching_goals.index[0])
        goal_id = int(matching_goals.iloc[0]["goal_id"])

        # Load features and rule scores for this period
        feat_file = os.path.join(_root, f"features_raw_p{current_period}.csv")
        rule_file = os.path.join(_root, f"rule_scores_p{current_period}.csv")
        if current_period == 12:
            feat_file = os.path.join(_root, "features_raw_poc.csv")
            rule_file = os.path.join(_root, "rule_scores_poc.csv")

        if not os.path.exists(feat_file) or not os.path.exists(rule_file):
            return None

        feat_df = pd.read_csv(feat_file)
        rule_df = pd.read_csv(rule_file)

        # Find goal row by goal_id if column exists, else use positional index
        if "goal_id" in feat_df.columns:
            feat_rows = feat_df[feat_df["goal_id"] == goal_id]
            rule_rows = rule_df[rule_df["goal_id"] == goal_id] if "goal_id" in rule_df.columns else rule_df.iloc[:1]
            if len(feat_rows) == 0:
                return None
            goal_feat_row = feat_rows.iloc[0]
            goal_rule_row = rule_rows.iloc[0] if len(rule_rows) > 0 else rule_df.iloc[0]
        else:
            if goal_row_idx >= len(feat_df):
                return None
            goal_feat_row = feat_df.iloc[goal_row_idx]
            goal_rule_row = rule_df.iloc[goal_row_idx] if goal_row_idx < len(rule_df) else rule_df.iloc[0]

        # Merge extra signal columns (shock features etc.)
        import pandas as _pd
        feat_single = _pd.DataFrame([goal_feat_row])
        if os.path.exists(os.path.join(_root, f"period_{current_period}_poc.csv")):
            period_ref = _pd.read_csv(os.path.join(_root, f"period_{current_period}_poc.csv"))
            feat_single = merge_extra_columns(feat_single, period_ref)
        feat_row = feat_single.iloc[0]

        result = _sg(feat_row, goal_rule_row, verbose=False, current_period=current_period)
        result["goal_id"] = goal_id
        return result

    except Exception as e:
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  PER-GOAL VIEW — reusable component for all tabs and the output page
# ═════════════════════════════════════════════════════════════════════════════
def _render_per_goal_view(s: dict, key_prefix: str = "pg"):
    """Render the full per-goal coherence view for a score result dict."""
    if not s:
        return

    ov = s.get("overall", s.get("composite", 0))
    c  = score_color(ov)
    conf_line = "High Confidence" if not s.get("uncertain", False) else "⚠️ Elevated Uncertainty"
    gp_std = s.get("gp_std", 0)
    n_llm  = s.get("n_llm_ok", 0)

    # Headline card
    st.markdown(f"""
    <div class="goal-card-hl">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px">
            <div>
                <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:0.08em">
                    Goal {s.get('goal_id','?')} · Composite Coherence
                </div>
                <div style="font-size:42px;font-weight:700;color:{c};line-height:1">{ov:.1%}</div>
                <div style="font-size:12px;color:#94A3B8;margin-top:4px">
                    {conf_line} · σ={gp_std:.4f}
                </div>
            </div>
            <div style="text-align:right">
                <div style="font-size:11px;color:#64748B">Status</div>
                <div style="font-size:20px;font-weight:700;color:{c}">{risk_label(ov)}</div>
                <div style="font-size:12px;color:#64748B;margin-top:4px">LLMs: {n_llm}/2</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dimension bars
    dim_cols = st.columns(4)
    for i, d in enumerate(DIMS):
        v = s.get(d, 0); c2 = score_color(v)
        with dim_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="color:{c2}">{d.capitalize()}</div>
                <div class="metric-value" style="color:{c2}">{v:.1%}</div>
                <div class="bar-wrap">
                    <div class="bar-fill" style="width:{v*100:.0f}%;background:{c2}"></div>
                </div>
                <div class="metric-sub">25% weight</div>
            </div>
            """, unsafe_allow_html=True)

    # Radar + ranked bar
    br1, br2 = st.columns(2)
    with br1:
        cats = [d.capitalize() for d in DIMS]
        vals = [s.get(d, 0) for d in DIMS]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name="This goal",
            line=dict(color=COLORS["composite"], width=2.5),
            fillcolor="rgba(129,114,178,0.25)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.35] * (len(cats) + 1), theta=cats + [cats[0]],
            name="Risk threshold",
            line=dict(color=COLORS["at_risk"], width=1.5, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(radialaxis=dict(range=[0, 1])),
            title="Dimension Radar", height=340,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_radar")

    with br2:
        sorted_dims = sorted(DIMS, key=lambda d: s.get(d, 0))
        fig = go.Figure(go.Bar(
            x=[s.get(d, 0) for d in sorted_dims],
            y=[d.capitalize() for d in sorted_dims],
            orientation="h",
            marker_color=[score_color(s.get(d, 0)) for d in sorted_dims],
            text=[f"{s.get(d,0):.1%}" for d in sorted_dims],
            textposition="outside",
        ))
        fig.add_vline(x=0.35, line_dash="dash", line_color=COLORS["at_risk"],
                      annotation_text="Risk threshold")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_range=[0, 1.05], title="Dimensions Ranked",
            height=340, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_ranked")

    # Reasoning cards
    reasoning = s.get("reasoning", {})
    if reasoning:
        st.markdown('<div class="step-label">Why these scores?</div>', unsafe_allow_html=True)
        for d in DIMS:
            v = s.get(d, 0); c2 = score_color(v)
            r = reasoning.get(d, "")
            if r:
                st.markdown(f"""
                <div class="goal-card" style="margin-bottom:8px">
                    <div style="display:flex;justify-content:space-between;align-items:baseline">
                        <span style="font-size:14px;font-weight:600;color:#F8FAFC">{d.capitalize()}</span>
                        <span style="font-size:18px;font-weight:700;color:{c2}">{v:.1%}</span>
                    </div>
                    <div class="bar-wrap">
                        <div class="bar-fill" style="width:{v*100:.0f}%;background:{c2}"></div>
                    </div>
                    <div style="font-size:12px;color:#E5E7EB;border-left:3px solid {c2};
                                padding-left:10px;margin-top:8px">{r}</div>
                </div>
                """, unsafe_allow_html=True)


def _per_goal_selector(comp_view, fwd, key_prefix, title="Drill into a goal", comp_full=None):
    """Per-goal detail view. comp_full = full comp with all periods for period-aware lookup."""
    if comp_view is None or len(comp_view) == 0 or "goal_id" not in comp_view.columns:
        return

    st.divider()
    st.markdown("""
    <div style="background:rgba(30,39,97,0.35);border:1px solid rgba(147,197,253,0.2);
                border-radius:12px;padding:14px 18px 10px 18px;margin-bottom:4px">
        <div style="font-size:11px;font-weight:700;color:#93C5FD;letter-spacing:0.08em;
                    text-transform:uppercase;margin-bottom:10px">
            🔍 Per-Goal Detail View
        </div>
    """, unsafe_allow_html=True)

    _all_ids = sorted(comp_view["goal_id"].dropna().unique().tolist())
    _labels = []
    for gid in _all_ids:
        row = comp_view[comp_view["goal_id"] == gid].iloc[0]
        l2 = row.get("l2_name", "") if "l2_name" in row.index else ""
        _labels.append(f"G{int(gid)} — {l2}" if l2 else f"G{int(gid)}")

    _gc1, _gc2 = st.columns([3, 1])
    with _gc1:
        _sel_label = st.selectbox("Select goal", _labels, key=f"{key_prefix}_goal_sel")
    with _gc2:
        _src = comp_full if comp_full is not None and "period_id" in comp_full.columns else comp_view
        _avail_p = sorted(_src["period_id"].dropna().unique().tolist()) if "period_id" in _src.columns else [18]
        _def_idx = _avail_p.index(18) if 18 in _avail_p else len(_avail_p) - 1
        _sel_p   = st.selectbox("Period", _avail_p, index=_def_idx, key=f"{key_prefix}_period_sel")

    st.markdown("</div>", unsafe_allow_html=True)

    _sel_gid = int(_sel_label.split("—")[0].strip()[1:])
    _lookup  = comp_full if comp_full is not None else comp_view
    if "period_id" in _lookup.columns:
        _pr = _lookup[(_lookup["goal_id"] == _sel_gid) & (_lookup["period_id"] == _sel_p)]
        _row = _pr.iloc[0] if len(_pr) > 0 else comp_view[comp_view["goal_id"] == _sel_gid].iloc[0]
    else:
        _row = comp_view[comp_view["goal_id"] == _sel_gid].iloc[0]

    # NaN-safe getter — verified_ columns are only populated at snapshot periods (6/12/18)
    # At other periods (e.g. p24) fall back cleanly to non-verified columns
    def _safe(row, *keys, default=0.0):
        import math
        for k in keys:
            v = row.get(k, None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
        return default

    _cv = _safe(_row, "final_composite", "verified_composite", "composite_adjusted", "overall")
    _s = {
        "goal_id"      : _sel_gid,
        "overall"      : _cv,
        "attainability": _safe(_row, "verified_attainability", "attainability"),
        "coherence"    : _safe(_row, "verified_coherence",     "coherence"),
        "relevance"    : _safe(_row, "verified_relevance",     "relevance"),
        "integrity"    : _safe(_row, "verified_integrity",     "integrity"),
        "gp_std"       : _safe(_row, "gp_std"),
        "uncertain"    : bool(_row.get("uncertain", False)),
        "n_llm_ok"     : 2,
        "reasoning"    : {},
        "period_shown" : _sel_p,
    }

    # ── Full trajectory chart: p1-p18 history + p24/p30 projection ──────
    if fwd is not None and "goal_id" in fwd.columns:
        _frow = fwd[fwd["goal_id"] == _sel_gid]
        if len(_frow) > 0:
            _fr = _frow.iloc[0]
            _hist_df = None
            if comp_full is not None and "period_id" in comp_full.columns and "goal_id" in comp_full.columns:
                _hist_df = comp_full[comp_full["goal_id"] == _sel_gid].sort_values("period_id")

            _p24_dim = {
                "coherence"    : float(_fr.get("coherence_p6",  _s["coherence"])),
                "attainability": float(_fr.get("attain_p6",     _s["attainability"])),
                "relevance"    : float(_fr.get("relevance_p6",  _s["relevance"])),
                "integrity"    : float(_fr.get("integrity_p6",  _s["integrity"])),
            }
            _p30_dim = {
                "coherence"    : float(_fr.get("coherence_p12", _p24_dim["coherence"])),
                "attainability": float(_fr.get("attain_p12",    _p24_dim["attainability"])),
                "relevance"    : float(_fr.get("relevance_p12", _p24_dim["relevance"])),
                "integrity"    : float(_fr.get("integrity_p12", _p24_dim["integrity"])),
            }
            _comp_p24 = float(_fr.get("composite_p6",  _cv))
            _comp_p30 = float(_fr.get("composite_p12", _cv))

            import math as _math
            _yfig = go.Figure()

            # Shock bands
            _yfig.add_vrect(x0=9.5, x1=12.5, fillcolor=COLORS["budget_shock"],
                            opacity=0.12, layer="below", line_width=0,
                            annotation_text="Budget shock", annotation_position="top left",
                            annotation_font_size=9)
            _yfig.add_vrect(x0=13.5, x1=17.5, fillcolor=COLORS["market_shock"],
                            opacity=0.10, layer="below", line_width=0,
                            annotation_text="Market shock", annotation_position="top right",
                            annotation_font_size=9)

            # 4 dimension traces: one continuous line each (solid history, dashed proj)
            for _dim in DIMS:
                _hx, _hy = [], []
                if _hist_df is not None and _dim in _hist_df.columns:
                    for _, _hr in _hist_df.iterrows():
                        _v = float(_hr[_dim])
                        if not _math.isnan(_v):
                            _hx.append(int(_hr["period_id"]))
                            _hy.append(_v)
                _anchor_v = _hy[-1] if _hy else _s.get(_dim, 0)
                if _hx:
                    _yfig.add_trace(go.Scatter(
                        x=_hx, y=_hy, mode="lines",
                        name=_dim.capitalize(),
                        line=dict(color=COLORS[_dim], width=1.2, dash="dot"),
                        opacity=0.55, showlegend=True,
                    ))
                _yfig.add_trace(go.Scatter(
                    x=[_hx[-1] if _hx else 18, 24, 30],
                    y=[_anchor_v, _p24_dim[_dim], _p30_dim[_dim]],
                    mode="lines+markers",
                    line=dict(color=COLORS[_dim], width=1.5, dash="dash"),
                    marker=dict(size=5), opacity=0.75, showlegend=False,
                    name=_dim.capitalize(),
                ))

            # Composite history
            _cx, _cy = [], []
            if _hist_df is not None:
                _ccol = (
                    "final_composite" if "final_composite" in _hist_df.columns
                    else "verified_composite" if "verified_composite" in _hist_df.columns
                    else "composite_adjusted" if "composite_adjusted" in _hist_df.columns
                    else "composite"
                )
                if _ccol in _hist_df.columns:
                    for _, _hr in _hist_df.iterrows():
                        _v = float(_hr[_ccol])
                        if not _math.isnan(_v):
                            _cx.append(int(_hr["period_id"]))
                            _cy.append(_v)
            _base_c = _cy[-1] if _cy else _cv
            if _cx:
                _yfig.add_trace(go.Scatter(
                    x=_cx, y=_cy, mode="lines",
                    name="Composite", line=dict(color=COLORS["composite"], width=3),
                ))
            # Composite projection
            _yfig.add_trace(go.Scatter(
                x=[_cx[-1] if _cx else 18, 24, 30],
                y=[_base_c, _comp_p24, _comp_p30],
                mode="lines+markers", name="Composite",
                line=dict(color=COLORS["composite"], width=3, dash="dash"),
                marker=dict(size=10,
                            color=[score_color(v) for v in [_base_c, _comp_p24, _comp_p30]],
                            line=dict(color="#F8FAFC", width=1.5)),
                showlegend=False,
            ))

            _yfig.add_vline(x=18, line_dash="dash", line_color="#93C5FD",
                            opacity=0.5, annotation_text="p18 anchor",
                            annotation_position="top", annotation_font_size=9)
            _yfig.add_hline(y=0.35, line_dash="dash", line_color=COLORS["at_risk"],
                            annotation_text="Risk threshold",
                            annotation_position="top right", annotation_font_size=9)

            _all_v = _cy + [_comp_p24, _comp_p30] + list(_p24_dim.values()) + list(_p30_dim.values())
            _yfig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title=f"G{_sel_gid} — History p1-p18 (dotted) · Projection p24/p30 (dashed)",
                xaxis=dict(title="Period", tickmode="linear", tick0=1, dtick=3),
                yaxis=dict(title="Score",
                           range=[max(0.0, min(_all_v)-0.06), min(1.0, max(_all_v)+0.06)]),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(size=10)),
                hovermode="x unified",
            )
            st.plotly_chart(_yfig, use_container_width=True, key=f"{key_prefix}_fwd")

    _render_per_goal_view(_s, key_prefix=f"{key_prefix}_{_sel_gid}_{_sel_p}")



_DEFAULTS = {
    "page": "portfolio", "nl_input": "", "extracted": None,
    "show_fields": False, "show_preview": False,
    "l1": "", "l2": "", "l3": "", "metric_name": "", "metric_unit": "score",
    "target_value": "", "initial_value": "", "periods": 24,
    "scenario": "optimal", "scope": "goal", "shock_type": "none",
    "payload_json": None, "goal_history": [],
    "score_result": None, "feedback": None,
    "aws_region": os.getenv("AWS_REGION", "us-west-2"),
    "bedrock_model_id": os.getenv(
        "BEDROCK_MODEL_ID",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    ),
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

def _make_extractor():
    """Fresh extractor per call - uses AWS Bedrock Claude Haiku."""
    return GoalExtractor(
        model_id=st.session_state.get(
            "bedrock_model_id",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        ),
        region_name=st.session_state.get("aws_region", "us-west-2"),
        use_llm=True,
    )

OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

def metric_card(title, value, subtitle="", color="#4C72B0"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:{color};">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def score_color(s):
    return "#4ade80" if s >= 0.5 else "#fbbf24" if s >= 0.35 else "#f87171"
 
def risk_label(s):
    return "Critical" if s < 0.2 else "At Risk" if s < 0.35 else "On Track"

def bucket_path():
    p = st.session_state.l1
    if st.session_state.l2: p += f" → {st.session_state.l2}"
    if st.session_state.l3: p += f" → {st.session_state.l3}"
    return p

def build_payload():
    ext = st.session_state.extracted or {}
    g = GoalInput(
        goal_title=ext.get("goal_title", ""),
        scope=st.session_state.scope,
        bucket_l1=st.session_state.l1,
        bucket_l2=st.session_state.l2,
        bucket_l3=st.session_state.l3 or None,
        metric_name=st.session_state.metric_name,
        metric_unit=st.session_state.metric_unit,
        target_value=float(st.session_state.target_value) if st.session_state.target_value else None,
        initial_value=float(st.session_state.initial_value) if st.session_state.initial_value else None,
        periods=st.session_state.periods,
        scenario_story=st.session_state.scenario,
        shock_type=st.session_state.shock_type,
    )
    return System3Payload(
        goal=g, scope=st.session_state.scope,
        raw_nl_input=st.session_state.nl_input,
    )

def reset_input():
    for k in ("nl_input", "l1", "l2", "l3", "metric_name", "target_value", "initial_value"):
        st.session_state[k] = ""
    st.session_state.update(
        metric_unit="score", periods=24, scenario="optimal", scope="goal",
        extracted=None, show_fields=False, show_preview=False,
        payload_json=None, score_result=None, feedback=None, page="input",
    )

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR - nav + filters (filters only shown on portfolio page)
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;margin-bottom:6px">
        <span class="decidr-mark"></span>
        <span style="font-size:1.2rem;font-weight:700;letter-spacing:-0.02em;color:#F8FAFC">
            Decidr Coherence Engine
        </span>
    </div>
    <div class="brand-bar"></div>
    """, unsafe_allow_html=True)
    st.markdown("**Team 14-02 · iLab Capstone 36127**")
    st.divider()

    # Nav is only visible in the "stable" pages - processing/output hide it
    # so the user follows the flow without side-stepping mid-scoring.
    # Order: Dashboard (core), Goal Input (add-on), About (context for non-tech viewers).
    if st.session_state.page in ("input", "portfolio", "about"):
        nav_labels = ["📊 Portfolio Dashboard", "🎯 Goal Input", "ℹ️ About"]
        nav_pages  = ["portfolio", "input", "about"]
        try:
            current_idx = nav_pages.index(st.session_state.page)
        except ValueError:
            current_idx = 0
        nav = st.radio(
            "Navigation", nav_labels,
            label_visibility="collapsed", index=current_idx,
        )
        target = nav_pages[nav_labels.index(nav)]
        if target != st.session_state.page:
            st.session_state.page = target; st.rerun()
    elif st.session_state.page == "processing":
        st.info("⏳ Scoring in progress…")
    elif st.session_state.page == "output":
        st.success("✓ Score ready")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Dashboard", use_container_width=True):
                st.session_state.page = "portfolio"; st.rerun()
        with c2:
            if st.button("+ New Goal", use_container_width=True):
                reset_input(); st.session_state.page = "input"; st.rerun()
 
    if st.session_state.goal_history:
        st.divider()
        st.caption(f"**{len(st.session_state.goal_history)}** goal(s) submitted this session")
    
    # ── Data controls — refresh / reload / cache clear ──────────────────────
    if st.session_state.page == "portfolio":
        st.divider()
        st.markdown("**Data Controls**")
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("🔄 Refresh", use_container_width=True,
                         help="Re-render the page (state preserved)"):
                st.rerun()
        with dc2:
            if st.button("📥 Reload Data", use_container_width=True,
                         help="Clear cache and re-read CSVs from disk"):
                load_data.clear()
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1: INPUT - NL entry + extraction + review + preview
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "input":

    st.markdown('<div class="badge">◆ GOAL INPUT</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-panel">
        <div class="hero-title">Score a Goal</div>
        <div class="hero-subtitle">Describe your goal in plain English - we'll map it to the organisational hierarchy and submit for coherence scoring.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: NL capture ───────────────────────────────────────────────────
    _has_key = True
    _mode_color = "#4ade80"
    _mode_label = "LLM extraction"
    _mode_note  = "AWS Bedrock Claude Haiku"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:8px 0 4px">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{_mode_color}"></span>
        <span style="font-size:12px;font-weight:600;color:{_mode_color}">{_mode_label}</span>
        <span style="font-size:11px;color:#7A6B4E">· {_mode_note}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="step-label">1 · Describe your goal</div>', unsafe_allow_html=True)
    nl = st.text_area(
        "goal_nl", value=st.session_state.nl_input, height=130, max_chars=1600,
        placeholder=(
            "Describe your goal in plain English.\n\n"
            "Examples:\n"
            "• Improve NPS score from 60 to 85 over 24 months\n"
            "• Increase organic traffic across all Content & SEO channels by 40%\n"
            "• How are paid acquisition channels performing?"
        ),
        label_visibility="collapsed", key="nlw",
    )
    st.session_state.nl_input = nl
    wc = len(nl.split()) if nl.strip() else 0
    wc_color = "#f87171" if wc > 200 else "#64748B"
    st.markdown(f'<div class="wc" style="color:{wc_color}">{wc}/200 words</div>', unsafe_allow_html=True)

    if st.button("Extract Goal Details",
                type="primary", use_container_width=True,
                disabled=not nl.strip() or wc > 200):
        with st.spinner("Parsing with AI…" if _has_key else "Parsing with heuristics…"):
            r = _make_extractor().extract(nl)
            st.session_state.extracted = r
            st.session_state.show_fields = True
            st.session_state.show_preview = False
            for k in ("scope", "bucket_l1", "bucket_l2"):
                if r.get(k): st.session_state[k.replace("bucket_", "")] = r[k]
            st.session_state.l3 = r.get("bucket_l3") or ""
            if r.get("target_value")  is not None: st.session_state.target_value  = str(r["target_value"])
            if r.get("initial_value") is not None: st.session_state.initial_value = str(r["initial_value"])
            if r.get("metric_suggestion"): st.session_state.metric_name = r["metric_suggestion"]
            if r.get("unit_suggestion"):   st.session_state.metric_unit = r["unit_suggestion"]
            st.rerun()

    # ── Step 2: Review extracted fields ──────────────────────────────────────
    if st.session_state.show_fields and st.session_state.extracted:
        ext = st.session_state.extracted
        st.markdown('<div class="step-label">2 · Review extracted details</div>', unsafe_allow_html=True)

        new_title = st.text_input("Goal Title", value=ext.get("goal_title", ""), key="ti")
        st.session_state.extracted["goal_title"] = new_title

        scope = st.session_state.scope
        scls  = {"goal": "scope-goal", "l2_bucket": "scope-l2", "l1_bucket": "scope-l1"}.get(scope, "scope-goal")
        slab  = {"goal": "Single Goal", "l2_bucket": "Department Group", "l1_bucket": "Division Group"}.get(scope, "Goal")
        st.markdown(f'<span class="scope-badge {scls}">{slab}</span>', unsafe_allow_html=True)

        st.markdown("**Organisational Bucket**")
        b1, b2, b3 = st.columns(3)
        with b1:
            opts = list(BUCKET_HIERARCHY.keys())
            idx  = (opts.index(st.session_state.l1) + 1) if st.session_state.l1 in opts else 0
            v    = st.selectbox("L1 Division", ["- Select -"] + opts, index=idx, key="l1s")
            v    = "" if v.startswith("-") else v
            if v != st.session_state.l1:
                st.session_state.l2 = ""; st.session_state.l3 = ""
            st.session_state.l1 = v
        with b2:
            if st.session_state.l1:
                l2o = get_l2_for_l1(st.session_state.l1)
                l2i = (l2o.index(st.session_state.l2) + 1) if st.session_state.l2 in l2o else 0
                v2  = st.selectbox("L2 Department", ["- Select -"] + l2o, index=l2i, key="l2s")
                v2  = "" if v2.startswith("-") else v2
                if v2 != st.session_state.l2: st.session_state.l3 = ""
                st.session_state.l2 = v2
            else:
                st.selectbox("L2 Department", ["- Select L1 first -"], disabled=True, key="l2d")
                st.session_state.l2 = ""
        with b3:
            if st.session_state.l1 and st.session_state.l2:
                l3o = get_l3_for_l2(st.session_state.l1, st.session_state.l2)
                l3i = (l3o.index(st.session_state.l3) + 1) if st.session_state.l3 and st.session_state.l3 in l3o else 0
                v3  = st.selectbox("L3 Function (optional)", ["- All -"] + l3o, index=l3i, key="l3s")
                st.session_state.l3 = "" if v3.startswith("-") else v3
            else:
                st.selectbox("L3 Function", ["- Select L2 first -"], disabled=True, key="l3d")
                st.session_state.l3 = ""

        # Scope is derived from which levels are filled, not manually set.
        if st.session_state.l3:   st.session_state.scope = "goal"
        elif st.session_state.l2: st.session_state.scope = "l2_bucket"
        elif st.session_state.l1: st.session_state.scope = "l1_bucket"

        st.markdown("**Metric & Targets**")
        st.session_state.metric_name = st.text_input("Metric Name", value=st.session_state.metric_name, key="mn")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.initial_value = st.text_input("Initial Value", value=st.session_state.initial_value, key="iv")
        with c2:
            st.session_state.target_value  = st.text_input("Target Value",  value=st.session_state.target_value,  key="tv")
        with c3:
            st.session_state.periods = st.number_input("Periods", value=st.session_state.periods, min_value=1, max_value=48, key="pr")

        u1, u2, u3 = st.columns(3)
        with u1:
            unit_idx = METRIC_UNITS.index(st.session_state.metric_unit) if st.session_state.metric_unit in METRIC_UNITS else 0
            st.session_state.metric_unit = st.selectbox("Unit", METRIC_UNITS, index=unit_idx, key="un")
        with u2:
            scn_idx = SCENARIOS.index(st.session_state.scenario) if st.session_state.scenario in SCENARIOS else 0
            st.session_state.scenario = st.selectbox("Scenario", SCENARIOS, index=scn_idx, key="scn")
        with u3:
            sk_idx = SHOCK_TYPES.index(st.session_state.shock_type) if st.session_state.shock_type in SHOCK_TYPES else 0
            st.session_state.shock_type = st.selectbox("Shock Type", SHOCK_TYPES, index=sk_idx, key="skt",
                                                    help="What kind of shock event to model? Affects scoring stress test.")

        ready = bool(ext.get("goal_title") and st.session_state.l1 and st.session_state.l2 and st.session_state.metric_name)
        if st.button("Review Goal", type="primary", use_container_width=True, disabled=not ready):
            st.session_state.show_preview = True; st.rerun()

    # ── Step 3: Preview + submit ─────────────────────────────────────────────
    if st.session_state.show_preview and st.session_state.extracted:
        ext = st.session_state.extracted
        st.markdown('<div class="step-label">3 · Goal Preview</div>', unsafe_allow_html=True)
        scope = st.session_state.scope
        scls  = {"goal": "scope-goal", "l2_bucket": "scope-l2", "l1_bucket": "scope-l1"}.get(scope, "scope-goal")
        slab  = {"goal": "Single Goal", "l2_bucket": "Department Group", "l1_bucket": "Division Group"}.get(scope, "Goal")
        bp = bucket_path()
        st.markdown(f"""
        <div class="goal-card-glow">
            <div style="font-size:22px;font-weight:700;color:#F8FAFC;margin-bottom:6px">
                {ext['goal_title']}
                <span class="scope-badge {scls}" style="vertical-align:middle;margin-left:8px">{slab}</span>
            </div>
            <div style="font-size:13px;color:#94A3B8;margin-bottom:14px">{bp}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
                <div><span style="font-size:11px;color:#64748B">Metric</span><br>
                     <span style="color:#E5E7EB">{st.session_state.metric_name} ({st.session_state.metric_unit})</span></div>
                <div><span style="font-size:11px;color:#64748B">Range</span><br>
                     <span style="color:#E5E7EB">{st.session_state.initial_value or '?'} → {st.session_state.target_value or '?'}</span></div>
                <div><span style="font-size:11px;color:#64748B">Timeline</span><br>
                     <span style="color:#E5E7EB">{st.session_state.periods} periods</span></div>
                <div><span style="font-size:11px;color:#64748B">Scenario</span><br>
                     <span style="color:#E5E7EB">{st.session_state.scenario}</span></div>
                <div><span style="font-size:11px;color:#64748B">Shock Type</span><br>
                     <span style="color:{'#FCD34D' if st.session_state.shock_type != 'none' else '#E5E7EB'};font-weight:{'600' if st.session_state.shock_type != 'none' else '400'}">{st.session_state.shock_type}{' ⚡' if st.session_state.shock_type != 'none' else ''}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dev-only: show raw payload when running with DECIDR_DEV=1 in the environment.
        # Sponsors and end users don't need this.
        if os.getenv("DECIDR_DEV") == "1":
            with st.expander("View raw JSON payload"):
                st.json(json.loads(build_payload().to_json()))

        a1, a2 = st.columns([1, 2])
        with a1:
            if st.button("✏️ Edit", use_container_width=True):
                st.session_state.show_preview = False; st.rerun()
        with a2:
            if st.button("Submit for Scoring →", type="primary", use_container_width=True):
                pl = build_payload()
                ts_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                (OUT_DIR / f"goal_{ts_stamp}.json").write_text(pl.to_json())
                st.session_state.payload_json = pl.to_json()
                st.session_state.goal_history.append({
                    "title": ext["goal_title"], "scope": scope,
                    "bucket": bp, "time": ts_stamp,
                })
                st.session_state.page = "processing"; st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2: PROCESSING - fake progress bar (real work happens in System 3)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "processing":
    title = st.session_state.extracted["goal_title"] if st.session_state.extracted else "Goal"
    _, cc, _ = st.columns([1, 2, 1])
    with cc:
        st.markdown(
            '<div style="text-align:center;padding:60px 0 16px">'
            '<div class="dot"></div><div class="dot"></div><div class="dot"></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<h2 style="text-align:center;font-weight:600">Scoring in progress</h2>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center;color:#94A3B8">Analysing <strong style="color:#93C5FD">{title}</strong></p>', unsafe_allow_html=True)
        bar = st.progress(0); status = st.empty()
        stages = ["Sending to engine…", "Computing features…", "Dimension analysis…",
                  "Ensemble blending…", "Generating report…"]
        for i, msg in enumerate(stages):
            status.caption(f"⏳ {msg}"); bar.progress((i + 1) / len(stages)); time.sleep(0.5)
        status.caption("✓ Complete")
    demo = DEMO_SCORE.copy()
    demo["goal_title"]  = title
    demo["bucket_path"] = bucket_path()

    # ── Live scoring attempt ──────────────────────────────────────────────
    # Try to call score_goal() with the submitted goal's L2 bucket.
    # Falls back to DEMO_SCORE if Bedrock is unavailable or no matching goal.
    _l2_submitted = st.session_state.get("l2", "")
    if _l2_submitted:
        try:
            _live = _live_score_goal(_l2_submitted, current_period=18)
            if _live and _live.get("status") == "ok":
                # Merge live scores into the result dict — keep demo keys for UI fields
                for _k in ["attainability","relevance","coherence","integrity","overall",
                           "gp_mean","gp_std","gp_weight","llm_weight","baseline",
                           "uncertain","n_llm_ok","reasoning","ensemble_meta",
                           "verified_attainability","verified_relevance",
                           "verified_coherence","verified_integrity",
                           "verified_composite","flags","narrative"]:
                    if _k in _live:
                        demo[_k] = _live[_k]
                demo["goal_id"]    = _live.get("goal_id", demo["goal_id"])
                demo["live_score"] = True
        except Exception:
            demo["live_score"] = False
    # ─────────────────────────────────────────────────────────────────────
    st.session_state.score_result = demo
    st.session_state["query"] = st.session_state.nl_input  # handoff key for Subhan's dashboard page
    time.sleep(0.3)
    st.session_state.page = "output"; st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3: OUTPUT - single-goal coherence report
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "output":
    s = st.session_state.score_result
    if not s:
        st.warning("No score data. Submit a goal first.")
        st.stop()
 
    title = s.get("goal_title", "Goal")
    bp    = s.get("bucket_path", "")
 
    st.markdown('<div class="badge">◆ COHERENCE REPORT</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hero-panel">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{bp} · {st.session_state.metric_name} · {st.session_state.scenario}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Composite headline card ──────────────────────────────────────────────
    ov = s["overall"]; c = score_color(ov)
    conf_line = "High Confidence" if not s["uncertain"] else "⚠️ Elevated Uncertainty"
    st.markdown(f"""
    <div class="goal-card-hl">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px">
            <div>
                <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:0.08em">Composite Coherence</div>
                <div style="font-size:48px;font-weight:700;color:{c};line-height:1">{ov:.1%}</div>
                <div style="font-size:12px;color:#94A3B8;margin-top:4px">{conf_line} · σ={s['gp_std']:.4f}</div>
            </div>
            <div style="text-align:right">
                <div style="font-size:11px;color:#64748B">Status</div>
                <div style="font-size:22px;font-weight:700;color:{c}">{risk_label(ov)}</div>
                <div style="font-size:12px;color:#64748B;margin-top:4px">LLMs: {s['n_llm_ok']}/3</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dimension KPI strip (quick read) ─────────────────────────────────────
    st.markdown('<div class="step-label">Dimension Scores</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, d in enumerate(DIMS):
        v = s[d]; c2 = score_color(v)
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="color:{c2}">{d.capitalize()}</div>
                <div class="metric-value" style="color:{c2}">{v:.1%}</div>
                <div class="bar-wrap"><div class="bar-fill" style="width:{v*100:.0f}%;background:{c2}"></div></div>
                <div class="metric-sub">{WEIGHTS[d]:.0%} weight</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Dimension Breakdown: radar + ranked bar (mirrors dashboard Tab 1) ────
    st.markdown('<div class="step-label">Dimension Breakdown</div>', unsafe_allow_html=True)
    br1, br2 = st.columns(2)
 
    with br1:
        cats = [d.capitalize() for d in DIMS]
        vals = [s[d] for d in DIMS]
        weights_ring = [WEIGHTS[d] * 2 for d in DIMS]  # scaled so weight ring is visible on 0-1 scale
        fig = go.Figure()
        # Weight reference ring (what each dim should contribute if evenly weighted)
        fig.add_trace(go.Scatterpolar(
            r=weights_ring + [weights_ring[0]],
            theta=cats + [cats[0]],
            fill="toself", name="Weight profile",
            line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot"),
            fillcolor="rgba(148,163,184,0.08)",
        ))
        # Actual dimension scores
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself", name="This goal",
            line=dict(color=COLORS["composite"], width=2.5),
            fillcolor="rgba(129,114,178,0.25)",
        ))
        # Risk threshold ring at 0.35
        fig.add_trace(go.Scatterpolar(
            r=[0.35] * (len(cats) + 1),
            theta=cats + [cats[0]],
            name="Risk threshold",
            line=dict(color=COLORS["at_risk"], width=1.5, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(15,23,42,0.3)",
                radialaxis=dict(range=[0, 1], tickfont=dict(size=9), gridcolor="rgba(148,163,184,0.15)"),
                angularaxis=dict(tickfont=dict(size=11), gridcolor="rgba(148,163,184,0.15)"),
            ),
            title="Dimension Radar", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=10)),
            margin=dict(l=40, r=40, t=50, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)
        viz.render(viz.caption_per_goal_radar(s))
 
    with br2:
        sorted_dims  = sorted(DIMS, key=lambda d: s[d])
        bar_colors   = [score_color(s[d]) for d in sorted_dims]
        fig = go.Figure(go.Bar(
            x=[s[d] for d in sorted_dims],
            y=[d.capitalize() for d in sorted_dims],
            orientation="h", marker_color=bar_colors,
            text=[f"{s[d]:.1%}" for d in sorted_dims],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
        ))
        fig.add_vline(x=0.35, line_dash="dash", line_color=COLORS["at_risk"],
                      annotation_text="Risk threshold", annotation_position="top")
        fig.add_vline(x=0.20, line_dash="dot", line_color="#8B0000",
                      annotation_text="Critical", annotation_position="bottom")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="Dimensions Ranked (red=at risk, green=on track)",
            xaxis_range=[0, 1.05],
            xaxis_title="Score", yaxis_title="",
            height=380, showlegend=False,
            margin=dict(l=20, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    # ── Projected Trajectory (mirrors dashboard's headline time-series) ──────
    st.markdown('<div class="step-label">Projected Coherence Trajectory</div>', unsafe_allow_html=True)
    st.caption(f"Simulated {st.session_state.periods}-period path from baseline to predicted end-state. Real trajectory comes from System 2 once wired in.")
 
    def _simulate_trajectory(final_score, periods, baseline, seed):
        """Deterministic sigmoid curve from baseline → final_score with small noise."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, periods)
        sig = 1 / (1 + np.exp(-5 * (t - 0.5)))
        trajectory = baseline + (final_score - baseline) * sig
        trajectory += rng.normal(0, 0.015, periods)
        return np.clip(trajectory, 0, 1)
 
    periods_n = int(st.session_state.periods) if st.session_state.periods else 24
    x_axis = np.arange(1, periods_n + 1)
    baseline_val = s["baseline"]
 
    fig = go.Figure()
    # 4 dimension traces (dotted, muted — like the dashboard's dimension overlays)
    seeds = {"coherence": 1, "attainability": 2, "relevance": 3, "integrity": 4}
    for d in DIMS:
        traj = _simulate_trajectory(s[d], periods_n, baseline_val, seeds[d])
        fig.add_trace(go.Scatter(
            x=x_axis, y=traj, mode="lines", name=d.capitalize(),
            line=dict(color=COLORS[d], width=1.5, dash="dot"), opacity=0.75,
        ))
    # Composite trace (solid, prominent)
    comp_traj = _simulate_trajectory(s["overall"], periods_n, baseline_val, 0)
    fig.add_trace(go.Scatter(
        x=x_axis, y=comp_traj, mode="lines+markers", name="Composite (projected)",
        line=dict(color=COLORS["composite"], width=3), marker=dict(size=5),
    ))
    # Risk threshold
    fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS["at_risk"],
                  annotation_text="Risk threshold (0.35)")
    # Endpoint marker for final composite
    fig.add_trace(go.Scatter(
        x=[periods_n], y=[s["overall"]], mode="markers+text",
        marker=dict(size=14, color=score_color(s["overall"]), symbol="star",
                    line=dict(color="#F8FAFC", width=1.5)),
        text=[f" End: {s['overall']:.1%}"], textposition="middle right",
        textfont=dict(size=11, color="#F8FAFC"),
        name="Final composite", showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Period", yaxis_title="Score", yaxis_range=[0, 1],
        xaxis_range=[0.5, periods_n + 2.5],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # ── Ensemble Composition (transparency into how scores were blended) ─────
    st.markdown('<div class="step-label">Ensemble Composition</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns([3, 2])
 
    with ec1:
        # Stacked horizontal bar — per dimension, show weight split across GP/Rule/LLM
        rows = []
        for d in DIMS:
            meta = s["ensemble_meta"].get(d, {})
            gp_w   = meta.get("gp_weight",   0)
            rule_w = meta.get("rule_weight", 0)
            llm_w  = meta.get("llm_weight",  0)
            if gp_w:   rows.append({"Dimension": d.capitalize(), "Model": "Gaussian Process", "Weight": gp_w})
            if rule_w: rows.append({"Dimension": d.capitalize(), "Model": "Rule-based",       "Weight": rule_w})
            if llm_w:  rows.append({"Dimension": d.capitalize(), "Model": "LLM Ensemble",     "Weight": llm_w})
        df_ens = pd.DataFrame(rows)
        if len(df_ens) > 0:
            fig = px.bar(
                df_ens, x="Weight", y="Dimension", color="Model", orientation="h",
                color_discrete_map={
                    "Gaussian Process": COLORS["coherence"],
                    "Rule-based":       COLORS["relevance"],
                    "LLM Ensemble":     COLORS["attainability"],
                },
                text=df_ens["Weight"].apply(lambda w: f"{w:.0%}"),
                category_orders={"Dimension": [d.capitalize() for d in DIMS]},
                title="Weight split per dimension",
            )
            fig.update_traces(textposition="inside", insidetextanchor="middle", textfont=dict(color="#F8FAFC", size=11))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                barmode="stack", xaxis_range=[0, 1], xaxis_tickformat=".0%",
                xaxis_title="Weight share", yaxis_title="", height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
 
    with ec2:
        # Confidence KPIs (moved here — they belong with ensemble stats)
        st.markdown(f"""
        <div class="goal-card" style="height:320px">
            <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:14px">Confidence</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                <div>
                    <div style="font-size:10px;color:#64748B">Baseline</div>
                    <div style="font-size:20px;font-weight:700;color:#94A3B8">{s['baseline']:.1%}</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#64748B">GP Mean</div>
                    <div style="font-size:20px;font-weight:700;color:{COLORS['coherence']}">{s['gp_mean']:.1%}</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#64748B">GP Weight</div>
                    <div style="font-size:20px;font-weight:700;color:{COLORS['relevance']}">{s['gp_weight']:.0%}</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#64748B">GP σ</div>
                    <div style="font-size:20px;font-weight:700;color:{COLORS['attainability']}">{s['gp_std']:.4f}</div>
                </div>
                <div style="grid-column:1/3;padding-top:10px;border-top:1px solid rgba(148,163,184,0.15)">
                    <div style="font-size:10px;color:#64748B">LLM agreement</div>
                    <div style="font-size:20px;font-weight:700;color:#F8FAFC">{s['n_llm_ok']}/3 models</div>
                </div>
                <div style="grid-column:1/3">
                    <div style="font-size:10px;color:#64748B">Overall status</div>
                    <div style="font-size:14px;font-weight:600;color:{score_color(ov)}">
                        {"✓ High confidence" if not s["uncertain"] else "⚠️ Elevated uncertainty"}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Dimension analysis with reasoning (kept - qualitative insight) ───────
    st.markdown('<div class="step-label">Why these scores?</div>', unsafe_allow_html=True)
    dim_desc = {
        "coherence":     "Are decisions consistent across levels, goals, and time?",
        "attainability": "Is the goal realistically achievable?",
        "relevance":     "Is allocation justified against stated goals?",
        "integrity":     "Are assumptions transparent and auditable?",
    }
    for d in DIMS:
        v = s[d]; c2 = score_color(v)
        reasoning = s["reasoning"].get(d, "")
        meta = s["ensemble_meta"].get(d, {})
        meta_parts = []
        if "gp_weight"   in meta: meta_parts.append(f"GP {meta['gp_weight']:.0%}")
        if "rule_weight" in meta: meta_parts.append(f"Rule {meta['rule_weight']:.0%}")
        if "llm_weight"  in meta: meta_parts.append(f"LLM {meta['llm_weight']:.0%}")
        if "variance"    in meta: meta_parts.append(f"σ²={meta['variance']:.4f}")
        meta_str = " · ".join(meta_parts)
        card_cls = "goal-card-hl" if d == "coherence" else "goal-card"
        st.markdown(f"""
        <div class="{card_cls}">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
                <div>
                    <span style="font-size:15px;font-weight:600;color:#F8FAFC">{d.capitalize()}</span>
                    <span style="font-size:12px;color:#64748B"> — {dim_desc.get(d, '')}</span>
                </div>
                <span style="font-size:22px;font-weight:700;color:{c2}">{v:.1%}</span>
            </div>
            <div class="bar-wrap"><div class="bar-fill" style="width:{v*100:.0f}%;background:{c2}"></div></div>
            <div style="font-size:13px;color:#E5E7EB;line-height:1.6;margin:10px 0 6px;border-left:3px solid {c2};padding-left:12px">{reasoning}</div>
            <div style="font-size:11px;color:#64748B">{meta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Forward Projection (this goal) ───────────────────────────────────────
    st.markdown('<div class="step-label">Forward Projection (this goal)</div>', unsafe_allow_html=True)
    fwd_data = s.get("forward", {})
    if fwd_data:
        traj_class  = fwd_data.get("trajectory_class", "stable")
        traj_color  = {"improving": "#4ade80", "stable": "#94A3B8", "degrading": "#f87171"}.get(traj_class, "#94A3B8")
        traj_arrow  = {"improving": "↑", "stable": "→", "degrading": "↓"}.get(traj_class, "→")
 
        fc1, fc2, fc3 = st.columns([1, 1, 1])
        with fc1: metric_card("+6 periods",  f"{fwd_data['p6_projected']:.1%}",
                              f"CI [{fwd_data['p6_lower']:.0%}, {fwd_data['p6_upper']:.0%}]",  COLORS["coherence"])
        with fc2: metric_card("+12 periods", f"{fwd_data['p12_projected']:.1%}",
                              f"CI [{fwd_data['p12_lower']:.0%}, {fwd_data['p12_upper']:.0%}]", COLORS["composite"])
        with fc3: metric_card("+18 periods", f"{fwd_data['p18_projected']:.1%}",
                              f"CI [{fwd_data['p18_lower']:.0%}, {fwd_data['p18_upper']:.0%}]", COLORS["relevance"])
 
        # Projection chart with confidence band
        x_proj = [0, 6, 12, 18]
        y_proj = [s["overall"], fwd_data["p6_projected"], fwd_data["p12_projected"], fwd_data["p18_projected"]]
        y_low  = [s["overall"], fwd_data["p6_lower"],     fwd_data["p12_lower"],     fwd_data["p18_lower"]]
        y_hi   = [s["overall"], fwd_data["p6_upper"],     fwd_data["p12_upper"],     fwd_data["p18_upper"]]
        fig = go.Figure()
        # Confidence band
        fig.add_trace(go.Scatter(x=x_proj+x_proj[::-1], y=y_hi+y_low[::-1],
                                 fill='toself', fillcolor='rgba(76,114,176,0.18)',
                                 line=dict(color='rgba(0,0,0,0)'), name='90% CI',
                                 hoverinfo='skip', showlegend=True))
        # Mean projection — dashes per spec
        fig.add_trace(go.Scatter(x=x_proj, y=y_proj, mode='lines+markers',
                                 line=dict(color=COLORS["composite"], width=3, dash='dash'),
                                 marker=dict(size=10, color=traj_color, line=dict(color='#F8FAFC', width=1)),
                                 name='Projected composite'))
        fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS["at_risk"],
                      annotation_text="Risk threshold")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Periods ahead", yaxis_title="Composite",
            yaxis_range=[0, 1], height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
 
        viz.render(viz.caption_per_goal_trajectory(s, st.session_state.periods))
 
    # ── Shock Analysis (this goal) ───────────────────────────────────────────
    st.markdown('<div class="step-label">Shock Resilience (this goal)</div>', unsafe_allow_html=True)
    shock_data = s.get("shock", {})
    if shock_data:
        vuln_class  = shock_data.get("vulnerability_class", "moderate")
        vuln_color  = {"resilient": "#4ade80", "moderate": "#fbbf24", "vulnerable": "#f87171"}.get(vuln_class, "#fbbf24")
 
        sc1, sc2, sc3, sc4 = st.columns(4)
        budget_drop = shock_data["budget_shock"] - shock_data["pre_shock"]
        market_drop = shock_data["market_shock"] - shock_data["pre_shock"]
        recovery_d  = shock_data["post_shock"]   - shock_data["budget_shock"]
        with sc1: metric_card("Pre-shock",       f"{shock_data['pre_shock']:.3f}",  "Baseline composite", COLORS["safe"])
        with sc2: metric_card("Budget shock Δ",  f"{budget_drop:+.3f}",             f"At {shock_data['budget_shock']:.3f}", COLORS["budget_shock"])
        with sc3: metric_card("Market shock Δ",  f"{market_drop:+.3f}",             f"At {shock_data['market_shock']:.3f}", COLORS["market_shock"])
        with sc4: metric_card("Recovery",        f"{recovery_d:+.3f}",              f"In {shock_data['recovery_periods']} periods", COLORS["relevance"])
 
        # Shock-phase bar chart
        phases = ["Pre-shock", "Budget shock", "Market shock", "Post-shock"]
        vals   = [shock_data["pre_shock"], shock_data["budget_shock"],
                  shock_data["market_shock"], shock_data["post_shock"]]
        bar_colors = [COLORS["safe"], COLORS["budget_shock"], COLORS["market_shock"], COLORS["relevance"]]
        fig = go.Figure(go.Bar(
            x=phases, y=vals, marker_color=bar_colors,
            text=[f"{v:.2f}" for v in vals], textposition="outside",
        ))
        fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS["at_risk"],
                      annotation_text="Risk threshold")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Composite", yaxis_range=[0, 1], height=320,
            showlegend=False, margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
 
        # Vulnerability badge + caption
        max_drop = min(budget_drop, market_drop)
        st.markdown(f"""
        <div style="display:inline-block;padding:6px 14px;border-radius:14px;background:rgba({{
            'resilient':'74,222,128','moderate':'251,191,36','vulnerable':'248,113,113'
        }}.get('{vuln_class}','251,191,36'),0.15);
        border:1px solid {vuln_color};color:{vuln_color};font-weight:700;font-size:12px">
        SHOCK PROFILE: {vuln_class.upper()}
        </div>
        """, unsafe_allow_html=True)
        viz.render(viz.caption_per_goal_shock(s))

    # ── Portfolio Context (mini-dashboard at the bottom of output page) ─────
    portfolio_data = DATA.get("composite") if "DATA" in dir() else None
    if portfolio_data is not None and len(portfolio_data) > 0:
        st.markdown('<div class="step-label">Portfolio Context</div>', unsafe_allow_html=True)
        st.caption("Where this goal sits relative to the rest of the portfolio.")
 
        # Pick a snapshot period from the data
        pf_period = 18 if (
            "period_id" in portfolio_data.columns and 18 in portfolio_data["period_id"].values
        ) else (
            portfolio_data["period_id"].max() if "period_id" in portfolio_data.columns else None
        )
        pf_view = portfolio_data[portfolio_data["period_id"] == pf_period].copy() \
                  if pf_period is not None else portfolio_data.copy()
 
        if len(pf_view) > 0:
            pcA, pcB, pcC, pcD = st.columns(4)
            avg_comp_pf = pf_view["composite"].mean() if "composite" in pf_view.columns else 0
            n_risk_pf   = int(pf_view["at_risk"].sum()) if "at_risk" in pf_view.columns else 0
            n_total_pf  = len(pf_view)
            this_score  = s["overall"]
            # Where does this goal rank?
            if "composite" in pf_view.columns:
                better_than = (pf_view["composite"] < this_score).sum()
                pct_rank = 100 * better_than / max(n_total_pf, 1)
            else:
                pct_rank = 50
 
            with pcA: metric_card("Portfolio avg",     f"{avg_comp_pf:.3f}", f"Across {n_total_pf} goals",      COLORS["composite"])
            with pcB: metric_card("This goal",         f"{this_score:.3f}",  f"{this_score - avg_comp_pf:+.3f} vs avg", score_color(this_score))
            with pcC: metric_card("Portfolio %ile",    f"{pct_rank:.0f}",     "Higher = better",                 COLORS["relevance"])
            with pcD: metric_card("Goals at risk",     f"{n_risk_pf}/{n_total_pf}", "Below 0.35",                COLORS["at_risk"])
 
            # Mini headline chart: this goal's score on top of portfolio time-series
            ts_data = DATA.get("coherence_ts") if "DATA" in dir() else None
            if ts_data is not None and "period_id" in ts_data.columns and "avg_composite" in ts_data.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_data["period_id"], y=ts_data["avg_composite"],
                    mode="lines", name="Portfolio avg",
                    line=dict(color=COLORS["composite"], width=2),
                ))
                # Place this goal's score as a horizontal reference line
                fig.add_hline(y=this_score, line_dash="dot", line_color=score_color(this_score),
                              annotation_text=f"This goal: {this_score:.2f}",
                              annotation_position="top right")
                fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS["at_risk"],
                              annotation_text="Risk threshold")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Period", yaxis_title="Composite", yaxis_range=[0, 1],
                    height=280, margin=dict(l=20, r=20, t=20, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
 
            viz.render(viz.caption_portfolio_context(this_score, avg_comp_pf, pct_rank))

    # ── Feedback + downloads ─────────────────────────────────────────────────
    st.markdown('<div class="step-label">Feedback</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        if st.button("👍 Looks Right", use_container_width=True): st.session_state.feedback = "positive"
    with f2:
        if st.button("🤔 Not Sure",    use_container_width=True): st.session_state.feedback = "neutral"
    with f3:
        if st.button("👎 Seems Off",   use_container_width=True): st.session_state.feedback = "negative"
    if st.session_state.feedback:
        msg = {"positive": "Thanks! Recorded.", "neutral": "Noted.",
            "negative": "Thanks for flagging."}.get(st.session_state.feedback, "")
        st.caption(msg)

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇ Score Report (JSON)",
            data=json.dumps(s, indent=2),
            file_name=f"report_{title.replace(' ', '_')}.json",
            mime="application/json", use_container_width=True,
        )
    with d2:
        if st.session_state.payload_json:
            st.download_button(
                "⬇ Input Payload (JSON)",
                data=st.session_state.payload_json,
                file_name=f"input_{title.replace(' ', '_')}.json",
                mime="application/json", use_container_width=True,
            )
    st.divider()
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("📊 Back to Dashboard", use_container_width=True):
            st.session_state.page = "portfolio"; st.rerun()
    with a2:
        if st.button("+ New Goal", type="primary", use_container_width=True):
            reset_input(); st.rerun()
    with a3:
        if st.button("🔄 Re-run Scoring", use_container_width=True):
            st.session_state.page = "processing"; st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4: PORTFOLIO - teammate's app(1).py code, verbatim
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "portfolio":

    # ── Load data (teammate's exact logic, using shared DATA cache) ─────────
    data = dict(DATA)  # shallow copy so we can override
    mapping   = dict(data.get("_mapping", {}))
    inventory = data.get("_inventory", {})

    # ── Manual overrides: if user picked a file for any slot, load that ─────
    manual = st.session_state.get("manual_mapping", {})
    for slot, fname in manual.items():
        if fname and fname in inventory:
            for folder in (".", "data"):
                p = os.path.join(folder, fname)
                if os.path.exists(p):
                    try:
                        df = pd.read_csv(p)
                        data[slot] = _normalise_cols(df, slot)
                        mapping[slot] = fname
                    except Exception:
                        pass
                    break

    # ── If composite is STILL missing, show rich diagnostic + file picker ───
    if data.get("composite") is None:
        st.error("⚠️ Goal-level scores not auto-detected. Pick the right file below.")
        st.markdown("### 🔍 Inventory of CSVs in your folder")
        st.caption("These are all the CSV files found. For each slot below, pick the file that contains the matching data. Goal-level scores is the critical one - that file needs columns like `goal_id`, `coherence`, `composite`.")

        # Show every CSV with its columns, so you can see what's in each
        with st.expander("📄 Show all CSV columns in folder", expanded=True):
            for fname, cols in inventory.items():
                st.markdown(f"**`{fname}`**  \n{', '.join(f'`{c}`' for c in cols[:25])}" +
                            ("  …" if len(cols) > 25 else ""))

        # Manual override dropdowns - pick which file maps to which slot
        st.markdown("### 🛠️ Manual file assignment")
        slot_labels = {
            "composite"   : "Goal-level scores (REQUIRED)",
            "coherence_ts": "Coherence time series (optional)",
            "portfolio_ts": "Bucket time series (optional)",
            "forward_proj": "Forward projection (optional)",
            "portfolio"   : "Portfolio summary (optional)",
        }
        file_options = ["- none -"] + sorted(inventory.keys())
        new_manual = dict(manual)
        for slot, label in slot_labels.items():
            current = manual.get(slot, mapping.get(slot, "- none -"))
            if current not in file_options:
                current = "- none -"
            pick = st.selectbox(
                label, file_options,
                index=file_options.index(current),
                key=f"pick_{slot}",
            )
            new_manual[slot] = pick if pick != "- none -" else ""

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Apply & Load Dashboard", type="primary", use_container_width=True):
                st.session_state.manual_mapping = new_manual
                load_data.clear()  # invalidate cache so canonical path re-runs
                st.rerun()
        with c2:
            if st.button("📊 Try auto-detect again", use_container_width=True):
                st.session_state.manual_mapping = {}
                load_data.clear()
                st.rerun()

        st.info("💡 **Tip:** The goal-level scores file probably has columns like `goal_id`, `coherence`, `composite`, `at_risk`. Best candidates are usually `analytical_flat.csv`, `meta_learner_predictions_poc.csv`, `meta_learner_results_poc.csv`, or `outputs.csv`.")
        st.stop()

    comp  = data.get("composite")
    ts    = data.get("coherence_ts")
    pts   = data.get("portfolio_ts")
    fwd   = data.get("forward_proj")
    port  = data.get("portfolio")

    DIMS   = ["coherence","attainability","relevance","integrity"]

    def _prepare_goal_scores(df):
        if df is None or len(df) == 0:
            return df
        df = df.copy()
        if "composite" not in df.columns:
            for alt in ["final_composite", "verified_composite", "composite_adjusted", "composite_score", "overall", "overall_score", "final_score", "score"]:
                if alt in df.columns:
                    df["composite"] = df[alt]
                    break
        if "at_risk" not in df.columns and "composite" in df.columns:
            df["at_risk"] = df["composite"] < 0.35
        if "critical" not in df.columns and "composite" in df.columns:
            df["critical"] = df["composite"] < 0.20
        if "weakest_dim" not in df.columns and all(d in df.columns for d in DIMS):
            df["weakest_dim"] = df[DIMS].idxmin(axis=1)
        return df

    def _build_portfolio_summary(source_df):
        if source_df is None or len(source_df) == 0 or "l2_name" not in source_df.columns:
            return None
        agg = {}
        if "composite" in source_df.columns:
            agg["composite"] = "mean"
        for d in DIMS:
            if d in source_df.columns:
                agg[d] = "mean"
        if "at_risk" in source_df.columns:
            agg["at_risk"] = "sum"
        if not agg:
            return None
        out = source_df.groupby("l2_name", as_index=False).agg(agg)
        rename = {"composite": "avg_composite"}
        for d in DIMS:
            rename[d] = f"avg_{d}"
        if "at_risk" in out.columns:
            rename["at_risk"] = "at_risk_count"
        out = out.rename(columns=rename)
        if "avg_composite" not in out.columns and any(f"avg_{d}" in out.columns for d in DIMS):
            dim_cols = [f"avg_{d}" for d in DIMS if f"avg_{d}" in out.columns]
            out["avg_composite"] = out[dim_cols].mean(axis=1)
        if "at_risk_count" not in out.columns:
            out["at_risk_count"] = 0
        return out

    def _build_timeseries(source_df):
        if source_df is None or len(source_df) == 0 or "period_id" not in source_df.columns:
            return None
        agg = {}
        if "composite" in source_df.columns:
            agg["composite"] = "mean"
        for d in DIMS:
            if d in source_df.columns:
                agg[d] = "mean"
        if "at_risk" in source_df.columns:
            agg["at_risk"] = "sum"
        if not agg:
            return None
        out = source_df.groupby("period_id", as_index=False).agg(agg)
        rename = {"composite": "avg_composite"}
        for d in DIMS:
            rename[d] = f"avg_{d}"
        if "at_risk" in out.columns:
            rename["at_risk"] = "at_risk_count"
        out = out.rename(columns=rename)
        if "avg_composite" not in out.columns and any(f"avg_{d}" in out.columns for d in DIMS):
            dim_cols = [f"avg_{d}" for d in DIMS if f"avg_{d}" in out.columns]
            out["avg_composite"] = out[dim_cols].mean(axis=1)
        return out

    def _build_bucket_timeseries(source_df):
        if source_df is None or len(source_df) == 0 or not all(c in source_df.columns for c in ["period_id", "l2_name"]):
            return None
        value_col = "composite" if "composite" in source_df.columns else next((d for d in DIMS if d in source_df.columns), None)
        if value_col is None:
            return None
        return source_df.groupby(["period_id", "l2_name"], as_index=False)[value_col].mean().rename(columns={value_col: "avg_composite"})

    comp = _prepare_goal_scores(comp)
    if ts is None or len(ts) == 0:
        ts = _build_timeseries(comp)
    if pts is None or len(pts) == 0:
        pts = _build_bucket_timeseries(comp)

    # ── Snapshot anchor for KPI cards (always p18, the pipeline anchor) ────
    # Individual tabs carry their own period selectors and filters inline.
    _all_periods = sorted(comp['period_id'].dropna().unique().tolist()) \
                   if comp is not None and 'period_id' in comp.columns else [18]
    selected_period = 18 if 18 in _all_periods else (_all_periods[-1] if _all_periods else 12)
    comp_view = comp[comp['period_id'] == selected_period].copy() \
                if comp is not None and 'period_id' in comp.columns else \
                (comp.copy() if comp is not None else pd.DataFrame())

    if port is None or len(port) == 0:
        port = _build_portfolio_summary(comp_view if comp_view is not None and len(comp_view) > 0 else comp)



    # ── Header (with "Score a New Goal" CTA so input flow is discoverable) ──
    hero_col, cta_col = st.columns([4, 1])
    with hero_col:
        st.markdown("""
        <div class="hero-panel">
            <div class="hero-title">Decidr Lens</div>
            <div class="hero-subtitle">Portfolio health dashboard</div>
        </div>
        """, unsafe_allow_html=True)
    with cta_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("➕ Score a New Goal", use_container_width=True, type="primary", key="cta_new_goal"):
            st.session_state.page = "input"; st.rerun()
        st.caption("Add a goal to this portfolio")

    # ── Period selector + KPI cards + expandable rows ─────────────────────
    _all_kpi_periods = sorted(comp['period_id'].dropna().unique().tolist()) \
                       if comp is not None and 'period_id' in comp.columns else [18]
    _kpi_sel_col, _kpi_spacer = st.columns([2, 6])
    with _kpi_sel_col:
        _kpi_sel_p = st.selectbox(
            "Snapshot period",
            _all_kpi_periods,
            index=_all_kpi_periods.index(18) if 18 in _all_kpi_periods else len(_all_kpi_periods) - 1,
            key="kpi_period_sel",
            help="Change to see KPI cards and goal breakdowns at any period. p18 is the pipeline anchor."
        )
    comp_view = comp[comp['period_id'] == _kpi_sel_p].copy() \
                if comp is not None and 'period_id' in comp.columns else comp_view

    if comp_view is not None and len(comp_view) > 0:
        avg_comp  = comp_view['composite'].mean() if 'composite' in comp_view.columns else 0
        n_risk    = int(comp_view['at_risk'].sum()) if 'at_risk' in comp_view.columns else 0
        n_crit    = int(comp_view['critical'].sum()) if 'critical' in comp_view.columns else 0
        n_goals   = len(comp_view)
        avg_coh   = comp_view['coherence'].mean() if 'coherence' in comp_view.columns else 0
        n_p1      = int((comp_view['goal_priority'] == 'P1').sum()) if 'goal_priority' in comp_view.columns else n_crit
        retention = round((1 - n_risk / max(n_goals, 1)) * 100, 1)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            metric_card("Portfolio Coherence", f"{avg_comp:.3f}", "Normalized composite score", COLORS["composite"])
        with col2:
            metric_card("Coherence Score", f"{avg_coh:.3f}", "Coherence dimension mean", COLORS["coherence"])
        with col3:
            metric_card("Goals at Risk", f"{n_risk}/{n_goals}", "Goals below risk threshold", COLORS["at_risk"])
        with col4:
            metric_card("P1 Priority Goals", f"{n_p1}", "Critical or shock-hit goals", "#FF6B35")
        with col5:
            metric_card("Goal Retention Rate", f"{retention}%", "% of goals not at risk", COLORS["relevance"])

        # ── Expandable detail rows ────────────────────────────────────────
        _dcol1, _dcol2, _dcol3 = st.columns(3)

        with _dcol1:
            with st.expander(f"▾ {n_risk} at-risk goals" if n_risk else "▾ No at-risk goals"):
                _at_risk_df = comp_view[comp_view['at_risk'] == True] \
                              if 'at_risk' in comp_view.columns else pd.DataFrame()
                if len(_at_risk_df) > 0:
                    _show_cols = [c for c in ['goal_id','l2_name','composite','coherence',
                                              'attainability','weakest_dim','goal_priority',
                                              'shock_effect'] if c in _at_risk_df.columns]
                    st.dataframe(_at_risk_df[_show_cols].sort_values('composite').round(3),
                                 use_container_width=True, hide_index=True)
                else:
                    st.caption("No goals below the 0.35 risk threshold at this period.")

        with _dcol2:
            with st.expander(f"▾ {n_p1} P1 priority goals" if n_p1 else "▾ No P1 priority goals"):
                _p1_df = comp_view[comp_view['goal_priority'] == 'P1'] \
                         if 'goal_priority' in comp_view.columns \
                         else comp_view[comp_view['critical'] == True]
                if len(_p1_df) > 0:
                    _show_cols = [c for c in ['goal_id','l2_name','composite','weakest_dim',
                                              'shock_effect','goal_priority'] if c in _p1_df.columns]
                    st.dataframe(_p1_df[_show_cols].sort_values('composite').round(3),
                                 use_container_width=True, hide_index=True)
                    st.caption("P1 = composite < 0.20, or at-risk with shock exposure, or uncertain and at-risk.")
                else:
                    st.caption("No P1 priority goals at this period.")

        with _dcol3:
            with st.expander("▾ Priority breakdown"):
                if 'goal_priority' in comp_view.columns:
                    _pr = comp_view['goal_priority'].value_counts().reindex(
                        ['P1','P2','P3','P4'], fill_value=0).reset_index()
                    _pr.columns = ['Priority', 'Goals']
                    _pr_desc = {'P1': 'Critical / shock-hit / uncertain+at-risk',
                                'P2': 'At risk or uncertain',
                                'P3': 'Watch — one weak dimension',
                                'P4': 'Healthy'}
                    _pr['Description'] = _pr['Priority'].map(_pr_desc)
                    st.dataframe(_pr, use_container_width=True, hide_index=True)
                elif 'weakest_dim' in comp_view.columns:
                    _wk = comp_view['weakest_dim'].value_counts().reset_index()
                    _wk.columns = ['Dimension', 'Goals']
                    st.dataframe(_wk, use_container_width=True, hide_index=True)
                    _top_wk = _wk.iloc[0]['Dimension'] if len(_wk) > 0 else "N/A"
                    st.caption(f"{_top_wk.capitalize()} is the most common weakest dimension "
                               f"across {n_goals} goals at period {_kpi_sel_p}.")
                else:
                    st.caption("Priority data not available — run composite_score.py.")





    # ── Tab layout ────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Coherence Over Time",
        "🎯 Goal Breakdown",
        "🏢 Portfolio by Bucket",
        "⚡ Shock Analysis",
        "🔮 Forward Projection",
    ])

    # ── TAB 1: Coherence Over Time ────────────────────────────────────────
    with tab1:
        st.subheader("Portfolio Coherence Over Periods")

        # ── Inline period range controls ──────────────────────────────────
        delta_df = None
        p_start = p_end = None
        if comp is not None and 'period_id' in comp.columns and comp['period_id'].nunique() > 1:
            _periods = sorted(comp['period_id'].dropna().unique().tolist())
            _fc1, _fc2, _fc3 = st.columns([2, 2, 3])
            with _fc1:
                _prev = st.selectbox("From period", _periods, index=0, key="t1_prev_period")
            _valid_later = [p for p in _periods if p > _prev] or _periods
            with _fc2:
                _later = st.selectbox("To period", _valid_later, index=len(_valid_later)-1, key="t1_later_period")
            with _fc3:
                _all24 = st.checkbox("All 24 periods", value=False, key="t1_all_periods")
            if _all24:
                _prev, _later = min(_periods), max(_periods)
            if _prev < _later:
                p_start, p_end = _prev, _later
                _ds = comp[comp['period_id'] == p_start].copy()
                _de = comp[comp['period_id'] == p_end].copy()
                if len(_ds) > 0 and len(_de) > 0:
                    delta_df = _de.merge(_ds, on="goal_id", suffixes=("_end", "_start"))
                    for _dc in ["composite","coherence","attainability","relevance","integrity"]:
                        if f"{_dc}_end" in delta_df.columns and f"{_dc}_start" in delta_df.columns:
                            delta_df[f"{_dc}_delta"] = delta_df[f"{_dc}_end"] - delta_df[f"{_dc}_start"]
            st.divider()

        if ts is not None and len(ts) > 0:
            fig = go.Figure()

            if 'budget_shock' in ts.columns:
                shock_b = ts[ts['budget_shock'] == 1]['period_id']
                if len(shock_b):
                    fig.add_vrect(x0=shock_b.min()-0.5, x1=shock_b.max()+0.5,
                                fillcolor=COLORS['budget_shock'], opacity=0.15,
                                layer="below", line_width=0,
                                annotation_text="Budget Shock", annotation_position="top left")

            if 'market_shock' in ts.columns:
                shock_m = ts[ts['market_shock'] == 1]['period_id']
                if len(shock_m):
                    fig.add_vrect(x0=shock_m.min()-0.5, x1=shock_m.max()+0.5,
                                fillcolor=COLORS['market_shock'], opacity=0.15,
                                layer="below", line_width=0,
                                annotation_text="Market Shock", annotation_position="top right")

            fig.add_trace(go.Scatter(
                x=ts['period_id'], y=ts['avg_composite'],
                mode='lines+markers', name='Composite (normalized)',
                line=dict(color=COLORS['composite'], width=3),
                marker=dict(size=6),
            ))

            dim_cols = {
                'avg_coherence': 'Coherence',
                'avg_attainability': 'Attainability',
                'avg_relevance': 'Relevance',
                'avg_integrity': 'Integrity'
            }
            dim_colors = [
                COLORS['coherence'],
                COLORS['attainability'],
                COLORS['relevance'],
                COLORS['integrity']
            ]
            for (col, label), color in zip(dim_cols.items(), dim_colors):
                if col in ts.columns:
                    fig.add_trace(go.Scatter(
                        x=ts['period_id'], y=ts[col],
                        mode='lines', name=label,
                        line=dict(color=color, width=1.5, dash='dot'),
                        opacity=0.7,
                    ))

            fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS['at_risk'],
                        annotation_text="Risk threshold (0.35)")

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(10,8,4,0)",
                plot_bgcolor="rgba(15,11,5,0.6)",
                xaxis_title="Period",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            viz.render(viz.caption_coherence_over_time(ts))

            if 'at_risk_count' in ts.columns:
                _max_risk = int(ts['at_risk_count'].max()) if len(ts) > 0 else 35
                fig2 = go.Figure()

                # Shock bands first
                if 'budget_shock' in ts.columns:
                    _sb = ts[ts['budget_shock'] == 1]['period_id']
                    if len(_sb):
                        fig2.add_vrect(x0=_sb.min()-0.5, x1=_sb.max()+0.5,
                                       fillcolor=COLORS['budget_shock'], opacity=0.12,
                                       layer="below", line_width=0,
                                       annotation_text="Budget shock",
                                       annotation_position="top left",
                                       annotation_font_size=10)
                if 'market_shock' in ts.columns:
                    _sm = ts[ts['market_shock'] == 1]['period_id']
                    if len(_sm):
                        fig2.add_vrect(x0=_sm.min()-0.5, x1=_sm.max()+0.5,
                                       fillcolor=COLORS['market_shock'], opacity=0.10,
                                       layer="below", line_width=0,
                                       annotation_text="Market shock",
                                       annotation_position="top right",
                                       annotation_font_size=10)

                # Bars coloured by severity: green=safe, yellow=warning, red=high risk
                _colors = []
                for v in ts['at_risk_count']:
                    if v <= 8:    _colors.append('#55A868')
                    elif v <= 18: _colors.append('#DD8452')
                    else:         _colors.append('#C44E52')

                fig2.add_trace(go.Bar(
                    x=ts['period_id'],
                    y=ts['at_risk_count'],
                    marker_color=_colors,
                    text=ts['at_risk_count'].astype(int),
                    textposition='outside',
                    textfont=dict(size=11, color='#CBD5E1'),
                    hovertemplate='Period %{x}<br>Goals at risk: %{y}<extra></extra>',
                ))

                # Reference line at 35 (total goals)
                fig2.add_hline(y=35, line_dash="dot", line_color="#64748B",
                               opacity=0.5, annotation_text="All 35 goals",
                               annotation_position="right",
                               annotation_font_size=10)

                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title="Goals at Risk per Period",
                    xaxis=dict(
                        title="Period",
                        tickmode='linear', tick0=1, dtick=1,
                        tickfont=dict(size=11),
                    ),
                    yaxis=dict(
                        title="Goals at risk",
                        range=[0, 40],
                        tickfont=dict(size=11),
                    ),
                    height=320,
                    showlegend=False,
                    bargap=0.25,
                    margin=dict(t=50, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)
                viz.render(viz.caption_at_risk_per_period(ts))


            if delta_df is not None and 'composite_delta' in delta_df.columns:
                st.subheader(f"📊 Change from Period {p_start} → {p_end}")

                avg_delta = delta_df['composite_delta'].mean()
                improving = (delta_df['composite_delta'] > 0).sum()
                degrading = (delta_df['composite_delta'] < 0).sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Δ Composite", f"{avg_delta:.3f}")
                col2.metric("Improving Goals", int(improving))
                col3.metric("Declining Goals", int(degrading))

                _sorted_delta = delta_df.sort_values("composite_delta")
                fig_delta = px.bar(
                    _sorted_delta,
                    x="composite_delta",
                    y=[f"G{int(g)}" for g in _sorted_delta['goal_id']],
                    orientation="h",
                    color="composite_delta",
                    color_continuous_scale="RdYlGn",
                    title=""
                )
                fig_delta.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=max(400, len(delta_df) * 20)
                )
                st.plotly_chart(fig_delta, use_container_width=True)

                show_delta_cols = [
                    c for c in ["goal_id", "composite_start", "composite_end", "composite_delta"]
                    if c in delta_df.columns
                ]
                if show_delta_cols:
                    st.dataframe(delta_df[show_delta_cols].round(3), use_container_width=True)
        else:
            st.info("Run composite_score.py with full predictions to see time series.")

        # ── Per-goal drill-down ───────────────────────────────────────────
        _per_goal_selector(
            comp_view=comp[comp['period_id'] == 18].copy() if comp is not None and 'period_id' in comp.columns else comp,
            fwd=fwd, key_prefix="t1",
            title="Drill into any goal — Coherence Over Time view",
            comp_full=comp,
        )

    # ── TAB 2: Goal Breakdown ─────────────────────────────────────────────
    with tab2:
        # ── Inline controls ───────────────────────────────────────────────
        _t2c1, _t2c2, _t2c3, _t2c4 = st.columns([2, 2, 2, 2])
        with _t2c1:
            _t2_periods = sorted(comp['period_id'].dropna().unique().tolist()) \
                          if comp is not None and 'period_id' in comp.columns else [18]
            _t2_sel_p = st.selectbox("Period", _t2_periods,
                                     index=_t2_periods.index(18) if 18 in _t2_periods else len(_t2_periods)-1,
                                     key="t2_period")
        with _t2c2:
            _t2_l1_opts = ["All"] + (sorted(comp["l1_name"].dropna().unique().tolist())
                                     if comp is not None and "l1_name" in comp.columns else [])
            _t2_l1 = st.selectbox("L1 Department", _t2_l1_opts, key="t2_l1")
        with _t2c3:
            if _t2_l1 != "All" and comp is not None and "l2_name" in comp.columns:
                _t2_l2_pool = comp[comp["l1_name"] == _t2_l1]["l2_name"].dropna().unique().tolist()
            else:
                _t2_l2_pool = comp["l2_name"].dropna().unique().tolist() if comp is not None and "l2_name" in comp.columns else []
            _t2_l2 = st.selectbox("L2 Bucket", ["All"] + sorted(_t2_l2_pool), key="t2_l2")
        with _t2c4:
            _t2_atrisk = st.checkbox("At-risk only", False, key="t2_atrisk")

        # Build filtered comp_view for this tab
        _t2_pool = comp.copy() if comp is not None else pd.DataFrame()
        if _t2_l1 != "All" and "l1_name" in _t2_pool.columns:
            _t2_pool = _t2_pool[_t2_pool["l1_name"] == _t2_l1]
        if _t2_l2 != "All" and "l2_name" in _t2_pool.columns:
            _t2_pool = _t2_pool[_t2_pool["l2_name"] == _t2_l2]
        _t2_goal_ids = sorted(_t2_pool["goal_id"].dropna().unique().tolist()) if "goal_id" in _t2_pool.columns else []
        _t2_cv = comp[comp['period_id'] == _t2_sel_p].copy() if comp is not None and 'period_id' in comp.columns else comp_view.copy()
        if _t2_goal_ids:
            _t2_cv = _t2_cv[_t2_cv["goal_id"].isin(_t2_goal_ids)]
        if _t2_atrisk and "at_risk" in _t2_cv.columns:
            _t2_cv = _t2_cv[_t2_cv["at_risk"]]
        comp_view = _t2_cv
        st.subheader(f"Goal Scores at Period {_t2_sel_p}")
        st.divider()

        if comp_view is not None and len(comp_view) > 0:
            col_a, col_b = st.columns([2, 1])

            with col_a:
                dim_data = comp_view[DIMS].values if all(d in comp_view.columns for d in DIMS) else None
                if dim_data is not None:
                    fig = px.imshow(
                        dim_data.T,
                        labels=dict(x="Goal", y="Dimension", color="Score"),
                        x=[f"G{int(g)}" for g in comp_view['goal_id']],
                        y=[d.capitalize() for d in DIMS],
                        color_continuous_scale='RdYlGn',
                        zmin=0, zmax=1,
                        title="Dimension Scores Heatmap (red=poor, green=good)",
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    viz.render(viz.caption_dimension_heatmap(comp_view))

            with col_b:
                if 'weakest_dim' in comp_view.columns:
                    wk = comp_view['weakest_dim'].value_counts().reset_index()
                    wk.columns = ['Dimension','Count']
                    fig = px.bar(
                        wk,
                        x='Dimension',
                        y='Count',
                        title="Most Weakest Dimension",
                        color='Dimension',
                        color_discrete_map={d: COLORS.get(d,'#888') for d in DIMS}
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    viz.render(viz.caption_weakest_dim_distribution(comp_view))

            if 'composite' in comp_view.columns:
                sorted_comp = comp_view.sort_values('composite')
                colors_bar  = [COLORS['at_risk'] if r else COLORS['safe']
                            for r in sorted_comp.get('at_risk', [False]*len(sorted_comp))]
                fig = go.Figure(go.Bar(
                    x=sorted_comp['composite'],
                    y=[f"G{int(g)}" for g in sorted_comp['goal_id']],
                    orientation='h',
                    marker_color=colors_bar,
                    text=sorted_comp['weakest_dim'] if 'weakest_dim' in sorted_comp.columns else None,
                    textposition='outside',
                ))
                fig.add_vline(x=0.35, line_dash="dash", line_color=COLORS['at_risk'])
                fig.add_vline(x=0.20, line_dash="dot",  line_color="#8B0000")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title="",
                    xaxis_range=[0, 1],
                    height=max(400, len(sorted_comp)*20),
                    yaxis=dict(tickfont=dict(size=9)),
                )
                st.plotly_chart(fig, use_container_width=True)
                viz.render(viz.caption_goals_ranked(comp_view))

            st.subheader("Goal Detail Table")
            show_cols = ['goal_id','l2_name','coherence','attainability','relevance',
                        'integrity','composite','final_composite','at_risk','critical',
                        'weakest_dim','goal_priority','shock_effect']
            show_cols = [c for c in show_cols if c in comp_view.columns]
            if show_cols:
                sort_col = 'composite' if 'composite' in show_cols else show_cols[0]
                st.dataframe(
                    comp_view[show_cols].sort_values(sort_col).round(3),
                    use_container_width=True, height=400,
                )
            else:
                st.dataframe(comp_view.round(3), use_container_width=True, height=400)
        else:
            st.info("No goal-level rows are available for the selected period.")

        # ── Per-goal drill-down ───────────────────────────────────────────
        _per_goal_selector(
            comp_view=_t2_cv, fwd=fwd, key_prefix="t2",
            title="Drill into any goal — Full coherence breakdown",
            comp_full=comp,
        )

    # ── TAB 3: Portfolio by Bucket ────────────────────────────────────────
    with tab3:
        # ── Inline L1 filter ──────────────────────────────────────────────
        _t3_l1_opts = ["All"] + (sorted(comp["l1_name"].dropna().unique().tolist())
                                  if comp is not None and "l1_name" in comp.columns else [])
        _t3_l1 = st.selectbox("Filter by L1 Department", _t3_l1_opts, key="t3_l1")
        _t3_port = port.copy() if port is not None else pd.DataFrame()
        if _t3_l1 != "All" and "l1_name" in _t3_port.columns:
            _t3_port = _t3_port[_t3_port["l1_name"] == _t3_l1]
        elif _t3_l1 != "All" and comp is not None and "l1_name" in comp.columns and "l2_name" in comp.columns:
            _l2_in_l1 = comp[comp["l1_name"] == _t3_l1]["l2_name"].dropna().unique().tolist()
            if "l2_name" in _t3_port.columns:
                _t3_port = _t3_port[_t3_port["l2_name"].isin(_l2_in_l1)]
        port = _t3_port
        st.subheader("Portfolio Health by Bucket")
        st.divider()

        if port is not None and len(port) > 0:
            col_a, col_b, col_c = st.columns([1,1,1])

            with col_a:
                fig = px.bar(
                    port.sort_values('avg_composite'),
                    x='avg_composite',
                    y='l2_name',
                    orientation='h',
                    color='at_risk_count' if 'at_risk_count' in port.columns else None,
                    color_continuous_scale='RdYlGn_r',
                    title="",
                    text='avg_composite',
                )
                fig.add_vline(x=0.35, line_dash="dash", line_color=COLORS['at_risk'])
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=500,
                    xaxis_range=[0,1]
                )
                st.plotly_chart(fig, use_container_width=True)
                viz.render(viz.caption_avg_composite_per_bucket(port))

            with col_b:
                if all(c in port.columns for c in ['avg_coherence','avg_attainability',
                                                'avg_relevance','avg_integrity']):
                    top_buckets = port.nlargest(5,'avg_composite')['l2_name'].tolist()
                    fig = go.Figure()
                    cats = ['Coherence','Attainability','Relevance','Integrity']
                    for _, row in port[port['l2_name'].isin(top_buckets)].iterrows():
                        vals = [row['avg_coherence'],row['avg_attainability'],
                                row['avg_relevance'],row['avg_integrity']]
                        fig.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]],
                            theta=cats + [cats[0]],
                            fill='toself', name=row['l2_name'], opacity=0.6,
                        ))
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        polar=dict(radialaxis=dict(range=[0,1])),
                        title="Top 5 Buckets - Dimension Radar",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    viz.render(viz.caption_top_buckets_radar(port))

            if pts is not None:
                st.subheader("Bucket Coherence Over Time")
                bucket_sel = st.selectbox("Select Bucket", sorted(pts['l2_name'].unique()))
                pts_bucket = pts[pts['l2_name'] == bucket_sel]
                fig = px.line(
                    pts_bucket,
                    x='period_id',
                    y='avg_composite',
                    title=f"{bucket_sel} - Composite Over Time"
                )
                fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS['at_risk'])
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_range=[0,1],
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                viz.render(viz.caption_bucket_over_time(pts_bucket, bucket_sel))
        else:
            st.info("Portfolio bucket data is not available. Make sure the loaded scores include `l2_name` and `composite` columns.")

        # ── Per-goal drill-down ───────────────────────────────────────────
        _t3_anchor = comp[comp['period_id'] == 18].copy() if comp is not None and 'period_id' in comp.columns else comp
        if _t3_l1 != "All" and _t3_anchor is not None and "l1_name" in _t3_anchor.columns:
            _t3_anchor = _t3_anchor[_t3_anchor["l1_name"] == _t3_l1]

    # ── TAB 4: Shock Analysis ─────────────────────────────────────────────
    with tab4:
        # ── Inline filter strip (uniform with other tabs) ─────────────────
        _t4fc1, _t4fc2, _t4fc3 = st.columns([2, 2, 2])
        with _t4fc1:
            _t4_l1_opts = ["All"] + (sorted(comp["l1_name"].dropna().unique().tolist())
                                      if comp is not None and "l1_name" in comp.columns else [])
            _t4_l1 = st.selectbox("L1 Department", _t4_l1_opts, key="t4_l1")
        with _t4fc2:
            if _t4_l1 != "All" and comp is not None and "l2_name" in comp.columns:
                _t4_l2_pool = comp[comp["l1_name"] == _t4_l1]["l2_name"].dropna().unique().tolist()
            else:
                _t4_l2_pool = comp["l2_name"].dropna().unique().tolist() if comp is not None and "l2_name" in comp.columns else []
            _t4_l2 = st.selectbox("L2 Bucket", ["All"] + sorted(_t4_l2_pool), key="t4_l2")
        with _t4fc3:
            _t4_vuln_only = st.checkbox("Shock-vulnerable only", False, key="t4_vuln_only",
                                        help="Show only goals flagged as market shock vulnerable")
        st.subheader("Shock Impact Analysis")
        st.divider()

        # Build filtered comp for this tab
        _t4_comp = comp.copy() if comp is not None else pd.DataFrame()
        if _t4_l1 != "All" and "l1_name" in _t4_comp.columns:
            _t4_comp = _t4_comp[_t4_comp["l1_name"] == _t4_l1]
        if _t4_l2 != "All" and "l2_name" in _t4_comp.columns:
            _t4_comp = _t4_comp[_t4_comp["l2_name"] == _t4_l2]
        if _t4_vuln_only and "market_shock_vulnerable" in _t4_comp.columns:
            _t4_comp = _t4_comp[_t4_comp["market_shock_vulnerable"] > 0]
        _t4_goal_ids = sorted(_t4_comp["goal_id"].dropna().unique().tolist()) if "goal_id" in _t4_comp.columns else []

        # ── Rebuild ts for filtered goals ─────────────────────────────────
        _t4_ts = ts  # default to portfolio-wide
        if len(_t4_goal_ids) > 0 and len(_t4_goal_ids) < (comp["goal_id"].nunique() if comp is not None else 999):
            _t4_ts_raw = _t4_comp.groupby("period_id", as_index=False).agg({
                "composite": "mean", "coherence": "mean", "attainability": "mean",
                "relevance": "mean", "integrity": "mean",
            }).rename(columns={"composite": "avg_composite", "coherence": "avg_coherence",
                               "attainability": "avg_attainability", "relevance": "avg_relevance",
                               "integrity": "avg_integrity"})
            if ts is not None:
                _shock_cols = [c for c in ["budget_shock","market_shock","any_shock","shock_label"] if c in ts.columns]
                if _shock_cols:
                    _t4_ts_raw = _t4_ts_raw.merge(ts[["period_id"] + _shock_cols], on="period_id", how="left")
            _t4_ts = _t4_ts_raw

        if _t4_ts is not None and len(_t4_ts) > 0 and "avg_composite" in _t4_ts.columns:

            # ── 1. Shock phase summary + metrics ──────────────────────────
            col_a, col_b = st.columns(2)

            pre_shock   = _t4_ts[_t4_ts['period_id'].isin([7,8,9])]['avg_composite'].mean()
            during_b    = _t4_ts[_t4_ts['budget_shock'] == 1]['avg_composite'].mean() if 'budget_shock' in _t4_ts.columns else _t4_ts[_t4_ts['period_id'].isin([10,11,12])]['avg_composite'].mean()
            during_m    = _t4_ts[_t4_ts['market_shock'] == 1]['avg_composite'].mean() if 'market_shock' in _t4_ts.columns else _t4_ts[_t4_ts['period_id'].isin([14,15,16,17])]['avg_composite'].mean()
            post_shock  = _t4_ts[_t4_ts['period_id'].isin([20,21,22])]['avg_composite'].mean()

            with col_a:
                shock_data = pd.DataFrame({
                    'Phase'    : ['Pre-shock\n(P7-9)', 'Budget shock\n(P10-12)',
                                  'Market shock\n(P14-17)', 'Recovery\n(P20-22)'],
                    'Composite': [pre_shock, during_b, during_m or 0, post_shock],
                    'Color'    : ['#D4A843', COLORS['budget_shock'], COLORS['market_shock'], '#55A868'],
                })
                fig = px.bar(shock_data, x='Phase', y='Composite',
                             title="",
                             color='Phase', color_discrete_sequence=shock_data['Color'].tolist())
                fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS['at_risk'])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0,1],
                                  height=360, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                viz.render(viz.caption_shock_phases(pre_shock, during_b, during_m, post_shock))

            with col_b:
                st.metric("Pre-shock coherence",   f"{pre_shock:.3f}")
                st.metric("Budget shock drop",     f"{during_b - pre_shock:.3f}",
                          delta=f"{during_b - pre_shock:.3f}", delta_color="inverse")
                st.metric("Market shock impact",   f"{during_m:.3f}",
                          delta=f"{during_m - during_b:.3f}", delta_color="inverse")
                st.metric("Post-shock recovery",   f"{post_shock:.3f}",
                          delta=f"{post_shock - during_b:.3f}", delta_color="normal")
                _rp_cands = _t4_ts[_t4_ts['avg_composite'] >= pre_shock - 0.01]['period_id']
                _rp = _rp_cands[_rp_cands > 12].min() if len(_rp_cands) > 0 else "TBD"
                st.metric("Recovery period", str(_rp))

            st.divider()

            # ── 2. Shock phase × dimension breakdown (grouped bars) ────────
            st.subheader("Dimension Impact by Shock Phase")
            _dims_phase = []
            for _dim in ["coherence", "relevance", "integrity", "attainability"]:
                _col = f"avg_{_dim}"
                if _col in _t4_ts.columns:
                    _dims_phase.append({
                        "Dimension": _dim.capitalize(),
                        "Pre-shock (P7-9)": _t4_ts[_t4_ts['period_id'].isin([7,8,9])][_col].mean(),
                        "Budget shock (P10-12)": _t4_ts[_t4_ts['period_id'].isin([10,11,12])][_col].mean(),
                        "Market shock (P14-17)": _t4_ts[_t4_ts['period_id'].isin([14,15,17])][_col].mean(),
                        "Recovery (P20-22)": _t4_ts[_t4_ts['period_id'].isin([20,21,22])][_col].mean(),
                    })
            if _dims_phase:
                _dp_df = pd.DataFrame(_dims_phase).melt(
                    id_vars="Dimension", var_name="Phase", value_name="Score")
                fig = px.bar(_dp_df, x="Dimension", y="Score", color="Phase", barmode="group",
                             title="",
                             color_discrete_sequence=['#D4A843', COLORS['budget_shock'],
                                                      COLORS['market_shock'], '#55A868'])
                fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS['at_risk'],
                              annotation_text="Risk threshold", annotation_position="top right")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0,1], height=380,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Coherence took the largest drop during the budget shock. Attainability remained flat throughout — a dataset property rather than a shock response.")

            st.divider()

            # ── 3. Goal vs Period composite heatmap ────────────────────────
            st.subheader("Goal vs Period Heatmap")
            if _t4_comp is not None and "period_id" in _t4_comp.columns and "composite" in _t4_comp.columns:
                _hm = _t4_comp.pivot_table(index="goal_id", columns="period_id", values="composite")
                _hm_goals = [f"G{int(g)}" for g in _hm.index]
                fig = go.Figure(data=go.Heatmap(
                    z=_hm.values,
                    x=[str(int(p)) for p in _hm.columns],
                    y=_hm_goals,
                    colorscale="RdYlGn",
                    zmin=0, zmax=1,
                    colorbar=dict(title="Composite"),
                    hoverongaps=False,
                ))
                # Budget shock band p10-12, market shock p14-17
                fig.add_vrect(x0="9", x1="12", fillcolor=COLORS["budget_shock"],
                              opacity=0.15, layer="below", line_width=0,
                              annotation_text="Budget shock",
                              annotation_position="top left",
                              annotation_font_color="#C44E52")
                fig.add_vrect(x0="13", x1="17", fillcolor=COLORS["market_shock"],
                              opacity=0.12, layer="below", line_width=0,
                              annotation_text="Market shock",
                              annotation_position="top right",
                              annotation_font_color="#DD8452")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Period",
                    yaxis_title="Goal",
                    height=max(420, len(_hm_goals) * 18),
                    xaxis=dict(type="category"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red = low composite, green = high. Budget shock (p10-12) and market shock (p14-17) are visible as column-wide colour drops. Goals that are darker throughout are structurally at risk, not just shock-affected.")

            st.divider()

            # ── 4. Shock Effect per Goal ──────────────────────────────────
            st.subheader("Shock Effect per Goal (p18)")
            _se_df = _t4_comp[_t4_comp["period_id"] == 18].copy() \
                     if "period_id" in _t4_comp.columns else _t4_comp.copy()
            if "shock_effect" in _se_df.columns and len(_se_df) > 0:
                _se_df = _se_df.sort_values("shock_effect")
                _se_df["goal_label"] = [
                    f"G{int(g)} — {_se_df[_se_df['goal_id']==g]['l2_name'].values[0]}"
                    if "l2_name" in _se_df.columns and len(_se_df[_se_df['goal_id']==g]) > 0
                    else f"G{int(g)}"
                    for g in _se_df["goal_id"]
                ]
                _se_affected = (_se_df["shock_effect"] < -0.01).sum()
                fig = px.bar(
                    _se_df, x="shock_effect", y="goal_label", orientation="h",
                    color="shock_effect",
                    color_continuous_scale=[[0,"#C44E52"],[0.5,"#FBBF24"],[1,"#64748B"]],
                    labels={"shock_effect": "Shock Effect (Δ composite)", "goal_label": "Goal"},
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=max(380, len(_se_df) * 20),
                    coloraxis_showscale=False,
                    xaxis=dict(zeroline=True, zerolinecolor="#64748B", zerolinewidth=1.5),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"{_se_affected} goals had shock-rule adjustments applied at p18. "
                    f"Goals with no shock exposure show 0. "
                    f"Larger negative values indicate higher shock sensitivity."
                )
            else:
                st.caption("Shock effect not available — run composite_score.py to generate it.")

            st.divider()

            # ── 5. Composite Recovery Ranking (p18 → p24) ────────────────
            # Uses composite_p6 (projected p24) minus composite_adjusted (p18 anchor).
            # Attainability is excluded here because its ground truth is binary at 0.0
            # for all 35 goals at p24 — showing attainability delta produces a
            # misleading chart. Composite uses all four per-goal slopes and gives
            # genuine per-goal differentiation.
            st.subheader("Composite Recovery Ranking (p18 → p24)")
            if fwd is not None and "goal_id" in fwd.columns and "composite_p6" in fwd.columns \
                    and "composite_adjusted" in fwd.columns:
                _rec = fwd.copy()
                if _t4_goal_ids:
                    _rec = _rec[_rec["goal_id"].isin(_t4_goal_ids)]
                _rec["composite_delta"] = _rec["composite_p6"] - _rec["composite_adjusted"]
                _rec = _rec.sort_values("composite_delta")
                _rec["goal_label"] = [f"G{int(g)}" for g in _rec["goal_id"]]

                # Add L2 name for richer labels
                if comp is not None and "goal_id" in comp.columns and "l2_name" in comp.columns:
                    _l2map = comp[["goal_id","l2_name"]].drop_duplicates().set_index("goal_id")["l2_name"].to_dict()
                    _rec["goal_label"] = [
                        f"G{int(g)} — {_l2map.get(g,'')}" if _l2map.get(g) else f"G{int(g)}"
                        for g in _rec["goal_id"]
                    ]

                fig = px.bar(
                    _rec, x="composite_delta", y="goal_label", orientation="h",
                    color="composite_delta",
                    color_continuous_scale=[[0,"#C44E52"],[0.4,"#f0c040"],[1,"#55A868"]],
                    title="",
                    labels={"composite_delta": "Δ Composite", "goal_label": "Goal"},
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=max(420, len(_rec) * 22),
                    coloraxis_showscale=False,
                    xaxis=dict(zeroline=True, zerolinecolor="#64748B", zerolinewidth=1.5),
                )
                st.plotly_chart(fig, use_container_width=True)

                _n_imp = (_rec["composite_delta"] > 0.01).sum()
                _n_stb = ((_rec["composite_delta"] >= -0.01) & (_rec["composite_delta"] <= 0.01)).sum()
                _n_deg = (_rec["composite_delta"] < -0.01).sum()
                st.caption(
                    f"{_n_imp} goals projected to improve composite score by p24. "
                    f"{_n_stb} stable (within ±0.01). {_n_deg} goals projected to decline. "
                    f"Based on per-goal linear slopes across Coherence, Relevance, Integrity, "
                    f"and Attainability from p1 to p18."
                )

                # Note on attainability
                st.info(
                    "ℹ️  Attainability is excluded from this ranking because its ground truth label "
                    "is binary in this dataset — 0.1 at p18 and 0.0 at p24 for all 35 goals uniformly. "
                    "No slope-based projection adds signal. The composite delta above uses Coherence, "
                    "Relevance, and Integrity slopes where genuine per-goal variation exists."
                )

            st.divider()

            # ── 5. Market shock vulnerable goals ──────────────────────────
            if "market_shock_vulnerable" in _t4_comp.columns:
                st.subheader("Market Shock Vulnerable Goals")
                _vuln = _t4_comp[(_t4_comp.get("period_id", pd.Series(dtype=int)) == 18) & (_t4_comp["market_shock_vulnerable"] > 0)] \
                        if "period_id" in _t4_comp.columns else _t4_comp[_t4_comp["market_shock_vulnerable"] > 0]
                _show_cols = [c for c in ["goal_id","l2_name","coherence","attainability","composite","at_risk"] if c in _vuln.columns]
                if len(_vuln) > 0 and _show_cols:
                    st.dataframe(_vuln[_show_cols].round(3), use_container_width=True)
                viz.render(viz.caption_market_shock_vulnerable(_vuln))

            # ── 6. Goal trajectory through shocks ─────────────────────────
            st.subheader("Goal Trajectories Through Shock Periods")
            _t4_goal_names = []
            for _gid in _t4_goal_ids:
                _row = _t4_comp[_t4_comp["goal_id"] == _gid]
                _l2 = _row["l2_name"].values[0] if "l2_name" in _row.columns and len(_row) > 0 else ""
                _t4_goal_names.append(f"G{int(_gid)} — {_l2}" if _l2 else f"G{int(_gid)}")

            _t4_selected_labels = st.multiselect(
                "Select goals to trace (up to 12)",
                _t4_goal_names,
                default=_t4_goal_names[:6] if len(_t4_goal_names) >= 6 else _t4_goal_names,
                key="t4_goal_select",
                help="Each selected goal gets a line showing its composite score across all 24 periods."
            )
            _t4_selected_goals = [int(lbl.split("—")[0].strip()[1:]) for lbl in _t4_selected_labels if lbl]

            if _t4_selected_goals and "period_id" in comp.columns and "composite" in comp.columns:
                _show_goals = _t4_selected_goals[:12]
                _traj_df = comp[comp["goal_id"].isin(_show_goals)].sort_values(["goal_id","period_id"])
                fig = go.Figure()
                if "budget_shock" in _t4_ts.columns:
                    _sb = _t4_ts[_t4_ts["budget_shock"] == 1]["period_id"]
                    if len(_sb):
                        fig.add_vrect(x0=_sb.min()-0.5, x1=_sb.max()+0.5, fillcolor=COLORS["budget_shock"],
                                      opacity=0.15, layer="below", line_width=0,
                                      annotation_text="Budget shock", annotation_position="top left")
                if "market_shock" in _t4_ts.columns:
                    _sm = _t4_ts[_t4_ts["market_shock"] == 1]["period_id"]
                    if len(_sm):
                        fig.add_vrect(x0=_sm.min()-0.5, x1=_sm.max()+0.5, fillcolor=COLORS["market_shock"],
                                      opacity=0.15, layer="below", line_width=0,
                                      annotation_text="Market shock", annotation_position="top right")
                for _gid in _show_goals:
                    _g = _traj_df[_traj_df["goal_id"] == _gid]
                    _label_row = _t4_comp[_t4_comp["goal_id"] == _gid]
                    _l2 = _label_row["l2_name"].values[0] if "l2_name" in _label_row.columns and len(_label_row) > 0 else ""
                    fig.add_trace(go.Scatter(
                        x=_g["period_id"], y=_g["composite"],
                        mode="lines+markers",
                        name=f"G{int(_gid)} — {_l2}" if _l2 else f"G{int(_gid)}",
                        line=dict(width=2), marker=dict(size=5),
                    ))
                fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS["at_risk"],
                              annotation_text="Risk threshold")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Period",
                                  yaxis_title="Composite", yaxis_range=[0,1], height=440,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                  hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                viz.render(viz.caption_selected_goals_trajectory(_traj_df, _t4_ts, len(_show_goals)))

        else:
            st.info("Shock data not available. Run composite_score.py with full predictions.")

    # ── TAB 5: Forward Projection ─────────────────────────────────────────
    with tab5:
        # ── Inline goal filter ────────────────────────────────────────────
        _t5c1, _t5c2, _t5c3 = st.columns([2, 2, 2])
        with _t5c1:
            _t5_l1_opts = ["All"] + (sorted(comp["l1_name"].dropna().unique().tolist())
                                      if comp is not None and "l1_name" in comp.columns else [])
            _t5_l1 = st.selectbox("L1 Department", _t5_l1_opts, key="t5_l1")
        with _t5c2:
            if _t5_l1 != "All" and comp is not None and "l2_name" in comp.columns:
                _t5_l2_pool = comp[comp["l1_name"] == _t5_l1]["l2_name"].dropna().unique().tolist()
            else:
                _t5_l2_pool = comp["l2_name"].dropna().unique().tolist() if comp is not None and "l2_name" in comp.columns else []
            _t5_l2 = st.selectbox("L2 Bucket", ["All"] + sorted(_t5_l2_pool), key="t5_l2")
        with _t5c3:
            _t5_atrisk = st.checkbox("At-risk only", False, key="t5_atrisk")

        if fwd is not None and "goal_id" in fwd.columns:
            _t5_fwd = fwd.copy()
            if _t5_l1 != "All" and comp is not None and "goal_id" in comp.columns and "l1_name" in comp.columns:
                _t5_goal_ids = comp[comp["l1_name"] == _t5_l1]["goal_id"].unique().tolist()
                _t5_fwd = _t5_fwd[_t5_fwd["goal_id"].isin(_t5_goal_ids)]
            if _t5_l2 != "All" and comp is not None and "goal_id" in comp.columns and "l2_name" in comp.columns:
                _t5_goal_ids = comp[comp["l2_name"] == _t5_l2]["goal_id"].unique().tolist()
                _t5_fwd = _t5_fwd[_t5_fwd["goal_id"].isin(_t5_goal_ids)]
            if _t5_atrisk and "at_risk" in _t5_fwd.columns:
                _t5_fwd = _t5_fwd[_t5_fwd["at_risk"]]
            fwd = _t5_fwd

        st.subheader("Forward Projection")
        st.divider()

        if fwd is not None and len(fwd) > 0:

            # ── Row 1: portfolio status ───────────────────────────────────
            _n_imp = int(fwd['improving_p6'].sum()) if 'improving_p6' in fwd.columns else 0
            _n_deg = int(fwd['degrading_p6'].sum()) if 'degrading_p6' in fwd.columns else 0
            _n_stb = len(fwd) - _n_imp - _n_deg

            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Goals improving (+6)", _n_imp)
            r1c2.metric("Goals stable (+6)",    _n_stb)
            r1c3.metric("Goals degrading (+6)", _n_deg)
            r1c4.metric("Goals at risk",
                        int((fwd['composite_adjusted'] < 0.35).sum())
                        if 'composite_adjusted' in fwd.columns else "N/A")

            st.divider()

            # ── Row 2: MAE metrics ────────────────────────────────────────
            _comp_mae = (fwd["composite_p6"] - fwd["composite_adjusted"]).abs().mean()

            _has_dim_errors = any(c in fwd.columns for c in
                ['coherence_error_p24','relevance_error_p24',
                 'integrity_error_p24','attain_error_p24'])

            if _has_dim_errors:
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Composite MAE\np18→p24",    f"{_comp_mae:.4f}")
                m2.metric("Coherence MAE\np18→p24",
                          f"{fwd['coherence_error_p24'].dropna().mean():.4f}"
                          if 'coherence_error_p24' in fwd.columns else "N/A")
                m3.metric("Relevance MAE\np18→p24",
                          f"{fwd['relevance_error_p24'].dropna().mean():.4f}"
                          if 'relevance_error_p24' in fwd.columns else "N/A")
                m4.metric("Integrity MAE\np18→p24",
                          f"{fwd['integrity_error_p24'].dropna().mean():.4f}"
                          if 'integrity_error_p24' in fwd.columns else "N/A")
                m5.metric("Attainability MAE\np18→p24",
                          f"{fwd['attain_error_p24'].dropna().mean():.4f}"
                          if 'attain_error_p24' in fwd.columns else "N/A")
                m6.metric("Benchmark", "0.272")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Composite MAE (p18→p24)", f"{_comp_mae:.4f}")
                m2.metric("Benchmark", "0.272")
                m3.metric("Attainability MAE (p18→p24)",
                          f"{fwd['attain_error_p24'].dropna().mean():.4f}"
                          if 'attain_error_p24' in fwd.columns else "N/A")

            st.divider()

            projection_cols = [c for c in ['composite_adjusted','composite','composite_p6','composite_p12','actual_composite_p24'] if c in fwd.columns]

            if len(projection_cols) >= 2:
                fig = go.Figure()
                x = fwd["goal_id"].tolist()
                name_map = {
                    'composite_adjusted': 'Period 18',
                    'composite': 'Current composite',
                    'composite_p6': 'Period 24 (projected)',
                    'composite_p12': 'Period 30 (future)',
                    'actual_composite_p24': 'Period 24 (actual)',
                }
                for col in projection_cols:
                    fig.add_trace(go.Scatter(
                        x=x, y=fwd[col], mode='markers+lines',
                        name=name_map.get(col, col),
                        customdata=fwd["goal_id"],
                        hovertemplate="<b>G%{customdata}</b><br>Score: %{y:.4f}<extra>" + name_map.get(col, col) + "</extra>",
                    ))
                fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS['at_risk'])
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title="",
                    xaxis_title="Goal ID",
                    yaxis_title="Composite score",
                    yaxis_range=[0,1],
                    height=450,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
                viz.render(viz.caption_forward_projection(fwd))
            else:
                st.info("Forward projection file loaded, but no compatible projection columns were found.")
                st.dataframe(fwd.head(50), use_container_width=True)

            if 'actual_composite_p24' in fwd.columns:
                st.subheader("Projection Validation — Predicted vs Actual at Period 24")
                val_cols = ['goal_id','composite_adjusted','composite_p6',
                            'actual_composite_p24','composite_error_p24']
                val_cols = [c for c in val_cols if c in fwd.columns]
                val_df   = fwd[val_cols].sort_values('composite_error_p24').round(3)                     if 'composite_error_p24' in fwd.columns else fwd[val_cols].round(3)
                val_df.columns = ['Goal','P18 Composite','Projected P24',
                                  'Actual P24','Error'][:len(val_cols)]
                st.dataframe(val_df, use_container_width=True, height=400)
                viz.render(viz.caption_projection_validation(fwd))
        else:
            st.info("Forward projection data not available. Run composite_score.py.")

        # ── Per-goal drill-down ───────────────────────────────────────────
        _per_goal_selector(
            comp_view=comp[comp['period_id'] == 18].copy()
                if comp is not None and 'period_id' in comp.columns else comp,
            fwd=fwd, key_prefix="t5",
            title="Drill into any goal — Forward projection view",
            comp_full=comp,
        )
# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5: ABOUT — non-technical primer + team
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "about":
 
    # Hero
    st.markdown("""
    <div class="hero-panel">
        <div style="display:flex;align-items:center;margin-bottom:8px">
            <span class="decidr-mark"></span>
            <span style="font-size:2.4rem;font-weight:800;letter-spacing:-0.03em;color:#F8FAFC">
                Decidr Coherence Engine
            </span>
        </div>
        <div style="color:#94A3B8;font-size:1.05rem;margin-top:4px">
            What if your resources could tell you when they were lying to you?
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # The pitch
    st.markdown("""
### The problem
 
Organisations invest millions across competing goals with no reliable way to know whether
those investments make sense. Existing tools track *what* happened. None of them answer the
harder question: **is what we're doing appropriate?** Underfunding a critical goal is wasteful.
So is overfunding one. Most organisations can't tell the difference until it's too late.
 
### What this system does
 
The Coherence Engine asks a different question: not *what happened*, but *whether what
happened makes sense*. It scores every goal in your portfolio across four dimensions —
**coherence, attainability, relevance, integrity** — and rolls them up into a single
composite health score per goal, per period.
 
A goal drowning in resources scores just as poorly as one being starved. **Both directions matter.**
    """)
 
    # 4 steps panel
    st.markdown('<div class="step-label">How it works</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    step_data = [
        ("1", "Data Layer",         "Your data is organised — 8 source datasets, 35 goals across 24 periods, 33 engineered features."),
        ("2", "Input Layer",        "Your question is understood — plain English in, structured analysis brief out. No forms, no filters."),
        ("3", "Intelligence Core",  "Every goal is scored — multi-LLM scoring + Gaussian-process calibration + 9 verification rules."),
        ("4", "Output Layer",       "Your answer is written — scores become sentences, with reasoning and uncertainty flagged."),
    ]
    for col, (num, title, body) in zip([s1, s2, s3, s4], step_data):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="min-height:170px">
                <div style="font-size:11px;color:#a78bfa;font-weight:700;letter-spacing:0.1em">STEP {num}</div>
                <div style="font-size:1.05rem;font-weight:700;color:#F8FAFC;margin:6px 0 8px 0">{title}</div>
                <div style="font-size:0.86rem;color:#cbd5e1;line-height:1.5">{body}</div>
            </div>
            """, unsafe_allow_html=True)
 
    # 4 dimensions explainer
    st.markdown('<div class="step-label">The four dimensions</div>', unsafe_allow_html=True)
    dim_explain = [
        ("Coherence",     "Are decisions consistent across levels, goals, and time?",     "35%", COLORS["coherence"]),
        ("Attainability", "Is the goal realistically achievable given current trajectory?", "25%", COLORS["attainability"]),
        ("Relevance",     "Is the resource allocation justified against stated goals?",    "20%", COLORS["relevance"]),
        ("Integrity",     "Are assumptions transparent and auditable?",                    "20%", COLORS["integrity"]),
    ]
    for name, desc, weight, color in dim_explain:
        st.markdown(f"""
        <div class="goal-card" style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
                <div>
                    <span style="font-size:1.05rem;font-weight:700;color:{color}">{name}</span>
                    <span style="font-size:0.88rem;color:#94A3B8;margin-left:8px">{desc}</span>
                </div>
                <span style="font-size:0.95rem;font-weight:700;color:{color}">{weight}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    # What makes it different
    st.markdown('<div class="step-label">What makes it different</div>', unsafe_allow_html=True)
    st.markdown("""
- **Symmetric standard.** Too little is as problematic as too much. Almost no other system flags overfunding.
- **Four dimensions, not just performance** — alignment, delivery, achievability, consistency.
- **Every answer comes with reasoning** — not a number, a conclusion.
- **Reproducible** — the same question on Monday returns the same rigorous answer on Friday.
    """)
 
    # The team
    st.markdown('<div class="step-label">The team</div>', unsafe_allow_html=True)
    st.markdown("**Team 14-02 · UTS MDSI Capstone (36127)**")
    st.caption("In partnership with Decidr · Industry mentor: Pouya Salehpour · Product lead: Tom")
 
    team_members = [
        ("Unni",        "Project Lead",                   "Architecture · Stakeholder management"),
        ("Padmasri",          "System 1 — Dashboard and Front-end",           "Goal-input flow · Output reports · Dashboard improvements and UI/UX"),
        ("Subhan",     "System 1 — Dashboard",           "Portfolio dashboard · KPI design"),
        ("Anupam",            "System 2 — Verification",        "Verification rules · System 1+2 integration"),
        ("Adrian",            "System 2 — Calibration",         "Gaussian process · Composite scoring"),
        ("Fatemeh",           "System 2 — Forward projection",  "Trajectory modelling · Validation framework")
    ]
    cols_per_row = 3
    for i in range(0, len(team_members), cols_per_row):
        row = team_members[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (name, role, work) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="goal-card" style="min-height:140px">
                    <div style="font-size:0.95rem;font-weight:700;color:#F8FAFC">{name}</div>
                    <div style="font-size:0.78rem;color:#a78bfa;font-weight:600;margin:4px 0 8px 0;letter-spacing:0.04em">{role}</div>
                    <div style="font-size:0.78rem;color:#94A3B8;line-height:1.45">{work}</div>
                </div>
                """, unsafe_allow_html=True)
        # spacer
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
 
    # CTAs
    st.divider()
    cta1, cta2 = st.columns(2)
    with cta1:
        if st.button("📊 View the Dashboard", use_container_width=True, type="primary"):
            st.session_state.page = "portfolio"; st.rerun()
    with cta2:
        if st.button("🎯 Score a New Goal", use_container_width=True):
            st.session_state.page = "input"; st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption("Decidr Coherence Engine - Team 14-02 · iLab Capstone 36127 · UTS MDSI")
