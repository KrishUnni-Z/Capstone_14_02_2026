"""
viz.py — Plot caption helpers for the Decidr Coherence Engine dashboard.

Each function takes the data shown in a plot and returns a one-line
plain-English explanation. Captions are data-driven (if/else branches based
on what the numbers say) so they stay accurate when filters change.

Column-name expectations come from output_reference.md (System 2 spec):
- composite_scores_poc.csv  (per goal × period)
- coherence_timeseries_poc.csv  (per period)
- portfolio_summary_poc.csv  (per L2 bucket)
- forward_projection_poc.csv  (per goal)
- portfolio_timeseries_poc.csv  (per period × L2 bucket)

Pattern: caption_<chart_name>(args...) -> str
Call site: st.caption(caption_xxx(...))
"""

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_mean(series, default=0.0):
    """Return mean or default for empty/NaN series."""
    try:
        if series is None or len(series) == 0:
            return default
        v = series.mean()
        return default if v != v else float(v)   # NaN check
    except Exception:
        return default


def _trend_word(delta, threshold=0.02):
    """Map a numeric change to plain-English trend word."""
    if delta > threshold:   return "improving"
    if delta < -threshold:  return "degrading"
    return "stable"


def _risk_word(score, risk_thresh=0.35, crit_thresh=0.20):
    """Map a composite score to a risk band label."""
    if score < crit_thresh: return "critical"
    if score < risk_thresh: return "at risk"
    if score < 0.50:        return "below mean"
    if score < 0.70:        return "on track"
    return "strong"


def render(caption_text):
    """Render a caption with consistent styling."""
    if caption_text:
        st.caption(caption_text)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Coherence Over Time
# ─────────────────────────────────────────────────────────────────────────────
def caption_coherence_over_time(ts_df):
    """Headline time-series chart with shock shading.

    `ts_df` = coherence_timeseries_poc.csv: period_id, avg_composite,
              avg_coherence, budget_shock, market_shock, at_risk_count.
    """
    if ts_df is None or len(ts_df) == 0:
        return None

    pre  = _safe_mean(ts_df[ts_df["period_id"].isin([7, 8, 9])]["avg_composite"])
    post = _safe_mean(ts_df[ts_df["period_id"].isin([20, 21, 22])]["avg_composite"])

    trough = None
    if "budget_shock" in ts_df.columns and ts_df["budget_shock"].any():
        trough = _safe_mean(ts_df[ts_df["budget_shock"] == 1]["avg_composite"])

    if trough is not None and pre > 0:
        drop_pct = (pre - trough) / pre * 100
        if post >= pre - 0.02:
            return (f"📉 Composite dropped **{drop_pct:.0f}%** during budget shock "
                    f"and **fully recovered** to {post:.2f} by P22 (baseline {pre:.2f}).")
        else:
            gap = (pre - post) / pre * 100
            return (f"📉 Composite dropped **{drop_pct:.0f}%** during budget shock; "
                    f"**still {gap:.0f}% below baseline** by P22 ({post:.2f} vs {pre:.2f}).")
    avg = _safe_mean(ts_df["avg_composite"])
    band = "above" if avg >= 0.35 else "below"
    return f"📊 Portfolio composite averaged **{avg:.2f}** across {len(ts_df)} periods — {band} the 0.35 risk threshold."


def caption_at_risk_per_period(ts_df):
    """Bar chart of at-risk goal count per period."""
    if ts_df is None or len(ts_df) == 0 or "at_risk_count" not in ts_df.columns:
        return None
    peak_n = int(ts_df["at_risk_count"].max())
    if peak_n == 0:
        return "✅ **No goals breached** the risk threshold across any period."
    peak_p = int(ts_df.loc[ts_df["at_risk_count"].idxmax(), "period_id"])
    avg_n  = ts_df["at_risk_count"].mean()
    return f"⚠️ At-risk goals peaked at **{peak_n} at period {peak_p}** (avg {avg_n:.1f} per period across all 24)."


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Goal Breakdown
# ─────────────────────────────────────────────────────────────────────────────
def caption_dimension_heatmap(comp_view):
    """Per-goal × per-dimension heatmap.

    `comp_view` = composite_scores_poc filtered to anchor period.
    """
    if comp_view is None or len(comp_view) == 0:
        return None
    dims = ["coherence", "attainability", "relevance", "integrity"]
    means = {d: _safe_mean(comp_view[d]) for d in dims if d in comp_view.columns}
    if not means:
        return None
    weakest = min(means, key=means.get)
    strongest = max(means, key=means.get)
    spread = means[strongest] - means[weakest]
    if spread > 0.30:
        return (f"🔍 **{weakest.capitalize()}** is the portfolio's weakest dimension at "
                f"{means[weakest]:.2f}, far behind {strongest} at {means[strongest]:.2f} "
                f"(spread of {spread:.2f}) — uneven dimensional health.")
    return (f"🔍 Dimension scores are **roughly balanced** "
            f"({weakest} {means[weakest]:.2f} → {strongest} {means[strongest]:.2f}) — "
            "no single dimension is dragging the portfolio down.")


def caption_weakest_dim_distribution(comp_view):
    """Bar chart of how often each dimension is the weakest across goals."""
    if comp_view is None or len(comp_view) == 0 or "weakest_dim" not in comp_view.columns:
        return None
    counts = comp_view["weakest_dim"].value_counts()
    if len(counts) == 0:
        return None
    top_dim = counts.index[0]
    top_n   = int(counts.iloc[0])
    pct     = top_n / len(comp_view) * 100
    if pct > 60:
        return (f"📌 **{top_dim.capitalize()}** is the weakest dimension for {top_n}/{len(comp_view)} "
                f"goals ({pct:.0f}%) — a portfolio-wide weakness, not a few outliers.")
    return (f"📌 **{top_dim.capitalize()}** is most often the weakest dimension "
            f"({top_n}/{len(comp_view)} goals, {pct:.0f}%), but the issue is spread across goals.")


def caption_goals_ranked(comp_view, risk_thresh=0.35):
    """Horizontal bar of goals ranked by composite."""
    if comp_view is None or len(comp_view) == 0 or "composite" not in comp_view.columns:
        return None
    n_total = len(comp_view)
    n_risk  = int(comp_view["at_risk"].sum()) if "at_risk" in comp_view.columns else \
              int((comp_view["composite"] < risk_thresh).sum())
    median  = comp_view["composite"].median()
    top_score = comp_view["composite"].max()
    if n_risk == 0:
        return f"🎯 All {n_total} goals are above the 0.35 risk threshold (median {median:.2f}, top {top_score:.2f})."
    pct = n_risk / n_total * 100
    return (f"🎯 **{n_risk} of {n_total} goals ({pct:.0f}%)** are below the 0.35 risk threshold; "
            f"median composite is {median:.2f} (top performer {top_score:.2f}).")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Portfolio by Bucket
# ─────────────────────────────────────────────────────────────────────────────
def caption_avg_composite_per_bucket(port_df):
    """Bar chart of mean composite per L2 bucket.

    `port_df` = portfolio_summary_poc.csv: l2_name, avg_composite, at_risk_count.
    """
    if port_df is None or len(port_df) == 0 or "avg_composite" not in port_df.columns:
        return None
    best  = port_df.loc[port_df["avg_composite"].idxmax()]
    worst = port_df.loc[port_df["avg_composite"].idxmin()]
    spread = best["avg_composite"] - worst["avg_composite"]
    if spread > 0.30:
        return (f"🏢 **{best['l2_name']}** leads at {best['avg_composite']:.2f}; "
                f"**{worst['l2_name']}** trails at {worst['avg_composite']:.2f} — "
                f"a {spread:.2f} gap signals uneven portfolio health.")
    return (f"🏢 Buckets cluster tightly between {worst['avg_composite']:.2f} ({worst['l2_name']}) "
            f"and {best['avg_composite']:.2f} ({best['l2_name']}) — uniform health across departments.")


def caption_top_buckets_radar(port_df, top_n=5):
    """Radar chart of top-N buckets across 4 dimensions."""
    if port_df is None or len(port_df) == 0:
        return None
    dim_cols = ["avg_coherence", "avg_attainability", "avg_relevance", "avg_integrity"]
    if not all(c in port_df.columns for c in dim_cols):
        return None
    top = port_df.nlargest(top_n, "avg_composite")
    if len(top) == 0:
        return None
    avgs = {c.replace("avg_", "").capitalize(): top[c].mean() for c in dim_cols}
    weakest = min(avgs, key=avgs.get)
    return (f"🕸️ Across the top {len(top)} buckets, **{weakest}** is consistently weakest "
            f"({avgs[weakest]:.2f}) — a structural pattern, not bucket-specific.")


def caption_bucket_over_time(pts_bucket, bucket_name):
    """Line chart of one bucket's composite trajectory.

    `pts_bucket` = portfolio_timeseries_poc filtered to one l2_name.
    """
    if pts_bucket is None or len(pts_bucket) == 0 or "avg_composite" not in pts_bucket.columns:
        return None
    start = pts_bucket["avg_composite"].iloc[0]
    end   = pts_bucket["avg_composite"].iloc[-1]
    delta = end - start
    trend = _trend_word(delta)
    return (f"📈 **{bucket_name}** trended **{trend}** from {start:.2f} (P{int(pts_bucket['period_id'].iloc[0])}) "
            f"to {end:.2f} (P{int(pts_bucket['period_id'].iloc[-1])}) — net change {delta:+.2f}.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Shock Analysis
# ─────────────────────────────────────────────────────────────────────────────
def caption_shock_phases(pre, budget, market, post):
    """Bar chart of composite by shock phase (pre / budget / market / post)."""
    if pre is None or pre == 0:
        return None
    worst_phase, worst_val = "budget", budget
    if market is not None and market < budget:
        worst_phase, worst_val = "market", market
    drop_pct = (pre - worst_val) / pre * 100
    recovery = (post - worst_val) if post is not None else 0
    if post is not None and post >= pre - 0.05:
        return (f"⚡ Worst hit was **{worst_phase} shock** (-{drop_pct:.0f}% from pre-shock); "
                f"portfolio **fully recovered** to {post:.2f}.")
    return (f"⚡ Worst hit was **{worst_phase} shock** (-{drop_pct:.0f}% from pre-shock); "
            f"recovery only +{recovery:.2f}, **not back to baseline** ({post:.2f} vs {pre:.2f}).")


def caption_selected_goals_trajectory(comp_traj_df, ts_df, n_selected):
    """Per-goal lines through shock periods.

    `comp_traj_df` = composite_scores rows for selected goals.
    """
    if comp_traj_df is None or len(comp_traj_df) == 0:
        return None
    by_goal = comp_traj_df.groupby("goal_id")["composite"]
    if len(by_goal) == 0:
        return None
    deltas = by_goal.last() - by_goal.first()
    n_improving = int((deltas > 0.02).sum())
    n_degrading = int((deltas < -0.02).sum())
    n_stable    = len(deltas) - n_improving - n_degrading
    return (f"📊 Of {n_selected} selected goal(s): **{n_improving} improving**, "
            f"{n_stable} stable, **{n_degrading} degrading** across the period range.")


def caption_market_shock_vulnerable(vuln_df):
    """Table of goals flagged as market-shock vulnerable."""
    if vuln_df is None or len(vuln_df) == 0:
        return "✅ No goals flagged as market-shock vulnerable at this period."
    n = len(vuln_df)
    if "composite" in vuln_df.columns:
        worst = vuln_df.loc[vuln_df["composite"].idxmin()]
        return (f"⚠️ **{n} goals** flagged vulnerable to market shock; "
                f"worst is goal **{int(worst['goal_id'])}** at composite {worst['composite']:.2f}.")
    return f"⚠️ **{n} goals** flagged vulnerable to market shock."


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — Forward Projection
# ─────────────────────────────────────────────────────────────────────────────
def caption_forward_projection(fwd_df):
    """Multi-line projection chart at p18 → p24 → p30.

    `fwd_df` = forward_projection_poc.csv: composite_adjusted, composite_p6,
               composite_p12, improving_p6, degrading_p6.
    """
    if fwd_df is None or len(fwd_df) == 0:
        return None
    n_total = len(fwd_df)
    n_imp = int(fwd_df["improving_p6"].sum()) if "improving_p6" in fwd_df.columns else 0
    n_deg = int(fwd_df["degrading_p6"].sum()) if "degrading_p6" in fwd_df.columns else 0
    n_stb = max(0, n_total - n_imp - n_deg)
    if "composite_p6" in fwd_df.columns and "composite_adjusted" in fwd_df.columns:
        delta_avg = (fwd_df["composite_p6"] - fwd_df["composite_adjusted"]).mean()
        direction = "improve" if delta_avg > 0 else "degrade"
        return (f"🔮 **{n_imp} improving · {n_stb} stable · {n_deg} degrading** at +6 periods; "
                f"portfolio expected to {direction} by **{delta_avg:+.2f}** on average.")
    return f"🔮 **{n_imp} improving · {n_stb} stable · {n_deg} degrading** goals projected at +6 periods."


def caption_projection_validation(fwd_df, error_col="composite_error_p24"):
    """Validation table comparing P18 projection vs P24 actual.

    Falls back to attain_error_p24 if composite_error not present (per ref doc).
    """
    if fwd_df is None or len(fwd_df) == 0:
        return None
    col = error_col if error_col in fwd_df.columns else \
          ("attain_error_p24" if "attain_error_p24" in fwd_df.columns else None)
    if col is None:
        return None
    mae = fwd_df[col].mean()
    metric_name = "Composite" if col == "composite_error_p24" else "Attainability"
    if mae < 0.05:
        return f"✅ {metric_name} projection MAE = **{mae:.3f}** — high-credibility forward look."
    if mae < 0.10:
        return f"📊 {metric_name} projection MAE = **{mae:.3f}** — solid validation against P24 actuals."
    return f"⚠️ {metric_name} projection MAE = **{mae:.3f}** — moderate uncertainty in forward projections."


# ─────────────────────────────────────────────────────────────────────────────
#  Output page (per-goal report) captions
# ─────────────────────────────────────────────────────────────────────────────
def caption_per_goal_radar(score_dict):
    """Radar of one goal across 4 dimensions vs weight profile."""
    if not score_dict:
        return None
    dims = ["coherence", "attainability", "relevance", "integrity"]
    vals = {d: score_dict.get(d, 0) for d in dims}
    weakest = min(vals, key=vals.get)
    strongest = max(vals, key=vals.get)
    return (f"🎯 Strongest dimension is **{strongest}** ({vals[strongest]:.0%}); "
            f"weakest is **{weakest}** ({vals[weakest]:.0%}) — focus area for improvement.")


def caption_per_goal_trajectory(score_dict, periods):
    """Projected trajectory of one goal."""
    if not score_dict:
        return None
    fwd = score_dict.get("forward", {})
    overall = score_dict.get("overall", 0)
    if not fwd:
        return None
    p18 = fwd.get("p18_projected", overall)
    delta = p18 - overall
    traj_class = fwd.get("trajectory_class", _trend_word(delta))
    target_attainment = fwd.get("expected_target_attainment")
    if traj_class == "improving":
        ext = f" ({target_attainment:.0%} target attainment expected)" if target_attainment else ""
        return f"📈 Trajectory **improving** — projected to gain **{delta:+.1%}** by period +18{ext}."
    if traj_class == "degrading":
        return f"📉 Trajectory **degrading** — projected to lose **{delta:+.1%}** by period +18. Intervention recommended."
    return f"➡️ Trajectory **stable** — projected drift of **{delta:+.1%}** by period +18, within noise band."


def caption_per_goal_shock(score_dict):
    """Shock-phase bars for one goal."""
    if not score_dict:
        return None
    shock = score_dict.get("shock", {})
    if not shock:
        return None
    pre  = shock.get("pre_shock", 0)
    bud  = shock.get("budget_shock", 0)
    mkt  = shock.get("market_shock", 0)
    post = shock.get("post_shock", 0)
    rec  = shock.get("recovery_periods", 0)
    vuln = shock.get("vulnerability_class", "moderate")
    max_drop = min(bud - pre, mkt - pre)
    rel_note = shock.get("vs_portfolio_avg_drop", "")
    if vuln == "resilient":
        return f"🛡️ **Shock-resilient** — worst drop {max_drop:.1%}, recovers fully in {rec} periods. {rel_note}"
    if vuln == "vulnerable":
        return f"⚠️ **Shock-vulnerable** — drops {max_drop:.1%} under stress; recovery takes {rec} periods. Consider buffer allocation."
    return f"🟡 **Moderate** shock sensitivity — {max_drop:.1%} worst-case drop, {rec}-period recovery. {rel_note}"


def caption_portfolio_context(this_score, portfolio_avg, pct_rank):
    """Mini portfolio time-series with this goal as reference line."""
    if pct_rank >= 75:
        return (f"⭐ This goal scores in the **top {100-pct_rank:.0f}%** of the portfolio "
                f"({this_score:.3f} vs {portfolio_avg:.3f} mean).")
    if pct_rank >= 25:
        return (f"📊 This goal scores in the **middle band** of the portfolio "
                f"({this_score:.3f} vs {portfolio_avg:.3f} mean).")
    return (f"⚠️ This goal scores in the **bottom {pct_rank:.0f}%** of the portfolio "
            f"({this_score:.3f} vs {portfolio_avg:.3f} mean) — needs attention.")