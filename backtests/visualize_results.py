"""
Visualize ORB + Monday Range backtest results as an interactive HTML dashboard.
Reads the unified_signals.csv and generates charts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT = RESULTS_DIR / "dashboard.html"


def wr(df):
    """Win rate as float."""
    return float(df["win"].mean() * 100) if len(df) > 0 else 0.0


def wr_str(df):
    return f"{wr(df):.1f}%"


def load_data():
    unified = pd.read_csv(RESULTS_DIR / "unified_signals.csv")
    monday = pd.read_csv(RESULTS_DIR / "strategy_b_monday_range.csv")
    orb_all = pd.read_csv(RESULTS_DIR / "strategy_a_orb.csv")
    rr_path = RESULTS_DIR / "rr_simulation.csv"
    rr = pd.read_csv(rr_path) if rr_path.exists() else None
    lir_path = RESULTS_DIR / "let_it_run.csv"
    lir = pd.read_csv(lir_path) if lir_path.exists() else None

    # Load all turbo CSVs (buffer-specific only, not the comparison file)
    turbo_files = sorted(RESULTS_DIR.glob("turbo_*pct.csv"))
    turbos = {}
    for f in turbo_files:
        buf = f.stem.replace("turbo_", "").replace("pct", "")
        turbos[buf] = pd.read_csv(f)

    ts_path = RESULTS_DIR / "turbo_strategy_comparison.csv"
    turbos_strat = pd.read_csv(ts_path) if ts_path.exists() else None

    return unified, monday, orb_all, rr, lir, turbos, turbos_strat


def chart_cross_strategy(unified, monday, orb_all):
    """Bar chart: cross-strategy win rates."""
    rows = []

    def add(label, df, color_group="single"):
        if len(df) == 0:
            return
        total = len(df)
        w = wr(df)
        longs = df[df["direction"] == "long"]
        shorts = df[df["direction"] == "short"]
        rows.append({
            "label": label, "WR": w, "trades": total,
            "L_WR": wr(longs), "S_WR": wr(shorts),
            "group": color_group,
        })

    add("Baseline ORB", orb_all, "baseline")
    add("Monday Range (5Y)", monday, "baseline")
    add("+ Monday Confluence", unified[unified["confluence"]], "single")
    add("+ VWAP Confirmed", unified[unified["vwap_confirmed"]], "single")
    add("+ Trend-Aligned", unified[unified["trend_aligned"]], "single")
    add("+ S/R Break", unified[unified["sr_type"] == "sr_break"], "single")
    add("+ Inside Monday", unified[unified["gap_type"] == "inside_monday"], "single")
    add("Confluence + VWAP", unified[unified["confluence"] & unified["vwap_confirmed"]], "combo")
    add("Confluence + Trend", unified[unified["confluence"] & unified["trend_aligned"]], "combo")
    add("Confl + VWAP + Trend", unified[unified["confluence"] & unified["vwap_confirmed"] & unified["trend_aligned"]], "combo")

    df = pd.DataFrame(rows)

    colors = {"baseline": "#6b7280", "single": "#3b82f6", "combo": "#10b981"}
    bar_colors = [colors[g] for g in df["group"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(df["label"]), y=list(df["WR"]),
        marker_color=bar_colors,
        text=[f"{w:.1f}%<br>n={t}" for w, t in zip(df["WR"], df["trades"])],
        textposition="outside",
        name="Win Rate",
    ))
    fig.add_hline(y=float(df[df["label"] == "Baseline ORB"]["WR"].iloc[0]),
                  line_dash="dash", line_color="#9ca3af",
                  annotation_text="Baseline", annotation_position="top left")
    fig.update_layout(
        title="Cross-Strategy Win Rates",
        yaxis_title="Win Rate (%)", yaxis_range=[55, 85],
        xaxis_tickangle=-30, height=500,
        showlegend=False,
    )
    return fig


def chart_signal_tf(unified):
    """Grouped bar: win rate by signal timeframe, long vs short."""
    tf_order = ["5min", "10min", "15min", "30min", "60min"]
    tf_labels = ["5m", "10m", "15m", "30m", "1h"]

    overall, longs_wr, shorts_wr, counts = [], [], [], []
    for tf in tf_order:
        sub = unified[unified["signal_tf"] == tf]
        overall.append(wr(sub))
        longs_wr.append(wr(sub[sub["direction"] == "long"]))
        shorts_wr.append(wr(sub[sub["direction"] == "short"]))
        counts.append(len(sub))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=tf_labels, y=overall, name="Overall",
                         marker_color="#6366f1", text=[f"{v:.1f}%" for v in overall],
                         textposition="outside"))
    fig.add_trace(go.Bar(x=tf_labels, y=longs_wr, name="Long",
                         marker_color="#22c55e"))
    fig.add_trace(go.Bar(x=tf_labels, y=shorts_wr, name="Short",
                         marker_color="#ef4444"))
    fig.update_layout(
        title="Win Rate by Signal Timeframe",
        yaxis_title="Win Rate (%)", yaxis_range=[55, 92],
        barmode="group", height=450,
        xaxis_title="Signal Candle Size",
    )
    return fig


def chart_factor_heatmap(unified):
    """Heatmap: win rate for every pair of boolean factors."""
    factors = {
        "Confluence": unified["confluence"].values,
        "VWAP": unified["vwap_confirmed"].values,
        "Trend-Aligned": unified["trend_aligned"].values,
        "S/R Break": (unified["sr_type"] == "sr_break").values,
        "Inside Monday": (unified["gap_type"] == "inside_monday").values,
    }
    names = list(factors.keys())
    n = len(names)

    # Pairwise win rates — use plain Python lists to avoid numpy serialization issues
    z_vals = []
    text_vals = []
    for i in range(n):
        z_row = []
        text_row = []
        for j in range(n):
            if i == j:
                sub = unified[factors[names[i]]]
            else:
                sub = unified[factors[names[i]] & factors[names[j]]]
            w = wr(sub)
            c = len(sub)
            z_row.append(w)
            text_row.append(f"{w:.1f}%\nn={c}")
        z_vals.append(z_row)
        text_vals.append(text_row)

    fig = go.Figure(go.Heatmap(
        z=z_vals, x=names, y=names,
        text=text_vals, texttemplate="%{text}",
        colorscale="RdYlGn", zmin=65, zmax=82,
        colorbar_title="WR %",
    ))
    fig.update_layout(
        title="Factor Pair Win Rates (row AND column both true)",
        height=500,
    )
    return fig


def chart_stacking(unified):
    """Show how stacking factors progressively improves win rate."""
    steps = [
        ("All ORB signals", unified),
        ("+ VWAP confirmed", unified[unified["vwap_confirmed"]]),
        ("+ Confluence", unified[unified["vwap_confirmed"] & unified["confluence"]]),
        ("+ Trend-aligned", unified[unified["vwap_confirmed"] & unified["confluence"] & unified["trend_aligned"]]),
    ]

    labels = [s[0] for s in steps]
    wrs = [wr(s[1]) for s in steps]
    trades = [len(s[1]) for s in steps]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=wrs, name="Win Rate",
        marker_color=["#6b7280", "#3b82f6", "#8b5cf6", "#10b981"],
        text=[f"{w:.1f}%" for w in wrs], textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels, y=trades, name="Trade Count",
        mode="lines+markers+text", line=dict(color="#f59e0b", width=2),
        text=[str(t) for t in trades], textposition="top center",
        marker=dict(size=10),
    ), secondary_y=True)

    fig.update_layout(
        title="Factor Stacking: Win Rate vs Trade Count",
        height=450,
    )
    fig.update_yaxes(title_text="Win Rate (%)", range=[65, 85], secondary_y=False)
    fig.update_yaxes(title_text="Number of Trades", secondary_y=True)
    return fig


def chart_per_ticker(unified):
    """Horizontal bar: win rate per ticker, colored by category."""
    cat_colors = {"Sector ETF": "#3b82f6", "Stock": "#8b5cf6", "Commodity": "#f59e0b"}

    rows = []
    for ticker in unified["ticker"].unique():
        sub = unified[unified["ticker"] == ticker]
        cat = str(sub["category"].iloc[0])
        rows.append({"ticker": str(ticker), "WR": wr(sub), "trades": int(len(sub)), "category": cat})
    df = pd.DataFrame(rows).sort_values("WR", ascending=True)

    fig = go.Figure()
    for cat, color in cat_colors.items():
        cat_df = df[df["category"] == cat]
        if len(cat_df) == 0:
            continue
        fig.add_trace(go.Bar(
            y=list(cat_df["ticker"]), x=list(cat_df["WR"]), orientation="h",
            name=cat, marker_color=color,
            text=[f"{w:.1f}% (n={t})" for w, t in zip(cat_df["WR"], cat_df["trades"])],
            textposition="outside",
        ))
    fig.add_vline(x=float(unified["win"].mean() * 100), line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        title="Win Rate by Ticker",
        xaxis_title="Win Rate (%)", xaxis_range=[55, 90],
        height=600, barmode="group",
    )
    return fig


def chart_gap_type(unified):
    """Bar chart: ORB open position vs Monday range."""
    gap_order = ["below_monday", "inside_monday", "above_monday"]
    gap_labels = ["Below Monday Low", "Inside Monday Range", "Above Monday High"]
    colors = ["#ef4444", "#6366f1", "#22c55e"]

    fig = go.Figure()
    for gap, label, color in zip(gap_order, gap_labels, colors):
        sub = unified[unified["gap_type"] == gap]
        longs = sub[sub["direction"] == "long"]
        shorts = sub[sub["direction"] == "short"]
        fig.add_trace(go.Bar(
            x=[f"{label}<br>Long", f"{label}<br>Short"],
            y=[wr(longs), wr(shorts)],
            name=label, marker_color=color,
            text=[f"{wr(longs):.1f}%<br>n={len(longs)}", f"{wr(shorts):.1f}%<br>n={len(shorts)}"],
            textposition="outside",
        ))
    fig.update_layout(
        title="Win Rate by ORB Open vs Monday Range (Long vs Short)",
        yaxis_title="Win Rate (%)", yaxis_range=[55, 85],
        height=450, showlegend=True,
    )
    return fig


def chart_tf_by_combo(unified):
    """Line chart: win rate by signal TF for key factor combos."""
    tf_order = ["5min", "10min", "15min", "30min", "60min"]
    tf_labels = ["5m", "10m", "15m", "30m", "1h"]

    combos = [
        ("Baseline", unified, "#6b7280"),
        ("+ Confluence", unified[unified["confluence"]], "#3b82f6"),
        ("+ Confl + VWAP", unified[unified["confluence"] & unified["vwap_confirmed"]], "#8b5cf6"),
        ("+ Confl + VWAP + Trend", unified[unified["confluence"] & unified["vwap_confirmed"] & unified["trend_aligned"]], "#10b981"),
    ]

    fig = go.Figure()
    for name, df, color in combos:
        wrs = [wr(df[df["signal_tf"] == tf]) for tf in tf_order]
        counts = [len(df[df["signal_tf"] == tf]) for tf in tf_order]
        fig.add_trace(go.Scatter(
            x=tf_labels, y=wrs, name=name,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=8),
            hovertext=[f"n={c}" for c in counts],
        ))

    fig.update_layout(
        title="Win Rate by Signal TF Across Factor Combos",
        xaxis_title="Signal Candle Size",
        yaxis_title="Win Rate (%)", yaxis_range=[60, 95],
        height=450,
    )
    return fig


def chart_long_vs_short(unified):
    """Grouped bar: long vs short win rate for each factor."""
    factor_defs = [
        ("Baseline", unified),
        ("Confluence", unified[unified["confluence"]]),
        ("VWAP", unified[unified["vwap_confirmed"]]),
        ("Trend-Aligned", unified[unified["trend_aligned"]]),
        ("S/R Break", unified[unified["sr_type"] == "sr_break"]),
        ("Confl+VWAP", unified[unified["confluence"] & unified["vwap_confirmed"]]),
        ("Confl+VWAP+Trend", unified[unified["confluence"] & unified["vwap_confirmed"] & unified["trend_aligned"]]),
    ]

    labels = [f[0] for f in factor_defs]
    l_wrs = [wr(f[1][f[1]["direction"] == "long"]) for f in factor_defs]
    s_wrs = [wr(f[1][f[1]["direction"] == "short"]) for f in factor_defs]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=l_wrs, name="Long", marker_color="#22c55e",
                         text=[f"{v:.1f}%" for v in l_wrs], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=s_wrs, name="Short", marker_color="#ef4444",
                         text=[f"{v:.1f}%" for v in s_wrs], textposition="outside"))
    fig.update_layout(
        title="Long vs Short Win Rate by Factor",
        yaxis_title="Win Rate (%)", yaxis_range=[55, 85],
        barmode="group", height=450, xaxis_tickangle=-20,
    )
    return fig


def chart_rr_simulation(rr):
    """Bar chart: 1.5:1 R:R simulation results by signal TF and factors."""
    if rr is None or len(rr) == 0:
        return None

    # Chart 1: TP/SL/Hold breakdown by signal TF
    tf_order = ["5min", "10min", "15min", "30min", "60min"]
    tf_labels = ["5m", "10m", "15m", "30m", "1h"]

    tp_rates, sl_rates, hold_rates, expects, counts = [], [], [], [], []
    for tf in tf_order:
        sub = rr[rr["signal_tf"] == tf]
        n = len(sub)
        if n == 0:
            tp_rates.append(0); sl_rates.append(0); hold_rates.append(0)
            expects.append(0); counts.append(0)
            continue
        tp_rates.append((sub["outcome"] == "tp").sum() / n * 100)
        sl_rates.append((sub["outcome"] == "sl").sum() / n * 100)
        hold_rates.append((sub["outcome"] == "hold").sum() / n * 100)
        expects.append(sub["pnl_pct"].mean())
        counts.append(n)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["TP/SL/Hold Rate by Signal TF", "Expectancy by Signal TF (% per trade)"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(x=tf_labels, y=tp_rates, name="TP Hit",
                         marker_color="#22c55e", text=[f"{v:.1f}%" for v in tp_rates],
                         textposition="inside"), row=1, col=1)
    fig.add_trace(go.Bar(x=tf_labels, y=sl_rates, name="SL Hit",
                         marker_color="#ef4444", text=[f"{v:.1f}%" for v in sl_rates],
                         textposition="inside"), row=1, col=1)
    fig.add_trace(go.Bar(x=tf_labels, y=hold_rates, name="Held to Close",
                         marker_color="#6366f1", text=[f"{v:.1f}%" for v in hold_rates],
                         textposition="inside"), row=1, col=1)

    bar_colors = ["#22c55e" if e > 0 else "#ef4444" for e in expects]
    fig.add_trace(go.Bar(x=tf_labels, y=expects, name="Expectancy",
                         marker_color=bar_colors,
                         text=[f"{v:+.3f}%" for v in expects],
                         textposition="outside", showlegend=False), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", row=1, col=2)

    fig.update_layout(
        title="1.5:1 R:R Exit  |  TP = 1.5x risk (price touch)  |  SL = no-wick reversal past ORB",
        barmode="stack", height=450,
    )
    fig.update_yaxes(title_text="% of Trades", row=1, col=1)
    fig.update_yaxes(title_text="Expectancy (%)", row=1, col=2)
    return fig


def chart_rr_factors(rr):
    """Bar chart: R:R expectancy by factor combo (5min signals)."""
    if rr is None or len(rr) == 0:
        return None

    rr5 = rr[rr["signal_tf"] == "5min"]
    if len(rr5) < 20:
        return None

    combos = [
        ("Baseline\n(5min)", rr5),
        ("Confluence", rr5[rr5["confluence"]]),
        ("Trend", rr5[rr5["trend_aligned"]]),
        ("Confl+Trend", rr5[rr5["confluence"] & rr5["trend_aligned"]]),
        ("Confl+VWAP\n+Trend", rr5[rr5["confluence"] & rr5["vwap_confirmed"] & rr5["trend_aligned"]]),
        ("Inside\nMonday", rr5[rr5["gap_type"] == "inside_monday"]),
        ("Long+Confl\n+Trend", rr5[(rr5["direction"] == "long") & rr5["confluence"] & rr5["trend_aligned"]]),
    ]

    labels = [c[0] for c in combos]
    expects = [c[1]["pnl_pct"].mean() if len(c[1]) > 0 else 0 for c in combos]
    tp_rates = [(c[1]["outcome"] == "tp").mean() * 100 if len(c[1]) > 0 else 0 for c in combos]
    trade_counts = [len(c[1]) for c in combos]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_colors = ["#22c55e" if e > 0 else "#ef4444" for e in expects]
    fig.add_trace(go.Bar(
        x=labels, y=expects, name="Expectancy",
        marker_color=bar_colors,
        text=[f"{v:+.3f}%\nn={n}" for v, n in zip(expects, trade_counts)],
        textposition="outside",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=labels, y=tp_rates, name="TP Hit Rate",
        mode="lines+markers+text", line=dict(color="#60a5fa", width=2),
        text=[f"{v:.0f}%" for v in tp_rates], textposition="top center",
        marker=dict(size=8),
    ), secondary_y=True)

    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        title="1.5:1 R:R Expectancy by Factor (5min signal candle)",
        height=450, xaxis_tickangle=-20,
    )
    fig.update_yaxes(title_text="Expectancy (% per trade)", secondary_y=False)
    fig.update_yaxes(title_text="TP Hit Rate (%)", secondary_y=True)
    return fig


def chart_let_it_run_r_dist(lir):
    """Histogram of R-multiples + stacked bar by signal TF showing R-multiple buckets."""
    if lir is None or len(lir) == 0:
        return None

    # Chart 1: R-multiple histogram (all trades)
    # Chart 2: R-bucket breakdown by signal TF
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["R-Multiple Distribution (all trades)", "R-Multiple Buckets by Signal TF"],
        horizontal_spacing=0.12,
    )

    # Clip for histogram display
    r_vals = list(lir["r_multiple"].clip(-5, 15))
    fig.add_trace(go.Histogram(
        x=r_vals, nbinsx=60,
        marker_color="#6366f1",
        name="Trades",
    ), row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=2, row=1, col=1)
    avg_r = float(lir["r_multiple"].mean())
    fig.add_vline(x=avg_r, line_dash="dot", line_color="#22c55e", line_width=2, row=1, col=1,
                  annotation_text=f"Avg: {avg_r:+.2f}R", annotation_font_color="#22c55e")

    # R-bucket breakdown by TF
    tf_order = ["5min", "10min", "15min", "30min", "60min"]
    tf_labels = ["5m", "10m", "15m", "30m", "1h"]
    buckets = [
        ("<0R", lambda s: s < 0, "#ef4444"),
        ("0-1R", lambda s: (s >= 0) & (s < 1), "#f97316"),
        ("1-2R", lambda s: (s >= 1) & (s < 2), "#eab308"),
        ("2-3R", lambda s: (s >= 2) & (s < 3), "#84cc16"),
        ("3-5R", lambda s: (s >= 3) & (s < 5), "#22c55e"),
        ("5R+", lambda s: s >= 5, "#06b6d4"),
    ]

    for bname, bfunc, bcolor in buckets:
        vals = []
        for tf in tf_order:
            sub = lir[lir["signal_tf"] == tf]
            if len(sub) == 0:
                vals.append(0)
            else:
                vals.append(float(bfunc(sub["r_multiple"]).sum() / len(sub) * 100))
        fig.add_trace(go.Bar(
            x=tf_labels, y=vals, name=bname,
            marker_color=bcolor,
            text=[f"{v:.0f}%" if v > 3 else "" for v in vals],
            textposition="inside",
        ), row=1, col=2)

    fig.update_layout(
        title="Let-It-Run: R-Multiple Distribution  |  No TP cap, SL = no-wick reversal",
        barmode="stack", height=480,
    )
    fig.update_xaxes(title_text="R-Multiple", row=1, col=1)
    fig.update_yaxes(title_text="# Trades", row=1, col=1)
    fig.update_yaxes(title_text="% of Trades", row=1, col=2)
    return fig


def chart_let_it_run_factors(lir):
    """Bar chart: avg R-multiple and 3R+ rate by factor combo (5min signals)."""
    if lir is None or len(lir) == 0:
        return None

    lir5 = lir[lir["signal_tf"] == "5min"]
    if len(lir5) < 20:
        return None

    # Add the reverse-engineered filter
    big_orb = lir5[(lir5["orb_high"] - lir5["orb_low"]) / lir5["entry_price"] * 100 > 0.8]
    tight_risk = big_orb[big_orb["risk_pct"] < 0.25]
    filtered = tight_risk[tight_risk["trend_aligned"]]

    combos = [
        ("Baseline\n(5min)", lir5),
        ("Confluence", lir5[lir5["confluence"]]),
        ("Trend", lir5[lir5["trend_aligned"]]),
        ("Confl+Trend", lir5[lir5["confluence"] & lir5["trend_aligned"]]),
        ("Confl+VWAP\n+Trend", lir5[lir5["confluence"] & lir5["vwap_confirmed"] & lir5["trend_aligned"]]),
        ("Inside\nMonday", lir5[lir5["gap_type"] == "inside_monday"]),
        ("ORB>0.8%\nrisk<0.25%\n+trend", filtered),
    ]

    labels = [c[0] for c in combos]
    avg_rs = [float(c[1]["r_multiple"].mean()) if len(c[1]) > 0 else 0 for c in combos]
    r3_rates = [float((c[1]["r_multiple"] >= 3).mean() * 100) if len(c[1]) > 0 else 0 for c in combos]
    trade_counts = [len(c[1]) for c in combos]
    avg_mfes = [float(c[1]["mfe_r"].mean()) if len(c[1]) > 0 else 0 for c in combos]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_colors = ["#22c55e" if r > 0 else "#ef4444" for r in avg_rs]
    fig.add_trace(go.Bar(
        x=labels, y=avg_rs, name="Avg R-Multiple",
        marker_color=bar_colors,
        text=[f"{v:+.2f}R\nn={n}" for v, n in zip(avg_rs, trade_counts)],
        textposition="outside",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=labels, y=r3_rates, name="3R+ Win Rate",
        mode="lines+markers+text", line=dict(color="#06b6d4", width=2),
        text=[f"{v:.0f}%" for v in r3_rates], textposition="top center",
        marker=dict(size=8),
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=labels, y=avg_mfes, name="Avg MFE (R)",
        mode="lines+markers", line=dict(color="#a78bfa", width=2, dash="dot"),
        marker=dict(size=6),
    ), secondary_y=False)

    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        title="Let-It-Run: R-Multiples by Factor Combo (5min signals)  |  Avg R + 3R+ Rate + MFE",
        height=480, xaxis_tickangle=-20,
    )
    fig.update_yaxes(title_text="R-Multiple", secondary_y=False)
    fig.update_yaxes(title_text="3R+ Rate (%)", secondary_y=True)
    return fig


def chart_let_it_run_outcomes(lir):
    """Scatter: R-multiple achieved vs MFE (max favorable excursion)."""
    if lir is None or len(lir) == 0:
        return None

    lir5 = lir[lir["signal_tf"] == "5min"].copy()
    if len(lir5) < 20:
        return None

    # Color by outcome
    colors = []
    for _, row in lir5.iterrows():
        if row["outcome"] == "stopped":
            colors.append("#ef4444")
        elif row["r_multiple"] >= 3:
            colors.append("#22c55e")
        elif row["r_multiple"] >= 0:
            colors.append("#eab308")
        else:
            colors.append("#f97316")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(lir5["mfe_r"].clip(0, 20)),
        y=list(lir5["r_multiple"].clip(-5, 20)),
        mode="markers",
        marker=dict(
            color=colors,
            size=6, opacity=0.6,
            line=dict(width=0.5, color="#1e293b"),
        ),
        text=[f"{row['ticker']} {row['date']}<br>R={row['r_multiple']:.1f} MFE={row['mfe_r']:.1f}"
              for _, row in lir5.iterrows()],
        hoverinfo="text",
        showlegend=False,
    ))

    # Diagonal line (R = MFE → captured all favorable movement)
    fig.add_trace(go.Scatter(
        x=[0, 15], y=[0, 15], mode="lines",
        line=dict(color="#9ca3af", dash="dot", width=1),
        name="Captured all MFE",
    ))

    fig.update_layout(
        title="Let-It-Run: R Captured vs Max Favorable Excursion (5min signals)",
        xaxis_title="MFE (max price went in your favor, in R-multiples)",
        yaxis_title="Actual R-Multiple at Exit",
        height=500,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=1)
    return fig


def chart_turbo_buffer_comparison(turbos):
    """Bar chart comparing P&L, ROI, KO rate across buffer levels."""
    if not turbos:
        return None

    buffers = sorted(turbos.keys(), key=lambda x: float(x))
    labels = [f"{b}%" for b in buffers]
    leverages = [f"~{round(100/float(b))}x" for b in buffers]

    total_pnls = []
    rois = []
    ko_rates = []
    doubled_rates = []
    for b in buffers:
        df = turbos[b]
        n = len(df)
        total_pnls.append(float(df["turbo_pnl_sek"].sum()))
        rois.append(float(df["turbo_pnl_sek"].sum()) / (n * 500) * 100)
        ko_rates.append(float((df["outcome"] == "ko").sum() / n * 100))
        doubled_rates.append(float((df["turbo_pnl_sek"] > 500).sum() / n * 100))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Total P&L by Buffer (500 trades × 500 SEK)", "KO Rate vs Doubled+ Rate"],
        horizontal_spacing=0.12,
    )

    # P&L bars
    bar_colors = ["#22c55e" if p > 0 else "#ef4444" for p in total_pnls]
    fig.add_trace(go.Bar(
        x=labels, y=total_pnls, name="Total P&L",
        marker_color=bar_colors,
        text=[f"{p:+,.0f} SEK\n{leverages[i]}" for i, p in enumerate(total_pnls)],
        textposition="outside",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", row=1, col=1)

    # KO vs Doubled rates
    fig.add_trace(go.Bar(
        x=labels, y=ko_rates, name="KO Rate",
        marker_color="#ef4444",
        text=[f"{v:.0f}%" for v in ko_rates], textposition="outside",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=labels, y=doubled_rates, name="Doubled+ Rate",
        marker_color="#22c55e",
        text=[f"{v:.0f}%" for v in doubled_rates], textposition="outside",
    ), row=1, col=2)

    n_trades = len(turbos[buffers[0]])
    date_min = turbos[buffers[0]]["date"].min()
    date_max = turbos[buffers[0]]["date"].max()
    fig.update_layout(
        title=f"Turbo Simulation: Buffer Comparison  |  {n_trades} trades, {date_min} to {date_max}",
        barmode="group", height=480,
    )
    fig.update_xaxes(title_text="KO Buffer (%)", row=1, col=1)
    fig.update_xaxes(title_text="KO Buffer (%)", row=1, col=2)
    fig.update_yaxes(title_text="SEK", row=1, col=1)
    fig.update_yaxes(title_text="% of Trades", row=1, col=2)
    return fig


def chart_turbo_equity_curves(turbos):
    """Line chart: cumulative P&L (equity curve) for each buffer level."""
    if not turbos:
        return None

    fig = go.Figure()
    colors = {"0.3": "#f43f5e", "0.5": "#f97316", "0.8": "#eab308",
              "1.0": "#22c55e", "1.5": "#06b6d4", "2.0": "#6366f1", "3.0": "#a78bfa"}

    buffers = sorted(turbos.keys(), key=lambda x: float(x))
    for b in buffers:
        df = turbos[b].sort_values(["date", "time"])
        cumulative = list(df["turbo_pnl_sek"].cumsum())
        x_vals = list(range(1, len(cumulative) + 1))
        leverage = round(100 / float(b))
        fig.add_trace(go.Scatter(
            x=x_vals, y=cumulative,
            mode="lines", name=f"{b}% (~{leverage}x)",
            line=dict(color=colors.get(b, "#9ca3af"), width=2),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", line_width=1)
    fig.update_layout(
        title="Turbo Equity Curves  |  Cumulative P&L per trade (500 SEK each)",
        xaxis_title="Trade #",
        yaxis_title="Cumulative P&L (SEK)",
        height=480,
    )
    return fig


def chart_turbo_pnl_distribution(turbos):
    """Histogram: P&L distribution for selected buffer levels."""
    if not turbos:
        return None

    # Pick 3 representative buffers
    picks = [b for b in ["0.5", "1.0", "2.0"] if b in turbos]
    if not picks:
        picks = sorted(turbos.keys(), key=lambda x: float(x))[:3]

    colors = {"0.3": "#f43f5e", "0.5": "#f97316", "0.8": "#eab308",
              "1.0": "#22c55e", "1.5": "#06b6d4", "2.0": "#6366f1", "3.0": "#a78bfa"}

    fig = go.Figure()
    for b in picks:
        df = turbos[b]
        pnls = list(df["turbo_pnl_sek"].clip(-600, 3000))
        leverage = round(100 / float(b))
        fig.add_trace(go.Histogram(
            x=pnls, nbinsx=50, name=f"{b}% (~{leverage}x)",
            marker_color=colors.get(b, "#9ca3af"),
            opacity=0.6,
        ))

    fig.update_layout(
        title="Turbo P&L Distribution  |  Per-trade P&L in SEK (500 SEK position)",
        xaxis_title="P&L (SEK)",
        yaxis_title="# Trades",
        barmode="overlay", height=450,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=2)
    fig.add_vline(x=-500, line_dash="dot", line_color="#9ca3af",
                  annotation_text="KO (-500)", annotation_font_color="#9ca3af")
    return fig


def chart_turbo_strategy_comparison(turbos_strat):
    """Grouped bar charts: turbo ROI by strategy filter × buffer level, one row per signal TF."""
    if turbos_strat is None or len(turbos_strat) == 0:
        return None

    signal_tfs = sorted(turbos_strat["signal_tf"].unique()) if "signal_tf" in turbos_strat.columns else ["5min"]
    n_rows = len(signal_tfs)

    subtitles = []
    for tf in signal_tfs:
        subtitles.extend([f"{tf} — Avg P&L per Trade (SEK)", f"{tf} — KO Rate (%)"])

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=subtitles,
        horizontal_spacing=0.12, vertical_spacing=0.15,
    )

    buf_colors = {0.3: "#f43f5e", 0.5: "#f97316", 1.0: "#22c55e", 2.0: "#6366f1"}
    buffers = sorted(turbos_strat["buffer"].unique())

    for row_idx, tf in enumerate(signal_tfs, 1):
        tf_data = turbos_strat[turbos_strat["signal_tf"] == tf] if "signal_tf" in turbos_strat.columns else turbos_strat
        show_legend = row_idx == 1

        for buf in buffers:
            sub = tf_data[tf_data["buffer"] == buf]
            if len(sub) == 0:
                continue
            leverage = round(100 / buf)
            color = buf_colors.get(buf, "#9ca3af")

            # Add n_trades to labels
            x_labels = [f"{s}\n(n={n})" for s, n in zip(sub["strategy"], sub["n_trades"])]

            fig.add_trace(go.Bar(
                x=x_labels, y=list(sub["avg_pnl"]),
                name=f"{buf}% (~{leverage}x)",
                marker_color=color, showlegend=show_legend,
                text=[f"{v:+,.0f}" for v in sub["avg_pnl"]],
                textposition="outside", textfont_size=9,
            ), row=row_idx, col=1)

            fig.add_trace(go.Bar(
                x=x_labels, y=list(sub["ko_rate"]),
                name=f"{buf}% (~{leverage}x)",
                marker_color=color, showlegend=False,
                text=[f"{v:.0f}%" for v in sub["ko_rate"]],
                textposition="outside", textfont_size=9,
            ), row=row_idx, col=2)

        fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", row=row_idx, col=1)

    fig.update_layout(
        title="Turbo P&L by Strategy × Buffer × Signal TF  |  5min vs 1min signals",
        barmode="group", height=420 * n_rows,
    )
    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Avg P&L (SEK)", row=i, col=1)
        fig.update_yaxes(title_text="KO Rate (%)", row=i, col=2)
    return fig


def chart_example_trades(unified):
    """Candlestick charts showing representative example trades."""
    import yfinance as yf
    import zoneinfo

    # Pick 6 representative trades
    examples = []

    # 1. High-confluence winning long (best case)
    hc_wins = unified[
        unified["confluence"] & unified["vwap_confirmed"] &
        unified["trend_aligned"] & (unified["direction"] == "long") &
        unified["win"]
    ]
    if len(hc_wins) > 0:
        examples.append(("High-Confluence Long WIN", hc_wins.iloc[len(hc_wins) // 2]))

    # 2. High-confluence losing long (shows it's not perfect)
    hc_loss = unified[
        unified["confluence"] & unified["vwap_confirmed"] &
        unified["trend_aligned"] & (unified["direction"] == "long") &
        ~unified["win"]
    ]
    if len(hc_loss) > 0:
        examples.append(("High-Confluence Long LOSS", hc_loss.iloc[0]))

    # 3. Baseline winning short (no extra factors)
    base_short_win = unified[
        ~unified["confluence"] & ~unified["vwap_confirmed"] &
        (unified["direction"] == "short") & unified["win"]
    ]
    if len(base_short_win) > 0:
        examples.append(("Baseline Short WIN", base_short_win.iloc[len(base_short_win) // 3]))

    # 4. Confluence + VWAP winning long (the sweet spot combo)
    cv_win = unified[
        unified["confluence"] & unified["vwap_confirmed"] &
        ~unified["trend_aligned"] &
        (unified["direction"] == "long") & unified["win"]
    ]
    if len(cv_win) > 0:
        stock_cv = cv_win[cv_win["category"] == "Stock"]
        pick = stock_cv.iloc[0] if len(stock_cv) > 0 else cv_win.iloc[0]
        examples.append(("Confluence+VWAP Long WIN (Stock)", pick))

    # 5. Inside-Monday winning long
    inside_win = unified[
        (unified["gap_type"] == "inside_monday") &
        (unified["direction"] == "long") & unified["win"] &
        unified["confluence"]
    ]
    if len(inside_win) > 0:
        comm = inside_win[inside_win["category"] == "Commodity"]
        pick = comm.iloc[0] if len(comm) > 0 else inside_win.iloc[len(inside_win) // 2]
        examples.append(("Inside Monday + Confluence Long WIN", pick))

    # 6. Baseline loss (no factors)
    base_loss = unified[
        ~unified["confluence"] & ~unified["vwap_confirmed"] &
        ~unified["trend_aligned"] &
        (unified["direction"] == "long") & ~unified["win"]
    ]
    if len(base_loss) > 0:
        examples.append(("Baseline Long LOSS (no factors)", base_loss.iloc[0]))

    if not examples:
        return None

    # Fetch 5-min data for the needed tickers
    needed_tickers = {str(row["ticker"]) for _, row in examples}

    print(f"  Fetching intraday data for {len(examples)} example trades ({len(needed_tickers)} tickers)...")
    ticker_data = {}
    for ticker in needed_tickers:
        try:
            df = yf.download(ticker, period="60d", interval="5m", progress=False)
            if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(1)
            df.index = pd.to_datetime(df.index)
            ticker_data[ticker] = df
        except Exception as e:
            print(f"    Warning: could not fetch {ticker}: {e}")

    ny_tz = zoneinfo.ZoneInfo("America/New_York")

    # Build individual figures (one per example) for reliable rendering
    trade_figs = []
    for title, signal_row in examples:
        ticker = str(signal_row["ticker"])
        trade_date = str(signal_row["date"])
        direction = str(signal_row["direction"])
        orb_high = float(signal_row["orb_high"])
        orb_low = float(signal_row["orb_low"])
        entry_price = float(signal_row["entry_price"])
        daily_close = float(signal_row["daily_close"])
        win = bool(signal_row["win"])
        signal_time = str(signal_row["time"])

        if ticker not in ticker_data:
            continue

        df = ticker_data[ticker]
        day_mask = df.index.strftime("%Y-%m-%d") == trade_date
        day_data = df[day_mask]

        if len(day_data) == 0:
            continue

        # Convert x values to strings (HH:MM) to avoid datetime axis issues
        x_times = [t.strftime("%H:%M") for t in day_data.index]

        fig = go.Figure()

        # Candlestick for the trading day
        fig.add_trace(go.Candlestick(
            x=x_times,
            open=list(day_data["Open"]),
            high=list(day_data["High"]),
            low=list(day_data["Low"]),
            close=list(day_data["Close"]),
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            showlegend=False,
            name=ticker,
        ))

        # ORB high line
        fig.add_hline(y=orb_high, line_dash="dash", line_color="#60a5fa", line_width=1.5,
                      annotation_text="ORB High", annotation_position="top left",
                      annotation_font_color="#60a5fa", annotation_font_size=10)

        # ORB low line
        fig.add_hline(y=orb_low, line_dash="dash", line_color="#f97316", line_width=1.5,
                      annotation_text="ORB Low", annotation_position="bottom left",
                      annotation_font_color="#f97316", annotation_font_size=10)

        # ORB shaded zone
        fig.add_hrect(y0=orb_low, y1=orb_high,
                      fillcolor="rgba(99, 102, 241, 0.12)", line_width=0)

        # Entry marker — match timezone
        signal_time_str = signal_time[:5]  # "HH:MM"
        entry_marker_color = "#22c55e" if direction == "long" else "#ef4444"
        entry_symbol = "triangle-up" if direction == "long" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[signal_time_str], y=[entry_price],
            mode="markers+text",
            marker=dict(color=entry_marker_color, size=14, symbol=entry_symbol,
                        line=dict(width=2, color="white")),
            text=[f"Entry {direction.upper()} ${entry_price:.2f}"],
            textposition="top center" if direction == "long" else "bottom center",
            textfont=dict(size=10, color=entry_marker_color),
            showlegend=False,
        ))

        # Daily close marker
        close_color = "#22c55e" if win else "#ef4444"
        close_label = "WIN" if win else "LOSS"
        fig.add_trace(go.Scatter(
            x=[x_times[-1]], y=[daily_close],
            mode="markers+text",
            marker=dict(color=close_color, size=12, symbol="diamond",
                        line=dict(width=2, color="white")),
            text=[f"Close ${daily_close:.2f} {close_label}"],
            textposition="middle left",
            textfont=dict(size=10, color=close_color),
            showlegend=False,
        ))

        # Build factors text
        factors = []
        if signal_row["confluence"]:
            factors.append("Confluence")
        if signal_row["vwap_confirmed"]:
            factors.append("VWAP")
        if signal_row["trend_aligned"]:
            factors.append("Trend")
        sr = str(signal_row["sr_type"])
        if sr == "sr_break":
            factors.append("S/R Break")
        gap = str(signal_row["gap_type"])
        if gap != "no_monday_data":
            factors.append(gap.replace("_", " ").title())
        factors_str = ", ".join(factors) if factors else "None"

        outcome_str = "WIN" if win else "LOSS"
        fig.update_layout(
            title=f"{title}  |  {ticker} {trade_date}  |  Factors: {factors_str}",
            xaxis_title="Time (ET)",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=400,
        )

        trade_figs.append(fig)

    return trade_figs


def build_dashboard():
    print("Loading data...")
    unified, monday, orb_all, rr, lir, turbos, turbos_strat = load_data()
    print(f"  Unified signals: {len(unified)}")
    if rr is not None:
        print(f"  R:R simulation: {len(rr)}")
    if lir is not None:
        print(f"  Let-it-run: {len(lir)}")
    if turbos:
        print(f"  Turbo buffers: {', '.join(sorted(turbos.keys(), key=lambda x: float(x)))}")

    figs = [
        chart_cross_strategy(unified, monday, orb_all),
        chart_stacking(unified),
        chart_signal_tf(unified),
        chart_tf_by_combo(unified),
        chart_long_vs_short(unified),
        chart_factor_heatmap(unified),
        chart_per_ticker(unified),
        chart_gap_type(unified),
    ]

    # Add Let-It-Run charts (primary — matches user's trading style)
    lir_rdist = chart_let_it_run_r_dist(lir)
    if lir_rdist is not None:
        figs.append(lir_rdist)
    lir_factors = chart_let_it_run_factors(lir)
    if lir_factors is not None:
        figs.append(lir_factors)
    lir_outcomes = chart_let_it_run_outcomes(lir)
    if lir_outcomes is not None:
        figs.append(lir_outcomes)

    # Add Turbo simulation charts
    turbo_comp = chart_turbo_buffer_comparison(turbos)
    if turbo_comp is not None:
        figs.append(turbo_comp)
    turbo_eq = chart_turbo_equity_curves(turbos)
    if turbo_eq is not None:
        figs.append(turbo_eq)
    turbo_dist = chart_turbo_pnl_distribution(turbos)
    if turbo_dist is not None:
        figs.append(turbo_dist)

    # Add Turbo × Strategy comparison
    turbo_strat_fig = chart_turbo_strategy_comparison(turbos_strat)
    if turbo_strat_fig is not None:
        figs.append(turbo_strat_fig)

    # Add R:R simulation charts (for comparison)
    rr_fig = chart_rr_simulation(rr)
    if rr_fig is not None:
        figs.append(rr_fig)
    rr_factor_fig = chart_rr_factors(rr)
    if rr_factor_fig is not None:
        figs.append(rr_factor_fig)

    # Add example trade charts (returns a list of individual figures)
    print("  Generating example trades...")
    trade_figs = chart_example_trades(unified)
    if trade_figs:
        figs.extend(trade_figs)

    # Build single HTML using plotly's own to_html (handles encoding correctly)
    html_parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        "<title>ORB + Monday Range Backtest</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }",
        "h1 { text-align: center; color: #f8fafc; margin-bottom: 30px; }",
        ".chart { background: #1e293b; border-radius: 12px; margin: 20px auto; max-width: 1100px; padding: 10px; }",
        "</style></head><body>",
        "<h1>ORB + Monday Range Breakout Backtest</h1>",
        '<div style="max-width:1100px; margin:0 auto 30px; background:#1e293b; border-radius:12px; padding:24px 32px; line-height:1.7; font-size:14px;">',
        '<h2 style="margin-top:0; color:#f8fafc; font-size:18px; border-bottom:1px solid #334155; padding-bottom:10px;">Definitions</h2>',
        '<dl style="margin:0;">',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">ORB (Opening Range Breakout)</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">The high/low of the first 5-minute candle (9:30-9:35) defines the day\'s opening range. '
        'A signal fires when a later candle trades <em>entirely</em> beyond that range &mdash; open, close, high, and low all on the same side (no wick crossing back).</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">Signal Timeframe (1m / 5m / 10m / 15m / 30m / 1h)</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">The candle size used to detect the breakout. The ORB levels are always set by the first 5-min candle; '
        'what changes is how large the confirmation candle is. Larger candles = fewer but higher-conviction signals.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">Monday Range</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">Monday\'s daily high and low define the weekly reference range. '
        'If any day Tue&ndash;Fri closes beyond that range, it\'s a weekly breakout/breakdown. Measured against Friday\'s close.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">Monday Confluence</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">An ORB signal where the entry price is <em>also</em> beyond Monday\'s weekly high (for longs) or low (for shorts). '
        'Two levels breaking at once = higher conviction.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">VWAP Confirmed</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">The ORB signal\'s entry price is beyond 1 standard deviation from the day\'s VWAP (Volume Weighted Average Price). '
        'Indicates the breakout has momentum beyond normal intraday variance.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">Trend-Aligned (SMA Filter)</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">Uses 9 SMAs on the daily close (20, 50, 150, 200, 250, 300, 400, 500, 600). '
        '"Aligned" means going long when price is above the majority of SMAs (uptrend) or shorting when below (downtrend). Trading with the macro trend.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">S/R Break</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">Pivot-based support/resistance zones (15-bar lookback, ATR-scaled width, volume-weighted, with polarity flips). '
        'An ORB breakout that also pushes through a resistance level, or breakdown through support. Uses previous day\'s S/R state to avoid look-ahead bias.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">Gap Type (ORB Open vs Monday)</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">Where the day opens relative to Monday\'s range: '
        '<b>Below</b> = gap down past Monday\'s low, <b>Inside</b> = opens within Monday\'s range, <b>Above</b> = gap up past Monday\'s high.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">Let-It-Run (No TP Cap)</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">Risk = distance from entry to ORB level. '
        'No profit target &mdash; the trade runs until either a no-wick reversal candle exits you (same condition as entry, reversed) '
        'or the day ends. R-Multiple = actual move / risk. MFE = max favorable excursion in R-multiples (how far price went your way). '
        'This matches a trend-following style: let winners run, cut losers at the defined level.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">1.5:1 R:R Exit</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">Risk = distance from entry to ORB level (the natural invalidation). '
        'TP = 1.5&times; that risk from entry (price touch &mdash; any wick counts). '
        'SL = no-wick candle fully retreating past the ORB level (same condition as entry, but reversed). '
        'If neither hits before close, the trade is &ldquo;held&rdquo; and exits at daily close.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">Win Condition</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">A trade "wins" if the market close (last candle of the day) finishes beyond the ORB level in the signal\'s direction. '
        'No take-profit or stop-loss &mdash; purely whether the breakout direction held until end of day.</dd>',

        '<dt style="color:#60a5fa; font-weight:600; margin-top:12px;">Data</dt>',
        '<dd style="margin:2px 0 0 0; color:#94a3b8;">Intraday strategies: ~60 trading days of 5-min data (yfinance max). '
        'Monday Range (standalone): 5 years of daily data. 20 tickers across S&amp;P sector ETFs, individual stocks, and commodity ETFs.</dd>',

        '</dl></div>',
    ]

    for i, fig in enumerate(figs):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font_color="#e2e8f0",
            title_font_size=18,
        )
        fig.update_xaxes(gridcolor="#334155")
        fig.update_yaxes(gridcolor="#334155")

        # Use plotly's to_html which handles binary encoding properly
        # First chart includes plotly.js, rest don't
        include_js = "cdn" if i == 0 else False
        chart_html = fig.to_html(
            full_html=False,
            include_plotlyjs=include_js,
            div_id=f"chart_{i}",
        )
        html_parts.append(f'<div class="chart">{chart_html}</div>')

    html_parts.append("</body></html>")

    OUTPUT.write_text("\n".join(html_parts))
    print(f"\nDashboard saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    build_dashboard()
