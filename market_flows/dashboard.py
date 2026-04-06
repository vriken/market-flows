"""Dashboard generation with Plotly charts and Jinja2 rendering."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots

from .config import DATA_DIR, ETF_GROUPS

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _plotly_dark_layout(fig, title=""):
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", size=12),
        margin=dict(l=50, r=30, t=50, b=40),
        height=400,
    )
    fig.update_xaxes(gridcolor="#30363d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#30363d", zerolinecolor="#30363d")
    return fig


def _signal_text(row):
    """Generate signal text for a COT row."""
    inverse = row["contract"] == "VIX"
    if row["percentile"] >= 90:
        return "Extreme low" if inverse else "Extreme high"
    if row["percentile"] <= 10:
        return "Extreme high" if inverse else "Extreme low"
    if row.get("change") is not None:
        if row["change"] > 0 and row["net"] > 0:
            return "Adding Short" if inverse else "Adding Long"
        if row["change"] < 0 and row["net"] < 0:
            return "Adding Long" if inverse else "Adding Short"
        if row["change"] > 0:
            return "Trimming Longs" if inverse else "Covering Shorts"
        if row["change"] < 0:
            return "Covering Shorts" if inverse else "Trimming Longs"
    return ""


def _percentile_color(pct):
    """Return a hex color based on percentile (red at extremes, green in middle)."""
    if pct >= 90 or pct <= 10:
        return "#f85149"
    if pct >= 80 or pct <= 20:
        return "#f0883e"
    if pct >= 70 or pct <= 30:
        return "#d29922"
    return "#3fb950"


def build_alerts(cot_rows):
    """Build alert items for contracts at extreme percentiles."""
    alerts = []
    for r in cot_rows:
        inverse = r["contract"] == "VIX"
        if r["percentile"] >= 95:
            level = "extreme-high" if not inverse else "extreme-low"
            label = "Extreme High" if not inverse else "Extreme Low"
            alerts.append({
                "contract": r["contract"],
                "level": level,
                "level_label": label,
                "message": f"Net positioning at {r['percentile']:.0f}th percentile ({r['percentile_label']})",
            })
        elif r["percentile"] <= 5:
            level = "extreme-low" if not inverse else "extreme-high"
            label = "Extreme Low" if not inverse else "Extreme High"
            alerts.append({
                "contract": r["contract"],
                "level": level,
                "level_label": label,
                "message": f"Net positioning at {r['percentile']:.0f}th percentile ({r['percentile_label']})",
            })
        elif r["percentile"] >= 90 or r["percentile"] <= 10:
            alerts.append({
                "contract": r["contract"],
                "level": "elevated",
                "level_label": "Elevated",
                "message": f"Net positioning at {r['percentile']:.0f}th percentile ({r['percentile_label']})",
            })
    return alerts


def _build_cot_bar_chart_single(rows, title, include_plotlyjs="cdn"):
    """Build a horizontal bar chart for a group of COT rows."""
    if not rows:
        return ""

    labels = [r["contract"] for r in rows]
    nets = [r["net"] for r in rows]
    colors = [_percentile_color(r["percentile"]) for r in rows]
    hover = [
        f"{r['contract']}<br>Net: {r['net']:+,}<br>Percentile: {r['percentile']:.0f}%<br>{_signal_text(r)}"
        for r in rows
    ]

    fig = go.Figure(go.Bar(
        y=labels,
        x=nets,
        orientation="h",
        marker_color=colors,
        hovertext=hover,
        hoverinfo="text",
    ))
    _plotly_dark_layout(fig, title)
    fig.update_layout(height=max(250, len(labels) * 32 + 80))
    fig.update_yaxes(autorange="reversed")
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, config={"displayModeBar": False})


_FINANCIALS_SUBGROUPS = {
    "Rates": {"US T-Bonds", "2Y T-Notes", "10Y T-Notes"},
    "Equity Indices": {"S&P 500", "Nasdaq 100", "Russell 2000"},
    "Currencies": {"EUR/USD", "JPY", "GBP"},
    "Volatility & Crypto": {"VIX", "Bitcoin"},
}


def build_cot_groups(cot_rows):
    """Split COT rows into logical groups and build charts + tables for each."""
    commodities = [r for r in cot_rows if r["trader_type"] == "Managed Money"]
    financials = [r for r in cot_rows if r["trader_type"] == "Leveraged Money"]

    groups = []
    first = True

    # Commodities as one group
    if commodities:
        chart = _build_cot_bar_chart_single(commodities, "Commodities (Managed Money)",
                                            include_plotlyjs="cdn")
        first = False
        groups.append({"label": "Commodities (Managed Money)", "chart": chart, "rows": commodities})

    # Split financials into sub-groups so different-scale contracts get their own chart
    for sub_label, contracts in _FINANCIALS_SUBGROUPS.items():
        sub_rows = [r for r in financials if r["contract"] in contracts]
        if not sub_rows:
            continue
        full_label = f"Financials — {sub_label}"
        chart = _build_cot_bar_chart_single(sub_rows, full_label,
                                            include_plotlyjs="cdn" if first else False)
        first = False
        groups.append({"label": full_label, "chart": chart, "rows": sub_rows})

    # Catch any financials not in a sub-group
    categorized = set().union(*_FINANCIALS_SUBGROUPS.values())
    other = [r for r in financials if r["contract"] not in categorized]
    if other:
        chart = _build_cot_bar_chart_single(other, "Financials — Other",
                                            include_plotlyjs="cdn" if first else False)
        first = False
        groups.append({"label": "Financials — Other", "chart": chart, "rows": other})

    return groups


def build_etf_groups(etf_rows):
    """Split ETF rows into the configured ETF_GROUPS for grouped display."""
    if not etf_rows:
        return []

    ticker_to_row = {r["ticker"]: r for r in etf_rows}
    groups = []
    for group_name, tickers in ETF_GROUPS.items():
        group_rows = [ticker_to_row[t] for t in tickers if t in ticker_to_row]
        if group_rows:
            group_rows.sort(key=lambda r: r["flow_m"] or 0, reverse=True)
            has_flow = any(r["flow_m"] is not None for r in group_rows)
            groups.append({"label": group_name, "rows": group_rows, "has_flow": has_flow})

    return groups


def build_cot_history_charts(cot_rows, data_dir=None):
    """Build line charts for contracts at extreme percentiles using parquet history."""
    if data_dir is None:
        data_dir = DATA_DIR
    parquet_path = Path(data_dir) / "cot_history.parquet"
    if not parquet_path.exists():
        return []

    history = pd.read_parquet(parquet_path)
    extreme = [r for r in cot_rows if r["percentile"] >= 90 or r["percentile"] <= 10]
    if not extreme:
        return []

    charts = []
    for r in extreme[:6]:
        contract_data = history[history["contract"] == r["contract"]].sort_values("date")
        if contract_data.empty:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=contract_data["date"],
            y=contract_data["net_position"],
            mode="lines",
            name="Net Position",
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.1)",
        ))
        # Mark current value
        fig.add_trace(go.Scatter(
            x=[contract_data["date"].iloc[-1]],
            y=[r["net"]],
            mode="markers",
            name="Current",
            marker=dict(color=_percentile_color(r["percentile"]), size=10),
            showlegend=False,
        ))
        _plotly_dark_layout(fig, f"{r['contract']} — Net Position ({r['trader_type']})")
        fig.update_layout(height=300, showlegend=False)
        charts.append(fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False}))

    return charts


def build_ratio_time_series_chart(ratio_series):
    """Build a subplot grid of ratio time series line charts."""
    if not ratio_series:
        return ""
    n = len(ratio_series)
    rows = (n + 1) // 2
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[s["label"] for s in ratio_series],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )
    for i, s in enumerate(ratio_series):
        r = i // 2 + 1
        c = i % 2 + 1
        fig.add_trace(go.Scatter(
            x=s["dates"], y=s["values"],
            mode="lines",
            line=dict(color="#58a6ff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
            hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=r, col=c)
    _plotly_dark_layout(fig, "")
    fig.update_layout(height=280 * rows, margin=dict(l=50, r=30, t=40, b=30))
    fig.update_annotations(font_size=12, font_color="#8b949e")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_sector_rotation_heatmap(rotation_data):
    """Build a heatmap of weekly sector returns."""
    if not rotation_data:
        return ""
    returns = rotation_data["returns"]
    fig = go.Figure(go.Heatmap(
        z=list(zip(*returns, strict=True)),  # transpose: sectors on Y, weeks on X
        x=rotation_data["week_labels"],
        y=rotation_data["sectors"],
        colorscale=[
            [0.0, "#f85149"],
            [0.5, "#161b22"],
            [1.0, "#3fb950"],
        ],
        zmid=0,
        text=[[f"{v*100:+.1f}%" for v in week] for week in zip(*returns, strict=True)],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Sector: %{y}<br>Week: %{x}<br>Return: %{text}<extra></extra>",
        colorbar=dict(title="Return", tickformat=".0%"),
    ))
    _plotly_dark_layout(fig, "")
    fig.update_layout(height=450, margin=dict(l=80, r=30, t=20, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_etf_flow_history_chart(flow_data):
    """Build a multi-line chart of cumulative ETF flows."""
    if not flow_data or not flow_data.get("has_data"):
        return ""
    colors = ["#58a6ff", "#3fb950", "#f0883e", "#f85149", "#d29922",
              "#bc8cff", "#79c0ff", "#56d364", "#ffa657", "#ff7b72"]
    fig = go.Figure()
    for i, s in enumerate(flow_data["series"]):
        fig.add_trace(go.Scatter(
            x=s["dates"], y=s["cumulative_flows"],
            mode="lines+markers",
            name=s["ticker"],
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=5),
            hovertemplate="%{x}<br>$%{y:+,.0f}M<extra>" + s["ticker"] + "</extra>",
        ))
    _plotly_dark_layout(fig, "")
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Cumulative Est. Flow ($M)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_yield_curve_chart(yield_data):
    """Build a line chart of current yield curve (maturities vs yield %)."""
    if not yield_data or "yields" not in yield_data:
        return ""

    yields = yield_data["yields"]
    maturities = list(yields.keys())
    values = list(yields.values())

    # Color segments: green if normal (rising), red if inverted (falling)
    colors = []
    for i in range(len(values)):
        if i == 0 or values[i] >= values[i - 1]:
            colors.append("#3fb950")
        else:
            colors.append("#f85149")

    fig = go.Figure()
    # Draw segments with appropriate colors
    for i in range(len(maturities) - 1):
        segment_color = colors[i + 1]
        fig.add_trace(go.Scatter(
            x=[maturities[i], maturities[i + 1]],
            y=[values[i], values[i + 1]],
            mode="lines+markers",
            line=dict(color=segment_color, width=3),
            marker=dict(size=8, color=segment_color),
            showlegend=False,
            hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
        ))

    _plotly_dark_layout(fig, "Treasury Yield Curve")
    fig.update_layout(
        height=350,
        xaxis_title="Maturity",
        yaxis_title="Yield (%)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_yield_spread_chart(yield_history):
    """Build a dual-line chart of 2s10s and 3m10y spreads over time."""
    if not yield_history:
        return ""

    dates = yield_history["dates"]
    fig = go.Figure()

    if "spread_2s10s" in yield_history:
        fig.add_trace(go.Scatter(
            x=dates, y=yield_history["spread_2s10s"],
            mode="lines", name="2s10s Spread",
            line=dict(color="#58a6ff", width=2),
            hovertemplate="%{x}<br>2s10s: %{y:.3f}%<extra></extra>",
        ))

    if "spread_3m10y" in yield_history:
        fig.add_trace(go.Scatter(
            x=dates, y=yield_history["spread_3m10y"],
            mode="lines", name="3m10y Spread",
            line=dict(color="#bc8cff", width=2),
            hovertemplate="%{x}<br>3m10y: %{y:.3f}%<extra></extra>",
        ))

    # Zero line for inversion threshold
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5,
                  annotation_text="Inversion", annotation_position="bottom right",
                  annotation_font_color="#8b949e")

    _plotly_dark_layout(fig, "Yield Spreads (1Y)")
    fig.update_layout(
        height=350,
        yaxis_title="Spread (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_margin_debt_chart(margin_data):
    """Build a line chart of FINRA margin debit balances over time."""
    if not margin_data:
        return ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=margin_data["dates"], y=margin_data["debit_balances"],
        mode="lines", name="Debit Balances",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        hovertemplate="%{x}<br>$%{y:,.1f}" + margin_data.get("unit", "M") + "<extra></extra>",
    ))

    _plotly_dark_layout(fig, "FINRA Margin Debt — Debit Balances")
    fig.update_layout(
        height=400,
        yaxis_title=f"Debit Balances (${margin_data.get('unit', 'M')})",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_fund_flows_chart(flows_data):
    """Build a multi-line chart of FRED fund flow series."""
    if not flows_data or not flows_data.get("has_data"):
        return ""

    colors = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff"]
    fig = go.Figure()

    for i, s in enumerate(flows_data["series"]):
        fig.add_trace(go.Scatter(
            x=s["dates"], y=s["values"],
            mode="lines+markers",
            name=s["name"],
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=3),
            hovertemplate="%{x}<br>%{y:,.0f}<extra>" + s["name"] + "</extra>",
        ))

    _plotly_dark_layout(fig, "Fund Flows (FRED, Quarterly)")
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_aaii_sentiment_chart(aaii_data):
    """Build a stacked area chart of AAII bull/neutral/bear sentiment."""
    if not aaii_data:
        return ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aaii_data["dates"], y=aaii_data["bullish"],
        mode="lines", name="Bullish",
        line=dict(width=0.5, color="#3fb950"),
        fillcolor="rgba(63,185,80,0.4)",
        stackgroup="sentiment",
        hovertemplate="%{x}<br>Bullish: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=aaii_data["dates"], y=aaii_data["neutral"],
        mode="lines", name="Neutral",
        line=dict(width=0.5, color="#8b949e"),
        fillcolor="rgba(139,148,158,0.3)",
        stackgroup="sentiment",
        hovertemplate="%{x}<br>Neutral: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=aaii_data["dates"], y=aaii_data["bearish"],
        mode="lines", name="Bearish",
        line=dict(width=0.5, color="#f85149"),
        fillcolor="rgba(248,81,73,0.4)",
        stackgroup="sentiment",
        hovertemplate="%{x}<br>Bearish: %{y:.1f}%<extra></extra>",
    ))

    _plotly_dark_layout(fig, "AAII Investor Sentiment")
    fig.update_layout(
        height=400,
        yaxis_title="Percentage (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_real_yields_chart(ry_data):
    """Build a multi-line chart of real yields and inflation breakevens."""
    if not ry_data or "series" not in ry_data:
        return ""

    colors = {
        "10Y Real Yield": "#58a6ff",
        "10Y Breakeven": "#f0883e",
        "5Y Breakeven": "#d29922",
        "5Y5Y Fwd Inflation": "#bc8cff",
    }
    fig = go.Figure()
    for label, series in ry_data["series"].items():
        fig.add_trace(go.Scatter(
            x=series["dates"], y=series["values"],
            mode="lines", name=label,
            line=dict(color=colors.get(label, "#58a6ff"), width=2),
            hovertemplate="%{x}<br>" + label + ": %{y:.3f}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.4)

    _plotly_dark_layout(fig, "Real Yields & Inflation Breakevens")
    fig.update_layout(
        height=400,
        yaxis_title="Yield / Rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_jobless_claims_chart(claims_data):
    """Build a line chart of initial jobless claims with 4-week MA."""
    if not claims_data or "series" not in claims_data:
        return ""

    ic = claims_data["series"].get("Initial Claims")
    if not ic:
        return ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ic["dates"], y=ic["values"],
        mode="lines", name="Initial Claims",
        line=dict(color="rgba(88,166,255,0.3)", width=1),
        hovertemplate="%{x}<br>Claims: %{y:,.0f}<extra></extra>",
    ))

    if claims_data.get("ma4"):
        fig.add_trace(go.Scatter(
            x=ic["dates"], y=claims_data["ma4"],
            mode="lines", name="4-Week MA",
            line=dict(color="#58a6ff", width=2),
            hovertemplate="%{x}<br>4W MA: %{y:,.0f}<extra></extra>",
        ))

    cc = claims_data["series"].get("Continued Claims")
    if cc:
        fig.add_trace(go.Scatter(
            x=cc["dates"], y=cc["values"],
            mode="lines", name="Continued Claims",
            line=dict(color="#f0883e", width=1.5),
            yaxis="y2",
            hovertemplate="%{x}<br>Continued: %{y:,.0f}<extra></extra>",
        ))

    _plotly_dark_layout(fig, "Jobless Claims (Weekly)")
    fig.update_layout(
        height=400,
        yaxis=dict(title="Initial Claims"),
        yaxis2=dict(
            title="Continued Claims", titlefont_color="#f0883e",
            overlaying="y", side="right",
            gridcolor="rgba(48,54,61,0.3)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_nfci_chart(nfci_data):
    """Build a line chart of the Chicago Fed NFCI."""
    if not nfci_data:
        return ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nfci_data["dates"], y=nfci_data["values"],
        mode="lines", name="NFCI",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        hovertemplate="%{x}<br>NFCI: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5,
                  annotation_text="Zero (avg conditions)", annotation_position="bottom right",
                  annotation_font_color="#8b949e")

    _plotly_dark_layout(fig, "Chicago Fed National Financial Conditions Index")
    fig.update_layout(
        height=350,
        yaxis_title="NFCI",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_move_chart(msd_data):
    """Build a line chart of the MOVE index (bond volatility)."""
    if not msd_data or "MOVE" not in msd_data:
        return ""

    move = msd_data["MOVE"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=move["dates"], y=move["values"],
        mode="lines", name="MOVE",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        hovertemplate="%{x}<br>MOVE: %{y:.1f}<extra></extra>",
    ))

    # Reference levels
    fig.add_hline(y=120, line_dash="dot", line_color="#f0883e", opacity=0.5,
                  annotation_text="120 (Elevated)", annotation_position="top right",
                  annotation_font_color="#f0883e")
    fig.add_hline(y=140, line_dash="dot", line_color="#f85149", opacity=0.5,
                  annotation_text="140 (Crisis)", annotation_position="top right",
                  annotation_font_color="#f85149")

    _plotly_dark_layout(fig, "ICE BofA MOVE Index (Bond Volatility)")
    fig.update_layout(height=350)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_skew_chart(msd_data):
    """Build a line chart of the CBOE SKEW index (tail risk)."""
    if not msd_data or "SKEW" not in msd_data:
        return ""

    skew = msd_data["SKEW"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=skew["dates"], y=skew["values"],
        mode="lines", name="SKEW",
        line=dict(color="#bc8cff", width=2),
        fill="tozeroy",
        fillcolor="rgba(188,140,255,0.08)",
        hovertemplate="%{x}<br>SKEW: %{y:.1f}<extra></extra>",
    ))

    fig.add_hline(y=130, line_dash="dot", line_color="#d29922", opacity=0.5,
                  annotation_text="130 (Moderate tail risk)", annotation_position="top right",
                  annotation_font_color="#d29922")
    fig.add_hline(y=150, line_dash="dot", line_color="#f85149", opacity=0.5,
                  annotation_text="150 (Elevated tail risk)", annotation_position="top right",
                  annotation_font_color="#f85149")

    _plotly_dark_layout(fig, "CBOE SKEW Index (Tail Risk)")
    fig.update_layout(height=350)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_dxy_chart(msd_data):
    """Build a line chart of DXY with 50/200 DMA overlay."""
    if not msd_data or "DXY" not in msd_data:
        return ""

    dxy = msd_data["DXY"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dxy["dates"], y=dxy["values"],
        mode="lines", name="DXY",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="%{x}<br>DXY: %{y:.2f}<extra></extra>",
    ))

    if "ma50" in dxy:
        fig.add_trace(go.Scatter(
            x=dxy["ma50_dates"], y=dxy["ma50"],
            mode="lines", name="50 DMA",
            line=dict(color="#d29922", width=1, dash="dash"),
            hovertemplate="%{x}<br>50 DMA: %{y:.2f}<extra></extra>",
        ))

    if "ma200" in dxy:
        fig.add_trace(go.Scatter(
            x=dxy["ma200_dates"], y=dxy["ma200"],
            mode="lines", name="200 DMA",
            line=dict(color="#f0883e", width=1, dash="dot"),
            hovertemplate="%{x}<br>200 DMA: %{y:.2f}<extra></extra>",
        ))

    _plotly_dark_layout(fig, "US Dollar Index (DXY)")
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_putcall_chart(putcall_data):
    """Build a line chart of live equity put/call ratio with 20-day MA."""
    if not putcall_data or not putcall_data.get("ratios"):
        return ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=putcall_data["dates"], y=putcall_data["ratios"],
        mode="lines", name="P/C Ratio",
        line=dict(color="rgba(88,166,255,0.3)", width=1),
        hovertemplate="%{x}<br>P/C: %{y:.3f}<extra></extra>",
    ))

    if putcall_data.get("ma20"):
        fig.add_trace(go.Scatter(
            x=putcall_data["dates"], y=putcall_data["ma20"],
            mode="lines", name="20-Day MA",
            line=dict(color="#58a6ff", width=2),
            hovertemplate="%{x}<br>MA(20): %{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0.7, line_dash="dash", line_color="#3fb950", opacity=0.5,
                  annotation_text="0.70 (Bullish)", annotation_position="bottom right",
                  annotation_font_color="#3fb950")
    fig.add_hline(y=1.0, line_dash="dash", line_color="#f85149", opacity=0.5,
                  annotation_text="1.00 (Bearish)", annotation_position="top right",
                  annotation_font_color="#f85149")

    _plotly_dark_layout(fig, "Equity Put/Call Ratio (SPY Options)")
    fig.update_layout(
        height=350,
        yaxis_title="Put/Call Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_fear_greed_chart(fg_data):
    """Build a gauge + history chart for CNN Fear & Greed Index."""
    if not fg_data:
        return ""

    value = fg_data.get("current_value", 50)

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "CNN Fear & Greed Index", "font": {"color": "#e6edf3", "size": 16}},
        number={"font": {"color": "#e6edf3", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#30363d", "tickfont": {"color": "#8b949e"}},
            "bar": {"color": "#58a6ff"},
            "bgcolor": "#21262d",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(248,81,73,0.3)"},
                {"range": [25, 45], "color": "rgba(240,136,62,0.3)"},
                {"range": [45, 55], "color": "rgba(139,148,158,0.2)"},
                {"range": [55, 75], "color": "rgba(63,185,80,0.2)"},
                {"range": [75, 100], "color": "rgba(63,185,80,0.4)"},
            ],
            "threshold": {
                "line": {"color": "#e6edf3", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))

    _plotly_dark_layout(fig, "")
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10))

    gauge_html = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

    # History line chart (if we have accumulated data)
    history_html = ""
    dates = fg_data.get("dates", [])
    values = fg_data.get("values", [])
    if len(dates) > 1:
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode="lines+markers",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
            hovertemplate="%{x}<br>F&G: %{y:.0f}<extra></extra>",
        ))
        # Zone reference lines
        hist_fig.add_hline(y=25, line_dash="dot", line_color="#f85149", opacity=0.4,
                           annotation_text="Extreme Fear", annotation_position="bottom right",
                           annotation_font_color="#f85149", annotation_font_size=10)
        hist_fig.add_hline(y=75, line_dash="dot", line_color="#3fb950", opacity=0.4,
                           annotation_text="Extreme Greed", annotation_position="top right",
                           annotation_font_color="#3fb950", annotation_font_size=10)
        _plotly_dark_layout(hist_fig, "Fear & Greed History")
        hist_fig.update_layout(height=300, yaxis=dict(range=[0, 100]))
        history_html = hist_fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

    return gauge_html, history_html


def build_credit_spread_chart(credit_data):
    """Build a dual-line chart of HY OAS and IG OAS credit spreads."""
    if not credit_data:
        return ""

    fig = go.Figure()

    if "hy_oas" in credit_data:
        hy = credit_data["hy_oas"]
        fig.add_trace(go.Scatter(
            x=hy["dates"], y=hy["values"],
            mode="lines", name="HY OAS",
            line=dict(color="#f85149", width=2),
            hovertemplate="%{x}<br>HY OAS: %{y:.0f} bps<extra></extra>",
        ))

    if "ig_oas" in credit_data:
        ig = credit_data["ig_oas"]
        fig.add_trace(go.Scatter(
            x=ig["dates"], y=ig["values"],
            mode="lines", name="IG OAS",
            line=dict(color="#58a6ff", width=2),
            yaxis="y2",
            hovertemplate="%{x}<br>IG OAS: %{y:.0f} bps<extra></extra>",
        ))

    _plotly_dark_layout(fig, "Credit Spreads (OAS)")
    fig.update_layout(
        height=400,
        yaxis=dict(title="HY OAS (bps)", titlefont_color="#f85149"),
        yaxis2=dict(
            title="IG OAS (bps)", titlefont_color="#58a6ff",
            overlaying="y", side="right",
            gridcolor="rgba(48,54,61,0.3)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    # Annotate current values
    annotations = []
    if credit_data.get("current_hy") is not None:
        annotations.append(f"HY: {credit_data['current_hy']:.0f}bps ({credit_data.get('hy_percentile', 0):.0f}th %ile)")
    if credit_data.get("current_ig") is not None:
        annotations.append(f"IG: {credit_data['current_ig']:.0f}bps ({credit_data.get('ig_percentile', 0):.0f}th %ile)")
    if annotations:
        fig.add_annotation(
            text=" · ".join(annotations),
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(color="#8b949e", size=11),
        )

    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_fed_liquidity_chart(liquidity_data):
    """Build an area chart of Fed Net Liquidity with component lines."""
    if not liquidity_data:
        return ""

    dates = liquidity_data["dates"]
    fig = go.Figure()

    # Net liquidity as filled area (primary)
    fig.add_trace(go.Scatter(
        x=dates, y=liquidity_data["net_liquidity"],
        mode="lines", name="Net Liquidity",
        line=dict(color="#3fb950", width=2),
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.1)",
        hovertemplate="%{x}<br>Net Liq: $%{y:,.0f}M<extra></extra>",
    ))

    # Component lines (secondary, thinner)
    if "walcl" in liquidity_data:
        fig.add_trace(go.Scatter(
            x=dates, y=liquidity_data["walcl"],
            mode="lines", name="Fed Balance Sheet",
            line=dict(color="#58a6ff", width=1, dash="dot"),
            visible="legendonly",
            hovertemplate="%{x}<br>WALCL: $%{y:,.0f}M<extra></extra>",
        ))
    if "rrp" in liquidity_data:
        fig.add_trace(go.Scatter(
            x=dates, y=liquidity_data["rrp"],
            mode="lines", name="Reverse Repo",
            line=dict(color="#f0883e", width=1, dash="dot"),
            visible="legendonly",
            hovertemplate="%{x}<br>RRP: $%{y:,.0f}M<extra></extra>",
        ))
    if "tga" in liquidity_data:
        fig.add_trace(go.Scatter(
            x=dates, y=liquidity_data["tga"],
            mode="lines", name="TGA",
            line=dict(color="#bc8cff", width=1, dash="dot"),
            visible="legendonly",
            hovertemplate="%{x}<br>TGA: $%{y:,.0f}M<extra></extra>",
        ))

    _plotly_dark_layout(fig, "Fed Net Liquidity (WALCL − RRP − TGA)")
    fig.update_layout(
        height=400,
        yaxis_title="$ Millions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    # Annotate 4-week change
    if liquidity_data.get("net_change_4w_pct") is not None:
        chg = liquidity_data["net_change_4w_pct"]
        color = "#3fb950" if chg > 0 else "#f85149"
        fig.add_annotation(
            text=f"4W Change: {chg:+.2f}%",
            xref="paper", yref="paper", x=0.98, y=0.95,
            showarrow=False, font=dict(color=color, size=12),
            xanchor="right",
        )

    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_breadth_chart(breadth_data):
    """Build a dual-line chart of S&P 500 % above 50 DMA and 200 DMA."""
    if not breadth_data:
        return ""

    fig = go.Figure()

    if "pct_above_50dma" in breadth_data:
        d50 = breadth_data["pct_above_50dma"]
        fig.add_trace(go.Scatter(
            x=d50["dates"], y=d50["values"],
            mode="lines", name="% > 50 DMA",
            line=dict(color="#58a6ff", width=2),
            hovertemplate="%{x}<br>Above 50 DMA: %{y:.1f}%<extra></extra>",
        ))

    if "pct_above_200dma" in breadth_data:
        d200 = breadth_data["pct_above_200dma"]
        fig.add_trace(go.Scatter(
            x=d200["dates"], y=d200["values"],
            mode="lines", name="% > 200 DMA",
            line=dict(color="#3fb950", width=2),
            hovertemplate="%{x}<br>Above 200 DMA: %{y:.1f}%<extra></extra>",
        ))

    # Reference lines
    fig.add_hline(y=50, line_dash="dash", line_color="#8b949e", opacity=0.4,
                  annotation_text="50% (neutral)", annotation_position="bottom right",
                  annotation_font_color="#8b949e")
    fig.add_hline(y=80, line_dash="dot", line_color="#3fb950", opacity=0.3,
                  annotation_text="80%", annotation_position="top right",
                  annotation_font_color="#3fb950")
    fig.add_hline(y=20, line_dash="dot", line_color="#f85149", opacity=0.3,
                  annotation_text="20%", annotation_position="bottom right",
                  annotation_font_color="#f85149")

    _plotly_dark_layout(fig, "S&P 500 Market Breadth")
    fig.update_layout(
        height=400,
        yaxis_title="% of Constituents",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_highs_lows_chart(breadth_data):
    """Build a bar chart of daily new 52-week highs vs lows."""
    if not breadth_data or "highs_lows_history" not in breadth_data:
        return ""

    hl = breadth_data["highs_lows_history"]
    dates = hl["dates"]
    highs = hl["highs"]
    lows = hl["lows"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=highs,
        name="New 52W Highs",
        marker_color="#3fb950",
        hovertemplate="%{x}<br>Highs: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=dates, y=[-v for v in lows],
        name="New 52W Lows",
        marker_color="#f85149",
        hovertemplate="%{x}<br>Lows: " + "%{customdata}<extra></extra>",
        customdata=lows,
    ))

    _plotly_dark_layout(fig, "S&P 500 — New 52-Week Highs vs Lows")
    fig.update_layout(
        height=350,
        barmode="relative",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def render_dashboard(cot_rows, etf_rows=None, sentiment_data=None,
                     data_dir=None, output_path=None,
                     ratio_series=None, rotation_data=None, flow_data=None,
                     external_data=None, orb_conditions=None, regime=None,
                     credit_data=None, liquidity_data=None, breadth_data=None,
                     fear_greed_data=None, real_yields_data=None,
                     jobless_data=None, nfci_data=None, msd_data=None,
                     putcall_data=None, freshness=None,
                     strategy_matrix=None):
    """Render the full dashboard HTML and write to output_path."""
    if data_dir is None:
        data_dir = DATA_DIR
    if output_path is None:
        output_path = Path(data_dir) / "dashboard.html"

    # Add signal text to rows for template
    for r in cot_rows:
        r["signal"] = _signal_text(r)

    alerts = build_alerts(cot_rows)
    cot_groups = build_cot_groups(cot_rows)
    cot_history_charts = build_cot_history_charts(cot_rows, data_dir)
    etf_groups = build_etf_groups(etf_rows)

    # Data range
    parquet_path = Path(data_dir) / "cot_history.parquet"
    if parquet_path.exists():
        history = pd.read_parquet(parquet_path)
        data_range = f"{history['date'].min().date()} to {history['date'].max().date()}"
    else:
        data_range = "Current year only"

    # Build new visualization charts
    ratio_chart_html = build_ratio_time_series_chart(ratio_series) if ratio_series else ""
    rotation_chart_html = build_sector_rotation_heatmap(rotation_data) if rotation_data else ""
    flow_chart_html = build_etf_flow_history_chart(flow_data) if flow_data else ""

    # Build yield curve charts (from sentiment_data)
    yield_curve_chart = ""
    yield_spread_chart = ""
    if sentiment_data:
        yield_curve_chart = build_yield_curve_chart(sentiment_data.get("yield_curve"))
        yield_spread_chart = build_yield_spread_chart(sentiment_data.get("yield_history"))

    # Build external data charts
    ext = external_data or {}
    margin_chart = build_margin_debt_chart(ext.get("margin_debt"))
    fund_flows_chart = build_fund_flows_chart(ext.get("fred_flows"))
    aaii_chart = build_aaii_sentiment_chart(ext.get("aaii"))

    # Build macro & liquidity charts
    credit_chart = build_credit_spread_chart(credit_data)
    liquidity_chart = build_fed_liquidity_chart(liquidity_data)
    breadth_chart = build_breadth_chart(breadth_data)
    highs_lows_chart = build_highs_lows_chart(breadth_data)

    # Build Fear & Greed chart
    fg_gauge_chart = ""
    fg_history_chart = ""
    if fear_greed_data:
        fg_result = build_fear_greed_chart(fear_greed_data)
        if fg_result:
            fg_gauge_chart, fg_history_chart = fg_result

    # Build Rates & Dollar charts
    real_yields_chart = build_real_yields_chart(real_yields_data)
    jobless_chart = build_jobless_claims_chart(jobless_data)
    nfci_chart = build_nfci_chart(nfci_data)
    move_chart = build_move_chart(msd_data)
    skew_chart = build_skew_chart(msd_data)
    dxy_chart = build_dxy_chart(msd_data)
    putcall_chart_html = build_putcall_chart(putcall_data)

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("dashboard.html")

    html = template.render(
        last_updated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        regime=regime or {},
        alerts=alerts,
        cot_groups=cot_groups,
        cot_history_charts=cot_history_charts,
        etf_groups=etf_groups,
        sentiment=sentiment_data or {},
        data_range=data_range,
        ratio_chart=ratio_chart_html,
        rotation_chart=rotation_chart_html,
        flow_chart=flow_chart_html,
        flow_data=flow_data or {},
        yield_curve_chart=yield_curve_chart,
        yield_spread_chart=yield_spread_chart,
        margin_data=ext.get("margin_debt"),
        margin_chart=margin_chart,
        aaii_data=ext.get("aaii"),
        aaii_chart=aaii_chart,
        fund_flows_chart=fund_flows_chart,
        orb=orb_conditions or {},
        credit_data=credit_data,
        credit_chart=credit_chart,
        liquidity_data=liquidity_data,
        liquidity_chart=liquidity_chart,
        breadth_data=breadth_data,
        breadth_chart=breadth_chart,
        highs_lows_chart=highs_lows_chart,
        fear_greed=fear_greed_data,
        fg_gauge_chart=fg_gauge_chart,
        fg_history_chart=fg_history_chart,
        real_yields_data=real_yields_data,
        real_yields_chart=real_yields_chart,
        jobless_data=jobless_data,
        jobless_chart=jobless_chart,
        nfci_data=nfci_data,
        nfci_chart=nfci_chart,
        msd_data=msd_data or {},
        move_chart=move_chart,
        skew_chart=skew_chart,
        dxy_chart=dxy_chart,
        putcall_data=putcall_data,
        putcall_chart=putcall_chart_html,
        freshness=freshness or {},
        strategy_matrix=strategy_matrix or [],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"  Dashboard written to {output_path}")
    return output_path
