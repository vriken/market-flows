"""Dashboard generation with Plotly charts and Jinja2 rendering."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

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
            return "Bearish" if inverse else "Bullish"
        if row["change"] < 0 and row["net"] < 0:
            return "Bullish" if inverse else "Bearish"
        if row["change"] > 0:
            return "Reducing" if inverse else "Covering"
        if row["change"] < 0:
            return "Covering" if inverse else "Reducing"
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


def build_cot_groups(cot_rows):
    """Split COT rows into logical groups and build charts + tables for each."""
    commodities = [r for r in cot_rows if r["trader_type"] == "Managed Money"]
    financials = [r for r in cot_rows if r["trader_type"] == "Leveraged Money"]

    groups = []
    first = True
    for label, rows in [("Commodities (Managed Money)", commodities),
                        ("Financials (Leveraged Money)", financials)]:
        if not rows:
            continue
        # Only the first chart includes plotly.js CDN, rest reuse it
        chart = _build_cot_bar_chart_single(rows, label, include_plotlyjs="cdn" if first else False)
        first = False
        groups.append({"label": label, "chart": chart, "rows": rows})

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


def render_dashboard(cot_rows, etf_rows=None, sentiment_data=None,
                     data_dir=None, output_path=None):
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

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)
    template = env.get_template("dashboard.html")

    html = template.render(
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        alerts=alerts,
        cot_groups=cot_groups,
        cot_history_charts=cot_history_charts,
        etf_groups=etf_groups,
        sentiment=sentiment_data or {},
        data_range=data_range,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"  Dashboard written to {output_path}")
    return output_path
