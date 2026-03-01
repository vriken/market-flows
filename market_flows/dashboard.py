"""Dashboard generation with Plotly charts and Jinja2 rendering."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

from .config import DATA_DIR

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


def build_cot_bar_chart(cot_rows):
    """Build a horizontal bar chart of net positions colored by percentile."""
    if not cot_rows:
        return ""

    labels = [r["contract"] for r in cot_rows]
    nets = [r["net"] for r in cot_rows]
    colors = [_percentile_color(r["percentile"]) for r in cot_rows]
    hover = [
        f"{r['contract']}<br>Net: {r['net']:+,}<br>Percentile: {r['percentile']:.0f}%<br>{_signal_text(r)}"
        for r in cot_rows
    ]

    fig = go.Figure(go.Bar(
        y=labels,
        x=nets,
        orientation="h",
        marker_color=colors,
        hovertext=hover,
        hoverinfo="text",
    ))
    _plotly_dark_layout(fig, "COT Net Positioning")
    fig.update_layout(height=max(300, len(labels) * 28 + 80))
    fig.update_yaxes(autorange="reversed")
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})


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


def render_dashboard(cot_rows, etf_rows=None, data_dir=None, output_path=None):
    """Render the full dashboard HTML and write to output_path."""
    if data_dir is None:
        data_dir = DATA_DIR
    if output_path is None:
        output_path = Path(data_dir) / "dashboard.html"

    # Add signal text to rows for template
    for r in cot_rows:
        r["signal"] = _signal_text(r)

    alerts = build_alerts(cot_rows)
    cot_bar_chart = build_cot_bar_chart(cot_rows)
    cot_history_charts = build_cot_history_charts(cot_rows, data_dir)

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
        cot_bar_chart=cot_bar_chart,
        cot_rows=cot_rows,
        cot_history_charts=cot_history_charts,
        etf_rows=etf_rows or [],
        data_range=data_range,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"  Dashboard written to {output_path}")
    return output_path
