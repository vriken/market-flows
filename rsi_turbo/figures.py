"""Charts shared by the local dashboard and the GitHub page, styled for market-flows' dark theme."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from param_health import PARTS, WINDOW_YEARS

INK, MUTED, GRID, BASELINE = "#e6edf3", "#8b949e", "#21262d", "#30363d"
POSITIVE, NEGATIVE, NEUTRAL = "#58a6ff", "#f85149", "#30363d"
BAND = "rgba(88,166,255,0.18)"
T_LIMIT = 3


def _layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, size=12), showlegend=False,
                      hoverlabel=dict(bgcolor="#161b22", bordercolor=BASELINE, font=dict(color=INK)))
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, zerolinecolor=BASELINE, tickfont=dict(color=MUTED))
    return fig


def monthly_results(journal):
    """(figure or None, totals) for closed trades, by the month they closed, in SEK."""
    def num(col):
        return pd.to_numeric(journal[col], errors="coerce")
    staked = num("fill_price") * num("qty")
    month = pd.to_datetime(journal.closed_on.where(journal.closed_on != "", journal.bought_on)).dt.to_period("M")
    done = pd.DataFrame({"month": month, "staked": staked, "pnl": staked * num("result_pct") / 100}).dropna()
    totals = {"trades": len(done), "pnl": done.pnl.sum(), "staked": done.staked.sum(),
              "unpriced": int((journal.exit_price == "sold").sum())}
    if done.empty:
        return None, totals
    by = done.groupby("month").agg(pnl=("pnl", "sum"), staked=("staked", "sum"), trades=("pnl", "size"),
                                   wins=("pnl", lambda s: int((s > 0).sum())))
    by = by.reindex(pd.period_range(by.index.min(), by.index.max(), freq="M"), fill_value=0)
    on_stake = np.where(by.staked > 0, by.pnl / by.staked.where(by.staked > 0, 1) * 100, 0)
    x = by.index.to_timestamp()
    fig = go.Figure()
    fig.add_bar(x=x, y=by.pnl, name="Month", marker=dict(color=np.where(by.pnl >= 0, POSITIVE, NEGATIVE), cornerradius=4),
                customdata=np.c_[by.trades, by.wins, on_stake],
                hovertemplate="%{x|%b %Y}: %{y:+,.0f} SEK<br>%{customdata[0]} closed, %{customdata[1]} won, "
                              "%{customdata[2]:+.1f}% on the money staked<extra></extra>")
    fig.add_scatter(x=x, y=by.pnl.cumsum(), name="Running total", mode="lines+markers", line=dict(color=INK, width=2),
                    marker=dict(size=8, line=dict(color="#0d1117", width=2)),
                    hovertemplate="Running total %{y:+,.0f} SEK<extra></extra>")
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    fig.update_yaxes(title=dict(text="SEK", font=dict(color=MUTED)), zeroline=True)
    _layout(fig, 320).update_layout(bargap=0.35, showlegend=True, legend=dict(orientation="h", y=1.12, x=0))
    return fig, totals


def health_heatmap(table):
    """Parts x years, coloured by the year's t-statistic (blue helped, red hurt, grey no evidence either way)."""
    yearly = table[table.window == "year"]
    parts = list(PARTS)
    grid = yearly.pivot(index="part", columns="year", values="t").reindex(parts)
    edge = yearly.pivot(index="part", columns="year", values="edge").reindex(parts)
    trades = yearly.pivot(index="part", columns="year", values="trades").reindex(parts)
    units = np.array([[PARTS[p][1]] * grid.shape[1] for p in parts])
    fig = go.Figure(go.Heatmap(
        z=grid.clip(-T_LIMIT, T_LIMIT).to_numpy(), x=grid.columns, y=[PARTS[p][0] for p in parts],
        zmin=-T_LIMIT, zmid=0, zmax=T_LIMIT, xgap=2, ygap=2,
        colorscale=[[0, NEGATIVE], [0.5, NEUTRAL], [1, POSITIVE]],
        customdata=np.dstack([edge.to_numpy(), trades.to_numpy(), grid.to_numpy()]), text=units,
        hovertemplate="%{y}, %{x}<br>edge %{customdata[0]:+.2f} (%{text})<br>t = %{customdata[2]:.1f} · "
                      "%{customdata[1]:.0f} trades<extra></extra>",
        colorbar=dict(title=dict(text="t", font=dict(color=MUTED)), tickvals=[-3, -2, 0, 2, 3],
                      ticktext=["≤ −3 hurt", "−2", "0", "2", "≥ 3 helped"], tickfont=dict(color=MUTED),
                      thickness=10, outlinewidth=0)))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(dtick=1, gridcolor="rgba(0,0,0,0)")
    return _layout(fig, 300)


def health_trend(table):
    """Rolling edge per part with a ±2 standard error band; the band crossing zero means no proof either way."""
    rolling = table[table.window == f"{WINDOW_YEARS}y"]
    rolling = rolling[rolling.year >= rolling.year.min() + WINDOW_YEARS - 1]
    parts = list(PARTS)
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.09, horizontal_spacing=0.07,
                        subplot_titles=[PARTS[p][0] for p in parts])
    for i, part in enumerate(parts):
        row, col = i // 2 + 1, i % 2 + 1
        s = rolling[rolling.part == part].sort_values("year")
        fig.add_scatter(x=np.r_[s.year, s.year[::-1]], y=np.r_[s.edge + 2 * s.se, (s.edge - 2 * s.se)[::-1]],
                        fill="toself", fillcolor=BAND, line=dict(width=0), hoverinfo="skip", row=row, col=col)
        fig.add_scatter(x=s.year, y=s.edge, mode="lines+markers", line=dict(color=POSITIVE, width=2),
                        marker=dict(size=8, color=np.where(s.edge >= 0, POSITIVE, NEGATIVE),
                                    line=dict(color="#0d1117", width=2)),
                        customdata=np.c_[s.year - WINDOW_YEARS + 1, s.t, s.trades],
                        hovertemplate=f"%{{customdata[0]}}–%{{x}}: %{{y:+.2f}} ({PARTS[part][1]})<br>"
                                      "t = %{customdata[1]:.1f} · %{customdata[2]} trades<extra></extra>",
                        row=row, col=col)
        fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=row, col=col)
    fig.update_annotations(font=dict(size=12, color=INK))
    return _layout(fig, 620)
