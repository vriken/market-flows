"""Regime-tagged performance reporting for backtest results.

Generates summary tables, strategy comparisons, and the regime x strategy
performance matrix that drives strategy selection logic.
"""

from __future__ import annotations

import html as html_mod
import logging
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default position size used for ROI calculations when not provided.
DEFAULT_POSITION_SIZE = 10_000.0


class BacktestReport:
    """Analyse a DataFrame of completed trades, sliced by regime and strategy.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Each row is a completed :class:`Trade` (or dict with the same fields).
        Expected columns (see ``strategies.base.Trade``):
            ticker, direction, entry_price, exit_price, pnl, pnl_pct,
            outcome, days_held, quality_score, strategy,
            regime_composite, regime_volatility, regime_cycle,
            regime_risk, regime_monetary.
    position_size : float
        Nominal position size in account currency for ROI calculation.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        position_size: float = DEFAULT_POSITION_SIZE,
    ) -> None:
        self.trades = trades_df.copy()
        self.position_size = position_size

        # Normalise column names: allow both "regime_composite" and
        # "composite_label" variants.
        rename_map = {
            "composite_label": "regime_composite",
            "volatility_state": "regime_volatility",
            "cycle_state": "regime_cycle",
            "risk_state": "regime_risk",
        }
        self.trades.rename(columns={k: v for k, v in rename_map.items() if k in self.trades.columns}, inplace=True)

        # Fill missing regime columns so groupbys never crash.
        for col in ("regime_composite", "regime_volatility", "regime_cycle", "regime_risk", "regime_monetary"):
            if col not in self.trades.columns:
                self.trades[col] = "Unknown"
            else:
                self.trades[col] = self.trades[col].fillna("Unknown")

        if "strategy" not in self.trades.columns:
            self.trades["strategy"] = "Unknown"

    # ------------------------------------------------------------------
    # Core metrics helper
    # ------------------------------------------------------------------

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        """Compute standard metrics for a subset of trades."""
        n = len(df)
        if n == 0:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "ko_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "roi_pct": 0.0,
                "avg_days": 0.0,
                "profit_factor": 0.0,
                "max_consec_wins": 0,
                "max_consec_losses": 0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }

        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        kos = df[df["outcome"] == "ko"] if "outcome" in df.columns else pd.DataFrame()

        total_pnl = df["pnl"].sum()
        gross_wins = wins["pnl"].sum() if len(wins) else 0.0
        gross_losses = abs(losses["pnl"].sum()) if len(losses) else 0.0
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

        # Max consecutive wins / losses
        pnl_signs = (df["pnl"] > 0).astype(int)
        max_consec_wins = _max_consecutive(pnl_signs, target=1)
        max_consec_losses = _max_consecutive(pnl_signs, target=0)

        return {
            "trades": n,
            "win_rate": len(wins) / n * 100,
            "ko_rate": len(kos) / n * 100 if n else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / n,
            "roi_pct": total_pnl / (n * self.position_size) * 100,
            "avg_days": df["days_held"].mean() if "days_held" in df.columns else 0.0,
            "profit_factor": profit_factor,
            "max_consec_wins": max_consec_wins,
            "max_consec_losses": max_consec_losses,
            "best_trade": df["pnl"].max(),
            "worst_trade": df["pnl"].min(),
        }

    # ------------------------------------------------------------------
    # Public reporting methods
    # ------------------------------------------------------------------

    def summary_table(self, group_by: str = "regime_composite") -> pd.DataFrame:
        """Return a DataFrame summarising performance by *group_by* column.

        Default groups by composite regime label.  Pass ``"strategy"`` to
        group by strategy name instead.

        Columns: regime (or strategy), trades, win_rate, ko_rate, roi_pct,
        avg_pnl, avg_days, profit_factor, best_trade, worst_trade.
        """
        if group_by not in self.trades.columns:
            logger.warning("Column %r not in trades — returning empty summary", group_by)
            return pd.DataFrame()

        rows: list[dict] = []
        for label, grp in self.trades.groupby(group_by, sort=False):
            m = self._compute_metrics(grp)
            rows.append({"group": label, **m})

        summary = pd.DataFrame(rows)
        if len(summary):
            summary = summary.sort_values("roi_pct", ascending=False).reset_index(drop=True)
        return summary

    def strategy_comparison(self) -> pd.DataFrame:
        """Compare all strategies side by side (one row per strategy)."""
        return self.summary_table(group_by="strategy")

    def regime_matrix(self) -> pd.DataFrame:
        """Strategy x Regime ROI% matrix — the crown jewel.

        Rows = composite regime labels, columns = strategy names.
        Each cell is the ROI% for that strategy in that regime.
        """
        if self.trades.empty:
            return pd.DataFrame()

        pivot_data: list[dict] = []
        for (regime, strategy), grp in self.trades.groupby(
            ["regime_composite", "strategy"], sort=False
        ):
            m = self._compute_metrics(grp)
            pivot_data.append({
                "regime": regime,
                "strategy": strategy,
                "roi_pct": m["roi_pct"],
                "trades": m["trades"],
            })

        if not pivot_data:
            return pd.DataFrame()

        pdf = pd.DataFrame(pivot_data)
        matrix = pdf.pivot_table(
            index="regime",
            columns="strategy",
            values="roi_pct",
            aggfunc="first",
        )
        return matrix

    def regime_matrix_counts(self) -> pd.DataFrame:
        """Like :meth:`regime_matrix` but cells contain trade counts."""
        if self.trades.empty:
            return pd.DataFrame()

        pivot_data: list[dict] = []
        for (regime, strategy), grp in self.trades.groupby(
            ["regime_composite", "strategy"], sort=False
        ):
            pivot_data.append({
                "regime": regime,
                "strategy": strategy,
                "trades": len(grp),
            })

        pdf = pd.DataFrame(pivot_data)
        return pdf.pivot_table(
            index="regime",
            columns="strategy",
            values="trades",
            aggfunc="first",
        ).fillna(0).astype(int)

    def best_strategy_per_regime(self) -> pd.DataFrame:
        """For each regime, which strategy has the highest ROI%.

        Returns DataFrame with columns: regime, best_strategy, roi_pct, trades.
        """
        matrix = self.regime_matrix()
        if matrix.empty:
            return pd.DataFrame(columns=["regime", "best_strategy", "roi_pct", "trades"])

        counts = self.regime_matrix_counts()
        rows: list[dict] = []
        for regime in matrix.index:
            best_col = matrix.loc[regime].idxmax()
            roi = matrix.loc[regime, best_col]
            n = counts.loc[regime, best_col] if regime in counts.index and best_col in counts.columns else 0
            rows.append({
                "regime": regime,
                "best_strategy": best_col,
                "roi_pct": roi,
                "trades": int(n),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Pretty-print a full analysis to the console."""
        print()
        print("=" * 72)
        print("  BACKTEST REPORT")
        print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  {len(self.trades)} trades, position size {self.position_size:,.0f}")
        print("=" * 72)

        # Overall metrics
        overall = self._compute_metrics(self.trades)
        print()
        print("--- Overall ---")
        _print_metrics(overall)

        # Strategy comparison
        strat_df = self.strategy_comparison()
        if not strat_df.empty:
            print()
            print("--- Strategy Comparison ---")
            print()
            _print_summary_df(strat_df)

        # Regime summary
        regime_df = self.summary_table(group_by="regime_composite")
        if not regime_df.empty:
            print()
            print("--- Performance by Regime ---")
            print()
            _print_summary_df(regime_df)

        # Regime matrix
        matrix = self.regime_matrix()
        if not matrix.empty:
            print()
            print("--- Strategy x Regime ROI% Matrix ---")
            print()
            _print_regime_matrix(matrix)

        # Best strategy per regime
        best = self.best_strategy_per_regime()
        if not best.empty:
            print()
            print("--- Best Strategy per Regime ---")
            print()
            for _, row in best.iterrows():
                print(f"  {row['regime']:<28s} -> {row['best_strategy']:<16s}  "
                      f"ROI {row['roi_pct']:+.1f}%  ({int(row['trades'])} trades)")

        print()
        print("=" * 72)
        print()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Export raw trades and summary tables to CSV files.

        Creates ``<path>_trades.csv``, ``<path>_summary.csv``, and
        ``<path>_matrix.csv``.
        """
        base = Path(path)
        parent = base.parent
        stem = base.stem

        parent.mkdir(parents=True, exist_ok=True)

        trades_path = parent / f"{stem}_trades.csv"
        self.trades.to_csv(trades_path, index=False)
        logger.info("Trades written to %s", trades_path)

        summary_path = parent / f"{stem}_summary.csv"
        self.summary_table().to_csv(summary_path, index=False)
        logger.info("Summary written to %s", summary_path)

        matrix = self.regime_matrix()
        if not matrix.empty:
            matrix_path = parent / f"{stem}_matrix.csv"
            matrix.to_csv(matrix_path)
            logger.info("Matrix written to %s", matrix_path)

    def to_html(self, path: str | Path) -> None:
        """Generate a standalone HTML report with dark-themed tables."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sections: list[str] = []

        # Overall
        overall = self._compute_metrics(self.trades)
        sections.append(_html_metrics_card("Overall", overall))

        # Strategy comparison
        strat_df = self.strategy_comparison()
        if not strat_df.empty:
            sections.append(_html_table("Strategy Comparison", strat_df))

        # Regime summary
        regime_df = self.summary_table(group_by="regime_composite")
        if not regime_df.empty:
            sections.append(_html_table("Performance by Regime", regime_df))

        # Regime matrix
        matrix = self.regime_matrix()
        if not matrix.empty:
            sections.append(_html_regime_matrix("Strategy x Regime ROI%", matrix))

        # Best strategy per regime
        best = self.best_strategy_per_regime()
        if not best.empty:
            sections.append(_html_table("Best Strategy per Regime", best))

        body = "\n".join(sections)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        html_content = _HTML_TEMPLATE.replace("{{TIMESTAMP}}", ts).replace("{{BODY}}", body)

        path.write_text(html_content, encoding="utf-8")
        logger.info("HTML report written to %s", path)


# ======================================================================
# Private helpers
# ======================================================================


def _max_consecutive(series: pd.Series, target: int = 1) -> int:
    """Count the longest consecutive run of *target* in a 0/1 Series."""
    if series.empty:
        return 0
    best = current = 0
    for val in series:
        if val == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _print_metrics(m: dict) -> None:
    """Print a single metrics dict to console."""
    pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "inf"
    print(f"  Trades: {m['trades']:<8d} Win rate: {m['win_rate']:.1f}%   KO rate: {m['ko_rate']:.1f}%")
    print(f"  Total PnL: {m['total_pnl']:>+10,.0f}   Avg PnL: {m['avg_pnl']:>+8,.0f}   ROI: {m['roi_pct']:+.1f}%")
    print(f"  Avg days: {m['avg_days']:.1f}     Profit factor: {pf_str}")
    print(f"  Best trade: {m['best_trade']:>+10,.0f}   Worst: {m['worst_trade']:>+10,.0f}")
    print(f"  Max consec wins: {m['max_consec_wins']}   Max consec losses: {m['max_consec_losses']}")


def _print_summary_df(df: pd.DataFrame) -> None:
    """Print a summary DataFrame in a fixed-width table."""
    # Header
    print(f"  {'Group':<28s} {'#':>5s} {'Win%':>6s} {'KO%':>6s} "
          f"{'ROI%':>7s} {'AvgPnL':>9s} {'Days':>5s} {'PF':>6s}")
    print(f"  {'-' * 28} {'-' * 5} {'-' * 6} {'-' * 6} "
          f"{'-' * 7} {'-' * 9} {'-' * 5} {'-' * 6}")

    for _, row in df.iterrows():
        pf = row.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(
            f"  {str(row['group']):<28s} {int(row['trades']):>5d} "
            f"{row['win_rate']:>5.1f}% {row['ko_rate']:>5.1f}% "
            f"{row['roi_pct']:>+6.1f}% {row['avg_pnl']:>+9,.0f} "
            f"{row['avg_days']:>5.1f} {pf_str:>6s}"
        )


def _print_regime_matrix(matrix: pd.DataFrame) -> None:
    """Print the regime x strategy matrix with formatting."""
    col_width = 12
    label_width = 28

    # Header
    header = f"  {'Regime':<{label_width}s}"
    for col in matrix.columns:
        header += f"{col:>{col_width}s}"
    print(header)
    print(f"  {'-' * label_width}" + ("-" * col_width) * len(matrix.columns))

    for regime in matrix.index:
        line = f"  {str(regime):<{label_width}s}"
        for col in matrix.columns:
            val = matrix.loc[regime, col]
            if pd.isna(val):
                cell = "    ---"
            else:
                cell = f"{val:>+7.1f}%"
            line += f"{cell:>{col_width}s}"
        print(line)


# ======================================================================
# HTML helpers
# ======================================================================


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest Report</title>
<style>
  body {
    background: #0d1117; color: #e6edf3; font-family: system-ui, sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 2rem;
  }
  h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: .5rem; }
  h2 { color: #e6edf3; margin-top: 2rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { padding: .4rem .8rem; text-align: right; border: 1px solid #30363d; }
  th { background: #161b22; color: #8b949e; font-size: .85rem; }
  td { font-size: .9rem; }
  td:first-child, th:first-child { text-align: left; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
          padding: 1rem 1.5rem; margin: 1rem 0; }
  .card dt { color: #8b949e; font-size: .8rem; }
  .card dd { color: #e6edf3; font-size: 1.1rem; margin: 0 0 .5rem 0; }
  .pos { color: #3fb950; } .neg { color: #f85149; }
  .footer { color: #484f58; font-size: .8rem; margin-top: 2rem;
            border-top: 1px solid #30363d; padding-top: .5rem; }
</style>
</head>
<body>
<h1>Backtest Report</h1>
<p style="color:#8b949e">Generated {{TIMESTAMP}}</p>
{{BODY}}
<div class="footer">market-flows backtest engine</div>
</body>
</html>"""


def _pnl_class(val: float) -> str:
    """Return CSS class for positive / negative values."""
    if val > 0:
        return "pos"
    if val < 0:
        return "neg"
    return ""


def _html_metrics_card(title: str, m: dict) -> str:
    """Return an HTML card with overall metrics."""
    pf = m["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "&infin;"
    return f"""
<h2>{html_mod.escape(title)}</h2>
<div class="card">
<dl style="display:grid; grid-template-columns: repeat(4, 1fr); gap: .2rem 1rem;">
  <dt>Trades</dt><dd>{m['trades']}</dd>
  <dt>Win rate</dt><dd>{m['win_rate']:.1f}%</dd>
  <dt>KO rate</dt><dd>{m['ko_rate']:.1f}%</dd>
  <dt>ROI</dt><dd class="{_pnl_class(m['roi_pct'])}">{m['roi_pct']:+.1f}%</dd>
  <dt>Total PnL</dt><dd class="{_pnl_class(m['total_pnl'])}">{m['total_pnl']:+,.0f}</dd>
  <dt>Avg PnL</dt><dd class="{_pnl_class(m['avg_pnl'])}">{m['avg_pnl']:+,.0f}</dd>
  <dt>Profit factor</dt><dd>{pf_str}</dd>
  <dt>Avg days held</dt><dd>{m['avg_days']:.1f}</dd>
  <dt>Best trade</dt><dd class="pos">{m['best_trade']:+,.0f}</dd>
  <dt>Worst trade</dt><dd class="neg">{m['worst_trade']:+,.0f}</dd>
  <dt>Max consec wins</dt><dd>{m['max_consec_wins']}</dd>
  <dt>Max consec losses</dt><dd>{m['max_consec_losses']}</dd>
</dl>
</div>"""


def _html_table(title: str, df: pd.DataFrame) -> str:
    """Render a DataFrame as a dark-themed HTML table."""
    buf = StringIO()
    buf.write(f"<h2>{html_mod.escape(title)}</h2>\n<table>\n<thead><tr>")
    for col in df.columns:
        buf.write(f"<th>{html_mod.escape(str(col))}</th>")
    buf.write("</tr></thead>\n<tbody>\n")
    for _, row in df.iterrows():
        buf.write("<tr>")
        for col in df.columns:
            val = row[col]
            css = ""
            if isinstance(val, (int, float)) and col in ("roi_pct", "total_pnl", "avg_pnl"):
                css = f' class="{_pnl_class(val)}"'
            if isinstance(val, float):
                cell = f"{val:+.1f}" if "pct" in col or "rate" in col or col == "roi_pct" else f"{val:,.1f}"
            else:
                cell = html_mod.escape(str(val))
            buf.write(f"<td{css}>{cell}</td>")
        buf.write("</tr>\n")
    buf.write("</tbody>\n</table>\n")
    return buf.getvalue()


def _html_regime_matrix(title: str, matrix: pd.DataFrame) -> str:
    """Render the regime x strategy matrix as colour-coded HTML table."""
    buf = StringIO()
    buf.write(f"<h2>{html_mod.escape(title)}</h2>\n<table>\n<thead><tr><th>Regime</th>")
    for col in matrix.columns:
        buf.write(f"<th>{html_mod.escape(str(col))}</th>")
    buf.write("</tr></thead>\n<tbody>\n")
    for regime in matrix.index:
        buf.write(f"<tr><td>{html_mod.escape(str(regime))}</td>")
        for col in matrix.columns:
            val = matrix.loc[regime, col]
            if pd.isna(val):
                buf.write('<td style="color:#484f58">---</td>')
            else:
                css_class = _pnl_class(val)
                buf.write(f'<td class="{css_class}">{val:+.1f}%</td>')
        buf.write("</tr>\n")
    buf.write("</tbody>\n</table>\n")
    return buf.getvalue()
