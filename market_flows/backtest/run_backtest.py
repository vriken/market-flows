"""CLI entry point for running regime-aware backtests.

Usage
-----
    python -m market_flows.backtest.run_backtest [options]

Options
-------
    --strategy {orb,vwap,fvg,momentum,pdhl,all}  Strategy to test (default: all)
    --tickers TICKER [TICKER ...]                  Ticker symbols (default: ORB universe)
    --start YYYY-MM-DD                             Start date (default: 2025-01-01)
    --end YYYY-MM-DD                               End date (default: today)
    --output {console,csv,html}                    Report format (default: console)
    --output-path PATH                             File path for csv/html output
    --regime-filter LABEL                          Only include trades in this regime
    --position-size N                              Nominal position size (default: 10000)
    --verbose / -v                                 Enable debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .report import BacktestReport, DEFAULT_POSITION_SIZE

logger = logging.getLogger(__name__)

# ======================================================================
# ORB universe (~60 instruments)
# ======================================================================

_SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLB", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC",
]

_STOCKS = [
    # Financials
    "JPM", "BRK-B", "V", "MA", "BAC",
    # Tech / Semis
    "INTC", "NVDA", "AAPL", "MSFT", "AVGO", "MU", "GOOGL", "GOOG",
    # Internet / Media
    "NFLX", "META", "AMZN", "VZ",
    # Energy
    "XOM", "CVX", "COP", "SLB", "WMB",
    # Healthcare
    "LLY", "JNJ", "ABBV", "MRK", "UNH",
    # Materials
    "LIN", "NEM", "FCX", "CRH", "SHW",
    # Industrials / Aerospace
    "GE", "CAT", "RTX", "GEV", "BA",
    # Consumer Discretionary
    "TSLA", "HD", "MCD", "TJX",
    # Consumer Staples
    "WMT", "COST", "PG", "KO", "PM",
    # Utilities
    "NEE", "SO", "DUK", "CEG", "AEP",
    # REITs
    "WELL", "PLD", "AMT", "EQIX", "SPG",
]

_COMMODITIES = ["GLD", "SLV", "CPER", "PPLT"]

DEFAULT_TICKERS: list[str] = _SECTOR_ETFS + _STOCKS + _COMMODITIES

# Strategy name -> import path mapping
STRATEGY_MAP = {
    "orb": "ORBStrategy",
    "vwap": "VWAPReversionStrategy",
    "fvg": "FVGStrategy",
    "momentum": "MomentumStrategy",
    "pdhl": "PDHLStrategy",
}


# ======================================================================
# Strategy loader
# ======================================================================


def _load_strategies(names: list[str]):
    """Lazily import and instantiate requested strategy classes.

    Returns list of strategy instances.
    """
    from .strategies import (  # noqa: F401 — conditional import
        FVGStrategy,
        MomentumStrategy,
        ORBStrategy,
        PDHLStrategy,
        VWAPReversionStrategy,
    )

    class_map = {
        "ORBStrategy": ORBStrategy,
        "VWAPReversionStrategy": VWAPReversionStrategy,
        "FVGStrategy": FVGStrategy,
        "MomentumStrategy": MomentumStrategy,
        "PDHLStrategy": PDHLStrategy,
    }

    strategies = []
    for name in names:
        cls_name = STRATEGY_MAP.get(name)
        if cls_name is None:
            logger.warning("Unknown strategy %r — skipping", name)
            continue
        cls = class_map.get(cls_name)
        if cls is None:
            logger.warning("Strategy class %r not found — skipping", cls_name)
            continue
        strategies.append(cls())
    return strategies


# ======================================================================
# Regime history builder
# ======================================================================


def _build_regime_history(start: date, end: date) -> dict[date, dict]:
    """Build a mapping of date -> regime classification.

    Uses the market_flows.regime module with historical data when
    available, falling back to a static 'Unknown' regime otherwise.

    Returns:
        dict mapping dates to regime dicts with keys matching
        Trade regime fields.
    """
    try:
        from ..sentiment import (
            fetch_leverage_ratios,
            fetch_market_ratios,
            fetch_vix_term_structure,
            fetch_yield_curve,
            fetch_yield_curve_history,
        )
        from ..regime import classify_regime

        # Fetch current regime as a baseline (historical reconstruction
        # is strategy-engine work; here we provide a snapshot).
        sentiment_data = {
            "vix": fetch_vix_term_structure(),
            "leverage": fetch_leverage_ratios(),
            "ratios": fetch_market_ratios(),
            "yield_curve": fetch_yield_curve(),
            "yield_history": fetch_yield_curve_history(),
        }
        regime = classify_regime(sentiment_data)

        # For now, apply the current regime to all dates.  A full
        # historical reconstruction requires time-series data per day
        # and will be added in a later sprint.
        regime_entry = {
            "regime_composite": regime["composite_label"],
            "regime_volatility": next(
                (d["state"] for d in regime["dimensions"] if d["name"] == "Volatility"), "Unknown"
            ),
            "regime_cycle": next(
                (d["state"] for d in regime["dimensions"] if d["name"] == "Cycle"), "Unknown"
            ),
            "regime_risk": next(
                (d["state"] for d in regime["dimensions"] if d["name"] == "Risk"), "Unknown"
            ),
            "regime_monetary": next(
                (d["state"] for d in regime["dimensions"] if d["name"] == "Monetary"), "Unknown"
            ),
        }
        logger.info("Using current regime snapshot: %s", regime_entry["regime_composite"])

    except Exception as e:
        logger.warning("Could not build regime history (%s) — using 'Unknown'", e)
        regime_entry = {
            "regime_composite": "Unknown",
            "regime_volatility": "Unknown",
            "regime_cycle": "Unknown",
            "regime_risk": "Unknown",
            "regime_monetary": "Unknown",
        }

    # Map the same regime to every trading day in the range.
    history: dict[date, dict] = {}
    for d in pd.bdate_range(start, end):
        history[d.date()] = regime_entry

    return history


# ======================================================================
# Price data fetcher
# ======================================================================


def _fetch_price_data(
    tickers: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for each ticker via yfinance.

    Returns dict of ticker -> DataFrame (with columns Open, High, Low,
    Close, Volume and a DatetimeIndex).
    """
    import yfinance as yf

    logger.info("Fetching price data for %d tickers (%s to %s)...", len(tickers), start, end)

    # Use yfinance bulk download for efficiency.
    try:
        raw = yf.download(
            tickers,
            start=str(start),
            end=str(end),
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error("yfinance download failed: %s", e)
        return {}

    result: dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        # yfinance returns flat columns for a single ticker.
        ticker = tickers[0]
        df = raw.copy()
        if not df.empty:
            result[ticker] = df
    else:
        for ticker in tickers:
            try:
                df = raw[ticker].dropna(how="all")
                if not df.empty:
                    result[ticker] = df
            except (KeyError, TypeError):
                logger.debug("No data for %s", ticker)
                continue

    logger.info("Fetched data for %d / %d tickers", len(result), len(tickers))
    return result


# ======================================================================
# Main runner
# ======================================================================


def _run_backtest(
    strategies,
    tickers: list[str],
    start: date,
    end: date,
    regime_history: dict[date, dict],
) -> list[dict]:
    """Run each strategy through the backtest engine and collect trades.

    Returns a list of trade dicts ready for ``pd.DataFrame()``.
    """
    try:
        from .engine import BacktestEngine
    except ImportError:
        logger.error(
            "BacktestEngine not yet implemented. "
            "Create market_flows/backtest/engine.py to enable live backtesting."
        )
        return []

    price_data = _fetch_price_data(tickers, start, end)
    if not price_data:
        logger.error("No price data available — aborting backtest")
        return []

    engine = BacktestEngine(
        regime_history=regime_history,
    )

    all_trades: list[dict] = []

    for strategy in strategies:
        logger.info("Running %s on %d tickers...", strategy.name, len(price_data))
        try:
            trades = engine.run(strategy, price_data, start, end)
            for t in trades:
                # Convert Trade dataclass to dict
                trade_dict = {
                    "ticker": t.ticker,
                    "date": t.date,
                    "strategy": t.strategy,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "stop_price": t.stop_price,
                    "target_price": t.target_price,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "outcome": t.outcome,
                    "days_held": t.days_held,
                    "quality_score": t.quality_score,
                    "regime_composite": t.regime_composite,
                    "regime_volatility": t.regime_volatility,
                    "regime_cycle": t.regime_cycle,
                    "regime_risk": t.regime_risk,
                    "regime_monetary": t.regime_monetary,
                }
                all_trades.append(trade_dict)
            logger.info("  %s: %d trades generated", strategy.name, len(trades))
        except Exception as e:
            logger.error("  %s failed: %s", strategy.name, e)
            continue

    return all_trades


# ======================================================================
# CLI
# ======================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        prog="python -m market_flows.backtest.run_backtest",
        description="Run regime-aware strategy backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s                                 Run all strategies on ORB universe
  %(prog)s --strategy orb --tickers AAPL MSFT
  %(prog)s --strategy vwap --start 2025-06-01 --output html
  %(prog)s --regime-filter "Healthy Expansion" --output csv
  %(prog)s --output html --output-path reports/backtest.html
""",
    )

    ap.add_argument(
        "--strategy",
        choices=list(STRATEGY_MAP.keys()) + ["all"],
        default="all",
        help="Strategy to backtest (default: all)",
    )
    ap.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Ticker symbols to test (default: ORB universe ~60 instruments)",
    )
    ap.add_argument(
        "--start",
        type=str,
        default="2025-01-01",
        help="Start date YYYY-MM-DD (default: 2025-01-01)",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    ap.add_argument(
        "--output",
        choices=["console", "csv", "html"],
        default="console",
        help="Output format (default: console)",
    )
    ap.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="File path for csv/html output (default: data/backtest_<timestamp>)",
    )
    ap.add_argument(
        "--regime-filter",
        type=str,
        default=None,
        metavar="LABEL",
        help="Only include trades where regime_composite matches this label",
    )
    ap.add_argument(
        "--position-size",
        type=float,
        default=DEFAULT_POSITION_SIZE,
        help=f"Nominal position size for ROI calculation (default: {DEFAULT_POSITION_SIZE:,.0f})",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the backtest CLI runner."""
    args = _parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    # Parse dates
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    if start >= end:
        print(f"Error: start date ({start}) must be before end date ({end})")
        sys.exit(1)

    # Resolve tickers
    tickers = args.tickers if args.tickers else DEFAULT_TICKERS

    # Resolve strategies
    if args.strategy == "all":
        strategy_names = list(STRATEGY_MAP.keys())
    else:
        strategy_names = [args.strategy]

    # Print header
    print()
    print("+" + "-" * 58 + "+")
    print("|              MARKET-FLOWS BACKTEST RUNNER                |")
    print(f"|  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'):^56s}  |")
    print("+" + "-" * 58 + "+")
    print()
    print(f"  Strategies: {', '.join(strategy_names)}")
    print(f"  Tickers:    {len(tickers)} instruments")
    print(f"  Period:     {start} to {end}")
    print(f"  Position:   {args.position_size:,.0f}")
    if args.regime_filter:
        print(f"  Regime:     {args.regime_filter}")
    print()

    # Load strategies
    try:
        strategies = _load_strategies(strategy_names)
    except ImportError as e:
        print(f"Error loading strategies: {e}")
        print("Some strategy modules may not be implemented yet.")
        sys.exit(1)

    if not strategies:
        print("No valid strategies to run. Available: " + ", ".join(STRATEGY_MAP.keys()))
        sys.exit(1)

    # Build regime history
    print("  Building regime history...")
    regime_history = _build_regime_history(start, end)
    print(f"  Regime history: {len(regime_history)} trading days")
    print()

    # Run backtest
    print("  Running backtest...")
    trade_dicts = _run_backtest(strategies, tickers, start, end, regime_history)
    print(f"  Completed: {len(trade_dicts)} total trades")
    print()

    if not trade_dicts:
        print("  No trades generated. The engine may not be implemented yet,")
        print("  or no signals fired for the given parameters.")
        print()
        return

    # Build trades DataFrame
    trades_df = pd.DataFrame(trade_dicts)

    # Apply regime filter
    if args.regime_filter:
        before = len(trades_df)
        trades_df = trades_df[trades_df["regime_composite"] == args.regime_filter]
        print(f"  Regime filter '{args.regime_filter}': {before} -> {len(trades_df)} trades")
        print()
        if trades_df.empty:
            print("  No trades match the regime filter.")
            print()
            return

    # Generate report
    report = BacktestReport(trades_df, position_size=args.position_size)

    if args.output == "console":
        report.print_report()

    elif args.output == "csv":
        out_path = args.output_path or _default_output_path("csv")
        report.to_csv(out_path)
        print(f"  CSV files written to {Path(out_path).parent}/")
        print()

    elif args.output == "html":
        out_path = args.output_path or _default_output_path("html")
        report.to_html(out_path)
        print(f"  HTML report written to {out_path}")
        print()


def _default_output_path(ext: str) -> str:
    """Generate a default output path with timestamp."""
    from ..config import DATA_DIR

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = DATA_DIR / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"backtest_{ts}.{ext}")


if __name__ == "__main__":
    main()
