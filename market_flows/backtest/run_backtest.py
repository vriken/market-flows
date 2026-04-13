"""CLI entry point for running regime-aware backtests.

Usage
-----
    python -m market_flows.backtest.run_backtest [options]

Options
-------
    --strategy {orb,momentum,pdhl,vwap,fvg,all}   Strategy to test (default: all)
    --tickers TICKER [TICKER ...]                  Ticker symbols (default: ORB universe)
    --start YYYY-MM-DD                             Start date (default: 2025-01-01)
    --end YYYY-MM-DD                               End date (default: today)
    --output {console,csv,html}                    Report format (default: console)
    --output-path PATH                             File path for csv/html output
    --regime-filter LABEL                          Only include trades in this regime
    --position-size N                              Nominal position size (default: 500)
    --verbose / -v                                 Enable debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================================
# Default position sizing (500 SEK, 2% KO = 50x)
# ======================================================================
DEFAULT_POSITION_SIZE = 500

# ======================================================================
# ORB universe (~71 instruments)
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

# Strategy name -> class name mapping
STRATEGY_MAP = {
    "orb": "ORBStrategy",
    "momentum": "MomentumStrategy",
    "pdhl": "PDHLStrategy",
}

RETIRED_STRATEGIES = {
    "fvg": "Retired: 8138 unfiltered trades with no quality gating, enters at open on daily signal",
}

PAUSED_STRATEGIES = {
    "vwap": "Paused: no trend/regime context, fades moves blindly — needs filters before re-enabling",
}


# ======================================================================
# Strategy loader
# ======================================================================


def _load_strategies(names: list[str], *, include_paused: bool = False):
    """Lazily import and instantiate requested strategy classes.

    Retired strategies are always skipped with a message.
    Paused strategies are skipped unless *include_paused* is True or
    the user explicitly requested the strategy by name.
    """
    from .strategies import (
        MomentumStrategy,
        ORBStrategy,
        PDHLStrategy,
        VWAPReversionStrategy,
    )

    class_map = {
        "ORBStrategy": ORBStrategy,
        "VWAPReversionStrategy": VWAPReversionStrategy,
        "MomentumStrategy": MomentumStrategy,
        "PDHLStrategy": PDHLStrategy,
    }

    # When include_paused, merge paused strategies into the active map
    active_map = dict(STRATEGY_MAP)
    if include_paused:
        for key in PAUSED_STRATEGIES:
            if key == "vwap":
                active_map["vwap"] = "VWAPReversionStrategy"

    strategies = []
    for name in names:
        # Check retired
        if name in RETIRED_STRATEGIES:
            print(f"  Skipping {name}: {RETIRED_STRATEGIES[name]}")
            continue
        # Check paused (only skip if not explicitly requested)
        if name in PAUSED_STRATEGIES and not include_paused:
            print(f"  ⚠ {name}: {PAUSED_STRATEGIES[name]}")
            continue

        cls_name = active_map.get(name)
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


def _build_regime_history(start: date, end: date) -> pd.DataFrame:
    """Build historical regime classifications for each trading day."""
    try:
        from .regime_history import build_regime_history
        regime_df = build_regime_history(str(start), str(end))
        logger.info("Built regime history: %d days, regimes: %s",
                     len(regime_df),
                     regime_df["composite_label"].value_counts().to_dict())
        return regime_df
    except Exception as e:
        logger.warning("Could not build regime history (%s) — returning empty", e)
        return pd.DataFrame()


# ======================================================================
# Price data fetcher — uses data.py with caching
# ======================================================================


def _fetch_price_data(
    tickers: list[str],
    start: date,
    end: date,
    need_intraday: bool = False,
) -> dict[str, dict]:
    """Fetch price data for all tickers, with optional intraday.

    Returns dict of ticker -> {"daily": DataFrame, "intraday": DataFrame or None}
    """
    from .data import fetch_daily, fetch_intraday

    result: dict[str, dict] = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        try:
            daily = fetch_daily(ticker, str(start), str(end))
            if daily.empty:
                continue

            intraday = None
            if need_intraday:
                try:
                    # yfinance only serves ~60 days of intraday data; clamp
                    # the start so we at least get recent bars.  Older dates
                    # will be served from the per-date parquet cache if a
                    # previous run already downloaded them.
                    from .data import _YF_INTRADAY_MAX_DAYS
                    intraday_start = max(start, date.today() - timedelta(days=_YF_INTRADAY_MAX_DAYS))
                    intraday = fetch_intraday(ticker, str(intraday_start), str(end))
                    if intraday is not None and intraday.empty:
                        intraday = None
                except Exception as e:
                    logger.debug("No intraday for %s: %s", ticker, e)

            result[ticker] = {"daily": daily, "intraday": intraday}

            if i % 10 == 0 or i == total:
                logger.info("  Fetched %d/%d tickers", i, total)

        except Exception as e:
            logger.debug("No data for %s: %s", ticker, e)

    return result


# ======================================================================
# Main runner
# ======================================================================


def _run_backtest(
    strategies,
    tickers: list[str],
    start: date,
    end: date,
    regime_history: pd.DataFrame,
    position_size: float = DEFAULT_POSITION_SIZE,
    ko_buffer: float = 0.02,
    slippage_pct: float = 0.0,
    risk_normalize: bool = False,
    base_risk_pct: float = 0.01,
    skip_wide_stops: bool = True,
) -> pd.DataFrame:
    """Run each strategy through the backtest engine and collect trades."""
    from .engine import BacktestEngine

    # Check if any strategy needs intraday data
    need_intraday = any(s.requires_intraday for s in strategies)
    daily_only = [s for s in strategies if not s.requires_intraday]
    intraday_strats = [s for s in strategies if s.requires_intraday]

    if daily_only and intraday_strats:
        logger.info("Strategies: %d daily, %d intraday",
                     len(daily_only), len(intraday_strats))

    # Compute warmup: fetch extra history so SMA-based strategies have
    # enough data from the first trading day of the user's range.
    max_warmup = max((s.warmup_days for s in strategies), default=0)
    fetch_start = start - timedelta(days=max_warmup) if max_warmup > 0 else start

    # Fetch price data
    print(f"  Fetching price data for {len(tickers)} tickers"
          f"{' (+ intraday)' if need_intraday else ''}...")
    price_data = _fetch_price_data(tickers, fetch_start, end, need_intraday=need_intraday)
    print(f"  Got data for {len(price_data)} / {len(tickers)} tickers")

    if not price_data:
        logger.error("No price data available — aborting backtest")
        return pd.DataFrame()

    all_dfs: list[pd.DataFrame] = []

    for strategy in strategies:
        print(f"  Running {strategy.name}...")

        # Build the right data dict for this strategy
        if strategy.requires_intraday:
            # Intraday strategies get {"daily": DF, "intraday": DF} bundles
            # Skip tickers that don't have intraday data
            strat_data = {}
            for t, bundle in price_data.items():
                if bundle.get("intraday") is not None:
                    strat_data[t] = bundle
                else:
                    # Fall back to daily-only (strategy may handle it)
                    strat_data[t] = bundle
        else:
            # Daily strategies just get the daily DataFrame directly
            strat_data = {t: bundle["daily"] for t, bundle in price_data.items()}

        try:
            engine = BacktestEngine(strategy, regime_history=regime_history)
            trades = engine.run(strat_data, str(start), str(end),
                              position_size=position_size,
                              ko_buffer=ko_buffer,
                              slippage_pct=slippage_pct,
                              risk_normalize=risk_normalize,
                              base_risk_pct=base_risk_pct,
                              skip_wide_stops=skip_wide_stops)
            print(f"    {len(trades)} trades")
            if trades:
                all_dfs.append(engine.results_df())
        except Exception as e:
            logger.error("  %s failed: %s", strategy.name, e, exc_info=True)
            print(f"    FAILED: {e}")
            continue

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


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
  %(prog)s --strategy orb --start 2025-06-01 --output html
  %(prog)s --regime-filter "Healthy Expansion" --output csv
  %(prog)s --output html --output-path reports/backtest.html
""",
    )

    all_strategy_names = (
        list(STRATEGY_MAP.keys())
        + list(PAUSED_STRATEGIES.keys())
        + list(RETIRED_STRATEGIES.keys())
    )
    ap.add_argument(
        "--strategy",
        choices=all_strategy_names + ["all", "daily", "intraday"],
        default="all",
        help="Strategy to backtest. 'daily' = momentum, 'intraday' = orb+pdhl (default: all)",
    )
    ap.add_argument(
        "--include-paused",
        action="store_true",
        help="Include paused strategies when running --strategy all/intraday",
    )
    ap.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Ticker symbols to test (default: ORB universe ~71 instruments)",
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
        help=f"Nominal position size (default: {DEFAULT_POSITION_SIZE})",
    )
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Only backtest new dates since last run (reads existing results_trades.csv)",
    )
    ap.add_argument(
        "--confluence",
        action="store_true",
        help="Run confluence analysis: compare performance when multiple strategies agree",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    ap.add_argument("--ko-buffer", type=float, default=0.02, help="KO buffer pct (default: 0.02 = 2%%)")
    ap.add_argument("--slippage-bps", type=int, default=10, help="Slippage in basis points (default: 10)")
    ap.add_argument("--risk-normalize", action="store_true", help="Normalize position sizing by stop distance")
    ap.add_argument("--base-risk", type=float, default=0.01, help="Base risk pct for risk normalization (default: 0.01)")
    ap.add_argument("--skip-wide-stops", action="store_true", default=True, help="Skip trades where stop > KO distance (default: True)")
    ap.add_argument("--no-skip-wide-stops", dest="skip_wide_stops", action="store_false")
    ap.add_argument("--analyze", action="store_true", help="Run trade analysis after backtest")

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the backtest CLI runner."""
    args = _parse_args(argv)

    # Configure logging — INFO by default so progress is visible in CI
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    # Parse dates
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    # Resolve tickers
    tickers = args.tickers if args.tickers else DEFAULT_TICKERS

    # Incremental mode: find last trade date from existing results
    existing_trades_df = pd.DataFrame()
    if args.incremental:
        trades_csv = Path(
            args.output_path or "data/backtest/results"
        ).parent / (Path(args.output_path or "data/backtest/results").stem + "_trades.csv")
        if trades_csv.exists():
            existing_trades_df = pd.read_csv(trades_csv, parse_dates=["date"])
            existing_trades_df["date"] = existing_trades_df["date"].dt.date
            last_date = str(existing_trades_df["date"].max())
            # Start from the day after the last trade
            incremental_start = date.fromisoformat(last_date) + timedelta(days=1)
            if incremental_start >= end:
                print(f"  Already up to date (last trade: {last_date})")
                sys.exit(0)
            print(f"  Incremental: last trade {last_date}, running from {incremental_start}")
            start = incremental_start
        else:
            print("  No existing trades found — running full backtest")

    if start >= end:
        print(f"Error: start date ({start}) must be before end date ({end})")
        sys.exit(1)

    # Resolve strategies
    include_paused = args.include_paused
    if args.strategy == "all":
        strategy_names = list(STRATEGY_MAP.keys())
        if include_paused:
            strategy_names += list(PAUSED_STRATEGIES.keys())
    elif args.strategy == "daily":
        strategy_names = ["momentum"]
    elif args.strategy == "intraday":
        strategy_names = ["orb", "pdhl"]
        if include_paused:
            strategy_names += [k for k in PAUSED_STRATEGIES if k == "vwap"]
    elif args.strategy in PAUSED_STRATEGIES:
        # Explicitly requested paused strategy — allow it with a warning
        print(f"  ⚠ Warning: {args.strategy} is paused — {PAUSED_STRATEGIES[args.strategy]}")
        strategy_names = [args.strategy]
        include_paused = True  # force load
    else:
        strategy_names = [args.strategy]

    # Print header
    print()
    print("+" + "-" * 58 + "+")
    print("|              MARKET-FLOWS BACKTEST RUNNER                |")
    print(f"|  {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC'):^56s}  |")
    print("+" + "-" * 58 + "+")
    print()
    print(f"  Strategies: {', '.join(strategy_names)}")
    print(f"  Tickers:    {len(tickers)} instruments")
    print(f"  Period:     {start} to {end}")
    ko_pct = args.ko_buffer * 100
    leverage = int(1 / args.ko_buffer)
    print(f"  Position:   {args.position_size:,.0f} SEK ({ko_pct:.0f}% KO = {leverage}x leverage)")
    if args.slippage_bps:
        print(f"  Slippage:   {args.slippage_bps} bps per side")
    if args.regime_filter:
        print(f"  Regime:     {args.regime_filter}")
    print()

    # Load strategies
    try:
        strategies = _load_strategies(strategy_names, include_paused=include_paused)
    except ImportError as e:
        print(f"Error loading strategies: {e}")
        sys.exit(1)

    if not strategies:
        print("No valid strategies to run.")
        print("  Active:  " + ", ".join(STRATEGY_MAP.keys()))
        if PAUSED_STRATEGIES:
            print("  Paused:  " + ", ".join(PAUSED_STRATEGIES.keys()) + "  (use --include-paused)")
        if RETIRED_STRATEGIES:
            print("  Retired: " + ", ".join(RETIRED_STRATEGIES.keys()))
        sys.exit(1)

    # Build regime history
    print("  Building regime history...")
    regime_history = _build_regime_history(start, end)
    print(f"  Regime history: {len(regime_history)} trading days")
    print()

    # Run backtest
    trades_df = _run_backtest(strategies, tickers, start, end, regime_history,
                              position_size=args.position_size,
                              ko_buffer=args.ko_buffer,
                              slippage_pct=args.slippage_bps / 10000,
                              risk_normalize=args.risk_normalize,
                              base_risk_pct=args.base_risk,
                              skip_wide_stops=args.skip_wide_stops)
    print()
    print(f"  Total trades: {len(trades_df)}")
    print()

    # Merge with existing trades in incremental mode
    if args.incremental and not existing_trades_df.empty:
        if not trades_df.empty:
            trades_df = pd.concat([existing_trades_df, trades_df], ignore_index=True)
            trades_df = trades_df.drop_duplicates(
                subset=["ticker", "date", "strategy", "entry_time"], keep="last"
            )
            trades_df = trades_df.sort_values("date").reset_index(drop=True)
            print(f"  Merged: {len(existing_trades_df)} existing + {len(trades_df) - len(existing_trades_df)} new = {len(trades_df)} total")
        else:
            trades_df = existing_trades_df
            print("  No new trades — keeping existing results")

    if trades_df.empty:
        print("  No trades generated.")
        print()
        return

    # Apply regime filter
    if args.regime_filter:
        before = len(trades_df)
        trades_df = trades_df[trades_df["regime_composite"] == args.regime_filter]
        print(f"  Regime filter '{args.regime_filter}': {before} -> {len(trades_df)} trades")
        print()
        if trades_df.empty:
            print("  No trades match the regime filter.")
            return

    # Confluence analysis (runs on all-strategy results before individual report)
    if args.confluence:
        from .confluence import analyse_confluence, print_confluence_report
        confluence_results = analyse_confluence(trades_df)
        print_confluence_report(confluence_results)

    # Generate report
    report = BacktestReport(trades_df, position_size=args.position_size)

    if args.output == "console":
        report.print_report()

    elif args.output == "csv":
        out_path = args.output_path or _default_output_path("csv")
        report.to_csv(out_path)
        print(f"  CSV files written to {Path(out_path).parent}/")

    elif args.output == "html":
        out_path = args.output_path or _default_output_path("html")
        report.to_html(out_path)
        print(f"  HTML report written to {out_path}")

    # Optional trade analysis
    if args.analyze and not trades_df.empty:
        from .analysis import print_analysis
        print_analysis(trades_df, ko_buffer=args.ko_buffer)


def _default_output_path(ext: str) -> str:
    """Generate a default output path with timestamp."""
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"backtest_{ts}.{ext}")


# Import here to avoid circular — report module is simple
from .report import BacktestReport  # noqa: E402

if __name__ == "__main__":
    main()
