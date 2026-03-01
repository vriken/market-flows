"""CLI entry point for market-flow tracker."""

import argparse
import warnings
from datetime import datetime

import pandas as pd

from .cot import backfill_cot_history, fetch_cot, print_cot, update_cot_history
from .etf import fetch_etfs, print_etfs
from .sentiment import print_sentiment

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def main():
    ap = argparse.ArgumentParser(
        description="Weekly market-flow tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s                          Show all data
  %(prog)s --update                 Save AUM snapshot & show all
  %(prog)s --cot gold oil bitcoin   COT for specific contracts
  %(prog)s --etf SPY QQQ XLK       ETF data for specific tickers
  %(prog)s --etf AAPL MSFT         Works with any Yahoo Finance ticker
  %(prog)s --cot gold --etf GLD    Mix COT and ETF filters
  %(prog)s --backfill              Download 5 years of COT history
""",
    )
    ap.add_argument("--update", action="store_true",
                    help="Save current AUM snapshot for flow calculations")
    ap.add_argument("--cot-only", action="store_true",
                    help="Only show COT data (all contracts)")
    ap.add_argument("--etf-only", action="store_true",
                    help="Only show ETF data (all tickers)")
    ap.add_argument("--cot", nargs="+", metavar="NAME",
                    help="Show specific COT contracts (e.g. gold oil bitcoin)")
    ap.add_argument("--etf", nargs="+", metavar="TICKER",
                    help="Show specific ETFs (e.g. SPY QQQ XLK)")
    ap.add_argument("--backfill", action="store_true",
                    help="Download 5 years of CFTC COT history into parquet")
    args = ap.parse_args()

    # Handle backfill mode
    if args.backfill:
        print("\n━━━ COT History Backfill ━━━\n")
        backfill_cot_history()
        print()
        return

    # Determine what to show
    has_cot_filter = args.cot is not None
    has_etf_filter = args.etf is not None

    if has_cot_filter or has_etf_filter:
        show_cot = has_cot_filter
        show_etf = has_etf_filter
    else:
        show_cot = not args.etf_only
        show_etf = not args.cot_only

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║           WEEKLY MARKET FLOWS SUMMARY            ║")
    print(f"║           {datetime.now().strftime('%Y-%m-%d %H:%M'):^40} ║")
    print("╚══════════════════════════════════════════════════╝")

    if show_cot:
        print("\n━━━ COT Positioning ━━━\n")
        try:
            rows = fetch_cot(filter_contracts=args.cot)
            print_cot(rows)
        except Exception as e:
            print(f"  Error fetching COT data: {e}")

    etf_results = []
    if show_etf:
        print("\n━━━ ETF Flows ━━━\n")
        if not args.update and not has_etf_filter:
            print("  Tip: run with --update to save AUM data for flow estimates.\n")
        try:
            etf_results = fetch_etfs(
                filter_tickers=args.etf,
                update=args.update,
            )
            print_etfs(etf_results, filter_tickers=args.etf)
        except Exception as e:
            print(f"  Error fetching ETF data: {e}")

        if not any(r.get("flow_m") for r in etf_results):
            print("\n  Note: flow estimates need 2+ snapshots.")
            print("  Run with --update weekly to build history.")
            print("  For instant data: https://www.etfdb.com/etf/flow-tool/")

    if show_cot or (not has_cot_filter and not has_etf_filter):
        print("\n━━━ Market Leverage & Sentiment ━━━\n")
        try:
            print_sentiment()
        except Exception as e:
            print(f"  Error fetching sentiment data: {e}")

    print()
