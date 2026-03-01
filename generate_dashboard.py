#!/usr/bin/env python3
"""CI entry point: fetch latest data and generate the static dashboard."""

import argparse
import sys
import warnings

import pandas as pd

from market_flows.cot import fetch_cot, update_cot_history
from market_flows.dashboard import render_dashboard
from market_flows.etf import fetch_etfs

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def main():
    ap = argparse.ArgumentParser(description="Generate Market Flows dashboard")
    ap.add_argument("--data-dir", default="data",
                    help="Directory containing parquet/JSON data files (default: data)")
    ap.add_argument("--output", default="docs/index.html",
                    help="Output HTML path (default: docs/index.html)")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="Skip fetching new data, render from cached files only")
    args = ap.parse_args()

    if not args.skip_fetch:
        print("━━━ Updating COT history ━━━\n")
        try:
            update_cot_history()
        except Exception as e:
            print(f"  Warning: COT update failed ({e}), using cached data")

        print("\n━━━ Fetching live COT data ━━━\n")
        try:
            cot_rows = fetch_cot()
        except Exception as e:
            print(f"  Error fetching COT data: {e}")
            cot_rows = []

        print("\n━━━ Fetching ETF data ━━━\n")
        try:
            etf_rows = fetch_etfs(update=True)
        except Exception as e:
            print(f"  Error fetching ETF data: {e}")
            etf_rows = []
    else:
        print("Skipping data fetch, rendering from cache...\n")
        try:
            cot_rows = fetch_cot()
        except Exception:
            cot_rows = []
        etf_rows = []

    print("\n━━━ Generating dashboard ━━━\n")
    render_dashboard(
        cot_rows=cot_rows,
        etf_rows=etf_rows,
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
