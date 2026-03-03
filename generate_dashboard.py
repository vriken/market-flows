#!/usr/bin/env python3
"""CI entry point: fetch latest data and generate the static dashboard."""

import argparse
import sys
import warnings

import pandas as pd

from market_flows.cot import fetch_cot, update_cot_history
from market_flows.dashboard import render_dashboard
from market_flows.etf import build_flow_history, fetch_etfs
from market_flows.sentiment import (
    fetch_leverage_ratios,
    fetch_market_ratios,
    fetch_ratio_time_series,
    fetch_sector_rotation,
    fetch_vix_term_structure,
)

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

    print("\n━━━ Fetching sentiment data ━━━\n")
    sentiment_data = {}
    price_data = None
    try:
        sentiment_data["vix"] = fetch_vix_term_structure()
        print("  VIX term structure: OK")
    except Exception as e:
        print(f"  VIX term structure failed: {e}")
    try:
        sentiment_data["leverage"] = fetch_leverage_ratios()
        print("  Leverage ratios: OK")
    except Exception as e:
        print(f"  Leverage ratios failed: {e}")
    try:
        ratios_result = fetch_market_ratios(period="1y", include_history=True)
        sentiment_data["ratios"], price_data = ratios_result
        print("  Market ratios: OK")
    except Exception as e:
        print(f"  Market ratios failed: {e}")

    print("\n━━━ Building historical visualizations ━━━\n")
    ratio_series = None
    rotation_data = None
    flow_data = None

    try:
        ratio_series = fetch_ratio_time_series(price_data=price_data)
        print(f"  Ratio time series: {len(ratio_series)} series")
    except Exception as e:
        print(f"  Ratio time series failed: {e}")

    try:
        rotation_data = fetch_sector_rotation(weeks=12)
        print(f"  Sector rotation: {'OK' if rotation_data else 'no data'}")
    except Exception as e:
        print(f"  Sector rotation failed: {e}")

    try:
        flow_data = build_flow_history()
        print(f"  ETF flow history: {'OK' if flow_data.get('has_data') else flow_data.get('message', 'no data')}")
    except Exception as e:
        print(f"  ETF flow history failed: {e}")

    print("\n━━━ Generating dashboard ━━━\n")
    render_dashboard(
        cot_rows=cot_rows,
        etf_rows=etf_rows,
        sentiment_data=sentiment_data,
        data_dir=args.data_dir,
        output_path=args.output,
        ratio_series=ratio_series,
        rotation_data=rotation_data,
        flow_data=flow_data,
    )


if __name__ == "__main__":
    main()
