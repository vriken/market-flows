#!/usr/bin/env python3
"""CI entry point: fetch latest data and generate the static dashboard."""

import argparse
import warnings

import pandas as pd

from market_flows.breadth import fetch_market_breadth
from market_flows.cot import fetch_cot, update_cot_history
from market_flows.dashboard import render_dashboard
from market_flows.etf import build_flow_history, fetch_etfs
from market_flows.external import fetch_credit_spreads, fetch_fed_liquidity, fetch_margin_debt
from market_flows.regime import classify_regime
from market_flows.sentiment import (
    fetch_leverage_ratios,
    fetch_market_ratios,
    fetch_orb_conditions,
    fetch_ratio_time_series,
    fetch_sector_rotation,
    fetch_vix_term_structure,
    fetch_yield_curve,
    fetch_yield_curve_history,
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
    try:
        sentiment_data["yield_curve"] = fetch_yield_curve()
        print(f"  Yield curve: {'OK' if sentiment_data['yield_curve'] else 'no data'}")
    except Exception as e:
        print(f"  Yield curve failed: {e}")
    try:
        sentiment_data["yield_history"] = fetch_yield_curve_history()
        print(f"  Yield history: {'OK' if sentiment_data['yield_history'] else 'no data'}")
    except Exception as e:
        print(f"  Yield history failed: {e}")

    print("\n━━━ ORB strategy conditions ━━━\n")
    orb_conditions = None
    try:
        vix_price = sentiment_data.get("vix", {}).get("vix") if sentiment_data.get("vix") else None
        orb_conditions = fetch_orb_conditions(vix_price=vix_price)
        print(f"  ORB regime: {orb_conditions.get('overall', 'unknown')} (VIX: {orb_conditions.get('vix', '?')})")
    except Exception as e:
        print(f"  ORB conditions failed: {e}")

    print("\n━━━ Fetching external data sources ━━━\n")
    external_data = {}
    try:
        external_data["margin_debt"] = fetch_margin_debt()
        print(f"  Margin debt: {'OK' if external_data['margin_debt'] else 'no data'}")
    except Exception as e:
        print(f"  Margin debt failed: {e}")
    # These require API keys — enable when keys are configured:
    # external_data["fred_flows"] = fetch_fred_fund_flows()   # needs FRED_API_KEY
    # external_data["aaii"] = fetch_aaii_sentiment()           # needs NASDAQ_DATA_LINK_API_KEY
    # external_data["putcall"] = fetch_putcall_ratio()         # needs FRED_API_KEY (EQUITYPC series)

    print("\n━━━ Fetching macro & liquidity data ━━━\n")
    credit_data = None
    liquidity_data = None
    breadth_data = None
    try:
        credit_data = fetch_credit_spreads()
        print(f"  Credit spreads: {'OK' if credit_data else 'no data (need FRED_API_KEY)'}")
    except Exception as e:
        print(f"  Credit spreads failed: {e}")
    try:
        liquidity_data = fetch_fed_liquidity()
        print(f"  Fed liquidity: {'OK' if liquidity_data else 'no data (need FRED_API_KEY)'}")
    except Exception as e:
        print(f"  Fed liquidity failed: {e}")
    try:
        breadth_data = fetch_market_breadth()
        status = f"OK — {breadth_data.get('total_tickers', 0)} tickers" if breadth_data else "no data"
        print(f"  Market breadth: {status}")
    except Exception as e:
        print(f"  Market breadth failed: {e}")

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

    print("\n━━━ Classifying regime ━━━\n")
    try:
        regime = classify_regime(sentiment_data, cot_rows,
                                credit_data=credit_data,
                                liquidity_data=liquidity_data,
                                breadth_data=breadth_data)
        print(f"  Regime: {regime['composite_label']} (confidence: {regime['confidence']:.0%})")
        for dim in regime["dimensions"]:
            print(f"    {dim['name']}: {dim['state']}")
    except Exception as e:
        print(f"  Regime classification failed: {e}")
        regime = None

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
        external_data=external_data,
        orb_conditions=orb_conditions,
        regime=regime,
        credit_data=credit_data,
        liquidity_data=liquidity_data,
        breadth_data=breadth_data,
    )


if __name__ == "__main__":
    main()
