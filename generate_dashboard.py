#!/usr/bin/env python3
"""CI entry point: fetch latest data and generate the static dashboard."""

import argparse
import time
import warnings
from datetime import UTC, datetime

import pandas as pd

from market_flows.breadth import fetch_market_breadth
from market_flows.cot import fetch_cot, update_cot_history
from market_flows.dashboard import render_dashboard
from market_flows.etf import build_flow_history, fetch_etfs
from market_flows.external import fetch_credit_spreads, fetch_fear_greed, fetch_fed_liquidity, fetch_margin_debt
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


def _fetch_source(name, fn, *args, **kwargs):
    """Fetch a data source, returning (result, status) tuple."""
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        if result is None:
            print(f"  {name}: no data ({elapsed:.1f}s)")
            return None, "missing"
        stale = isinstance(result, dict) and result.get("_stale")
        status = "stale" if stale else "ok"
        print(f"  {name}: {'STALE' if stale else 'OK'} ({elapsed:.1f}s)")
        return result, status
    except Exception as e:
        elapsed = time.time() - start
        print(f"  {name}: FAILED ({e}) ({elapsed:.1f}s)")
        return None, "failed"


def main():
    ap = argparse.ArgumentParser(description="Generate Market Flows dashboard")
    ap.add_argument("--data-dir", default="data",
                    help="Directory containing parquet/JSON data files (default: data)")
    ap.add_argument("--output", default="docs/index.html",
                    help="Output HTML path (default: docs/index.html)")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="Skip fetching new data, render from cached files only")
    args = ap.parse_args()

    freshness = {}

    # --- COT data ---
    print("━━━ COT data ━━━\n")
    if not args.skip_fetch:
        _, freshness["cot_update"] = _fetch_source("COT history update", update_cot_history)
    cot_rows, freshness["cot"] = _fetch_source("COT live", fetch_cot)
    if cot_rows is None:
        cot_rows = []

    # --- ETF data ---
    print("\n━━━ ETF data ━━━\n")
    if not args.skip_fetch:
        etf_rows, freshness["etf"] = _fetch_source("ETF snapshot", fetch_etfs, update=True)
    else:
        etf_rows = []
        freshness["etf"] = "skipped"
    if etf_rows is None:
        etf_rows = []

    # --- Sentiment data ---
    print("\n━━━ Sentiment data ━━━\n")
    sentiment_data = {}
    price_data = None

    sources = [
        ("vix", "VIX term structure", fetch_vix_term_structure),
        ("leverage", "Leverage ratios", fetch_leverage_ratios),
        ("yield_curve", "Yield curve", fetch_yield_curve),
        ("yield_history", "Yield history", fetch_yield_curve_history),
    ]
    for key, name, fn in sources:
        result, freshness[key] = _fetch_source(name, fn)
        if result is not None:
            sentiment_data[key] = result

    # Market ratios returns a tuple (ratios, price_data)
    try:
        ratios_result = fetch_market_ratios(period="1y", include_history=True)
        sentiment_data["ratios"], price_data = ratios_result
        freshness["ratios"] = "ok"
        print("  Market ratios: OK")
    except Exception as e:
        freshness["ratios"] = "failed"
        print(f"  Market ratios: FAILED ({e})")

    # --- ORB conditions ---
    print("\n━━━ ORB conditions ━━━\n")
    vix_price = sentiment_data.get("vix", {}).get("vix") if sentiment_data.get("vix") else None
    orb_conditions, freshness["orb"] = _fetch_source("ORB conditions", fetch_orb_conditions, vix_price=vix_price)

    # --- External data ---
    print("\n━━━ External data ━━━\n")
    external_data = {}

    ext_sources = [
        ("margin_debt", "Margin debt", fetch_margin_debt),
        ("fear_greed", "Fear & Greed", fetch_fear_greed),
    ]
    for key, name, fn in ext_sources:
        result, freshness[key] = _fetch_source(name, fn)
        if result is not None:
            external_data[key] = result

    fear_greed_data = external_data.get("fear_greed")

    # --- Macro & liquidity ---
    print("\n━━━ Macro & liquidity ━━━\n")
    credit_data, freshness["credit"] = _fetch_source("Credit spreads", fetch_credit_spreads)
    liquidity_data, freshness["liquidity"] = _fetch_source("Fed liquidity", fetch_fed_liquidity)
    breadth_data, freshness["breadth"] = _fetch_source("Market breadth", fetch_market_breadth)

    # --- Historical visualizations ---
    print("\n━━━ Historical visualizations ━━━\n")
    ratio_series, _ = _fetch_source("Ratio time series", fetch_ratio_time_series, price_data=price_data)
    rotation_data, _ = _fetch_source("Sector rotation", fetch_sector_rotation, weeks=12)
    flow_data, _ = _fetch_source("ETF flow history", build_flow_history)

    # --- Regime classification ---
    print("\n━━━ Regime ━━━\n")
    regime = None
    try:
        regime = classify_regime(sentiment_data, cot_rows,
                                credit_data=credit_data,
                                liquidity_data=liquidity_data,
                                breadth_data=breadth_data)
        print(f"  Regime: {regime['composite_label']} (confidence: {regime['confidence']:.0%})")
        freshness["regime"] = "ok"
    except Exception as e:
        print(f"  Regime: FAILED ({e})")
        freshness["regime"] = "failed"

    # --- Freshness summary ---
    failed = [k for k, v in freshness.items() if v == "failed"]
    stale = [k for k, v in freshness.items() if v == "stale"]
    ok = [k for k, v in freshness.items() if v == "ok"]
    print("\n━━━ Data freshness ━━━\n")
    print(f"  OK: {len(ok)}  Stale: {len(stale)}  Failed: {len(failed)}")
    if failed:
        print(f"  Failed sources: {', '.join(failed)}")
    if stale:
        print(f"  Stale sources: {', '.join(stale)}")

    # Build freshness metadata for dashboard
    freshness_meta = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "sources": freshness,
        "ok_count": len(ok),
        "stale_count": len(stale),
        "failed_count": len(failed),
    }

    # --- Render ---
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
        fear_greed_data=fear_greed_data,
        freshness=freshness_meta,
    )


if __name__ == "__main__":
    main()
