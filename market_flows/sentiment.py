"""Market leverage, sentiment indicators, and ratio calculations."""

import pandas as pd
import yfinance as yf

from .config import LEVERAGE_PAIRS, MARKET_RATIOS, VIX_TICKERS, YIELD_CURVE_TICKERS

SECTOR_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLB", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC"]


def fetch_yield_curve():
    """Fetch current Treasury yields and compute key spreads.

    Returns dict with yields, spreads (2s10s, 3m10y), and inversion signal.
    Handles ^IRX scaling quirk (divide by 10 if > 20).
    """
    try:
        maturity_order = ["3m", "2y", "5y", "10y", "30y"]
        yields = {}

        for label, ticker in YIELD_CURVE_TICKERS.items():
            try:
                tkr = yf.Ticker(ticker)
                price = tkr.info.get("regularMarketPrice") or tkr.info.get("previousClose")
                if price is not None:
                    # ^IRX reports in basis-point-like format; normalize
                    if ticker == "^IRX" and price > 20:
                        price = price / 10
                    yields[label] = round(price, 3)
            except Exception:
                continue

        if "10y" not in yields:
            return None

        spreads = {}
        if "2y" in yields:
            spreads["2s10s"] = round(yields["10y"] - yields["2y"], 3)
        if "3m" in yields:
            spreads["3m10y"] = round(yields["10y"] - yields["3m"], 3)

        # Determine signal
        inverted = any(v < 0 for v in spreads.values())
        if inverted:
            signal = "Inverted — recession risk elevated"
        elif all(v > 0.5 for v in spreads.values()):
            signal = "Normal — healthy steepness"
        else:
            signal = "Flat — late-cycle or transition"

        return {
            "yields": {k: yields[k] for k in maturity_order if k in yields},
            "spreads": spreads,
            "signal": signal,
        }
    except Exception:
        return None


def fetch_yield_curve_history(period="1y"):
    """Fetch historical yield data and compute daily spread series.

    Returns dict with dates, spread_2s10s, spread_3m10y, and individual yields.
    """
    try:
        histories = {}
        for label, ticker in YIELD_CURVE_TICKERS.items():
            try:
                tkr = yf.Ticker(ticker)
                hist = tkr.history(period=period)
                if not hist.empty:
                    series = hist["Close"].copy()
                    # Normalize ^IRX
                    if ticker == "^IRX":
                        series = series.where(series <= 20, series / 10)
                    histories[label] = series
            except Exception:
                continue

        if "10y" not in histories:
            return None

        # Align all series on shared dates
        combined = pd.DataFrame(histories).dropna()
        if combined.empty:
            return None

        dates = [d.strftime("%Y-%m-%d") for d in combined.index]
        result = {
            "dates": dates,
            "yields": {col: combined[col].tolist() for col in combined.columns},
        }

        if "2y" in combined.columns:
            result["spread_2s10s"] = (combined["10y"] - combined["2y"]).round(3).tolist()
        if "3m" in combined.columns:
            result["spread_3m10y"] = (combined["10y"] - combined["3m"]).round(3).tolist()

        return result
    except Exception:
        return None


def fetch_vix_term_structure():
    """Fetch VIX and VIX3M to compute term structure (contango/backwardation)."""
    try:
        vix = yf.Ticker(VIX_TICKERS["vix"])
        vix3m = yf.Ticker(VIX_TICKERS["vix3m"])

        vix_price = vix.info.get("regularMarketPrice") or vix.info.get("previousClose")
        vix3m_price = vix3m.info.get("regularMarketPrice") or vix3m.info.get("previousClose")

        if not vix_price or not vix3m_price:
            return None

        spread = vix3m_price - vix_price
        ratio = vix_price / vix3m_price

        if ratio > 1.0:
            structure = "Backwardation"
            signal = "Fear — near-term vol elevated"
        elif ratio < 0.85:
            structure = "Steep contango"
            signal = "Complacency — potential snap-back risk"
        else:
            structure = "Contango"
            signal = "Normal"

        return {
            "vix": vix_price,
            "vix3m": vix3m_price,
            "spread": spread,
            "ratio": ratio,
            "structure": structure,
            "signal": signal,
        }
    except Exception:
        return None


def fetch_leverage_ratios():
    """Fetch AUM for leveraged bull/bear ETF pairs and compute ratios."""
    results = []
    for bull_tick, bear_tick, label in LEVERAGE_PAIRS:
        try:
            bull = yf.Ticker(bull_tick)
            bear = yf.Ticker(bear_tick)

            bull_aum = bull.info.get("totalAssets")
            bear_aum = bear.info.get("totalAssets")

            if not bull_aum or not bear_aum:
                continue

            ratio = bull_aum / bear_aum
            bull_b = bull_aum / 1e9
            bear_b = bear_aum / 1e9

            if ratio > 8:
                signal = "Extreme bull crowding"
            elif ratio > 5:
                signal = "Heavy bull lean"
            elif ratio > 3:
                signal = "Moderate bull lean"
            elif ratio > 1:
                signal = "Slight bull lean"
            else:
                signal = "Bear lean — unusual"

            results.append({
                "label": label,
                "bull_ticker": bull_tick,
                "bear_ticker": bear_tick,
                "bull_aum_b": bull_b,
                "bear_aum_b": bear_b,
                "ratio": ratio,
                "signal": signal,
            })
        except Exception:
            continue

    return results


def fetch_market_ratios(period="1y", include_history=False):
    """Fetch price ratios for key market pairs.

    When include_history=True, returns (results, prices) where prices contains
    full history DataFrames for reuse by fetch_ratio_time_series().
    """
    results = []
    # Batch-fetch all unique tickers
    all_tickers = set()
    for num, den, _, _ in MARKET_RATIOS:
        all_tickers.add(num)
        all_tickers.add(den)

    prices = {}
    for tick in all_tickers:
        try:
            tkr = yf.Ticker(tick)
            hist = tkr.history(period=period)
            if len(hist) >= 2:
                prices[tick] = {
                    "current": hist["Close"].iloc[-1],
                    "prev_week": hist["Close"].iloc[-5] if len(hist) >= 5 else hist["Close"].iloc[0],
                    "prev_month": hist["Close"].iloc[-22] if len(hist) >= 22 else hist["Close"].iloc[0],
                    "hist": hist,
                }
        except Exception:
            continue

    for num, den, label, interpretation in MARKET_RATIOS:
        if num not in prices or den not in prices:
            continue

        current_ratio = prices[num]["current"] / prices[den]["current"]
        prev_week_ratio = prices[num]["prev_week"] / prices[den]["prev_week"]
        prev_month_ratio = prices[num]["prev_month"] / prices[den]["prev_month"]

        wk_chg = (current_ratio / prev_week_ratio - 1) * 100
        mo_chg = (current_ratio / prev_month_ratio - 1) * 100

        results.append({
            "label": label,
            "numerator": num,
            "denominator": den,
            "ratio": current_ratio,
            "wk_change_pct": wk_chg,
            "mo_change_pct": mo_chg,
            "interpretation": interpretation,
        })

    if include_history:
        return results, prices
    return results


def fetch_ratio_time_series(price_data=None):
    """Compute daily ratio time series for each MARKET_RATIOS pair.

    Reuses price_data from fetch_market_ratios(include_history=True) to avoid
    additional API calls.
    """
    if price_data is None:
        return []

    series = []
    for num, den, label, interpretation in MARKET_RATIOS:
        if num not in price_data or den not in price_data:
            continue
        num_hist = price_data[num]["hist"]["Close"]
        den_hist = price_data[den]["hist"]["Close"]
        # Align on shared dates
        aligned = pd.concat([num_hist, den_hist], axis=1, keys=["num", "den"]).dropna()
        if aligned.empty:
            continue
        ratio = aligned["num"] / aligned["den"]
        series.append({
            "label": label,
            "interpretation": interpretation,
            "dates": [d.strftime("%Y-%m-%d") for d in ratio.index],
            "values": ratio.tolist(),
        })

    return series


def fetch_sector_rotation(weeks=12):
    """Fetch sector ETF prices and compute weekly returns for a heatmap.

    Single batch yf.download() call for all 11 sector tickers.
    """
    try:
        data = yf.download(SECTOR_TICKERS, period="6mo", progress=False)
        if data.empty:
            return None
        close = data["Close"] if "Close" in data.columns else data
        # Resample to weekly (Friday close)
        weekly = close.resample("W-FRI").last()
        returns = weekly.pct_change().dropna()
        if returns.empty:
            return None
        # Keep only the last N weeks
        returns = returns.tail(weeks)
        week_labels = [d.strftime("%b %d") for d in returns.index]
        return {
            "sectors": list(returns.columns),
            "week_labels": week_labels,
            "returns": returns.values.tolist(),
        }
    except Exception:
        return None


def fetch_orb_conditions(vix_price=None):
    """Compute ORB strategy regime conditions based on VIX and SPY.

    Backtest findings (60d, 71 US tickers):
      VIX < 18  → +28% ROI (go)
      VIX 18-22 → -6% ROI  (caution)
      VIX > 22  → -57% ROI (sit out)
      SPY up days → +36% ROI, SPY down >0.5% → -10.3% ROI
      Fridays → +46.6% ROI (best day)
    """
    from datetime import datetime

    result = {}

    # VIX — reuse if already fetched, otherwise fetch
    if vix_price is None:
        try:
            vix = yf.Ticker("^VIX")
            vix_price = vix.info.get("regularMarketPrice") or vix.info.get("previousClose")
        except Exception:
            pass

    result["vix"] = vix_price
    if vix_price is not None:
        if vix_price < 18:
            result["vix_regime"] = "GO"
            result["vix_color"] = "green"
        elif vix_price < 22:
            result["vix_regime"] = "CAUTION"
            result["vix_color"] = "yellow"
        else:
            result["vix_regime"] = "SIT OUT"
            result["vix_color"] = "red"

    # SPY — 30d history for SMA20 and daily return
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="30d")
        if not spy_hist.empty and len(spy_hist) >= 2:
            result["spy_close"] = round(spy_hist["Close"].iloc[-1], 2)
            daily_ret = (spy_hist["Close"].iloc[-1] / spy_hist["Close"].iloc[-2] - 1) * 100
            result["spy_daily_return"] = round(daily_ret, 2)

            if len(spy_hist) >= 20:
                sma20 = spy_hist["Close"].rolling(20).mean()
                sma_vals = sma20.dropna()
                if len(sma_vals) >= 5:
                    slope = (sma_vals.iloc[-1] - sma_vals.iloc[-5]) / sma_vals.iloc[-5] * 100
                    result["spy_sma20_slope"] = round(slope, 3)
                    if slope > 0.1:
                        result["spy_trend"] = "UP"
                    elif slope < -0.1:
                        result["spy_trend"] = "DOWN"
                    else:
                        result["spy_trend"] = "FLAT"
                result["spy_above_sma20"] = spy_hist["Close"].iloc[-1] > sma20.iloc[-1]
    except Exception:
        pass

    # Day of week
    now = datetime.now()
    result["day_of_week"] = now.strftime("%A")
    result["is_friday"] = now.weekday() == 4

    # Overall signal
    vix_ok = vix_price is not None and vix_price < 18
    spy_down_big = result.get("spy_daily_return", 0) is not None and result.get("spy_daily_return", 0) < -0.5
    vix_danger = vix_price is not None and vix_price >= 22

    if vix_danger:
        result["overall"] = "SIT OUT"
        result["overall_color"] = "red"
        result["overall_reason"] = "VIX too high — strategy loses money in high-vol regimes"
    elif spy_down_big and not vix_ok:
        result["overall"] = "SIT OUT"
        result["overall_color"] = "red"
        result["overall_reason"] = "VIX elevated + SPY down > 0.5%"
    elif not vix_ok:
        result["overall"] = "CAUTION"
        result["overall_color"] = "yellow"
        result["overall_reason"] = "VIX elevated — reduced edge"
    elif spy_down_big:
        result["overall"] = "CAUTION"
        result["overall_color"] = "yellow"
        result["overall_reason"] = "SPY down > 0.5% today — breakouts less reliable"
    else:
        result["overall"] = "GO"
        result["overall_color"] = "green"
        result["overall_reason"] = "Low VIX + favorable conditions"

    return result


def print_sentiment():
    """Print all sentiment indicators to terminal."""
    print("  VIX Term Structure")
    vix = fetch_vix_term_structure()
    if vix:
        print(f"    VIX: {vix['vix']:.1f}  |  VIX3M: {vix['vix3m']:.1f}  |  Ratio: {vix['ratio']:.2f}")
        print(f"    Structure: {vix['structure']} — {vix['signal']}")
    else:
        print("    Unavailable")

    print("\n  Leveraged ETF Bull/Bear Ratios")
    lev = fetch_leverage_ratios()
    if lev:
        for r in lev:
            print(f"    {r['label']:<12} {r['bull_ticker']}/{r['bear_ticker']}  "
                  f"AUM: ${r['bull_aum_b']:.1f}B / ${r['bear_aum_b']:.1f}B  "
                  f"Ratio: {r['ratio']:.1f}x  — {r['signal']}")
    else:
        print("    Unavailable")

    print("\n  Market Ratios")
    ratios = fetch_market_ratios()
    if ratios:
        print(f"    {'Ratio':<22} {'Value':>8} {'1W Chg':>8} {'1M Chg':>8}  Reading")
        print(f"    {'─'*22} {'─'*8} {'─'*8} {'─'*8}  {'─'*20}")
        for r in ratios:
            print(f"    {r['label']:<22} {r['ratio']:>8.3f} {r['wk_change_pct']:>+7.1f}% {r['mo_change_pct']:>+7.1f}%  {r['interpretation']}")
    else:
        print("    Unavailable")
    print()
