"""Build a day-by-day historical regime classification.

Reconstructs the sentiment_data dict that :func:`classify_regime` expects
for each trading day in the requested range, using historical price data
from yfinance and (optionally) FRED credit-spread data.

The result is a DataFrame indexed by date with the regime dimensions,
composite label, and confidence score — suitable for overlaying on
backtest results to analyse strategy performance per regime.
"""

import logging
import os
from datetime import datetime

import pandas as pd
import yfinance as yf

from ..config import CREDIT_SPREAD_SERIES, DATA_DIR, YIELD_CURVE_TICKERS
from ..regime import classify_regime

logger = logging.getLogger(__name__)

BACKTEST_DATA_DIR = DATA_DIR / "backtest"
REGIME_CACHE_PATH = BACKTEST_DATA_DIR / "regime_history.parquet"

# Tickers needed for historical reconstruction
_VIX_TICKER = "^VIX"
_VIX3M_TICKER = "^VIX3M"

# Ratio pairs: (numerator, denominator, label) — must match MARKET_RATIOS labels
_RATIO_PAIRS = [
    ("CPER", "GLD", "Copper/Gold"),
    ("XLY", "XLP", "Discretionary/Staples"),
    ("HYG", "LQD", "High Yield/Inv Grade"),
    ("IWM", "SPY", "Small Cap/Large Cap"),
]

# Number of trailing business days to compute 1-month % change
_MONTH_LOOKBACK = 22

# Number of trailing business days for the 2Y yield rolling windows
_YIELD_RECENT_WINDOW = 20
_YIELD_OLDER_START = 60
_YIELD_OLDER_END = 40


def _download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Batch-download daily close prices via yf.download.

    Returns a DataFrame with tickers as columns and a DatetimeIndex.
    """
    try:
        raw = yf.download(tickers, start=start, end=end, progress=False, threads=True)
    except Exception as e:
        logger.error("Batch download failed: %s", e)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        # Single ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return close.ffill()


def _fetch_hy_oas_history(start: str, end: str) -> pd.Series | None:
    """Fetch HY OAS from FRED if an API key is available.

    Returns a Series indexed by date, or None.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None

    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        series_id = CREDIT_SPREAD_SERIES["HY OAS"]
        data = fred.get_series(series_id, observation_start=start, observation_end=end)
        if data is not None and not data.empty:
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            return data
    except Exception as e:
        logger.warning("FRED HY OAS fetch failed: %s", e)

    return None


def _compute_credit_data(
    hy_oas: pd.Series | None,
    hyg_close: pd.Series | None,
    lqd_close: pd.Series | None,
    date: pd.Timestamp,
) -> dict | None:
    """Compute credit_data dict for a single date.

    Uses FRED HY OAS when available, otherwise approximates from HYG/LQD
    spread as a directional proxy.
    """
    if hy_oas is not None and date in hy_oas.index:
        # Use actual OAS data with 5-year rolling percentile
        lookback_start = date - pd.Timedelta(days=365 * 5)
        window = hy_oas.loc[lookback_start:date]
        if len(window) >= 20:
            current = hy_oas.loc[date]
            percentile = (window <= current).sum() / len(window) * 100
            return {
                "current_hy": current,
                "hy_percentile": round(percentile, 0),
            }

    # Proxy: use inverse of HYG/LQD ratio changes as a spread proxy
    if hyg_close is not None and lqd_close is not None:
        ratio = hyg_close / lqd_close
        # Falling HYG/LQD → widening spreads → higher stress
        lookback_start = date - pd.Timedelta(days=365 * 2)
        window = ratio.loc[lookback_start:date].dropna()
        if len(window) >= 20:
            current = window.iloc[-1]
            # Invert: lower ratio = wider spread = higher percentile
            percentile_raw = (window >= current).sum() / len(window) * 100
            return {
                "current_hy": 0,  # unknown in proxy mode
                "hy_percentile": round(percentile_raw, 0),
            }

    return None


def build_regime_history(
    start_date: str | datetime,
    end_date: str | datetime,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Classify the market regime for each trading day in [start_date, end_date].

    This reconstructs the ``sentiment_data`` dict that
    :func:`market_flows.regime.classify_regime` expects, using historical
    daily prices from yfinance and (optionally) FRED credit-spread data.

    **Data sources:**

    - VIX / VIX3M: ``^VIX``, ``^VIX3M`` (yfinance)
    - Yields: ``^IRX``, ``2YY=F``, ``^FVX``, ``^TNX``, ``^TYX`` (yfinance)
    - Ratios: 1-month % changes of CPER/GLD, XLY/XLP, HYG/LQD, IWM/SPY
    - Credit: FRED HY OAS if ``FRED_API_KEY`` is set; else HYG/LQD proxy
    - Breadth / Liquidity: skipped (set to None)

    Results are cached to ``data/backtest/regime_history.parquet``.

    Args:
        start_date: First date to classify (inclusive).
        end_date:   Last date to classify (inclusive).
        force:      If True, bypass cache and recompute.

    Returns:
        DataFrame indexed by date with columns:
        volatility_state, cycle_state, risk_state, monetary_state,
        credit_state, composite_label, confidence.
    """
    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    # Check cache
    if not force and REGIME_CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(REGIME_CACHE_PATH)
            mask = (cached.index >= start_str) & (cached.index <= end_str)
            subset = cached.loc[mask]
            if not subset.empty:
                # Check coverage: do we have ≥90% of requested trading days?
                expected = len(pd.bdate_range(start_str, end_str))
                if len(subset) >= expected * 0.9:
                    logger.debug(
                        "Regime history cache hit (%d/%d days)", len(subset), expected,
                    )
                    return subset
        except Exception:
            logger.debug("Regime cache read failed, rebuilding")

    # --- Gather all tickers we need -------------------------------------------
    yield_tickers = list(YIELD_CURVE_TICKERS.values())
    ratio_tickers = list({t for num, den, _ in _RATIO_PAIRS for t in (num, den)})
    all_tickers = [_VIX_TICKER, _VIX3M_TICKER] + yield_tickers + ratio_tickers

    # Extra lookback for rolling windows (yields need ~60 days, ratios ~22 days)
    lookback_start = (pd.Timestamp(start_str) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    # Add 1 day to end for yfinance's exclusive end
    dl_end = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Downloading price history for regime classification (%s → %s)", lookback_start, end_str)
    prices = _download_batch(all_tickers, lookback_start, dl_end)

    if prices.empty:
        logger.error("No price data returned, cannot build regime history")
        return pd.DataFrame()

    # --- Fix ^IRX scaling quirk (same as sentiment.py) -----------------------
    if "^IRX" in prices.columns:
        prices["^IRX"] = prices["^IRX"].where(prices["^IRX"] <= 20, prices["^IRX"] / 10)

    # --- Fetch FRED credit data ----------------------------------------------
    hy_oas = _fetch_hy_oas_history(lookback_start, end_str)

    # --- Pre-compute ratio series --------------------------------------------
    ratio_series: dict[str, pd.Series] = {}
    for num_tick, den_tick, label in _RATIO_PAIRS:
        if num_tick in prices.columns and den_tick in prices.columns:
            ratio_series[label] = prices[num_tick] / prices[den_tick]

    # --- Yield history keyed by maturity label --------------------------------
    yield_label_to_ticker = YIELD_CURVE_TICKERS  # {"3m": "^IRX", ...}

    # --- Classify each trading day --------------------------------------------
    trading_dates = pd.bdate_range(start_str, end_str)
    records: list[dict] = []

    for date in trading_dates:
        if date not in prices.index:
            continue

        # -- VIX data ----------------------------------------------------------
        vix_data = None
        vix_val = prices.at[date, _VIX_TICKER] if _VIX_TICKER in prices.columns else None
        vix3m_val = prices.at[date, _VIX3M_TICKER] if _VIX3M_TICKER in prices.columns else None

        if pd.notna(vix_val) and pd.notna(vix3m_val) and vix3m_val != 0:
            vix_data = {
                "vix": float(vix_val),
                "vix3m": float(vix3m_val),
                "ratio": float(vix_val / vix3m_val),
            }

        # -- Yield curve -------------------------------------------------------
        yields_today: dict[str, float] = {}
        for label, ticker in yield_label_to_ticker.items():
            if ticker in prices.columns and pd.notna(prices.at[date, ticker]):
                yields_today[label] = float(prices.at[date, ticker])

        yield_curve = None
        if "10y" in yields_today:
            spreads: dict[str, float] = {}
            if "2y" in yields_today:
                spreads["2s10s"] = round(yields_today["10y"] - yields_today["2y"], 3)
            if "3m" in yields_today:
                spreads["3m10y"] = round(yields_today["10y"] - yields_today["3m"], 3)
            yield_curve = {
                "yields": yields_today,
                "spreads": spreads,
            }

        # -- Yield history (for monetary regime) --------------------------------
        yield_history = None
        if "2y" in yield_label_to_ticker:
            y2_ticker = yield_label_to_ticker["2y"]
            if y2_ticker in prices.columns:
                y2_to_date = prices.loc[:date, y2_ticker].dropna()
                if len(y2_to_date) >= _YIELD_OLDER_START:
                    yield_history = {
                        "yields": {
                            "2y": y2_to_date.values.tolist(),
                        },
                    }

        # -- Ratio data (1-month % changes) ------------------------------------
        ratios_list: list[dict] = []
        for num_tick, den_tick, label in _RATIO_PAIRS:
            if label not in ratio_series:
                continue
            rs = ratio_series[label]
            if date not in rs.index:
                continue
            current_val = rs.loc[date]
            # Look back ~22 trading days for 1-month change
            prior_idx = rs.loc[:date].index
            if len(prior_idx) > _MONTH_LOOKBACK:
                prev_val = rs.iloc[prior_idx.get_loc(date) - _MONTH_LOOKBACK]
                if prev_val != 0 and pd.notna(prev_val) and pd.notna(current_val):
                    mo_change = (float(current_val) / float(prev_val) - 1) * 100
                    ratios_list.append({
                        "label": label,
                        "numerator": num_tick,
                        "denominator": den_tick,
                        "ratio": float(current_val),
                        "mo_change_pct": mo_change,
                    })

        # -- Credit data -------------------------------------------------------
        hyg_close = prices[_ratio_tick("HYG")] if _ratio_tick("HYG") in prices.columns else None
        lqd_close = prices[_ratio_tick("LQD")] if _ratio_tick("LQD") in prices.columns else None
        credit_data = _compute_credit_data(
            hy_oas,
            hyg_close.loc[:date] if hyg_close is not None else None,
            lqd_close.loc[:date] if lqd_close is not None else None,
            date,
        )

        # -- Build the sentiment_data dict -------------------------------------
        sentiment_data: dict = {
            "vix": vix_data,
            "yield_curve": yield_curve,
            "yield_history": yield_history,
            "ratios": ratios_list,
            "leverage": [],  # Historical AUM data not available
        }

        # -- Classify -----------------------------------------------------------
        result = classify_regime(
            sentiment_data,
            credit_data=credit_data,
            liquidity_data=None,   # Skipped — requires daily FRED snapshots
            breadth_data=None,     # Skipped — too expensive to compute historically
        )

        # Extract dimension states
        dim_map = {d["name"]: d["state"] for d in result["dimensions"]}
        records.append({
            "date": date,
            "volatility_state": dim_map.get("Volatility"),
            "cycle_state": dim_map.get("Cycle"),
            "risk_state": dim_map.get("Risk"),
            "monetary_state": dim_map.get("Monetary"),
            "credit_state": dim_map.get("Credit"),
            "composite_label": result["composite_label"],
            "confidence": result["confidence"],
        })

    if not records:
        logger.warning("No regime records produced for %s → %s", start_str, end_str)
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")
    df.index.name = "date"

    # --- Cache ----------------------------------------------------------------
    _save_regime_cache(df)

    logger.info(
        "Built regime history: %d days, %d unique labels",
        len(df), df["composite_label"].nunique(),
    )
    return df


def _ratio_tick(tick: str) -> str:
    """Return the ticker string as it appears in the price columns.

    Helper for accessing individual tickers from the batch download.
    """
    return tick


def _save_regime_cache(df: pd.DataFrame) -> None:
    """Merge *df* into the on-disk regime history cache."""
    BACKTEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if REGIME_CACHE_PATH.exists():
        try:
            existing = pd.read_parquet(REGIME_CACHE_PATH)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined.to_parquet(REGIME_CACHE_PATH)
            return
        except Exception:
            pass

    df.to_parquet(REGIME_CACHE_PATH)
