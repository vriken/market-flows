"""Previous Day High/Low (PDHL) Breakout strategy.

Tracks the previous day's high (PDH) and low (PDL), then enters when
price breaks through those levels with volume confirmation.

Entry: intraday breakout through PDH (long) or PDL (short) with volume spike.
Quality: above/below session VWAP adds confidence.
Stop: midpoint of previous day's range.
Exit: EOD or 1R target (risk = distance from entry to stop).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .base import BaseStrategy, Exit, Signal

ATR_PERIOD = 14
ATR_STOP_MULT = 1.5  # stop = entry ± ATR * multiplier

MARKET_OPEN = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()

# Skip the first 15 min to avoid opening-auction noise
ENTRY_START = dt.time(9, 45)

# Volume spike threshold: signal bar volume >= this × 20-period rolling avg
VOLUME_SPIKE_THRESHOLD = 1.5


class PDHLStrategy(BaseStrategy):
    """Previous Day High/Low breakout strategy.

    Uses 5-min intraday bars for entry timing and daily data for
    computing PDH/PDL levels. Enters on the first confirmed breakout
    candle with volume confirmation. Targets 1R profit (symmetric to risk).
    """

    name = "PDHL Breakout"
    requires_intraday = True

    def __init__(
        self,
        volume_threshold: float = 1.5,
        use_vwap_filter: bool = True,
        stop_mode: str = "atr",  # "atr", "midpoint", "level"
        atr_mult: float = ATR_STOP_MULT,
    ):
        self.volume_threshold = volume_threshold
        self.use_vwap_filter = use_vwap_filter
        self.stop_mode = stop_mode
        self.atr_mult = atr_mult
        self._pdhl_cache: dict[str, dict] = {}  # ticker -> {date: {pdh, pdl}}
        self._vol_sma_cache: dict[str, pd.Series] = {}
        self._atr_cache: dict[tuple, pd.Series] = {}

    # ── Signal generation ──────────────────────────────────────────────────

    def generate_signals(self, data: dict, date: dt.date) -> list[Signal]:
        intraday = data.get("intraday")
        daily = data.get("daily")
        ticker = data.get("ticker", "")

        if intraday is None or intraday.empty:
            return []

        # Get PDH/PDL from previous day
        pdhl = self._get_pdhl(intraday, daily, ticker, date)
        if pdhl is None:
            return []

        pdh = pdhl["pdh"]
        pdl = pdhl["pdl"]
        pd_mid = (pdh + pdl) / 2  # midpoint for stop (legacy)

        # Compute ATR from daily data for stop sizing
        atr_val = self._get_atr(daily, ticker, date)

        # Get today's intraday bars
        day_bars = intraday[intraday.index.date == date]
        if day_bars.empty:
            return []

        # Filter to after entry start time
        post_open = day_bars[day_bars.index.time >= ENTRY_START]
        if post_open.empty:
            return []

        # Volume rolling average for spike detection
        vol_sma = self._get_vol_sma(intraday, ticker)

        # Optionally compute session VWAP for quality filter
        vwap = self._compute_session_vwap(day_bars) if self.use_vwap_filter else None

        for ts, row in post_open.iterrows():
            close = float(row["Close"])
            volume = float(row["Volume"])

            # Volume confirmation
            avg_vol = vol_sma.get(ts, volume) if vol_sma is not None else volume
            vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0
            has_volume = vol_ratio >= self.volume_threshold

            if not has_volume:
                continue

            # Current VWAP value
            vwap_val = None
            if vwap is not None and ts in vwap.index:
                vwap_val = float(vwap.loc[ts])

            # Long: close above PDH (breakout)
            if close > pdh:
                # VWAP quality: above VWAP confirms long
                vwap_confirmed = vwap_val is not None and close > vwap_val

                stop_price = self._compute_stop(close, "long", pd_mid, pdl, atr_val)
                risk = close - stop_price
                if risk <= 0:
                    continue

                quality_parts = []
                if vwap_confirmed:
                    quality_parts.append("W")  # VWAP confirmed
                if vol_ratio >= 2.0:
                    quality_parts.append("V")  # strong volume spike

                # Require all quality flags to enter
                if len(quality_parts) < 2:
                    continue

                target_price = close + risk  # 1R target

                return [Signal(
                    date=date,
                    time=ts.time() if hasattr(ts, "time") else None,
                    ticker=ticker,
                    direction="long",
                    entry_price=close,
                    stop_price=stop_price,
                    target_price=target_price,
                    quality_score="".join(quality_parts),
                    quality_flags={
                        "pdh": pdh,
                        "pdl": pdl,
                        "vol_ratio": round(vol_ratio, 2),
                        "vwap_confirmed": vwap_confirmed,
                        "vwap_value": vwap_val,
                    },
                    metadata={
                        "pdh": pdh,
                        "pdl": pdl,
                        "pd_mid": pd_mid,
                        "risk": risk,
                        "max_hold_days": 1,
                    },
                )]

            # Short: close below PDL (breakdown)
            elif close < pdl:
                vwap_confirmed = vwap_val is not None and close < vwap_val

                stop_price = self._compute_stop(close, "short", pd_mid, pdh, atr_val)
                risk = stop_price - close
                if risk <= 0:
                    continue

                quality_parts = []
                if vwap_confirmed:
                    quality_parts.append("W")
                if vol_ratio >= 2.0:
                    quality_parts.append("V")

                # Require all quality flags to enter
                if len(quality_parts) < 2:
                    continue

                target_price = close - risk  # 1R target

                return [Signal(
                    date=date,
                    time=ts.time() if hasattr(ts, "time") else None,
                    ticker=ticker,
                    direction="short",
                    entry_price=close,
                    stop_price=stop_price,
                    target_price=target_price,
                    quality_score="".join(quality_parts),
                    quality_flags={
                        "pdh": pdh,
                        "pdl": pdl,
                        "vol_ratio": round(vol_ratio, 2),
                        "vwap_confirmed": vwap_confirmed,
                        "vwap_value": vwap_val,
                    },
                    metadata={
                        "pdh": pdh,
                        "pdl": pdl,
                        "pd_mid": pd_mid,
                        "risk": risk,
                        "max_hold_days": 1,
                    },
                )]

        return []

    # ── Exit logic ─────────────────────────────────────────────────────────

    def check_exit(
        self,
        signal: Signal,
        current_bar: dict,
        bars_since_entry: int,
        day_index: int,
    ) -> Exit | None:
        target = signal.target_price
        stop = signal.stop_price

        # Target hit (1.5R)
        if target is not None:
            if (signal.direction == "long" and current_bar["High"] >= target) or (signal.direction == "short" and current_bar["Low"] <= target):
                return Exit(
                    should_exit=True,
                    exit_price=float(target),
                    reason="target",
                    metadata={"trigger": "1.5r_target"},
                )

        # Stop hit (midpoint of previous day's range)
        if signal.direction == "long" and current_bar["Close"] < stop:
            return Exit(
                should_exit=True,
                exit_price=current_bar["Close"],
                reason="stop",
                metadata={"trigger": "close_below_pd_mid"},
            )
        elif signal.direction == "short" and current_bar["Close"] > stop:
            return Exit(
                should_exit=True,
                exit_price=current_bar["Close"],
                reason="stop",
                metadata={"trigger": "close_above_pd_mid"},
            )

        # EOD exit
        bar_time = current_bar.get("time")
        if bar_time is not None and bar_time >= dt.time(15, 55):
            return Exit(
                should_exit=True,
                exit_price=current_bar["Close"],
                reason="close",
                metadata={"trigger": "eod"},
            )

        return None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _get_pdhl(
        self,
        intraday: pd.DataFrame,
        daily: pd.DataFrame | None,
        ticker: str,
        date: dt.date,
    ) -> dict | None:
        """Get previous day high/low for the given date.

        Prefers daily data if available; falls back to intraday aggregation.
        """
        cache_key = f"{ticker}_{date}"
        if cache_key in self._pdhl_cache:
            return self._pdhl_cache[cache_key]

        if daily is not None and not daily.empty:
            idx = daily.index
            date_vals = [d.date() if hasattr(d, "date") else d for d in idx]
            # Find the trading day immediately before `date`
            prev_dates = [d for d in date_vals if d < date]
            if prev_dates:
                prev_date = prev_dates[-1]
                prev_mask = [d == prev_date for d in date_vals]
                prev_bar = daily[prev_mask]
                if not prev_bar.empty:
                    result = {
                        "pdh": float(prev_bar.iloc[0]["High"]),
                        "pdl": float(prev_bar.iloc[0]["Low"]),
                    }
                    self._pdhl_cache[cache_key] = result
                    return result

        # Fallback: derive from intraday
        all_dates = sorted(set(intraday.index.date))
        if date not in all_dates:
            return None
        date_idx = all_dates.index(date)
        if date_idx < 1:
            return None

        prev_date = all_dates[date_idx - 1]
        prev_bars = intraday[intraday.index.date == prev_date]
        if prev_bars.empty:
            return None

        result = {
            "pdh": float(prev_bars["High"].max()),
            "pdl": float(prev_bars["Low"].min()),
        }
        self._pdhl_cache[cache_key] = result
        return result

    def _get_vol_sma(self, intraday: pd.DataFrame, ticker: str) -> pd.Series:
        """20-period rolling volume SMA (session-aware)."""
        if ticker not in self._vol_sma_cache:
            self._vol_sma_cache[ticker] = self.session_volume_sma(intraday, window=20)
        return self._vol_sma_cache[ticker]

    def _compute_stop(
        self,
        entry: float,
        direction: str,
        pd_mid: float,
        level: float,
        atr_val: float | None,
    ) -> float:
        """Compute stop price based on stop_mode."""
        if self.stop_mode == "atr" and atr_val is not None and atr_val > 0:
            if direction == "long":
                return entry - self.atr_mult * atr_val
            else:
                return entry + self.atr_mult * atr_val
        elif self.stop_mode == "level":
            return level  # PDL for longs, PDH for shorts
        else:
            return pd_mid  # legacy midpoint

    def _get_atr(
        self,
        daily: pd.DataFrame | None,
        ticker: str,
        date: dt.date,
    ) -> float | None:
        """Get ATR value for the previous trading day."""
        if daily is None or daily.empty:
            return None

        cache_key = (ticker, len(daily))
        if cache_key not in self._atr_cache:
            high = daily["High"].values
            low = daily["Low"].values
            close = daily["Close"].values

            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1)),
                ),
            )
            tr[0] = high[0] - low[0]
            self._atr_cache[cache_key] = pd.Series(
                tr, index=daily.index
            ).rolling(ATR_PERIOD, min_periods=1).mean()

        atr_series = self._atr_cache[cache_key]
        # Use the ATR from the day before `date`
        date_vals = [d.date() if hasattr(d, "date") else d for d in atr_series.index]
        prev_dates = [d for d in date_vals if d < date]
        if not prev_dates:
            return None
        prev_date = prev_dates[-1]
        idx = date_vals.index(prev_date)
        return float(atr_series.iloc[idx])

    @staticmethod
    def _compute_session_vwap(day_bars: pd.DataFrame) -> pd.Series | None:
        """Compute session VWAP for a single day."""
        if day_bars.empty or "Volume" not in day_bars.columns:
            return None

        tp = (day_bars["High"] + day_bars["Low"] + day_bars["Close"]) / 3
        tp_vol = tp * day_bars["Volume"]
        cum_vol = day_bars["Volume"].cumsum().replace(0, np.nan)
        vwap = tp_vol.cumsum() / cum_vol

        return vwap
