"""Fair Value Gap (FVG) Fill strategy.

Detects FVGs on the daily chart and enters when price fills back into the gap.

FVG detection (3-candle pattern):
  - Bullish FVG: bar[0].low > bar[2].high  (gap up — buyers left a void)
  - Bearish FVG: bar[0].high < bar[2].low  (gap down — sellers left a void)

Entry: next day, if price enters the FVG zone, enter in the direction of the fill.
  - Long into bullish FVG (buying the dip into the gap-up zone)
  - Short into bearish FVG (selling the rally into the gap-down zone)
Stop: opposite side of FVG + buffer (ATR-based).
Target: full FVG fill (opposite edge of the gap).
Exit: hit target, hit stop, or 3-day max hold.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BaseStrategy, Exit, Signal

MAX_HOLD_DAYS = 3
FVG_BUFFER_ATR_MULT = 0.25  # stop buffer = 25% of ATR beyond FVG edge


@dataclass
class FVGZone:
    """A detected Fair Value Gap zone."""

    date: dt.date  # date of the middle candle (bar[1])
    direction: str  # "bullish" or "bearish"
    upper: float  # top of the FVG zone
    lower: float  # bottom of the FVG zone
    atr: float  # ATR at detection for buffer sizing


class FVGStrategy(BaseStrategy):
    """Fair Value Gap fill strategy on daily OHLCV data.

    Scans for 3-candle FVG patterns, then enters when the next day's price
    moves into the gap zone. Targets full gap fill with a 3-day max hold.
    """

    name = "FVG Fill"
    requires_intraday = False

    def __init__(self, max_hold_days: int = 3, buffer_atr_mult: float = 0.25):
        self.max_hold_days = max_hold_days
        self.buffer_atr_mult = buffer_atr_mult
        self._fvg_cache: dict[str, list[FVGZone]] = {}
        self._atr_cache: dict[str, pd.Series] = {}

    # ── Signal generation ──────────────────────────────────────────────────

    def generate_signals(self, data: dict, date: dt.date) -> list[Signal]:
        daily = data.get("daily")
        ticker = data.get("ticker", "")

        if daily is None or len(daily) < 5:
            return []

        # Detect all FVGs in history
        fvg_zones = self._detect_fvgs(daily, ticker)

        idx = daily.index
        date_vals = [d.date() if hasattr(d, "date") else d for d in idx]

        # Use YESTERDAY's completed bar to check zone entry (no look-ahead)
        prev_dates = [d for d in date_vals if d < date]
        if not prev_dates:
            return []
        prev_date = prev_dates[-1]
        prev_mask = [d == prev_date for d in date_vals]
        prev_bar = daily[prev_mask]
        if prev_bar.empty:
            return []

        prev = prev_bar.iloc[0]
        prev_high = float(prev["High"])
        prev_low = float(prev["Low"])
        prev_close = float(prev["Close"])

        # Get today's open for entry price (known at market open)
        today_mask = [d == date for d in date_vals]
        today_bars = daily[today_mask]
        if today_bars.empty:
            return []
        today_open = float(today_bars.iloc[0]["Open"])

        # Active FVGs: detected before yesterday, max 20 trading days old
        max_age = dt.timedelta(days=30)  # ~20 trading days
        active_fvgs = [f for f in fvg_zones if f.date < prev_date and (prev_date - f.date) <= max_age]
        if not active_fvgs:
            return []

        # Check most recent FVGs first (priority to freshest gaps)
        for fvg in reversed(active_fvgs[-5:]):
            if fvg.direction == "bullish":
                # Yesterday's low dipped into bullish FVG zone = fill started
                if prev_low <= fvg.upper and prev_close > fvg.lower:
                    entry_price = today_open
                    stop_price = fvg.lower - self.buffer_atr_mult * fvg.atr
                    risk = entry_price - stop_price
                    if risk <= 0:
                        continue
                    target_price = entry_price + 1.5 * risk  # 1.5R target

                    return [Signal(
                        date=date,
                        time=None,
                        ticker=ticker,
                        direction="long",
                        entry_price=round(entry_price, 4),
                        stop_price=round(stop_price, 4),
                        target_price=round(target_price, 4),
                        quality_score="",
                        quality_flags={
                            "fvg_type": "bullish",
                            "fvg_upper": fvg.upper,
                            "fvg_lower": fvg.lower,
                            "fvg_size_pct": (fvg.upper - fvg.lower) / fvg.lower * 100,
                        },
                        metadata={
                            "fvg_upper": fvg.upper,
                            "fvg_lower": fvg.lower,
                            "fvg_date": str(fvg.date),
                            "max_hold_days": self.max_hold_days,
                        },
                    )]

            elif fvg.direction == "bearish":
                # Yesterday's high rose into bearish FVG zone = fill started
                if prev_high >= fvg.lower and prev_close < fvg.upper:
                    entry_price = today_open
                    stop_price = fvg.upper + self.buffer_atr_mult * fvg.atr
                    risk = stop_price - entry_price
                    if risk <= 0:
                        continue
                    target_price = entry_price - 1.5 * risk  # 1.5R target

                    return [Signal(
                        date=date,
                        time=None,
                        ticker=ticker,
                        direction="short",
                        entry_price=round(entry_price, 4),
                        stop_price=round(stop_price, 4),
                        target_price=round(target_price, 4),
                        quality_score="",
                        quality_flags={
                            "fvg_type": "bearish",
                            "fvg_upper": fvg.upper,
                            "fvg_lower": fvg.lower,
                            "fvg_size_pct": (fvg.upper - fvg.lower) / fvg.lower * 100,
                        },
                        metadata={
                            "fvg_upper": fvg.upper,
                            "fvg_lower": fvg.lower,
                            "fvg_date": str(fvg.date),
                            "max_hold_days": self.max_hold_days,
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

        # Stop hit
        if (signal.direction == "long" and current_bar["Low"] <= stop) or (signal.direction == "short" and current_bar["High"] >= stop):
            return Exit(
                should_exit=True,
                exit_price=float(stop),
                reason="stop",
                metadata={"trigger": "fvg_stop"},
            )

        # Max hold days
        if day_index >= self.max_hold_days - 1:
            return Exit(
                should_exit=True,
                exit_price=current_bar["Close"],
                reason="max_hold",
                metadata={"trigger": f"max_hold_{self.max_hold_days}d"},
            )

        return None

    # ── FVG detection ──────────────────────────────────────────────────────

    def _detect_fvgs(self, daily: pd.DataFrame, ticker: str) -> list[FVGZone]:
        """Detect all FVG zones in the daily data."""
        cache_key = (ticker, len(daily))
        if cache_key in self._fvg_cache:
            return self._fvg_cache[cache_key]

        atr = self._compute_atr(daily, ticker)
        zones: list[FVGZone] = []

        idx = daily.index
        high = daily["High"].values
        low = daily["Low"].values

        for i in range(2, len(daily)):
            bar0_low = float(low[i])
            bar0_high = float(high[i])
            bar2_high = float(high[i - 2])
            bar2_low = float(low[i - 2])

            date_i = idx[i]
            d = date_i.date() if hasattr(date_i, "date") else date_i
            atr_val = float(atr.iloc[i]) if i < len(atr) and pd.notna(atr.iloc[i]) else 1.0

            # Bullish FVG: bar[0].low > bar[2].high (gap up)
            if bar0_low > bar2_high:
                zones.append(FVGZone(
                    date=d,
                    direction="bullish",
                    upper=bar0_low,
                    lower=bar2_high,
                    atr=atr_val,
                ))

            # Bearish FVG: bar[0].high < bar[2].low (gap down)
            elif bar0_high < bar2_low:
                zones.append(FVGZone(
                    date=d,
                    direction="bearish",
                    upper=bar2_low,
                    lower=bar0_high,
                    atr=atr_val,
                ))

        self._fvg_cache[cache_key] = zones
        return zones

    def _compute_atr(self, daily: pd.DataFrame, ticker: str) -> pd.Series:
        """Compute 14-period ATR on daily data."""
        cache_key = (ticker, len(daily))
        if cache_key in self._atr_cache:
            return self._atr_cache[cache_key]

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
        atr = pd.Series(tr, index=daily.index).rolling(14, min_periods=1).mean()

        self._atr_cache[cache_key] = atr
        return atr
