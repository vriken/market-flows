"""Trend/Momentum Following strategy.

Uses a daily SMA stack (20, 50, 100, 200) to identify strongly trending
markets. Entry occurs when the stack is fully aligned and price breaks
above/below the 20 SMA.

Entry: SMA stack fully aligned + price breaks above 20 SMA (long) or below (short).
  - Bullish stack: SMA20 > SMA50 > SMA100 > SMA200, price > SMA20
  - Bearish stack: SMA20 < SMA50 < SMA100 < SMA200, price < SMA20
Direction: Long when bullish stack, Short when bearish stack.
Stop: close below 50 SMA (long) or above 50 SMA (short).
Exit: stack breaks down (loses full alignment), or 10-day max hold.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .base import BaseStrategy, Exit, Signal

SMA_PERIODS = [20, 50, 100, 200]
MAX_HOLD_DAYS = 10


class MomentumStrategy(BaseStrategy):
    """SMA-stack trend-following strategy on daily data.

    Enters when all four SMAs are perfectly stacked and price confirms
    by closing beyond the fast (20) SMA. Exits when the stack breaks
    or after 10 trading days.
    """

    name = "Momentum"
    requires_intraday = False

    def __init__(self, max_hold_days: int = 10):
        self.max_hold_days = max_hold_days
        self._sma_cache: dict[str, pd.DataFrame] = {}

    # ── Signal generation ──────────────────────────────────────────────────

    def generate_signals(self, data: dict, date: dt.date) -> list[Signal]:
        daily = data.get("daily")
        ticker = data.get("ticker", "")

        if daily is None or len(daily) < 200:
            return []

        sma_df = self._compute_smas(daily, ticker)

        # Find today's row and yesterday's row
        idx = sma_df.index
        date_vals = [d.date() if hasattr(d, "date") else d for d in idx]
        date_to_iloc: dict = {d: i for i, d in enumerate(date_vals)}

        today_idx = date_to_iloc.get(date)
        if today_idx is None or today_idx < 1:
            return []

        today = sma_df.iloc[today_idx]
        yesterday = sma_df.iloc[today_idx - 1]

        # Check that all SMAs are available
        if any(pd.isna(today[f"sma{p}"]) for p in SMA_PERIODS):
            return []

        sma20 = float(today["sma20"])
        sma50 = float(today["sma50"])
        sma100 = float(today["sma100"])
        sma200 = float(today["sma200"])
        close = float(today["Close"])
        prev_close = float(yesterday["Close"])
        prev_sma20 = float(yesterday["sma20"])

        signals: list[Signal] = []

        # Bullish stack: SMA20 > SMA50 > SMA100 > SMA200
        bullish_stack = sma20 > sma50 > sma100 > sma200

        # Bearish stack: SMA20 < SMA50 < SMA100 < SMA200
        bearish_stack = sma20 < sma50 < sma100 < sma200

        if bullish_stack and close > sma20 and prev_close <= prev_sma20:
            # Price just broke above 20 SMA with full bullish alignment
            # Compute gradient for quality assessment
            gradient = self._compute_gradient(sma_df, today_idx)

            # Require all quality flags to enter
            if self._quality_label(gradient, "long") != "GS":
                return signals

            signals.append(Signal(
                date=date,
                time=None,
                ticker=ticker,
                direction="long",
                entry_price=close,
                stop_price=sma50,  # stop at 50 SMA
                target_price=None,  # no fixed target; exit on stack break or max hold
                quality_score=self._quality_label(gradient, "long"),
                quality_flags={
                    "stack": "bullish",
                    "gradient_20": gradient.get("gradient_20", 0),
                    "gradient_50": gradient.get("gradient_50", 0),
                    "sma_spread_pct": (sma20 - sma200) / sma200 * 100,
                },
                metadata={
                    "sma20": sma20,
                    "sma50": sma50,
                    "sma100": sma100,
                    "sma200": sma200,
                    "max_hold_days": self.max_hold_days,
                },
            ))

        elif bearish_stack and close < sma20 and prev_close >= prev_sma20:
            # Price just broke below 20 SMA with full bearish alignment
            gradient = self._compute_gradient(sma_df, today_idx)

            # Require all quality flags to enter
            if self._quality_label(gradient, "short") != "GS":
                return signals

            signals.append(Signal(
                date=date,
                time=None,
                ticker=ticker,
                direction="short",
                entry_price=close,
                stop_price=sma50,  # stop at 50 SMA
                target_price=None,
                quality_score=self._quality_label(gradient, "short"),
                quality_flags={
                    "stack": "bearish",
                    "gradient_20": gradient.get("gradient_20", 0),
                    "gradient_50": gradient.get("gradient_50", 0),
                    "sma_spread_pct": (sma200 - sma20) / sma200 * 100,
                },
                metadata={
                    "sma20": sma20,
                    "sma50": sma50,
                    "sma100": sma100,
                    "sma200": sma200,
                    "max_hold_days": self.max_hold_days,
                },
            ))

        return signals

    # ── Exit logic ─────────────────────────────────────────────────────────

    def check_exit(
        self,
        signal: Signal,
        current_bar: dict,
        bars_since_entry: int,
        day_index: int,
    ) -> Exit | None:
        stop = signal.stop_price

        # Stop: close beyond 50 SMA
        if signal.direction == "long" and current_bar["Close"] < stop:
            return Exit(
                should_exit=True,
                exit_price=current_bar["Close"],
                reason="stop",
                metadata={"trigger": "close_below_sma50"},
            )
        elif signal.direction == "short" and current_bar["Close"] > stop:
            return Exit(
                should_exit=True,
                exit_price=current_bar["Close"],
                reason="stop",
                metadata={"trigger": "close_above_sma50"},
            )

        # Stack break: check if alignment is lost
        # We store SMA values in metadata at entry; for ongoing bars we compare
        # current close to the SMAs. Since daily bars = one bar per day, we use
        # a simplified check: if price reverses to the other side of sma20
        sma20 = signal.metadata.get("sma20", 0)
        if sma20:
            if signal.direction == "long" and current_bar["Close"] < sma20 * 0.99:
                return Exit(
                    should_exit=True,
                    exit_price=current_bar["Close"],
                    reason="stack_break",
                    metadata={"trigger": "price_back_below_sma20"},
                )
            elif signal.direction == "short" and current_bar["Close"] > sma20 * 1.01:
                return Exit(
                    should_exit=True,
                    exit_price=current_bar["Close"],
                    reason="stack_break",
                    metadata={"trigger": "price_back_above_sma20"},
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

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _compute_smas(self, daily: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Compute SMAs and cache per ticker."""
        if ticker in self._sma_cache:
            return self._sma_cache[ticker]

        df = daily.copy()
        for p in SMA_PERIODS:
            df[f"sma{p}"] = df["Close"].rolling(p).mean()

        self._sma_cache[ticker] = df
        return df

    def _compute_gradient(self, sma_df: pd.DataFrame, today_idx: int, lookback: int = 5) -> dict:
        """Compute SMA slope as % change over lookback days."""
        result = {}
        for p in [20, 50]:
            col = f"sma{p}"
            if today_idx >= lookback:
                current = sma_df.iloc[today_idx][col]
                past = sma_df.iloc[today_idx - lookback][col]
                if pd.notna(current) and pd.notna(past) and past != 0:
                    result[f"gradient_{p}"] = round((current - past) / past * 100, 4)
                else:
                    result[f"gradient_{p}"] = 0.0
            else:
                result[f"gradient_{p}"] = 0.0
        return result

    @staticmethod
    def _quality_label(gradient: dict, direction: str) -> str:
        """Build a quality label from gradient strength."""
        g20 = gradient.get("gradient_20", 0)
        g50 = gradient.get("gradient_50", 0)

        parts = []
        # Strong gradient = fast SMA accelerating
        if direction == "long":
            if g20 > 0.5:
                parts.append("G")  # strong gradient
            if g50 > 0.2:
                parts.append("S")  # slow SMA confirming
        else:
            if g20 < -0.5:
                parts.append("G")
            if g50 < -0.2:
                parts.append("S")

        return "".join(parts)
