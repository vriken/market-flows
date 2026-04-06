"""VWAP Mean Reversion strategy.

Computes session VWAP with standard-deviation bands from 5-min intraday data.
Trades mean reversion when price touches and reverses from the outer bands.

Entry: price touches/pierces VWAP +/- 1.5 sigma band, then the next candle
       closes back inside the band (reversal confirmation).
Direction: Long at lower band, Short at upper band.
Stop: price moves 0.5 sigma beyond the band.
Target: VWAP (mean reversion target).
Exit: hit target, hit stop, or EOD.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Exit, Signal

MARKET_OPEN = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()

BAND_MULTIPLIER = 1.5   # sigma multiplier for entry bands
STOP_SIGMA = 0.5        # additional sigma beyond band for stop


class VWAPReversionStrategy(BaseStrategy):
    """Intraday VWAP mean-reversion strategy.

    Trades occur when price touches the VWAP +/- 1.5 sigma band and then
    reverses (candle closes back inside). Target is VWAP itself; stop is
    0.5 sigma past the band.
    """

    name = "VWAP Reversion"
    requires_intraday = True

    def __init__(self, band_sigma: float = 1.5, stop_sigma: float = 0.5):
        self.band_sigma = band_sigma
        self.stop_sigma = stop_sigma

    # ── Signal generation ──────────────────────────────────────────────────

    def generate_signals(self, data: dict, date: dt.date) -> list[Signal]:
        intraday = data.get("intraday")
        ticker = data.get("ticker", "")

        if intraday is None or intraday.empty:
            return []

        # Compute VWAP + bands for this date
        day_bars = intraday[intraday.index.date == date]
        if len(day_bars) < 10:  # need enough bars for meaningful VWAP
            return []

        vwap_data = self._compute_vwap_bands(day_bars)
        if vwap_data is None:
            return []

        # Skip the first 30 min (ORB zone — let volatility settle)
        skip_until = dt.time(10, 0)
        vwap_data = vwap_data[vwap_data.index.time >= skip_until]

        if len(vwap_data) < 3:
            return []

        signals: list[Signal] = []

        for idx in range(1, len(vwap_data)):
            prev = vwap_data.iloc[idx - 1]
            curr = vwap_data.iloc[idx]
            ts = vwap_data.index[idx]

            vwap = curr["vwap"]
            upper = curr["vwap_upper"]
            lower = curr["vwap_lower"]
            sigma = curr["sigma"]

            if np.isnan(vwap) or np.isnan(sigma) or sigma <= 0:
                continue

            # Long signal: previous bar pierced lower band, current bar closes back inside
            if prev["Low"] <= lower and curr["Close"] > lower:
                stop_price = lower - self.stop_sigma * sigma
                signal = Signal(
                    date=date,
                    time=ts.time() if hasattr(ts, "time") else None,
                    ticker=ticker,
                    direction="long",
                    entry_price=float(curr["Close"]),
                    stop_price=float(stop_price),
                    target_price=float(vwap),
                    quality_score="",
                    quality_flags={
                        "vwap": float(vwap),
                        "band_touched": "lower",
                        "sigma": float(sigma),
                    },
                    metadata={
                        "vwap_at_entry": float(vwap),
                        "sigma_at_entry": float(sigma),
                        "upper_band": float(upper),
                        "lower_band": float(lower),
                        "max_hold_days": 1,
                    },
                )
                signals.append(signal)
                break  # one signal per day

            # Short signal: previous bar pierced upper band, current bar closes back inside
            if prev["High"] >= upper and curr["Close"] < upper:
                stop_price = upper + self.stop_sigma * sigma
                signal = Signal(
                    date=date,
                    time=ts.time() if hasattr(ts, "time") else None,
                    ticker=ticker,
                    direction="short",
                    entry_price=float(curr["Close"]),
                    stop_price=float(stop_price),
                    target_price=float(vwap),
                    quality_score="",
                    quality_flags={
                        "vwap": float(vwap),
                        "band_touched": "upper",
                        "sigma": float(sigma),
                    },
                    metadata={
                        "vwap_at_entry": float(vwap),
                        "sigma_at_entry": float(sigma),
                        "upper_band": float(upper),
                        "lower_band": float(lower),
                        "max_hold_days": 1,
                    },
                )
                signals.append(signal)
                break  # one signal per day

        return signals

    # ── Exit logic ─────────────────────────────────────────────────────────

    def check_exit(
        self,
        signal: Signal,
        current_bar: dict,
        bars_since_entry: int,
        day_index: int,
    ) -> Optional[Exit]:
        target = signal.target_price
        stop = signal.stop_price

        # Target hit (mean reversion to VWAP)
        if target is not None:
            if signal.direction == "long" and current_bar["High"] >= target:
                return Exit(
                    should_exit=True,
                    exit_price=float(target),
                    reason="target",
                    metadata={"trigger": "vwap_target"},
                )
            elif signal.direction == "short" and current_bar["Low"] <= target:
                return Exit(
                    should_exit=True,
                    exit_price=float(target),
                    reason="target",
                    metadata={"trigger": "vwap_target"},
                )

        # Stop hit
        if signal.direction == "long" and current_bar["Low"] <= stop:
            return Exit(
                should_exit=True,
                exit_price=float(stop),
                reason="stop",
                metadata={"trigger": "below_lower_band_stop"},
            )
        elif signal.direction == "short" and current_bar["High"] >= stop:
            return Exit(
                should_exit=True,
                exit_price=float(stop),
                reason="stop",
                metadata={"trigger": "above_upper_band_stop"},
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

    # ── VWAP computation ───────────────────────────────────────────────────

    def _compute_vwap_bands(self, day_bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute session VWAP with sigma bands for a single day."""
        df = day_bars.copy()
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_vol = tp * df["Volume"]

        cum_vol = df["Volume"].cumsum().replace(0, np.nan)
        vwap = tp_vol.cumsum() / cum_vol

        # Variance: E[X^2] - E[X]^2
        variance = ((tp**2 * df["Volume"]).cumsum() / cum_vol) - vwap**2
        sigma = np.sqrt(variance.clip(lower=0))

        df["vwap"] = vwap
        df["sigma"] = sigma
        df["vwap_upper"] = vwap + self.band_sigma * sigma
        df["vwap_lower"] = vwap - self.band_sigma * sigma

        return df
