"""ORB (Opening Range Breakout) strategy.

Ported from orb_monday_range.py — the core logic for:
- Computing the 5-min opening range (high/low of first N minutes)
- Detecting no-wick breakout/breakdown candles
- Quality scoring: T (trend), M (Monday range), B (breakout size), V (volume spike)
- Stop: price reverses through ORB range
- Exit: EOD or runner hold (gradient-based)
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .base import BaseStrategy, Exit, Signal

# ── Constants ──────────────────────────────────────────────────────────────

MARKET_OPEN = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()

SMA_PERIODS = [20, 50, 150, 200]


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout with no-wick signal detection.

    Computes the ORB from the first ``orb_window_minutes`` of each trading day
    on 5-min intraday bars. A signal fires when the first candle whose *entire*
    body is beyond the ORB level appears (no-wick condition).

    Quality flags:
        T — SMA trend aligned with direction
        M — Monday range confluence (entry beyond Mon high/low)
        B — breakout size >= 1% beyond ORB level
        V — volume spike (signal candle volume >= 2x 20-period SMA)

    Exit on same-day close or ORB full-reversal stop (close through opposite
    side of ORB range).
    """

    name = "ORB Breakout"
    requires_intraday = True

    def __init__(
        self,
        orb_window_minutes: int = 5,
        signal_interval: str = "5min",
        use_runner: bool = False,
    ):
        self.orb_window_minutes = orb_window_minutes
        self.signal_interval = signal_interval
        self.use_runner = use_runner

        # Cached per-ticker computations (populated in generate_signals)
        self._orb_cache: dict[str, dict] = {}
        self._trend_cache: dict[str, dict] = {}
        self._monday_cache: dict[str, dict] = {}
        self._vol_sma_cache: dict[str, pd.Series] = {}

    # ── Signal generation ──────────────────────────────────────────────────

    def generate_signals(self, data: dict, date: dt.date) -> list[Signal]:
        intraday = data.get("intraday")
        daily = data.get("daily")
        ticker = data.get("ticker", "")

        if intraday is None or intraday.empty:
            return []

        # Compute ORB for this date
        orb = self._compute_orb_for_date(intraday, date)
        if orb is None:
            return []

        orb_high = orb["orb_high"]
        orb_low = orb["orb_low"]

        # Get bars after ORB window
        orb_end_time = (
            pd.Timestamp("09:30") + pd.Timedelta(minutes=self.orb_window_minutes)
        ).time()

        day_bars = intraday[intraday.index.date == date]
        post_orb = day_bars[day_bars.index.time >= orb_end_time]

        if post_orb.empty:
            return []

        # Optionally resample to signal interval
        if self.signal_interval != "5min":
            tf_minutes = {"1min": 1, "5min": 5, "10min": 10, "15min": 15, "30min": 30, "60min": 60}
            mins = tf_minutes.get(self.signal_interval, 5)
            if mins > 1:
                post_orb = post_orb.resample(f"{mins}min", closed="left", label="left").agg({
                    "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
                }).dropna(subset=["Close"])
                post_orb = post_orb[post_orb.index.time >= MARKET_OPEN]
                post_orb = post_orb[post_orb.index.time < MARKET_CLOSE]

        # Volume SMA for spike detection
        vol_sma = self._get_vol_sma(intraday, ticker)

        # Trend data from daily
        trend_label = "none"
        if daily is not None and not daily.empty:
            trend = self._get_trend(daily, ticker)
            t = trend.get(date, {"label": "none"})
            trend_label = t["label"]

        # Monday range
        mon_high, mon_low = None, None
        if daily is not None and not daily.empty:
            mon_lookup = self._get_monday_lookup(daily, ticker)
            mon = mon_lookup.get(date)
            if mon:
                mon_high, mon_low = mon

        # Scan for first no-wick breakout/breakdown
        for ts, row in post_orb.iterrows():
            direction = None
            # Bullish: entire candle above ORB high (Low > orb_high)
            if row["Low"] > orb_high:
                direction = "long"
            # Bearish: entire candle below ORB low (High < orb_low)
            elif row["High"] < orb_low:
                direction = "short"

            if direction is None:
                continue

            entry_price = float(row["Close"])

            # Stop: opposite side of ORB range
            stop_price = float(orb_low) if direction == "long" else float(orb_high)

            # Quality scoring
            flags: dict = {}
            score_chars = []

            # T: trend aligned
            trend_aligned = (
                (direction == "long" and trend_label in ("up", "strong_up"))
                or (direction == "short" and trend_label in ("down", "strong_down"))
            )
            flags["trend_aligned"] = trend_aligned
            flags["trend_label"] = trend_label
            if trend_aligned:
                score_chars.append("T")

            # M: Monday range confluence
            mon_confluence = False
            if mon_high is not None and mon_low is not None:
                if direction == "long" and entry_price > mon_high or direction == "short" and entry_price < mon_low:
                    mon_confluence = True
            flags["monday_confluence"] = mon_confluence
            if mon_confluence:
                score_chars.append("M")

            # B: breakout size >= 1%
            breakout_pct = abs(entry_price - (orb_high if direction == "long" else orb_low)) / entry_price * 100
            flags["breakout_pct"] = round(breakout_pct, 2)
            if breakout_pct >= 1.0:
                score_chars.append("B")

            # V: volume spike
            avg_vol = vol_sma.get(ts, row["Volume"]) if vol_sma is not None else row["Volume"]
            vol_ratio = row["Volume"] / avg_vol if avg_vol > 0 else 1.0
            flags["vol_ratio"] = round(vol_ratio, 2)
            if vol_ratio >= 2.0:
                score_chars.append("V")

            quality_score = "".join(score_chars)

            # Require at least one quality flag to enter
            if not score_chars:
                continue

            # 1.5R target
            risk = abs(entry_price - stop_price)
            target_price = entry_price + 1.5 * risk if direction == "long" else entry_price - 1.5 * risk

            signal = Signal(
                date=date,
                time=ts.time() if hasattr(ts, "time") else None,
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quality_score=quality_score,
                quality_flags=flags,
                metadata={
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "max_hold_days": 1,
                },
            )
            return [signal]  # only first signal per day

        return []

    # ── Exit logic ─────────────────────────────────────────────────────────

    def check_exit(
        self,
        signal: Signal,
        current_bar: dict,
        bars_since_entry: int,
        day_index: int,
    ) -> Exit | None:
        orb_high = signal.metadata.get("orb_high", 0)
        orb_low = signal.metadata.get("orb_low", 0)

        # Target hit (1.5R)
        target = signal.target_price
        if target is not None:
            if (signal.direction == "long" and current_bar["High"] >= target) or (signal.direction == "short" and current_bar["Low"] <= target):
                return Exit(
                    should_exit=True,
                    exit_price=float(target),
                    reason="target",
                    metadata={"trigger": "1.5r_target"},
                )

        # Full-reversal stop: close through entire ORB range
        if signal.direction == "long" and current_bar["Close"] < orb_low:
            return Exit(
                should_exit=True,
                exit_price=current_bar["Close"],
                reason="stop",
                metadata={"trigger": "full_reversal_below_orb_low"},
            )
        elif signal.direction == "short" and current_bar["Close"] > orb_high:
            return Exit(
                should_exit=True,
                exit_price=current_bar["Close"],
                reason="stop",
                metadata={"trigger": "full_reversal_above_orb_high"},
            )

        # EOD exit: if this is the last bar of the day (close to market close)
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

    def _compute_orb_for_date(
        self, intraday: pd.DataFrame, date: dt.date
    ) -> dict | None:
        """Compute ORB high/low for a given date from intraday data."""
        day_data = intraday[intraday.index.date == date]
        if day_data.empty:
            return None

        candles_needed = max(1, self.orb_window_minutes // 5)
        opening = day_data.iloc[:candles_needed]
        if opening.empty:
            return None

        return {
            "orb_open": float(opening["Open"].iloc[0]),
            "orb_high": float(opening["High"].max()),
            "orb_low": float(opening["Low"].min()),
        }

    def _get_vol_sma(self, intraday: pd.DataFrame, ticker: str) -> pd.Series:
        """20-period rolling volume SMA for spike detection."""
        cache_key = ticker
        if cache_key not in self._vol_sma_cache:
            self._vol_sma_cache[cache_key] = intraday["Volume"].rolling(20, min_periods=1).mean()
        return self._vol_sma_cache[cache_key]

    def _get_trend(self, daily: pd.DataFrame, ticker: str) -> dict:
        """Compute SMA trend labels per date (cached)."""
        if ticker in self._trend_cache:
            return self._trend_cache[ticker]

        df = daily.copy()
        for p in SMA_PERIODS:
            df[f"sma{p}"] = df["Close"].rolling(p).mean()

        trend: dict[dt.date, dict] = {}
        for i, row in df.iterrows():
            d = i.date() if hasattr(i, "date") else i
            available = [(p, row[f"sma{p}"]) for p in SMA_PERIODS if pd.notna(row[f"sma{p}"])]
            if len(available) < 2:
                trend[d] = {"label": "none", "above_count": 0, "total_smas": 0}
                continue
            above = sum(1 for _, v in available if row["Close"] > v)
            total = len(available)
            if above == total:
                label = "strong_up"
            elif above >= total * 2 / 3:
                label = "up"
            elif (total - above) == total:
                label = "strong_down"
            elif (total - above) >= total * 2 / 3:
                label = "down"
            else:
                label = "none"
            trend[d] = {"label": label, "above_count": above, "total_smas": total}

        self._trend_cache[ticker] = trend
        return trend

    def _get_monday_lookup(self, daily: pd.DataFrame, ticker: str) -> dict:
        """Build Monday range lookup: date -> (mon_high, mon_low)."""
        if ticker in self._monday_cache:
            return self._monday_cache[ticker]

        df = daily.copy()
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("America/New_York")

        # Group by ISO week, pick first trading day
        seen_weeks: set = set()
        monday_ranges = []
        for i, row in df.iterrows():
            d = i.date() if hasattr(i, "date") else i
            iso_key = d.isocalendar()[:2]
            if iso_key in seen_weeks:
                continue
            seen_weeks.add(iso_key)
            monday_ranges.append({
                "week_start": d,
                "mon_high": float(row["High"]),
                "mon_low": float(row["Low"]),
            })

        # Map every weekday of each week to (mon_high, mon_low)
        lookup: dict = {}
        for mon in monday_ranges:
            ws = mon["week_start"]
            iso_monday = ws - dt.timedelta(days=ws.weekday())
            for d in range(5):
                lookup[iso_monday + dt.timedelta(days=d)] = (mon["mon_high"], mon["mon_low"])

        self._monday_cache[ticker] = lookup
        return lookup
