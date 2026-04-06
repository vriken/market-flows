"""Core backtesting engine that runs strategies across tickers and dates."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict

import pandas as pd

from .strategies.base import BaseStrategy, Signal, Trade

# ── Timezone / market hours helpers (mirror orb_monday_range patterns) ─────

MARKET_OPEN = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()


def _to_ny(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with index in US/Eastern time."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    else:
        try:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        except TypeError:
            # Already tz-naive non-datetime index (e.g. daily data)
            pass
    return df


def _market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular market hours. Assumes Eastern-time index."""
    if not hasattr(df.index, "time"):
        return df  # daily data — no filtering needed
    times = df.index.time
    return df[(times >= MARKET_OPEN) & (times < MARKET_CLOSE)]


def _safe_tz_convert(df: pd.DataFrame) -> pd.DataFrame:
    """Safely handle timezone conversion, allowing already-naive data."""
    if df.empty:
        return df
    if hasattr(df.index, "tz") and df.index.tz is not None:
        return _to_ny(df)
    # Try to localise; if it fails (non-datetime index), just return as-is
    try:
        return _to_ny(df)
    except Exception:
        return df


# ── Engine ─────────────────────────────────────────────────────────────────


class BacktestEngine:
    """Run a strategy across tickers and date ranges, producing Trade results.

    Supports two call patterns:

    Pattern A (direct API):
        engine = BacktestEngine(strategy, regime_history=df)
        trades = engine.run(tickers_dict, start, end)

    Pattern B (run_backtest.py compatibility):
        engine = BacktestEngine(regime_history=regime_dict)
        trades = engine.run(strategy, price_data, start, end)

    Args:
        strategy: optional BaseStrategy subclass instance (can be passed
            to run() instead).
        regime_history: optional regime data.  Accepts either:
            - pd.DataFrame indexed by date with regime columns
            - dict[date, dict] mapping dates to regime classification dicts
    """

    def __init__(
        self,
        strategy: BaseStrategy | None = None,
        regime_history: pd.DataFrame | dict | None = None,
    ):
        self.strategy = strategy
        self._regime_df = self._normalise_regime_history(regime_history)
        self._trades: list[Trade] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        *args,
        position_size: float = 500.0,
        ko_buffer: float = 0.04,
        **kwargs,
    ) -> list[Trade]:
        """Run the strategy across all tickers for the given date range.

        Accepts two call signatures:

        1. run(tickers, start_date, end_date, ...)
           tickers: dict[str, dict] mapping ticker -> {"intraday": DF, "daily": DF}
                    or dict[str, DataFrame] mapping ticker -> daily DataFrame

        2. run(strategy, price_data, start_date, end_date, ...)
           strategy: BaseStrategy (used instead of self.strategy)
           price_data: dict[str, DataFrame] mapping ticker -> daily DataFrame
        """
        strategy, tickers, start_date, end_date = self._parse_run_args(args, kwargs)

        if isinstance(start_date, str):
            start_date = dt.date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = dt.date.fromisoformat(end_date)
        if hasattr(start_date, "date") and not isinstance(start_date, dt.date):
            start_date = start_date.date()
        if hasattr(end_date, "date") and not isinstance(end_date, dt.date):
            end_date = end_date.date()

        self._trades = []

        for ticker, data_bundle in tickers.items():
            self._run_ticker(
                ticker,
                data_bundle,
                strategy,
                start_date,
                end_date,
                position_size,
                ko_buffer,
            )

        return self._trades

    def results_df(self) -> pd.DataFrame:
        """Return completed trades as a DataFrame with regime columns."""
        if not self._trades:
            return pd.DataFrame()
        rows = [asdict(t) for t in self._trades]
        return pd.DataFrame(rows)

    def summary(self, group_by: str = "regime") -> pd.DataFrame:
        """Aggregate statistics grouped by a column.

        Args:
            group_by: one of "regime", "ticker", "direction", "outcome",
                      "quality_score", or any column name in results_df().
        """
        df = self.results_df()
        if df.empty:
            return pd.DataFrame()

        col_map = {
            "regime": "regime_composite",
            "ticker": "ticker",
            "direction": "direction",
            "outcome": "outcome",
            "quality": "quality_score",
        }
        col = col_map.get(group_by, group_by)
        if col not in df.columns:
            raise ValueError(f"Unknown group_by column: {group_by!r}. Available: {list(df.columns)}")

        rows = []
        for name, grp in df.groupby(col):
            total = len(grp)
            winners = grp[grp["pnl"] > 0]
            losers = grp[grp["pnl"] <= 0]
            rows.append({
                col: name,
                "trades": total,
                "wins": len(winners),
                "win_rate": len(winners) / total if total else 0,
                "avg_pnl": grp["pnl"].mean(),
                "total_pnl": grp["pnl"].sum(),
                "avg_winner": winners["pnl"].mean() if len(winners) else 0,
                "avg_loser": losers["pnl"].mean() if len(losers) else 0,
                "best_trade": grp["pnl"].max(),
                "worst_trade": grp["pnl"].min(),
                "avg_days_held": grp["days_held"].mean(),
                "profit_factor": (
                    abs(winners["pnl"].sum() / losers["pnl"].sum())
                    if len(losers) and losers["pnl"].sum() != 0
                    else float("inf") if len(winners) else 0
                ),
            })

        summary_df = pd.DataFrame(rows)
        return summary_df.sort_values("total_pnl", ascending=False).reset_index(drop=True)

    # ── Argument parsing ──────────────────────────────────────────────────

    def _parse_run_args(self, args, kwargs):
        """Disambiguate the two call signatures for run().

        Signature A: run(tickers_dict, start_date, end_date)
        Signature B: run(strategy, price_data_dict, start_date, end_date)
        """
        # Detect based on first positional argument type
        if len(args) >= 1 and isinstance(args[0], BaseStrategy):
            # Signature B: run(strategy, price_data, start, end)
            strategy = args[0]
            price_data = args[1] if len(args) > 1 else kwargs.get("price_data", {})
            start_date = args[2] if len(args) > 2 else kwargs.get("start_date")
            end_date = args[3] if len(args) > 3 else kwargs.get("end_date")

            # price_data is dict[str, DataFrame] — wrap each in a bundle
            tickers = {}
            for ticker, df in price_data.items():
                if isinstance(df, pd.DataFrame):
                    tickers[ticker] = {"daily": df, "intraday": None}
                elif isinstance(df, dict):
                    tickers[ticker] = df
                else:
                    continue
        elif len(args) >= 1 and isinstance(args[0], dict):
            # Signature A: run(tickers_dict, start_date, end_date)
            strategy = self.strategy
            raw_tickers = args[0]
            start_date = args[1] if len(args) > 1 else kwargs.get("start_date")
            end_date = args[2] if len(args) > 2 else kwargs.get("end_date")

            # Normalise: accept dict[str, DataFrame] or dict[str, dict]
            tickers = {}
            for ticker, val in raw_tickers.items():
                if isinstance(val, pd.DataFrame):
                    tickers[ticker] = {"daily": val, "intraday": None}
                elif isinstance(val, dict):
                    tickers[ticker] = val
                else:
                    continue
        else:
            raise TypeError(
                "run() requires either (tickers_dict, start, end) or "
                "(strategy, price_data, start, end) as positional arguments."
            )

        if strategy is None:
            raise ValueError("No strategy provided. Pass it to __init__ or run().")

        return strategy, tickers, start_date, end_date

    # ── Internal ───────────────────────────────────────────────────────────

    def _run_ticker(
        self,
        ticker: str,
        data_bundle: dict,
        strategy: BaseStrategy,
        start_date: dt.date,
        end_date: dt.date,
        position_size: float,
        ko_buffer: float,
    ):
        """Run the strategy for a single ticker."""
        intraday_df = data_bundle.get("intraday")
        daily_df = data_bundle.get("daily")

        if strategy.requires_intraday and intraday_df is None:
            return
        if not strategy.requires_intraday and daily_df is None:
            return

        # Normalise intraday to NY market hours
        if intraday_df is not None and not intraday_df.empty:
            intraday_df = _market_hours(_safe_tz_convert(intraday_df))

        # Build date list
        if strategy.requires_intraday and intraday_df is not None and not intraday_df.empty:
            all_dates = sorted(set(intraday_df.index.date))
        elif daily_df is not None and not daily_df.empty:
            idx = daily_df.index
            if hasattr(idx, "date"):
                all_dates = sorted(set(
                    d.date() if hasattr(d, "date") and callable(d.date) else d
                    for d in idx
                ))
            else:
                all_dates = sorted(set(idx))
        else:
            return

        trade_dates = [d for d in all_dates if start_date <= d <= end_date]

        for trade_date in trade_dates:
            # Regime gating: skip if strategy matrix says SIT OUT
            if self._regime_blocks_trade(strategy.name, trade_date):
                continue

            data = {
                "intraday": intraday_df,
                "daily": daily_df,
                "ticker": ticker,
            }

            signals = strategy.generate_signals(data, trade_date)

            for signal in signals:
                trade = self._simulate_trade(
                    signal,
                    strategy,
                    intraday_df,
                    daily_df,
                    all_dates,
                    trade_date,
                    position_size,
                    ko_buffer,
                )
                if trade is not None:
                    self._trades.append(trade)

    def _simulate_trade(
        self,
        signal: Signal,
        strategy: BaseStrategy,
        intraday_df: pd.DataFrame | None,
        daily_df: pd.DataFrame | None,
        all_dates: list[dt.date],
        trade_date: dt.date,
        position_size: float,
        ko_buffer: float,
    ) -> Trade | None:
        """Simulate a single trade from signal to exit."""
        entry = signal.entry_price
        direction = signal.direction

        # KO level for leveraged product
        ko_level = entry * (1 - ko_buffer) if direction == "long" else entry * (1 + ko_buffer)

        # Determine max hold days from strategy metadata or default
        max_hold_days = signal.metadata.get("max_hold_days", 1)

        # Find subsequent dates for multi-day holds
        try:
            date_idx = all_dates.index(trade_date)
        except ValueError:
            return None

        sim_dates = all_dates[date_idx: date_idx + max_hold_days]
        if not sim_dates:
            return None

        # Collect bars to simulate through
        outcome = "close"
        exit_price = entry
        exit_time = None
        bars_total = 0
        days_held = 0

        for day_offset, sim_date in enumerate(sim_dates):
            days_held = day_offset + 1

            if strategy.requires_intraday and intraday_df is not None:
                day_bars = intraday_df[intraday_df.index.date == sim_date]
                # On entry day, skip bars before signal time
                if sim_date == trade_date and signal.time is not None:
                    day_bars = day_bars[day_bars.index.time > signal.time]
            elif daily_df is not None:
                idx = daily_df.index
                if hasattr(idx, "date") and hasattr(idx[0], "date") and callable(getattr(idx[0], "date", None)):
                    day_bars = daily_df[[
                        (d.date() if hasattr(d, "date") and callable(d.date) else d) == sim_date
                        for d in idx
                    ]]
                else:
                    day_bars = daily_df[daily_df.index == sim_date]
            else:
                continue

            if day_bars.empty:
                continue

            for ts, bar_row in day_bars.iterrows():
                bar = {
                    "Open": float(bar_row["Open"]),
                    "High": float(bar_row["High"]),
                    "Low": float(bar_row["Low"]),
                    "Close": float(bar_row["Close"]),
                    "Volume": float(bar_row.get("Volume", 0)),
                    "time": ts.time() if hasattr(ts, "time") and callable(ts.time) else None,
                    "date": sim_date,
                }

                # Check KO first (wick touches KO level)
                if (direction == "long" and bar["Low"] <= ko_level) or (direction == "short" and bar["High"] >= ko_level):
                    outcome = "ko"
                    exit_price = ko_level
                    exit_time = str(bar["time"]) if bar["time"] else ""
                    break

                # Then check strategy exit
                exit_decision = strategy.check_exit(signal, bar, bars_total, day_offset)
                bars_total += 1

                if exit_decision is not None and exit_decision.should_exit:
                    outcome = exit_decision.reason
                    exit_price = exit_decision.exit_price
                    exit_time = str(bar["time"]) if bar["time"] else ""
                    break
            else:
                # No exit triggered during this day; use day close as provisional exit
                exit_price = float(day_bars.iloc[-1]["Close"])
                last_ts = day_bars.index[-1]
                exit_time = str(last_ts.time()) if hasattr(last_ts, "time") and callable(last_ts.time) else ""
                continue  # proceed to next day if multi-day hold

            # If we broke out of the bar loop (KO or exit), stop day iteration
            break

        # Calculate P&L: leveraged product formula
        # pnl = position_size * (price_change_pct / ko_buffer), capped at -position_size
        if outcome == "ko":
            pnl = -position_size
        else:
            price_change_pct = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
            pnl = position_size * (price_change_pct / ko_buffer)
            pnl = max(pnl, -position_size)  # can't lose more than position

        pnl_pct = ((exit_price - entry) / entry * 100) if direction == "long" else (
            (entry - exit_price) / entry * 100
        )

        # Build regime tags
        regime_data = self._get_regime(trade_date)

        trade = Trade(
            ticker=signal.ticker,
            date=trade_date,
            strategy=strategy.name,
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            entry_time=signal.time,
            exit_time=exit_time,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            outcome=outcome,
            days_held=days_held,
            quality_score=signal.quality_score,
            quality_flags=signal.quality_flags,
            regime_composite=regime_data.get("regime_composite", regime_data.get("composite_label", "")),
            regime_volatility=regime_data.get("regime_volatility", regime_data.get("volatility", "")),
            regime_cycle=regime_data.get("regime_cycle", regime_data.get("cycle", "")),
            regime_risk=regime_data.get("regime_risk", regime_data.get("risk", "")),
            regime_monetary=regime_data.get("regime_monetary", regime_data.get("monetary", "")),
            metadata=signal.metadata,
        )
        return trade

    def _regime_blocks_trade(self, strategy_name: str, trade_date: dt.date) -> bool:
        """Check if the strategy matrix says SIT OUT for this date's regime."""
        regime_data = self._get_regime(trade_date)
        if not regime_data:
            return False

        from ..regime import STRATEGIES

        # Find matching strategy rules
        strat_rules = None
        # Map strategy class names to regime matrix names
        name_aliases = {
            "Momentum": "SMA Gradient Runner",
        }
        lookup_name = name_aliases.get(strategy_name, strategy_name)
        for s in STRATEGIES:
            if s["name"] == lookup_name:
                strat_rules = s["rules"]
                break
        if strat_rules is None:
            return False

        # Map regime_history columns to dimension names
        dim_col_map = {
            "Volatility": ("volatility_state", "regime_volatility"),
            "Cycle": ("cycle_state", "regime_cycle"),
            "Risk": ("risk_state", "regime_risk"),
            "Monetary": ("monetary_state", "regime_monetary"),
            "Credit": ("credit_state", "regime_credit"),
        }

        for dim_name, rule_map in strat_rules.items():
            col_names = dim_col_map.get(dim_name, ())
            state = None
            for col in col_names:
                state = regime_data.get(col)
                if state:
                    break
            if state and rule_map.get(state) == "SIT OUT":
                return True

        return False

    def _get_regime(self, date: dt.date) -> dict:
        """Look up regime classification for a given date."""
        if self._regime_df is not None and not self._regime_df.empty:
            # Normalize to match index type
            import pandas as pd
            lookup = pd.Timestamp(date)
            if hasattr(self._regime_df.index, 'tz') and self._regime_df.index.tz is not None:
                lookup = lookup.tz_localize(self._regime_df.index.tz)
            elif not hasattr(lookup, 'tz') or lookup.tz is not None:
                lookup = lookup.tz_localize(None) if hasattr(lookup, 'tz_localize') and lookup.tz is not None else lookup
            # DataFrame-based lookup
            if lookup in self._regime_df.index:
                row = self._regime_df.loc[lookup]
            else:
                prior = self._regime_df[self._regime_df.index <= lookup]
                if prior.empty:
                    return {}
                row = prior.iloc[-1]

            result = {}
            for col in row.index:
                result[col] = str(row[col])
            return result

        return {}

    @staticmethod
    def _normalise_regime_history(
        regime_history: pd.DataFrame | dict | None,
    ) -> pd.DataFrame | None:
        """Convert regime_history to a DataFrame regardless of input format.

        Accepts:
        - pd.DataFrame indexed by date
        - dict[date, dict] from run_backtest._build_regime_history()
        - None
        """
        if regime_history is None:
            return None

        if isinstance(regime_history, pd.DataFrame):
            return regime_history

        if isinstance(regime_history, dict):
            if not regime_history:
                return None
            records = []
            for d, regime_dict in regime_history.items():
                records.append({"date": d, **regime_dict})
            df = pd.DataFrame(records).set_index("date")
            df.index.name = "date"
            return df

        return None
