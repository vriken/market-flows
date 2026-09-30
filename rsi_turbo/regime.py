import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data import CACHE_DIR, daily

REGIME_DIMENSIONS = ("volatility_state", "cycle_state", "risk_state", "monetary_state", "credit_state",
                     "composite_label")
MONETARY_INPUT = "2YY=F"
MONETARY_MIN_HISTORY = 60


def _build(market_flows_repo, start, end):
    path = CACHE_DIR / f"regime_history_{date.today():%Y%m%d}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    sys.path.insert(0, str(Path(market_flows_repo).resolve()))
    from market_flows.backtest.regime_history import build_regime_history
    history = build_regime_history(start, end, force=True)
    CACHE_DIR.mkdir(exist_ok=True)
    history.to_csv(path)
    return history


def load_regime_history(market_flows_repo, start, end):
    """market-flows reports monetary 'Pause' when the 2Y yield is missing, so blank it until it exists."""
    history = _build(market_flows_repo, start, end)
    monetary_from = daily(MONETARY_INPUT).index[MONETARY_MIN_HISTORY - 1]
    history.loc[history.index < monetary_from, ["monetary_state", "composite_label"]] = None
    return history


def prior_state(states, dates):
    states = states.dropna()
    idx = states.index.searchsorted(dates, side="left") - 1
    values = states.to_numpy(dtype=object)[np.clip(idx, 0, None)]
    return np.where(idx >= 0, values, None)
