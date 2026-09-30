from itertools import combinations

import numpy as np
import pandas as pd

GRADIENT_SMAS = (20, 50, 100, 150, 200, 250, 300, 400, 500, 600)


def rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return 100 - 100 / (1 + gain / loss)


def combined_gradient(close):
    smas = np.column_stack([close.rolling(p).mean().to_numpy() for p in GRADIENT_SMAS])
    valid = ~np.isnan(smas)

    bull = np.zeros(len(close))
    bear = np.zeros(len(close))
    for i, j in combinations(range(len(GRADIENT_SMAS)), 2):
        both = valid[:, i] & valid[:, j]
        shorter_above = smas[:, i] > smas[:, j]
        bull += both & shorter_above
        bear += both & ~shorter_above
    pairs = bull + bear
    stack = np.divide(bull - bear, pairs, out=np.zeros_like(pairs), where=pairs > 0)

    weights = 1 / np.array(GRADIENT_SMAS)
    side = np.where(close.to_numpy()[:, None] > smas, 1.0, -1.0)
    num = (side * weights * valid).sum(axis=1)
    den = (weights * valid).sum(axis=1)
    price_pos = np.divide(num, den, out=np.zeros_like(den), where=den > 0)

    return pd.Series((stack + price_pos) / 2, index=close.index)


def prior_value(series, dates):
    series = series.dropna()
    idx = series.index.searchsorted(dates, side="left") - 1
    values = series.to_numpy()[np.clip(idx, 0, None)]
    return np.where(idx >= 0, values, np.nan)
