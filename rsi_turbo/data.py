import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "data"

US = ["AMZN", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "NFLX", "AMD", "INTC",
      "JPM", "BAC", "GE", "BA", "XOM", "CVX", "JNJ", "PFE", "KO", "WMT", "HD", "CAT",
      "DIS", "NKE", "SPY", "QQQ"]
SE = ["VOLV-B.ST", "ERIC-B.ST", "INVE-B.ST", "ATCO-A.ST", "SAND.ST", "HM-B.ST",
      "SEB-A.ST", "SWED-A.ST", "AZN.ST", "ABB.ST"]
UNIVERSE = US + SE


def _cached(key, fetch, refresh=False):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}_{date.today():%Y%m%d}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    df = fetch()
    df.to_pickle(path)
    return df


def _normalise(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna(subset=["close"])


def _drop_inconsistent_ohlc(df, tolerance=0.001):
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    consistent = (df.high >= body_top * (1 - tolerance)) & (df.low <= body_bottom * (1 + tolerance))
    return df[consistent]


def daily(ticker, refresh=False):
    return _drop_inconsistent_ohlc(_cached(f"{ticker}_1d", lambda: _normalise(
        yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)), refresh))


def recent_daily(tickers, period="2y", chunk=100):
    """Uncached batch downloads, including today's bar while the market is open."""
    frames = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        raw = yf.download(batch, period=period, interval="1d", auto_adjust=True, progress=False, group_by="ticker",
                          threads=True)
        for ticker in batch:
            if ticker not in raw.columns.get_level_values(0):
                continue
            df = raw[ticker].copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["close"])
            if len(df):
                frames[ticker] = _drop_inconsistent_ohlc(df)
    for ticker in (t for t in tickers if t not in frames):
        single = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if len(single):
            frames[ticker] = _drop_inconsistent_ohlc(_normalise(single))
    return frames


def index_universe():
    """Today's S&P 500 and OMX Stockholm 30 members as Yahoo tickers, from Wikipedia, cached per day."""
    path = CACHE_DIR / f"index_universe_{date.today():%Y%m%d}.csv"
    if path.exists():
        return pd.read_csv(path).ticker.tolist()
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    def get(url):
        return pd.read_html(io.StringIO(requests.get(url, headers=headers, timeout=30).text))
    sp500 = get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0].Symbol.str.replace(".", "-")
    omx = get("https://en.wikipedia.org/wiki/OMX_Stockholm_30")[1].Ticker.str.strip().str.replace(" ", "-")
    omx = omx.where(omx.str.endswith(".ST"), omx + ".ST")
    tickers = list(dict.fromkeys([*sp500, *omx]))
    CACHE_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
    return tickers


def intraday_5m(ticker):
    bars = _cached(f"{ticker}_5m", lambda: _normalise(
        yf.download(ticker, period="60d", interval="5m", auto_adjust=True, progress=False, prepost=False)))
    local_day = bars.index.tz_localize(None).normalize()
    return bars[local_day < pd.Timestamp(date.today())]
