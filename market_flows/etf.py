"""ETF flow estimation via AUM snapshots."""

import json
from datetime import datetime, timezone

import yfinance as yf

from .config import DATA_DIR, ETF_GROUPS


def _load_history():
    p = DATA_DIR / "etf_aum.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    # Support both legacy single-snapshot and new array-of-snapshots format
    if data and isinstance(next(iter(data.values())), dict) and "aum" in next(iter(data.values())):
        # Legacy format: {ticker: {aum, price, date}} — convert to new format
        return {t: [v] for t, v in data.items()}
    return data


def _save_history(h):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "etf_aum.json").write_text(json.dumps(h, indent=2))


def _latest_snapshot(history, ticker):
    """Get the most recent snapshot for a ticker from array-format history."""
    snapshots = history.get(ticker, [])
    if not snapshots:
        return None
    return snapshots[-1]


def _previous_snapshot(history, ticker):
    """Get the second-most-recent snapshot for flow calculation."""
    snapshots = history.get(ticker, [])
    if len(snapshots) < 2:
        return None
    return snapshots[-2]


def fetch_etfs(filter_tickers=None, update=False):
    """Get current ETF data and estimate flows vs last saved snapshot."""
    hist = _load_history()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    all_known = {t: n for grp in ETF_GROUPS.values() for t, n in grp.items()}
    if filter_tickers:
        tickers = {}
        for t in filter_tickers:
            t_upper = t.upper()
            tickers[t_upper] = all_known.get(t_upper, t_upper)
    else:
        tickers = all_known

    print(f"  Fetching data for {len(tickers)} ETFs...")

    results = []
    snapshot = {}

    for tick, name in tickers.items():
        try:
            tkr = yf.Ticker(tick)
            info = tkr.info
            aum = info.get("totalAssets")
            price = info.get("previousClose") or info.get("regularMarketPrice")

            prices = tkr.history(period="5d")
            wk_ret = None
            if len(prices) >= 2:
                wk_ret = prices["Close"].iloc[-1] / prices["Close"].iloc[0] - 1

            # Estimate flow vs previous snapshot
            flow = None
            prev = _latest_snapshot(hist, tick)
            if prev and aum and prev.get("aum") and prev.get("price") and price:
                price_ret = price / prev["price"] - 1
                flow = aum - prev["aum"] * (1 + price_ret)

            if aum and price:
                snapshot[tick] = {"aum": aum, "price": price, "date": today}

            results.append({
                "ticker": tick,
                "name": name,
                "aum_b": aum / 1e9 if aum else None,
                "wk_ret": wk_ret,
                "flow_m": flow / 1e6 if flow else None,
            })
        except Exception:
            results.append({
                "ticker": tick, "name": name,
                "aum_b": None, "wk_ret": None, "flow_m": None,
            })

    if update:
        for t, s in snapshot.items():
            if t not in hist:
                hist[t] = []
            # Don't duplicate same-day snapshots
            if not hist[t] or hist[t][-1].get("date") != today:
                hist[t].append(s)
            else:
                hist[t][-1] = s
        _save_history(hist)
        print(f"  Saved AUM snapshot for {len(snapshot)} ETFs.\n")
    else:
        print()

    return results


def build_flow_history(min_snapshots=3):
    """Build cumulative flow history from saved AUM snapshots.

    Returns {has_data, series, message} where series is a list of
    {ticker, dates, cumulative_flows} for tickers with enough snapshots.
    """
    hist = _load_history()
    if not hist:
        return {"has_data": False, "series": [], "message": "No AUM history available yet."}

    # Filter to sector + major index tickers for readability
    relevant_groups = {"S&P 500 Sectors", "Major Indices"}
    relevant_tickers = {t for grp_name, grp in ETF_GROUPS.items()
                        if grp_name in relevant_groups for t in grp}

    series = []
    for ticker, snapshots in hist.items():
        if ticker not in relevant_tickers:
            continue
        if len(snapshots) < min_snapshots:
            continue

        dates = []
        cumulative_flows = []
        cumulative = 0.0
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            if not (prev.get("aum") and prev.get("price") and
                    curr.get("aum") and curr.get("price")):
                continue
            price_ret = curr["price"] / prev["price"] - 1
            flow = curr["aum"] - prev["aum"] * (1 + price_ret)
            cumulative += flow / 1e6  # convert to $M
            dates.append(curr.get("date", ""))
            cumulative_flows.append(round(cumulative, 1))

        if dates:
            series.append({
                "ticker": ticker,
                "dates": dates,
                "cumulative_flows": cumulative_flows,
            })

    if not series:
        return {
            "has_data": False,
            "series": [],
            "message": f"Need at least {min_snapshots} snapshots per ETF. Keep running the dashboard to accumulate data.",
        }

    return {"has_data": True, "series": series, "message": ""}


def print_etfs(results, filter_tickers=None):
    has_flow = any(r["flow_m"] is not None for r in results)

    if filter_tickers:
        print(f"\n  Custom ETFs")
        hdr = f"  {'Tick':<6} {'Name':<18} {'AUM ($B)':>9} {'Return':>8}"
        div = f"  {'─'*6} {'─'*18} {'─'*9} {'─'*8}"
        if has_flow:
            hdr += f" {'Est Flow ($M)':>14}"
            div += f" {'─'*14}"
        print(hdr)
        print(div)
        for r in sorted(results, key=lambda r: r["flow_m"] or 0, reverse=True):
            _print_etf_row(r, has_flow)
    else:
        for group, tickers in ETF_GROUPS.items():
            grp = [r for r in results if r["ticker"] in tickers]
            grp.sort(key=lambda r: r["flow_m"] or 0, reverse=True)

            print(f"\n  {group}")
            hdr = f"  {'Tick':<6} {'Name':<18} {'AUM ($B)':>9} {'Return':>8}"
            div = f"  {'─'*6} {'─'*18} {'─'*9} {'─'*8}"
            if has_flow:
                hdr += f" {'Est Flow ($M)':>14}"
                div += f" {'─'*14}"
            print(hdr)
            print(div)
            for r in grp:
                _print_etf_row(r, has_flow)


def _print_etf_row(r, has_flow):
    aum_s = f"{r['aum_b']:>9.1f}" if r["aum_b"] else f"{'—':>9}"
    ret_s = (
        f"{r['wk_ret']*100:>+7.1f}%"
        if r["wk_ret"] is not None else f"{'—':>8}"
    )
    line = f"  {r['ticker']:<6} {r['name']:<18} {aum_s} {ret_s}"
    if has_flow:
        fl_s = (
            f"{r['flow_m']:>+13.0f}M"
            if r["flow_m"] is not None else f"{'—':>14}"
        )
        line += f" {fl_s}"
    print(line)
