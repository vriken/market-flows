"""CFTC Commitment of Traders data fetching, backfill, and percentiles."""

import io
import logging
import zipfile
from datetime import UTC, datetime

import pandas as pd
import requests

from .config import COT_CONTRACTS, DATA_DIR
from .http import get_session

logger = logging.getLogger(__name__)


def _fetch_cftc_zip(report_type, year=None):
    """Download a CFTC annual zip and return a DataFrame with headers."""
    if year is None:
        year = datetime.now(UTC).year
    if report_type == "disagg":
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    else:
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"

    r = get_session().get(url, timeout=60)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, low_memory=False)

    df.columns = df.columns.str.strip()

    date_col = next(
        (c for c in df.columns if "Report_Date" in c and "YYYY" in c), None
    )
    if date_col:
        df["date"] = pd.to_datetime(df[date_col])
    else:
        date_col = next(c for c in df.columns if "date" in c.lower())
        df["date"] = pd.to_datetime(df[date_col], format="mixed")

    return df


def _resolve_columns(df, report_type):
    """Return (name_col, oi_col, long_col, short_col, trader_label) for a report type."""
    name_col = next(c for c in df.columns if "Market_and_Exchange" in c)
    oi_col = next(c for c in df.columns if "Open_Interest_All" in c)

    if report_type == "disagg":
        long_col = next(
            c for c in df.columns
            if "M_Money" in c and "Long" in c and "All" in c
            and "Pct" not in c and "Change" not in c and "Spread" not in c
        )
        short_col = next(
            c for c in df.columns
            if "M_Money" in c and "Short" in c and "All" in c
            and "Pct" not in c and "Change" not in c and "Spread" not in c
        )
        trader_label = "Managed Money"
    else:
        long_col = next(
            c for c in df.columns
            if "Lev_Money" in c and "Long" in c and "All" in c
            and "Pct" not in c and "Change" not in c and "Spread" not in c
        )
        short_col = next(
            c for c in df.columns
            if "Lev_Money" in c and "Short" in c and "All" in c
            and "Pct" not in c and "Change" not in c and "Spread" not in c
        )
        trader_label = "Leveraged Money"

    return name_col, oi_col, long_col, short_col, trader_label


def _extract_contract_rows(df, pattern, report_type):
    """Extract standardized rows for one contract from a raw CFTC dataframe."""
    name_col, oi_col, long_col, short_col, trader_label = _resolve_columns(df, report_type)

    mask = df[name_col].str.upper().str.contains(pattern, na=False)
    matched = df.loc[mask].copy()
    if matched.empty:
        return pd.DataFrame()

    long_vals = matched[long_col].astype(int)
    short_vals = matched[short_col].astype(int)
    oi_vals = matched[oi_col].astype(int)
    net_vals = long_vals - short_vals

    result = pd.DataFrame({
        "date": matched["date"].values,
        "long_positions": long_vals.values,
        "short_positions": short_vals.values,
        "net_position": net_vals.values,
        "open_interest": oi_vals.values,
        "trader_type": trader_label,
    })
    result["net_pct_oi"] = result["net_position"] / result["open_interest"] * 100
    result.loc[result["open_interest"] == 0, "net_pct_oi"] = 0.0

    return result


def backfill_cot_history(start_year=2021):
    """Download CFTC data from start_year through current year and save to parquet."""
    current_year = datetime.now(UTC).year
    years = range(start_year, current_year + 1)

    all_records = []
    for year in years:
        for report_type in ("disagg", "fin"):
            logger.info("Fetching %s %d", report_type, year)
            try:
                df = _fetch_cftc_zip(report_type, year)
            except requests.RequestException as e:
                logger.warning("Skipped %s %d: %s", report_type, year, e)
                continue

            for pattern, label, rtype in COT_CONTRACTS:
                if rtype != report_type:
                    continue
                rows = _extract_contract_rows(df, pattern, rtype)
                if rows.empty:
                    continue
                rows["contract"] = label
                rows["report_type"] = rtype
                all_records.append(rows)

    if not all_records:
        logger.warning("No COT data fetched during backfill")
        return

    history = pd.concat(all_records, ignore_index=True)
    history = history.drop_duplicates(subset=["date", "contract", "report_type"])
    history = history.sort_values(["contract", "date"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "cot_history.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    history.to_parquet(tmp, index=False)
    tmp.rename(out)

    span = f"{history['date'].min().date()} to {history['date'].max().date()}"
    logger.info("Saved %d rows to %s (%s)", len(history), out, span)


def update_cot_history():
    """Fetch current year's data and merge into existing parquet history."""
    parquet_path = DATA_DIR / "cot_history.parquet"
    existing = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    current_year = datetime.now(UTC).year
    new_records = []
    for report_type in ("disagg", "fin"):
        logger.info("Fetching %s %d", report_type, current_year)
        try:
            df = _fetch_cftc_zip(report_type, current_year)
        except requests.RequestException as e:
            logger.warning("Skipped %s %d: %s", report_type, current_year, e)
            continue

        for pattern, label, rtype in COT_CONTRACTS:
            if rtype != report_type:
                continue
            rows = _extract_contract_rows(df, pattern, rtype)
            if rows.empty:
                continue
            rows["contract"] = label
            rows["report_type"] = rtype
            new_records.append(rows)

    if not new_records:
        logger.warning("No new COT data fetched")
        return

    new_data = pd.concat(new_records, ignore_index=True)

    combined = pd.concat([existing, new_data], ignore_index=True) if not existing.empty else new_data

    combined = combined.drop_duplicates(subset=["date", "contract", "report_type"])
    combined = combined.sort_values(["contract", "date"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "cot_history.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.rename(out)

    span = f"{combined['date'].min().date()} to {combined['date'].max().date()}"
    logger.info("Updated %s: %d rows (%s)", out, len(combined), span)


def _rank_percentile(series, value):
    """Rank-based percentile: % of historical values strictly below current."""
    below = (series < value).sum()
    return below / len(series) * 100 if len(series) > 0 else 50.0


def fetch_cot(filter_contracts=None, use_history=True):
    """Fetch COT data and return summary rows.

    If use_history=True and parquet exists, uses 5-year history for percentiles.
    Otherwise falls back to current-year-only (original behavior).
    """
    contracts = COT_CONTRACTS
    if filter_contracts:
        lc = [f.lower() for f in filter_contracts]
        contracts = [
            c for c in contracts
            if any(term in c[1].lower() for term in lc)
        ]
        if not contracts:
            logger.warning("No matching COT contracts for filter: %s", filter_contracts)
            return []

    # Load historical data if available
    parquet_path = DATA_DIR / "cot_history.parquet"
    history = None
    if use_history and parquet_path.exists():
        history = pd.read_parquet(parquet_path)

    need_disagg = any(t == "disagg" for _, _, t in contracts)
    need_fin = any(t == "fin" for _, _, t in contracts)

    dfs = {}
    logger.info("Fetching COT data from CFTC")
    if need_disagg:
        dfs["disagg"] = _fetch_cftc_zip("disagg")
    if need_fin:
        dfs["fin"] = _fetch_cftc_zip("fin")

    rows = []
    for pattern, label, rtype in contracts:
        df = dfs.get(rtype)
        if df is None:
            continue

        name_col, oi_col, long_col, short_col, trader_label = _resolve_columns(df, rtype)

        mask = df[name_col].str.upper().str.contains(pattern, na=False)
        all_rows = df.loc[mask].sort_values("date", ascending=False)
        if all_rows.empty:
            continue

        latest = all_rows.iloc[0]
        net = int(latest[long_col]) - int(latest[short_col])
        oi = int(latest[oi_col])
        net_pct = net / oi * 100 if oi else 0

        chg = None
        if len(all_rows) >= 2:
            prev = all_rows.iloc[1]
            prev_net = int(prev[long_col]) - int(prev[short_col])
            chg = net - prev_net

        # Percentile and z-score: use 5-year history if available, else YTD
        if history is not None and label in history["contract"].values:
            hist_nets = history.loc[history["contract"] == label, "net_position"]
            percentile = _rank_percentile(hist_nets, net)
            percentile_label = "5Y %ile"
            weeks = len(hist_nets)
            std = hist_nets.std()
            zscore = (net - hist_nets.mean()) / std if std > 0 else 0.0
        else:
            all_nets = (
                all_rows[long_col].astype(int) - all_rows[short_col].astype(int)
            )
            percentile = _rank_percentile(all_nets, net)
            percentile_label = "YTD %ile"
            weeks = len(all_nets)
            std = all_nets.std()
            zscore = (net - all_nets.mean()) / std if std > 0 else 0.0

        rows.append({
            "contract": label,
            "net": net,
            "net_pct_oi": net_pct,
            "change": chg,
            "percentile": percentile,
            "percentile_label": percentile_label,
            "zscore": round(zscore, 2),
            "weeks": weeks,
            "date": latest["date"].strftime("%Y-%m-%d"),
            "trader_type": trader_label,
        })

    return rows


def print_cot(rows):
    if not rows:
        print("  No COT data found.")
        return

    print(f"  Report date: {rows[0]['date']}\n")

    ptile_label = rows[0].get("percentile_label", "YTD %ile")

    disagg = [r for r in rows if r["trader_type"] == "Managed Money"]
    fin = [r for r in rows if r["trader_type"] == "Leveraged Money"]

    for label, group in [("Commodities (Managed Money)", disagg),
                         ("Financials (Leveraged Money)", fin)]:
        if not group:
            continue
        print(f"  {label}")
        print(f"  {'Contract':<16} {'Net Pos':>11} {'% of OI':>8} {'Wk Chg':>10} {ptile_label:>8} {'Z-Score':>8}  Signal")
        print(f"  {'─'*16} {'─'*11} {'─'*8} {'─'*10} {'─'*8} {'─'*8}  {'─'*16}")

        for r in group:
            net_s = f"{r['net']:>+11,}"
            pct_s = f"{r['net_pct_oi']:>+7.1f}%"
            chg_s = f"{r['change']:>+10,}" if r["change"] is not None else f"{'—':>10}"
            ptile = f"{r['percentile']:>7.0f}%"
            zs = f"{r['zscore']:>+7.2f}" if r.get("zscore") is not None else f"{'—':>8}"

            inverse = r["contract"] == "VIX"

            sig = ""
            if r["percentile"] >= 90:
                sig = "!! Extreme low" if inverse else "!! Extreme high"
            elif r["percentile"] <= 10:
                sig = "!! Extreme high" if inverse else "!! Extreme low"
            elif r["change"] is not None:
                if r["change"] > 0 and r["net"] > 0:
                    sig = "▼ Adding Short" if inverse else "▲ Adding Long"
                elif r["change"] < 0 and r["net"] < 0:
                    sig = "▲ Adding Long" if inverse else "▼ Adding Short"
                elif r["change"] > 0:
                    sig = "↘ Trimming Longs" if inverse else "↗ Covering Shorts"
                elif r["change"] < 0:
                    sig = "↗ Covering Shorts" if inverse else "↘ Trimming Longs"

            print(f"  {r['contract']:<16} {net_s} {pct_s} {chg_s} {ptile} {zs}  {sig}")
        print()
