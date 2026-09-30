"""Walk-forward gradient boosting on stock-day features, with the market regime as an input.

Every regime feature is the previous trading day's value, so each prediction conditions on the regime
the trade is actually entered in; nothing about the coming year's regime is assumed. Each test year is
predicted by a model trained on all earlier years, so the first fold already contains the 2000–02 bear.
The leave-one-bear-out test trains on data after the episode too, so it measures how well the model
generalises to an unseen bear, not what it would have earned live.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

import swing
from data import UNIVERSE, daily
from indicators import prior_value, rsi
from regime import load_regime_history

HOLD = 5
FIRST_TEST_YEAR = 2008
PURGE = pd.Timedelta(days=10)
MIN_REGIME_ROWS = 500
HIGH_VIX = 22
RESULTS_DIR = Path(__file__).parent / "results"

REGIME_CODES = {
    "volatility_state": {"Low Vol": 0, "Normal": 1, "Elevated": 2, "Crisis": 3},
    "cycle_state": {"Expansion": 0, "Late Cycle": 1, "Contraction": 2, "Recovery": 3},
    "risk_state": {"Risk-On": 0, "Mixed": 1, "Risk-Off": 2},
    "credit_state": {"Complacent": 0, "Normal": 1, "Stress": 2},
}
STOCK_FEATURES = ["rsi", "rsi_prev", "rsi_chg5", "dist_sma20", "dist_sma50", "dist_sma200", "ret1", "ret5",
                  "ret20", "vol20", "vol_ratio", "drawdown252", "volume_ratio", "gradient", "stockholm"]
MARKET_FEATURES = ["vix", "vix_chg5", "vix_term", "spy_dist200", "spy_drawdown", "spy_ret20", "spy_rsi"]
FEATURES = STOCK_FEATURES + MARKET_FEATURES + list(REGIME_CODES)

BEAR_EPISODES = {
    "2008–09 financial crisis": ("2008-01-01", "2009-06-30"),
    "2011 euro crisis": ("2011-07-01", "2011-12-31"),
    "2015–16 selloff": ("2015-08-01", "2016-02-29"),
    "2018 Q4 selloff": ("2018-10-01", "2018-12-31"),
    "2020 covid crash": ("2020-02-15", "2020-06-30"),
    "2022 rate bear": ("2022-01-01", "2022-12-31"),
    "2025 tariff selloff": ("2025-02-15", "2025-05-31"),
}


def market_features(dates, vix_close):
    spy = daily("SPY").close
    vix3m = daily("^VIX3M").close
    series = {
        "vix_chg5": vix_close.pct_change(5),
        "vix_term": vix_close / vix3m,
        "spy_dist200": spy / spy.rolling(200).mean() - 1,
        "spy_drawdown": spy / spy.rolling(252, min_periods=20).max() - 1,
        "spy_ret20": spy.pct_change(20),
        "spy_rsi": rsi(spy),
    }
    return pd.DataFrame({name: prior_value(s, dates) for name, s in series.items()}, index=dates)


def symbol_rows(sym, vix_close):
    f = sym.f
    c = f.close
    logret = np.log(c).diff()
    vol20 = logret.rolling(20).std()
    rows = pd.DataFrame(index=f.index)
    rows["rsi"] = f.rsi
    rows["rsi_prev"] = f.rsi_prev
    rows["rsi_chg5"] = f.rsi - f.rsi.shift(5)
    rows["dist_sma20"] = c / c.rolling(20).mean() - 1
    rows["dist_sma50"] = c / f.sma50 - 1
    rows["dist_sma200"] = c / f.trend_sma - 1
    rows["ret1"] = c.pct_change()
    rows["ret5"] = c.pct_change(5)
    rows["ret20"] = c.pct_change(20)
    rows["vol20"] = vol20
    rows["vol_ratio"] = vol20 / logret.rolling(100).std()
    rows["drawdown252"] = c / f.high.rolling(252, min_periods=20).max() - 1
    rows["volume_ratio"] = f.volume / f.volume.replace(0, np.nan).rolling(20).mean()
    rows["gradient"] = f.grad
    rows["stockholm"] = float(sym.ticker.endswith(".ST"))
    rows["vix"] = f.vix_prev
    rows = rows.join(market_features(f.index, vix_close))
    for dim, codes in REGIME_CODES.items():
        rows[dim] = pd.Series(f[dim], index=f.index).map(codes).astype(float)
    rows["fwd"] = c.shift(-HOLD) / c - 1
    rows["rsi_signal"] = sym.setups[("rsi<30", 1)]
    rows["ticker"] = sym.ticker
    return rows.rename_axis("date").reset_index()


def fit(train, features=FEATURES):
    target = train.fwd.clip(*train.fwd.quantile([0.01, 0.99]))
    model = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=500,
                                          l2_regularization=1.0, random_state=0,
                                          categorical_features=[f in REGIME_CODES for f in features])
    model.fit(train[features], target)
    fitted = model.predict(train[features])
    dips = fitted[train.rsi_signal.to_numpy()]
    cuts = {"top_cut": np.quantile(fitted, 0.99), "bottom_cut": np.quantile(fitted, 0.01),
            "dip_low_cut": np.quantile(dips, 1 / 3), "dip_high_cut": np.quantile(dips, 2 / 3)}
    return model, cuts


def rank_ic(pred, outcome):
    return spearmanr(pred, outcome).statistic


def market_regime(rows):
    trend = np.where(rows.spy_dist200 > 0, "SPY above 200d", "SPY below 200d")
    fear = np.where(rows.vix >= HIGH_VIX, f"VIX ≥ {HIGH_VIX}", f"VIX < {HIGH_VIX}")
    return pd.Series(trend, index=rows.index) + ", " + pd.Series(fear, index=rows.index)


def permutation_drops(model, rows, rng):
    base = rank_ic(model.predict(rows[FEATURES]), rows.fwd)
    drops = {}
    for feature in FEATURES:
        shuffled = rows[FEATURES].copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        drops[feature] = base - rank_ic(model.predict(shuffled), rows.fwd)
    return base, drops


def walk_forward(df):
    df = df.copy()
    for col in ("pred", "top_cut", "bottom_cut", "dip_low_cut", "dip_high_cut"):
        df[col] = np.nan
    folds, importance, regime_importance = [], [], []
    for year in range(FIRST_TEST_YEAR, df.date.dt.year.max() + 1):
        test_mask = (df.date.dt.year == year) & df.fwd.notna()
        train = df[(df.date < pd.Timestamp(year, 1, 1) - PURGE) & df.fwd.notna()]
        model, cuts = fit(train)
        test = df[test_mask]
        pred = model.predict(test[FEATURES])
        df.loc[test_mask, "pred"] = pred
        for name, value in cuts.items():
            df.loc[test_mask, name] = value
        rng = np.random.default_rng(year)
        base, drops = permutation_drops(model, test, rng)
        regimes = market_regime(test)
        folds.append({"year": year, "train_rows": len(train), "test_rows": len(test), "rank_ic": base,
                      "share_below_200d": (test.spy_dist200 <= 0).mean(), "mean_vix": test.vix.mean()})
        importance += [{"year": year, "feature": f, "ic_drop": d} for f, d in drops.items()]
        for regime_name, idx in test.groupby(regimes).groups.items():
            if len(idx) < MIN_REGIME_ROWS:
                continue
            regime_base, regime_drops = permutation_drops(model, test.loc[idx], rng)
            regime_importance += [{"year": year, "regime": regime_name, "rows": len(idx), "rank_ic": regime_base,
                                   "feature": f, "ic_drop": d} for f, d in regime_drops.items()]
        print(f"  {year}: trained on {len(train):,} rows, rank IC {base:+.3f}")
    return df, pd.DataFrame(folds), pd.DataFrame(importance), pd.DataFrame(regime_importance)


def strategy_masks(df):
    oos = df.pred.notna()
    dip = df.rsi_signal & oos
    overbought = (df.rsi > 70) & (df.rsi_prev <= 70) & oos
    return {
        "own every stock (baseline)": (oos, 1),
        "RSI<30 rule": (dip, 1),
        "RSI<30, model top third": (dip & (df.pred >= df.dip_high_cut), 1),
        "RSI<30, model middle third": (dip & (df.pred < df.dip_high_cut) & (df.pred >= df.dip_low_cut), 1),
        "RSI<30, model bottom third": (dip & (df.pred < df.dip_low_cut), 1),
        "model top 1% of all days": (oos & (df.pred >= df.top_cut), 1),
        "SHORT every stock (baseline)": (oos, -1),
        "SHORT RSI>70 cross": (overbought, -1),
        "SHORT RSI>70 cross, model below median": (overbought & (df.pred < 0), -1),
        "SHORT model bottom 1% of all days": (oos & (df.pred <= df.bottom_cut), -1),
    }


def simulate(symbols, df, market):
    masks = strategy_masks(df)
    def ones(f):
        return np.ones(len(f), dtype=bool)
    rows = []
    for name, (mask, direction) in masks.items():
        parts = []
        for sym in symbols:
            sub = df[df.ticker == sym.ticker]
            key = (name, direction)
            sym.setups[key] = pd.Series(mask[sub.index].to_numpy(), index=sub.date).reindex(sym.f.index, fill_value=False).to_numpy()
            if (t := sym.trades(key, "none", ones, HOLD)) is not None:
                parts.append(t)
        trades = pd.concat(parts, ignore_index=True)
        trades["market"] = np.where(prior_value(market.spy_dist200, trades.entry_date) > 0, "bull", "bear")
        trades["period"] = np.where(trades.oos, "2015–26", "2008–14")
        for split, groups in (("all", [("2008–26", trades)]), ("period", trades.groupby("period")),
                              ("market", trades.groupby("market"))):
            for label, g in groups:
                rows.append({"strategy": name, "split": split, "slice": label,
                             **swing.summarise(g, symbols[0].args.leverages)})
    return pd.DataFrame(rows)


def bear_holdout(df):
    rows = []
    for episode, (start, end) in BEAR_EPISODES.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        labelled = df.fwd.notna()
        inside = labelled & (df.date >= start) & (df.date <= end)
        train = df[labelled & ((df.date < start - PURGE) | (df.date > end + PURGE))]
        model, cuts = fit(train)
        test = df[inside].copy()
        test["pred"] = model.predict(test[FEATURES])
        dips = test[test.rsi_signal]
        top_dips = dips[dips.pred >= cuts["dip_high_cut"]]
        low_dips = dips[dips.pred < cuts["dip_low_cut"]]
        rows.append({
            "episode": episode, "rank_ic": rank_ic(test.pred, test.fwd),
            "all_days_%": test.fwd.mean() * 100,
            "rsi30_n": len(dips), "rsi30_%": dips.fwd.mean() * 100,
            "top_third_n": len(top_dips), "top_third_%": top_dips.fwd.mean() * 100,
            "bottom_third_n": len(low_dips), "bottom_third_%": low_dips.fwd.mean() * 100,
            "model_top1%_n": int((test.pred >= cuts["top_cut"]).sum()),
            "model_top1%_%": test.fwd[test.pred >= cuts["top_cut"]].mean() * 100,
        })
    return pd.DataFrame(rows)


def realised_by_decile(df, feature):
    oos = df[df.pred.notna() & df[feature].notna()]
    buckets = pd.qcut(oos[feature], 10, duplicates="drop")
    g = oos.groupby(buckets, observed=True).fwd
    return pd.DataFrame({"range": [f"{b.left:.3g} … {b.right:.3g}" for b in g.mean().index],
                         "rows": g.size().to_numpy(), "mean_5d_%": (g.mean() * 100).round(3).to_numpy()})


def load_dataset():
    args = swing.parse_args(["--leverages", "3", "5", "10", "15", "20", "--holds", str(HOLD),
                             "--report-hold", str(HOLD)])
    vix_close, tbill = daily("^VIX").close, daily("^IRX").close
    regime = load_regime_history(args.market_flows, args.start, pd.Timestamp.today())
    symbols = [swing.Symbol(t, swing.features(daily(t), vix_close, tbill, regime, args), args) for t in UNIVERSE]
    df = pd.concat([symbol_rows(s, vix_close) for s in symbols], ignore_index=True)
    market = market_features(pd.DatetimeIndex(sorted(df.date.unique())), vix_close)
    return symbols, df, market


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    symbols, df, market = load_dataset()
    print(f"{len(df):,} stock-days, {len(FEATURES)} features, test years {FIRST_TEST_YEAR}–{df.date.dt.year.max()}\n")

    print("Walk-forward folds:")
    df, folds, importance, regime_importance = walk_forward(df)
    importance.to_csv(RESULTS_DIR / "ml_importance_by_year.csv", index=False)
    regime_importance.to_csv(RESULTS_DIR / "ml_regime_importance.csv", index=False)
    df.to_parquet(RESULTS_DIR / "ml_predictions.parquet")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)

    print(f"\n== 1. Prediction quality: mean rank IC {folds.rank_ic.mean():+.4f}, positive in "
          f"{(folds.rank_ic > 0).sum()} of {len(folds)} years ==")
    print(folds.round(4).to_string(index=False))

    ranked = (importance.groupby("feature").ic_drop.agg(["mean", "std", lambda x: (x > 0).mean()])
              .set_axis(["mean_ic_drop", "std", "share_of_years_helping"], axis=1)
              .sort_values("mean_ic_drop", ascending=False))
    ranked.to_csv(RESULTS_DIR / "ml_importance.csv")
    print("\n== 2. Feature importance (rank-IC lost when the feature is shuffled, averaged over years) ==")
    print(ranked.round(4).to_string())

    results = simulate(symbols, df, market)
    results.to_csv(RESULTS_DIR / "ml_strategies.csv", index=False)
    cols = ["strategy", "slice", "trades", "win_%", "mean_%", "excess_%", "t_excess", "turbo5_mean_%", "turbo5_ko_%",
            "turbo10_mean_%", "turbo10_ko_%", "turbo20_mean_%", "turbo20_ko_%"]
    for split in ("all", "period", "market"):
        print(f"\n== 3. Strategies, 5-day hold, split by {split} ==")
        print(results[results.split == split][cols].round(2).to_string(index=False))

    print("\n== 4. Leave-one-bear-out: train on everything except the episode, mean 5-day return ==")
    print(bear_holdout(df).round(3).to_string(index=False))

    print("\n== 5. Realised 5-day return by decile of the top features (all test years) ==")
    for feature in ranked.index[:6]:
        if feature in REGIME_CODES:
            g = df[df.pred.notna()].groupby(feature).fwd
            codes = {v: k for k, v in REGIME_CODES[feature].items()}
            table = pd.DataFrame({"state": [codes[int(i)] for i in g.mean().index], "rows": g.size().to_numpy(),
                                  "mean_5d_%": (g.mean() * 100).round(3).to_numpy()})
        else:
            table = realised_by_decile(df, feature)
        print(f"-- {feature}")
        print(table.to_string(index=False))

    print("\n== 6. Top 3 features each year (rank-IC lost when shuffled) ==")
    top3 = (importance.sort_values("ic_drop", ascending=False).groupby("year").head(3)
            .groupby("year").apply(lambda g: ", ".join(f"{f} {d:+.3f}" for f, d in zip(g.feature, g.ic_drop, strict=False)),
                                   include_groups=False).rename("top_3"))
    print(folds.set_index("year")[["rank_ic", "share_below_200d", "mean_vix"]].join(top3).round(3).to_string())

    print(f"\n== 7. Feature importance by market regime at entry (mean over years with ≥{MIN_REGIME_ROWS} rows) ==")
    per_regime = regime_importance.groupby(["regime", "feature"]).ic_drop.agg(["mean", lambda x: (x > 0).mean()])
    per_regime.columns = ["mean_ic_drop", "share_of_years_helping"]
    regime_ic = regime_importance.drop_duplicates(["regime", "year"]).groupby("regime").agg(
        years=("year", "size"), mean_rank_ic=("rank_ic", "mean"), ic_positive_years=("rank_ic", lambda x: (x > 0).sum()))
    print(regime_ic.round(3).to_string())
    pivot = per_regime.mean_ic_drop.unstack("regime")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    print("\nMean rank-IC lost when shuffled (higher = the model leans on it more in that regime):")
    print(pivot.round(4).to_string())
    print("\nShare of years in which the feature helped:")
    print(per_regime.share_of_years_helping.unstack("regime").loc[pivot.index].round(2).to_string())


if __name__ == "__main__":
    main()
