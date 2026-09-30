"""How many features, and which groups, does the model need? Ranked on 2008–14, judged on 2015–26.

Needs results/ml_importance_by_year.csv from ml.py for the 2008–14 ranking.
"""
import numpy as np
import pandas as pd

from ml import (
    FEATURES,
    MARKET_FEATURES,
    PURGE,
    REGIME_CODES,
    RESULTS_DIR,
    STOCK_FEATURES,
    fit,
    load_dataset,
    market_regime,
    rank_ic,
)

SELECTION_YEARS = range(2008, 2015)
EVALUATION_YEARS = range(2015, 2027)
TOP_K = (1, 3, 5, 8, 12, 18)


def evaluate(df, features):
    folds, dip_rows = [], []
    for year in EVALUATION_YEARS:
        test = df[(df.date.dt.year == year) & df.fwd.notna()]
        if test.empty:
            continue
        train = df[(df.date < pd.Timestamp(year, 1, 1) - PURGE) & df.fwd.notna()]
        model, cuts = fit(train, features)
        pred = model.predict(test[features])
        regimes = market_regime(test).to_numpy()
        row = {"year": year, "ic": rank_ic(pred, test.fwd)}
        for name in np.unique(regimes):
            inside = regimes == name
            if inside.sum() >= 500:
                row[name] = rank_ic(pred[inside], test.fwd[inside])
        folds.append(row)
        dips = test.rsi_signal.to_numpy()
        dip_rows.append(pd.DataFrame({"fwd": test.fwd[dips], "top": pred[dips] >= cuts["dip_high_cut"],
                                      "bottom": pred[dips] < cuts["dip_low_cut"]}))
    folds = pd.DataFrame(folds)
    dips = pd.concat(dip_rows)
    summary = {"features": len(features), "mean_ic": folds.ic.mean(), "ic_positive_years": int((folds.ic > 0).sum()),
               "years": len(folds)}
    for name in sorted(c for c in folds.columns if c.startswith("SPY")):
        summary[f"ic {name}"] = folds[name].mean()
    summary["dips_all_5d_%"] = dips.fwd.mean() * 100
    summary["dips_top_third_5d_%"] = dips.fwd[dips.top].mean() * 100
    summary["dips_bottom_third_5d_%"] = dips.fwd[dips.bottom].mean() * 100
    return summary


def main():
    _, df, _ = load_dataset()
    by_year = pd.read_csv(RESULTS_DIR / "ml_importance_by_year.csv")
    ranking = (by_year[by_year.year.isin(SELECTION_YEARS)].groupby("feature").ic_drop.mean()
               .sort_values(ascending=False).index.tolist())
    print("Ranking from 2008–14 only:", ", ".join(ranking), "\n")

    configs = {f"top {k} (ranked on 2008–14)": ranking[:k] for k in TOP_K}
    configs["all 26"] = FEATURES
    configs["market fear only"] = MARKET_FEATURES
    configs["stock only"] = STOCK_FEATURES
    configs["market-flows regime only"] = list(REGIME_CODES)
    configs["stock + market fear"] = STOCK_FEATURES + MARKET_FEATURES
    configs["market fear + market-flows regime"] = MARKET_FEATURES + list(REGIME_CODES)

    rows = []
    for name, features in configs.items():
        print(f"  evaluating {name} ({len(features)} features)")
        rows.append({"config": name, **evaluate(df, features)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "ml_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n== Out-of-sample 2015–26: rank IC overall and by regime, and RSI<30 dips split by the model ==")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
