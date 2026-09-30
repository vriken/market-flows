from datetime import date

import pandas as pd

import figures
import scan


def signal_log(**overrides):
    row = {c: "" for c in scan.SIGNAL_COLUMNS} | {
        "signal_date": "2026-09-28", "ticker": "BAC", "close": "55.47", "leverage": "20", "rule": "v2",
        "sell_on": "2026-10-05",
    }
    return pd.DataFrame([row | overrides])


def empty_journal():
    return pd.DataFrame(columns=scan.JOURNAL_COLUMNS)


def bought(journal, signals, ticker="BAC", day=date(2026, 9, 29), financing=49.1365, stop=0.0, fill=5.94, qty=170):
    return scan.log_trade(signals, journal, ticker, day, "T LONG TEST", financing, stop, fill, qty, 10, 10.0)[0]


def lows(ticker, values, start="2026-09-28"):
    index = pd.bdate_range(start, periods=len(values))
    return {ticker: pd.DataFrame({"low": values, "close": values}, index=index)}


def test_trade_links_to_recent_signal():
    journal, message = scan.log_trade(signal_log(), empty_journal(), "BAC", date(2026, 9, 29), "T LONG TEST",
                                      49.1365, 0, 5.94, 170, 10, 10.0)
    assert journal.loc[0, "rule"] == "v2"
    assert journal.loc[0, "signal_date"] == "2026-09-28"
    assert journal.loc[0, "fill_underlying"] == "55.08"
    assert journal.loc[0, "leverage"] == "9.3"
    assert "Linked" in message


def test_trade_without_signal_is_off_rule():
    journal = bought(empty_journal(), signal_log(), ticker="NVDA", financing=224.38, fill=9.13)
    assert journal.loc[0, "rule"] == "off-rule"
    assert journal.loc[0, "signal_date"] == ""


def test_close_trade_records_result():
    journal = scan.close_trade(bought(empty_journal(), signal_log()), 0, 6.534, date(2026, 10, 1))
    assert journal.loc[0, "result_pct"] == "10.0"
    assert journal.loc[0, "knocked_out"] == "no"
    assert not scan.open_trades(journal).any()


def test_turbo_knocked_out_when_low_reaches_financing():
    journal = scan.settle_trades(bought(empty_journal(), signal_log()), lows("BAC", [55.0, 54.0, 49.0, 52.0]))
    assert journal.loc[0, ["knocked_out", "result_pct", "closed_on"]].tolist() == ["yes", "-100.0", "2026-09-30"]


def test_mini_long_stop_pays_back_what_is_left():
    journal = bought(empty_journal(), signal_log(), financing=45.0, stop=50.0, fill=10.0)
    journal = scan.settle_trades(journal, lows("BAC", [56.0, 55.0, 49.5]))
    assert journal.loc[0, "knocked_out"] == "yes"
    assert journal.loc[0, "result_pct"] == "-50.0"


def test_held_trade_stays_due_after_rule_exit():
    signals = signal_log(result_stock_pct="2.0", result_turbo_pct="30.0", result_ko="no", sold_on="2026-10-01",
                         exit_reason="RSI back to 40")
    due = scan.exits_due(signals, bought(empty_journal(), signals), [], "2026-10-02", {"US"})
    assert due.ticker.tolist() == ["BAC"]
    assert bool(due.yours.iloc[0])
    assert due.why.iloc[0].startswith("rule exit 2026-10-01")


def test_paper_signal_due_when_rsi_recovers():
    due = scan.exits_due(signal_log(), empty_journal(), [{"ticker": "BAC", "rsi": 41.0}], "2026-09-30", {"US"})
    assert due.why.tolist() == ["RSI back to 41"]
    assert not due.yours.iloc[0]


def test_trade_on_revoked_signal_becomes_off_rule():
    journal = bought(empty_journal(), signal_log())
    journal = scan.unlink_revoked(journal, signal_log(rule="didn't hold"))
    assert journal.loc[0, "rule"] == "off-rule"


def test_scorecard_uses_your_result_where_closed():
    signals = signal_log(result_stock_pct="2.0", result_turbo_pct="30.0", result_ko="no", sold_on="2026-10-01")
    journal = scan.close_trade(bought(empty_journal(), signals), 0, 6.534, date(2026, 10, 1))
    assert scan.scorecard(signals)["avg_turbo_pct"] == 30.0
    assert scan.scorecard(signals, journal)["avg_turbo_pct"] == 10.0


def test_monthly_results_count_closed_trades_by_month():
    journal = bought(bought(empty_journal(), signal_log()), signal_log(), ticker="NVDA", financing=224.38, fill=9.13,
                     qty=137)
    journal = scan.close_trade(journal, 0, 6.534, date(2026, 10, 1))
    journal.loc[1, ["exit_price", "closed_on"]] = ["sold", "2026-09-29"]
    fig, totals = figures.monthly_results(journal)
    assert totals["trades"] == 1
    assert totals["unpriced"] == 1
    assert round(totals["pnl"], 2) == round(5.94 * 170 * 0.10, 2)
    assert fig is not None
