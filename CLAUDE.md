# market-flows

Personal trading toolkit — weekly market flow analysis, backtesting engine, and TradingView indicator suite.

## Project Structure

```
market_flows/
  cli.py              # CLI entry point (`market-flows` command)
  regime.py            # VIX regime classification (Normal/Elevated/Crisis)
  breadth.py           # Market breadth indicators
  sentiment.py         # Fear & Greed, put/call ratios
  cot.py               # COT positioning data
  etf.py               # ETF sector flow analysis
  dashboard.py         # HTML dashboard generation
  config.py            # Configuration and constants
  cache.py             # Data caching utilities
  http.py              # HTTP request helpers
  external.py          # External data source integrations
  backtest/
    engine.py          # Core backtesting engine (trade simulation, regime gating, KO mechanics)
    data.py            # Price data fetching and caching (yfinance)
    report.py          # HTML report generation for backtest results
    run_backtest.py    # CLI runner for backtests
    regime_history.py  # Historical regime data for backtests
    strategies/
      base.py          # Signal, Exit, Trade dataclasses + BaseStrategy ABC
      orb.py           # ORB (Opening Range Breakout) strategy
      fvg.py           # FVG (Fair Value Gap) fill strategy
      pdhl.py          # PDHL (Previous Day High/Low) breakout strategy
      vwap_reversion.py # VWAP mean reversion strategy
      momentum.py      # SMA Stack Momentum (gradient runner) strategy
tests/
  test_backtest.py     # Backtest engine and strategy tests
```

## Setup

```bash
cd market-flows
pip install -e ".[dev]"
```

## Running Backtests

```bash
# All strategies, default tickers, from 2025-01-01
python -m market_flows.backtest.run_backtest --strategy all --start 2025-01-01 --output html --output-path data/backtest/report.html

# Specific strategy
python -m market_flows.backtest.run_backtest --strategy fvg --start 2025-01-01 --tickers AAPL MSFT

# CSV export
python -m market_flows.backtest.run_backtest --strategy all --start 2025-01-01 --output csv --output-path data/backtest/results
```

## Tests

```bash
cd market-flows && python -m pytest tests/
```

## TradingView MCP Integration

This project has 5 Pine Script indicators deployed on TradingView that mirror the backtest strategies. The TradingView MCP server (`tradingview` in `~/.claude/.mcp.json`) allows reading chart data and indicator output programmatically.

### Pine Script Indicators

| Indicator | Strategies Covered | Key Output |
|-----------|-------------------|------------|
| ORB + Monday Range + Volume | ORB Breakout | Labels: `LONG/SHORT [n/4] flags`, table: VIX regime |
| PDHL Breakout | PDHL | Labels: `LONG PDH/SHORT PDL [n/3] flags`, risk in 1R |
| FVG Fill Strategy | FVG | Labels: `LONG/SHORT [n/3] flags`, FVG zone prices |
| SMA Stack Momentum | Momentum | Labels: `LONG/SHORT [n/3] G S V`, table: SMA stack state |
| VWAP Mean Reversion | VWAP | Labels: `LONG/SHORT [n/3] flags`, target: VWAP |

### Running a Full Instrument Scan

To scan instruments for trade signals using TradingView MCP:

#### 1. Check regime first
```
chart_get_state → get current chart info
data_get_pine_tables → read VIX regime from any indicator table
```
- VIX < 20: all strategies active
- VIX 20-25: ORB/VWAP/FVG show "SIT OUT" — only Momentum and PDHL are tradeable
- VIX > 25: Crisis — most strategies blocked

#### 2. For each instrument, scan both timeframes

**Daily (3-month context):**
```
chart_set_timeframe → "D"
chart_set_symbol → "EXCHANGE:TICKER"
sleep 2-3s (wait for indicators to render)
data_get_ohlcv summary=true → 3-month price summary
data_get_pine_tables → SMA stack state (BULL/BEAR/MIXED), VIX regime
data_get_pine_labels study_filter="Momentum" → active momentum signals
```

**5-minute (intraday signals):**
```
chart_set_timeframe → "5"
sleep 2-3s
data_get_pine_labels → all indicator signals (or use study_filter per indicator)
data_get_pine_tables → regime + stack confirmation
quote_get → current price
```

#### 3. Reading indicator signals

Pine labels follow this format:
- **Entry**: `"LONG [2/3] G S\nGrad: 0.8\nVol: 2.1x"` — direction, quality score, flags
- **Hold**: `"HOLD d3 2.5%\nGrad: 0.8"` — days held, unrealized P&L
- **Exit**: `"EXIT tmrw -0.5%"` or `"TP HIT +2.1%"` or `"STOP -1.3%"`

Quality flags per strategy:
- **ORB** [0-4]: T=trend, M=monday, B=breakout size, V=volume, G=gradient
- **PDHL** [0-3]: V=volume, W=wide range, T=trend
- **FVG** [0-3]: T=trend, S=size, V=volume
- **Momentum** [0-3]: G=gradient, S=stack alignment, V=volume
- **VWAP** [0-3]: T=trend, D=distance, V=volume

Pine tables show:
- VIX regime: `"VIX 24.2"` + `"SIT OUT"` or `"ACTIVE"`
- SMA stack: `"SMA 20/50/100/200"` + `"BULL STACK"` / `"BEAR STACK"` / `"MIXED"`

#### 4. Indicator visibility

All 5 indicators must be **visible** on the chart to return data. If `data_get_pine_labels` returns empty, toggle visibility:
```
indicator_toggle_visibility entity_id="<id>" visible=true
```
Get entity IDs from `chart_get_state`.

### Symbol Exchange Mapping

| Market | Exchange Prefix | Examples |
|--------|----------------|---------|
| Stockholm | OMXSTO | OMXSTO:ERIC_B, OMXSTO:EQT, OMXSTO:VOLV_B |
| Copenhagen | OMXCOP | OMXCOP:NOVO_B |
| Frankfurt | XETR | XETR:SAP, XETR:SIE |
| NASDAQ | NASDAQ | NASDAQ:TSLA, NASDAQ:META, NASDAQ:PLTR |
| NYSE | NYSE | NYSE:JPM, NYSE:KLAR |
| Crypto | BITSTAMP | BITSTAMP:BTCUSD |
| Gold | OANDA | OANDA:XAUUSD |
| Oil | NYMEX | NYMEX:CL1! |
| Silver | COMEX | COMEX:SI1! |
| Copper | COMEX | COMEX:HG1! |
| Platinum | NYMEX | NYMEX:PL1! |

Use `symbol_search` if unsure about the correct exchange prefix.

## Trading Philosophy

- **React, don't predict** — use indicators for awareness, enter when signals confirm
- **Regime first** — check VIX before looking at individual signals
- **Quality over quantity** — prefer [3/3] signals over [1/3]
- **Stack alignment** — BULL STACK + long signal = highest conviction; avoid counter-trend

## Backtest CI

Weekly backtest runs via GitHub Actions (`.github/workflows/backtest.yaml`):
- Scheduled Saturday 10 AM UTC
- Results published to gh-pages branch
- Week-over-week tracking via `data/backtest/history.jsonl`
- Manual trigger: `gh workflow run backtest.yaml -f strategy=all -f start_date=2025-01-01`

## Conventions

- Python 3.11+, type hints (PEP 585), `str | None` not `Optional[str]`
- Logging: `logger = logging.getLogger(__name__)`, lazy formatting
- Linting: `ruff check`, line length 120
- Tests: `pytest`, test file mirrors source structure
