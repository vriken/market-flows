"""Watchlists and constants for market-flow tracking."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ETF_GROUPS = {
    "S&P 500 Sectors": {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLB": "Materials",
        "XLI": "Industrials",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication",
    },
    "Major Indices": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "IWM": "Russell 2000",
        "DIA": "Dow Jones",
        "EFA": "Intl Developed",
        "EEM": "Emerging Markets",
    },
    "Commodities & Crypto": {
        "GLD": "Gold",
        "SLV": "Silver",
        "CPER": "Copper",
        "PPLT": "Platinum",
        "USO": "WTI Crude",
        "BNO": "Brent Crude",
        "IBIT": "Bitcoin",
    },
    "Bonds": {
        "TLT": "20+ Yr Treasury",
        "IEF": "7-10 Yr Treasury",
        "HYG": "High Yield",
        "LQD": "Inv Grade Corp",
    },
    "Currency": {
        "UUP": "US Dollar",
    },
    "Leveraged (Bull/Bear Pairs)": {
        "TQQQ": "Nasdaq 3x Bull",
        "SQQQ": "Nasdaq 3x Bear",
        "SPXL": "S&P 500 3x Bull",
        "SPXU": "S&P 500 3x Bear",
    },
}

# Leveraged ETF pairs for bull/bear ratio calculation
# Each pair: (bull_ticker, bear_ticker, label)
LEVERAGE_PAIRS = [
    ("TQQQ", "SQQQ", "Nasdaq"),
    ("SPXL", "SPXU", "S&P 500"),
]

# VIX tickers for term structure
VIX_TICKERS = {
    "vix": "^VIX",
    "vix3m": "^VIX3M",
}

# Market ratios: (numerator, denominator, label, interpretation)
# interpretation: what a rising ratio means
MARKET_RATIOS = [
    ("GLD", "SLV", "Gold/Silver", "Defensive / monetary stress"),
    ("CPER", "GLD", "Copper/Gold", "Growth optimism"),
    ("XLY", "XLP", "Discretionary/Staples", "Risk appetite"),
    ("HYG", "LQD", "High Yield/Inv Grade", "Credit risk appetite"),
    ("IWM", "SPY", "Small Cap/Large Cap", "Breadth / risk-on"),
    ("EEM", "SPY", "Emerging/US", "Global risk appetite"),
]

# COT contracts: (search_pattern, display_name, report_type)
#   report_type: "disagg" = commodities, "fin" = financial futures
COT_CONTRACTS = [
    # Commodities (disaggregated report -- uses Managed Money)
    ("GOLD", "Gold", "disagg"),
    ("SILVER", "Silver", "disagg"),
    ("CRUDE OIL, LIGHT SWEET", "Crude Oil", "disagg"),
    ("CORN", "Corn", "disagg"),
    ("SOYBEANS", "Soybeans", "disagg"),
    ("WHEAT-SRW", "Wheat", "disagg"),
    ("COPPER", "Copper", "disagg"),
    ("PLATINUM", "Platinum", "disagg"),
    # Financial futures (TFF report -- uses Asset Managers / Leveraged Money)
    ("NASDAQ MINI", "Nasdaq 100", "fin"),
    ("RUSSELL E-MINI", "Russell 2000", "fin"),
    ("EURO FX - C", "EUR/USD", "fin"),
    ("JAPANESE YEN", "JPY", "fin"),
    ("BRITISH POUND", "GBP", "fin"),
    ("UST BOND", "US T-Bonds", "fin"),
    ("UST 10Y NOTE", "10Y T-Notes", "fin"),
    ("VIX FUTURES", "VIX", "fin"),
    ("BITCOIN -", "Bitcoin", "fin"),
]
