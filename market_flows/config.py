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
}

# COT contracts: (search_pattern, display_name, report_type)
#   report_type: "disagg" = commodities, "fin" = financial futures
COT_CONTRACTS = [
    # Commodities (disaggregated report -- uses Managed Money)
    ("GOLD", "Gold", "disagg"),
    ("SILVER", "Silver", "disagg"),
    ("CRUDE OIL, LIGHT SWEET", "Crude Oil", "disagg"),
    ("NATURAL GAS", "Natural Gas", "disagg"),
    ("CORN", "Corn", "disagg"),
    ("SOYBEANS", "Soybeans", "disagg"),
    ("WHEAT-SRW", "Wheat", "disagg"),
    ("COPPER", "Copper", "disagg"),
    ("PLATINUM", "Platinum", "disagg"),
    # Financial futures (TFF report -- uses Asset Managers / Leveraged Money)
    ("E-MINI S&P 500", "S&P 500", "fin"),
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
