"""Regime classifier using data already fetched by the dashboard."""

from .config import REGIME_THRESHOLDS


def _classify_volatility(sentiment_data):
    """Classify volatility regime from VIX data.

    Returns (state, signals) where state is one of:
    Crisis, Elevated, Normal, Low Vol
    """
    vix = sentiment_data.get("vix")
    if not vix:
        return None, []

    t = REGIME_THRESHOLDS["volatility"]
    level = vix["vix"]
    ratio = vix["ratio"]
    signals = []

    if level >= t["crisis"]:
        state = "Crisis"
        signals.append(f"VIX {level:.0f} (crisis)")
    elif level >= t["elevated"]:
        state = "Elevated"
        signals.append(f"VIX {level:.0f} (elevated)")
    elif level <= t["low_vol"]:
        state = "Low Vol"
        signals.append(f"VIX {level:.0f} (low)")
    else:
        state = "Normal"
        signals.append(f"VIX {level:.0f}")

    if ratio > t["backwardation"]:
        signals.append("VIX backwardation")
        if state == "Normal":
            state = "Elevated"
    elif ratio < t["steep_contango"]:
        signals.append("Steep contango")

    return state, signals


def _classify_cycle(sentiment_data):
    """Classify business cycle regime from yield spreads and ratios.

    Returns (state, signals) where state is one of:
    Expansion, Late Cycle, Contraction, Recovery
    """
    t = REGIME_THRESHOLDS["cycle"]
    signals = []
    scores = {"Expansion": 0, "Late Cycle": 0, "Contraction": 0, "Recovery": 0}

    # Yield spreads
    yc = sentiment_data.get("yield_curve")
    if yc and yc.get("spreads"):
        s2s10 = yc["spreads"].get("2s10s")
        s3m10y = yc["spreads"].get("3m10y")

        if s2s10 is not None:
            if s2s10 < t["deep_inversion"]:
                scores["Contraction"] += 2
                signals.append(f"2s10s deeply inverted ({s2s10:+.3f})")
            elif s2s10 < t["inversion"]:
                scores["Late Cycle"] += 2
                signals.append(f"2s10s inverted ({s2s10:+.3f})")
            elif s2s10 > t["steep_curve"]:
                scores["Recovery"] += 1
                scores["Expansion"] += 1
                signals.append(f"2s10s steep ({s2s10:+.3f})")
            else:
                scores["Expansion"] += 1

        if s3m10y is not None:
            if s3m10y < 0:
                scores["Contraction"] += 1
                scores["Late Cycle"] += 1
            else:
                scores["Expansion"] += 1

    # Copper/Gold ratio trend (growth proxy)
    ratios = sentiment_data.get("ratios", [])
    cu_au = next((r for r in ratios if r["label"] == "Copper/Gold"), None)
    if cu_au:
        if cu_au["mo_change_pct"] > t["cu_au_growth"]:
            scores["Expansion"] += 1
            signals.append(f"Cu/Au rising ({cu_au['mo_change_pct']:+.1f}% 1M)")
        elif cu_au["mo_change_pct"] < t["cu_au_contraction"]:
            scores["Contraction"] += 1
            signals.append(f"Cu/Au falling ({cu_au['mo_change_pct']:+.1f}% 1M)")

    # Consumer Discretionary/Staples trend
    xly_xlp = next((r for r in ratios if r["label"] == "Discretionary/Staples"), None)
    if xly_xlp:
        if xly_xlp["mo_change_pct"] > t["xly_xlp_growth"]:
            scores["Expansion"] += 1
        elif xly_xlp["mo_change_pct"] < t["xly_xlp_contraction"]:
            scores["Contraction"] += 1
            scores["Late Cycle"] += 1

    if not any(scores.values()):
        return None, signals

    state = max(scores, key=scores.get)
    return state, signals


def _classify_risk_appetite(sentiment_data, breadth_data=None):
    """Classify risk appetite from credit spreads, small/large cap, leverage, and breadth.

    Returns (state, signals) where state is one of:
    Risk-On, Mixed, Risk-Off
    """
    t = REGIME_THRESHOLDS["risk_appetite"]
    signals = []
    risk_on_votes = 0
    risk_off_votes = 0

    ratios = sentiment_data.get("ratios", [])

    # HYG/LQD (credit risk appetite)
    hyg_lqd = next((r for r in ratios if r["label"] == "High Yield/Inv Grade"), None)
    if hyg_lqd:
        if hyg_lqd["mo_change_pct"] > t["hyg_lqd_risk_on"]:
            risk_on_votes += 1
            signals.append(f"HYG/LQD rising ({hyg_lqd['mo_change_pct']:+.1f}%)")
        elif hyg_lqd["mo_change_pct"] < t["hyg_lqd_risk_off"]:
            risk_off_votes += 1
            signals.append(f"HYG/LQD falling ({hyg_lqd['mo_change_pct']:+.1f}%)")

    # IWM/SPY (breadth / risk-on)
    iwm_spy = next((r for r in ratios if r["label"] == "Small Cap/Large Cap"), None)
    if iwm_spy:
        if iwm_spy["mo_change_pct"] > t["iwm_spy_risk_on"]:
            risk_on_votes += 1
            signals.append(f"IWM/SPY rising ({iwm_spy['mo_change_pct']:+.1f}%)")
        elif iwm_spy["mo_change_pct"] < t["iwm_spy_risk_off"]:
            risk_off_votes += 1
            signals.append(f"IWM/SPY falling ({iwm_spy['mo_change_pct']:+.1f}%)")

    # Leveraged bull/bear ratios
    leverage = sentiment_data.get("leverage", [])
    for lev in leverage:
        if lev["ratio"] > t["leverage_extreme"]:
            risk_on_votes += 1
            signals.append(f"{lev['label']} bull/bear {lev['ratio']:.0f}x (extreme)")
        elif lev["ratio"] < t["leverage_low"]:
            risk_off_votes += 1
            signals.append(f"{lev['label']} bull/bear {lev['ratio']:.1f}x (low)")

    # Market breadth (% above 200 DMA)
    if breadth_data and breadth_data.get("current_200") is not None:
        pct200 = breadth_data["current_200"]
        if pct200 > t["breadth_strong"]:
            risk_on_votes += 1
            signals.append(f"Breadth strong ({pct200:.0f}% > 200 DMA)")
        elif pct200 < t["breadth_weak"]:
            risk_off_votes += 1
            signals.append(f"Breadth weak ({pct200:.0f}% > 200 DMA)")

    if risk_on_votes > risk_off_votes:
        state = "Risk-On"
    elif risk_off_votes > risk_on_votes:
        state = "Risk-Off"
    else:
        state = "Mixed"

    return state, signals


def _classify_monetary(sentiment_data, liquidity_data=None):
    """Classify monetary regime from yield levels, curve shape, rate direction, and Fed liquidity.

    Returns (state, signals) where state is one of:
    Tightening, Pause, Easing
    """
    t = REGIME_THRESHOLDS["monetary"]
    signals = []
    yc = sentiment_data.get("yield_curve")
    yh = sentiment_data.get("yield_history")

    if not yc and not liquidity_data:
        return None, signals

    # Rate direction from yield history (compare recent to older)
    rate_signal = None  # True=tightening, False=easing
    if yc and yh and "yields" in yh and "2y" in yh["yields"]:
        y2_series = yh["yields"]["2y"]
        if len(y2_series) >= 60:
            recent_avg = sum(y2_series[-20:]) / 20
            older_avg = sum(y2_series[-60:-40]) / 20
            if recent_avg > older_avg + t["rate_change_threshold"]:
                rate_signal = True
                signals.append(f"2Y yield rising ({older_avg:.2f}→{recent_avg:.2f})")
            elif recent_avg < older_avg - t["rate_change_threshold"]:
                rate_signal = False
                signals.append(f"2Y yield falling ({older_avg:.2f}→{recent_avg:.2f})")

    # Net Liquidity direction (4-week change)
    liq_signal = None  # True=expanding (easing), False=contracting (tightening)
    if liquidity_data and liquidity_data.get("net_change_4w_pct") is not None:
        chg = liquidity_data["net_change_4w_pct"]
        if chg > t["liquidity_change_threshold"]:
            liq_signal = True
            signals.append(f"Net liquidity expanding ({chg:+.2f}% 4W)")
        elif chg < -t["liquidity_change_threshold"]:
            liq_signal = False
            signals.append(f"Net liquidity contracting ({chg:+.2f}% 4W)")

    # Combine rate direction and liquidity
    tight_votes = 0
    ease_votes = 0
    if rate_signal is True:
        tight_votes += 1
    elif rate_signal is False:
        ease_votes += 1
    if liq_signal is True:
        ease_votes += 1
    elif liq_signal is False:
        tight_votes += 1

    if tight_votes > ease_votes:
        state = "Tightening"
    elif ease_votes > tight_votes:
        state = "Easing"
        if yc:
            spreads = yc.get("spreads", {})
            s2s10 = spreads.get("2s10s")
            if s2s10 is not None and s2s10 > t["curve_steepening"]:
                signals.append("Curve steepening")
    else:
        state = "Pause"
        if yc:
            yields_dict = yc.get("yields", {})
            y2 = yields_dict.get("2y")
            if y2 is not None:
                signals.append(f"2Y at {y2:.2f}%")

    return state, signals


def _classify_credit(credit_data):
    """Classify credit conditions from HY OAS percentile.

    Returns (state, signals) where state is one of:
    Stress, Normal, Complacent
    """
    if not credit_data or credit_data.get("hy_percentile") is None:
        return None, []

    t = REGIME_THRESHOLDS["credit"]
    pctile = credit_data["hy_percentile"]
    current = credit_data.get("current_hy", 0)
    signals = []

    if pctile >= t["stress_percentile"]:
        state = "Stress"
        signals.append(f"HY OAS {current:.0f}bps ({pctile:.0f}th %ile — wide)")
    elif pctile <= t["complacent_percentile"]:
        state = "Complacent"
        signals.append(f"HY OAS {current:.0f}bps ({pctile:.0f}th %ile — tight)")
    else:
        state = "Normal"
        signals.append(f"HY OAS {current:.0f}bps ({pctile:.0f}th %ile)")

    return state, signals


def _composite_label(vol, cycle, risk, monetary, credit=None):
    """Derive a short composite label from the dimension states."""
    # 5-tuple patterns: (vol, cycle, risk, monetary, credit)
    # None = wildcard (matches anything)
    labels = {
        # Crisis conditions — credit stress reinforces
        ("Crisis", None, "Risk-Off", None, "Stress"): "Credit Crisis",
        ("Crisis", None, "Risk-Off", None, None): "Crisis Mode",
        ("Crisis", None, None, None, None): "Vol Spike",
        ("Elevated", None, "Risk-Off", None, "Stress"): "Credit Stress",
        ("Elevated", None, "Risk-Off", None, None): "Stress Rising",

        # Complacency — tight credit confirms
        ("Low Vol", "Expansion", "Risk-On", None, "Complacent"): "Max Complacency",
        ("Low Vol", "Expansion", "Risk-On", None, None): "Complacent Bull",
        ("Low Vol", "Late Cycle", "Risk-On", None, None): "Complacent Late Cycle",
        ("Low Vol", None, "Risk-On", None, None): "Low Vol Risk-On",

        # Late cycle
        (None, "Late Cycle", "Risk-Off", "Tightening", None): "Late Cycle Stress",
        (None, "Late Cycle", "Risk-On", None, None): "Fragile Risk-On",
        (None, "Late Cycle", None, None, None): "Late Cycle",

        # Contraction
        (None, "Contraction", "Risk-Off", None, None): "Defensive Pivot",
        (None, "Contraction", None, "Easing", None): "Easing into Weakness",
        (None, "Contraction", None, None, None): "Contraction",

        # Recovery
        (None, "Recovery", "Risk-On", "Easing", None): "Recovery Rally",
        (None, "Recovery", None, None, None): "Early Recovery",

        # Expansion
        (None, "Expansion", "Risk-On", None, None): "Healthy Expansion",
        (None, "Expansion", None, None, None): "Expansion",
    }

    dims = [vol, cycle, risk, monetary, credit]
    for pattern, label in labels.items():
        match = True
        for actual, expected in zip(dims, pattern, strict=True):
            if expected is not None and actual != expected:
                match = False
                break
        if match:
            return label

    # Fallback: concatenate available dimensions
    parts = [d for d in [cycle, risk] if d]
    if vol and vol not in ("Normal",):
        parts.insert(0, vol)
    return " / ".join(parts) if parts else "Indeterminate"


_REGIME_COLORS = {
    "Credit Crisis": "#f85149",
    "Crisis Mode": "#f85149",
    "Vol Spike": "#f85149",
    "Credit Stress": "#f0883e",
    "Stress Rising": "#f0883e",
    "Late Cycle Stress": "#f0883e",
    "Defensive Pivot": "#f0883e",
    "Contraction": "#f85149",
    "Easing into Weakness": "#d29922",
    "Max Complacency": "#d29922",
    "Complacent Bull": "#d29922",
    "Complacent Late Cycle": "#d29922",
    "Fragile Risk-On": "#d29922",
    "Late Cycle": "#d29922",
    "Low Vol Risk-On": "#3fb950",
    "Healthy Expansion": "#3fb950",
    "Expansion": "#3fb950",
    "Recovery Rally": "#3fb950",
    "Early Recovery": "#58a6ff",
}


def _narrative(composite, vol, cycle, risk, monetary, credit=None):
    """Generate a one-sentence regime narrative."""
    parts = []
    if vol:
        parts.append(f"volatility is {vol.lower()}")
    if cycle:
        parts.append(f"cycle indicators point to {cycle.lower()}")
    if risk:
        parts.append(f"risk appetite is {risk.lower().replace('-', ' ')}")
    if monetary:
        parts.append(f"monetary conditions suggest {monetary.lower()}")
    if credit:
        parts.append(f"credit conditions are {credit.lower()}")

    if not parts:
        return "Insufficient data for regime classification."

    sentence = "Market regime: " + ", ".join(parts) + "."
    return sentence[0].upper() + sentence[1:]


def classify_regime(sentiment_data, cot_rows=None,
                    credit_data=None, liquidity_data=None, breadth_data=None):
    """Classify current market regime from sentiment and macro data.

    Args:
        sentiment_data: dict with vix, leverage, ratios, yield_curve, yield_history.
        cot_rows: optional COT positioning rows (reserved for future use).
        credit_data: optional dict from fetch_credit_spreads().
        liquidity_data: optional dict from fetch_fed_liquidity().
        breadth_data: optional dict from fetch_market_breadth().

    Returns:
        dict with keys: dimensions, composite_label, color, signals, confidence, narrative
    """
    vol_state, vol_signals = _classify_volatility(sentiment_data)
    cycle_state, cycle_signals = _classify_cycle(sentiment_data)
    risk_state, risk_signals = _classify_risk_appetite(sentiment_data, breadth_data)
    mon_state, mon_signals = _classify_monetary(sentiment_data, liquidity_data)
    credit_state, credit_signals = _classify_credit(credit_data)

    composite = _composite_label(vol_state, cycle_state, risk_state, mon_state, credit_state)

    all_signals = vol_signals + cycle_signals + risk_signals + mon_signals + credit_signals
    available = sum(1 for s in [vol_state, cycle_state, risk_state, mon_state, credit_state] if s is not None)
    confidence = available / 5.0

    dimensions = []
    for name, state, color_map in [
        ("Volatility", vol_state, {"Crisis": "#f85149", "Elevated": "#f0883e", "Normal": "#8b949e", "Low Vol": "#3fb950"}),
        ("Cycle", cycle_state, {"Contraction": "#f85149", "Late Cycle": "#d29922", "Expansion": "#3fb950", "Recovery": "#58a6ff"}),
        ("Risk", risk_state, {"Risk-Off": "#f85149", "Mixed": "#d29922", "Risk-On": "#3fb950"}),
        ("Monetary", mon_state, {"Tightening": "#f85149", "Pause": "#d29922", "Easing": "#3fb950"}),
        ("Credit", credit_state, {"Stress": "#f85149", "Normal": "#8b949e", "Complacent": "#d29922"}),
    ]:
        if state:
            dimensions.append({
                "name": name,
                "state": state,
                "color": color_map.get(state, "#8b949e"),
            })

    return {
        "dimensions": dimensions,
        "composite_label": composite,
        "color": _REGIME_COLORS.get(composite, "#8b949e"),
        "signals": all_signals,
        "confidence": confidence,
        "narrative": _narrative(composite, vol_state, cycle_state, risk_state, mon_state, credit_state),
    }
