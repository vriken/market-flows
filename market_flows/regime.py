"""Regime classifier using data already fetched by the dashboard."""


def _classify_volatility(sentiment_data):
    """Classify volatility regime from VIX data.

    Returns (state, signals) where state is one of:
    Crisis, Elevated, Normal, Low Vol
    """
    vix = sentiment_data.get("vix")
    if not vix:
        return None, []

    level = vix["vix"]
    ratio = vix["ratio"]
    signals = []

    if level >= 30:
        state = "Crisis"
        signals.append(f"VIX {level:.0f} (crisis)")
    elif level >= 20:
        state = "Elevated"
        signals.append(f"VIX {level:.0f} (elevated)")
    elif level <= 13:
        state = "Low Vol"
        signals.append(f"VIX {level:.0f} (low)")
    else:
        state = "Normal"
        signals.append(f"VIX {level:.0f}")

    if ratio > 1.0:
        signals.append("VIX backwardation")
        if state == "Normal":
            state = "Elevated"
    elif ratio < 0.85:
        signals.append("Steep contango")

    return state, signals


def _classify_cycle(sentiment_data):
    """Classify business cycle regime from yield spreads and ratios.

    Returns (state, signals) where state is one of:
    Expansion, Late Cycle, Contraction, Recovery
    """
    signals = []
    scores = {"Expansion": 0, "Late Cycle": 0, "Contraction": 0, "Recovery": 0}

    # Yield spreads
    yc = sentiment_data.get("yield_curve")
    if yc and yc.get("spreads"):
        s2s10 = yc["spreads"].get("2s10s")
        s3m10y = yc["spreads"].get("3m10y")

        if s2s10 is not None:
            if s2s10 < -0.5:
                scores["Contraction"] += 2
                signals.append(f"2s10s deeply inverted ({s2s10:+.3f})")
            elif s2s10 < 0:
                scores["Late Cycle"] += 2
                signals.append(f"2s10s inverted ({s2s10:+.3f})")
            elif s2s10 > 1.5:
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
        if cu_au["mo_change_pct"] > 3:
            scores["Expansion"] += 1
            signals.append(f"Cu/Au rising ({cu_au['mo_change_pct']:+.1f}% 1M)")
        elif cu_au["mo_change_pct"] < -3:
            scores["Contraction"] += 1
            signals.append(f"Cu/Au falling ({cu_au['mo_change_pct']:+.1f}% 1M)")

    # Consumer Discretionary/Staples trend
    xly_xlp = next((r for r in ratios if r["label"] == "Discretionary/Staples"), None)
    if xly_xlp:
        if xly_xlp["mo_change_pct"] > 2:
            scores["Expansion"] += 1
        elif xly_xlp["mo_change_pct"] < -2:
            scores["Contraction"] += 1
            scores["Late Cycle"] += 1

    if not any(scores.values()):
        return None, signals

    state = max(scores, key=scores.get)
    return state, signals


def _classify_risk_appetite(sentiment_data):
    """Classify risk appetite from credit spreads, small/large cap, and leverage ratios.

    Returns (state, signals) where state is one of:
    Risk-On, Mixed, Risk-Off
    """
    signals = []
    risk_on_votes = 0
    risk_off_votes = 0

    ratios = sentiment_data.get("ratios", [])

    # HYG/LQD (credit risk appetite)
    hyg_lqd = next((r for r in ratios if r["label"] == "High Yield/Inv Grade"), None)
    if hyg_lqd:
        if hyg_lqd["mo_change_pct"] > 1:
            risk_on_votes += 1
            signals.append(f"HYG/LQD rising ({hyg_lqd['mo_change_pct']:+.1f}%)")
        elif hyg_lqd["mo_change_pct"] < -1:
            risk_off_votes += 1
            signals.append(f"HYG/LQD falling ({hyg_lqd['mo_change_pct']:+.1f}%)")

    # IWM/SPY (breadth / risk-on)
    iwm_spy = next((r for r in ratios if r["label"] == "Small Cap/Large Cap"), None)
    if iwm_spy:
        if iwm_spy["mo_change_pct"] > 2:
            risk_on_votes += 1
            signals.append(f"IWM/SPY rising ({iwm_spy['mo_change_pct']:+.1f}%)")
        elif iwm_spy["mo_change_pct"] < -2:
            risk_off_votes += 1
            signals.append(f"IWM/SPY falling ({iwm_spy['mo_change_pct']:+.1f}%)")

    # Leveraged bull/bear ratios
    leverage = sentiment_data.get("leverage", [])
    for lev in leverage:
        if lev["ratio"] > 8:
            risk_on_votes += 1
            signals.append(f"{lev['label']} bull/bear {lev['ratio']:.0f}x (extreme)")
        elif lev["ratio"] < 2:
            risk_off_votes += 1
            signals.append(f"{lev['label']} bull/bear {lev['ratio']:.1f}x (low)")

    if risk_on_votes > risk_off_votes:
        state = "Risk-On"
    elif risk_off_votes > risk_on_votes:
        state = "Risk-Off"
    else:
        state = "Mixed"

    return state, signals


def _classify_monetary(sentiment_data):
    """Classify monetary regime from yield levels, curve shape, and rate direction.

    Returns (state, signals) where state is one of:
    Tightening, Pause, Easing
    """
    signals = []
    yc = sentiment_data.get("yield_curve")
    yh = sentiment_data.get("yield_history")

    if not yc:
        return None, signals

    yields_dict = yc.get("yields", {})
    y2 = yields_dict.get("2y")

    # Rate direction from yield history (compare recent to older)
    rate_rising = None
    if yh and "yields" in yh and "2y" in yh["yields"]:
        y2_series = yh["yields"]["2y"]
        if len(y2_series) >= 60:
            recent_avg = sum(y2_series[-20:]) / 20
            older_avg = sum(y2_series[-60:-40]) / 20
            if recent_avg > older_avg + 0.15:
                rate_rising = True
                signals.append(f"2Y yield rising ({older_avg:.2f}→{recent_avg:.2f})")
            elif recent_avg < older_avg - 0.15:
                rate_rising = False
                signals.append(f"2Y yield falling ({older_avg:.2f}→{recent_avg:.2f})")

    # Curve shape
    spreads = yc.get("spreads", {})
    s2s10 = spreads.get("2s10s")

    if rate_rising is True:
        state = "Tightening"
    elif rate_rising is False:
        if s2s10 is not None and s2s10 > 0.5:
            state = "Easing"
            signals.append("Curve steepening")
        else:
            state = "Easing"
    else:
        state = "Pause"
        if y2 is not None:
            signals.append(f"2Y at {y2:.2f}%")

    return state, signals


def _composite_label(vol, cycle, risk, monetary):
    """Derive a short composite label from the four dimension states."""
    labels = {
        # Crisis conditions
        ("Crisis", None, "Risk-Off", None): "Crisis Mode",
        ("Crisis", None, None, None): "Vol Spike",
        ("Elevated", None, "Risk-Off", None): "Stress Rising",

        # Complacency
        ("Low Vol", "Expansion", "Risk-On", None): "Complacent Bull",
        ("Low Vol", "Late Cycle", "Risk-On", None): "Complacent Late Cycle",
        ("Low Vol", None, "Risk-On", None): "Low Vol Risk-On",

        # Late cycle
        (None, "Late Cycle", "Risk-Off", "Tightening"): "Late Cycle Stress",
        (None, "Late Cycle", "Risk-On", None): "Fragile Risk-On",
        (None, "Late Cycle", None, None): "Late Cycle",

        # Contraction
        (None, "Contraction", "Risk-Off", None): "Defensive Pivot",
        (None, "Contraction", None, "Easing"): "Easing into Weakness",
        (None, "Contraction", None, None): "Contraction",

        # Recovery
        (None, "Recovery", "Risk-On", "Easing"): "Recovery Rally",
        (None, "Recovery", None, None): "Early Recovery",

        # Expansion
        (None, "Expansion", "Risk-On", None): "Healthy Expansion",
        (None, "Expansion", None, None): "Expansion",
    }

    # Try exact matches first, then partial matches with None as wildcard
    for pattern, label in labels.items():
        match = True
        for actual, expected in zip([vol, cycle, risk, monetary], pattern):
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
    "Crisis Mode": "#f85149",
    "Vol Spike": "#f85149",
    "Stress Rising": "#f0883e",
    "Late Cycle Stress": "#f0883e",
    "Defensive Pivot": "#f0883e",
    "Contraction": "#f85149",
    "Easing into Weakness": "#d29922",
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


def _narrative(composite, vol, cycle, risk, monetary):
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

    if not parts:
        return "Insufficient data for regime classification."

    sentence = "Market regime: " + ", ".join(parts) + "."
    return sentence[0].upper() + sentence[1:]


def classify_regime(sentiment_data, cot_rows=None):
    """Classify current market regime from sentiment data already fetched.

    Args:
        sentiment_data: dict from generate_dashboard with vix, leverage, ratios,
                       yield_curve, yield_history keys.
        cot_rows: optional COT positioning rows (reserved for future use).

    Returns:
        dict with keys: dimensions, composite_label, color, signals, confidence, narrative
    """
    vol_state, vol_signals = _classify_volatility(sentiment_data)
    cycle_state, cycle_signals = _classify_cycle(sentiment_data)
    risk_state, risk_signals = _classify_risk_appetite(sentiment_data)
    mon_state, mon_signals = _classify_monetary(sentiment_data)

    composite = _composite_label(vol_state, cycle_state, risk_state, mon_state)

    all_signals = vol_signals + cycle_signals + risk_signals + mon_signals
    available = sum(1 for s in [vol_state, cycle_state, risk_state, mon_state] if s is not None)
    confidence = available / 4.0

    dimensions = []
    for name, state, color_map in [
        ("Volatility", vol_state, {"Crisis": "#f85149", "Elevated": "#f0883e", "Normal": "#8b949e", "Low Vol": "#3fb950"}),
        ("Cycle", cycle_state, {"Contraction": "#f85149", "Late Cycle": "#d29922", "Expansion": "#3fb950", "Recovery": "#58a6ff"}),
        ("Risk", risk_state, {"Risk-Off": "#f85149", "Mixed": "#d29922", "Risk-On": "#3fb950"}),
        ("Monetary", mon_state, {"Tightening": "#f85149", "Pause": "#d29922", "Easing": "#3fb950"}),
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
        "narrative": _narrative(composite, vol_state, cycle_state, risk_state, mon_state),
    }
