"""Tests for regime classifier with historical market condition fixtures."""


from market_flows.regime import (
    _classify_credit,
    _classify_cycle,
    _classify_monetary,
    _classify_risk_appetite,
    _classify_volatility,
    _composite_label,
    classify_regime,
)

# --- Fixtures simulating known market conditions ---

def _make_sentiment(
    vix_level=16, vix_ratio=0.9,
    s2s10=0.5, s3m10y=0.3,
    cu_au_mo=0.0, xly_xlp_mo=0.0,
    hyg_lqd_mo=0.0, iwm_spy_mo=0.0,
    leverage_ratios=None,
    yield_2y=None, yield_history_2y=None,
):
    """Build a sentiment_data dict from scalar inputs."""
    data = {
        "vix": {"vix": vix_level, "vix3m": vix_level / vix_ratio if vix_ratio else vix_level, "ratio": vix_ratio},
        "yield_curve": {
            "yields": {"2y": yield_2y or 4.0, "10y": 4.0 + s2s10},
            "spreads": {"2s10s": s2s10, "3m10y": s3m10y},
        },
        "ratios": [
            {"label": "Copper/Gold", "mo_change_pct": cu_au_mo},
            {"label": "Discretionary/Staples", "mo_change_pct": xly_xlp_mo},
            {"label": "High Yield/Inv Grade", "mo_change_pct": hyg_lqd_mo},
            {"label": "Small Cap/Large Cap", "mo_change_pct": iwm_spy_mo},
        ],
        "leverage": leverage_ratios or [],
    }
    if yield_history_2y:
        data["yield_history"] = {"yields": {"2y": yield_history_2y}}
    return data


# --- Volatility classification ---

class TestClassifyVolatility:
    def test_crisis(self):
        state, signals = _classify_volatility(_make_sentiment(vix_level=35))
        assert state == "Crisis"
        assert any("crisis" in s for s in signals)

    def test_elevated(self):
        state, _ = _classify_volatility(_make_sentiment(vix_level=25))
        assert state == "Elevated"

    def test_normal(self):
        state, _ = _classify_volatility(_make_sentiment(vix_level=16))
        assert state == "Normal"

    def test_low_vol(self):
        state, _ = _classify_volatility(_make_sentiment(vix_level=12))
        assert state == "Low Vol"

    def test_backwardation_escalates_normal(self):
        state, signals = _classify_volatility(_make_sentiment(vix_level=18, vix_ratio=1.05))
        assert state == "Elevated"
        assert any("backwardation" in s for s in signals)

    def test_steep_contango(self):
        state, signals = _classify_volatility(_make_sentiment(vix_level=14, vix_ratio=0.80))
        assert state == "Normal"
        assert any("contango" in s.lower() for s in signals)

    def test_no_vix_data(self):
        state, signals = _classify_volatility({})
        assert state is None
        assert signals == []


# --- Cycle classification ---

class TestClassifyCycle:
    def test_deep_inversion_contraction(self):
        state, signals = _classify_cycle(_make_sentiment(s2s10=-0.8, s3m10y=-0.3))
        assert state == "Contraction"

    def test_mild_inversion_late_cycle(self):
        state, _ = _classify_cycle(_make_sentiment(s2s10=-0.2, s3m10y=0.1))
        assert state == "Late Cycle"

    def test_steep_curve_expansion(self):
        state, _ = _classify_cycle(_make_sentiment(s2s10=1.8, s3m10y=1.5))
        assert state in ("Expansion", "Recovery")

    def test_copper_gold_rising_expansion(self):
        state, signals = _classify_cycle(_make_sentiment(s2s10=0.3, cu_au_mo=5.0))
        assert state == "Expansion"
        assert any("Cu/Au rising" in s for s in signals)

    def test_copper_gold_falling_contraction(self):
        state, signals = _classify_cycle(_make_sentiment(s2s10=-0.6, cu_au_mo=-5.0))
        assert state == "Contraction"

    def test_no_data(self):
        state, signals = _classify_cycle({"ratios": []})
        assert state is None


# --- Risk appetite classification ---

class TestClassifyRiskAppetite:
    def test_risk_on(self):
        state, _ = _classify_risk_appetite(
            _make_sentiment(hyg_lqd_mo=2.0, iwm_spy_mo=3.0),
            breadth_data={"current_200": 75},
        )
        assert state == "Risk-On"

    def test_risk_off(self):
        state, _ = _classify_risk_appetite(
            _make_sentiment(hyg_lqd_mo=-2.0, iwm_spy_mo=-3.0),
            breadth_data={"current_200": 25},
        )
        assert state == "Risk-Off"

    def test_mixed(self):
        state, _ = _classify_risk_appetite(
            _make_sentiment(hyg_lqd_mo=2.0, iwm_spy_mo=-3.0),
        )
        assert state == "Mixed"

    def test_leverage_extreme_risk_on(self):
        leverage = [{"label": "Nasdaq", "ratio": 10.0}]
        state, signals = _classify_risk_appetite(
            _make_sentiment(leverage_ratios=leverage),
        )
        assert any("extreme" in s for s in signals)

    def test_breadth_weak_risk_off(self):
        state, signals = _classify_risk_appetite(
            _make_sentiment(hyg_lqd_mo=-1.5),
            breadth_data={"current_200": 20},
        )
        assert state == "Risk-Off"


# --- Monetary classification ---

class TestClassifyMonetary:
    def test_tightening_from_yield_rise(self):
        # 2Y rising from 3.5 to 4.5 over 60 days
        y2_series = [3.5] * 40 + [4.5] * 20
        state, signals = _classify_monetary(
            _make_sentiment(yield_history_2y=y2_series),
        )
        assert state == "Tightening"

    def test_easing_from_yield_fall(self):
        y2_series = [4.5] * 40 + [3.5] * 20
        state, signals = _classify_monetary(
            _make_sentiment(yield_history_2y=y2_series),
        )
        assert state == "Easing"

    def test_easing_from_liquidity_expansion(self):
        state, _ = _classify_monetary(
            _make_sentiment(),
            liquidity_data={"net_change_4w_pct": 1.5},
        )
        assert state == "Easing"

    def test_tightening_from_liquidity_contraction(self):
        state, _ = _classify_monetary(
            _make_sentiment(),
            liquidity_data={"net_change_4w_pct": -1.5},
        )
        assert state == "Tightening"

    def test_pause_when_mixed(self):
        # Yields rising but liquidity expanding → cancel out
        y2_series = [3.5] * 40 + [4.5] * 20
        state, _ = _classify_monetary(
            _make_sentiment(yield_history_2y=y2_series),
            liquidity_data={"net_change_4w_pct": 1.5},
        )
        assert state == "Pause"

    def test_no_data(self):
        state, signals = _classify_monetary({})
        assert state is None


# --- Credit classification ---

class TestClassifyCredit:
    def test_stress(self):
        state, _ = _classify_credit({"hy_percentile": 90, "current_hy": 500})
        assert state == "Stress"

    def test_complacent(self):
        state, _ = _classify_credit({"hy_percentile": 10, "current_hy": 300})
        assert state == "Complacent"

    def test_normal(self):
        state, _ = _classify_credit({"hy_percentile": 50, "current_hy": 400})
        assert state == "Normal"

    def test_no_data(self):
        state, _ = _classify_credit(None)
        assert state is None


# --- Composite label ---

class TestCompositeLabel:
    def test_crisis_mode(self):
        assert _composite_label("Crisis", None, "Risk-Off", None, None) == "Crisis Mode"

    def test_credit_crisis(self):
        assert _composite_label("Crisis", None, "Risk-Off", None, "Stress") == "Credit Crisis"

    def test_healthy_expansion(self):
        assert _composite_label("Normal", "Expansion", "Risk-On", None, None) == "Healthy Expansion"

    def test_complacent_bull(self):
        assert _composite_label("Low Vol", "Expansion", "Risk-On", None, None) == "Complacent Bull"

    def test_late_cycle_stress(self):
        assert _composite_label(None, "Late Cycle", "Risk-Off", "Tightening", None) == "Late Cycle Stress"

    def test_recovery_rally(self):
        assert _composite_label(None, "Recovery", "Risk-On", "Easing", None) == "Recovery Rally"

    def test_fallback_concatenation(self):
        label = _composite_label("Elevated", "Expansion", "Risk-On", "Easing", "Normal")
        # No exact pattern match — should fallback
        assert isinstance(label, str)
        assert len(label) > 0


# --- Full regime classification (integration) ---

class TestClassifyRegime:
    def test_march_2020_crisis(self):
        """Simulate March 2020: VIX 80, curve flat, everything risk-off."""
        sentiment = _make_sentiment(
            vix_level=80, vix_ratio=1.2,
            s2s10=0.1, s3m10y=-0.5,
            cu_au_mo=-8.0, xly_xlp_mo=-10.0,
            hyg_lqd_mo=-5.0, iwm_spy_mo=-8.0,
        )
        credit = {"hy_percentile": 95, "current_hy": 1000}
        result = classify_regime(sentiment, credit_data=credit)
        assert result["composite_label"] == "Credit Crisis"
        assert result["color"] == "#f85149"
        assert result["confidence"] > 0

    def test_bull_market_2021(self):
        """Simulate 2021 bull: low VIX, everything risk-on, expansion."""
        sentiment = _make_sentiment(
            vix_level=12, vix_ratio=0.82,
            s2s10=1.2, s3m10y=1.5,
            cu_au_mo=5.0, xly_xlp_mo=4.0,
            hyg_lqd_mo=2.0, iwm_spy_mo=3.0,
            leverage_ratios=[{"label": "Nasdaq", "ratio": 10.0}],
        )
        credit = {"hy_percentile": 15, "current_hy": 280}
        result = classify_regime(
            sentiment,
            credit_data=credit,
            breadth_data={"current_200": 85},
        )
        assert "Complacen" in result["composite_label"] or "Low Vol" in result["composite_label"]

    def test_tightening_late_cycle(self):
        """Simulate late 2022: elevated VIX, inverted curve, tightening."""
        y2_series = [3.0] * 40 + [4.5] * 20
        sentiment = _make_sentiment(
            vix_level=25, vix_ratio=0.95,
            s2s10=-0.3, s3m10y=-0.1,
            cu_au_mo=-2.0, xly_xlp_mo=-3.0,
            hyg_lqd_mo=-1.5, iwm_spy_mo=-2.5,
            yield_history_2y=y2_series,
        )
        result = classify_regime(sentiment)
        assert result["dimensions"]
        # Should be elevated + late cycle or contraction
        vol_dim = next((d for d in result["dimensions"] if d["name"] == "Volatility"), None)
        assert vol_dim and vol_dim["state"] == "Elevated"

    def test_missing_data_graceful(self):
        """Empty sentiment data should not crash."""
        result = classify_regime({})
        # With no VIX/cycle/monetary/credit data, only risk appetite resolves (Mixed by default)
        assert isinstance(result["composite_label"], str)
        assert result["confidence"] <= 0.2

    def test_narrative_generated(self):
        sentiment = _make_sentiment(vix_level=15, s2s10=0.5)
        result = classify_regime(sentiment)
        assert "Market regime:" in result["narrative"]
        assert len(result["narrative"]) > 20

    def test_all_dimensions_present(self):
        """When all data is provided, all 5 dimensions should be classified."""
        y2_series = [4.0] * 60
        sentiment = _make_sentiment(
            vix_level=15, s2s10=0.5, s3m10y=0.3,
            yield_history_2y=y2_series,
        )
        credit = {"hy_percentile": 50, "current_hy": 400}
        liquidity = {"net_change_4w_pct": 0.1}
        breadth = {"current_200": 55}
        result = classify_regime(
            sentiment,
            credit_data=credit,
            liquidity_data=liquidity,
            breadth_data=breadth,
        )
        assert result["confidence"] == 1.0
        dim_names = {d["name"] for d in result["dimensions"]}
        assert dim_names == {"Volatility", "Cycle", "Risk", "Monetary", "Credit"}
