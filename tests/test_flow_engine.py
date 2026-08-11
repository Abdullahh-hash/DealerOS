import json
import math
from pathlib import Path

import pytest

from app.engines.flow_engine import FlowEngine

from app.models.flow_snapshot import (
    FlowCounts,
    FlowSnapshot,
    FlowStrike,
    PremiumSide,
)

from app.models.flow_summary import FlowSummary

from app.services.flow_parser import parse_flow


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

REFERENCE_FLOW = (
    PROJECT_ROOT
    / "data"
    / "flow_snapshots"
    / "flow_2026-08-10_reference.json"
)


# ============================================================
# HELPERS
# ============================================================

def make_snapshot(
    *,
    net_directional=0.0,
    strikes=None,
    buy_count=10,
    sell_count=10,
    classified_count=20,
):
    return FlowSnapshot(
        symbol="TEST",
        expiry="2026-08-11",
        timestamp="2026-08-11T10:00:00",

        spot=100.0,
        dte=0,
        window_min=0,

        net_premium=0.0,
        bull_premium=0.0,
        bear_premium=0.0,

        net_directional=net_directional,
        net_delta_notional=0.0,

        call_premium=PremiumSide(
            buy=0.0,
            sell=0.0,
        ),

        put_premium=PremiumSide(
            buy=0.0,
            sell=0.0,
        ),

        put_call_prem_ratio=0.0,

        counts=FlowCounts(
            classified=classified_count,
            buy=buy_count,
            sell=sell_count,
        ),

        by_strike=(
            strikes
            if strikes is not None
            else []
        ),
    )


def load_reference_flow():
    with open(
        REFERENCE_FLOW,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(f)

    return (
        raw,
        parse_flow(raw),
    )


# ============================================================
# REFERENCE FLOW REGRESSION
# ============================================================

def test_reference_flow_summary_regression():
    """
    Lock the known FreeFlow result used during
    Flow Engine semantic validation.
    """

    raw, snapshot = (
        load_reference_flow()
    )

    result = FlowEngine(
        snapshot
    ).summary()

    assert isinstance(
        result,
        FlowSummary,
    )

    assert result.symbol == "NDX"
    assert result.expiry == "2026-08-10"

    assert result.spot == pytest.approx(
        29722.303
    )

    assert (
        result.directional_flow_state
        == "Bearish"
    )

    # --------------------------------------------------------
    # Directional premium
    # --------------------------------------------------------

    assert result.bull_premium == pytest.approx(
        5945258.0
    )

    assert result.bear_premium == pytest.approx(
        19344604.0
    )

    assert result.net_directional == pytest.approx(
        -13399346.0
    )

    assert result.net_delta_notional == pytest.approx(
        -156872906.94
    )

    # --------------------------------------------------------
    # Aggressor premium
    # --------------------------------------------------------

    assert result.net_premium == pytest.approx(
        -3622598.0
    )

    assert result.call_buy_premium == pytest.approx(
        5698620.0
    )

    assert result.call_sell_premium == pytest.approx(
        14209592.0
    )

    assert result.put_buy_premium == pytest.approx(
        5135012.0
    )

    assert result.put_sell_premium == pytest.approx(
        246638.0
    )

    assert (
        result.put_call_premium_ratio
        == pytest.approx(0.27)
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    assert result.classified_count == 444
    assert result.buy_count == 259
    assert result.sell_count == 185

    assert (
        result.classified_buy_sell_ratio
        == pytest.approx(1.4)
    )

    # --------------------------------------------------------
    # Returned strike subset
    # --------------------------------------------------------

    assert result.returned_strike_count == 20

    assert (
        result.largest_call_buy_strike
        == pytest.approx(29320.0)
    )

    assert (
        result.largest_call_buy_value
        == pytest.approx(2353000.0)
    )

    assert (
        result.largest_call_sell_strike
        == pytest.approx(24500.0)
    )

    assert (
        result.largest_call_sell_value
        == pytest.approx(-3092736.0)
    )

    assert (
        result.largest_put_buy_strike
        == pytest.approx(31000.0)
    )

    assert (
        result.largest_put_buy_value
        == pytest.approx(670310.0)
    )

    assert (
        result.largest_put_sell_strike
        == pytest.approx(28890.0)
    )

    assert (
        result.largest_put_sell_value
        == pytest.approx(-360.0)
    )

    assert (
        result.largest_premium_buy_strike
        == pytest.approx(29320.0)
    )

    assert (
        result.largest_premium_buy_value
        == pytest.approx(2355815.0)
    )

    assert (
        result.largest_premium_sell_strike
        == pytest.approx(24500.0)
    )

    assert (
        result.largest_premium_sell_value
        == pytest.approx(-3092736.0)
    )

    assert (
        result.strongest_bullish_directional_strike
        == pytest.approx(29320.0)
    )

    assert (
        result.strongest_bullish_directional_value
        == pytest.approx(2350185.0)
    )

    assert (
        result.strongest_bearish_directional_strike
        == pytest.approx(24500.0)
    )

    assert (
        result.strongest_bearish_directional_value
        == pytest.approx(-3092736.0)
    )

    assert len(
        result.returned_strikes
    ) == 20

    assert raw["symbol"] == result.symbol


# ============================================================
# TOP-LEVEL FLOW IDENTITIES
# ============================================================

def test_reference_flow_accounting_identities():

    _, snapshot = (
        load_reference_flow()
    )

    call_buy = (
        snapshot.call_premium.buy
    )

    call_sell = (
        snapshot.call_premium.sell
    )

    put_buy = (
        snapshot.put_premium.buy
    )

    put_sell = (
        snapshot.put_premium.sell
    )

    expected_bull = (
        call_buy
        + put_sell
    )

    expected_bear = (
        call_sell
        + put_buy
    )

    expected_directional = (
        expected_bull
        - expected_bear
    )

    expected_net_premium = (
        call_buy
        + put_buy
        - call_sell
        - put_sell
    )

    assert snapshot.bull_premium == pytest.approx(
        expected_bull
    )

    assert snapshot.bear_premium == pytest.approx(
        expected_bear
    )

    assert snapshot.net_directional == pytest.approx(
        expected_directional
    )

    assert snapshot.net_premium == pytest.approx(
        expected_net_premium
    )


# ============================================================
# PROVIDER STRIKE IDENTITY
# ============================================================

def test_reference_by_strike_identity():
    """
    FreeFlow by_strike net is:

        call_net + put_net

    It is an aggressor premium balance, not a
    directional premium balance.
    """

    _, snapshot = (
        load_reference_flow()
    )

    assert len(
        snapshot.by_strike
    ) == 20

    for item in snapshot.by_strike:

        assert item.net == pytest.approx(
            item.call_net
            + item.put_net
        )


# ============================================================
# FLOW STRIKE PROPERTIES
# ============================================================

def test_flow_strike_balance_properties():

    item = FlowStrike(
        strike=100.0,
        call_net=100.0,
        put_net=40.0,
        net=140.0,
    )

    assert item.aggressor_net == pytest.approx(
        140.0
    )

    assert item.directional_net == pytest.approx(
        60.0
    )


# ============================================================
# DIRECTIONAL FLOW STATE
# ============================================================

@pytest.mark.parametrize(
    (
        "net_directional",
        "expected",
    ),
    [
        (
            100.0,
            "Bullish",
        ),
        (
            -100.0,
            "Bearish",
        ),
        (
            0.0,
            "Neutral",
        ),
    ],
)
def test_directional_flow_state(
    net_directional,
    expected,
):

    snapshot = make_snapshot(
        net_directional=net_directional
    )

    analyzer = FlowEngine(
        snapshot
    )

    assert (
        analyzer.directional_flow_state()
        == expected
    )


# ============================================================
# CLASSIFIED COUNT RATIO
# ============================================================

def test_classified_buy_sell_ratio():

    snapshot = make_snapshot(
        buy_count=30,
        sell_count=20,
        classified_count=50,
    )

    analyzer = FlowEngine(
        snapshot
    )

    assert (
        analyzer.classified_buy_sell_ratio()
        == pytest.approx(1.5)
    )


def test_zero_sell_count_ratio():

    snapshot = make_snapshot(
        buy_count=10,
        sell_count=0,
        classified_count=10,
    )

    analyzer = FlowEngine(
        snapshot
    )

    assert math.isinf(
        analyzer.classified_buy_sell_ratio()
    )


# ============================================================
# STRIKE SELECTION
# ============================================================

def test_strike_balance_selection():
    """
    Verify that aggressor balance and directional
    balance remain distinct concepts.
    """

    strikes = [
        FlowStrike(
            strike=95.0,
            call_net=120.0,
            put_net=10.0,
            net=130.0,
        ),
        FlowStrike(
            strike=100.0,
            call_net=-220.0,
            put_net=20.0,
            net=-200.0,
        ),
        FlowStrike(
            strike=105.0,
            call_net=10.0,
            put_net=150.0,
            net=160.0,
        ),
        FlowStrike(
            strike=110.0,
            call_net=60.0,
            put_net=-100.0,
            net=-40.0,
        ),
    ]

    snapshot = make_snapshot(
        strikes=strikes
    )

    analyzer = FlowEngine(
        snapshot
    )

    # --------------------------------------------------------
    # Call balances
    # --------------------------------------------------------

    call_buy = (
        analyzer.largest_call_buy_balance()
    )

    call_sell = (
        analyzer.largest_call_sell_balance()
    )

    assert call_buy is not None
    assert call_buy.strike == 95.0
    assert call_buy.call_net == 120.0

    assert call_sell is not None
    assert call_sell.strike == 100.0
    assert call_sell.call_net == -220.0

    # --------------------------------------------------------
    # Put balances
    # --------------------------------------------------------

    put_buy = (
        analyzer.largest_put_buy_balance()
    )

    put_sell = (
        analyzer.largest_put_sell_balance()
    )

    assert put_buy is not None
    assert put_buy.strike == 105.0
    assert put_buy.put_net == 150.0

    assert put_sell is not None
    assert put_sell.strike == 110.0
    assert put_sell.put_net == -100.0

    # --------------------------------------------------------
    # Aggressor premium balance
    # --------------------------------------------------------

    premium_buy = (
        analyzer.largest_premium_buy_balance()
    )

    premium_sell = (
        analyzer.largest_premium_sell_balance()
    )

    assert premium_buy is not None
    assert premium_buy.strike == 105.0
    assert premium_buy.aggressor_net == 160.0

    assert premium_sell is not None
    assert premium_sell.strike == 100.0
    assert premium_sell.aggressor_net == -200.0

    # --------------------------------------------------------
    # Directional premium balance
    # --------------------------------------------------------

    bullish = (
        analyzer
        .strongest_bullish_directional_strike()
    )

    bearish = (
        analyzer
        .strongest_bearish_directional_strike()
    )

    assert bullish is not None
    assert bullish.strike == 110.0

    assert bullish.directional_net == pytest.approx(
        160.0
    )

    assert bearish is not None
    assert bearish.strike == 100.0

    assert bearish.directional_net == pytest.approx(
        -240.0
    )


# ============================================================
# EMPTY RETURNED STRIKES
# ============================================================

def test_empty_returned_strikes():

    snapshot = make_snapshot(
        strikes=[]
    )

    result = FlowEngine(
        snapshot
    ).summary()

    assert result.returned_strike_count == 0
    assert result.returned_strikes == []

    assert (
        result.largest_call_buy_strike
        is None
    )

    assert (
        result.largest_call_sell_strike
        is None
    )

    assert (
        result.largest_put_buy_strike
        is None
    )

    assert (
        result.largest_put_sell_strike
        is None
    )

    assert (
        result.largest_premium_buy_strike
        is None
    )

    assert (
        result.largest_premium_sell_strike
        is None
    )

    assert (
        result.strongest_bullish_directional_strike
        is None
    )

    assert (
        result.strongest_bearish_directional_strike
        is None
    )


# ============================================================
# PUBLIC FLOW INTERFACE
# ============================================================

def test_public_flow_summary_uses_factual_names():

    snapshot = make_snapshot(
        net_directional=100.0
    )

    result = FlowEngine(
        snapshot
    ).summary()

    assert isinstance(
        result,
        FlowSummary,
    )

    assert not isinstance(
        result,
        dict,
    )

    assert (
        result.directional_flow_state
        == "Bullish"
    )

    # Legacy / ambiguous public summary names
    # should not return.
    assert not hasattr(
        result,
        "flow_bias",
    )

    assert not hasattr(
        result,
        "largest_positive_strike",
    )

    assert not hasattr(
        result,
        "largest_negative_strike",
    )