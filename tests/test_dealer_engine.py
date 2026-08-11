import json
from pathlib import Path

import pytest

from app.engines.dealer_engine import DealerEngine
from app.engines.gamma_analyzer import GammaAnalyzer
from app.engines.oi_analyzer import OIAnalyzer

from app.models.dealer_snapshot import DealerSnapshot
from app.models.option_contract import OptionContract

from app.services.snapshot_parser import parse_snapshot


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

REFERENCE_SNAPSHOT = (
    PROJECT_ROOT
    / "data"
    / "rnd_snapshots"
    / (
        "adaptive_probe_"
        "2026-08-11_"
        "20260811_134044.json"
    )
)


# ============================================================
# HELPERS
# ============================================================

def make_snapshot(
    contracts=None,
    spot=100.0,
    total_gex=1.0,
):
    return DealerSnapshot(
        symbol="TEST",
        exp="2026-08-11",
        timestamp="2026-08-11T10:00:00",
        spot=spot,
        dte=0,
        total_gex=total_gex,
        contracts=contracts or [],
    )


def load_reference_snapshot():
    with open(
        REFERENCE_SNAPSHOT,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(f)

    return parse_snapshot(
        raw
    )


# ============================================================
# REFERENCE SNAPSHOT REGRESSION
# ============================================================

def test_reference_dealer_summary_regression():
    """
    Lock the known Dealer structure from the saved
    August 11 NDX snapshot.
    """

    snapshot = load_reference_snapshot()

    result = DealerEngine(
        snapshot
    ).summary()

    assert result.symbol == "NDX"

    assert result.spot == pytest.approx(
        29621.805
    )

    assert result.gamma_state == "Long Gamma"

    assert result.total_gex == pytest.approx(
        530627804.54
    )

    # --------------------------------------------------------
    # Net-GEX profile
    # --------------------------------------------------------

    assert (
        result.net_gex_sign_change_strike
        == pytest.approx(29610.0)
    )

    # --------------------------------------------------------
    # Contract GEX
    # --------------------------------------------------------

    assert (
        result.largest_call_gex_strike
        == pytest.approx(29800.0)
    )

    assert (
        result.largest_call_gex_value
        == pytest.approx(
            292141556.57
        )
    )

    assert (
        result.largest_put_gex_strike
        == pytest.approx(29400.0)
    )

    assert (
        result.largest_put_gex_value
        == pytest.approx(
            -139088985.66
        )
    )

    # --------------------------------------------------------
    # Net-GEX concentrations
    # --------------------------------------------------------

    assert (
        result.strongest_positive_net_gex_strike
        == pytest.approx(29800.0)
    )

    assert (
        result.strongest_positive_net_gex_value
        == pytest.approx(
            291050373.48
        )
    )

    assert (
        result.strongest_negative_net_gex_strike
        == pytest.approx(29490.0)
    )

    assert (
        result.strongest_negative_net_gex_value
        == pytest.approx(
            -34995583.55
        )
    )

    # --------------------------------------------------------
    # Open interest
    # --------------------------------------------------------

    assert (
        result.largest_call_oi_strike
        == pytest.approx(29800.0)
    )

    assert result.largest_call_oi_value == 266

    assert (
        result.largest_put_oi_strike
        == pytest.approx(29400.0)
    )

    assert result.largest_put_oi_value == 153

    assert (
        result.largest_total_oi_strike
        == pytest.approx(29400.0)
    )

    assert result.largest_total_oi_value == 320


# ============================================================
# GEX PROFILE
# ============================================================

def test_gex_profile_aggregation():

    contracts = [
        OptionContract(
            strike=95.0,
            right="C",
            gex=10.0,
        ),
        OptionContract(
            strike=95.0,
            right="P",
            gex=-2.0,
        ),
        OptionContract(
            strike=100.0,
            right="C",
            gex=6.0,
        ),
        OptionContract(
            strike=100.0,
            right="P",
            gex=-1.0,
        ),
        OptionContract(
            strike=105.0,
            right="P",
            gex=-9.0,
        ),
    ]

    snapshot = make_snapshot(
        contracts=contracts
    )

    levels = GammaAnalyzer(
        snapshot
    ).gex_levels()

    assert len(levels) == 3

    level_95 = levels[0]
    level_100 = levels[1]
    level_105 = levels[2]

    assert level_95.strike == 95.0
    assert level_95.call_gex == 10.0
    assert level_95.put_gex == -2.0
    assert level_95.net_gex == 8.0

    assert level_100.strike == 100.0
    assert level_100.call_gex == 6.0
    assert level_100.put_gex == -1.0
    assert level_100.net_gex == 5.0

    assert level_105.strike == 105.0
    assert level_105.call_gex == 0.0
    assert level_105.put_gex == -9.0
    assert level_105.net_gex == -9.0


# ============================================================
# NET-GEX CONCENTRATIONS
# ============================================================

def test_net_gex_concentration_selection():

    contracts = [
        OptionContract(
            strike=95.0,
            right="C",
            gex=12.0,
        ),
        OptionContract(
            strike=95.0,
            right="P",
            gex=-2.0,
        ),
        OptionContract(
            strike=100.0,
            right="C",
            gex=5.0,
        ),
        OptionContract(
            strike=105.0,
            right="P",
            gex=-15.0,
        ),
    ]

    snapshot = make_snapshot(
        contracts=contracts
    )

    analyzer = GammaAnalyzer(
        snapshot
    )

    positive = (
        analyzer
        .strongest_positive_net_gex()
    )

    negative = (
        analyzer
        .strongest_negative_net_gex()
    )

    assert positive is not None
    assert positive.strike == 95.0
    assert positive.net_gex == 10.0

    assert negative is not None
    assert negative.strike == 105.0
    assert negative.net_gex == -15.0


# ============================================================
# NET-GEX SIGN CHANGE
# ============================================================

def test_net_gex_sign_change_is_nearest_to_spot():
    """
    This tests the current factual metric only.

    It is a strike-level sign change in the GEX profile,
    NOT a true portfolio zero-gamma calculation.
    """

    contracts = [
        OptionContract(
            strike=90.0,
            right="C",
            gex=10.0,
        ),
        OptionContract(
            strike=100.0,
            right="P",
            gex=-5.0,
        ),
        OptionContract(
            strike=110.0,
            right="C",
            gex=20.0,
        ),
    ]

    snapshot = make_snapshot(
        contracts=contracts,
        spot=104.0,
    )

    analyzer = GammaAnalyzer(
        snapshot
    )

    result = (
        analyzer
        .nearest_net_gex_sign_change_strike()
    )

    # Crossings occur at 100 and 110.
    # 100 is nearer to spot=104.
    assert result == 100.0


# ============================================================
# OPEN INTEREST PROFILE
# ============================================================

def test_oi_profile_and_largest_levels():

    contracts = [
        OptionContract(
            strike=95.0,
            right="C",
            oi=10,
        ),
        OptionContract(
            strike=95.0,
            right="P",
            oi=20,
        ),
        OptionContract(
            strike=100.0,
            right="C",
            oi=40,
        ),
        OptionContract(
            strike=105.0,
            right="P",
            oi=50,
        ),
    ]

    snapshot = make_snapshot(
        contracts=contracts
    )

    analyzer = OIAnalyzer(
        snapshot
    )

    levels = analyzer.oi_levels()

    assert len(levels) == 3

    assert levels[0].strike == 95.0
    assert levels[0].call_oi == 10
    assert levels[0].put_oi == 20
    assert levels[0].total_oi == 30

    assert levels[1].strike == 100.0
    assert levels[1].call_oi == 40
    assert levels[1].put_oi == 0
    assert levels[1].total_oi == 40

    assert levels[2].strike == 105.0
    assert levels[2].call_oi == 0
    assert levels[2].put_oi == 50
    assert levels[2].total_oi == 50

    largest_call = (
        analyzer.largest_call_oi()
    )

    largest_put = (
        analyzer.largest_put_oi()
    )

    largest_total = (
        analyzer.largest_total_oi_level()
    )

    assert largest_call is not None
    assert largest_call.strike == 100.0
    assert largest_call.oi == 40

    assert largest_put is not None
    assert largest_put.strike == 105.0
    assert largest_put.oi == 50

    assert largest_total is not None
    assert largest_total.strike == 105.0
    assert largest_total.total_oi == 50


# ============================================================
# GAMMA STATE
# ============================================================

@pytest.mark.parametrize(
    (
        "total_gex",
        "expected",
    ),
    [
        (
            100.0,
            "Long Gamma",
        ),
        (
            -100.0,
            "Short Gamma",
        ),
        (
            None,
            "Unknown",
        ),
    ],
)
def test_gamma_state(
    total_gex,
    expected,
):

    snapshot = make_snapshot(
        total_gex=total_gex
    )

    analyzer = GammaAnalyzer(
        snapshot
    )

    assert (
        analyzer.gamma_state()
        == expected
    )


# ============================================================
# MISSING DATA
# ============================================================

def test_missing_gex_and_oi_are_ignored():

    contracts = [
        OptionContract(
            strike=100.0,
            right="C",
            gex=None,
            oi=None,
        ),
        OptionContract(
            strike=100.0,
            right="P",
            gex=None,
            oi=None,
        ),
    ]

    snapshot = make_snapshot(
        contracts=contracts
    )

    gamma = GammaAnalyzer(
        snapshot
    )

    oi = OIAnalyzer(
        snapshot
    )

    assert gamma.gex_levels() == []

    assert (
        gamma.largest_call_gex()
        is None
    )

    assert (
        gamma.largest_put_gex()
        is None
    )

    assert (
        gamma
        .strongest_positive_net_gex()
        is None
    )

    assert (
        gamma
        .strongest_negative_net_gex()
        is None
    )

    assert (
        gamma
        .nearest_net_gex_sign_change_strike()
        is None
    )

    assert oi.oi_levels() == []

    assert (
        oi.largest_call_oi()
        is None
    )

    assert (
        oi.largest_put_oi()
        is None
    )

    assert (
        oi.largest_total_oi_level()
        is None
    )


# ============================================================
# PUBLIC DEALER INTERFACE
# ============================================================

def test_public_summary_uses_factual_names():
    """
    Prevent misleading legacy terminology from
    silently returning to the public Dealer result.
    """

    snapshot = make_snapshot(
        total_gex=100.0
    )

    result = DealerEngine(
        snapshot
    ).summary()

    assert result.gamma_state == (
        "Long Gamma"
    )

    # Legacy/misleading concepts must not exist.
    assert not hasattr(
        result,
        "dealer_bias",
    )

    assert not hasattr(
        result,
        "gamma_flip",
    )

    assert not hasattr(
        result,
        "support",
    )

    assert not hasattr(
        result,
        "resistance",
    )