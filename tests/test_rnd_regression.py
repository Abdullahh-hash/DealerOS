import json
import math
from pathlib import Path

import pytest

from app.engines.rnd_engine import (
    MIN_RAW_AREA,
    RNDEngine,
)

from app.services.rnd_surface_builder import (
    RND_GRID_STEP,
    RND_SIGMA_MULTIPLIER,
    RND_SPLINE_LAMBDA,
    build_rnd_surface,
)

from app.services.snapshot_parser import (
    parse_snapshot,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SNAPSHOT_DIR = (
    PROJECT_ROOT
    / "data"
    / "rnd_snapshots"
)


# ============================================================
# SAVED REGRESSION SNAPSHOTS
# ============================================================

SNAPSHOT_FILES = sorted(
    list(
        SNAPSHOT_DIR.glob(
            "snapshot_*.json"
        )
    )
    + list(
        SNAPSHOT_DIR.glob(
            "adaptive_probe_*.json"
        )
    )
)


REFERENCE_SNAPSHOT = (
    SNAPSHOT_DIR
    / (
        "adaptive_probe_"
        "2026-08-11_"
        "20260811_134044.json"
    )
)


# ============================================================
# HELPERS
# ============================================================

def load_snapshot(
    path: Path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(f)

    return parse_snapshot(
        raw
    )


def build_production_rnd(
    path: Path,
):
    snapshot = load_snapshot(
        path
    )

    surface = build_rnd_surface(
        snapshot
    )

    engine = RNDEngine(
        spot=surface.spot,
        time_to_expiry=surface.time_to_expiry,
        risk_free_rate=surface.risk_free_rate,
    )

    result = engine.build_from_surface(
        surface
    )

    return (
        snapshot,
        surface,
        result,
    )


# ============================================================
# CONFIGURATION LOCK
# ============================================================

def test_rnd_validated_configuration():
    """
    Lock the currently validated production
    RND configuration.
    """

    assert RND_SIGMA_MULTIPLIER == pytest.approx(
        3.10
    )

    assert RND_SPLINE_LAMBDA == pytest.approx(
        0.002
    )

    assert RND_GRID_STEP == pytest.approx(
        5.0
    )

    assert MIN_RAW_AREA == pytest.approx(
        0.99
    )


# ============================================================
# SNAPSHOT SET
# ============================================================

def test_regression_snapshot_set_exists():
    """
    We currently have at least the 11 live snapshots
    used to validate the production RND methodology.
    """

    assert len(
        SNAPSHOT_FILES
    ) >= 11

    assert REFERENCE_SNAPSHOT.exists()


# ============================================================
# FULL PRODUCTION REGRESSION
# ============================================================

@pytest.mark.parametrize(
    "snapshot_path",
    SNAPSHOT_FILES,
    ids=lambda path: path.name,
)
def test_production_rnd_integrity(
    snapshot_path,
):
    """
    Every saved live snapshot must pass all production
    RND integrity gates.
    """

    (
        snapshot,
        surface,
        result,
    ) = build_production_rnd(
        snapshot_path
    )

    # --------------------------------------------------------
    # Basic surface sanity
    # --------------------------------------------------------

    assert surface.source_count >= 10

    assert len(
        surface.points
    ) >= 3

    assert surface.fit_range > 0

    # --------------------------------------------------------
    # Adaptive range formula
    # --------------------------------------------------------

    expected_sigma_move = (
        float(snapshot.spot)
        * (
            float(snapshot.atm_iv)
            / 100.0
        )
        * math.sqrt(
            surface.time_to_expiry
        )
    )

    expected_range = (
        RND_SIGMA_MULTIPLIER
        * expected_sigma_move
    )

    assert surface.fit_range == pytest.approx(
        expected_range,
        rel=1e-12,
        abs=1e-9,
    )

    # --------------------------------------------------------
    # Structural integrity
    # --------------------------------------------------------

    assert (
        result.monotonicity_failures
        == 0
    )

    assert (
        result.negative_density_points
        == 0
    )

    # --------------------------------------------------------
    # Raw probability coverage
    # --------------------------------------------------------

    assert result.raw_area >= (
        MIN_RAW_AREA
    )

    assert result.raw_area <= 1.01

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    assert result.normalized_area == pytest.approx(
        1.0,
        abs=1e-10,
    )

    # --------------------------------------------------------
    # Density
    # --------------------------------------------------------

    assert result.min_raw_density >= 0

    assert result.max_raw_density > 0

    # --------------------------------------------------------
    # Distribution ordering
    # --------------------------------------------------------

    assert (
        result.q05
        < result.q25
        < result.median
        < result.q75
        < result.q95
    )

    # --------------------------------------------------------
    # Distribution width
    # --------------------------------------------------------

    assert result.std > 0

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    assert (
        0.0
        <= result.probability_above_spot
        <= 1.0
    )

    assert (
        0.0
        <= result.probability_above_forward
        <= 1.0
    )

    # --------------------------------------------------------
    # Public result metadata
    # --------------------------------------------------------

    assert result.symbol == (
        snapshot.symbol
    )

    assert result.exp == (
        snapshot.exp
    )

    assert result.spot == pytest.approx(
        snapshot.spot
    )

    assert result.forward == pytest.approx(
        surface.forward
    )

    assert result.fit_range == pytest.approx(
        surface.fit_range
    )

    assert result.source_count == (
        surface.source_count
    )

    assert result.surface_points == len(
        surface.points
    )

    assert len(
        result.points
    ) == result.surface_points


# ============================================================
# EXACT NUMERICAL REGRESSION
# ============================================================

def test_reference_snapshot_numerical_regression():
    """
    Lock a known high-IV 0DTE production result.

    This catches subtle mathematical changes even when
    general integrity checks still pass.
    """

    (
        snapshot,
        surface,
        result,
    ) = build_production_rnd(
        REFERENCE_SNAPSHOT
    )

    # --------------------------------------------------------
    # Market / model inputs
    # --------------------------------------------------------

    assert snapshot.symbol == "NDX"

    assert snapshot.exp == "2026-08-11"

    assert snapshot.spot == pytest.approx(
        29621.805,
        abs=1e-9,
    )

    assert surface.time_to_expiry * 365 * 24 == pytest.approx(
        11.997331016809099,
        abs=1e-9,
    )

    assert surface.risk_free_rate * 100 == pytest.approx(
        5.319329,
        abs=1e-5,
    )

    # --------------------------------------------------------
    # Adaptive surface
    # --------------------------------------------------------

    assert surface.fit_range == pytest.approx(
        831.5673889637129,
        abs=1e-9,
    )

    assert surface.source_count == 134

    assert len(
        surface.points
    ) == 331

    # --------------------------------------------------------
    # Raw RND
    # --------------------------------------------------------

    assert result.raw_area == pytest.approx(
        0.9915000906287206,
        abs=1e-10,
    )

    assert result.normalized_area == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert result.monotonicity_failures == 0

    assert result.negative_density_points == 0

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    assert result.mean_minus_forward == pytest.approx(
        5.240227344598679,
        abs=1e-8,
    )

    assert result.std == pytest.approx(
        248.55442535078626,
        abs=1e-8,
    )

    # --------------------------------------------------------
    # Central 50% interval
    # --------------------------------------------------------

    assert result.q25 == pytest.approx(
        29476.02045975744,
        abs=1e-8,
    )

    assert result.q75 == pytest.approx(
        29799.931726724626,
        abs=1e-8,
    )

    # --------------------------------------------------------
    # Central 90% interval
    # --------------------------------------------------------

    assert result.q05 == pytest.approx(
        29199.143779991733,
        abs=1e-8,
    )

    assert result.q95 == pytest.approx(
        30021.0502193687,
        abs=1e-8,
    )

    # --------------------------------------------------------
    # Risk-neutral probability
    # --------------------------------------------------------

    assert result.probability_above_spot == pytest.approx(
        0.5154144925457397,
        abs=1e-10,
    )


# ============================================================
# PUBLIC RESULT CONVENIENCE INTERFACE
# ============================================================

def test_rnd_result_public_interface():
    """
    Verify the canonical public result exposes the fields
    future DealerOS components need.
    """

    (
        _,
        _,
        result,
    ) = build_production_rnd(
        REFERENCE_SNAPSHOT
    )

    assert result.model_hours > 0

    assert result.coverage_pct >= 99.0

    lower_50, upper_50 = (
        result.range_50
    )

    lower_90, upper_90 = (
        result.range_90
    )

    assert lower_90 < lower_50

    assert lower_50 < upper_50

    assert upper_50 < upper_90