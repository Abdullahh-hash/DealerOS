import json
from datetime import datetime
from pathlib import Path

import numpy as np

from app.api.client import FreeFlowClient
from app.config.settings import settings
from app.engines.rnd_engine import RNDEngine
from app.services.snapshot_parser import parse_snapshot

import app.services.rnd_surface_builder as rnd_builder


# ============================================================
# CONFIGURATION
# ============================================================

RANGES = [
    450.0,
    500.0,
    550.0,
    600.0,
    650.0,
    700.0,
    750.0,
    800.0,
    850.0,
    900.0,
    950.0,
    1000.0,
]

FOLDS = 5

OUTPUT_DIR = Path("data") / "rnd_snapshots"


# ============================================================
# BUILD RAW OBSERVATIONS FOR A SPECIFIC RANGE
# ============================================================

def get_observations(
    snapshot,
    forward,
    fit_range,
):
    strikes = []
    ivs_pct = []

    for contract in snapshot.contracts:

        if contract.iv_pct is None:
            continue

        iv_pct = float(
            contract.iv_pct
        )

        if iv_pct <= 0:
            continue

        strike = float(
            contract.strike
        )

        right = contract.right.upper()

        if abs(
            strike - forward
        ) > fit_range:
            continue

        use_contract = (
            (
                strike < forward
                and right == "P"
            )
            or
            (
                strike > forward
                and right == "C"
            )
        )

        if not use_contract:
            continue

        strikes.append(
            strike
        )

        ivs_pct.append(
            iv_pct
        )

    strikes = np.asarray(
        strikes,
        dtype=float,
    )

    ivs_pct = np.asarray(
        ivs_pct,
        dtype=float,
    )

    order = np.argsort(
        strikes
    )

    strikes = strikes[
        order
    ]

    ivs_pct = ivs_pct[
        order
    ]

    unique_strikes, indices = np.unique(
        strikes,
        return_index=True,
    )

    return (
        unique_strikes,
        ivs_pct[indices],
    )


# ============================================================
# FIT A SURFACE FROM ARBITRARY TRAINING DATA
# ============================================================

def fit_training_surface(
    snapshot,
    forward,
    time_to_expiry,
    risk_free_rate,
    train_strikes,
    train_ivs_pct,
):
    from app.models.rnd_surface import (
        RNDSurface,
        RNDSurfacePoint,
    )

    strike_grid, fitted_iv_pct = (
        rnd_builder.smooth_iv_surface(
            strikes=train_strikes,
            ivs_pct=train_ivs_pct,
            forward=forward,
        )
    )

    points = [
        RNDSurfacePoint(
            strike=float(strike),
            iv=float(iv_pct) / 100.0,
            right="SMOOTH",
        )
        for strike, iv_pct
        in zip(
            strike_grid,
            fitted_iv_pct,
        )
    ]

    return RNDSurface(
        spot=float(snapshot.spot),
        forward=float(forward),
        time_to_expiry=float(
            time_to_expiry
        ),
        risk_free_rate=float(
            risk_free_rate
        ),
        source_count=len(
            train_strikes
        ),
        calibration_count=0,
        fit_range=0.0,
        grid_step=rnd_builder.RND_GRID_STEP,
        smoothing_lambda=(
            rnd_builder.RND_SPLINE_LAMBDA
        ),
        points=points,
    )


# ============================================================
# INSPECT A SURFACE WITHOUT HIDING FAILURES
# ============================================================

def inspect_surface(
    surface,
):
    engine = RNDEngine(
        spot=surface.spot,
        time_to_expiry=surface.time_to_expiry,
        risk_free_rate=surface.risk_free_rate,
    )

    prices = np.asarray(
        [
            point.strike
            for point in surface.points
        ],
        dtype=float,
    )

    ivs = np.asarray(
        [
            point.iv
            for point in surface.points
        ],
        dtype=float,
    )

    calls = engine.build_call_prices(
        prices=prices,
        ivs=ivs,
    )

    mono = (
        engine.count_monotonicity_failures(
            calls
        )
    )

    density = engine.build_density(
        prices=prices,
        call_prices=calls,
    )

    neg = (
        engine.count_negative_density_points(
            density
        )
    )

    area = float(
        np.trapezoid(
            density,
            prices,
        )
    )

    clean = (
        mono == 0
        and neg == 0
        and np.isfinite(area)
        and area > 0
    )

    return {
        "mono": mono,
        "neg": neg,
        "area": area,
        "clean": clean,
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

def validate_range(
    snapshot,
    forward,
    time_to_expiry,
    risk_free_rate,
    strikes,
    ivs_pct,
):
    n = len(
        strikes
    )

    interior = np.arange(
        1,
        n - 1,
        dtype=int,
    )

    bad_folds = 0
    mono_total = 0
    neg_total = 0

    for fold in range(
        FOLDS
    ):

        test_indices = interior[
            np.arange(
                len(interior)
            )
            % FOLDS
            == fold
        ]

        test_set = set(
            test_indices.tolist()
        )

        train_indices = np.array(
            [
                i
                for i in range(n)
                if i not in test_set
            ],
            dtype=int,
        )

        train_strikes = strikes[
            train_indices
        ]

        train_ivs = ivs_pct[
            train_indices
        ]

        surface = fit_training_surface(
            snapshot=snapshot,
            forward=forward,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            train_strikes=train_strikes,
            train_ivs_pct=train_ivs,
        )

        result = inspect_surface(
            surface
        )

        mono_total += result[
            "mono"
        ]

        neg_total += result[
            "neg"
        ]

        if not result[
            "clean"
        ]:
            bad_folds += 1

    return {
        "bad_folds": bad_folds,
        "mono": mono_total,
        "neg": neg_total,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    client = FreeFlowClient()

    expirations = client.get_expirations(
        settings.default_symbol
    )

    selected_expiry = expirations[
        "expirations"
    ][0]

    # --------------------------------------------------------
    # Fetch exactly ONE snapshot.
    # --------------------------------------------------------

    raw = client.get_snapshot(
        settings.default_symbol,
        selected_expiry,
    )

    # --------------------------------------------------------
    # Save the exact tested snapshot.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        OUTPUT_DIR
        / (
            f"adaptive_probe_"
            f"{selected_expiry}_"
            f"{stamp}.json"
        )
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            raw,
            f,
            indent=2,
        )

    snapshot = parse_snapshot(
        raw
    )

    (
        time_to_expiry,
        risk_free_rate,
        calibration_count,
    ) = rnd_builder.calibrate_model_inputs(
        snapshot
    )

    forward = (
        snapshot.spot
        * np.exp(
            risk_free_rate
            * time_to_expiry
        )
    )

    print()
    print("=" * 105)
    print(
        "DEALEROS LIVE ADAPTIVE RANGE VALIDATION"
    )
    print("=" * 105)

    print(
        f"Saved snapshot : {output_file}"
    )

    print(
        f"Timestamp      : {snapshot.timestamp}"
    )

    print(
        f"Spot           : {snapshot.spot}"
    )

    print(
        f"Forward        : {forward}"
    )

    print(
        f"ATM IV         : {snapshot.atm_iv}"
    )

    print(
        f"T hours        : "
        f"{time_to_expiry * 365 * 24:.6f}"
    )

    print(
        f"Rate           : "
        f"{risk_free_rate * 100:.6f}%"
    )

    print(
        f"Calibration N  : {calibration_count}"
    )

    print()

    print(
        f"{'RANGE':>8}"
        f"{'OBS':>8}"
        f"{'AREA':>12}"
        f"{'MISSING':>12}"
        f"{'FULL M':>9}"
        f"{'FULL N':>9}"
        f"{'BAD CV':>9}"
        f"{'CV MONO':>10}"
        f"{'CV NEG':>9}"
        f"{'STATUS':>10}"
    )

    print("-" * 105)

    largest_valid = None

    for fit_range in RANGES:

        strikes, ivs_pct = (
            get_observations(
                snapshot=snapshot,
                forward=forward,
                fit_range=fit_range,
            )
        )

        if len(strikes) < 10:
            print(
                f"{fit_range:>8.0f}"
                f"{len(strikes):>8}"
                f"{'NA':>12}"
                f"{'NA':>12}"
                f"{'NA':>9}"
                f"{'NA':>9}"
                f"{'NA':>9}"
                f"{'NA':>10}"
                f"{'NA':>9}"
                f"{'ERROR':>10}"
            )
            continue

        full_surface = (
            fit_training_surface(
                snapshot=snapshot,
                forward=forward,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                train_strikes=strikes,
                train_ivs_pct=ivs_pct,
            )
        )

        full = inspect_surface(
            full_surface
        )

        cv = validate_range(
            snapshot=snapshot,
            forward=forward,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            strikes=strikes,
            ivs_pct=ivs_pct,
        )

        clean = (
            full["clean"]
            and cv["bad_folds"] == 0
            and cv["mono"] == 0
            and cv["neg"] == 0
        )

        if clean:
            largest_valid = (
                fit_range
            )

        status = (
            "PASS"
            if clean
            else "FAIL"
        )

        missing = (
            1.0
            - full["area"]
        )

        print(
            f"{fit_range:>8.0f}"
            f"{len(strikes):>8}"
            f"{full['area']:>12.6f}"
            f"{missing:>12.6f}"
            f"{full['mono']:>9}"
            f"{full['neg']:>9}"
            f"{cv['bad_folds']:>9}"
            f"{cv['mono']:>10}"
            f"{cv['neg']:>9}"
            f"{status:>10}"
        )

    print("-" * 105)

    print()

    if largest_valid is None:
        print(
            "No range passed the complete integrity gate."
        )
    else:
        print(
            "Largest fully validated range: "
            f"+/- {largest_valid:.0f}"
        )

    print()


if __name__ == "__main__":
    main()