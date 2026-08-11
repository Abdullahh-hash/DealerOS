import numpy as np

from app.api.client import FreeFlowClient
from app.config.settings import settings
from app.services.snapshot_parser import parse_snapshot

import app.services.rnd_surface_builder as rnd_builder

from app.engines.rnd_engine import RNDEngine


RANGES = [
    350.0,
    400.0,
    450.0,
    500.0,
    550.0,
    600.0,
    650.0,
    700.0,
]


def inspect_surface(surface):
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

    call_prices = engine.build_call_prices(
        prices=prices,
        ivs=ivs,
    )

    mono = engine.count_monotonicity_failures(
        call_prices
    )

    raw_density = engine.build_density(
        prices=prices,
        call_prices=call_prices,
    )

    neg = engine.count_negative_density_points(
        raw_density
    )

    area = float(
        np.trapezoid(
            raw_density,
            prices,
        )
    )

    clean = (
        mono == 0
        and neg == 0
        and np.isfinite(area)
        and area > 0
    )

    std = None
    mean_fwd = None

    if clean:
        result = engine.build_from_surface(
            surface
        )

        std = result.std
        mean_fwd = result.mean_minus_forward

    return {
        "mono": mono,
        "neg": neg,
        "area": area,
        "clean": clean,
        "std": std,
        "mean_fwd": mean_fwd,
    }


def main():
    print()
    print("=" * 110)
    print("DEALEROS LIVE RND RANGE PROBE")
    print("=" * 110)

    client = FreeFlowClient()

    expirations = client.get_expirations(
        settings.default_symbol
    )

    selected_expiry = expirations[
        "expirations"
    ][0]

    # Fetch ONCE.
    # Every range below is tested against the exact
    # same market snapshot.
    raw_snapshot = client.get_snapshot(
        settings.default_symbol,
        selected_expiry,
    )

    snapshot = parse_snapshot(
        raw_snapshot
    )

    print(
        f"Symbol       : {snapshot.symbol}"
    )

    print(
        f"Expiry       : {snapshot.exp}"
    )

    print(
        f"Timestamp    : {snapshot.timestamp}"
    )

    print(
        f"Spot         : {snapshot.spot}"
    )

    print(
        f"DTE          : {snapshot.dte}"
    )

    print(
        f"ATM IV       : {snapshot.atm_iv}"
    )

    print(
        f"Contracts    : {len(snapshot.contracts)}"
    )

    print()

    print(
        f"{'RANGE':>8}"
        f"{'SRC':>8}"
        f"{'PTS':>8}"
        f"{'MONO':>8}"
        f"{'NEG':>8}"
        f"{'AREA':>12}"
        f"{'MISSING':>12}"
        f"{'STD':>12}"
        f"{'MEAN-FWD':>12}"
        f"{'STATUS':>10}"
    )

    print("-" * 110)

    original_range = (
        rnd_builder.RND_FIT_RANGE
    )

    try:
        for fit_range in RANGES:

            rnd_builder.RND_FIT_RANGE = (
                float(fit_range)
            )

            try:
                surface = (
                    rnd_builder.build_rnd_surface(
                        snapshot
                    )
                )

                diagnostics = (
                    inspect_surface(
                        surface
                    )
                )

                area = diagnostics[
                    "area"
                ]

                missing = (
                    1.0 - area
                )

                if diagnostics["std"] is None:
                    std_text = "NA"
                    mean_fwd_text = "NA"
                else:
                    std_text = (
                        f"{diagnostics['std']:.3f}"
                    )

                    mean_fwd_text = (
                        f"{diagnostics['mean_fwd']:.3f}"
                    )

                status = (
                    "PASS"
                    if diagnostics["clean"]
                    else "FAIL"
                )

                print(
                    f"{fit_range:>8.0f}"
                    f"{surface.source_count:>8}"
                    f"{len(surface):>8}"
                    f"{diagnostics['mono']:>8}"
                    f"{diagnostics['neg']:>8}"
                    f"{area:>12.6f}"
                    f"{missing:>12.6f}"
                    f"{std_text:>12}"
                    f"{mean_fwd_text:>12}"
                    f"{status:>10}"
                )

            except Exception as exc:
                print(
                    f"{fit_range:>8.0f}"
                    f"{'NA':>8}"
                    f"{'NA':>8}"
                    f"{'NA':>8}"
                    f"{'NA':>8}"
                    f"{'NA':>12}"
                    f"{'NA':>12}"
                    f"{'NA':>12}"
                    f"{'NA':>12}"
                    f"{'ERROR':>10}"
                )

                print(
                    f"    Reason: {exc}"
                )

    finally:
        # Restore production configuration.
        rnd_builder.RND_FIT_RANGE = (
            original_range
        )

    print("-" * 110)

    print()
    print(
        "Production RND_FIT_RANGE restored to:",
        rnd_builder.RND_FIT_RANGE,
    )

    print()


if __name__ == "__main__":
    main()