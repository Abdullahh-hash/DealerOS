import json
import math
from pathlib import Path

import numpy as np

from app.services.snapshot_parser import parse_snapshot
import app.services.rnd_surface_builder as rnd_builder

from research.test_live_adaptive_range import (
    get_observations,
    fit_training_surface,
    inspect_surface,
    validate_range,
)


# ============================================================
# CONFIGURATION
# ============================================================

SNAPSHOT_DIR = Path("data") / "rnd_snapshots"

SIGMA_MULTIPLIERS = [
    2.50,
    2.75,
    3.00,
    3.10,
    3.20,
    3.25,
    3.30,
    3.40,
    3.50,
]

TARGET_MIN_RAW_AREA = 0.99


# ============================================================
# LOAD SAVED LIVE SNAPSHOTS
# ============================================================

files = sorted(
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

if not files:
    raise RuntimeError(
        "No saved RND snapshots found."
    )


print()
print("=" * 145)
print(
    "DEALEROS RND — VOLATILITY-SCALED RANGE POLICY"
)
print("=" * 145)

print(
    f"Snapshots       : {len(files)}"
)

print(
    f"Lambda          : {rnd_builder.RND_SPLINE_LAMBDA}"
)

print(
    f"Grid step       : {rnd_builder.RND_GRID_STEP}"
)

print(
    f"Target min area : {TARGET_MIN_RAW_AREA}"
)

print()


# ============================================================
# RESULTS STORAGE
# ============================================================

results = {}


for multiplier in SIGMA_MULTIPLIERS:

    bad_snapshots = 0
    bad_folds = 0

    full_mono = 0
    full_neg = 0

    cv_mono = 0
    cv_neg = 0

    areas = []
    ranges = []
    observation_counts = []

    failures = []

    for path in files:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            raw = json.load(f)

        snapshot = parse_snapshot(
            raw
        )

        if snapshot.atm_iv is None:
            raise RuntimeError(
                f"ATM IV missing in {path.name}"
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
            * math.exp(
                risk_free_rate
                * time_to_expiry
            )
        )

        # ----------------------------------------------------
        # Model-implied one-sigma move in index points.
        # ----------------------------------------------------

        sigma_move = (
            float(snapshot.spot)
            * (
                float(snapshot.atm_iv)
                / 100.0
            )
            * math.sqrt(
                time_to_expiry
            )
        )

        fit_range = (
            multiplier
            * sigma_move
        )

        strikes, ivs_pct = (
            get_observations(
                snapshot=snapshot,
                forward=forward,
                fit_range=fit_range,
            )
        )

        if len(strikes) < 10:

            bad_snapshots += 1

            failures.append(
                (
                    str(
                        snapshot.timestamp
                    )[:19],
                    fit_range,
                    "NOT_ENOUGH_OBSERVATIONS",
                )
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

        ranges.append(
            fit_range
        )

        areas.append(
            full["area"]
        )

        observation_counts.append(
            len(strikes)
        )

        full_mono += (
            full["mono"]
        )

        full_neg += (
            full["neg"]
        )

        bad_folds += (
            cv["bad_folds"]
        )

        cv_mono += (
            cv["mono"]
        )

        cv_neg += (
            cv["neg"]
        )

        full_clean = (
            full["clean"]
        )

        cv_clean = (
            cv["bad_folds"] == 0
            and cv["mono"] == 0
            and cv["neg"] == 0
        )

        if not (
            full_clean
            and cv_clean
        ):

            bad_snapshots += 1

            failures.append(
                (
                    str(
                        snapshot.timestamp
                    )[:19],
                    fit_range,
                    (
                        f"full_mono={full['mono']} "
                        f"full_neg={full['neg']} "
                        f"bad_cv={cv['bad_folds']} "
                        f"cv_mono={cv['mono']} "
                        f"cv_neg={cv['neg']}"
                    ),
                )
            )

    results[
        multiplier
    ] = {
        "bad_snapshots": (
            bad_snapshots
        ),
        "bad_folds": (
            bad_folds
        ),
        "full_mono": (
            full_mono
        ),
        "full_neg": (
            full_neg
        ),
        "cv_mono": (
            cv_mono
        ),
        "cv_neg": (
            cv_neg
        ),
        "avg_area": float(
            np.mean(
                areas
            )
        ),
        "min_area": float(
            np.min(
                areas
            )
        ),
        "max_area": float(
            np.max(
                areas
            )
        ),
        "avg_range": float(
            np.mean(
                ranges
            )
        ),
        "min_range": float(
            np.min(
                ranges
            )
        ),
        "max_range": float(
            np.max(
                ranges
            )
        ),
        "avg_obs": float(
            np.mean(
                observation_counts
            )
        ),
        "failures": failures,
    }


# ============================================================
# AGGREGATE TABLE
# ============================================================

print("=" * 145)
print("AGGREGATE SIGMA-RANGE RESULTS")
print("=" * 145)

print(
    f"{'SIGMA':>8}"
    f"{'AVG RNG':>10}"
    f"{'MIN RNG':>10}"
    f"{'MAX RNG':>10}"
    f"{'AVG OBS':>10}"
    f"{'AVG AREA':>11}"
    f"{'MIN AREA':>11}"
    f"{'BAD SNAP':>10}"
    f"{'BAD CV':>9}"
    f"{'FULL M':>9}"
    f"{'FULL N':>9}"
    f"{'CV M':>9}"
    f"{'CV N':>9}"
    f"{'STATUS':>10}"
)

print("-" * 145)


eligible = []


for multiplier in SIGMA_MULTIPLIERS:

    r = results[
        multiplier
    ]

    integrity_pass = (
        r["bad_snapshots"] == 0
        and r["bad_folds"] == 0
        and r["full_mono"] == 0
        and r["full_neg"] == 0
        and r["cv_mono"] == 0
        and r["cv_neg"] == 0
    )

    coverage_pass = (
        r["min_area"]
        >= TARGET_MIN_RAW_AREA
    )

    passed = (
        integrity_pass
        and coverage_pass
    )

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    if passed:
        eligible.append(
            multiplier
        )

    print(
        f"{multiplier:>8.2f}"
        f"{r['avg_range']:>10.1f}"
        f"{r['min_range']:>10.1f}"
        f"{r['max_range']:>10.1f}"
        f"{r['avg_obs']:>10.1f}"
        f"{r['avg_area']:>11.6f}"
        f"{r['min_area']:>11.6f}"
        f"{r['bad_snapshots']:>10}"
        f"{r['bad_folds']:>9}"
        f"{r['full_mono']:>9}"
        f"{r['full_neg']:>9}"
        f"{r['cv_mono']:>9}"
        f"{r['cv_neg']:>9}"
        f"{status:>10}"
    )


print("-" * 145)

print()
print("=" * 100)
print("POLICY RESULT")
print("=" * 100)

if eligible:

    selected = min(
        eligible
    )

    print(
        "Smallest sigma multiplier passing "
        "integrity + coverage:"
    )

    print(
        f"{selected:.2f} sigma"
    )

else:

    print(
        "No tested sigma multiplier passed "
        "both integrity and coverage."
    )


# ============================================================
# FAILURE DETAILS
# ============================================================

print()
print("=" * 100)
print("FAILURE DETAILS")
print("=" * 100)

any_failures = False

for multiplier in SIGMA_MULTIPLIERS:

    failures = results[
        multiplier
    ][
        "failures"
    ]

    if not failures:
        continue

    any_failures = True

    print()
    print(
        f"{multiplier:.2f} SIGMA"
    )

    for (
        timestamp,
        fit_range,
        reason,
    ) in failures:

        print(
            f"{timestamp}"
            f" | range={fit_range:.2f}"
            f" | {reason}"
        )


if not any_failures:
    print(
        "No structural failures."
    )

print()
print("=" * 100)
print("DONE")
print("=" * 100)
print()