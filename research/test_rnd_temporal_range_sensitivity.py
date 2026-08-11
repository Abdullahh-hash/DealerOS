import json
import math
from pathlib import Path
from statistics import NormalDist, median

import numpy as np
from scipy.interpolate import make_smoothing_spline

from app.services.black_scholes import black_scholes_call


# ============================================================
# CONFIGURATION
# ============================================================

SNAPSHOT_DIR = Path("data") / "rnd_snapshots"

FIT_RANGES = [
    350.0,
    400.0,
    450.0,
    500.0,
    550.0,
    600.0,
    700.0,
]

GRID_STEP = 5.0

SPLINE_LAMBDA = 2e-3

X_SCALE = 1000.0

ATM_CALIBRATION_RANGE = 100.0

FOLDS = 5

MONO_TOL = 1e-10
RND_NEG_TOL = -1e-12


normal = NormalDist()


# ============================================================
# FREEFLOW MODEL INPUT CALIBRATION
# ============================================================

def calibrate_model_inputs(data):

    spot = float(data["spot"])

    inferred_times = []
    inferred_rates = []

    for row in data["rows"]:

        if row.get("right") != "C":
            continue

        strike = row.get("strike")
        iv_pct = row.get("iv_pct")
        delta = row.get("delta")
        gamma = row.get("gamma")
        vega = row.get("vega")

        if any(
            value is None
            for value in (
                strike,
                iv_pct,
                delta,
                gamma,
                vega,
            )
        ):
            continue

        strike = float(strike)
        sigma = float(iv_pct) / 100.0
        delta = float(delta)
        gamma = float(gamma)
        vega = float(vega)

        if abs(strike - spot) > ATM_CALIBRATION_RANGE:
            continue

        if sigma <= 0:
            continue

        if gamma <= 0:
            continue

        if vega <= 0:
            continue

        if not (0.0 < delta < 1.0):
            continue

        # ----------------------------------------------------
        # Infer T from FreeFlow gamma / vega.
        # Vega is per 1 volatility percentage point.
        # ----------------------------------------------------

        T = (
            100.0
            * vega
            / (
                gamma
                * spot
                * spot
                * sigma
            )
        )

        if T <= 0:
            continue

        # ----------------------------------------------------
        # Infer r from call delta assuming q = 0.
        # ----------------------------------------------------

        d1 = normal.inv_cdf(delta)

        r = (
            (
                d1
                * sigma
                * math.sqrt(T)
                - math.log(spot / strike)
            )
            / T
            - 0.5 * sigma * sigma
        )

        inferred_times.append(T)
        inferred_rates.append(r)

    if not inferred_times:
        raise RuntimeError(
            "Could not infer model T."
        )

    if not inferred_rates:
        raise RuntimeError(
            "Could not infer model rate."
        )

    return (
        median(inferred_times),
        median(inferred_rates),
    )


# ============================================================
# BUILD OTM IV OBSERVATIONS
# ============================================================

def build_otm_observations(
    data,
    forward,
    fit_range,
):

    strikes = []
    ivs = []

    for row in data["rows"]:

        strike = row.get("strike")
        right = row.get("right")
        iv_pct = row.get("iv_pct")

        if strike is None:
            continue

        if right is None:
            continue

        if iv_pct is None:
            continue

        strike = float(strike)
        iv_pct = float(iv_pct)

        if iv_pct <= 0:
            continue

        if abs(strike - forward) > fit_range:
            continue

        # ----------------------------------------------------
        # OTM source construction:
        #
        # put below forward
        # call above forward
        # ----------------------------------------------------

        if (
            strike < forward
            and right == "P"
        ):
            use_row = True

        elif (
            strike > forward
            and right == "C"
        ):
            use_row = True

        else:
            use_row = False

        if not use_row:
            continue

        strikes.append(strike)
        ivs.append(iv_pct)

    if len(strikes) < 10:
        raise RuntimeError(
            "Not enough OTM observations."
        )

    order = np.argsort(strikes)

    strikes = np.asarray(
        strikes,
        dtype=float,
    )[order]

    ivs = np.asarray(
        ivs,
        dtype=float,
    )[order]

    # Defensive uniqueness
    unique_strikes, unique_indices = np.unique(
        strikes,
        return_index=True,
    )

    strikes = unique_strikes
    ivs = ivs[unique_indices]

    return strikes, ivs


# ============================================================
# SPLINE
# ============================================================

def fit_spline(
    train_strikes,
    train_ivs,
    eval_strikes,
    forward,
):

    x_train = (
        train_strikes - forward
    ) / X_SCALE

    x_eval = (
        eval_strikes - forward
    ) / X_SCALE

    spline = make_smoothing_spline(
        x_train,
        train_ivs,
        lam=SPLINE_LAMBDA,
    )

    return spline(
        x_eval
    )


# ============================================================
# RND
# ============================================================

def build_rnd(
    train_strikes,
    train_ivs,
    grid,
    spot,
    forward,
    T,
    rate,
):

    fitted_iv_pct = fit_spline(
        train_strikes=train_strikes,
        train_ivs=train_ivs,
        eval_strikes=grid,
        forward=forward,
    )

    if not np.all(
        np.isfinite(
            fitted_iv_pct
        )
    ):
        return {
            "valid": False,
            "mono": 0,
            "neg": 0,
            "area": float("nan"),
            "mean_fwd": None,
            "std": None,
        }

    if np.any(
        fitted_iv_pct <= 0
    ):
        return {
            "valid": False,
            "mono": 0,
            "neg": 0,
            "area": float("nan"),
            "mean_fwd": None,
            "std": None,
        }

    call_prices = np.array(
        [
            black_scholes_call(
                spot=spot,
                strike=float(K),
                time_to_expiry=T,
                volatility=float(iv) / 100.0,
                risk_free_rate=rate,
            )
            for K, iv
            in zip(
                grid,
                fitted_iv_pct,
            )
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Call monotonicity
    # --------------------------------------------------------

    mono = int(
        np.sum(
            np.diff(
                call_prices
            )
            > MONO_TOL
        )
    )

    # --------------------------------------------------------
    # Breeden-Litzenberger
    # --------------------------------------------------------

    first_derivative = np.gradient(
        call_prices,
        grid,
    )

    second_derivative = np.gradient(
        first_derivative,
        grid,
    )

    density = (
        np.exp(rate * T)
        * second_derivative
    )

    neg = int(
        np.sum(
            density < RND_NEG_TOL
        )
    )

    area = float(
        np.trapezoid(
            density,
            grid,
        )
    )

    clean = (
        mono == 0
        and neg == 0
        and area > 0
        and np.isfinite(area)
    )

    if not clean:

        return {
            "valid": True,
            "mono": mono,
            "neg": neg,
            "area": area,
            "mean_fwd": None,
            "std": None,
        }

    # --------------------------------------------------------
    # Normalize only AFTER integrity passes.
    # No clipping.
    # --------------------------------------------------------

    normalized = (
        density / area
    )

    mean_value = float(
        np.trapezoid(
            grid * normalized,
            grid,
        )
    )

    variance = float(
        np.trapezoid(
            (
                (grid - mean_value) ** 2
            )
            * normalized,
            grid,
        )
    )

    std = math.sqrt(
        max(
            variance,
            0.0,
        )
    )

    return {
        "valid": True,
        "mono": mono,
        "neg": neg,
        "area": area,
        "mean_fwd": (
            mean_value - forward
        ),
        "std": std,
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate(
    strikes,
    ivs,
    grid,
    spot,
    forward,
    T,
    rate,
):

    n = len(strikes)

    interior_indices = np.arange(
        1,
        n - 1,
        dtype=int,
    )

    all_errors = []

    bad_folds = 0

    for fold in range(FOLDS):

        test_indices = (
            interior_indices[
                np.arange(
                    len(
                        interior_indices
                    )
                )
                % FOLDS
                == fold
            ]
        )

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

        train_strikes = (
            strikes[
                train_indices
            ]
        )

        train_ivs = (
            ivs[
                train_indices
            ]
        )

        test_strikes = (
            strikes[
                test_indices
            ]
        )

        test_ivs = (
            ivs[
                test_indices
            ]
        )

        predicted = fit_spline(
            train_strikes=train_strikes,
            train_ivs=train_ivs,
            eval_strikes=test_strikes,
            forward=forward,
        )

        errors = (
            predicted
            - test_ivs
        )

        all_errors.extend(
            errors.tolist()
        )

        fold_rnd = build_rnd(
            train_strikes=train_strikes,
            train_ivs=train_ivs,
            grid=grid,
            spot=spot,
            forward=forward,
            T=T,
            rate=rate,
        )

        fold_bad = (
            not fold_rnd["valid"]
            or fold_rnd["mono"] != 0
            or fold_rnd["neg"] != 0
            or not np.isfinite(
                fold_rnd["area"]
            )
            or fold_rnd["area"] <= 0
        )

        if fold_bad:
            bad_folds += 1

    errors = np.asarray(
        all_errors,
        dtype=float,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                errors
            )
        )
    )

    max_error = float(
        np.max(
            np.abs(
                errors
            )
        )
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "bad_folds": bad_folds,
        "errors": errors,
    }


# ============================================================
# MAIN
# ============================================================

files = sorted(
    SNAPSHOT_DIR.glob(
        "snapshot_*.json"
    )
)

if not files:
    raise RuntimeError(
        "No temporal snapshots found."
    )


print()
print("=" * 130)
print(
    "DEALEROS RND — TEMPORAL RANGE SENSITIVITY"
)
print("=" * 130)

print(
    f"Snapshots       : {len(files)}"
)

print(
    f"Lambda          : {SPLINE_LAMBDA}"
)

print(
    f"Grid step       : {GRID_STEP}"
)

print(
    f"CV folds        : {FOLDS}"
)

print(
    "Ranges          : "
    + ", ".join(
        f"+/-{int(x)}"
        for x in FIT_RANGES
    )
)

print()


results = {}


for fit_range in FIT_RANGES:

    all_errors = []

    bad_folds_total = 0
    bad_snapshots = 0

    full_mono_total = 0
    full_neg_total = 0

    areas = []
    observation_counts = []
    mean_fwd_values = []
    std_values = []

    failures = []

    for path in files:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        spot = float(
            data["spot"]
        )

        T, rate = (
            calibrate_model_inputs(
                data
            )
        )

        forward = (
            spot
            * math.exp(
                rate * T
            )
        )

        strikes, ivs = (
            build_otm_observations(
                data=data,
                forward=forward,
                fit_range=fit_range,
            )
        )

        observation_counts.append(
            len(strikes)
        )

        grid = np.arange(
            strikes[0],
            strikes[-1]
            + GRID_STEP * 0.5,
            GRID_STEP,
            dtype=float,
        )

        cv = cross_validate(
            strikes=strikes,
            ivs=ivs,
            grid=grid,
            spot=spot,
            forward=forward,
            T=T,
            rate=rate,
        )

        full = build_rnd(
            train_strikes=strikes,
            train_ivs=ivs,
            grid=grid,
            spot=spot,
            forward=forward,
            T=T,
            rate=rate,
        )

        all_errors.extend(
            cv["errors"].tolist()
        )

        bad_folds_total += (
            cv["bad_folds"]
        )

        if full["valid"]:

            full_mono_total += (
                full["mono"]
            )

            full_neg_total += (
                full["neg"]
            )

        full_bad = (
            not full["valid"]
            or full["mono"] != 0
            or full["neg"] != 0
            or not np.isfinite(
                full["area"]
            )
            or full["area"] <= 0
        )

        if full_bad:

            bad_snapshots += 1

            failures.append(
                (
                    str(
                        data.get(
                            "timestamp",
                            path.name,
                        )
                    )[:19],
                    cv["bad_folds"],
                    full["mono"],
                    full["neg"],
                    full["area"],
                )
            )

        if (
            np.isfinite(
                full["area"]
            )
            and full["area"] > 0
        ):

            areas.append(
                full["area"]
            )

        if (
            full["mean_fwd"]
            is not None
        ):

            mean_fwd_values.append(
                full["mean_fwd"]
            )

        if (
            full["std"]
            is not None
        ):

            std_values.append(
                full["std"]
            )

    errors = np.asarray(
        all_errors,
        dtype=float,
    )

    results[
        fit_range
    ] = {
        "cv_rmse": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),
        "cv_mae": float(
            np.mean(
                np.abs(
                    errors
                )
            )
        ),
        "max_error": float(
            np.max(
                np.abs(
                    errors
                )
            )
        ),
        "bad_folds": bad_folds_total,
        "bad_snapshots": bad_snapshots,
        "mono": full_mono_total,
        "neg": full_neg_total,
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
        "avg_obs": float(
            np.mean(
                observation_counts
            )
        ),
        "avg_abs_mean_fwd": (
            float(
                np.mean(
                    np.abs(
                        mean_fwd_values
                    )
                )
            )
            if mean_fwd_values
            else float("nan")
        ),
        "avg_std": (
            float(
                np.mean(
                    std_values
                )
            )
            if std_values
            else float("nan")
        ),
        "failures": failures,
    }


# ============================================================
# AGGREGATE TABLE
# ============================================================

print("=" * 145)
print("AGGREGATE RANGE RESULTS")
print("=" * 145)

print(
    f"{'RANGE':>8}"
    f"{'AVG OBS':>10}"
    f"{'CV RMSE':>11}"
    f"{'CV MAE':>11}"
    f"{'MAX ERR':>11}"
    f"{'BAD FOLDS':>11}"
    f"{'BAD SNAP':>10}"
    f"{'MONO':>8}"
    f"{'NEG':>8}"
    f"{'AVG AREA':>11}"
    f"{'MIN AREA':>11}"
    f"{'MAX AREA':>11}"
    f"{'AVG |M-F|':>12}"
    f"{'AVG STD':>10}"
)

print("-" * 145)


for fit_range in FIT_RANGES:

    r = results[
        fit_range
    ]

    print(
        f"{fit_range:>8.0f}"
        f"{r['avg_obs']:>10.1f}"
        f"{r['cv_rmse']:>11.6f}"
        f"{r['cv_mae']:>11.6f}"
        f"{r['max_error']:>11.6f}"
        f"{r['bad_folds']:>11}"
        f"{r['bad_snapshots']:>10}"
        f"{r['mono']:>8}"
        f"{r['neg']:>8}"
        f"{r['avg_area']:>11.6f}"
        f"{r['min_area']:>11.6f}"
        f"{r['max_area']:>11.6f}"
        f"{r['avg_abs_mean_fwd']:>12.4f}"
        f"{r['avg_std']:>10.4f}"
    )


print("-" * 145)


# ============================================================
# STABILITY SUMMARY
# ============================================================

print()
print("=" * 100)
print("STABILITY SUMMARY")
print("=" * 100)


stable_ranges = []


for fit_range in FIT_RANGES:

    r = results[
        fit_range
    ]

    stable = (
        r["bad_folds"] == 0
        and r["bad_snapshots"] == 0
        and r["mono"] == 0
        and r["neg"] == 0
    )

    status = (
        "PASS"
        if stable
        else "FAIL"
    )

    print(
        f"+/- {fit_range:>4.0f} : "
        f"{status}"
        f" | bad folds={r['bad_folds']}"
        f" | bad snapshots={r['bad_snapshots']}"
        f" | mono={r['mono']}"
        f" | neg={r['neg']}"
        f" | avg area={r['avg_area']:.6f}"
    )

    if stable:
        stable_ranges.append(
            fit_range
        )


print()

if stable_ranges:

    largest_stable = max(
        stable_ranges
    )

    print(
        "Largest tested stable range: "
        f"+/- {largest_stable:.0f}"
    )

else:

    print(
        "No tested range passed "
        "the full integrity gate."
    )


# ============================================================
# FAILURE DETAILS
# ============================================================

print()
print("=" * 100)
print("FAILURE DETAILS")
print("=" * 100)


any_failures = False


for fit_range in FIT_RANGES:

    failures = results[
        fit_range
    ][
        "failures"
    ]

    if not failures:
        continue

    any_failures = True

    print()
    print(
        f"RANGE +/- {fit_range:.0f}"
    )

    for (
        api_time,
        bad_folds,
        mono,
        neg,
        area,
    ) in failures:

        print(
            f"{api_time}"
            f" | CV bad folds={bad_folds}"
            f" | mono={mono}"
            f" | neg={neg}"
            f" | area={area:.6f}"
        )


if not any_failures:

    print(
        "No full-snapshot failures."
    )


print()
print("=" * 100)
print("DONE")
print("=" * 100)
print()