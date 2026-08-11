import json
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import make_smoothing_spline

from app.services.black_scholes import black_scholes_call


# --------------------------------------------------
# Fixed configuration
# --------------------------------------------------

SNAPSHOT = r"data\snapshot_2026-08-10.json"

RATE = 0.04
CORE_RANGE = 1000.0
GRID_STEP = 5.0

POLY_DEGREE = 4

SPLINE_LAMBDAS = [
    1e-3,
    1e-2,
    1e-1,
    1.0,
]

X_SCALE = 1000.0

FOLDS = 5

TOL = 1e-12


# --------------------------------------------------
# Load snapshot
# --------------------------------------------------

with open(SNAPSHOT, "r") as f:
    data = json.load(f)

S = float(data["spot"])
T = float(data["dte"]) / 365.0
r = RATE

forward = S * math.exp(r * T)


# --------------------------------------------------
# Collect IV by strike / right
# --------------------------------------------------

by_strike = {}

for row in data["rows"]:

    iv_pct = row.get("iv_pct")

    if iv_pct is None:
        continue

    if float(iv_pct) <= 0:
        continue

    K = float(row["strike"])
    right = row["right"]

    by_strike.setdefault(K, {})

    by_strike[K][right] = (
        float(iv_pct) / 100.0
    )


# --------------------------------------------------
# Same OTM IV selection
# --------------------------------------------------

otm_points = []

for K in sorted(by_strike):

    if abs(K - S) > CORE_RANGE:
        continue

    sides = by_strike[K]

    if K < forward:

        if "P" in sides:
            otm_points.append(
                (K, sides["P"])
            )

    else:

        if "C" in sides:
            otm_points.append(
                (K, sides["C"])
            )


strikes = np.array(
    [p[0] for p in otm_points],
    dtype=float,
)

ivs = np.array(
    [p[1] for p in otm_points],
    dtype=float,
)


if len(strikes) < 20:
    raise ValueError(
        "Not enough OTM IV observations."
    )


# --------------------------------------------------
# Normalized coordinate for splines
# --------------------------------------------------

x = (
    (strikes - forward)
    / X_SCALE
)


# --------------------------------------------------
# Common grid
# --------------------------------------------------

grid_min = (
    math.ceil(
        strikes.min() / GRID_STEP
    )
    * GRID_STEP
)

grid_max = (
    math.floor(
        strikes.max() / GRID_STEP
    )
    * GRID_STEP
)

grid = np.arange(
    grid_min,
    grid_max + GRID_STEP / 2.0,
    GRID_STEP,
)

grid_x = (
    (grid - forward)
    / X_SCALE
)


# --------------------------------------------------
# Build fold assignments
#
# First and last strikes always remain training
# points so we don't judge extrapolation.
# --------------------------------------------------

interior_indices = np.arange(
    1,
    len(strikes) - 1,
)

fold_assignment = {}

for n, index in enumerate(interior_indices):
    fold_assignment[index] = (
        n % FOLDS
    )


# --------------------------------------------------
# Fit helper
# --------------------------------------------------

def fit_method(
    method_name,
    train_indices,
    eval_strikes,
):

    train_strikes = strikes[
        train_indices
    ]

    train_ivs = ivs[
        train_indices
    ]


    # ----------------------------------------------
    # Polynomial
    # ----------------------------------------------

    if method_name == "POLY4":

        model = Polynomial.fit(
            train_strikes,
            train_ivs,
            deg=POLY_DEGREE,
        )

        return model(
            eval_strikes
        )


    # ----------------------------------------------
    # Spline
    # ----------------------------------------------

    if method_name.startswith("SPLINE_"):

        lam = float(
            method_name.split("_")[1]
        )

        train_x = (
            (train_strikes - forward)
            / X_SCALE
        )

        eval_x = (
            (eval_strikes - forward)
            / X_SCALE
        )

        model = make_smoothing_spline(
            train_x,
            train_ivs,
            lam=lam,
        )

        return model(
            eval_x
        )


    raise ValueError(
        f"Unknown method: {method_name}"
    )


# --------------------------------------------------
# RND integrity helper
# --------------------------------------------------

def rnd_integrity(
    fitted_iv,
):

    if np.any(
        fitted_iv <= 0
    ):
        return {
            "mono": 999,
            "negative": 999,
        }


    calls = np.array(
        [
            black_scholes_call(
                spot=S,
                strike=float(K),
                time_to_expiry=T,
                volatility=float(sigma),
                risk_free_rate=r,
            )
            for K, sigma in zip(
                grid,
                fitted_iv,
            )
        ],
        dtype=float,
    )


    mono = int(
        np.sum(
            np.diff(calls) > TOL
        )
    )


    h = GRID_STEP

    d2 = (
        calls[:-2]
        - 2.0 * calls[1:-1]
        + calls[2:]
    ) / (h * h)


    rnd = (
        math.exp(r * T)
        * d2
    )


    negative = int(
        np.sum(
            rnd < -TOL
        )
    )


    return {
        "mono": mono,
        "negative": negative,
    }


# --------------------------------------------------
# Full-distribution helper
# --------------------------------------------------

def full_distribution(
    method_name,
):

    all_indices = np.arange(
        len(strikes)
    )

    fitted_iv = fit_method(
        method_name,
        all_indices,
        grid,
    )


    calls = np.array(
        [
            black_scholes_call(
                spot=S,
                strike=float(K),
                time_to_expiry=T,
                volatility=float(sigma),
                risk_free_rate=r,
            )
            for K, sigma in zip(
                grid,
                fitted_iv,
            )
        ],
        dtype=float,
    )


    mono = int(
        np.sum(
            np.diff(calls) > TOL
        )
    )


    h = GRID_STEP

    d2 = (
        calls[:-2]
        - 2.0 * calls[1:-1]
        + calls[2:]
    ) / (h * h)


    rnd_strikes = grid[1:-1]

    rnd = (
        math.exp(r * T)
        * d2
    )


    negative = int(
        np.sum(
            rnd < -TOL
        )
    )


    area = float(
        np.trapezoid(
            rnd,
            rnd_strikes,
        )
    )


    if (
        mono != 0
        or negative != 0
        or area <= 0
    ):
        return {
            "mono": mono,
            "negative": negative,
            "area": area,
            "mean": None,
            "median": None,
            "mode": None,
            "std": None,
        }


    pdf = (
        rnd / area
    )


    mean = float(
        np.trapezoid(
            rnd_strikes * pdf,
            rnd_strikes,
        )
    )


    variance = float(
        np.trapezoid(
            ((rnd_strikes - mean) ** 2)
            * pdf,
            rnd_strikes,
        )
    )


    std = math.sqrt(
        max(
            variance,
            0.0,
        )
    )


    mode = float(
        rnd_strikes[
            int(np.argmax(pdf))
        ]
    )


    cdf = np.zeros_like(
        pdf
    )

    increments = (
        0.5
        * (pdf[:-1] + pdf[1:])
        * np.diff(rnd_strikes)
    )

    cdf[1:] = np.cumsum(
        increments
    )

    cdf /= cdf[-1]


    median = float(
        np.interp(
            0.50,
            cdf,
            rnd_strikes,
        )
    )


    return {
        "mono": mono,
        "negative": negative,
        "area": area,
        "mean": mean,
        "median": median,
        "mode": mode,
        "std": std,
    }


# --------------------------------------------------
# Methods
# --------------------------------------------------

methods = [
    "POLY4",
    "SPLINE_1e-3",
    "SPLINE_1e-2",
    "SPLINE_1e-1",
    "SPLINE_1",
]


# --------------------------------------------------
# Cross-validation
# --------------------------------------------------

cv_results = {}


for method in methods:

    squared_errors = []

    absolute_errors = []

    invalid_folds = 0

    fold_mono_total = 0

    fold_negative_total = 0


    for fold in range(FOLDS):

        test_indices = np.array(
            [
                i
                for i in interior_indices
                if fold_assignment[i] == fold
            ],
            dtype=int,
        )


        train_indices = np.array(
            [
                i
                for i in range(
                    len(strikes)
                )
                if i not in set(
                    test_indices
                )
            ],
            dtype=int,
        )


        # ------------------------------------------
        # Predict hidden IV observations
        # ------------------------------------------

        predicted = fit_method(
            method,
            train_indices,
            strikes[test_indices],
        )


        errors = (
            predicted
            - ivs[test_indices]
        )


        squared_errors.extend(
            (
                errors ** 2
            ).tolist()
        )

        absolute_errors.extend(
            np.abs(
                errors
            ).tolist()
        )


        # ------------------------------------------
        # Build complete smooth surface using
        # training strikes only
        # ------------------------------------------

        fitted_grid = fit_method(
            method,
            train_indices,
            grid,
        )


        integrity = rnd_integrity(
            fitted_grid
        )


        fold_mono_total += (
            integrity["mono"]
        )

        fold_negative_total += (
            integrity["negative"]
        )


        if (
            integrity["mono"] != 0
            or integrity["negative"] != 0
        ):
            invalid_folds += 1


    rmse = math.sqrt(
        float(
            np.mean(
                squared_errors
            )
        )
    )


    mae = float(
        np.mean(
            absolute_errors
        )
    )


    max_error = float(
        np.max(
            absolute_errors
        )
    )


    cv_results[method] = {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "invalid_folds": invalid_folds,
        "mono_total": fold_mono_total,
        "negative_total": fold_negative_total,
    }


# --------------------------------------------------
# Full-data distributions
# --------------------------------------------------

full_results = {
    method: full_distribution(
        method
    )
    for method in methods
}


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("=" * 115)
print("RND SMOOTHER OUT-OF-SAMPLE CROSS-VALIDATION")
print("=" * 115)

print(f"Spot                 : {S}")
print(f"Forward              : {forward}")
print(f"DTE                  : {data['dte']}")
print(f"Fit range            : +/- {CORE_RANGE}")
print(f"Grid step            : {GRID_STEP}")
print(f"OTM observations     : {len(strikes)}")
print(f"Cross-validation     : {FOLDS} folds")

print()


print("=" * 115)
print("OUT-OF-SAMPLE IV PREDICTION + FOLD RND INTEGRITY")
print("=" * 115)

print(
    f"{'METHOD':>14s} "
    f"{'CV_RMSE':>12s} "
    f"{'CV_MAE':>12s} "
    f"{'MAX_ERR':>12s} "
    f"{'BAD_FOLDS':>12s} "
    f"{'MONO_TOTAL':>12s} "
    f"{'NEG_TOTAL':>12s}"
)


for method in methods:

    result = cv_results[
        method
    ]

    print(
        f"{method:>14s} "
        f"{result['rmse'] * 100:12.6f} "
        f"{result['mae'] * 100:12.6f} "
        f"{result['max_error'] * 100:12.6f} "
        f"{result['invalid_folds']:12d} "
        f"{result['mono_total']:12d} "
        f"{result['negative_total']:12d}"
    )


print()
print("=" * 115)
print("FULL-DATA RND")
print("=" * 115)

print(
    f"{'METHOD':>14s} "
    f"{'MONO':>8s} "
    f"{'NEG':>8s} "
    f"{'AREA':>12s} "
    f"{'MEAN':>14s} "
    f"{'MEAN-FWD':>12s} "
    f"{'MEDIAN':>14s} "
    f"{'MODE':>10s} "
    f"{'STD':>12s}"
)


for method in methods:

    result = full_results[
        method
    ]


    if result["mean"] is None:

        print(
            f"{method:>14s} "
            f"{result['mono']:8d} "
            f"{result['negative']:8d} "
            f"{result['area']:12.6f} "
            f"{'INVALID':>14s}"
        )

        continue


    print(
        f"{method:>14s} "
        f"{result['mono']:8d} "
        f"{result['negative']:8d} "
        f"{result['area']:12.6f} "
        f"{result['mean']:14.4f} "
        f"{result['mean'] - forward:12.4f} "
        f"{result['median']:14.4f} "
        f"{result['mode']:10.1f} "
        f"{result['std']:12.4f}"
    )


print()
print("=" * 115)
print("TEST COMPLETE")
print("=" * 115)