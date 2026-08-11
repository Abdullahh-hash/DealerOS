import json
import math

import numpy as np
from numpy.polynomial import Polynomial

from app.services.black_scholes import black_scholes_call


SNAPSHOT = r"data\snapshot_2026-08-10.json"

RATE = 0.04
GRID_STEP = 5.0
POLY_DEGREE = 4

RANGES = [
    500.0,
    750.0,
    1000.0,
    1250.0,
    1500.0,
]

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
# Collect IV by strike
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
# Test one range
# --------------------------------------------------

def test_range(core_range):

    # ----------------------------------------------
    # Build OTM IV observations
    # ----------------------------------------------

    otm_points = []

    for K in sorted(by_strike):

        if abs(K - S) > core_range:
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


    if len(otm_points) < 10:

        print()
        print("=" * 80)
        print(f"RANGE +/- {core_range}")
        print("=" * 80)
        print("NOT ENOUGH OTM OBSERVATIONS")

        return


    strikes = np.array(
        [item[0] for item in otm_points],
        dtype=float,
    )

    ivs = np.array(
        [item[1] for item in otm_points],
        dtype=float,
    )


    # ----------------------------------------------
    # Fit degree-4 IV curve
    # ----------------------------------------------

    poly = Polynomial.fit(
        strikes,
        ivs,
        deg=POLY_DEGREE,
    )


    # ----------------------------------------------
    # Dense grid
    # ----------------------------------------------

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

    fitted_iv = poly(grid)


    # ----------------------------------------------
    # Fit diagnostics
    # ----------------------------------------------

    fitted_original = poly(
        strikes
    )

    errors = (
        fitted_original - ivs
    )

    rmse = math.sqrt(
        float(
            np.mean(
                errors ** 2
            )
        )
    )

    max_error = float(
        np.max(
            np.abs(errors)
        )
    )


    if np.any(fitted_iv <= 0):

        print()
        print("=" * 80)
        print(f"RANGE +/- {core_range}")
        print("=" * 80)
        print("INVALID: NON-POSITIVE FITTED IV")

        return


    # ----------------------------------------------
    # IV -> calls
    # ----------------------------------------------

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


    # ----------------------------------------------
    # Call monotonicity
    # ----------------------------------------------

    mono_failures = int(
        np.sum(
            np.diff(calls) > TOL
        )
    )


    # ----------------------------------------------
    # RND
    # ----------------------------------------------

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

    negative_count = int(
        np.sum(
            rnd < -TOL
        )
    )

    minimum_rnd = float(
        np.min(rnd)
    )

    maximum_rnd = float(
        np.max(rnd)
    )

    mode_strike = float(
        rnd_strikes[
            int(np.argmax(rnd))
        ]
    )

    raw_area = float(
        np.trapezoid(
            rnd,
            rnd_strikes,
        )
    )


    # ----------------------------------------------
    # Distribution statistics
    # ----------------------------------------------

    mean = None
    median = None
    std_dev = None

    q05 = None
    q95 = None

    if (
        negative_count == 0
        and raw_area > 0
    ):

        pdf = rnd / raw_area

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

        std_dev = math.sqrt(
            max(variance, 0.0)
        )


        # ------------------------------------------
        # CDF
        # ------------------------------------------

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

        cdf = (
            cdf / cdf[-1]
        )


        median = float(
            np.interp(
                0.50,
                cdf,
                rnd_strikes,
            )
        )

        q05 = float(
            np.interp(
                0.05,
                cdf,
                rnd_strikes,
            )
        )

        q95 = float(
            np.interp(
                0.95,
                cdf,
                rnd_strikes,
            )
        )


    # ----------------------------------------------
    # Output
    # ----------------------------------------------

    print()
    print("=" * 80)
    print(f"RANGE +/- {core_range}")
    print("=" * 80)

    print(
        f"OTM observations        : "
        f"{len(otm_points)}"
    )

    print(
        f"Actual strike minimum   : "
        f"{strikes.min()}"
    )

    print(
        f"Actual strike maximum   : "
        f"{strikes.max()}"
    )

    print(
        f"IV RMSE                 : "
        f"{rmse * 100:.6f} vol points"
    )

    print(
        f"Maximum IV error        : "
        f"{max_error * 100:.6f} vol points"
    )

    print(
        f"Minimum fitted IV       : "
        f"{fitted_iv.min() * 100:.6f}%"
    )

    print(
        f"Maximum fitted IV       : "
        f"{fitted_iv.max() * 100:.6f}%"
    )

    print(
        f"Call monotonic failures : "
        f"{mono_failures}"
    )

    print(
        f"Negative RND points     : "
        f"{negative_count}"
    )

    print(
        f"Minimum RND             : "
        f"{minimum_rnd}"
    )

    print(
        f"Maximum RND             : "
        f"{maximum_rnd}"
    )

    print(
        f"Raw density area        : "
        f"{raw_area}"
    )

    print(
        f"Approx missing mass     : "
        f"{1.0 - raw_area}"
    )

    if mean is not None:

        print(
            f"Mean                    : "
            f"{mean}"
        )

        print(
            f"Mean - Forward          : "
            f"{mean - forward}"
        )

        print(
            f"Median                  : "
            f"{median}"
        )

        print(
            f"Mode                    : "
            f"{mode_strike}"
        )

        print(
            f"Standard deviation      : "
            f"{std_dev}"
        )

        print(
            f"5% quantile             : "
            f"{q05}"
        )

        print(
            f"95% quantile            : "
            f"{q95}"
        )

    else:

        print(
            "Distribution stats      : "
            "NOT CALCULATED"
        )


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("=" * 80)
print("RND RANGE SENSITIVITY")
print("=" * 80)

print(f"Spot              : {S}")
print(f"Forward           : {forward}")
print(f"DTE               : {data['dte']}")
print(f"Polynomial degree : {POLY_DEGREE}")
print(f"Grid step         : {GRID_STEP}")


for core_range in RANGES:
    test_range(core_range)