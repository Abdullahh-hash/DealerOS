import json
import math

import numpy as np
from numpy.polynomial import Polynomial

from app.services.black_scholes import black_scholes_call


SNAPSHOT = r"data\snapshot_2026-08-10.json"

RATE = 0.04
CORE_RANGE = 1000.0
GRID_STEP = 5.0

DEGREES = [3, 4, 5, 6]

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
# Collect IVs
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
# Build identical OTM-IV observations
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
    [item[0] for item in otm_points],
    dtype=float,
)

ivs = np.array(
    [item[1] for item in otm_points],
    dtype=float,
)


# --------------------------------------------------
# Same dense grid for every degree
# --------------------------------------------------

grid_min = (
    math.ceil(strikes.min() / GRID_STEP)
    * GRID_STEP
)

grid_max = (
    math.floor(strikes.max() / GRID_STEP)
    * GRID_STEP
)

grid = np.arange(
    grid_min,
    grid_max + GRID_STEP / 2.0,
    GRID_STEP,
)


# --------------------------------------------------
# Test one polynomial degree
# --------------------------------------------------

def test_degree(degree):

    poly = Polynomial.fit(
        strikes,
        ivs,
        deg=degree,
    )

    fitted_iv = poly(grid)

    original_fit = poly(strikes)

    errors = original_fit - ivs

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


    # ----------------------------------------------
    # Invalid fitted IV check
    # ----------------------------------------------

    if np.any(fitted_iv <= 0):

        print(
            f"{degree:6d} | "
            f"INVALID IV FIT"
        )

        return


    # ----------------------------------------------
    # IV -> BS calls
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
    # Monotonicity
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

    raw_area = float(
        np.trapezoid(
            rnd,
            rnd_strikes,
        )
    )


    # ----------------------------------------------
    # Distribution stats only if density passes
    # ----------------------------------------------

    mean = None
    mode = None
    median = None

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

        mode = float(
            rnd_strikes[
                int(np.argmax(pdf))
            ]
        )


        cdf = np.zeros_like(pdf)

        increments = (
            0.5
            * (pdf[:-1] + pdf[1:])
            * np.diff(rnd_strikes)
        )

        cdf[1:] = np.cumsum(
            increments
        )

        cdf = cdf / cdf[-1]

        median = float(
            np.interp(
                0.50,
                cdf,
                rnd_strikes,
            )
        )


    # ----------------------------------------------
    # Output
    # ----------------------------------------------

    print()
    print("=" * 80)
    print(f"POLYNOMIAL DEGREE {degree}")
    print("=" * 80)

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
        f"Raw density area        : "
        f"{raw_area}"
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
            f"{mode}"
        )

    else:

        print(
            "Probability stats       : "
            "NOT CALCULATED - density failed"
        )


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("=" * 80)
print("RND POLYNOMIAL-DEGREE SENSITIVITY")
print("=" * 80)

print(f"Spot                : {S}")
print(f"Forward             : {forward}")
print(f"DTE                 : {data['dte']}")
print(f"Range               : +/- {CORE_RANGE}")
print(f"Grid step           : {GRID_STEP}")
print(f"OTM observations    : {len(otm_points)}")


for degree in DEGREES:
    test_degree(degree)