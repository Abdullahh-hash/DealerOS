import json
import math

import numpy as np
from numpy.polynomial import Polynomial

from app.services.black_scholes import black_scholes_call


SNAPSHOT = r"data\snapshot_2026-08-10.json"

RATE = 0.04
CORE_RANGE = 1000.0
POLY_DEGREE = 4

GRID_STEPS = [
    2.5,
    5.0,
    10.0,
    20.0,
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
# Build fixed OTM-IV sample
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
    [x[0] for x in otm_points],
    dtype=float,
)

ivs = np.array(
    [x[1] for x in otm_points],
    dtype=float,
)


# --------------------------------------------------
# Fit ONCE
#
# Only grid spacing changes.
# --------------------------------------------------

poly = Polynomial.fit(
    strikes,
    ivs,
    deg=POLY_DEGREE,
)


# --------------------------------------------------
# Test one grid step
# --------------------------------------------------

def test_grid(grid_step):

    grid_min = (
        math.ceil(strikes.min() / grid_step)
        * grid_step
    )

    grid_max = (
        math.floor(strikes.max() / grid_step)
        * grid_step
    )

    grid = np.arange(
        grid_min,
        grid_max + grid_step / 2.0,
        grid_step,
    )


    fitted_iv = poly(grid)

    if np.any(fitted_iv <= 0):

        print(
            f"GRID {grid_step}: INVALID IV"
        )

        return


    # ----------------------------------------------
    # IV -> call price
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
    # Second derivative
    # ----------------------------------------------

    h = grid_step

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

    raw_area = float(
        np.trapezoid(
            rnd,
            rnd_strikes,
        )
    )


    # ----------------------------------------------
    # Probability stats
    # ----------------------------------------------

    mean = None
    median = None
    mode = None
    std_dev = None

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


    # ----------------------------------------------
    # Output
    # ----------------------------------------------

    print()
    print("=" * 80)
    print(f"GRID STEP {grid_step}")
    print("=" * 80)

    print(
        f"Grid points              : "
        f"{len(grid)}"
    )

    print(
        f"Call monotonic failures  : "
        f"{mono_failures}"
    )

    print(
        f"Negative RND points      : "
        f"{negative_count}"
    )

    print(
        f"Minimum RND              : "
        f"{float(np.min(rnd))}"
    )

    print(
        f"Maximum RND              : "
        f"{float(np.max(rnd))}"
    )

    print(
        f"Raw density area         : "
        f"{raw_area}"
    )

    if mean is not None:

        print(
            f"Mean                     : "
            f"{mean}"
        )

        print(
            f"Mean - Forward           : "
            f"{mean - forward}"
        )

        print(
            f"Median                   : "
            f"{median}"
        )

        print(
            f"Mode                     : "
            f"{mode}"
        )

        print(
            f"Standard deviation       : "
            f"{std_dev}"
        )


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("=" * 80)
print("RND GRID-STEP SENSITIVITY")
print("=" * 80)

print(f"Spot              : {S}")
print(f"Forward           : {forward}")
print(f"DTE               : {data['dte']}")
print(f"Fit range         : +/- {CORE_RANGE}")
print(f"Polynomial degree : {POLY_DEGREE}")
print(f"OTM observations  : {len(otm_points)}")


for grid_step in GRID_STEPS:
    test_grid(grid_step)