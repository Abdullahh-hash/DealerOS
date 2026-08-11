import json
import math

import numpy as np
from numpy.polynomial import Polynomial

from app.services.black_scholes import black_scholes_call


SNAPSHOT = r"data\snapshot_2026-08-10.json"

RATE = 0.04

# Diagnostic window only.
# This is NOT yet the final DealerOS RND range.
CORE_RANGE = 1000.0

# Dense strike grid for differentiation.
GRID_STEP = 5.0

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
# Collect call and put IV by strike
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
# Build ONE OTM-IV curve
#
# Below forward:
#     use put IV
#
# At/above forward:
#     use call IV
#
# IMPORTANT:
# We use the IVs here.
# We do NOT convert raw puts into call prices.
# --------------------------------------------------

otm_points = []

for K in sorted(by_strike):

    if abs(K - S) > CORE_RANGE:
        continue

    sides = by_strike[K]

    if K < forward:

        if "P" in sides:
            otm_points.append(
                (K, sides["P"], "P")
            )

    else:

        if "C" in sides:
            otm_points.append(
                (K, sides["C"], "C")
            )


if len(otm_points) < 10:
    raise ValueError(
        "Not enough OTM IV points."
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
# Build dense regular strike grid
# --------------------------------------------------

grid_min = math.ceil(
    strikes.min() / GRID_STEP
) * GRID_STEP

grid_max = math.floor(
    strikes.max() / GRID_STEP
) * GRID_STEP

grid = np.arange(
    grid_min,
    grid_max + GRID_STEP / 2.0,
    GRID_STEP,
)


# --------------------------------------------------
# METHOD 1:
# Linear interpolation of IV
#
# Still not our final solution.
# This is the baseline comparison.
# --------------------------------------------------

linear_iv = np.interp(
    grid,
    strikes,
    ivs,
)


# --------------------------------------------------
# METHOD 2:
# Smooth degree-4 IV fit
#
# Diagnostic only.
# This is NOT yet the final production smoother.
#
# Polynomial.fit scales the strike domain
# internally, which is numerically safer than
# fitting raw 29,000-style strike values directly.
# --------------------------------------------------

poly = Polynomial.fit(
    strikes,
    ivs,
    deg=4,
)

smooth_iv = poly(grid)


if np.any(smooth_iv <= 0):
    raise ValueError(
        "Smooth IV fit produced non-positive IV."
    )


# --------------------------------------------------
# Convert IV curve -> BS CALL curve
# --------------------------------------------------

def build_call_curve(iv_curve):

    return np.array(
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
                iv_curve,
            )
        ],
        dtype=float,
    )


linear_calls = build_call_curve(
    linear_iv
)

smooth_calls = build_call_curve(
    smooth_iv
)


# --------------------------------------------------
# Audit the resulting CALL curve / RND
# --------------------------------------------------

def audit_method(
    name,
    iv_curve,
    call_curve,
):

    # ----------------------------------------------
    # Monotonic call-price violations
    # ----------------------------------------------

    call_diffs = np.diff(
        call_curve
    )

    monotonic_failures = int(
        np.sum(
            call_diffs > TOL
        )
    )


    # ----------------------------------------------
    # Uniform-grid second derivative
    # ----------------------------------------------

    h = GRID_STEP

    d2 = (
        call_curve[:-2]
        - 2.0 * call_curve[1:-1]
        + call_curve[2:]
    ) / (h * h)

    rnd = (
        math.exp(r * T)
        * d2
    )

    rnd_strikes = grid[1:-1]


    negative_mask = (
        rnd < -TOL
    )

    negative_count = int(
        np.sum(negative_mask)
    )


    # ----------------------------------------------
    # Raw area
    #
    # Do NOT normalize.
    # Do NOT clip negative values.
    #
    # This is only a diagnostic window,
    # therefore the area is not expected
    # to equal 1.
    # ----------------------------------------------

    raw_area = np.trapezoid(
        rnd,
        rnd_strikes,
    )


    minimum_index = int(
        np.argmin(rnd)
    )

    maximum_index = int(
        np.argmax(rnd)
    )


    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Grid points              : "
        f"{len(grid)}"
    )

    print(
        f"Call monotonic failures  : "
        f"{monotonic_failures}"
    )

    print(
        f"Negative RND points      : "
        f"{negative_count}"
    )

    print(
        f"Minimum RND              : "
        f"{rnd[minimum_index]}"
    )

    print(
        f"Minimum RND strike       : "
        f"{rnd_strikes[minimum_index]}"
    )

    print(
        f"Maximum RND              : "
        f"{rnd[maximum_index]}"
    )

    print(
        f"Maximum RND strike       : "
        f"{rnd_strikes[maximum_index]}"
    )

    print(
        f"Raw density area         : "
        f"{raw_area}"
    )

    print(
        f"Minimum fitted IV        : "
        f"{iv_curve.min() * 100:.4f}%"
    )

    print(
        f"Maximum fitted IV        : "
        f"{iv_curve.max() * 100:.4f}%"
    )

    print()


# --------------------------------------------------
# Smooth-fit error at ORIGINAL observations
# --------------------------------------------------

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


# --------------------------------------------------
# Find source switch around forward
# --------------------------------------------------

below = [
    item
    for item in otm_points
    if item[0] < forward
]

above = [
    item
    for item in otm_points
    if item[0] >= forward
]


print()
print("=" * 80)
print("RND IV-SMOOTHING TEST")
print("=" * 80)

print(f"Spot                    : {S}")
print(f"Forward                 : {forward}")
print(f"DTE                     : {data['dte']}")
print(f"Diagnostic range        : +/- {CORE_RANGE}")
print(f"OTM IV input points     : {len(otm_points)}")
print(f"Dense grid step         : {GRID_STEP}")

print()

if below:
    print(
        "Nearest OTM put below F : "
        f"K={below[-1][0]} "
        f"IV={below[-1][1] * 100:.4f}%"
    )

if above:
    print(
        "Nearest OTM call above F: "
        f"K={above[0][0]} "
        f"IV={above[0][1] * 100:.4f}%"
    )

print()

print(
    f"Degree-4 fit RMSE       : "
    f"{rmse * 100:.4f} vol points"
)

print(
    f"Degree-4 max IV error   : "
    f"{max_error * 100:.4f} vol points"
)

print()


audit_method(
    "1. LINEAR INTERPOLATION OF OTM IV",
    linear_iv,
    linear_calls,
)

audit_method(
    "2. SMOOTH DEGREE-4 OTM IV FIT",
    smooth_iv,
    smooth_calls,
)