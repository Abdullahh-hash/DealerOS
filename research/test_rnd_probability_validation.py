import json
import math

import numpy as np
from numpy.polynomial import Polynomial

from app.services.black_scholes import black_scholes_call


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SNAPSHOT = r"data\snapshot_2026-08-10.json"

RATE = 0.04

# Same diagnostic window as the successful
# IV-smoothing test.
CORE_RANGE = 1000.0

GRID_STEP = 5.0

POLY_DEGREE = 4

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
# Collect IV by strike / option right
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
# Build OTM IV observations
#
# Below forward:
#     PUT IV
#
# At / above forward:
#     CALL IV
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
# Smooth IV curve
# --------------------------------------------------

poly = Polynomial.fit(
    strikes,
    ivs,
    deg=POLY_DEGREE,
)


# --------------------------------------------------
# Dense regular strike grid
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

smooth_iv = poly(grid)


if np.any(smooth_iv <= 0):
    raise ValueError(
        "Smooth IV fit produced non-positive IV."
    )


# --------------------------------------------------
# Convert smooth IV -> Black-Scholes calls
# --------------------------------------------------

call_prices = np.array(
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
            smooth_iv,
        )
    ],
    dtype=float,
)


# --------------------------------------------------
# Call-price monotonicity
# --------------------------------------------------

call_diffs = np.diff(
    call_prices
)

monotonic_failures = int(
    np.sum(call_diffs > TOL)
)


# --------------------------------------------------
# Breeden-Litzenberger density
#
# Uniform grid:
#
# d²C/dK² =
# (C[K-h] - 2C[K] + C[K+h]) / h²
#
# q(K) = exp(rT) * d²C/dK²
#
# IMPORTANT:
# NO negative clipping.
# --------------------------------------------------

h = GRID_STEP

d2 = (
    call_prices[:-2]
    - 2.0 * call_prices[1:-1]
    + call_prices[2:]
) / (h * h)

rnd_strikes = grid[1:-1]

raw_density = (
    math.exp(r * T)
    * d2
)


# --------------------------------------------------
# Raw-density diagnostics
# --------------------------------------------------

negative_mask = (
    raw_density < -TOL
)

negative_count = int(
    np.sum(negative_mask)
)

minimum_index = int(
    np.argmin(raw_density)
)

maximum_index = int(
    np.argmax(raw_density)
)

raw_area = float(
    np.trapezoid(
        raw_density,
        rnd_strikes,
    )
)


# --------------------------------------------------
# Stop if raw density is genuinely invalid
#
# We do NOT hide negatives by clipping.
# --------------------------------------------------

if negative_count > 0:

    print("=" * 80)
    print("RND PROBABILITY VALIDATION")
    print("=" * 80)

    print(f"Spot                    : {S}")
    print(f"Forward                 : {forward}")
    print(f"DTE                     : {data['dte']}")
    print(f"Raw density area        : {raw_area}")
    print(f"Negative RND points     : {negative_count}")
    print(
        f"Minimum RND             : "
        f"{raw_density[minimum_index]}"
    )
    print(
        f"Minimum RND strike      : "
        f"{rnd_strikes[minimum_index]}"
    )

    raise ValueError(
        "Raw density contains negative values. "
        "Probability statistics not calculated."
    )


if raw_area <= 0:
    raise ValueError(
        "Raw density area must be positive."
    )


# --------------------------------------------------
# Normalize ONLY AFTER raw-density validation
#
# This is legitimate probability normalization,
# not negative-value clipping.
# --------------------------------------------------

pdf = (
    raw_density
    / raw_area
)

normalized_area = float(
    np.trapezoid(
        pdf,
        rnd_strikes,
    )
)


# --------------------------------------------------
# Expected terminal value
#
# IMPORTANT:
# This is the CONDITIONAL / TRUNCATED mean over
# our +/-1000 diagnostic domain.
#
# For a complete full-support RND, the mean should
# correspond to the forward.
# --------------------------------------------------

mean = float(
    np.trapezoid(
        rnd_strikes * pdf,
        rnd_strikes,
    )
)

mean_minus_forward = (
    mean - forward
)

mean_minus_spot = (
    mean - S
)


# --------------------------------------------------
# Variance / standard deviation
# --------------------------------------------------

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


# --------------------------------------------------
# Mode
# --------------------------------------------------

mode_index = int(
    np.argmax(pdf)
)

mode = float(
    rnd_strikes[mode_index]
)


# --------------------------------------------------
# Build numerical CDF
# --------------------------------------------------

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

if cdf[-1] <= 0:
    raise ValueError(
        "CDF has non-positive total probability."
    )

cdf = (
    cdf / cdf[-1]
)


# --------------------------------------------------
# Percentile helper
# --------------------------------------------------

def percentile(probability):

    return float(
        np.interp(
            probability,
            cdf,
            rnd_strikes,
        )
    )


median = percentile(0.50)

q25 = percentile(0.25)

q75 = percentile(0.75)

q05 = percentile(0.05)

q95 = percentile(0.95)


# --------------------------------------------------
# CDF probability helper
# --------------------------------------------------

def probability_below(price):

    if price <= rnd_strikes[0]:
        return 0.0

    if price >= rnd_strikes[-1]:
        return 1.0

    return float(
        np.interp(
            price,
            rnd_strikes,
            cdf,
        )
    )


prob_below_spot = probability_below(
    S
)

prob_above_spot = (
    1.0 - prob_below_spot
)

prob_below_forward = probability_below(
    forward
)

prob_above_forward = (
    1.0 - prob_below_forward
)


# --------------------------------------------------
# Mass omitted by diagnostic strike window
#
# This assumes a complete RND would integrate
# to approximately 1.
# --------------------------------------------------

missing_mass = (
    1.0 - raw_area
)


# --------------------------------------------------
# IV fit diagnostics
# --------------------------------------------------

original_fit = poly(
    strikes
)

iv_errors = (
    original_fit - ivs
)

iv_rmse = math.sqrt(
    float(
        np.mean(
            iv_errors ** 2
        )
    )
)

max_iv_error = float(
    np.max(
        np.abs(iv_errors)
    )
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("=" * 80)
print("RND PROBABILITY VALIDATION")
print("=" * 80)

print(f"Spot                       : {S}")
print(f"Forward                    : {forward}")
print(f"DTE                        : {data['dte']}")
print(f"Time                       : {T}")
print(f"Rate                       : {r}")

print()

print(f"OTM IV observations        : {len(otm_points)}")
print(f"Polynomial degree          : {POLY_DEGREE}")
print(f"Grid step                  : {GRID_STEP}")
print(f"RND grid points            : {len(rnd_strikes)}")

print()

print("=" * 80)
print("RAW RND INTEGRITY")
print("=" * 80)

print(
    f"Call monotonic failures     : "
    f"{monotonic_failures}"
)

print(
    f"Negative RND points         : "
    f"{negative_count}"
)

print(
    f"Minimum raw RND             : "
    f"{raw_density[minimum_index]}"
)

print(
    f"Minimum RND strike          : "
    f"{rnd_strikes[minimum_index]}"
)

print(
    f"Maximum raw RND             : "
    f"{raw_density[maximum_index]}"
)

print(
    f"Maximum RND strike          : "
    f"{rnd_strikes[maximum_index]}"
)

print(
    f"Raw density area            : "
    f"{raw_area}"
)

print(
    f"Approx missing mass         : "
    f"{missing_mass}"
)

print()

print("=" * 80)
print("NORMALIZED DISTRIBUTION")
print("=" * 80)

print(
    f"Normalized area             : "
    f"{normalized_area}"
)

print(
    f"Mean                        : "
    f"{mean}"
)

print(
    f"Forward                     : "
    f"{forward}"
)

print(
    f"Mean - Forward              : "
    f"{mean_minus_forward}"
)

print(
    f"Mean - Spot                 : "
    f"{mean_minus_spot}"
)

print(
    f"Mode                        : "
    f"{mode}"
)

print(
    f"Median                      : "
    f"{median}"
)

print(
    f"Standard deviation          : "
    f"{std_dev}"
)

print()

print("=" * 80)
print("DISTRIBUTION QUANTILES")
print("=" * 80)

print(f"5%                          : {q05}")
print(f"25%                         : {q25}")
print(f"50% / Median                : {median}")
print(f"75%                         : {q75}")
print(f"95%                         : {q95}")

print()

print("=" * 80)
print("DIRECTIONAL PROBABILITY")
print("=" * 80)

print(
    f"P(K <= Spot)                : "
    f"{prob_below_spot:.6f}"
)

print(
    f"P(K > Spot)                 : "
    f"{prob_above_spot:.6f}"
)

print(
    f"P(K <= Forward)             : "
    f"{prob_below_forward:.6f}"
)

print(
    f"P(K > Forward)              : "
    f"{prob_above_forward:.6f}"
)

print()

print("=" * 80)
print("IV FIT QUALITY")
print("=" * 80)

print(
    f"IV fit RMSE                 : "
    f"{iv_rmse * 100:.4f} vol points"
)

print(
    f"Maximum IV fit error        : "
    f"{max_iv_error * 100:.4f} vol points"
)

print(
    f"Minimum fitted IV           : "
    f"{smooth_iv.min() * 100:.4f}%"
)

print(
    f"Maximum fitted IV           : "
    f"{smooth_iv.max() * 100:.4f}%"
)

print()

print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)