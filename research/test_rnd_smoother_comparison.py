import json
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import make_smoothing_spline

from app.services.black_scholes import black_scholes_call


# --------------------------------------------------
# Fixed configuration
#
# We are changing ONLY the smoother.
# --------------------------------------------------

SNAPSHOT = r"data\snapshot_2026-08-10.json"

RATE = 0.04

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
# Build identical OTM-IV observations
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
                (K, sides["P"])
            )

    else:

        if "C" in sides:
            otm_points.append(
                (K, sides["C"])
            )


if len(otm_points) < 10:
    raise ValueError(
        "Not enough OTM IV observations."
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
# Sanity check:
# smoothing spline requires increasing x values.
# --------------------------------------------------

if np.any(np.diff(strikes) <= 0):
    raise ValueError(
        "OTM strikes are not strictly increasing."
    )


# --------------------------------------------------
# Common dense strike grid
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


# --------------------------------------------------
# METHOD A
#
# Degree-4 polynomial
# --------------------------------------------------

poly = Polynomial.fit(
    strikes,
    ivs,
    deg=POLY_DEGREE,
)

poly_iv_grid = poly(
    grid
)

poly_iv_original = poly(
    strikes
)


# --------------------------------------------------
# METHOD B
#
# Cubic smoothing spline
#
# lam=None:
# SciPy chooses smoothing parameter using GCV.
# --------------------------------------------------

spline = make_smoothing_spline(
    strikes,
    ivs,
    lam=None,
)

spline_iv_grid = spline(
    grid
)

spline_iv_original = spline(
    strikes
)


# --------------------------------------------------
# Reject invalid IV before pricing
# --------------------------------------------------

if np.any(poly_iv_grid <= 0):
    raise ValueError(
        "Polynomial produced non-positive IV."
    )

if np.any(spline_iv_grid <= 0):
    raise ValueError(
        "Smoothing spline produced non-positive IV."
    )


# --------------------------------------------------
# Helper:
# IV -> Black-Scholes call curve
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


poly_calls = build_call_curve(
    poly_iv_grid
)

spline_calls = build_call_curve(
    spline_iv_grid
)


# --------------------------------------------------
# Helper:
# build and audit RND
#
# NO clipping.
# --------------------------------------------------

def analyze_method(
    name,
    fitted_iv_grid,
    fitted_iv_original,
    call_curve,
):

    # ----------------------------------------------
    # IV fit quality
    # ----------------------------------------------

    errors = (
        fitted_iv_original - ivs
    )

    iv_rmse = math.sqrt(
        float(
            np.mean(
                errors ** 2
            )
        )
    )

    iv_max_error = float(
        np.max(
            np.abs(errors)
        )
    )


    # ----------------------------------------------
    # Call monotonicity
    # ----------------------------------------------

    mono_failures = int(
        np.sum(
            np.diff(call_curve) > TOL
        )
    )


    # ----------------------------------------------
    # Second derivative
    # ----------------------------------------------

    h = GRID_STEP

    d2 = (
        call_curve[:-2]
        - 2.0 * call_curve[1:-1]
        + call_curve[2:]
    ) / (h * h)

    rnd_strikes = grid[1:-1]

    raw_rnd = (
        math.exp(r * T)
        * d2
    )


    # ----------------------------------------------
    # Raw RND integrity
    # ----------------------------------------------

    negative_mask = (
        raw_rnd < -TOL
    )

    negative_count = int(
        np.sum(negative_mask)
    )

    min_index = int(
        np.argmin(raw_rnd)
    )

    max_index = int(
        np.argmax(raw_rnd)
    )

    raw_area = float(
        np.trapezoid(
            raw_rnd,
            rnd_strikes,
        )
    )


    result = {
        "name": name,
        "iv_grid": fitted_iv_grid,
        "calls": call_curve,
        "rnd_strikes": rnd_strikes,
        "raw_rnd": raw_rnd,
        "iv_rmse": iv_rmse,
        "iv_max_error": iv_max_error,
        "mono_failures": mono_failures,
        "negative_count": negative_count,
        "minimum_rnd": float(
            raw_rnd[min_index]
        ),
        "minimum_rnd_strike": float(
            rnd_strikes[min_index]
        ),
        "maximum_rnd": float(
            raw_rnd[max_index]
        ),
        "maximum_rnd_strike": float(
            rnd_strikes[max_index]
        ),
        "raw_area": raw_area,
        "valid_probability": False,
    }


    # ----------------------------------------------
    # Probability statistics only if valid
    # ----------------------------------------------

    if (
        mono_failures == 0
        and negative_count == 0
        and raw_area > 0
    ):

        pdf = (
            raw_rnd / raw_area
        )

        normalized_area = float(
            np.trapezoid(
                pdf,
                rnd_strikes,
            )
        )


        mean = float(
            np.trapezoid(
                rnd_strikes * pdf,
                rnd_strikes,
            )
        )


        variance = float(
            np.trapezoid(
                (
                    (rnd_strikes - mean) ** 2
                )
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

        cdf /= cdf[-1]


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

        q25 = float(
            np.interp(
                0.25,
                cdf,
                rnd_strikes,
            )
        )

        q75 = float(
            np.interp(
                0.75,
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


        result.update(
            {
                "valid_probability": True,
                "pdf": pdf,
                "normalized_area": normalized_area,
                "mean": mean,
                "median": median,
                "mode": mode,
                "std_dev": std_dev,
                "q05": q05,
                "q25": q25,
                "q75": q75,
                "q95": q95,
            }
        )


    return result


poly_result = analyze_method(
    name="DEGREE-4 POLYNOMIAL",
    fitted_iv_grid=poly_iv_grid,
    fitted_iv_original=poly_iv_original,
    call_curve=poly_calls,
)

spline_result = analyze_method(
    name="GCV SMOOTHING SPLINE",
    fitted_iv_grid=spline_iv_grid,
    fitted_iv_original=spline_iv_original,
    call_curve=spline_calls,
)


# --------------------------------------------------
# Print one method
# --------------------------------------------------

def print_method(result):

    print()
    print("=" * 80)
    print(result["name"])
    print("=" * 80)

    print(
        f"IV RMSE                 : "
        f"{result['iv_rmse'] * 100:.6f} vol points"
    )

    print(
        f"Maximum IV error        : "
        f"{result['iv_max_error'] * 100:.6f} vol points"
    )

    print(
        f"Minimum fitted IV       : "
        f"{result['iv_grid'].min() * 100:.6f}%"
    )

    print(
        f"Maximum fitted IV       : "
        f"{result['iv_grid'].max() * 100:.6f}%"
    )

    print(
        f"Call monotonic failures : "
        f"{result['mono_failures']}"
    )

    print(
        f"Negative RND points     : "
        f"{result['negative_count']}"
    )

    print(
        f"Minimum RND             : "
        f"{result['minimum_rnd']}"
    )

    print(
        f"Minimum RND strike      : "
        f"{result['minimum_rnd_strike']}"
    )

    print(
        f"Maximum RND             : "
        f"{result['maximum_rnd']}"
    )

    print(
        f"Maximum RND strike      : "
        f"{result['maximum_rnd_strike']}"
    )

    print(
        f"Raw density area        : "
        f"{result['raw_area']}"
    )


    if result["valid_probability"]:

        print(
            f"Normalized area         : "
            f"{result['normalized_area']}"
        )

        print(
            f"Mean                    : "
            f"{result['mean']}"
        )

        print(
            f"Mean - Forward          : "
            f"{result['mean'] - forward}"
        )

        print(
            f"Median                  : "
            f"{result['median']}"
        )

        print(
            f"Mode                    : "
            f"{result['mode']}"
        )

        print(
            f"Standard deviation      : "
            f"{result['std_dev']}"
        )

        print(
            f"5% quantile             : "
            f"{result['q05']}"
        )

        print(
            f"25% quantile            : "
            f"{result['q25']}"
        )

        print(
            f"75% quantile            : "
            f"{result['q75']}"
        )

        print(
            f"95% quantile            : "
            f"{result['q95']}"
        )

    else:

        print(
            "Probability statistics  : "
            "NOT CALCULATED - raw curve failed"
        )


# --------------------------------------------------
# Method comparison
# --------------------------------------------------

iv_difference = (
    poly_iv_grid
    - spline_iv_grid
)

call_difference = (
    poly_calls
    - spline_calls
)

rnd_difference = (
    poly_result["raw_rnd"]
    - spline_result["raw_rnd"]
)


iv_diff_rmse = math.sqrt(
    float(
        np.mean(
            iv_difference ** 2
        )
    )
)

iv_diff_max = float(
    np.max(
        np.abs(iv_difference)
    )
)


call_diff_rmse = math.sqrt(
    float(
        np.mean(
            call_difference ** 2
        )
    )
)

call_diff_max = float(
    np.max(
        np.abs(call_difference)
    )
)


rnd_diff_rmse = math.sqrt(
    float(
        np.mean(
            rnd_difference ** 2
        )
    )
)

rnd_diff_max = float(
    np.max(
        np.abs(rnd_difference)
    )
)


integrated_abs_rnd_difference = float(
    np.trapezoid(
        np.abs(rnd_difference),
        poly_result["rnd_strikes"],
    )
)


# --------------------------------------------------
# If both densities are valid:
# compare normalized PDFs directly.
# --------------------------------------------------

total_variation = None

if (
    poly_result["valid_probability"]
    and spline_result["valid_probability"]
):

    pdf_difference = np.abs(
        poly_result["pdf"]
        - spline_result["pdf"]
    )

    total_variation = float(
        0.5
        * np.trapezoid(
            pdf_difference,
            poly_result["rnd_strikes"],
        )
    )


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("=" * 80)
print("RND SMOOTHER COMPARISON")
print("=" * 80)

print(f"Spot                 : {S}")
print(f"Forward              : {forward}")
print(f"DTE                  : {data['dte']}")
print(f"Fit range            : +/- {CORE_RANGE}")
print(f"Grid step            : {GRID_STEP}")
print(f"OTM observations     : {len(otm_points)}")
print(f"Strike minimum       : {strikes.min()}")
print(f"Strike maximum       : {strikes.max()}")


print_method(
    poly_result
)

print_method(
    spline_result
)


print()
print("=" * 80)
print("POINT-BY-POINT DIFFERENCE")
print("=" * 80)

print(
    f"IV difference RMSE    : "
    f"{iv_diff_rmse * 100:.6f} vol points"
)

print(
    f"Maximum IV difference : "
    f"{iv_diff_max * 100:.6f} vol points"
)

print(
    f"Call-price diff RMSE  : "
    f"{call_diff_rmse:.6f} points"
)

print(
    f"Maximum call diff     : "
    f"{call_diff_max:.6f} points"
)

print(
    f"RND difference RMSE   : "
    f"{rnd_diff_rmse}"
)

print(
    f"Maximum RND difference: "
    f"{rnd_diff_max}"
)

print(
    f"Integrated abs RND diff: "
    f"{integrated_abs_rnd_difference}"
)


if total_variation is not None:

    print(
        f"PDF total variation   : "
        f"{total_variation}"
    )

else:

    print(
        "PDF total variation   : "
        "NOT CALCULATED"
    )


# --------------------------------------------------
# Distribution-stat difference
# --------------------------------------------------

print()
print("=" * 80)
print("DISTRIBUTION DIFFERENCE")
print("=" * 80)

if (
    poly_result["valid_probability"]
    and spline_result["valid_probability"]
):

    print(
        f"Mean difference        : "
        f"{spline_result['mean'] - poly_result['mean']}"
    )

    print(
        f"Median difference      : "
        f"{spline_result['median'] - poly_result['median']}"
    )

    print(
        f"Mode difference        : "
        f"{spline_result['mode'] - poly_result['mode']}"
    )

    print(
        f"Std-dev difference     : "
        f"{spline_result['std_dev'] - poly_result['std_dev']}"
    )

    print(
        f"5% quantile difference : "
        f"{spline_result['q05'] - poly_result['q05']}"
    )

    print(
        f"95% quantile difference: "
        f"{spline_result['q95'] - poly_result['q95']}"
    )

else:

    print(
        "Cannot compare probability statistics "
        "because at least one raw density failed."
    )


print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)