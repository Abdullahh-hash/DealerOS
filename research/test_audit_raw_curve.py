import json
import math


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        if option_type == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma <= 0:
        if option_type == "C":
            return max(S - K * math.exp(-r * T), 0.0)
        return max(K * math.exp(-r * T) - S, 0.0)

    sqrt_t = math.sqrt(T)

    d1 = (
        math.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * sqrt_t)

    d2 = d1 - sigma * sqrt_t

    if option_type == "C":
        return (
            S * norm_cdf(d1)
            - K * math.exp(-r * T) * norm_cdf(d2)
        )

    return (
        K * math.exp(-r * T) * norm_cdf(-d2)
        - S * norm_cdf(-d1)
    )


def put_to_call(S, K, T, r, put_price):
    return put_price + S - K * math.exp(-r * T)


# --------------------------------------------------
# Load exact snapshot
# --------------------------------------------------

with open("data/snapshot_2026-08-10.json", "r") as f:
    data = json.load(f)

S = float(data["spot"])
T = float(data["dte"]) / 365.0
r = 0.04

PRICE_TOL = 1e-9
CURVATURE_TOL = 1e-12


# --------------------------------------------------
# Build direct calls and parity-derived calls
# --------------------------------------------------

calls = {}
puts = {}

call_iv = {}
put_iv = {}

for row in data["rows"]:
    strike = float(row["strike"])
    right = row["right"]
    iv_pct = row.get("iv_pct")

    if iv_pct is None:
        continue

    iv_pct = float(iv_pct)
    sigma = iv_pct / 100.0

    price = black_scholes_price(
        S=S,
        K=strike,
        T=T,
        r=r,
        sigma=sigma,
        option_type=right,
    )

    if right == "C":
        calls[strike] = price
        call_iv[strike] = iv_pct

    elif right == "P":
        puts[strike] = price
        put_iv[strike] = iv_pct


# --------------------------------------------------
# Build EXACT SAME 301-point call surface
#
# Rule from current research test:
#   use direct call when available
#   otherwise use P -> C parity
# --------------------------------------------------

call_surface = {}
provenance = {}

for strike in sorted(set(calls) | set(puts)):
    if strike in calls:
        call_surface[strike] = calls[strike]
        provenance[strike] = "C"

    elif strike in puts:
        call_surface[strike] = put_to_call(
            S=S,
            K=strike,
            T=T,
            r=r,
            put_price=puts[strike],
        )
        provenance[strike] = "P->C"


points = sorted(call_surface.items())


# --------------------------------------------------
# Basic counts
# --------------------------------------------------

direct_used = sum(
    1 for strike in call_surface
    if provenance[strike] == "C"
)

parity_used = sum(
    1 for strike in call_surface
    if provenance[strike] == "P->C"
)


print("=" * 100)
print("CORRECT NON-UNIFORM RAW CALL-SURFACE AUDIT")
print("=" * 100)

print(f"Spot                    : {S}")
print(f"DTE                     : {data['dte']}")
print(f"T                       : {T}")
print(f"Rate                    : {r}")
print(f"API call rows           : {len(calls)}")
print(f"API put rows            : {len(puts)}")
print(f"Unified surface points  : {len(points)}")
print(f"Direct calls USED       : {direct_used}")
print(f"Parity P->C USED        : {parity_used}")
print()


# --------------------------------------------------
# Strike-gap audit
# --------------------------------------------------

gaps = []

for i in range(1, len(points)):
    k0 = points[i - 1][0]
    k1 = points[i][0]

    gaps.append(
        (
            k1 - k0,
            k0,
            k1,
        )
    )


print("=" * 100)
print("15 LARGEST RAW STRIKE GAPS")
print("=" * 100)

for gap, k0, k1 in sorted(
    gaps,
    reverse=True,
)[:15]:
    print(
        f"{k0:10.1f} -> {k1:10.1f} | "
        f"gap = {gap:8.1f}"
    )


# --------------------------------------------------
# TRUE monotonicity audit
#
# A call-price curve should be non-increasing:
#
# C(K_next) <= C(K_previous)
# --------------------------------------------------

monotonic_failures = []

for i in range(1, len(points)):
    k0, c0 = points[i - 1]
    k1, c1 = points[i]

    if c1 > c0 + PRICE_TOL:
        monotonic_failures.append(
            {
                "k0": k0,
                "c0": c0,
                "source0": provenance[k0],
                "k1": k1,
                "c1": c1,
                "source1": provenance[k1],
                "increase": c1 - c0,
            }
        )


print()
print("=" * 100)
print("TRUE MONOTONICITY FAILURES")
print("=" * 100)

print(f"Count: {len(monotonic_failures)}")
print()

for item in monotonic_failures:
    print(
        f"{item['k0']:8.1f} "
        f"{item['source0']:4s} "
        f"{item['c0']:12.6f}"
        f"  ->  "
        f"{item['k1']:8.1f} "
        f"{item['source1']:4s} "
        f"{item['c1']:12.6f}"
        f" | increase={item['increase']:.6f}"
    )


# --------------------------------------------------
# CORRECT convexity audit for NON-UNIFORM strikes
#
# For:
#
# K0 < K1 < K2
#
# slope_left  = (C1-C0)/(K1-K0)
# slope_right = (C2-C1)/(K2-K1)
#
# Convexity requires:
#
# slope_right >= slope_left
#
# Equivalent non-uniform second derivative:
#
# 2 * (slope_right - slope_left) / (h_left + h_right)
# --------------------------------------------------

convexity_failures = []

all_curvature = []

for i in range(1, len(points) - 1):
    k0, c0 = points[i - 1]
    k1, c1 = points[i]
    k2, c2 = points[i + 1]

    h_left = k1 - k0
    h_right = k2 - k1

    if h_left <= 0 or h_right <= 0:
        continue

    slope_left = (c1 - c0) / h_left
    slope_right = (c2 - c1) / h_right

    d2 = (
        2.0
        * (slope_right - slope_left)
        / (h_left + h_right)
    )

    raw_rnd = math.exp(r * T) * d2

    record = {
        "k0": k0,
        "k1": k1,
        "k2": k2,
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "source": provenance[k1],
        "h_left": h_left,
        "h_right": h_right,
        "slope_left": slope_left,
        "slope_right": slope_right,
        "d2": d2,
        "raw_rnd": raw_rnd,
    }

    all_curvature.append(record)

    if d2 < -CURVATURE_TOL:
        convexity_failures.append(record)


print()
print("=" * 100)
print("TRUE CONVEXITY FAILURES - NON-UNIFORM GRID")
print("=" * 100)

print(f"Count: {len(convexity_failures)}")
print()

for item in convexity_failures:
    print(
        f"K0={item['k0']:8.1f} "
        f"K1={item['k1']:8.1f} "
        f"K2={item['k2']:8.1f} | "
        f"{item['source']:4s} | "
        f"hL={item['h_left']:6.1f} "
        f"hR={item['h_right']:6.1f} | "
        f"slopeL={item['slope_left']: .8f} "
        f"slopeR={item['slope_right']: .8f} | "
        f"d2={item['d2']: .10f} | "
        f"rawRND={item['raw_rnd']: .10f}"
    )


# --------------------------------------------------
# Core NDX window around spot
# --------------------------------------------------

print()
print("=" * 100)
print("CURVATURE WITHIN +/- 1000 POINTS OF SPOT")
print("=" * 100)

for item in all_curvature:
    if abs(item["k1"] - S) <= 1000:
        status = (
            "FAIL"
            if item["d2"] < -CURVATURE_TOL
            else "OK"
        )

        print(
            f"K={item['k1']:8.1f} | "
            f"{item['source']:4s} | "
            f"hL={item['h_left']:5.1f} "
            f"hR={item['h_right']:5.1f} | "
            f"d2={item['d2']: .10f} | "
            f"RND={item['raw_rnd']: .10f} | "
            f"{status}"
        )


# --------------------------------------------------
# Compare direct call vs SAME-STRIKE P->C
#
# This does NOT alter the surface.
# It only measures disagreement where both exist.
# --------------------------------------------------

same_strike_differences = []

shared_strikes = sorted(
    set(calls) & set(puts)
)

for strike in shared_strikes:
    direct_call = calls[strike]

    parity_call = put_to_call(
        S=S,
        K=strike,
        T=T,
        r=r,
        put_price=puts[strike],
    )

    difference = parity_call - direct_call

    same_strike_differences.append(
        (
            abs(difference),
            strike,
            call_iv[strike],
            put_iv[strike],
            direct_call,
            parity_call,
            difference,
        )
    )


print()
print("=" * 100)
print("20 LARGEST SAME-STRIKE CALL vs P->C DIFFERENCES")
print("=" * 100)

for (
    abs_difference,
    strike,
    c_iv,
    p_iv,
    direct_call,
    parity_call,
    difference,
) in sorted(
    same_strike_differences,
    reverse=True,
)[:20]:

    print(
        f"K={strike:8.1f} | "
        f"C IV={c_iv:8.2f}% | "
        f"P IV={p_iv:8.2f}% | "
        f"Direct C={direct_call:12.6f} | "
        f"P->C={parity_call:12.6f} | "
        f"diff={difference: .6f}"
    )


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print()
print("=" * 100)
print("FINAL AUDIT SUMMARY")
print("=" * 100)

print(f"Surface points              : {len(points)}")
print(f"Direct calls used           : {direct_used}")
print(f"Parity-derived calls used   : {parity_used}")
print(f"Monotonicity failures       : {len(monotonic_failures)}")
print(f"True convexity failures     : {len(convexity_failures)}")
print(f"Shared C/P strikes compared : {len(shared_strikes)}")