import json
import math

from app.services.black_scholes import black_scholes_call


# ============================================================
# LOAD SNAPSHOT
# ============================================================

with open(r"data\snapshot_2026-08-10.json", "r") as f:
    data = json.load(f)


spot = data["spot"]
dte = data["dte"]
rate = 0.04

T = dte / 365.0


# ============================================================
# BUILD CALL PRICE SURFACE
# ============================================================

call_prices = {}

for row in data["rows"]:

    if row["right"] != "C":
        continue

    iv_pct = row.get("iv_pct")

    if iv_pct is None:
        continue

    strike = row["strike"]

    # API gives IV as percentage:
    # 14.20 -> 0.1420
    volatility = iv_pct / 100.0

    price = black_scholes_call(
        spot=spot,
        strike=strike,
        time_to_expiry=T,
        volatility=volatility,
        risk_free_rate=rate,
    )

    call_prices[strike] = price


call_points = sorted(call_prices.items())


# ============================================================
# BUILD REGULAR STRIKE GRID
# ============================================================

# Use a regular 25-point strike grid.
# This removes the irregular spacing problem before
# calculating the second derivative.

min_strike = math.ceil(call_points[0][0] / 25.0) * 25.0
max_strike = math.floor(call_points[-1][0] / 25.0) * 25.0

regular_strikes = []

k = min_strike

while k <= max_strike:
    regular_strikes.append(k)
    k += 25.0


# ============================================================
# LINEAR INTERPOLATION OF CALL PRICES
# ============================================================

interpolated_calls = []

j = 0

for strike in regular_strikes:

    while (
        j < len(call_points) - 2
        and call_points[j + 1][0] < strike
    ):
        j += 1

    k1, c1 = call_points[j]
    k2, c2 = call_points[j + 1]

    if k2 == k1:
        continue

    weight = (strike - k1) / (k2 - k1)

    call_price = c1 + weight * (c2 - c1)

    interpolated_calls.append(
        (strike, call_price)
    )


# ============================================================
# CALCULATE SECOND DERIVATIVE
# ============================================================

rnd_points = []

h = 25.0

for i in range(1, len(interpolated_calls) - 1):

    k_prev, c_prev = interpolated_calls[i - 1]
    k, c = interpolated_calls[i]
    k_next, c_next = interpolated_calls[i + 1]

    # Regular grid:
    # h = 25 points

    d2c = (
        c_prev
        - 2.0 * c
        + c_next
    ) / (h * h)

    rnd = math.exp(rate * T) * d2c

    rnd_points.append(
        (k, rnd)
    )
# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("RND DENSITY TEST")
print("=" * 60)

print(f"Spot        : {spot}")
print(f"DTE         : {dte}")
print(f"Rate        : {rate}")
print(f"Raw call points        : {len(call_points)}")
print(f"Regular grid points    : {len(interpolated_calls)}")
print(f"RND points             : {len(rnd_points)}")

print()
print("First 30 RND points")
print("-" * 60)

for strike, density in rnd_points[:30]:

    print(
        f"Strike : {strike:8.1f} | "
        f"RND : {density:.10f}"
    )