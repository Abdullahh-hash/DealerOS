import math
import json


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        if option_type == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (
        math.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

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
    """
    Put-call parity:

        C = P + S - K * exp(-rT)
    """

    return put_price + S - K * math.exp(-r * T)


# --------------------------------------------------
# Load snapshot
# --------------------------------------------------

with open("data/snapshot_2026-08-10.json", "r") as f:
    data = json.load(f)


S = data["spot"]

# Use the actual snapshot DTE
T = data["dte"] / 365

# Temporary rate
r = 0.04


# --------------------------------------------------
# Build call-price surface
# --------------------------------------------------

calls = {}
puts = {}

for row in data["rows"]:

    strike = row["strike"]
    right = row["right"]
    iv_pct = row.get("iv_pct")

    if iv_pct is None:
        continue

    # API gives IV in percentage form:
    # 14.20 -> 0.1420
    sigma = iv_pct / 100.0

    price = black_scholes_price(
        S=S,
        K=strike,
        T=T,
        r=r,
        sigma=sigma,
        option_type=right
    )
    

    if right == "C":
        calls[strike] = price

    elif right == "P":
        puts[strike] = price


# --------------------------------------------------
# Normalize into one call-price curve
# --------------------------------------------------

call_surface = {}

for strike in sorted(set(calls) | set(puts)):

    if strike in calls:

        # Direct call price
        call_surface[strike] = calls[strike]

    elif strike in puts:

        # Derive call from put-call parity
        call_surface[strike] = put_to_call(
            S=S,
            K=strike,
            T=T,
            r=r,
            put_price=puts[strike],
        )


# --------------------------------------------------
# Output
# --------------------------------------------------

print("=" * 60)
print("RND CALL SURFACE TEST")
print("=" * 60)

print(f"Spot        : {S}")
print(f"DTE         : {data['dte']}")
print(f"Rate        : {r}")
print(f"Call points : {len(call_surface)}")
print()

for strike, price in call_surface.items():

    if abs(strike - S) <= 500:

        print(
            f"Strike : {strike:8.1f} | "
            f"Call Price : {price:10.4f}"
        )