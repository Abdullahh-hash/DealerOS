import json
import math


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(S, K, T, r, sigma, right):
    if T <= 0 or sigma <= 0:
        return None, None

    d1 = (
        math.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    gamma = norm_pdf(d1) / (
        S * sigma * math.sqrt(T)
    )

    if right == "C":
        delta = norm_cdf(d1)
    else:
        delta = norm_cdf(d1) - 1.0

    return delta, gamma


# --------------------------------------------------
# Load snapshot
# --------------------------------------------------

with open("data/snapshot_2026-08-10.json", "r") as f:
    data = json.load(f)

S = data["spot"]
T = data["dte"] / 365.0
r = 0.04

print("=" * 80)
print("RND SURFACE CONSISTENCY AUDIT")
print("=" * 80)

print(f"Spot : {S}")
print(f"DTE  : {data['dte']}")
print(f"T    : {T}")
print(f"Rate : {r}")
print()


# --------------------------------------------------
# Inspect suspicious strikes
# --------------------------------------------------

suspicious_strikes = {
    29530,
    29720,
    30450,
    30510,
    30725,
    30775,
    30875,
    30975,
    31025,
    31100,
    31400,
    31600,
    31800,
    32300,
    32500,
    35000,
    36000,
    37000,
}


print("=" * 80)
print("SUSPICIOUS ROWS")
print("=" * 80)

for row in data["rows"]:

    strike = float(row["strike"])

    if strike not in suspicious_strikes:
        continue

    right = row["right"]
    iv_pct = row.get("iv_pct")

    if iv_pct is None:
        continue

    sigma = iv_pct / 100.0

    api_delta = row.get("delta")
    api_gamma = row.get("gamma")

    bs_delta, bs_gamma = bs_greeks(
        S=S,
        K=strike,
        T=T,
        r=r,
        sigma=sigma,
        right=right,
    )

    print()
    print(f"Strike       : {strike}")
    print(f"Right        : {right}")
    print(f"IV %         : {iv_pct}")
    print(f"IV decimal   : {sigma}")
    print(f"API delta    : {api_delta}")
    print(f"BS delta     : {bs_delta}")
    print(f"Delta error  : {None if bs_delta is None else api_delta - bs_delta}")
    print(f"API gamma    : {api_gamma}")
    print(f"BS gamma     : {bs_gamma}")
    print(f"Gamma error  : {None if bs_gamma is None else api_gamma - bs_gamma}")


# --------------------------------------------------
# Full Greek consistency sweep
# --------------------------------------------------

delta_failures = []
gamma_failures = []

for row in data["rows"]:

    strike = float(row["strike"])
    right = row["right"]
    iv_pct = row.get("iv_pct")

    if iv_pct is None:
        continue

    api_delta = row.get("delta")
    api_gamma = row.get("gamma")

    if api_delta is None or api_gamma is None:
        continue

    sigma = iv_pct / 100.0

    bs_delta, bs_gamma = bs_greeks(
        S=S,
        K=strike,
        T=T,
        r=r,
        sigma=sigma,
        right=right,
    )

    if bs_delta is not None:
        if abs(api_delta - bs_delta) > 0.005:
            delta_failures.append(
                (
                    strike,
                    right,
                    iv_pct,
                    api_delta,
                    bs_delta,
                    api_delta - bs_delta,
                )
            )

    if bs_gamma is not None:
        if abs(api_gamma - bs_gamma) > 0.00005:
            gamma_failures.append(
                (
                    strike,
                    right,
                    iv_pct,
                    api_gamma,
                    bs_gamma,
                    api_gamma - bs_gamma,
                )
            )


print()
print("=" * 80)
print("FULL GREEK CONSISTENCY")
print("=" * 80)

print(f"Delta failures : {len(delta_failures)}")
print(f"Gamma failures : {len(gamma_failures)}")

print()
print("Largest delta discrepancies:")

for item in sorted(
    delta_failures,
    key=lambda x: abs(x[-1]),
    reverse=True,
)[:10]:

    print(item)

print()
print("Largest gamma discrepancies:")

for item in sorted(
    gamma_failures,
    key=lambda x: abs(x[-1]),
    reverse=True,
)[:10]:

    print(item)


print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)