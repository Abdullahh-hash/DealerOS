import json
import math


# --------------------------------------------------
# Black-Scholes
# --------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S, K, T, r, sigma, right):
    if T <= 0:
        if right == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_t = math.sqrt(T)

    d1 = (
        math.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * sqrt_t)

    d2 = d1 - sigma * sqrt_t

    if right == "C":
        return (
            S * norm_cdf(d1)
            - K * math.exp(-r * T) * norm_cdf(d2)
        )

    return (
        K * math.exp(-r * T) * norm_cdf(-d2)
        - S * norm_cdf(-d1)
    )


def put_to_call(S, K, T, r, put_price):
    return (
        put_price
        + S
        - K * math.exp(-r * T)
    )


# --------------------------------------------------
# Load exact snapshot
# --------------------------------------------------

with open("data/snapshot_2026-08-10.json", "r") as f:
    data = json.load(f)

S = float(data["spot"])
T = float(data["dte"]) / 365.0
r = 0.04

forward = S * math.exp(r * T)

TOL = 1e-12


# --------------------------------------------------
# Build direct call and parity-call dictionaries
# --------------------------------------------------

calls = {}
parity_calls = {}

for row in data["rows"]:

    iv_pct = row.get("iv_pct")

    if iv_pct is None:
        continue

    K = float(row["strike"])
    right = row["right"]

    sigma = float(iv_pct) / 100.0

    price = black_scholes_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        right=right,
    )

    if right == "C":

        calls[K] = price

    elif right == "P":

        parity_calls[K] = put_to_call(
            S=S,
            K=K,
            T=T,
            r=r,
            put_price=price,
        )


# --------------------------------------------------
# Surface 1
#
# CURRENT METHOD
#
# Call first; only use put when call missing.
# --------------------------------------------------

current_surface = {}

for K in sorted(set(calls) | set(parity_calls)):

    if K in calls:
        current_surface[K] = calls[K]

    elif K in parity_calls:
        current_surface[K] = parity_calls[K]


# --------------------------------------------------
# Surface 2
#
# CALLS ONLY
# --------------------------------------------------

calls_only_surface = dict(calls)


# --------------------------------------------------
# Surface 3
#
# PUTS ONLY, transformed to calls
# --------------------------------------------------

puts_only_surface = dict(parity_calls)


# --------------------------------------------------
# Surface 4
#
# OTM SIDE
#
# Below forward:
#     use puts -> calls
#
# Above forward:
#     use direct calls
#
# No fallback.
# --------------------------------------------------

otm_surface = {}

all_strikes = sorted(
    set(calls) | set(parity_calls)
)

for K in all_strikes:

    if K < forward:

        if K in parity_calls:
            otm_surface[K] = parity_calls[K]

    else:

        if K in calls:
            otm_surface[K] = calls[K]


# --------------------------------------------------
# Audit function
# --------------------------------------------------

def audit_surface(name, surface):

    points = sorted(surface.items())

    monotonic_failures = []
    convexity_failures = []

    core_monotonic = []
    core_convexity = []

    worst_d2 = None
    worst_strike = None

    # ----------------------------------------------
    # Monotonicity
    # ----------------------------------------------

    for i in range(1, len(points)):

        k0, c0 = points[i - 1]
        k1, c1 = points[i]

        if c1 > c0 + TOL:

            monotonic_failures.append(k1)

            if (
                abs(k0 - S) <= 1000
                and abs(k1 - S) <= 1000
            ):
                core_monotonic.append(k1)

    # ----------------------------------------------
    # Non-uniform convexity
    # ----------------------------------------------

    for i in range(1, len(points) - 1):

        k0, c0 = points[i - 1]
        k1, c1 = points[i]
        k2, c2 = points[i + 1]

        h_left = k1 - k0
        h_right = k2 - k1

        slope_left = (
            (c1 - c0) / h_left
        )

        slope_right = (
            (c2 - c1) / h_right
        )

        d2 = (
            2.0
            * (slope_right - slope_left)
            / (h_left + h_right)
        )

        if worst_d2 is None or d2 < worst_d2:
            worst_d2 = d2
            worst_strike = k1

        if d2 < -TOL:

            convexity_failures.append(k1)

            if abs(k1 - S) <= 1000:
                core_convexity.append(k1)

    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Points                     : "
        f"{len(points)}"
    )

    print(
        f"Monotonicity failures      : "
        f"{len(monotonic_failures)}"
    )

    print(
        f"Convexity failures         : "
        f"{len(convexity_failures)}"
    )

    print(
        f"Core +/-1000 mono failures : "
        f"{len(core_monotonic)}"
    )

    print(
        f"Core +/-1000 conv failures : "
        f"{len(core_convexity)}"
    )

    print(
        f"Worst d2                   : "
        f"{worst_d2}"
    )

    print(
        f"Worst d2 strike            : "
        f"{worst_strike}"
    )

    print()


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("=" * 80)
print("RND SOURCE-SELECTION TEST")
print("=" * 80)

print(f"Spot    : {S}")
print(f"Forward : {forward}")
print(f"DTE     : {data['dte']}")
print()

audit_surface(
    "1. CURRENT HYBRID",
    current_surface,
)

audit_surface(
    "2. CALLS ONLY",
    calls_only_surface,
)

audit_surface(
    "3. PUTS ONLY -> CALLS",
    puts_only_surface,
)

audit_surface(
    "4. OTM PUTS BELOW FORWARD / CALLS ABOVE",
    otm_surface,
)