import math


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        if option_type == "C":
            return max(S - K, 0.0)
        else:
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


# --------------------------------------------------
# Test using our actual NDX snapshot
# --------------------------------------------------

S = 29722.303

# 2 DTE from the snapshot
T = 2 / 365

# Temporary risk-free rate
r = 0.04

tests = [
    (29725.0, 0.1420, "C"),
    (29725.0, 0.1467, "P"),
    (29800.0, 0.1412, "C"),
    (29800.0, 0.1461, "P"),
    (30000.0, 0.1412, "C"),
]

print("=" * 60)
print("BLACK-SCHOLES PRICE TEST")
print("=" * 60)

print(f"Spot : {S}")
print(f"DTE  : 2")
print(f"Rate : {r}")
print()

for K, iv, option_type in tests:

    price = black_scholes_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=iv,
        option_type=option_type,
    )

    print(
        f"Strike : {K:8.1f} | "
        f"IV : {iv:.4f} | "
        f"Right : {option_type} | "
        f"BS Price : {price:.4f}"
    )