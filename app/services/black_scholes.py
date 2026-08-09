from math import erf, exp, log, pi, sqrt


def normal_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes call option price.

    volatility must be decimal form.
    Example:
        14.84% -> 0.1484
    """

    if spot <= 0:
        raise ValueError("Spot must be greater than zero.")

    if strike <= 0:
        raise ValueError("Strike must be greater than zero.")

    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")

    if time_to_expiry <= 0:
        return max(spot - strike, 0.0)

    if volatility == 0:
        return max(
            spot - strike * exp(-risk_free_rate * time_to_expiry),
            0.0,
        )

    sqrt_t = sqrt(time_to_expiry)

    d1 = (
        log(spot / strike)
        + (
            risk_free_rate
            + 0.5 * volatility**2
        ) * time_to_expiry
    ) / (volatility * sqrt_t)

    d2 = d1 - volatility * sqrt_t

    return (
        spot * normal_cdf(d1)
        - strike
        * exp(-risk_free_rate * time_to_expiry)
        * normal_cdf(d2)
    )