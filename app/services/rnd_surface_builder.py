from math import exp, log, sqrt
from statistics import NormalDist, median

import numpy as np
from scipy.interpolate import make_smoothing_spline

from app.models.dealer_snapshot import DealerSnapshot
from app.models.rnd_surface import (
    RNDSurface,
    RNDSurfacePoint,
)


# ============================================================
# VALIDATED RND CONFIGURATION
# ============================================================

RND_SIGMA_MULTIPLIER = 3.10
RND_GRID_STEP = 5.0
RND_SPLINE_LAMBDA = 2e-3

RND_X_SCALE = 1000.0

ATM_CALIBRATION_RANGE = 100.0


normal = NormalDist()


# ============================================================
# FREEFLOW MODEL CALIBRATION
# ============================================================

def calibrate_model_inputs(
    snapshot: DealerSnapshot,
) -> tuple[float, float, int]:
    """
    Infer the Black-Scholes time-to-expiry and
    risk-free rate encoded in FreeFlow's Greeks.

    Time is inferred from the gamma/vega identity.

    Rate is inferred from call delta assuming q = 0.

    Returns:
        time_to_expiry
        risk_free_rate
        calibration_count
    """

    spot = float(snapshot.spot)

    if spot <= 0:
        raise ValueError(
            "Snapshot spot must be greater than zero."
        )

    inferred_times = []
    inferred_rates = []

    for contract in snapshot.contracts:

        if contract.right.upper() != "C":
            continue

        if contract.iv_pct is None:
            continue

        if contract.delta is None:
            continue

        if contract.gamma is None:
            continue

        if contract.vega is None:
            continue

        strike = float(contract.strike)
        sigma = float(contract.iv_pct) / 100.0

        delta = float(contract.delta)
        gamma = float(contract.gamma)
        vega = float(contract.vega)

        if abs(strike - spot) > ATM_CALIBRATION_RANGE:
            continue

        if sigma <= 0:
            continue

        if gamma <= 0:
            continue

        if vega <= 0:
            continue

        if not (0.0 < delta < 1.0):
            continue

        # ----------------------------------------------------
        # Infer T from:
        #
        # vega / gamma = S^2 * sigma * T
        #
        # FreeFlow vega is quoted per one volatility
        # percentage point, hence the factor of 100.
        # ----------------------------------------------------

        time_to_expiry = (
            100.0
            * vega
            / (
                gamma
                * spot
                * spot
                * sigma
            )
        )

        if time_to_expiry <= 0:
            continue

        # ----------------------------------------------------
        # FreeFlow call delta:
        #
        # delta = N(d1)
        #
        # Assuming q = 0:
        #
        # d1 =
        # [ln(S/K) + (r + sigma^2 / 2)T]
        # / (sigma * sqrt(T))
        #
        # Rearrange to solve for r.
        # ----------------------------------------------------

        d1 = normal.inv_cdf(delta)

        risk_free_rate = (
            (
                d1
                * sigma
                * sqrt(time_to_expiry)
                - log(spot / strike)
            )
            / time_to_expiry
            - 0.5 * sigma * sigma
        )

        inferred_times.append(
            time_to_expiry
        )

        inferred_rates.append(
            risk_free_rate
        )

    if not inferred_times:
        raise ValueError(
            "Unable to infer FreeFlow model time."
        )

    if not inferred_rates:
        raise ValueError(
            "Unable to infer FreeFlow risk-free rate."
        )

    calibrated_time = median(
        inferred_times
    )

    calibrated_rate = median(
        inferred_rates
    )

    return (
        calibrated_time,
        calibrated_rate,
        len(inferred_times),
    )


# ============================================================
# OTM SOURCE SELECTION
# ============================================================

def build_otm_observations(
    snapshot: DealerSnapshot,
    forward: float,
    fit_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the OTM IV observations used by the RND fit.

    Selection rule:

        puts below forward
        calls above forward

    The strike window is volatility-scaled upstream.
    """

    if fit_range <= 0:
        raise ValueError(
            "RND fit range must be greater than zero."
        )

    strikes = []
    ivs_pct = []

    for contract in snapshot.contracts:

        if contract.iv_pct is None:
            continue

        iv_pct = float(
            contract.iv_pct
        )

        if iv_pct <= 0:
            continue

        strike = float(
            contract.strike
        )

        right = contract.right.upper()

        if (
            abs(strike - forward)
            > fit_range
        ):
            continue

        use_contract = (
            (
                strike < forward
                and right == "P"
            )
            or
            (
                strike > forward
                and right == "C"
            )
        )

        if not use_contract:
            continue

        strikes.append(
            strike
        )

        ivs_pct.append(
            iv_pct
        )

    if len(strikes) < 10:
        raise ValueError(
            "Not enough OTM IV observations "
            "to build the RND surface."
        )

    strikes = np.asarray(
        strikes,
        dtype=float,
    )

    ivs_pct = np.asarray(
        ivs_pct,
        dtype=float,
    )

    order = np.argsort(
        strikes
    )

    strikes = strikes[
        order
    ]

    ivs_pct = ivs_pct[
        order
    ]

    unique_strikes, unique_indices = np.unique(
        strikes,
        return_index=True,
    )

    strikes = unique_strikes

    ivs_pct = ivs_pct[
        unique_indices
    ]

    if len(strikes) < 10:
        raise ValueError(
            "Not enough unique OTM strikes "
            "to build the RND surface."
        )

    return (
        strikes,
        ivs_pct,
    )
    # --------------------------------------------------------
    # Sort by strike.
    # --------------------------------------------------------

    order = np.argsort(
        strikes
    )

    strikes = strikes[
        order
    ]

    ivs_pct = ivs_pct[
        order
    ]

    # --------------------------------------------------------
    # Defensive duplicate-strike removal.
    # --------------------------------------------------------

    unique_strikes, unique_indices = np.unique(
        strikes,
        return_index=True,
    )

    strikes = unique_strikes

    ivs_pct = ivs_pct[
        unique_indices
    ]

    if len(strikes) < 10:
        raise ValueError(
            "Not enough unique OTM strikes "
            "to build the RND surface."
        )

    return (
        strikes,
        ivs_pct,
    )


# ============================================================
# SMOOTH IV SURFACE
# ============================================================

def smooth_iv_surface(
    strikes: np.ndarray,
    ivs_pct: np.ndarray,
    forward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the validated smoothing spline to the OTM IV
    observations and evaluate it on a 5-point strike grid.
    """

    x = (
        strikes - forward
    ) / RND_X_SCALE

    spline = make_smoothing_spline(
        x,
        ivs_pct,
        lam=RND_SPLINE_LAMBDA,
    )

    strike_grid = np.arange(
        strikes[0],
        strikes[-1]
        + RND_GRID_STEP * 0.5,
        RND_GRID_STEP,
        dtype=float,
    )

    grid_x = (
        strike_grid - forward
    ) / RND_X_SCALE

    fitted_iv_pct = spline(
        grid_x
    )

    if not np.all(
        np.isfinite(
            fitted_iv_pct
        )
    ):
        raise ValueError(
            "RND spline produced non-finite IV values."
        )

    if np.any(
        fitted_iv_pct <= 0
    ):
        raise ValueError(
            "RND spline produced non-positive IV values."
        )

    return (
        strike_grid,
        fitted_iv_pct,
    )


# ============================================================
# PUBLIC BUILDER
# ============================================================

def build_rnd_surface(
    snapshot: DealerSnapshot,
) -> RNDSurface:
    """
    Build the production-ready smoothed IV surface
    used by the RND engine.

    The fit range is volatility-scaled:

        range =
            sigma_multiplier
            * spot
            * ATM_IV
            * sqrt(T)
    """

    (
        time_to_expiry,
        risk_free_rate,
        calibration_count,
    ) = calibrate_model_inputs(
        snapshot
    )

    spot = float(
        snapshot.spot
    )

    if snapshot.atm_iv is None:
        raise ValueError(
            "ATM IV is required for adaptive RND range."
        )

    atm_iv = (
        float(snapshot.atm_iv)
        / 100.0
    )

    if atm_iv <= 0:
        raise ValueError(
            "ATM IV must be greater than zero."
        )

    forward = (
        spot
        * exp(
            risk_free_rate
            * time_to_expiry
        )
    )

    # --------------------------------------------------------
    # Volatility-scaled RND range.
    #
    # One model sigma in index points:
    #
    #     S * sigma * sqrt(T)
    #
    # Validated multiplier:
    #
    #     3.10 sigma
    # --------------------------------------------------------

    sigma_move = (
        spot
        * atm_iv
        * sqrt(
            time_to_expiry
        )
    )

    fit_range = (
        RND_SIGMA_MULTIPLIER
        * sigma_move
    )

    (
        source_strikes,
        source_ivs_pct,
    ) = build_otm_observations(
        snapshot=snapshot,
        forward=forward,
        fit_range=fit_range,
    )

    (
        strike_grid,
        fitted_iv_pct,
    ) = smooth_iv_surface(
        strikes=source_strikes,
        ivs_pct=source_ivs_pct,
        forward=forward,
    )

    points = [
        RNDSurfacePoint(
            strike=float(strike),
            iv=float(iv_pct) / 100.0,
            right="SMOOTH",
        )
        for strike, iv_pct
        in zip(
            strike_grid,
            fitted_iv_pct,
        )
    ]

    return RNDSurface(
    spot=spot,

    forward=float(
        forward
    ),

    time_to_expiry=float(
        time_to_expiry
    ),

    risk_free_rate=float(
        risk_free_rate
    ),

    source_count=len(
        source_strikes
    ),

    calibration_count=(
        calibration_count
    ),

    fit_range=float(
        fit_range
    ),

    grid_step=RND_GRID_STEP,

    smoothing_lambda=(
        RND_SPLINE_LAMBDA
    ),

    symbol=str(
        snapshot.symbol
    ),

    exp=str(
        snapshot.exp
    ),

    timestamp=str(
        snapshot.timestamp
    ),

    dte=int(
        snapshot.dte
    ),

    atm_iv_pct=float(
        snapshot.atm_iv
    ),

    sigma_multiplier=float(
        RND_SIGMA_MULTIPLIER
    ),

    points=points,
)