import numpy as np

from app.models.rnd_result import (
    RNDPoint,
    RNDResult,
)

from app.models.rnd_surface import RNDSurface

from app.services.black_scholes import (
    black_scholes_call,
)

# ============================================================
# VALIDATION TOLERANCES
# ============================================================

MONOTONICITY_TOL = 1e-10
NEGATIVE_DENSITY_TOL = -1e-12

# Minimum raw probability mass that must be captured
# by the fitted strike window before normalization.
#
# Empirically validated across the current live
# 0DTE snapshot set.
MIN_RAW_AREA = 0.99


# ============================================================
# ENGINE
# ============================================================

class RNDEngine:
    """
    Build a risk-neutral density from a validated,
    smoothed IV surface.

    Pipeline:

        smooth IV
        -> Black-Scholes calls
        -> monotonicity validation
        -> Breeden-Litzenberger density
        -> non-negativity validation
        -> normalization
        -> distribution statistics

    Negative density is never clipped.
    """

    def __init__(
        self,
        spot: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.0,
    ):
        if spot <= 0:
            raise ValueError(
                "Spot must be greater than zero."
            )

        if time_to_expiry <= 0:
            raise ValueError(
                "Time to expiry must be greater than zero."
            )

        self.spot = float(spot)
        self.time_to_expiry = float(
            time_to_expiry
        )
        self.risk_free_rate = float(
            risk_free_rate
        )

    # ========================================================
    # BLACK-SCHOLES CALL RECONSTRUCTION
    # ========================================================

    def build_call_prices(
        self,
        prices: np.ndarray,
        ivs: np.ndarray,
    ) -> np.ndarray:
        """
        Convert the smoothed IV grid into
        Black-Scholes call prices.

        IVs must be decimal values.

        Example:
            14.84% -> 0.1484
        """

        prices = np.asarray(
            prices,
            dtype=float,
        )

        ivs = np.asarray(
            ivs,
            dtype=float,
        )

        if len(prices) != len(ivs):
            raise ValueError(
                "Prices and IVs must have the same length."
            )

        if len(prices) < 3:
            raise ValueError(
                "At least three surface points are required."
            )

        if not np.all(
            np.isfinite(prices)
        ):
            raise ValueError(
                "Strike grid contains non-finite values."
            )

        if not np.all(
            np.isfinite(ivs)
        ):
            raise ValueError(
                "IV surface contains non-finite values."
            )

        if np.any(
            prices <= 0
        ):
            raise ValueError(
                "All strikes must be greater than zero."
            )

        if np.any(
            ivs <= 0
        ):
            raise ValueError(
                "All IV values must be greater than zero."
            )

        if np.any(
            np.diff(prices) <= 0
        ):
            raise ValueError(
                "Strike grid must be strictly increasing."
            )

        call_prices = np.array(
            [
                black_scholes_call(
                    spot=self.spot,
                    strike=float(strike),
                    time_to_expiry=self.time_to_expiry,
                    volatility=float(iv),
                    risk_free_rate=self.risk_free_rate,
                )
                for strike, iv
                in zip(
                    prices,
                    ivs,
                )
            ],
            dtype=float,
        )

        return call_prices

    # ========================================================
    # CALL-SURFACE VALIDATION
    # ========================================================

    def count_monotonicity_failures(
        self,
        call_prices: np.ndarray,
    ) -> int:
        """
        European call prices must not increase
        as strike increases.
        """

        call_prices = np.asarray(
            call_prices,
            dtype=float,
        )

        differences = np.diff(
            call_prices
        )

        return int(
            np.sum(
                differences
                > MONOTONICITY_TOL
            )
        )

    # ========================================================
    # BREEDEN-LITZENBERGER DENSITY
    # ========================================================

    def build_density(
        self,
        prices: np.ndarray,
        call_prices: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate raw risk-neutral density:

            q(K) = exp(rT) * d²C/dK²

        No clipping is performed.
        """

        prices = np.asarray(
            prices,
            dtype=float,
        )

        call_prices = np.asarray(
            call_prices,
            dtype=float,
        )

        if len(prices) != len(call_prices):
            raise ValueError(
                "Prices and call prices must have the same length."
            )

        if len(prices) < 3:
            raise ValueError(
                "At least three price points are required."
            )

        first_derivative = np.gradient(
            call_prices,
            prices,
        )

        second_derivative = np.gradient(
            first_derivative,
            prices,
        )

        discount_adjustment = np.exp(
            self.risk_free_rate
            * self.time_to_expiry
        )

        density = (
            discount_adjustment
            * second_derivative
        )

        return density

    # ========================================================
    # RAW DENSITY VALIDATION
    # ========================================================

    def count_negative_density_points(
        self,
        density: np.ndarray,
    ) -> int:
        """
        Count materially negative density values.

        Values above NEGATIVE_DENSITY_TOL are treated
        as floating-point noise, not arbitrage failures.

        Nothing is clipped.
        """

        density = np.asarray(
            density,
            dtype=float,
        )

        return int(
            np.sum(
                density
                < NEGATIVE_DENSITY_TOL
            )
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize_density(
        self,
        prices: np.ndarray,
        density: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Normalize a clean raw density.

        Returns:
            normalized_density
            raw_area
        """

        area = float(
            np.trapezoid(
                density,
                prices,
            )
        )

        if not np.isfinite(area):
            raise ValueError(
                "Density area is non-finite."
            )

        if area <= 0:
            raise ValueError(
                "Density area must be greater than zero."
            )

        normalized = (
            density / area
        )

        return (
            normalized,
            area,
        )

    # ========================================================
    # CDF
    # ========================================================

    def build_cdf(
        self,
        prices: np.ndarray,
        density: np.ndarray,
    ) -> np.ndarray:
        """
        Build a cumulative distribution using
        trapezoidal integration.
        """

        prices = np.asarray(
            prices,
            dtype=float,
        )

        density = np.asarray(
            density,
            dtype=float,
        )

        if len(prices) != len(density):
            raise ValueError(
                "Prices and density must have the same length."
            )

        cdf = np.zeros_like(
            density,
            dtype=float,
        )

        increments = (
            0.5
            * (
                density[:-1]
                + density[1:]
            )
            * np.diff(prices)
        )

        cdf[1:] = np.cumsum(
            increments
        )

        total = float(
            cdf[-1]
        )

        if total <= 0:
            raise ValueError(
                "CDF total probability must be greater than zero."
            )

        # Defensive normalization against tiny
        # numerical integration error.
        cdf = (
            cdf / total
        )

        return cdf

    # ========================================================
    # QUANTILES
    # ========================================================

    def quantile(
        self,
        prices: np.ndarray,
        cdf: np.ndarray,
        probability: float,
    ) -> float:
        """
        Return the price corresponding to a
        cumulative probability.
        """

        if not (
            0.0 <= probability <= 1.0
        ):
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        return float(
            np.interp(
                probability,
                cdf,
                prices,
            )
        )

    # ========================================================
    # PROBABILITY ABOVE LEVEL
    # ========================================================

    def probability_above(
        self,
        prices: np.ndarray,
        cdf: np.ndarray,
        level: float,
    ) -> float:
        """
        Calculate risk-neutral probability that
        terminal price is above a specified level.
        """

        minimum = float(
            prices[0]
        )

        maximum = float(
            prices[-1]
        )

        if level <= minimum:
            return 1.0

        if level >= maximum:
            return 0.0

        cumulative_probability = float(
            np.interp(
                level,
                prices,
                cdf,
            )
        )

        probability = (
            1.0
            - cumulative_probability
        )

        return float(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # BUILD FROM PRODUCTION SURFACE
    # ========================================================

    def build_from_surface(
        self,
        surface: RNDSurface,
    ) -> RNDResult:
        """
        Build, validate, normalize, and summarize
        the RND from a production RNDSurface.
        """

        if not surface.points:
            raise ValueError(
                "RND surface contains no points."
            )

        # ----------------------------------------------------
        # Prevent model-input mismatch.
        # ----------------------------------------------------

        if not np.isclose(
            self.spot,
            surface.spot,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "Engine spot does not match RND surface spot."
            )

        if not np.isclose(
            self.time_to_expiry,
            surface.time_to_expiry,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(
                "Engine time-to-expiry does not match "
                "RND surface."
            )

        if not np.isclose(
            self.risk_free_rate,
            surface.risk_free_rate,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Engine risk-free rate does not match "
                "RND surface."
            )

        prices = np.asarray(
            [
                point.strike
                for point in surface.points
            ],
            dtype=float,
        )

        ivs = np.asarray(
            [
                point.iv
                for point in surface.points
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # Smooth IV -> Black-Scholes calls
        # ----------------------------------------------------

        call_prices = self.build_call_prices(
            prices=prices,
            ivs=ivs,
        )

        # ----------------------------------------------------
        # Integrity Gate 1:
        # Call prices must decrease with strike.
        # ----------------------------------------------------

        monotonicity_failures = (
            self.count_monotonicity_failures(
                call_prices
            )
        )

        if monotonicity_failures != 0:
            raise ValueError(
                "RND call surface failed monotonicity "
                f"validation: {monotonicity_failures} failures."
            )

        # ----------------------------------------------------
        # Breeden-Litzenberger
        # ----------------------------------------------------

        raw_density = self.build_density(
            prices=prices,
            call_prices=call_prices,
        )

        # ----------------------------------------------------
        # Integrity Gate 2:
        # Raw density must be non-negative.
        # ----------------------------------------------------

        negative_density_points = (
            self.count_negative_density_points(
                raw_density
            )
        )

        if negative_density_points != 0:
            raise ValueError(
                "RND density failed non-negativity "
                f"validation: {negative_density_points} "
                "negative points."
            )

        # ----------------------------------------------------
        # Integrity Gate 3:
        # The fitted strike window must capture enough
        # raw probability mass BEFORE normalization.
        #
        # We do not allow normalization to hide a
        # materially truncated distribution.
        # ----------------------------------------------------

        raw_area_check = float(
            np.trapezoid(
                raw_density,
                prices,
            )
        )

        if not np.isfinite(
            raw_area_check
        ):
            raise ValueError(
                "RND raw probability area is non-finite."
            )

        if raw_area_check <= 0:
            raise ValueError(
                "RND raw probability area must be "
                "greater than zero."
            )

        if raw_area_check < MIN_RAW_AREA:
            missing_mass = (
                1.0 - raw_area_check
            )

            raise ValueError(
                "RND probability coverage failed: "
                f"raw area={raw_area_check:.6f}, "
                f"minimum={MIN_RAW_AREA:.6f}, "
                f"missing mass={missing_mass:.6f}."
            )

        # ----------------------------------------------------
        # Normalize only AFTER all three integrity gates pass:
        #
        # 1. Call monotonicity
        # 2. Density non-negativity
        # 3. Raw probability coverage
        # ----------------------------------------------------

        (
            normalized_density,
            raw_area,
        ) = self.normalize_density(
            prices=prices,
            density=raw_density,
        )

        normalized_area = float(
            np.trapezoid(
                normalized_density,
                prices,
            )
        )

        # ----------------------------------------------------
        # Distribution mean
        # ----------------------------------------------------

        mean = float(
            np.trapezoid(
                prices
                * normalized_density,
                prices,
            )
        )

        mean_minus_forward = (
            mean
            - float(
                surface.forward
            )
        )

        # ----------------------------------------------------
        # Standard deviation
        # ----------------------------------------------------

        variance = float(
            np.trapezoid(
                (
                    prices - mean
                ) ** 2
                * normalized_density,
                prices,
            )
        )

        std = float(
            np.sqrt(
                max(
                    variance,
                    0.0,
                )
            )
        )

        # ----------------------------------------------------
        # CDF
        # ----------------------------------------------------

        cdf = self.build_cdf(
            prices=prices,
            density=normalized_density,
        )

        # ----------------------------------------------------
        # Quantiles
        # ----------------------------------------------------

        q05 = self.quantile(
            prices,
            cdf,
            0.05,
        )

        q25 = self.quantile(
            prices,
            cdf,
            0.25,
        )

        median = self.quantile(
            prices,
            cdf,
            0.50,
        )

        q75 = self.quantile(
            prices,
            cdf,
            0.75,
        )

        q95 = self.quantile(
            prices,
            cdf,
            0.95,
        )

        # ----------------------------------------------------
        # Mode
        # ----------------------------------------------------

        mode = float(
            prices[
                np.argmax(
                    normalized_density
                )
            ]
        )

        # ----------------------------------------------------
        # Risk-neutral directional probabilities
        # ----------------------------------------------------

        probability_above_spot = (
            self.probability_above(
                prices=prices,
                cdf=cdf,
                level=self.spot,
            )
        )

        probability_above_forward = (
            self.probability_above(
                prices=prices,
                cdf=cdf,
                level=float(
                    surface.forward
                ),
            )
        )

        points = [
            RNDPoint(
                price=float(price),
                density=float(density),
            )
            for price, density
            in zip(
                prices,
                normalized_density,
            )
        ]

        return RNDResult(
    # --------------------------------------------------------
    # Market context
    # --------------------------------------------------------

    symbol=surface.symbol,
    exp=surface.exp,
    timestamp=surface.timestamp,

    spot=surface.spot,
    forward=surface.forward,

    dte=surface.dte,

    time_to_expiry=(
        surface.time_to_expiry
    ),

    risk_free_rate=(
        surface.risk_free_rate
    ),

    atm_iv_pct=(
        surface.atm_iv_pct
    ),

    # --------------------------------------------------------
    # Surface configuration
    # --------------------------------------------------------

    sigma_multiplier=(
        surface.sigma_multiplier
    ),

    fit_range=(
        surface.fit_range
    ),

    source_count=(
        surface.source_count
    ),

    calibration_count=(
        surface.calibration_count
    ),

    surface_points=len(
        surface.points
    ),

    grid_step=(
        surface.grid_step
    ),

    smoothing_lambda=(
        surface.smoothing_lambda
    ),

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    raw_area=raw_area,

    normalized_area=(
        normalized_area
    ),

    monotonicity_failures=(
        monotonicity_failures
    ),

    negative_density_points=(
        negative_density_points
    ),

    min_raw_density=float(
        np.min(
            raw_density
        )
    ),

    max_raw_density=float(
        np.max(
            raw_density
        )
    ),

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    mean=mean,

    mean_minus_forward=float(
        mean_minus_forward
    ),

    median=median,
    mode=mode,
    std=std,

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    q05=q05,
    q25=q25,
    q75=q75,
    q95=q95,

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probability_above_spot=(
        probability_above_spot
    ),

    probability_above_forward=(
        probability_above_forward
    ),

    # --------------------------------------------------------
    # Curve
    # --------------------------------------------------------

    points=points,
)

    # ========================================================
    # LEGACY LOW-LEVEL CURVE BUILDER
    # ========================================================

    def build_curve(
        self,
        prices: np.ndarray,
        call_prices: np.ndarray,
    ) -> list[RNDPoint]:
        """
        Build a normalized curve from externally supplied
        call prices.

        Invalid surfaces are rejected rather than clipped.
        """

        monotonicity_failures = (
            self.count_monotonicity_failures(
                call_prices
            )
        )

        if monotonicity_failures != 0:
            raise ValueError(
                "Call surface failed monotonicity "
                f"validation: {monotonicity_failures} failures."
            )

        density = self.build_density(
            prices=prices,
            call_prices=call_prices,
        )

        negative_density_points = (
            self.count_negative_density_points(
                density
            )
        )

        if negative_density_points != 0:
            raise ValueError(
                "Density failed non-negativity "
                f"validation: {negative_density_points} "
                "negative points."
            )

        (
            normalized_density,
            _,
        ) = self.normalize_density(
            prices=prices,
            density=density,
        )

        return [
            RNDPoint(
                price=float(price),
                density=float(value),
            )
            for price, value
            in zip(
                prices,
                normalized_density,
            )
        ]