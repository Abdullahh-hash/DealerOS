from dataclasses import dataclass


@dataclass
class RNDPoint:
    """
    One point on the normalized risk-neutral density.
    """

    price: float
    density: float


@dataclass
class RNDResult:
    """
    Final validated DealerOS risk-neutral density result.

    This is the public RND output intended for consumers
    such as:

        Regime Engine
        Dashboard
        Replay
        Backtesting
        AI narratives

    RNDSurface remains an internal construction object.
    """

    # ========================================================
    # MARKET CONTEXT
    # ========================================================

    symbol: str
    exp: str
    timestamp: str

    spot: float
    forward: float

    dte: int

    time_to_expiry: float
    risk_free_rate: float

    atm_iv_pct: float

    # ========================================================
    # SURFACE CONFIGURATION
    # ========================================================

    sigma_multiplier: float
    fit_range: float

    source_count: int
    calibration_count: int
    surface_points: int

    grid_step: float
    smoothing_lambda: float

    # ========================================================
    # QUALITY / INTEGRITY
    # ========================================================

    raw_area: float
    normalized_area: float

    monotonicity_failures: int
    negative_density_points: int

    min_raw_density: float
    max_raw_density: float

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    mean: float
    mean_minus_forward: float

    median: float
    mode: float

    std: float

    # ========================================================
    # QUANTILES
    # ========================================================

    q05: float
    q25: float
    q75: float
    q95: float

    # ========================================================
    # RISK-NEUTRAL PROBABILITIES
    # ========================================================

    probability_above_spot: float
    probability_above_forward: float

    # ========================================================
    # NORMALIZED CURVE
    # ========================================================

    points: list[RNDPoint]

    # ========================================================
    # CONVENIENCE PROPERTIES
    # ========================================================

    @property
    def model_hours(self) -> float:
        """
        Black-Scholes model time expressed in hours.
        """

        return (
            self.time_to_expiry
            * 365.0
            * 24.0
        )

    @property
    def coverage_pct(self) -> float:
        """
        Raw probability coverage before normalization,
        expressed as a percentage.
        """

        return (
            self.raw_area
            * 100.0
        )

    @property
    def range_50(self) -> tuple[float, float]:
        """
        Central 50% risk-neutral interval.
        """

        return (
            self.q25,
            self.q75,
        )

    @property
    def range_90(self) -> tuple[float, float]:
        """
        Central 90% risk-neutral interval.
        """

        return (
            self.q05,
            self.q95,
        )