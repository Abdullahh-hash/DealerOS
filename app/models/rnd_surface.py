from dataclasses import dataclass, field


@dataclass
class RNDSurfacePoint:
    """
    One point on the smoothed IV surface.

    IV is stored in decimal form.

    Example:
        14.84% -> 0.1484
    """

    strike: float
    iv: float
    right: str = "SMOOTH"


@dataclass
class RNDSurface:
    """
    Prepared IV surface used internally by the RND engine.
    """

    spot: float
    forward: float

    time_to_expiry: float
    risk_free_rate: float

    source_count: int
    calibration_count: int

    fit_range: float
    grid_step: float
    smoothing_lambda: float

    # --------------------------------------------------------
    # Snapshot metadata.
    #
    # Defaults preserve compatibility with older research
    # scripts that manually construct RNDSurface objects.
    # --------------------------------------------------------

    symbol: str = ""
    exp: str = ""
    timestamp: str = ""

    dte: int = 0
    atm_iv_pct: float = float("nan")

    sigma_multiplier: float = float("nan")

    points: list[RNDSurfacePoint] = field(
        default_factory=list
    )

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)