from dataclasses import dataclass
from typing import Optional


@dataclass
class OptionContract:
    """
    Represents a single option contract (one strike)
    returned by the FreeFlow API.
    """

    strike: float
    right: str

    oi: Optional[int] = None

    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    vanna: Optional[float] = None
    charm: Optional[float] = None

    gex: Optional[float] = None
    dex: Optional[float] = None
    vex: Optional[float] = None

    ag: Optional[float] = None
    dag: Optional[float] = None

    vegaex: Optional[float] = None
    charmex: Optional[float] = None

    iv_pct: Optional[float] = None