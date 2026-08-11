from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketSummary:
    """
    Final factual Dealer Engine market summary.

    Structural GEX and OI observations are exposed
    directly without labeling them support/resistance.
    """

    symbol: str
    spot: float

    gamma_state: str
    total_gex: Optional[float]

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    net_gex_sign_change_strike: Optional[float]

    largest_call_gex_strike: Optional[float]
    largest_call_gex_value: Optional[float]

    largest_put_gex_strike: Optional[float]
    largest_put_gex_value: Optional[float]

    strongest_positive_net_gex_strike: Optional[float]
    strongest_positive_net_gex_value: Optional[float]

    strongest_negative_net_gex_strike: Optional[float]
    strongest_negative_net_gex_value: Optional[float]

    # --------------------------------------------------------
    # OPEN INTEREST
    # --------------------------------------------------------

    largest_call_oi_strike: Optional[float]
    largest_call_oi_value: Optional[int]

    largest_put_oi_strike: Optional[float]
    largest_put_oi_value: Optional[int]

    largest_total_oi_strike: Optional[float]
    largest_total_oi_value: Optional[int]