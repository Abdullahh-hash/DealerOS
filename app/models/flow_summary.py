from dataclasses import dataclass, field
from typing import Optional

from app.models.flow_snapshot import FlowStrike


@dataclass
class FlowSummary:
    """
    Final factual DealerOS flow result.

    The strike list is the subset returned by FreeFlow's
    by_strike field. It must not be interpreted as a
    complete option-chain flow profile.
    """

    # ========================================================
    # MARKET CONTEXT
    # ========================================================

    symbol: str
    expiry: str
    timestamp: Optional[str]

    spot: float
    dte: int
    window_min: int

    # ========================================================
    # DIRECTIONAL FLOW
    # ========================================================

    directional_flow_state: str

    bull_premium: float
    bear_premium: float

    net_directional: float
    net_delta_notional: float

    # ========================================================
    # AGGRESSOR PREMIUM
    # ========================================================

    net_premium: float

    call_buy_premium: float
    call_sell_premium: float

    put_buy_premium: float
    put_sell_premium: float

    put_call_premium_ratio: float

    # ========================================================
    # CLASSIFIED TRADE COUNTS
    # ========================================================

    classified_count: int
    buy_count: int
    sell_count: int

    classified_buy_sell_ratio: float

    # ========================================================
    # RETURNED STRIKE ACTIVITY
    # ========================================================

    returned_strike_count: int

    largest_call_buy_strike: Optional[float]
    largest_call_buy_value: Optional[float]

    largest_call_sell_strike: Optional[float]
    largest_call_sell_value: Optional[float]

    largest_put_buy_strike: Optional[float]
    largest_put_buy_value: Optional[float]

    largest_put_sell_strike: Optional[float]
    largest_put_sell_value: Optional[float]

    largest_premium_buy_strike: Optional[float]
    largest_premium_buy_value: Optional[float]

    largest_premium_sell_strike: Optional[float]
    largest_premium_sell_value: Optional[float]

    strongest_bullish_directional_strike: Optional[float]
    strongest_bullish_directional_value: Optional[float]

    strongest_bearish_directional_strike: Optional[float]
    strongest_bearish_directional_value: Optional[float]

    # ========================================================
    # PROVIDER-RETURNED STRIKES
    # ========================================================

    returned_strikes: list[FlowStrike] = field(
        default_factory=list
    )