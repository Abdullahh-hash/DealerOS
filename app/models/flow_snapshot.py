from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FlowStrike:
    """
    Premium-flow activity for one provider-returned strike.

    net:
        Call net + Put net.
        Represents the aggressor premium balance.

    directional_net:
        Call net - Put net.
        Positive = bullish-classified premium dominates.
        Negative = bearish-classified premium dominates.
    """

    strike: float
    call_net: float
    put_net: float
    net: float

    @property
    def aggressor_net(self) -> float:
        return self.net

    @property
    def directional_net(self) -> float:
        return (
            self.call_net
            - self.put_net
        )


@dataclass
class FlowCounts:
    classified: int
    buy: int
    sell: int


@dataclass
class PremiumSide:
    buy: float
    sell: float


@dataclass
class FlowSnapshot:
    symbol: str
    expiry: str
    spot: float
    dte: int
    window_min: int

    net_premium: float
    bull_premium: float
    bear_premium: float
    net_directional: float
    net_delta_notional: float

    call_premium: PremiumSide
    put_premium: PremiumSide

    put_call_prem_ratio: float

    counts: FlowCounts

    by_strike: List[FlowStrike] = field(default_factory=list)

    timestamp: Optional[str] = None