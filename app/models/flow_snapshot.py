from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FlowStrike:
    strike: float
    call_net: float
    put_net: float
    net: float


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