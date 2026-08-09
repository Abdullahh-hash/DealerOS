from dataclasses import dataclass
from typing import Optional

from app.models.dealer_zone import DealerZone


@dataclass
class MarketSummary:

    symbol: str

    spot: float

    dealer_bias: str

    gamma_flip: Optional[float]

    largest_call_gex: Optional[float]

    largest_put_gex: Optional[float]

    support: Optional[DealerZone]

    resistance: Optional[DealerZone]