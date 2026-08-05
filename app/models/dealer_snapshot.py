from dataclasses import dataclass
from typing import Optional


@dataclass
class DealerSnapshot:
    """
    Represents a complete market snapshot
    returned by the FreeFlow API.
    """

    symbol: str

    total_gex: Optional[float] = None
    total_dex: Optional[float] = None

    call_wall: Optional[float] = None
    put_wall: Optional[float] = None

    gamma_flip: Optional[float] = None
    max_pain: Optional[float] = None
    vol_trigger: Optional[float] = None