from dataclasses import dataclass, field
from typing import List, Optional

from app.models.option_contract import OptionContract


@dataclass
class DealerSnapshot:
    """
    Represents the complete market snapshot
    returned by the FreeFlow API.
    """

    symbol: str
    timestamp: str

    spot: float

    total_gex: Optional[float] = None
    total_dex: Optional[float] = None

    total_ag: Optional[float] = None
    total_dag: Optional[float] = None

    total_vol: Optional[float] = None

    vol_trigger: Optional[float] = None

    contracts: List[OptionContract] = field(default_factory=list)