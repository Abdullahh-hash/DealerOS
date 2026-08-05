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
    exp: str
    timestamp: str

    spot: float
    dte: int

    total_gex: Optional[float] = None
    total_dex: Optional[float] = None

    total_ag: Optional[float] = None
    total_dag: Optional[float] = None

    net_premium: Optional[float] = None

    gross_dex: Optional[float] = None
    gross_vex: Optional[float] = None
    gross_charmex: Optional[float] = None

    total_vol: Optional[float] = None

    max_pain: Optional[float] = None
    vol_trigger: Optional[float] = None

    atm_iv: Optional[float] = None
    model: Optional[str] = None

    contracts: List[OptionContract] = field(default_factory=list)