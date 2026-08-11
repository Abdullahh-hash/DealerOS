from dataclasses import dataclass


@dataclass
class GEXLevel:
    """
    Aggregated option gamma exposure at one strike.

    FreeFlow GEX is OI-based and follows the provider's
    sign convention:

        Calls -> positive GEX
        Puts  -> negative GEX
    """

    strike: float

    call_gex: float = 0.0
    put_gex: float = 0.0

    net_gex: float = 0.0