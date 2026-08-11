from dataclasses import dataclass


@dataclass
class OILevel:
    """
    Aggregated open interest at one strike.
    """

    strike: float

    call_oi: int = 0
    put_oi: int = 0

    total_oi: int = 0