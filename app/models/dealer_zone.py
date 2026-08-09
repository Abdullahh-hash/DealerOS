from dataclasses import dataclass


@dataclass
class DealerZone:
    """
    Dealer support or resistance zone.
    """

    strike: float

    strength: float

    zone_type: str