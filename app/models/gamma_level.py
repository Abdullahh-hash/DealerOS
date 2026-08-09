from dataclasses import dataclass


@dataclass
class GammaLevel:
    """
    Aggregated dealer gamma at a strike.
    """

    strike: float

    call_gamma: float = 0.0
    put_gamma: float = 0.0

    net_gamma: float = 0.0