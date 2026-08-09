from app.models.dealer_zone import DealerZone
from app.models.gamma_level import GammaLevel


class ZoneAnalyzer:
    """
    Calculates dealer support and resistance
    from aggregated gamma levels.
    """

    def __init__(self, gamma_levels: list[GammaLevel]):
        self.gamma_levels = gamma_levels

    # ---------------------------------------------------------
    # Support
    # ---------------------------------------------------------

    def strongest_support(self):

        positives = [
            level
            for level in self.gamma_levels
            if level.net_gamma > 0
        ]

        if not positives:
            return None

        level = max(
            positives,
            key=lambda level: level.net_gamma,
        )

        return DealerZone(
            strike=level.strike,
            strength=level.net_gamma,
            zone_type="Support",
        )

    # ---------------------------------------------------------
    # Resistance
    # ---------------------------------------------------------

    def strongest_resistance(self):

        negatives = [
            level
            for level in self.gamma_levels
            if level.net_gamma < 0
        ]

        if not negatives:
            return None

        level = min(
            negatives,
            key=lambda level: level.net_gamma,
        )

        return DealerZone(
            strike=level.strike,
            strength=abs(level.net_gamma),
            zone_type="Resistance",
        )

    # ---------------------------------------------------------
    # Aliases
    # ---------------------------------------------------------

    def dealer_support(self):
        return self.strongest_support()

    def dealer_resistance(self):
        return self.strongest_resistance()