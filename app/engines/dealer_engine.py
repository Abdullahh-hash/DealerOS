from app.engines.gamma_analyzer import GammaAnalyzer
from app.engines.zone_analyzer import ZoneAnalyzer
from app.engines.summary_builder import SummaryBuilder


class DealerEngine:
    """
    Main Dealer Engine.

    Coordinates all dealer analysis components.
    """

    def __init__(self, snapshot):
        self.snapshot = snapshot

        self.gamma = GammaAnalyzer(snapshot)

        self.zones = ZoneAnalyzer(
            self.gamma.net_gamma_levels()
        )

    def summary(self):
        """
        Return complete market summary.
        """

        return SummaryBuilder(
            self.snapshot,
            self.gamma,
            self.zones,
        ).build()