from app.engines.gamma_analyzer import GammaAnalyzer
from app.engines.oi_analyzer import OIAnalyzer
from app.engines.summary_builder import SummaryBuilder


class DealerEngine:
    """
    Main Dealer Engine.

    Coordinates dealer-positioning analysis.
    """

    def __init__(self, snapshot):
        self.snapshot = snapshot

        self.gamma = GammaAnalyzer(
            snapshot
        )

        self.oi = OIAnalyzer(
            snapshot
        )

    def summary(self):
        """
        Return the complete DealerOS dealer summary.
        """

        return SummaryBuilder(
            self.snapshot,
            self.gamma,
            self.oi,
        ).build()