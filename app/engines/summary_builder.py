from app.models.market_summary import MarketSummary


class SummaryBuilder:
    """
    Builds the final DealerOS market summary.
    """

    def __init__(
        self,
        snapshot,
        gamma,
        zones,
    ):
        self.snapshot = snapshot
        self.gamma = gamma
        self.zones = zones

    def build(self):

        call = self.gamma.largest_call_gex()
        put = self.gamma.largest_put_gex()

        support = self.zones.dealer_support()
        resistance = self.zones.dealer_resistance()

        return MarketSummary(
            symbol=self.snapshot.symbol,
            spot=self.snapshot.spot,
            dealer_bias=self.gamma.dealer_bias(),
            gamma_flip=self.gamma.gamma_flip(),
            largest_call_gex=call.strike if call else None,
            largest_put_gex=put.strike if put else None,
            support=support,
            resistance=resistance,
        )