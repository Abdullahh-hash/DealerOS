from app.models.market_summary import MarketSummary


class SummaryBuilder:
    """
    Builds the final DealerOS dealer market summary.
    """

    def __init__(
        self,
        snapshot,
        gamma,
        oi,
    ):
        self.snapshot = snapshot
        self.gamma = gamma
        self.oi = oi

    def build(self):

        call_gex = (
            self.gamma.largest_call_gex()
        )

        put_gex = (
            self.gamma.largest_put_gex()
        )

        positive_net_gex = (
            self.gamma.strongest_positive_net_gex()
        )

        negative_net_gex = (
            self.gamma.strongest_negative_net_gex()
        )

        call_oi = (
            self.oi.largest_call_oi()
        )

        put_oi = (
            self.oi.largest_put_oi()
        )

        total_oi = (
            self.oi.largest_total_oi_level()
        )

        return MarketSummary(
            symbol=self.snapshot.symbol,
            spot=self.snapshot.spot,

            gamma_state=(
                self.gamma.gamma_state()
            ),

            total_gex=(
                self.snapshot.total_gex
            ),

            net_gex_sign_change_strike=(
                self.gamma
                .nearest_net_gex_sign_change_strike()
            ),

            largest_call_gex_strike=(
                call_gex.strike
                if call_gex
                else None
            ),

            largest_call_gex_value=(
                call_gex.gex
                if call_gex
                else None
            ),

            largest_put_gex_strike=(
                put_gex.strike
                if put_gex
                else None
            ),

            largest_put_gex_value=(
                put_gex.gex
                if put_gex
                else None
            ),

            strongest_positive_net_gex_strike=(
                positive_net_gex.strike
                if positive_net_gex
                else None
            ),

            strongest_positive_net_gex_value=(
                positive_net_gex.net_gex
                if positive_net_gex
                else None
            ),

            strongest_negative_net_gex_strike=(
                negative_net_gex.strike
                if negative_net_gex
                else None
            ),

            strongest_negative_net_gex_value=(
                negative_net_gex.net_gex
                if negative_net_gex
                else None
            ),

            largest_call_oi_strike=(
                call_oi.strike
                if call_oi
                else None
            ),

            largest_call_oi_value=(
                call_oi.oi
                if call_oi
                else None
            ),

            largest_put_oi_strike=(
                put_oi.strike
                if put_oi
                else None
            ),

            largest_put_oi_value=(
                put_oi.oi
                if put_oi
                else None
            ),

            largest_total_oi_strike=(
                total_oi.strike
                if total_oi
                else None
            ),

            largest_total_oi_value=(
                total_oi.total_oi
                if total_oi
                else None
            ),
        )