from app.models.flow_snapshot import FlowSnapshot, FlowStrike


class FlowEngine:
    """
    Analyzes options order-flow data from FreeFlow.
    """

    def __init__(self, snapshot: FlowSnapshot):
        self.snapshot = snapshot

    def flow_bias(self) -> str:
        """
        Determine directional flow bias.
        """

        if self.snapshot.net_directional > 0:
            return "Bullish"

        if self.snapshot.net_directional < 0:
            return "Bearish"

        return "Neutral"

    def net_premium(self) -> float:
        return self.snapshot.net_premium

    def call_buy_premium(self) -> float:
        return self.snapshot.call_premium.buy

    def call_sell_premium(self) -> float:
        return self.snapshot.call_premium.sell

    def put_buy_premium(self) -> float:
        return self.snapshot.put_premium.buy

    def put_sell_premium(self) -> float:
        return self.snapshot.put_premium.sell

    def largest_positive_flow(self) -> FlowStrike | None:
        if not self.snapshot.by_strike:
            return None

        return max(
            self.snapshot.by_strike,
            key=lambda x: x.net,
        )

    def largest_negative_flow(self) -> FlowStrike | None:
        if not self.snapshot.by_strike:
            return None

        return min(
            self.snapshot.by_strike,
            key=lambda x: x.net,
        )

    def call_flow_strike(self) -> FlowStrike | None:
        if not self.snapshot.by_strike:
            return None

        return max(
            self.snapshot.by_strike,
            key=lambda x: x.call_net,
        )

    def put_flow_strike(self) -> FlowStrike | None:
        if not self.snapshot.by_strike:
            return None

        return max(
            self.snapshot.by_strike,
            key=lambda x: x.put_net,
        )

    def buy_sell_ratio(self) -> float:
        sells = self.snapshot.counts.sell

        if sells == 0:
            return float("inf")

        return self.snapshot.counts.buy / sells

    def nearby_strikes(self, distance: float = 500):
        lower = self.snapshot.spot - distance
        upper = self.snapshot.spot + distance

        return [
            item
            for item in self.snapshot.by_strike
            if lower <= item.strike <= upper
        ]

    def largest_positive_flow_nearby(self, distance: float = 500):
        strikes = self.nearby_strikes(distance)

        positive_strikes = [
            item
            for item in strikes
            if item.net > 0
        ]

        if not positive_strikes:
            return None

        return max(
            positive_strikes,
            key=lambda x: x.net,
        )

    def largest_negative_flow_nearby(self, distance: float = 500):
        strikes = self.nearby_strikes(distance)

        negative_strikes = [
            item
            for item in strikes
            if item.net < 0
        ]

        if not negative_strikes:
            return None

        return min(
            negative_strikes,
            key=lambda x: x.net,
        )

    def summary(self) -> dict:
        positive = self.largest_positive_flow_nearby()
        negative = self.largest_negative_flow_nearby()
        return {
            "symbol": self.snapshot.symbol,
            "spot": self.snapshot.spot,
            "flow_bias": self.flow_bias(),
            "net_premium": self.snapshot.net_premium,
            "net_directional": self.snapshot.net_directional,
            "net_delta_notional": self.snapshot.net_delta_notional,
            "put_call_premium_ratio": self.snapshot.put_call_prem_ratio,
            "buy_count": self.snapshot.counts.buy,
            "sell_count": self.snapshot.counts.sell,
            "buy_sell_ratio": self.buy_sell_ratio(),
            "largest_positive_strike": (
                positive.strike if positive else None
            ),
            "largest_positive_flow": (
                positive.net if positive else None
            ),
            "largest_negative_strike": (
                negative.strike if negative else None
            ),
            "largest_negative_flow": (
                negative.net if negative else None
            ),
        }