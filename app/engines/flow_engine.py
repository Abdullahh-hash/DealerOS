from app.models.flow_snapshot import (
    FlowSnapshot,
    FlowStrike,
)

from app.models.flow_summary import FlowSummary


class FlowEngine:
    """
    Analyzes options premium-flow data from FreeFlow.

    Important distinction:

        net premium
            = premium buying minus premium selling

        directional premium
            = bullish-classified premium minus
              bearish-classified premium
    """

    def __init__(
        self,
        snapshot: FlowSnapshot,
    ):
        self.snapshot = snapshot

    # ---------------------------------------------------------
    # Directional State
    # ---------------------------------------------------------

    def directional_flow_state(
        self,
    ) -> str:
        """
        Classify the sign of FreeFlow net directional
        premium.

        This describes option-flow classification.
        It is not a DealerOS price prediction.
        """

        if self.snapshot.net_directional > 0:
            return "Bullish"

        if self.snapshot.net_directional < 0:
            return "Bearish"

        return "Neutral"

    # ---------------------------------------------------------
    # Classified Trade Count Ratio
    # ---------------------------------------------------------

    def classified_buy_sell_ratio(
        self,
    ) -> float:

        sells = self.snapshot.counts.sell

        if sells == 0:
            return float("inf")

        return (
            self.snapshot.counts.buy
            / sells
        )

    # ---------------------------------------------------------
    # Call Premium Balance
    # ---------------------------------------------------------

    def largest_call_buy_balance(
        self,
    ) -> FlowStrike | None:

        candidates = [
            item
            for item in self.snapshot.by_strike
            if item.call_net > 0
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: item.call_net,
        )

    def largest_call_sell_balance(
        self,
    ) -> FlowStrike | None:

        candidates = [
            item
            for item in self.snapshot.by_strike
            if item.call_net < 0
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: item.call_net,
        )

    # ---------------------------------------------------------
    # Put Premium Balance
    # ---------------------------------------------------------

    def largest_put_buy_balance(
        self,
    ) -> FlowStrike | None:

        candidates = [
            item
            for item in self.snapshot.by_strike
            if item.put_net > 0
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: item.put_net,
        )

    def largest_put_sell_balance(
        self,
    ) -> FlowStrike | None:

        candidates = [
            item
            for item in self.snapshot.by_strike
            if item.put_net < 0
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: item.put_net,
        )

    # ---------------------------------------------------------
    # Aggressor Premium Balance
    # ---------------------------------------------------------

    def largest_premium_buy_balance(
        self,
    ) -> FlowStrike | None:
        """
        Largest positive provider-returned strike net.

        net = call_net + put_net
        """

        candidates = [
            item
            for item in self.snapshot.by_strike
            if item.aggressor_net > 0
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: item.aggressor_net,
        )

    def largest_premium_sell_balance(
        self,
    ) -> FlowStrike | None:
        """
        Largest negative provider-returned strike net.

        net = call_net + put_net
        """

        candidates = [
            item
            for item in self.snapshot.by_strike
            if item.aggressor_net < 0
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: item.aggressor_net,
        )

    # ---------------------------------------------------------
    # Directional Strike Premium
    # ---------------------------------------------------------

    def strongest_bullish_directional_strike(
        self,
    ) -> FlowStrike | None:
        """
        Strongest positive directional balance among
        provider-returned strikes.

        directional_net = call_net - put_net
        """

        candidates = [
            item
            for item in self.snapshot.by_strike
            if item.directional_net > 0
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: item.directional_net,
        )

    def strongest_bearish_directional_strike(
        self,
    ) -> FlowStrike | None:
        """
        Strongest negative directional balance among
        provider-returned strikes.

        directional_net = call_net - put_net
        """

        candidates = [
            item
            for item in self.snapshot.by_strike
            if item.directional_net < 0
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: item.directional_net,
        )

    # ---------------------------------------------------------
    # Public Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> FlowSummary:

        call_buy = (
            self.largest_call_buy_balance()
        )

        call_sell = (
            self.largest_call_sell_balance()
        )

        put_buy = (
            self.largest_put_buy_balance()
        )

        put_sell = (
            self.largest_put_sell_balance()
        )

        premium_buy = (
            self.largest_premium_buy_balance()
        )

        premium_sell = (
            self.largest_premium_sell_balance()
        )

        bullish = (
            self.strongest_bullish_directional_strike()
        )

        bearish = (
            self.strongest_bearish_directional_strike()
        )

        return FlowSummary(
            # ------------------------------------------------
            # Market context
            # ------------------------------------------------

            symbol=self.snapshot.symbol,
            expiry=self.snapshot.expiry,
            timestamp=self.snapshot.timestamp,

            spot=self.snapshot.spot,
            dte=self.snapshot.dte,
            window_min=self.snapshot.window_min,

            # ------------------------------------------------
            # Directional flow
            # ------------------------------------------------

            directional_flow_state=(
                self.directional_flow_state()
            ),

            bull_premium=(
                self.snapshot.bull_premium
            ),

            bear_premium=(
                self.snapshot.bear_premium
            ),

            net_directional=(
                self.snapshot.net_directional
            ),

            net_delta_notional=(
                self.snapshot.net_delta_notional
            ),

            # ------------------------------------------------
            # Aggressor premium
            # ------------------------------------------------

            net_premium=(
                self.snapshot.net_premium
            ),

            call_buy_premium=(
                self.snapshot.call_premium.buy
            ),

            call_sell_premium=(
                self.snapshot.call_premium.sell
            ),

            put_buy_premium=(
                self.snapshot.put_premium.buy
            ),

            put_sell_premium=(
                self.snapshot.put_premium.sell
            ),

            put_call_premium_ratio=(
                self.snapshot.put_call_prem_ratio
            ),

            # ------------------------------------------------
            # Counts
            # ------------------------------------------------

            classified_count=(
                self.snapshot.counts.classified
            ),

            buy_count=(
                self.snapshot.counts.buy
            ),

            sell_count=(
                self.snapshot.counts.sell
            ),

            classified_buy_sell_ratio=(
                self.classified_buy_sell_ratio()
            ),

            # ------------------------------------------------
            # Returned strikes
            # ------------------------------------------------

            returned_strike_count=len(
                self.snapshot.by_strike
            ),

            largest_call_buy_strike=(
                call_buy.strike
                if call_buy
                else None
            ),

            largest_call_buy_value=(
                call_buy.call_net
                if call_buy
                else None
            ),

            largest_call_sell_strike=(
                call_sell.strike
                if call_sell
                else None
            ),

            largest_call_sell_value=(
                call_sell.call_net
                if call_sell
                else None
            ),

            largest_put_buy_strike=(
                put_buy.strike
                if put_buy
                else None
            ),

            largest_put_buy_value=(
                put_buy.put_net
                if put_buy
                else None
            ),

            largest_put_sell_strike=(
                put_sell.strike
                if put_sell
                else None
            ),

            largest_put_sell_value=(
                put_sell.put_net
                if put_sell
                else None
            ),

            largest_premium_buy_strike=(
                premium_buy.strike
                if premium_buy
                else None
            ),

            largest_premium_buy_value=(
                premium_buy.aggressor_net
                if premium_buy
                else None
            ),

            largest_premium_sell_strike=(
                premium_sell.strike
                if premium_sell
                else None
            ),

            largest_premium_sell_value=(
                premium_sell.aggressor_net
                if premium_sell
                else None
            ),

            strongest_bullish_directional_strike=(
                bullish.strike
                if bullish
                else None
            ),

            strongest_bullish_directional_value=(
                bullish.directional_net
                if bullish
                else None
            ),

            strongest_bearish_directional_strike=(
                bearish.strike
                if bearish
                else None
            ),

            strongest_bearish_directional_value=(
                bearish.directional_net
                if bearish
                else None
            ),

            returned_strikes=list(
                self.snapshot.by_strike
            ),
        )