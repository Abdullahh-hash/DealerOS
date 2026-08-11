from app.models.gex_level import GEXLevel
from app.models.option_contract import OptionContract


class GammaAnalyzer:
    """
    Performs DealerOS gamma-exposure analysis.

    FreeFlow contract GEX is OI-based gamma exposure.
    This analyzer aggregates that supplied exposure
    across strikes.
    """

    def __init__(self, snapshot):
        self.snapshot = snapshot

    # ---------------------------------------------------------
    # Dealer Gamma State
    # ---------------------------------------------------------

    def gamma_state(self) -> str:
        """
        Classify the sign of aggregate FreeFlow GEX.

        This describes the supplied GEX regime metric.
        It should not be interpreted as proof of actual
        dealer inventory direction without validating the
        provider's positioning assumptions.
        """

        if self.snapshot.total_gex is None:
            return "Unknown"

        if self.snapshot.total_gex >= 0:
            return "Long Gamma"

        return "Short Gamma"

    # ---------------------------------------------------------
    # Largest Contract GEX
    # ---------------------------------------------------------

    def largest_call_gex(
        self,
    ) -> OptionContract | None:
        """
        Largest Call GEX contract.
        """

        calls = [
            contract
            for contract in self.snapshot.contracts
            if (
                contract.right == "C"
                and contract.gex is not None
            )
        ]

        if not calls:
            return None

        return max(
            calls,
            key=lambda contract: contract.gex,
        )

    def largest_put_gex(
        self,
    ) -> OptionContract | None:
        """
        Most negative Put GEX contract under
        FreeFlow's signed GEX convention.
        """

        puts = [
            contract
            for contract in self.snapshot.contracts
            if (
                contract.right == "P"
                and contract.gex is not None
            )
        ]

        if not puts:
            return None

        return min(
            puts,
            key=lambda contract: contract.gex,
        )

    # ---------------------------------------------------------
    # GEX Profile
    # ---------------------------------------------------------

    def gex_levels(
        self,
    ) -> list[GEXLevel]:
        """
        Aggregate Call GEX, Put GEX and Net GEX
        by strike.
        """

        levels: dict[
            float,
            GEXLevel,
        ] = {}

        for contract in self.snapshot.contracts:

            if contract.gex is None:
                continue

            strike = float(
                contract.strike
            )

            if strike not in levels:
                levels[strike] = GEXLevel(
                    strike=strike
                )

            level = levels[strike]

            gex = float(
                contract.gex
            )

            if contract.right == "C":
                level.call_gex += gex

            elif contract.right == "P":
                level.put_gex += gex

            level.net_gex += gex

        return sorted(
            levels.values(),
            key=lambda level: level.strike,
        )

    def gex_profile(
        self,
    ) -> list[GEXLevel]:
        """
        Alias intended for dashboard/profile consumers.
        """

        return self.gex_levels()

    # ---------------------------------------------------------
    # Net-GEX Concentrations
    # ---------------------------------------------------------

    def strongest_positive_net_gex(
        self,
    ) -> GEXLevel | None:
        """
        Strike with the largest positive net GEX.
        """

        positive = [
            level
            for level in self.gex_levels()
            if level.net_gex > 0
        ]

        if not positive:
            return None

        return max(
            positive,
            key=lambda level: level.net_gex,
        )

    def strongest_negative_net_gex(
        self,
    ) -> GEXLevel | None:
        """
        Strike with the most negative net GEX.
        """

        negative = [
            level
            for level in self.gex_levels()
            if level.net_gex < 0
        ]

        if not negative:
            return None

        return min(
            negative,
            key=lambda level: level.net_gex,
        )

    # ---------------------------------------------------------
    # Net-GEX Sign Change
    # ---------------------------------------------------------

    def nearest_net_gex_sign_change_strike(
        self,
    ) -> float | None:
        """
        Find the nearest strike at which the strike-level
        net-GEX profile changes sign.

        IMPORTANT:

        This is NOT a true portfolio zero-gamma / gamma-flip
        calculation. It only describes a sign change in the
        current GEX profile across adjacent strikes.
        """

        levels = self.gex_levels()

        crossings: list[float] = []

        previous = None

        for level in levels:

            if previous is None:
                previous = level
                continue

            if previous.net_gex == 0:
                crossings.append(
                    previous.strike
                )

            elif level.net_gex == 0:
                crossings.append(
                    level.strike
                )

            elif (
                previous.net_gex < 0
                and level.net_gex > 0
            ):
                crossings.append(
                    level.strike
                )

            elif (
                previous.net_gex > 0
                and level.net_gex < 0
            ):
                crossings.append(
                    level.strike
                )

            previous = level

        if not crossings:
            return None

        return min(
            crossings,
            key=lambda strike: abs(
                strike - self.snapshot.spot
            ),
        )