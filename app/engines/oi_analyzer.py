from app.models.oi_level import OILevel
from app.models.option_contract import OptionContract


class OIAnalyzer:
    """
    Analyzes raw open-interest structure.

    OI describes where outstanding option positioning
    is concentrated independently of the current gamma
    sensitivity of those positions.
    """

    def __init__(self, snapshot):
        self.snapshot = snapshot

    # ---------------------------------------------------------
    # Largest Call OI
    # ---------------------------------------------------------

    def largest_call_oi(
        self,
    ) -> OptionContract | None:

        calls = [
            contract
            for contract in self.snapshot.contracts
            if (
                contract.right == "C"
                and contract.oi is not None
            )
        ]

        if not calls:
            return None

        return max(
            calls,
            key=lambda contract: contract.oi,
        )

    # ---------------------------------------------------------
    # Largest Put OI
    # ---------------------------------------------------------

    def largest_put_oi(
        self,
    ) -> OptionContract | None:

        puts = [
            contract
            for contract in self.snapshot.contracts
            if (
                contract.right == "P"
                and contract.oi is not None
            )
        ]

        if not puts:
            return None

        return max(
            puts,
            key=lambda contract: contract.oi,
        )

    # ---------------------------------------------------------
    # OI Profile
    # ---------------------------------------------------------

    def oi_levels(
        self,
    ) -> list[OILevel]:
        """
        Aggregate Call OI, Put OI and total OI by strike.
        """

        levels: dict[
            float,
            OILevel,
        ] = {}

        for contract in self.snapshot.contracts:

            if contract.oi is None:
                continue

            strike = float(
                contract.strike
            )

            if strike not in levels:
                levels[strike] = OILevel(
                    strike=strike
                )

            level = levels[strike]

            oi = int(
                contract.oi
            )

            if contract.right == "C":
                level.call_oi += oi

            elif contract.right == "P":
                level.put_oi += oi

            level.total_oi += oi

        return sorted(
            levels.values(),
            key=lambda level: level.strike,
        )

    def oi_profile(
        self,
    ) -> list[OILevel]:
        """
        Alias intended for dashboard/profile consumers.
        """

        return self.oi_levels()

    # ---------------------------------------------------------
    # Largest Combined OI
    # ---------------------------------------------------------

    def largest_total_oi_level(
        self,
    ) -> OILevel | None:

        levels = self.oi_levels()

        if not levels:
            return None

        return max(
            levels,
            key=lambda level: level.total_oi,
        )