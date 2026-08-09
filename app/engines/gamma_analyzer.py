from collections import defaultdict

from app.models.gamma_level import GammaLevel
from app.models.option_contract import OptionContract


class GammaAnalyzer:
    """
    Performs all gamma-related calculations.
    """

    def __init__(self, snapshot):
        self.snapshot = snapshot

    # ---------------------------------------------------------
    # Basic Dealer Metrics
    # ---------------------------------------------------------

    def dealer_bias(self) -> str:
        """
        Determine overall dealer gamma regime.
        """

        if self.snapshot.total_gex >= 0:
            return "Long Gamma"

        return "Short Gamma"

    def largest_call_gex(self) -> OptionContract | None:
        """
        Largest positive Call GEX contract.
        """

        calls = [
            c for c in self.snapshot.contracts
            if c.right == "C"
        ]

        if not calls:
            return None

        return max(calls, key=lambda c: c.gex)

    def largest_put_gex(self) -> OptionContract | None:
        """
        Largest magnitude Put GEX contract.
        """

        puts = [
            c for c in self.snapshot.contracts
            if c.right == "P"
        ]

        if not puts:
            return None

        return min(puts, key=lambda c: c.gex)

    # ---------------------------------------------------------
    # Gamma Profile
    # ---------------------------------------------------------

    def net_gamma_levels(self) -> list[GammaLevel]:
        """
        Aggregate gamma by strike.
        """

        levels = defaultdict(
            lambda: GammaLevel(strike=0)
        )

        for contract in self.snapshot.contracts:

            strike = contract.strike

            if strike not in levels:
                levels[strike] = GammaLevel(strike=strike)

            level = levels[strike]

            if contract.right == "C":
                level.call_gamma += contract.gex

            else:
                level.put_gamma += contract.gex

            level.net_gamma += contract.gex

        return sorted(
            levels.values(),
            key=lambda level: level.strike,
        )

    def gamma_profile(self) -> list[GammaLevel]:
        """
        Alias for dashboard usage.
        """

        return self.net_gamma_levels()

    # ---------------------------------------------------------
    # Gamma Flip
    # ---------------------------------------------------------

    def gamma_flip(self):
        """
        Find the gamma flip closest to the current spot price.
        """

        levels = self.net_gamma_levels()

        crossings = []

        previous = None

        for level in levels:

            if previous is None:
                previous = level
                continue

            if (
                previous.net_gamma < 0 <= level.net_gamma
                or
                previous.net_gamma > 0 >= level.net_gamma
            ):
                crossings.append(level.strike)

            previous = level

        if not crossings:
            return None

        return min(
            crossings,
            key=lambda strike: abs(strike - self.snapshot.spot),
        )