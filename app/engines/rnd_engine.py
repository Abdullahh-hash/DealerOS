from dataclasses import dataclass

import numpy as np


@dataclass
class RNDPoint:
    """
    One point on the risk-neutral density curve.
    """

    price: float
    density: float


class RNDEngine:
    """
    Builds a risk-neutral density from an option IV surface.

    The engine is expiry-agnostic.
    It receives the selected expiry and its option-chain data.
    """

    def __init__(
        self,
        spot: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.0,
    ):
        self.spot = spot
        self.time_to_expiry = time_to_expiry
        self.risk_free_rate = risk_free_rate

    def build_price_grid(
        self,
        strikes: list[float],
        points: int = 500,
    ) -> np.ndarray:
        """
        Build the price grid used for the RND curve.
        """

        if not strikes:
            raise ValueError("No strikes supplied.")

        minimum = min(strikes)
        maximum = max(strikes)

        if minimum >= maximum:
            raise ValueError("Invalid strike range.")

        return np.linspace(
            minimum,
            maximum,
            points,
        )

    def build_density(
        self,
        prices: np.ndarray,
        call_prices: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the raw risk-neutral density from
        the second derivative of call price with respect
        to strike.

        This is the first RND building block.
        """

        if len(prices) != len(call_prices):
            raise ValueError(
                "Prices and call prices must have the same length."
            )

        if len(prices) < 3:
            raise ValueError(
                "At least three price points are required."
            )

        first_derivative = np.gradient(
            call_prices,
            prices,
        )

        second_derivative = np.gradient(
            first_derivative,
            prices,
        )

        discount_factor = np.exp(
            self.risk_free_rate * self.time_to_expiry
        )

        density = discount_factor * second_derivative

        # Numerical interpolation can create tiny negative values.
        # The density itself cannot be negative.
        density = np.maximum(density, 0.0)

        return density

    def normalize_density(
        self,
        prices: np.ndarray,
        density: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize the density so its total area equals 1.
        """

        area = np.trapezoid(
            density,
            prices,
        )

        if area <= 0:
            raise ValueError(
                "Density area must be greater than zero."
            )

        return density / area

    def build_curve(
        self,
        prices: np.ndarray,
        call_prices: np.ndarray,
    ) -> list[RNDPoint]:
        """
        Build the final normalized (price, density) curve.
        """

        density = self.build_density(
            prices,
            call_prices,
        )

        density = self.normalize_density(
            prices,
            density,
        )

        return [
            RNDPoint(
                price=float(price),
                density=float(value),
            )
            for price, value in zip(
                prices,
                density,
            )
        ]