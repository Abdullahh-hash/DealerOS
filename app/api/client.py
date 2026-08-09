import requests

from app.config.settings import settings


class FreeFlowClient:
    """Client for communicating with the FreeFlow API."""

    def __init__(self):
        self.base_url = settings.base_url
        self.headers = {
            "X-API-Key": settings.api_key
        }
        self.timeout = settings.request_timeout

    def get_snapshot(self, symbol: str, expiry: str):
        """Retrieve a snapshot for a given symbol and expiry."""

        response = requests.get(
            f"{self.base_url}/snapshot",
            headers=self.headers,
            params={
                "symbol": symbol,
                "exp": expiry,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def get_expirations(self, symbol: str):
        """Retrieve available expirations."""

        response = requests.get(
            f"{self.base_url}/expirations",
            headers=self.headers,
            params={
                "symbol": symbol,
            },
            timeout=self.timeout,
        )

        print("=" * 60)
        print("FREEFLOW DEBUG")
        print("=" * 60)
        print("URL         :", response.url)
        print("Status Code :", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))
        print("Response:")
        print(response.text)
        print("=" * 60)

        response.raise_for_status()

        return response.json()

    def get_flow(self, symbol: str, expiry: str):
        """Retrieve flow data."""
        response = requests.get(
            "https://www.free-flow.site/public/flow",
            headers=self.headers,
            params={
                "symbol": symbol,
                "exp": expiry,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()