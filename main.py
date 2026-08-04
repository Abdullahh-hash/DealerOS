from datetime import datetime

from app.config.settings import settings
from app.api.client import FreeFlowClient


def print_banner():
    print("=" * 60)
    print("DealerOS v0.1")
    print("Heartbeat")
    print("=" * 60)
    print(f"Started : {datetime.now()}")
    print()


def main():
    print_banner()

    print("Configuration")
    print("-" * 60)

    print(f"Base URL : {settings.base_url}")
    print(f"Symbol   : {settings.default_symbol}")
    print(f"Expiry   : {settings.default_expiry}")
    print(f"Timeout  : {settings.request_timeout}")

    if settings.api_key:
        print("API Key  : Loaded ✅")
    else:
        print("API Key  : Missing ❌")


    print()
    print("API Test")
    print("-" * 60)

    try:
        client = FreeFlowClient()

        expirations = client.get_expirations(settings.default_symbol)

        print("Connection : Success ✅")
        print("Available Expirations:")
        print(expirations)

    except Exception as e:
        print("Connection : Failed ❌")
        print(f"Reason     : {e}")


if __name__ == "__main__":
    main()