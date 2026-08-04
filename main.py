from datetime import datetime

from app.config.settings import settings


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


if __name__ == "__main__":
    main()