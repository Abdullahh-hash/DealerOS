from datetime import datetime
from pprint import pprint
from app.services.snapshot_parser import parse_snapshot
from app.api.client import FreeFlowClient
from app.config.settings import settings


def print_banner():
    print("=" * 60)
    print("DealerOS v0.3")
    print("Snapshot Engine")
    print("=" * 60)
    print(f"Started : {datetime.now()}")
    print()


def main():
    print_banner()

    print("Configuration")
    print("-" * 60)

    print(f"Base URL : {settings.base_url}")
    print(f"Symbol   : {settings.default_symbol}")
    print(f"Timeout  : {settings.request_timeout}")

    if settings.api_key:
        print("API Key  : Loaded ✅")
    else:
        print("API Key  : Missing ❌")
        return

    print()
    print("Connecting to FreeFlow...")
    print("-" * 60)

    try:
        client = FreeFlowClient()

        expirations = client.get_expirations(settings.default_symbol)

        expiry_list = expirations["expirations"]
        selected_expiry = expiry_list[0]

        print("Connection       : Success ✅")
        print(f"Selected Expiry : {selected_expiry}")

        print()
        print("Downloading Snapshot...")
        print("-" * 60)

        raw_snapshot = client.get_snapshot(
            settings.default_symbol,
            selected_expiry,
        )

        snapshot = parse_snapshot(raw_snapshot)

        print()
        print("Parsed Snapshot")
        print("-" * 60)
        print(f"Symbol      : {snapshot.symbol}")
        print(f"Spot        : {snapshot.spot}")
        print(f"Expiry      : {snapshot.exp}")
        print(f"Total GEX   : {snapshot.total_gex}")
        print(f"Contracts   : {len(snapshot.contracts)}")

        if snapshot.contracts:
            first = snapshot.contracts[0]

            print()
            print("First Contract")
            print("-" * 60)
            print(f"Strike : {first.strike}")
            print(f"Right  : {first.right}")
            print(f"Gamma  : {first.gamma}")
            print(f"GEX    : {first.gex}")

       

    except Exception as e:
        print("Connection : Failed ❌")
        print(f"Reason     : {e}")


if __name__ == "__main__":
    main()