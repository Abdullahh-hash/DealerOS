from datetime import datetime


def print_banner():
    print("=" * 50)
    print("DealerOS v0.1")
    print("Heartbeat Initialized")
    print("=" * 50)
    print(f"Started: {datetime.now()}")
    print()


def main():
    print_banner()

    print("Status: READY")
    print("API Connection: Waiting...")
    print("Database: Waiting...")
    print("Dealer Engine: Waiting...")
    print()


if __name__ == "__main__":
    main()