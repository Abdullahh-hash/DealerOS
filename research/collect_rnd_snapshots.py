import json
import time
from datetime import datetime
from pathlib import Path
from getpass import getpass

import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_URL = "https://www.free-flow.site/public/snapshot"

SYMBOL = "NDX"

EXPIRY = "2026-08-10"

INTERVAL_MINUTES = 15

OUTPUT_DIR = Path("data") / "rnd_snapshots"

TIMEOUT = (10, 90)


# --------------------------------------------------
# Create output folder
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# API key
#
# Enter once when collector starts.
# It is NOT written to the file or source code.
# --------------------------------------------------

api_key = getpass(
    "Enter FreeFlow API key: "
)

headers = {
    "X-API-Key": api_key
}


# --------------------------------------------------
# Snapshot function
# --------------------------------------------------

def fetch_snapshot():

    params = {
        "symbol": SYMBOL,
        "exp": EXPIRY,
    }

    timestamp = datetime.now()

    filename = (
        f"snapshot_{EXPIRY}_"
        f"{timestamp:%Y%m%d_%H%M%S}.json"
    )

    filepath = (
        OUTPUT_DIR / filename
    )

    try:

        response = requests.get(
            BASE_URL,
            headers=headers,
            params=params,
            timeout=TIMEOUT,
        )

        print()
        print(
            f"[{timestamp:%Y-%m-%d %H:%M:%S}] "
            f"HTTP {response.status_code}"
        )

        response.raise_for_status()

        data = response.json()


        # ------------------------------------------
        # Basic validation
        # ------------------------------------------

        if not isinstance(data, dict):
            raise ValueError(
                "Snapshot response is not a JSON object."
            )

        if "rows" not in data:
            raise ValueError(
                "Snapshot JSON has no 'rows' field."
            )


        # ------------------------------------------
        # Save snapshot
        # ------------------------------------------

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
            )


        print(
            f"Saved : {filepath}"
        )

        print(
            f"Spot  : {data.get('spot')}"
        )

        print(
            f"DTE   : {data.get('dte')}"
        )

        print(
            f"ATM IV: {data.get('atm_iv')}"
        )

        print(
            f"Rows  : {len(data.get('rows', []))}"
        )


    except KeyboardInterrupt:
        raise

    except Exception as exc:

        print(
            f"ERROR : {exc}"
        )


# --------------------------------------------------
# Main loop
# --------------------------------------------------

print()
print("=" * 70)
print("DEALEROS RND SNAPSHOT COLLECTOR")
print("=" * 70)

print(f"Symbol     : {SYMBOL}")
print(f"Expiry     : {EXPIRY}")
print(
    f"Interval   : "
    f"{INTERVAL_MINUTES} minutes"
)
print(f"Output     : {OUTPUT_DIR}")

print()
print(
    "Press Ctrl+C whenever you want to stop."
)

print()


try:

    while True:

        fetch_snapshot()

        print(
            f"Next snapshot in "
            f"{INTERVAL_MINUTES} minutes..."
        )

        time.sleep(
            INTERVAL_MINUTES * 60
        )


except KeyboardInterrupt:

    print()
    print("=" * 70)
    print("Collector stopped.")
    print("=" * 70)