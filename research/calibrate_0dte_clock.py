import json
import math
from pathlib import Path
from statistics import NormalDist, median


SNAPSHOT_DIR = Path("data") / "rnd_snapshots"
ATM_RANGE = 100.0

normal = NormalDist()

files = sorted(
    SNAPSHOT_DIR.glob("snapshot_*.json")
)

print()
print("=" * 105)
print("DEALEROS — FREEFLOW 0DTE CLOCK CALIBRATION")
print("=" * 105)
print(f"Files found: {len(files)}")
print()

if not files:
    raise RuntimeError(
        "No snapshot files found."
    )

print(
    f"{'FILE':<42}"
    f"{'SPOT':>11}"
    f"{'N':>5}"
    f"{'T HOURS':>12}"
    f"{'MIN HRS':>12}"
    f"{'MAX HRS':>12}"
    f"{'RATE %':>11}"
)

print("-" * 105)


for path in files:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    spot = float(data["spot"])

    times = []
    rates = []

    for row in data["rows"]:

        if row.get("right") != "C":
            continue

        strike = row.get("strike")
        iv_pct = row.get("iv_pct")
        delta = row.get("delta")
        gamma = row.get("gamma")
        vega = row.get("vega")

        if (
            strike is None
            or iv_pct is None
            or delta is None
            or gamma is None
            or vega is None
        ):
            continue

        strike = float(strike)
        sigma = float(iv_pct) / 100.0
        delta = float(delta)
        gamma = float(gamma)
        vega = float(vega)

        if abs(strike - spot) > ATM_RANGE:
            continue

        if sigma <= 0:
            continue

        if gamma <= 0:
            continue

        if vega <= 0:
            continue

        if not (0.0 < delta < 1.0):
            continue

        # ------------------------------------------
        # Infer T from BS gamma + vega.
        #
        # Vega is quoted per 1 volatility point.
        # ------------------------------------------

        T = (
            100.0
            * vega
            / (
                gamma
                * spot
                * spot
                * sigma
            )
        )

        if T <= 0:
            continue

        # ------------------------------------------
        # Infer r from call delta assuming q = 0.
        # ------------------------------------------

        d1 = normal.inv_cdf(delta)

        r = (
            (
                d1
                * sigma
                * math.sqrt(T)
                - math.log(spot / strike)
            )
            / T
            - 0.5 * sigma * sigma
        )

        times.append(T)
        rates.append(r)

    if not times:

        print(
            f"{path.name:<42}"
            f"{spot:>11.3f}"
            f"{0:>5}"
            f"{'NO DATA':>47}"
        )

        continue

    hours = [
        T * 365.0 * 24.0
        for T in times
    ]

    median_hours = median(hours)
    min_hours = min(hours)
    max_hours = max(hours)

    median_rate = (
        median(rates) * 100.0
    )

    print(
        f"{path.name:<42}"
        f"{spot:>11.3f}"
        f"{len(times):>5}"
        f"{median_hours:>12.6f}"
        f"{min_hours:>12.6f}"
        f"{max_hours:>12.6f}"
        f"{median_rate:>11.4f}"
    )


print("-" * 105)

print()
print("Reference:")
print("0.5 calendar days = 12.000000 hours")
print()