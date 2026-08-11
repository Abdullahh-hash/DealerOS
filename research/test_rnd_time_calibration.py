import json
import math
from pathlib import Path
from statistics import NormalDist, median


SNAPSHOT_DIR = Path("data") / "rnd_snapshots"

ATM_RANGE = 100.0

normal = NormalDist()


def infer_time_and_rate(
    spot: float,
    strike: float,
    sigma: float,
    delta: float,
    gamma: float,
    vega: float,
):
    # --------------------------------------------------
    # FreeFlow vega appears to be quoted per 1 vol point.
    #
    # BS:
    #
    # gamma = phi(d1) / (S * sigma * sqrt(T))
    #
    # vega_per_1pct =
    #     S * phi(d1) * sqrt(T) / 100
    #
    # Therefore:
    #
    # T = 100 * vega /
    #     (gamma * S^2 * sigma)
    # --------------------------------------------------

    time_to_expiry = (
        100.0 * vega
        / (
            gamma
            * spot
            * spot
            * sigma
        )
    )

    if time_to_expiry <= 0:
        return None


    # --------------------------------------------------
    # Under q = 0:
    #
    # Call delta = N(d1)
    #
    # Use delta to infer the rate that would reproduce
    # FreeFlow's model.
    # --------------------------------------------------

    if not (0.0 < delta < 1.0):
        return None

    d1 = normal.inv_cdf(delta)

    sqrt_t = math.sqrt(
        time_to_expiry
    )

    rate = (
        (
            d1
            * sigma
            * sqrt_t
            - math.log(
                spot / strike
            )
        )
        / time_to_expiry
        - 0.5
        * sigma
        * sigma
    )

    return (
        time_to_expiry,
        rate,
    )


files = sorted(
    SNAPSHOT_DIR.glob(
        "snapshot_*.json"
    )
)

if not files:
    raise RuntimeError(
        "No snapshot files found."
    )


print()
print("=" * 118)
print(
    "FREEFLOW 0DTE TIME / RATE CALIBRATION"
)
print("=" * 118)

print(
    f"{'FILE TIME':<20}"
    f"{'API TIME':<20}"
    f"{'SPOT':>11}"
    f"{'N':>5}"
    f"{'T DAYS':>12}"
    f"{'T HOURS':>12}"
    f"{'T MIN HRS':>12}"
    f"{'T MAX HRS':>12}"
    f"{'RATE %':>12}"
)

print("-" * 118)


for path in files:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)


    spot = float(
        data["spot"]
    )

    inferred_times = []
    inferred_rates = []


    for row in data["rows"]:

        if row.get("right") != "C":
            continue

        strike = row.get("strike")
        iv_pct = row.get("iv_pct")
        delta = row.get("delta")
        gamma = row.get("gamma")
        vega = row.get("vega")

        if any(
            value is None
            for value in (
                strike,
                iv_pct,
                delta,
                gamma,
                vega,
            )
        ):
            continue


        strike = float(strike)
        sigma = float(iv_pct) / 100.0
        delta = float(delta)
        gamma = float(gamma)
        vega = float(vega)


        if (
            sigma <= 0
            or gamma <= 0
            or vega <= 0
        ):
            continue


        if (
            abs(
                strike - spot
            )
            > ATM_RANGE
        ):
            continue


        result = infer_time_and_rate(
            spot=spot,
            strike=strike,
            sigma=sigma,
            delta=delta,
            gamma=gamma,
            vega=vega,
        )

        if result is None:
            continue


        time_to_expiry, rate = result

        inferred_times.append(
            time_to_expiry
        )

        inferred_rates.append(
            rate
        )


    if not inferred_times:

        print(
            f"{path.name:<20} "
            f"NO VALID ROWS"
        )

        continue


    median_t = median(
        inferred_times
    )

    median_rate = median(
        inferred_rates
    )


    t_days = (
        median_t * 365.0
    )

    t_hours = (
        t_days * 24.0
    )


    min_hours = (
        min(inferred_times)
        * 365.0
        * 24.0
    )

    max_hours = (
        max(inferred_times)
        * 365.0
        * 24.0
    )


    # Filename timestamp
    #
    # snapshot_2026-08-10_20260810_203429.json
    #
    parts = path.stem.split("_")

    file_time = (
        f"{parts[-2]} "
        f"{parts[-1]}"
    )


    api_timestamp = str(
        data.get(
            "timestamp",
            ""
        )
    )

    api_time = (
        api_timestamp[:19]
        if api_timestamp
        else "None"
    )


    print(
        f"{file_time:<20}"
        f"{api_time:<20}"
        f"{spot:>11.3f}"
        f"{len(inferred_times):>5}"
        f"{t_days:>12.6f}"
        f"{t_hours:>12.6f}"
        f"{min_hours:>12.6f}"
        f"{max_hours:>12.6f}"
        f"{median_rate * 100:>12.4f}"
    )


print("-" * 118)

print()
print(
    "Reference:"
)

print(
    "0.5 / 365 years = "
    f"{0.5 / 365.0:.10f}"
)

print(
    "0.5 calendar days = "
    "12.000000 hours"
)

print()