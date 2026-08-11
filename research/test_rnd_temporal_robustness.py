import json
import math
from pathlib import Path
from statistics import NormalDist, median

import numpy as np
from scipy.interpolate import make_smoothing_spline

from app.services.black_scholes import black_scholes_call


# ============================================================
# CONFIGURATION
# ============================================================

SNAPSHOT_DIR = Path("data") / "rnd_snapshots"

FIT_RANGE = 500.0
GRID_STEP = 5.0

ATM_CALIBRATION_RANGE = 100.0

FOLDS = 5

METHODS = [
    "SPLINE_1e-3",
    "SPLINE_2e-3",
    "SPLINE_3e-3",
    "SPLINE_4e-3",
    "SPLINE_5e-3",
    "SPLINE_7e-3",
    "SPLINE_1e-2",
]

X_SCALE = 1000.0

MONO_TOL = 1e-10
RND_NEG_TOL = -1e-12


normal = NormalDist()


# ============================================================
# FREEFLOW MODEL-CLOCK CALIBRATION
# ============================================================

def calibrate_model_inputs(data):

    spot = float(data["spot"])

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

        if abs(strike - spot) > ATM_CALIBRATION_RANGE:
            continue

        if sigma <= 0:
            continue

        if gamma <= 0:
            continue

        if vega <= 0:
            continue

        if not (0.0 < delta < 1.0):
            continue

        # ----------------------------------------------------
        # Infer T from Black-Scholes gamma / vega relationship.
        #
        # FreeFlow vega is per 1 volatility percentage point.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Infer r from call delta assuming q = 0.
        #
        # delta = N(d1)
        # ----------------------------------------------------

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

        inferred_times.append(T)
        inferred_rates.append(r)

    if not inferred_times:
        raise RuntimeError(
            "Could not calibrate time-to-expiry."
        )

    if not inferred_rates:
        raise RuntimeError(
            "Could not calibrate risk-free rate."
        )

    return {
        "T": median(inferred_times),
        "rate": median(inferred_rates),
        "T_min": min(inferred_times),
        "T_max": max(inferred_times),
        "n": len(inferred_times),
    }


# ============================================================
# OTM IV SURFACE
# ============================================================

def build_otm_observations(
    data,
    forward,
):

    strikes = []
    ivs = []

    for row in data["rows"]:

        strike = row.get("strike")
        right = row.get("right")
        iv_pct = row.get("iv_pct")

        if strike is None:
            continue

        if right is None:
            continue

        if iv_pct is None:
            continue

        strike = float(strike)
        iv_pct = float(iv_pct)

        if iv_pct <= 0:
            continue

        if abs(strike - forward) > FIT_RANGE:
            continue

        # ----------------------------------------------------
        # OTM construction:
        #
        # puts below forward
        # calls above forward
        # ----------------------------------------------------

        use_row = (
            (
                strike < forward
                and right == "P"
            )
            or
            (
                strike > forward
                and right == "C"
            )
        )

        if not use_row:
            continue

        strikes.append(strike)
        ivs.append(iv_pct)

    if len(strikes) < 10:
        raise RuntimeError(
            "Not enough OTM observations."
        )

    order = np.argsort(strikes)

    strikes = np.asarray(
        strikes,
        dtype=float,
    )[order]

    ivs = np.asarray(
        ivs,
        dtype=float,
    )[order]

    # --------------------------------------------------------
    # Defensive uniqueness check.
    # --------------------------------------------------------

    unique_strikes, unique_indices = np.unique(
        strikes,
        return_index=True,
    )

    strikes = unique_strikes
    ivs = ivs[unique_indices]

    return strikes, ivs


# ============================================================
# SMOOTHERS
# ============================================================

def fit_predict(
    method,
    train_strikes,
    train_ivs,
    eval_strikes,
    forward,
):

    x_train = (
        train_strikes - forward
    ) / X_SCALE

    x_eval = (
        eval_strikes - forward
    ) / X_SCALE

    if method == "POLY4":

        if len(train_strikes) < 5:
            raise RuntimeError(
                "Not enough points for POLY4."
            )

        coefficients = np.polyfit(
            x_train,
            train_ivs,
            deg=4,
        )

        return np.polyval(
            coefficients,
            x_eval,
        )

    if method.startswith("SPLINE_"):

        lam = float(
            method.split("_")[1]
        )

        spline = make_smoothing_spline(
            x_train,
            train_ivs,
            lam=lam,
        )

        return spline(
            x_eval
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# ============================================================
# RND CALCULATION
# ============================================================

def build_rnd(
    method,
    train_strikes,
    train_ivs,
    grid,
    spot,
    forward,
    T,
    rate,
):

    fitted_iv_pct = fit_predict(
        method=method,
        train_strikes=train_strikes,
        train_ivs=train_ivs,
        eval_strikes=grid,
        forward=forward,
    )

    if not np.all(
        np.isfinite(fitted_iv_pct)
    ):
        return {
            "valid": False,
            "reason": "NONFINITE_IV",
        }

    if np.any(
        fitted_iv_pct <= 0
    ):
        return {
            "valid": False,
            "reason": "NONPOSITIVE_IV",
        }

    call_prices = np.array(
        [
            black_scholes_call(
                spot=spot,
                strike=float(K),
                time_to_expiry=T,
                volatility=float(iv) / 100.0,
                risk_free_rate=rate,
            )
            for K, iv
            in zip(
                grid,
                fitted_iv_pct,
            )
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Call-price monotonicity
    # --------------------------------------------------------

    call_diffs = np.diff(
        call_prices
    )

    mono_failures = int(
        np.sum(
            call_diffs > MONO_TOL
        )
    )

    # --------------------------------------------------------
    # Breeden-Litzenberger
    #
    # q(K) = exp(rT) * d²C/dK²
    # --------------------------------------------------------

    first_derivative = np.gradient(
        call_prices,
        grid,
    )

    second_derivative = np.gradient(
        first_derivative,
        grid,
    )

    density = (
        np.exp(rate * T)
        * second_derivative
    )

    negative_count = int(
        np.sum(
            density < RND_NEG_TOL
        )
    )

    area = float(
        np.trapezoid(
            density,
            grid,
        )
    )

    result = {
        "valid": True,
        "mono": mono_failures,
        "neg": negative_count,
        "area": area,
        "min_rnd": float(
            np.min(density)
        ),
        "max_rnd": float(
            np.max(density)
        ),
    }

    # --------------------------------------------------------
    # Do NOT clip negative density.
    #
    # Distribution statistics are calculated only if the
    # raw surface passes integrity checks.
    # --------------------------------------------------------

    if (
        mono_failures != 0
        or negative_count != 0
        or area <= 0
    ):

        result.update(
            {
                "mean": None,
                "mean_fwd": None,
                "median": None,
                "mode": None,
                "std": None,
            }
        )

        return result

    normalized = (
        density / area
    )

    normalized_area = float(
        np.trapezoid(
            normalized,
            grid,
        )
    )

    mean_value = float(
        np.trapezoid(
            grid * normalized,
            grid,
        )
    )

    variance = float(
        np.trapezoid(
            (
                (grid - mean_value) ** 2
            )
            * normalized,
            grid,
        )
    )

    std = math.sqrt(
        max(
            variance,
            0.0,
        )
    )

    # --------------------------------------------------------
    # CDF using trapezoidal accumulation
    # --------------------------------------------------------

    cdf = np.zeros_like(
        normalized
    )

    increments = (
        0.5
        * (
            normalized[:-1]
            + normalized[1:]
        )
        * np.diff(grid)
    )

    cdf[1:] = np.cumsum(
        increments
    )

    if cdf[-1] > 0:
        cdf = (
            cdf / cdf[-1]
        )

    median_value = float(
        np.interp(
            0.50,
            cdf,
            grid,
        )
    )

    mode_value = float(
        grid[
            np.argmax(
                normalized
            )
        ]
    )

    result.update(
        {
            "normalized_area": normalized_area,
            "mean": mean_value,
            "mean_fwd": (
                mean_value - forward
            ),
            "median": median_value,
            "mode": mode_value,
            "std": std,
        }
    )

    return result


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate(
    method,
    strikes,
    ivs,
    grid,
    spot,
    forward,
    T,
    rate,
):

    n = len(strikes)

    if n < 10:
        raise RuntimeError(
            "Not enough observations for CV."
        )

    interior_indices = np.arange(
        1,
        n - 1,
        dtype=int,
    )

    all_errors = []

    bad_folds = 0
    mono_total = 0
    neg_total = 0

    for fold in range(FOLDS):

        test_indices = (
            interior_indices[
                np.arange(
                    len(
                        interior_indices
                    )
                )
                % FOLDS
                == fold
            ]
        )

        test_set = set(
            test_indices.tolist()
        )

        train_indices = np.array(
            [
                i
                for i in range(n)
                if i not in test_set
            ],
            dtype=int,
        )

        train_strikes = (
            strikes[
                train_indices
            ]
        )

        train_ivs = (
            ivs[
                train_indices
            ]
        )

        test_strikes = (
            strikes[
                test_indices
            ]
        )

        test_ivs = (
            ivs[
                test_indices
            ]
        )

        predicted = fit_predict(
            method=method,
            train_strikes=train_strikes,
            train_ivs=train_ivs,
            eval_strikes=test_strikes,
            forward=forward,
        )

        errors = (
            predicted
            - test_ivs
        )

        all_errors.extend(
            errors.tolist()
        )

        fold_rnd = build_rnd(
            method=method,
            train_strikes=train_strikes,
            train_ivs=train_ivs,
            grid=grid,
            spot=spot,
            forward=forward,
            T=T,
            rate=rate,
        )

        if not fold_rnd["valid"]:

            bad_folds += 1

            continue

        mono_total += (
            fold_rnd["mono"]
        )

        neg_total += (
            fold_rnd["neg"]
        )

        if (
            fold_rnd["mono"] != 0
            or fold_rnd["neg"] != 0
        ):

            bad_folds += 1

    errors = np.asarray(
        all_errors,
        dtype=float,
    )

    if len(errors) == 0:
        raise RuntimeError(
            "No CV errors calculated."
        )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                errors
            )
        )
    )

    max_error = float(
        np.max(
            np.abs(
                errors
            )
        )
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "bad_folds": bad_folds,
        "mono_total": mono_total,
        "neg_total": neg_total,
        "errors": errors,
    }


# ============================================================
# MAIN
# ============================================================

files = sorted(
    SNAPSHOT_DIR.glob(
        "snapshot_*.json"
    )
)

if not files:
    raise RuntimeError(
        "No temporal snapshots found."
    )


print()
print("=" * 170)
print(
    "DEALEROS RND TEMPORAL ROBUSTNESS TEST"
)
print("=" * 170)

print(
    f"Snapshots            : {len(files)}"
)

print(
    f"Fit range            : +/- {FIT_RANGE}"
)

print(
    f"Grid step            : {GRID_STEP}"
)

print(
    f"Cross-validation     : {FOLDS} folds"
)

print(
    "Methods              : "
    + ", ".join(METHODS)
)

print()


aggregate = {
    method: {
        "errors": [],
        "bad_folds": 0,
        "bad_snapshots": 0,
        "mono_total": 0,
        "neg_total": 0,
        "areas": [],
        "wins": 0,
    }
    for method in METHODS
}


print("=" * 170)
print("PER-SNAPSHOT RESULTS")
print("=" * 170)

print(
    f"{'API TIME':<20}"
    f"{'SPOT':>10}"
    f"{'T HRS':>9}"
    f"{'RATE%':>9}"
    f"{'OBS':>6}"
    f"{'METHOD':>14}"
    f"{'CV RMSE':>11}"
    f"{'CV MAE':>11}"
    f"{'BAD F':>8}"
    f"{'MONO':>7}"
    f"{'NEG':>7}"
    f"{'AREA':>11}"
    f"{'MEAN-FWD':>11}"
    f"{'MEDIAN':>12}"
    f"{'MODE':>10}"
    f"{'STD':>10}"
)

print("-" * 170)


snapshot_winners = []


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

    calibration = (
        calibrate_model_inputs(
            data
        )
    )

    T = calibration["T"]
    rate = calibration["rate"]

    T_hours = (
        T * 365.0 * 24.0
    )

    forward = (
        spot
        * math.exp(
            rate * T
        )
    )

    strikes, ivs = (
        build_otm_observations(
            data=data,
            forward=forward,
        )
    )

    grid = np.arange(
        strikes[0],
        strikes[-1]
        + GRID_STEP * 0.5,
        GRID_STEP,
        dtype=float,
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
        else path.name
    )

    candidate_winners = []

    for method in METHODS:

        cv = cross_validate(
            method=method,
            strikes=strikes,
            ivs=ivs,
            grid=grid,
            spot=spot,
            forward=forward,
            T=T,
            rate=rate,
        )

        full = build_rnd(
            method=method,
            train_strikes=strikes,
            train_ivs=ivs,
            grid=grid,
            spot=spot,
            forward=forward,
            T=T,
            rate=rate,
        )

        aggregate[
            method
        ][
            "errors"
        ].extend(
            cv["errors"].tolist()
        )

        aggregate[
            method
        ][
            "bad_folds"
        ] += cv[
            "bad_folds"
        ]

        aggregate[
            method
        ][
            "mono_total"
        ] += (
            full["mono"]
            if full["valid"]
            else 0
        )

        aggregate[
            method
        ][
            "neg_total"
        ] += (
            full["neg"]
            if full["valid"]
            else 0
        )

        if (
            not full["valid"]
            or full["mono"] != 0
            or full["neg"] != 0
        ):

            aggregate[
                method
            ][
                "bad_snapshots"
            ] += 1

        if (
            full["valid"]
            and full["area"] > 0
        ):

            aggregate[
                method
            ][
                "areas"
            ].append(
                full["area"]
            )

        # ----------------------------------------------------
        # Eligible snapshot winner:
        #
        # 0 bad CV folds
        # clean full RND
        # ----------------------------------------------------

        if (
            cv["bad_folds"] == 0
            and full["valid"]
            and full["mono"] == 0
            and full["neg"] == 0
        ):

            candidate_winners.append(
                (
                    cv["rmse"],
                    method,
                )
            )

        if (
            full["mean_fwd"]
            is None
        ):

            mean_fwd_text = "NA"
            median_text = "NA"
            mode_text = "NA"
            std_text = "NA"

        else:

            mean_fwd_text = (
                f"{full['mean_fwd']:.4f}"
            )

            median_text = (
                f"{full['median']:.4f}"
            )

            mode_text = (
                f"{full['mode']:.1f}"
            )

            std_text = (
                f"{full['std']:.4f}"
            )

        print(
            f"{api_time:<20}"
            f"{spot:>10.3f}"
            f"{T_hours:>9.4f}"
            f"{rate * 100:>9.4f}"
            f"{len(strikes):>6}"
            f"{method:>14}"
            f"{cv['rmse']:>11.6f}"
            f"{cv['mae']:>11.6f}"
            f"{cv['bad_folds']:>8}"
            f"{full['mono']:>7}"
            f"{full['neg']:>7}"
            f"{full['area']:>11.6f}"
            f"{mean_fwd_text:>11}"
            f"{median_text:>12}"
            f"{mode_text:>10}"
            f"{std_text:>10}"
        )

    if candidate_winners:

        candidate_winners.sort()

        winner = (
            candidate_winners[0][1]
        )

        aggregate[
            winner
        ][
            "wins"
        ] += 1

        snapshot_winners.append(
            (
                api_time,
                winner,
                candidate_winners[0][0],
            )
        )

    else:

        snapshot_winners.append(
            (
                api_time,
                "NO_VALID_WINNER",
                None,
            )
        )

    print("-" * 170)


# ============================================================
# AGGREGATE RESULTS
# ============================================================

print()
print("=" * 125)
print("AGGREGATE TEMPORAL RESULTS")
print("=" * 125)

print(
    f"{'METHOD':>14}"
    f"{'CV RMSE':>12}"
    f"{'CV MAE':>12}"
    f"{'MAX ERR':>12}"
    f"{'BAD FOLDS':>12}"
    f"{'BAD SNAPS':>12}"
    f"{'MONO':>10}"
    f"{'NEG':>10}"
    f"{'AVG AREA':>12}"
    f"{'WINS':>8}"
)

print("-" * 125)


for method in METHODS:

    errors = np.asarray(
        aggregate[
            method
        ][
            "errors"
        ],
        dtype=float,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                errors
            )
        )
    )

    max_error = float(
        np.max(
            np.abs(
                errors
            )
        )
    )

    areas = aggregate[
        method
    ][
        "areas"
    ]

    avg_area = (
        float(
            np.mean(
                areas
            )
        )
        if areas
        else float("nan")
    )

    print(
        f"{method:>14}"
        f"{rmse:>12.6f}"
        f"{mae:>12.6f}"
        f"{max_error:>12.6f}"
        f"{aggregate[method]['bad_folds']:>12}"
        f"{aggregate[method]['bad_snapshots']:>12}"
        f"{aggregate[method]['mono_total']:>10}"
        f"{aggregate[method]['neg_total']:>10}"
        f"{avg_area:>12.6f}"
        f"{aggregate[method]['wins']:>8}"
    )


print("-" * 125)


# ============================================================
# SNAPSHOT WINNERS
# ============================================================

print()
print("=" * 80)
print("PER-SNAPSHOT WINNER")
print("=" * 80)

for (
    api_time,
    winner,
    rmse,
) in snapshot_winners:

    if rmse is None:

        print(
            f"{api_time}  "
            f"{winner}"
        )

    else:

        print(
            f"{api_time}  "
            f"{winner:<14} "
            f"CV_RMSE={rmse:.6f}"
        )


print()
print("=" * 80)
print("DONE")
print("=" * 80)
print()