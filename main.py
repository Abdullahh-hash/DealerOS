from datetime import datetime
import json

from app.api.client import FreeFlowClient
from app.config.settings import settings

from app.engines.dealer_engine import DealerEngine
from app.engines.flow_engine import FlowEngine
from app.engines.rnd_engine import RNDEngine

from app.services.snapshot_parser import parse_snapshot
from app.services.flow_parser import parse_flow
from app.services.rnd_surface_builder import build_rnd_surface


def print_banner():
    print("=" * 60)
    print("DealerOS v0.4")
    print("Dealer Engine")
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
        print("API Key  : Loaded")
    else:
        print("API Key  : Missing")
        return

    print()
    print("Connecting to FreeFlow...")
    print("-" * 60)

    try:
        client = FreeFlowClient()

        expirations = client.get_expirations(
            settings.default_symbol
        )

        selected_expiry = expirations["expirations"][0]

        print("Connection      : Success")
        print(f"Selected Expiry : {selected_expiry}")

        # ====================================================
        # SNAPSHOT
        # ====================================================

        print()
        print("Downloading Snapshot...")
        print("-" * 60)

        raw_snapshot = client.get_snapshot(
            settings.default_symbol,
            selected_expiry,
        )

        snapshot = parse_snapshot(
            raw_snapshot
        )

        # ====================================================
        # DEALER ENGINE
        # ====================================================

        dealer_engine = DealerEngine(
            snapshot
        )

        summary = dealer_engine.summary()

        # ====================================================
        # RND SURFACE
        # ====================================================

        rnd_surface = build_rnd_surface(
            snapshot
        )

        # ====================================================
        # RND ENGINE
        # ====================================================

        rnd_engine = RNDEngine(
            spot=rnd_surface.spot,
            time_to_expiry=rnd_surface.time_to_expiry,
            risk_free_rate=rnd_surface.risk_free_rate,
        )

        rnd_result = rnd_engine.build_from_surface(
            rnd_surface
        )

        # ====================================================
        # FLOW
        # ====================================================

        flow = client.get_flow(
            settings.default_symbol,
            selected_expiry,
        )

        flow_snapshot = parse_flow(
            flow
        )

        flow_engine = FlowEngine(
            flow_snapshot
        )

        flow_summary = flow_engine.summary()

        with open(
            "flow.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                flow,
                f,
                indent=2,
            )

        print("Flow saved to flow.json")

        # ====================================================
        # SNAPSHOT SUMMARY
        # ====================================================

        print()
        print("Parsed Snapshot")
        print("-" * 60)

        print(
            f"Symbol      : {snapshot.symbol}"
        )

        print(
            f"Spot        : {snapshot.spot}"
        )

        print(
            f"Expiry      : {snapshot.exp}"
        )

        print(
            f"Total GEX   : {snapshot.total_gex}"
        )

        print(
            f"Contracts   : {len(snapshot.contracts)}"
        )

        # ====================================================
        # DEALER SUMMARY
        # ====================================================

        print()
        print("=" * 60)
        print("MARKET SUMMARY")
        print("=" * 60)

        print(
            f"Dealer Bias       : "
            f"{summary.dealer_bias}"
        )

        print(
            f"Gamma Flip        : "
            f"{summary.gamma_flip}"
        )

        print(
            f"Largest Call GEX  : "
            f"{summary.largest_call_gex}"
        )

        print(
            f"Largest Put GEX   : "
            f"{summary.largest_put_gex}"
        )

        if summary.support:
            print(
                f"Support           : "
                f"{summary.support.strike}"
            )

        if summary.resistance:
            print(
                f"Resistance        : "
                f"{summary.resistance.strike}"
            )

        # ====================================================
        # FLOW SUMMARY
        # ====================================================

        print()
        print("=" * 60)
        print("FLOW SUMMARY")
        print("=" * 60)

        print(
            f"Flow Bias          : "
            f"{flow_summary['flow_bias']}"
        )

        print(
            f"Net Premium        : "
            f"{flow_summary['net_premium']:,.2f}"
        )

        print(
            f"Net Directional    : "
            f"{flow_summary['net_directional']:,.2f}"
        )

        print(
            f"Net Delta Notional : "
            f"{flow_summary['net_delta_notional']:,.2f}"
        )

        print(
            f"Put/Call Ratio     : "
            f"{flow_summary['put_call_premium_ratio']}"
        )

        print(
            f"Buy Count          : "
            f"{flow_summary['buy_count']}"
        )

        print(
            f"Sell Count         : "
            f"{flow_summary['sell_count']}"
        )

        print(
            f"Buy/Sell Ratio     : "
            f"{flow_summary['buy_sell_ratio']:.2f}"
        )

        print()
        print("Largest Positive Flow")
        print("-" * 60)

        if (
            flow_summary[
                "largest_positive_strike"
            ]
            is not None
        ):
            print(
                f"Strike : "
                f"{flow_summary['largest_positive_strike']}"
            )

            print(
                f"Net    : "
                f"{flow_summary['largest_positive_flow']:,.2f}"
            )

        else:
            print(
                "No positive flow within +/-500 points"
            )

        print()
        print("Largest Negative Flow")
        print("-" * 60)

        if (
            flow_summary[
                "largest_negative_strike"
            ]
            is not None
        ):
            print(
                f"Strike : "
                f"{flow_summary['largest_negative_strike']}"
            )

            print(
                f"Net    : "
                f"{flow_summary['largest_negative_flow']:,.2f}"
            )

        else:
            print(
                "No negative flow within +/-500 points"
            )

        # ====================================================
        # RND SUMMARY
        # ====================================================

        print()
        print("=" * 60)
        print("RISK-NEUTRAL DENSITY")
        print("=" * 60)

        print(
            f"Symbol            : "
            f"{rnd_result.symbol}"
        )

        print(
            f"Expiry            : "
            f"{rnd_result.exp}"
        )

        print(
            f"Spot              : "
            f"{rnd_result.spot:.3f}"
        )

        print(
            f"Forward           : "
            f"{rnd_result.forward:.3f}"
        )

        print(
            f"ATM IV            : "
            f"{rnd_result.atm_iv_pct:.2f}%"
        )

        print(
            f"Model Time        : "
            f"{rnd_result.model_hours:.4f} hours"
        )

        print(
            f"Model Rate        : "
            f"{rnd_result.risk_free_rate * 100:.4f}%"
        )

        print()
        print("Surface")
        print("-" * 60)

        print(
            f"Sigma Multiplier  : "
            f"{rnd_result.sigma_multiplier:.2f}"
        )

        print(
            f"Adaptive Range    : "
            f"+/-{rnd_result.fit_range:.2f}"
        )

        print(
            f"Source Contracts  : "
            f"{rnd_result.source_count}"
        )

        print(
            f"Surface Points    : "
            f"{rnd_result.surface_points}"
        )

        print(
            f"Grid Step         : "
            f"{rnd_result.grid_step:.2f}"
        )

        print(
            f"Spline Lambda     : "
            f"{rnd_result.smoothing_lambda}"
        )

        print()
        print("Quality")
        print("-" * 60)

        print(
            f"Raw Coverage      : "
            f"{rnd_result.coverage_pct:.4f}%"
        )

        print(
            f"Raw Area          : "
            f"{rnd_result.raw_area:.6f}"
        )

        print(
            f"Normalized Area   : "
            f"{rnd_result.normalized_area:.6f}"
        )

        print(
            f"Monotonic Fails   : "
            f"{rnd_result.monotonicity_failures}"
        )

        print(
            f"Negative Density  : "
            f"{rnd_result.negative_density_points}"
        )

        print()
        print("Distribution")
        print("-" * 60)

        print(
            f"Mean              : "
            f"{rnd_result.mean:.2f}"
        )

        print(
            f"Mean - Forward    : "
            f"{rnd_result.mean_minus_forward:+.2f}"
        )

        print(
            f"Median            : "
            f"{rnd_result.median:.2f}"
        )

        print(
            f"Mode              : "
            f"{rnd_result.mode:.2f}"
        )

        print(
            f"Std Dev           : "
            f"{rnd_result.std:.2f}"
        )

        print()
        print("Risk-Neutral Ranges")
        print("-" * 60)

        lower_50, upper_50 = (
            rnd_result.range_50
        )

        lower_90, upper_90 = (
            rnd_result.range_90
        )

        print(
            f"50% Range         : "
            f"{lower_50:.2f} -> "
            f"{upper_50:.2f}"
        )

        print(
            f"90% Range         : "
            f"{lower_90:.2f} -> "
            f"{upper_90:.2f}"
        )

        print()
        print("Risk-Neutral Probabilities")
        print("-" * 60)

        print(
            f"P(Price > Spot)   : "
            f"{rnd_result.probability_above_spot * 100:.2f}%"
        )

        print(
            f"P(Price > Forward): "
            f"{rnd_result.probability_above_forward * 100:.2f}%"
        )

    except Exception as e:
        print()
        print("DealerOS run failed")
        print(f"Reason : {e}")


if __name__ == "__main__":
    main()