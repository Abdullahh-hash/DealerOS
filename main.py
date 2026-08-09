from datetime import datetime
import json

from app.api.client import FreeFlowClient
from app.config.settings import settings
from app.engines.dealer_engine import DealerEngine
from app.services.snapshot_parser import parse_snapshot
from app.engines.flow_engine import FlowEngine
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
        print("API Key  : Loaded ✅")
    else:
        print("API Key  : Missing ❌")
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

        engine = DealerEngine(snapshot)

        summary = engine.summary()

        rnd_surface = build_rnd_surface(snapshot)

        flow = client.get_flow(
            settings.default_symbol,
            selected_expiry,
        )
        flow_snapshot = parse_flow(flow)

        flow_engine = FlowEngine(flow_snapshot)

        flow_summary = flow_engine.summary()

        with open("flow.json", "w") as f:
            json.dump(flow, f, indent=2)

        print("Flow saved to flow.json ✅")

        print()
        print("Parsed Snapshot")
        print("-" * 60)
        print(f"Symbol      : {snapshot.symbol}")
        print(f"Spot        : {snapshot.spot}")
        print(f"Expiry      : {snapshot.exp}")
        print(f"Total GEX   : {snapshot.total_gex}")
        print(f"Contracts   : {len(snapshot.contracts)}")

        print()
        print("=" * 60)
        print("MARKET SUMMARY")
        print("=" * 60)

        print(f"Dealer Bias       : {summary.dealer_bias}")
        print(f"Gamma Flip        : {summary.gamma_flip}")
        print(f"Largest Call GEX  : {summary.largest_call_gex}")
        print(f"Largest Put GEX   : {summary.largest_put_gex}")

        if summary.support:
            print(f"Support           : {summary.support.strike}")

        if summary.resistance:
            print(f"Resistance        : {summary.resistance.strike}")

        print()
        print("=" * 60)
        print("FLOW SUMMARY")
        print("=" * 60)

        print(f"Flow Bias          : {flow_summary['flow_bias']}")
        print(f"Net Premium        : {flow_summary['net_premium']:,.2f}")
        print(f"Net Directional    : {flow_summary['net_directional']:,.2f}")
        print(f"Net Delta Notional : {flow_summary['net_delta_notional']:,.2f}")
        print(f"Put/Call Ratio     : {flow_summary['put_call_premium_ratio']}")
        print(f"Buy Count          : {flow_summary['buy_count']}")
        print(f"Sell Count         : {flow_summary['sell_count']}")
        print(f"Buy/Sell Ratio     : {flow_summary['buy_sell_ratio']:.2f}")

        print()
        print("Largest Positive Flow")
        print("-" * 60)

        if flow_summary["largest_positive_strike"] is not None:
            print(f"Strike : {flow_summary['largest_positive_strike']}")
            print(f"Net    : {flow_summary['largest_positive_flow']:,.2f}")
        else:
            print("No positive flow within ±500 points")

        print()
        print("Largest Negative Flow")
        print("-" * 60)

        if flow_summary["largest_negative_strike"] is not None:
            print(f"Strike : {flow_summary['largest_negative_strike']}")
            print(f"Net    : {flow_summary['largest_negative_flow']:,.2f}")
        else:
            print("No negative flow within ±500 points")

        print()
        print("=" * 60)
        print("RND SURFACE TEST")
        print("=" * 60)

        print(f"Surface points : {len(rnd_surface)}")

        RND_RANGE = 500

        lower = snapshot.spot - RND_RANGE
        upper = snapshot.spot + RND_RANGE

        nearby = [
            point
            for point in rnd_surface
            if lower <= point.strike <= upper
        ]

        nearby.sort(key=lambda x: (x.strike, x.right))

        print()
        print(f"Strike range : {lower:.1f} → {upper:.1f}")
        print(f"Points       : {len(nearby)}")
        print()

        for point in nearby:
            print(
                f"Strike : {point.strike:8.1f} | "
                f"IV : {point.iv:.4f} | "
                f"Right : {point.right}"
            )

    except Exception as e:
        print()
        print("Connection : Failed ❌")
        print(f"Reason     : {e}")


if __name__ == "__main__":
    main()
