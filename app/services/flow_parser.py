from app.models.flow_snapshot import (
    FlowCounts,
    FlowSnapshot,
    FlowStrike,
    PremiumSide,
)


def parse_flow(data: dict) -> FlowSnapshot:
    """
    Convert FreeFlow flow JSON into a FlowSnapshot object.
    """

    strikes = []

    for item in data.get("by_strike", []):
        strikes.append(
            FlowStrike(
                strike=item["strike"],
                call_net=item.get("call_net", 0.0),
                put_net=item.get("put_net", 0.0),
                net=item.get("net", 0.0),
            )
        )

    call_data = data.get("call_premium", {})
    put_data = data.get("put_premium", {})
    count_data = data.get("counts", {})

    return FlowSnapshot(
        symbol=data["symbol"],
        expiry=data["exp"],
        spot=data["spot"],
        dte=data.get("dte", 0),
        window_min=data.get("window_min", 0),

        net_premium=data.get("net_premium", 0.0),
        bull_premium=data.get("bull_premium", 0.0),
        bear_premium=data.get("bear_premium", 0.0),
        net_directional=data.get("net_directional", 0.0),
        net_delta_notional=data.get("net_delta_notional", 0.0),

        call_premium=PremiumSide(
            buy=call_data.get("buy", 0.0),
            sell=call_data.get("sell", 0.0),
        ),

        put_premium=PremiumSide(
            buy=put_data.get("buy", 0.0),
            sell=put_data.get("sell", 0.0),
        ),

        put_call_prem_ratio=data.get("put_call_prem_ratio", 0.0),

        counts=FlowCounts(
            classified=count_data.get("classified", 0),
            buy=count_data.get("buy", 0),
            sell=count_data.get("sell", 0),
        ),

        by_strike=strikes,

        timestamp=data.get("timestamp"),
    )