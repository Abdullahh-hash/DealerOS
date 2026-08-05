from app.models.dealer_snapshot import DealerSnapshot
from app.models.option_contract import OptionContract


def parse_snapshot(data: dict) -> DealerSnapshot:
    """
    Convert a FreeFlow snapshot JSON dictionary into
    a DealerSnapshot object.
    """

    contracts = []

    for contract in data.get("contracts", []):
        contracts.append(
            OptionContract(
                strike=contract["strike"],
                right=contract["right"],
                oi=contract.get("oi"),
                delta=contract.get("delta"),
                gamma=contract.get("gamma"),
                vega=contract.get("vega"),
                vanna=contract.get("vanna"),
                charm=contract.get("charm"),
                gex=contract.get("gex"),
                dex=contract.get("dex"),
                vex=contract.get("vex"),
                ag=contract.get("ag"),
                dag=contract.get("dag"),
                vegaex=contract.get("vegaex"),
                charmex=contract.get("charmex"),
                iv_pct=contract.get("iv_pct"),
            )
        )

    snapshot = DealerSnapshot(
        symbol=data["symbol"],
        timestamp=data["timestamp"],
        spot=data["spot"],
        total_gex=data.get("total_gex"),
        total_dex=data.get("total_dex"),
        total_ag=data.get("total_ag"),
        total_dag=data.get("total_dag"),
        total_vol=data.get("total_vol"),
        vol_trigger=data.get("vol_trigger"),
        contracts=contracts,
    )

    return snapshot