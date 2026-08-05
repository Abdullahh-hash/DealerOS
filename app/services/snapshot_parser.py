from app.models.dealer_snapshot import DealerSnapshot
from app.models.option_contract import OptionContract


def parse_snapshot(data: dict) -> DealerSnapshot:
    """
    Convert a FreeFlow snapshot JSON dictionary into
    a DealerSnapshot object.
    """

    contracts = []

    for contract in data.get("rows", []):
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
        exp=data["exp"],
        timestamp=data["timestamp"],

        spot=data["spot"],
        dte=data["dte"],

        total_gex=data.get("total_gex"),
        total_dex=data.get("total_dex"),
        total_ag=data.get("total_ag"),
        total_dag=data.get("total_dag"),

        net_premium=data.get("net_premium"),

        gross_dex=data.get("gross_dex"),
        gross_vex=data.get("gross_vex"),
        gross_charmex=data.get("gross_charmex"),

        total_vol=data.get("total_vol"),

        max_pain=data.get("max_pain"),
        vol_trigger=data.get("vol_trigger"),

        atm_iv=data.get("atm_iv"),
        model=data.get("model"),

        contracts=contracts,
    )

    return snapshot