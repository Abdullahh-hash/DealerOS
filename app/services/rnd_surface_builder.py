from app.models.dealer_snapshot import DealerSnapshot
from app.models.rnd_surface import RNDSurfacePoint


def build_rnd_surface(snapshot: DealerSnapshot) -> list[RNDSurfacePoint]:
    surface = []

    for contract in snapshot.contracts:
        if contract.iv_pct is None:
            continue

        if contract.iv_pct <= 0:
            continue

        surface.append(
            RNDSurfacePoint(
                strike=contract.strike,
                iv=contract.iv_pct / 100.0,
                right=contract.right,
            )
        )

    return surface