from dataclasses import dataclass


@dataclass
class RNDSurfacePoint:
    strike: float
    iv: float
    right: str