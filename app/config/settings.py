from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    request_timeout: int
    default_symbol: str
    default_expiry: str


settings = Settings(
    api_key=os.getenv("FREEFLOW_API_KEY", ""),
    base_url=os.getenv("BASE_URL", ""),
    request_timeout=int(os.getenv("REQUEST_TIMEOUT", "10")),
    default_symbol=os.getenv("DEFAULT_SYMBOL", "NDX"),
    default_expiry=os.getenv("DEFAULT_EXPIRY", "0"),
)