"""Application settings, loaded from environment / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = (
        "postgresql+psycopg://tejassaiprashad@localhost:5432/f1_intelligence"
    )
    test_database_url: str = (
        "postgresql+psycopg://tejassaiprashad@localhost:5432/f1_intelligence_test"
    )

    # FastF1 caches raw HTTP responses here. Without a cache, every re-run
    # re-downloads session data, which is slow and rude to the upstream API.
    fastf1_cache_dir: str = "data/fastf1_cache"

    log_level: str = "INFO"

    # Echo SQL to stdout. Useful when learning what the ORM actually emits.
    sql_echo: bool = False

    @property
    def fastf1_cache_path(self) -> Path:
        p = PROJECT_ROOT / self.fastf1_cache_dir
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is read once per process."""
    return Settings()
