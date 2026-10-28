"""Central configuration for SupplySight.

All settings come from environment variables (optionally loaded from a `.env` file) so the same
code runs locally, in Docker and in CI. Nothing here contains real company data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

PROJECT_ROOT = Path(os.environ.get("SUPPLYSIGHT_HOME", Path(__file__).resolve().parents[2]))


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _from_env(name: str, default: str, cast=str):
    """Field whose value is read from the environment when Settings() is created."""
    return field(default_factory=lambda: cast(_env(name, default)))


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment (read at instantiation time)."""

    seed: int = _from_env("SEED", "42", int)
    data_dir: Path = _from_env("SUPPLYSIGHT_DATA_DIR", str(PROJECT_ROOT / "data"), Path)
    artifacts_dir: Path = _from_env("SUPPLYSIGHT_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"), Path)
    docs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "docs")
    database_url: str = _from_env("DATABASE_URL", f"sqlite:///{(PROJECT_ROOT / 'data' / 'supplysight.db').as_posix()}")
    jwt_secret: str = _from_env("JWT_SECRET", "change-me-in-production-please-32b")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = _from_env("JWT_EXPIRY_MINUTES", "480", int)
    gemini_api_key: str = _from_env("GEMINI_API_KEY", "")
    gemini_model: str = _from_env("GEMINI_MODEL", "gemini-2.5-flash")
    mlflow_tracking_uri: str = _from_env("MLFLOW_TRACKING_URI", "")
    api_url: str = _from_env("SUPPLYSIGHT_API_URL", "")
    approval_threshold_inr: float = _from_env("APPROVAL_THRESHOLD_INR", "500000", float)
    sql_row_limit: int = _from_env("SQL_ROW_LIMIT", "200", int)
    sql_timeout_s: float = _from_env("SQL_TIMEOUT_S", "5", float)
    nn_backend: str = _from_env("SUPPLYSIGHT_NN_BACKEND", "auto")  # auto | torch | sklearn
    gbm_backend: str = _from_env("SUPPLYSIGHT_GBM_BACKEND", "auto")  # auto | lightgbm | sklearn
    embedding_backend: str = _from_env("SUPPLYSIGHT_EMBEDDINGS", "auto")  # auto | st | lsa
    extra: dict = field(default_factory=dict)

    @property
    def sops_dir(self) -> Path:
        return self.docs_dir / "sops"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def ensure_dirs(self) -> None:
        """Create the data/artifact directories if they do not exist."""
        for p in (self.data_dir, self.artifacts_dir, self.models_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings object."""
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests that change env vars)."""
    get_settings.cache_clear()
