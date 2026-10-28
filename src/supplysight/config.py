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


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    seed: int = int(_env("SEED", "42"))
    data_dir: Path = Path(_env("SUPPLYSIGHT_DATA_DIR", str(PROJECT_ROOT / "data")))
    artifacts_dir: Path = Path(_env("SUPPLYSIGHT_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
    docs_dir: Path = PROJECT_ROOT / "docs"
    database_url: str = _env(
        "DATABASE_URL", f"sqlite:///{(PROJECT_ROOT / 'data' / 'supplysight.db').as_posix()}"
    )
    jwt_secret: str = _env("JWT_SECRET", "change-me-in-production-please-32b")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = int(_env("JWT_EXPIRY_MINUTES", "480"))
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    mlflow_tracking_uri: str = _env("MLFLOW_TRACKING_URI", "")
    api_url: str = _env("SUPPLYSIGHT_API_URL", "")
    approval_threshold_inr: float = float(_env("APPROVAL_THRESHOLD_INR", "500000"))
    sql_row_limit: int = int(_env("SQL_ROW_LIMIT", "200"))
    sql_timeout_s: float = float(_env("SQL_TIMEOUT_S", "5"))
    nn_backend: str = _env("SUPPLYSIGHT_NN_BACKEND", "auto")  # auto | torch | sklearn
    gbm_backend: str = _env("SUPPLYSIGHT_GBM_BACKEND", "auto")  # auto | lightgbm | sklearn
    embedding_backend: str = _env("SUPPLYSIGHT_EMBEDDINGS", "auto")  # auto | st | lsa
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
