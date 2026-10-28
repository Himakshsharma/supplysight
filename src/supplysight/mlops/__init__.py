"""Experiment tracking and model registry.

* :class:`Registry` is a small file-based registry (``artifacts/registry``) with versions and
  stages (``None`` / ``Staging`` / ``Production`` / ``Archived``). The API and dashboard load
  models from here by **registered name + stage**, so serving never depends on a running server.
* :func:`track_run` logs params/metrics/artifacts to a local JSONL run log and, when the
  ``mlflow`` package is installed, mirrors the run (and the registered model) to MLflow
  (``MLFLOW_TRACKING_URI``, e.g. the docker-compose MLflow server at http://mlflow:5000).
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from supplysight.config import PROJECT_ROOT, get_settings
from supplysight.utils import get_logger, optional_import

log = get_logger(__name__)
STAGES = ("None", "Staging", "Production", "Archived")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Registry:
    """File-based model registry with MLflow-like semantics."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or get_settings().artifacts_dir / "registry")
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"

    # ------------------------------------------------------------- index helpers
    def _index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {}

    def _save(self, idx: dict) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, indent=2, default=str))
        tmp.replace(self.index_path)

    # ------------------------------------------------------------- public API
    def register(
        self,
        name: str,
        model: Any,
        metrics: dict | None = None,
        params: dict | None = None,
        run_id: str | None = None,
        stage: str = "None",
    ) -> int:
        """Persist ``model`` as a new version of ``name`` and return the version number."""
        idx = self._index()
        entry = idx.setdefault(name, {"versions": {}})
        version = max([int(v) for v in entry["versions"]] or [0]) + 1
        vdir = self.root / name / f"v{version}"
        vdir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "version"):
            model.version = str(version)
        joblib.dump(model, vdir / "model.joblib")
        entry["versions"][str(version)] = {
            "version": version,
            "stage": "None",
            "created_at": _now(),
            "metrics": metrics or {},
            "params": params or {},
            "run_id": run_id,
        }
        self._save(idx)
        if stage != "None":
            self.transition(name, version, stage)
        log.info("Registered %s v%d", name, version)
        return version

    def transition(self, name: str, version: int, stage: str) -> None:
        """Move a version to a stage; promoting to Production archives the previous one."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}")
        idx = self._index()
        versions = idx[name]["versions"]
        if stage == "Production":
            for v in versions.values():
                if v["stage"] == "Production":
                    v["stage"] = "Archived"
        versions[str(version)]["stage"] = stage
        versions[str(version)]["stage_updated_at"] = _now()
        self._save(idx)

    def get_version(self, name: str, stage: str = "Production") -> dict | None:
        """Metadata of the latest version in ``stage`` (or None)."""
        idx = self._index().get(name)
        if not idx:
            return None
        cands = [v for v in idx["versions"].values() if v["stage"] == stage]
        return max(cands, key=lambda v: v["version"]) if cands else None

    def load(self, name: str, stage: str = "Production", version: int | None = None) -> tuple[Any, dict]:
        """Load a model by name + stage (or explicit version)."""
        meta = self._index()[name]["versions"][str(version)] if version is not None else self.get_version(name, stage)
        if meta is None:
            raise LookupError(f"no {stage} version of {name}")
        return joblib.load(self.root / name / f"v{meta['version']}" / "model.joblib"), meta

    def list(self) -> list[dict]:
        """Flat list of all versions of all models."""
        out = []
        for name, e in self._index().items():
            for v in e["versions"].values():
                out.append(
                    {
                        "name": name,
                        **{k: v[k] for k in ("version", "stage", "created_at")},
                        "metrics": v.get("metrics", {}),
                    }
                )
        return sorted(out, key=lambda r: (r["name"], r["version"]))

    def clear(self) -> None:
        """Delete the whole registry (tests / full rebuilds)."""
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)


@contextmanager
def track_run(experiment: str, run_name: str) -> Iterator[dict]:
    """Track one training run. Yields a dict; put ``params``, ``metrics``, ``artifacts``
    (list of file paths) and optionally ``model`` + ``model_name`` into it."""
    s = get_settings()
    rec: dict = {
        "run_id": uuid.uuid4().hex[:12],
        "experiment": experiment,
        "run_name": run_name,
        "started_at": _now(),
        "params": {},
        "metrics": {},
        "artifacts": [],
    }
    yield rec
    rec["ended_at"] = _now()
    rec["artifacts"] = [_relative(a) for a in rec["artifacts"]]
    runs = s.artifacts_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with open(runs / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: v for k, v in rec.items() if k != "model"}, default=str) + "\n")
    _mirror_to_mlflow(rec)


def _relative(path: str) -> str:
    """Store artifact paths relative to the project root so run logs are portable."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _mirror_to_mlflow(rec: dict) -> None:
    mlflow = optional_import("mlflow")
    if mlflow is None:
        return
    try:  # pragma: no cover - requires mlflow
        s = get_settings()
        if s.mlflow_tracking_uri:
            mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        mlflow.set_experiment(rec["experiment"])
        with mlflow.start_run(run_name=rec["run_name"]):
            mlflow.log_params({k: str(v)[:250] for k, v in rec["params"].items()})
            mlflow.log_metrics({k: float(v) for k, v in rec["metrics"].items() if v == v})
            for a in rec["artifacts"]:
                full = PROJECT_ROOT / a
                if full.exists():
                    mlflow.log_artifact(str(full))
            if rec.get("model") is not None and rec.get("model_name"):
                mlflow.sklearn.log_model(rec["model"], artifact_path="model", registered_model_name=rec["model_name"])
    except Exception as exc:  # pragma: no cover
        log.warning("MLflow mirror skipped: %s", exc)


def read_runs(experiment: str | None = None) -> list[dict]:
    """Read the local run log (newest last)."""
    p = get_settings().artifacts_dir / "runs" / "runs.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if experiment is None or r["experiment"] == experiment]
