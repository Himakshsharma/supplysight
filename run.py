"""Cross-platform task runner (works on Windows without `make`).

python run.py setup       # pip install -e ".[serve,dev]"
python run.py all         # data -> train -> drift report -> evaluation
python run.py api         # FastAPI on http://localhost:8000/docs
python run.py dashboard   # Streamlit on http://localhost:8501
python run.py test        # pytest with coverage
python run.py drift       # simulate a new quarter + drift report
python run.py retrain     # champion/challenger retraining on the new quarter
python run.py mlflow      # local MLflow UI on http://localhost:5000
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
TASKS: dict[str, list[list[str]]] = {
    "setup": [[PY, "-m", "pip", "install", "-e", ".[serve,dev]"]],
    "data": [[PY, "scripts/generate_data.py", "--seed", "42"]],
    "train": [[PY, "scripts/train_all.py"]],
    "drift": [[PY, "scripts/simulate_drift.py"], [PY, "scripts/retrain.py", "--check"]],
    "retrain": [[PY, "scripts/retrain.py"]],
    "eval": [[PY, "scripts/evaluate_all.py"], [PY, "scripts/make_eda_charts.py"]],
    "test": [[PY, "-m", "pytest", "--cov=src/supplysight", "--cov-report=term-missing"]],
    "api": [[PY, "-m", "uvicorn", "supplysight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]],
    "dashboard": [[PY, "-m", "streamlit", "run", "dashboard/app.py"]],
    "mlflow": [[PY, "-m", "mlflow", "ui", "--port", "5000"]],
    "lint": [[PY, "-m", "ruff", "check", "."], [PY, "-m", "ruff", "format", "--check", "."]],
}
TASKS["all"] = TASKS["data"] + TASKS["train"] + TASKS["drift"] + TASKS["eval"]


def main() -> None:
    """Run the named tasks in order."""
    names = sys.argv[1:] or ["all"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for name in names:
        if name not in TASKS:
            raise SystemExit(f"unknown task {name!r}; choose from: {', '.join(TASKS)}")
        for cmd in TASKS[name]:
            print("$", " ".join(cmd[1:] if cmd[0] == PY else cmd), flush=True)
            if subprocess.call(cmd, cwd=ROOT, env=env) != 0:
                raise SystemExit(f"task {name} failed")


if __name__ == "__main__":
    main()
