"""Check drift on the newest period and retrain/promote models if needed.

Usage:
    python scripts/retrain.py            # retrain only when drift crosses thresholds
    python scripts/retrain.py --force    # always train a challenger
    python scripts/retrain.py --check    # drift report only
"""

from __future__ import annotations

import argparse
import json

from supplysight.data import load_source_tables, read_csv_tables
from supplysight.monitoring.retrain import drift_check, drift_dir, run_retrain


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="only compute drift reports")
    args = ap.parse_args()
    if args.check:
        out = drift_check(load_source_tables(), read_csv_tables(drift_dir()))
    else:
        out = run_retrain(force=args.force)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
