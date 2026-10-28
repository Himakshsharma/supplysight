"""Generate synthetic SAP MM/PP data, write CSVs and load the database.

Usage:
    python scripts/generate_data.py --seed 42
    python scripts/generate_data.py --seed 7 --small        # tiny dataset for CI
"""

from __future__ import annotations

import argparse

from supplysight.config import get_settings
from supplysight.data import validate
from supplysight.datagen.generator import GenConfig, generate, write_csv
from supplysight.db import load_tables
from supplysight.utils import get_logger

log = get_logger("generate_data")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=get_settings().seed)
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--small", action="store_true", help="small catalogue + 1 year (CI)")
    ap.add_argument("--no-db", action="store_true", help="only write CSVs")
    args = ap.parse_args()

    cfg = GenConfig(seed=args.seed, start=args.start, end=args.end)
    if args.small:
        cfg.material_scale, cfg.n_vendors, cfg.start = 0.3, 25, "2025-04-01"
    s = get_settings()
    tables = generate(cfg)
    write_csv(tables, s.data_dir / "raw")
    problems = validate(tables)
    if problems:
        raise SystemExit("Integrity check failed: " + "; ".join(problems))
    if not args.no_db:
        load_tables(tables)
    log.info("Generated: %s", ", ".join(f"{k}={len(v)}" for k, v in tables.items()))
    log.info("Anomaly rate: %.2f%%", 100 * len(tables["anomaly_labels"]) / len(tables["ekpo"]))


if __name__ == "__main__":
    main()
