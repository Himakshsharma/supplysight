"""Generate a 'new quarter' (Oct–Dec 2026) with shifted behaviour into data/drift/.

Shifts: +18 % price inflation shock, festive demand surge for lipsticks / creams /
fragrances, and 15 % of POs moved to a set of brand-new vendors.

Usage:
    python scripts/simulate_drift.py               # default drift scenario
    python scripts/simulate_drift.py --no-drift    # same period without shifts (control)
"""

from __future__ import annotations

import argparse
import shutil

from supplysight.config import get_settings
from supplysight.datagen.generator import DRIFT_TABLES, drift_quarter_config, generate, write_csv
from supplysight.utils import get_logger

log = get_logger("simulate_drift")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=get_settings().seed)
    ap.add_argument("--start", default="2026-10-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--price-shock", type=float, default=1.18)
    ap.add_argument("--new-vendor-share", type=float, default=0.15)
    args = ap.parse_args()
    cfg = drift_quarter_config(
        args.seed, args.start, args.end, args.price_shock, args.new_vendor_share, drift=not args.no_drift
    )
    tables = generate(cfg)
    out = get_settings().data_dir / "drift"
    shutil.rmtree(out, ignore_errors=True)
    write_csv({k: v for k, v in tables.items() if k in DRIFT_TABLES}, out)
    log.info(
        "Wrote new period %s..%s to %s (%d POs, drift=%s)",
        args.start,
        args.end,
        out,
        len(tables["ekpo"]),
        bool(cfg.drift),
    )


if __name__ == "__main__":
    main()
