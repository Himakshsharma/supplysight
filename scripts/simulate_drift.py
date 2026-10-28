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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supplysight.config import get_settings  # noqa: E402
from supplysight.datagen.generator import GenConfig, generate, write_csv  # noqa: E402
from supplysight.utils import get_logger  # noqa: E402

log = get_logger("simulate_drift")
TRANSACTIONAL = ["lfa1", "eina", "eine", "eban", "ekko", "ekpo", "mkpf", "mseg", "afko", "afpo", "mard", "plaf",
                 "anomaly_labels"]


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
    drift = {} if args.no_drift else {
        "price_shock": args.price_shock,
        "demand_surge": {"FG-LIP": 1.5, "FG-CREAM": 1.3, "FG-FRAG": 1.45, "FG-FACE": 1.25},
        "new_vendor_share": args.new_vendor_share,
    }
    cfg = GenConfig(seed=args.seed, start=args.start, end=args.end, trend_anchor="2024-10-01",
                    id_offset=100_000, vendor_offset=100, drift=drift)
    tables = generate(cfg)
    out = get_settings().data_dir / "drift"
    shutil.rmtree(out, ignore_errors=True)
    write_csv({k: v for k, v in tables.items() if k in TRANSACTIONAL}, out)
    log.info("Wrote new period %s..%s to %s (%d POs, drift=%s)", args.start, args.end, out, len(tables["ekpo"]),
             bool(drift))


if __name__ == "__main__":
    main()
