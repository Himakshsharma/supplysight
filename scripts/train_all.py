"""Train, evaluate and register models; publish serving tables.

Usage:
    python scripts/train_all.py                    # anomaly + forecast + copilot index
    python scripts/train_all.py --module anomaly   # one module
    python scripts/train_all.py --quick            # 2 backtest origins (CI)
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")

from supplysight.pipeline import train_all


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", choices=["anomaly", "forecast", "copilot", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--promote",
        choices=["if_better", "always", "never"],
        default="always",
        help="initial training promotes by default; retrain.py uses champion/challenger",
    )
    args = ap.parse_args()
    mods = ("anomaly", "forecast", "copilot") if args.module == "all" else (args.module,)
    out = train_all(mods, quick=args.quick, promote=args.promote)
    summary = {k: v.get("metrics", v) for k, v in out.items()}
    print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
