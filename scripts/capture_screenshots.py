"""Capture README screenshots of every dashboard page with Playwright.

1. Start the dashboard:  python run.py dashboard      (http://localhost:8501)
2. pip install playwright && python -m playwright install chromium
3. python scripts/capture_screenshots.py              (writes docs/images/screenshot_*.png)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
PAGES = {"overview": "", "anomaly_review": "Anomaly_Review", "vendor_duplicates": "Vendor_Duplicates",
         "demand_reorder": "Demand_and_Reorder", "copilot": "Copilot", "model_health": "Model_Health"}


def main() -> None:
    """Sign in as manager and screenshot each page."""
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(URL)
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_timeout(4000)
        for name, path in PAGES.items():
            page.goto(f"{URL}/{path}")
            page.wait_for_load_state("networkidle")
            time.sleep(4)
            if name == "copilot":
                box = page.get_by_placeholder("Ask about vendors, POs, prices, stock, SOPs…")
                box.fill("Which vendor supplies glycerin and what is the lead time?")
                box.press("Enter")
                time.sleep(5)
            page.screenshot(path=str(OUT / f"screenshot_{name}.png"), full_page=True)
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
