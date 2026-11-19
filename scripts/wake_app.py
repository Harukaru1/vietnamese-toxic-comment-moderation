"""Open the Streamlit demo in a headless browser and wake it up if it is asleep.

Streamlit Community Cloud puts an app to sleep after a period without visitors; a sleeping app shows a
"get this app back up" button instead of the app. Run by .github/workflows/keep-alive.yml on a schedule.

    python scripts/wake_app.py [url]
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

URL = "https://vi-toxic-moderation.streamlit.app/"
WAKE_BUTTON = "text=/get this app back up/i"


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else URL
    with sync_playwright() as p:
        # CHROMIUM_PATH: optional browser binary (for running outside GitHub Actions)
        browser = p.chromium.launch(executable_path=os.getenv("CHROMIUM_PATH") or None)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(10_000)

        button = page.locator(WAKE_BUTTON)
        if button.count():
            print("App was asleep: clicking the wake-up button")
            button.first.click()
            page.wait_for_timeout(5_000)
        else:
            print("App is awake")

        # The app may be served inside an iframe; wait until a heading renders in any frame.
        for _ in range(36):
            if any(f.locator("h1").count() for f in page.frames):
                print("App content loaded")
                break
            page.wait_for_timeout(5_000)
        else:
            print("App content did not load within 3 minutes", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
