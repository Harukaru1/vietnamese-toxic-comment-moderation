"""Small safeguards for the public demo: spreadsheet-safe CSV export, input length caps and per-session quotas."""

from __future__ import annotations

from collections.abc import MutableMapping

import pandas as pd

MAX_TEXT_CHARS = 2000  # longer inputs are cut before they reach the model, the LLM or the feedback log
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def spreadsheet_safe(value):
    """Prefix ' to text that a spreadsheet app would run as a formula (CSV injection)."""
    if isinstance(value, str) and value.startswith(FORMULA_START):
        return "'" + value
    return value


def csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 CSV (with BOM, so Excel shows Vietnamese correctly) in which no cell can run as a formula."""
    safe = df.copy()
    safe.columns = [spreadsheet_safe(str(c)) for c in safe.columns]
    return safe.map(spreadsheet_safe).to_csv(index=False).encode("utf-8-sig")


def clip(text: object, limit: int = MAX_TEXT_CHARS) -> str:
    return str(text)[:limit]


def take_quota(state: MutableMapping, name: str, limit: int) -> bool:
    """Count one use of `name` in a session state; False once `limit` uses are spent."""
    key = f"quota_{name}"
    used = int(state.get(key, 0))
    if used >= limit:
        return False
    state[key] = used + 1
    return True
