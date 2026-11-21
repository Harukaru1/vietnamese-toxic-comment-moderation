import pandas as pd

from toxic_moderation.safety import MAX_TEXT_CHARS, clip, csv_bytes, spreadsheet_safe, take_quota


def test_formulas_are_neutralised_in_cells_and_headers():
    df = pd.DataFrame({"=cmd|' /C calc'!A0": ['=HYPERLINK("http://x")', "+1", "-2", "@SUM(A1)", "\tx", "bình thường"]})
    out = csv_bytes(df).decode("utf-8-sig").splitlines()
    assert out[0].startswith("'=cmd")
    assert all(line.startswith(("'", "\"'")) for line in out[1:6])
    assert out[6] == "bình thường"


def test_non_text_values_are_left_alone():
    assert spreadsheet_safe(-3) == -3
    assert spreadsheet_safe(None) is None


def test_csv_has_bom_for_excel():
    assert csv_bytes(pd.DataFrame({"a": ["x"]})).startswith("﻿".encode("utf-8"))


def test_clip_limits_length():
    assert len(clip("a" * (MAX_TEXT_CHARS + 50))) == MAX_TEXT_CHARS
    assert clip(123) == "123"


def test_quota_stops_after_limit():
    state = {}
    assert [take_quota(state, "video", 2) for _ in range(3)] == [True, True, False]
    assert take_quota(state, "llm", 1) is True  # separate counters
