import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxic_moderation.preprocess import clean_text, load_teencode, normalize_teencode  # noqa: E402


def test_lowercase_and_punctuation():
    assert clean_text("Xin CHÀO!!! các bạn...") == "xin chào các bạn"


def test_removes_urls_and_mentions():
    assert clean_text("xem https://abc.com nè @tri") == "xem nè"


def test_squeezes_repeated_chars():
    assert clean_text("điiiiii ngủ đêeeee") == "đii ngủ đêee"


def test_teencode_whole_tokens_only():
    mapping = {"ko": "không", "dc": "được"}
    assert normalize_teencode("ko biết dc ko", mapping) == "không biết được không"
    assert normalize_teencode("kong", mapping) == "kong"


def test_dictionary_loads():
    mapping = load_teencode()
    assert mapping["ko"] == "không"
    assert len(mapping) > 500
