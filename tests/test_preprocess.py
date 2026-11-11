import sys
from pathlib import Path



from toxic_moderation.preprocess import (  # noqa: E402
    clean_text,
    is_too_short,
    load_teencode,
    normalize_teencode,
)


def test_lowercase_and_punctuation():
    assert clean_text("Xin CHÀO!!! các bạn...") == "xin chào các bạn"


def test_removes_urls_and_emoji():
    assert clean_text("xem https://abc.com nè 😂😂") == "xem nè"


def test_mentions_become_generic_token():
    assert clean_text("@tri123 nói đúng") == "username nói đúng"


def test_squeezes_repeated_chars():
    assert clean_text("điiiiii ngủ ngonnnn") == "đi ngủ ngon"


def test_teencode_whole_tokens_only():
    mapping = {"ko": "không", "dc": "được"}
    assert normalize_teencode("ko biết dc ko", mapping) == "không biết được không"
    assert normalize_teencode("kong", mapping) == "kong"


def test_teencode_with_punctuation_attached():
    assert clean_text("ko, biết dc!", {"ko": "không", "dc": "được"}) == "không biết được"


def test_dictionary_lookup_before_squeezing():
    # "kkk" is laughing; it must not become "k" -> "không"
    assert clean_text("kkk vui", {"kkk": "ka ka ka", "k": "không"}) == "ka ka ka vui"


def test_squeezed_token_is_looked_up():
    assert clean_text("dcccc kooo", {"dc": "được", "ko": "không"}) == "được không"


def test_too_short_filter():
    assert is_too_short("quá hay")
    assert not is_too_short("bài này viết rất hay luôn")


def test_dictionary_loads():
    mapping = load_teencode()
    assert mapping["ko"] == "không"
    assert len(mapping) > 500


def test_fix_bad_token():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("fbt", Path(__file__).resolve().parents[1] / "scripts" / "fix_bad_token.py")
    fbt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fbt)
    assert fbt.fix("có sai bad đâu") == "có sai đéo đâu"
    assert fbt.fix("lý lẽ cái bad gì") == "lý lẽ cái đéo gì"
    assert fbt.fix("độ bad trip") == "độ bad trip"
    assert fbt.fix("too bad i forgot") == "too bad i forgot"
