"""Text normalization for Vietnamese social-media comments."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

DEFAULT_DICT = Path(__file__).resolve().parents[2] / "resources" / "teencode_dictionary.json"

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_REPEAT = re.compile(r"(\w)\1{2,}", flags=re.UNICODE)  # "điiiii" -> "đii"
_SPACES = re.compile(r"\s+")


@lru_cache(maxsize=4)
def load_teencode(path: str | Path = DEFAULT_DICT) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mappings", data)


def normalize_teencode(text: str, mapping: dict[str, str]) -> str:
    """Replace whole-token teencode/slang with its full form."""
    return " ".join(mapping.get(tok, tok) for tok in text.split())


def clean_text(text: str, teencode: dict[str, str] | None = None) -> str:
    """Lowercase, strip URLs/mentions/punctuation, squeeze repeats, normalize teencode."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    text = _REPEAT.sub(r"\1\1", text)
    text = _SPACES.sub(" ", text).strip()
    if teencode:
        text = normalize_teencode(text, teencode)
    return text
