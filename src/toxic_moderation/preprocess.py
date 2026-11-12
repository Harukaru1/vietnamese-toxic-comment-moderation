"""Text normalization for Vietnamese social-media comments.

Ported from the thesis `text_normalizer.py` (lowercase, @mention token, emoji removal,
repeated-character squeezing, teencode dictionary) and extended with URL/punctuation removal.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import emoji

DEFAULT_DICT = Path(__file__).resolve().parent / "teencode_dictionary.json"

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_REPEAT = re.compile(r"(\w)\1{2,}", flags=re.UNICODE)  # "yeuuuu" -> "yeu"
_SPACES = re.compile(r"\s+")


@lru_cache(maxsize=4)
def load_teencode(path: str | Path = DEFAULT_DICT) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mappings", data)


def _normalize_token(tok: str, mapping: dict[str, str] | None) -> str:
    # Look up the raw token first so entries like "kkk" (laughing) are not
    # squeezed to "k" (= "không") before the dictionary sees them.
    if mapping and tok in mapping:
        return mapping[tok]
    squeezed = _REPEAT.sub(r"\1", tok)
    if mapping:
        return mapping.get(squeezed, squeezed)
    return squeezed


def normalize_teencode(text: str, mapping: dict[str, str]) -> str:
    """Replace whole-token teencode/slang with its full form."""
    return " ".join(mapping.get(tok, tok) for tok in text.split())


def clean_text(text: str, teencode: dict[str, str] | None = None) -> str:
    """Normalize one comment. Safe to use both for training and at inference time."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" username ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = _NON_WORD.sub(" ", text)
    return " ".join(_normalize_token(tok, teencode) for tok in text.split())


def is_too_short(text: str, min_words: int = 5) -> bool:
    """Dataset-building filter from the thesis (drop comments with < min_words words).

    Only used when building a dataset; never applied at inference time.
    """
    return len(text.split()) < min_words


def has_text(text: str) -> bool:
    """False when nothing is left to classify after cleaning (emoji-only, punctuation-only, empty)."""
    return any(ch.isalnum() for ch in clean_text(text))
