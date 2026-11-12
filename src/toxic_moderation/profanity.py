"""Keyword flag for swear words, separate from the toxicity model.

The model follows the annotation guideline, where profanity used as an intensifier ("hay vcl") is clean.
Some communities (children's channels, school or brand pages) still want such comments flagged as a matter
of form. That is a policy choice and a word list answers it better than a model: it is transparent, cheap and
does not teach the classifier that "contains a swear word" means "toxic". On 12k YouTube comments about 3%
contain one of these words and three quarters of those are clean under the guideline.
"""

from __future__ import annotations

import re

from .preprocess import clean_text

# Matched on cleaned text (lowercased, repeated letters squeezed), whole tokens only.
_WORDS = [
    "vl", "vcl", "vkl", "vloz", "vãi", "vcc",
    "đm", "dm", "đmm", "dmm", "đcm", "dcm", "clm", "cmn", "cmnr", "đkm", "dkm",
    "đéo", "éo", "đếch", "dell", "đell", "méo", "đíu",
    "cc", "cặc", "cak", "lồn", "loz", "buồi", "địt", "đụ", "đù",
    "mẹ kiếp", "mẹ nó", "con mẹ", "bỏ mẹ", "mịa", "moẹ",
]
_PATTERN = re.compile(r"(?<![\w])(" + "|".join(sorted(map(re.escape, _WORDS), key=len, reverse=True)) + r")(?![\w])")


def find_profanity(text: str) -> list[str]:
    """Swear words found in the comment, in order of appearance, without duplicates."""
    found = _PATTERN.findall(clean_text(text))
    return list(dict.fromkeys(found))


def has_profanity(text: str) -> bool:
    return bool(_PATTERN.search(clean_text(text)))
