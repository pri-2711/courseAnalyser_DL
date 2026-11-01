"""Minimal text cleanup that preserves sentiment-bearing punctuation and words."""
import re
from html import unescape

_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def clean_text(value):
    text = unescape(str(value))
    text = _TAGS.sub(" ", text)
    return _SPACE.sub(" ", text).strip()
