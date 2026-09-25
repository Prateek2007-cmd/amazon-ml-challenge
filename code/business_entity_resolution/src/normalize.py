"""
Normalization utilities for business names and addresses.

Designed to be robust to unseen countries (e.g. France, which only
appears in the test set): everything here is pattern-based string
cleanup, never a lookup keyed on a fixed {US, India} vocabulary.
"""
import re
import unicodedata

# Canonicalize common legal-suffix / connector variants. Not
# exhaustive -- extend this as you find more patterns in the real data.
_NAME_ABBREV = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "and": "and",  # kept for symmetry after punctuation strip turns "&" -> "and"
}

_ADDRESS_ABBREV = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "apartment": "apt",
    "suite": "ste",
    "floor": "fl",
    "building": "bldg",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Folds accented characters to their closest ASCII form -- handles
    transliteration variants, relevant now that French addresses are
    in the test set."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _clean_base(text) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.lower() == "nan":
        return ""
    text = text.replace("&", " and ")
    text = _strip_accents(text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _apply_abbrev(tokens, mapping):
    return [mapping.get(tok, tok) for tok in tokens]


def normalize_name(raw_name) -> str:
    """Lowercase, strip accents/punctuation, canonicalize legal suffixes."""
    cleaned = _clean_base(raw_name)
    tokens = cleaned.split()
    tokens = _apply_abbrev(tokens, _NAME_ABBREV)
    return " ".join(tokens)


def normalize_address(raw_address) -> str:
    """Lowercase, strip accents/punctuation, canonicalize street-type words."""
    cleaned = _clean_base(raw_address)
    tokens = cleaned.split()
    tokens = _apply_abbrev(tokens, _ADDRESS_ABBREV)
    return " ".join(tokens)


def name_tokens(normalized_name: str):
    return set(normalized_name.split())


def address_numeric_tokens(normalized_address: str):
    """Digit-only tokens (street numbers, PIN/ZIP codes) -- a strong
    signal since these rarely get garbled the way free text does."""
    return set(tok for tok in normalized_address.split() if tok.isdigit())
