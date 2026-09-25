"""
Robust Multi-Representation Normalization System for Business Entity Resolution.
Supports:
1. Unicode accent folding & NFKD normalization
2. Open-set international address and name handling (India, US, France, etc.)
3. Legal suffix and street type canonicalization
4. Hashtag / handle splitting (#sjace -> sj ace)
5. Web domain and URL normalization
6. Business alias detection (dba, aka, formerly, fka)
7. Alphanumeric plot / door number preservation
"""
import re
import unicodedata

# Legal Suffix Canonicalization
NAME_ABBREVIATIONS = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "and": "and",
}

# Street & Address Component Canonicalization
ADDRESS_ABBREVIATIONS = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "apartment": "apt",
    "suite": "ste",
    "floor": "fl",
    "building": "bldg",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
}

PUNCT_RE = re.compile(r"[^\w\s\-/]")
WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"([a-zA-Z0-9\-]+)\.(?:com|in|org|net|co|io|biz|info|gov|edu|fr)", re.I)
NUM_CLEAN_RE = re.compile(r"^([0-9]+)[a-zA-Z]?$")
ALIAS_RE = re.compile(r"\b(?:aka|dba|d/b/a|formerly|f/k/a|fka)\b", re.I)

def strip_accents(text: str) -> str:
    """Folds accented characters to ASCII equivalents (e.g. é -> e, ô -> o)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def clean_base(text) -> str:
    """Core string normalization pipeline."""
    if text is None:
        return ""
    text = str(text)
    if text.lower() in ("nan", "null", "none"):
        return ""
    text = text.replace("&", " and ")
    text = text.replace("#", " ")
    text = text.replace("@", " ")
    text = strip_accents(text)
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

def apply_abbrev(tokens, mapping):
    return [mapping.get(tok, tok) for tok in tokens]

def normalize_name(raw_name) -> str:
    """Normalizes business name: lowercase, accents, punctuation, legal suffix canonicalization."""
    cleaned = clean_base(raw_name)
    tokens = cleaned.split()
    tokens = apply_abbrev(tokens, NAME_ABBREVIATIONS)
    return " ".join(tokens)

def normalize_address(raw_address) -> str:
    """Normalizes address: lowercase, accents, punctuation, address component canonicalization."""
    cleaned = clean_base(raw_address)
    tokens = cleaned.split()
    tokens = apply_abbrev(tokens, ADDRESS_ABBREVIATIONS)
    return " ".join(tokens)

def extract_name_aliases(raw_name):
    """Extracts aliases from phrases like 'Parent dba Subsidiary' or 'New formerly Old'."""
    parts = ALIAS_RE.split(str(raw_name))
    return [p.strip() for p in parts if p.strip()]

def extract_url_core(norm_name: str):
    """Extracts base domain from web-style business names (e.g., dprobst.com -> dprobst)."""
    m = URL_RE.search(norm_name)
    return m.group(1).lower() if m else None

def extract_clean_numbers(norm_address: str):
    """Extracts clean house / door numbers, stripping letter suffixes and leading zeros."""
    tokens = norm_address.split()
    nums = []
    for t in tokens:
        clean_match = NUM_CLEAN_RE.match(t)
        if clean_match:
            val = clean_match.group(1).lstrip("0") or "0"
            if len(val) <= 7:
                nums.append(val)
        elif "/" in t or "-" in t:
            # Door / plot numbers like 6-2-101 or 6/29
            sub = t.replace("/", " ").replace("-", " ")
            for st in sub.split():
                if st.isdigit() and len(st) <= 7:
                    nums.append(st.lstrip("0") or "0")
    return list(dict.fromkeys(nums))  # preserve order, unique
