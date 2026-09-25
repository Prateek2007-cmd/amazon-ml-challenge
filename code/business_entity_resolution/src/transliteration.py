"""
Local Deterministic Transliteration & Script-Aware Normalization Subsystem
For Multi-Lingual Business Entity Resolution (Amazon ML Challenge 2026).

Zero External APIs | Zero External Business Data | 100% Pure Python & Unicode

Supports:
1. Deterministic Brahmi-derived Unicode offset transliteration:
   - Devanagari (Hindi, Marathi, Sanskrit)
   - Bengali / Assamese
   - Gurmukhi (Punjabi)
   - Gujarati
   - Oriya / Odia
   - Tamil
   - Telugu
   - Kannada
   - Malayalam
2. Script detection and non-ASCII character categorization
3. Phonetic consonant skeleton extraction
4. Script-preserving and transliterated n-grams
"""
import unicodedata
import re

# Standard Brahmic Script Unicode Block Bases (128-byte offsets in Unicode standard)
SCRIPT_BASES = [
    0x0900,  # Devanagari (0x0900 - 0x097F)
    0x0980,  # Bengali (0x0980 - 0x09FF)
    0x0A00,  # Gurmukhi (0x0A00 - 0x0A7F)
    0x0A80,  # Gujarati (0x0A80 - 0x0AFF)
    0x0B00,  # Oriya (0x0B00 - 0x0B7F)
    0x0B80,  # Tamil (0x0B80 - 0x0BFF)
    0x0C00,  # Telugu (0x0C00 - 0x0C7F)
    0x0C80,  # Kannada (0x0C80 - 0x0CFF)
    0x0D00,  # Malayalam (0x0D00 - 0x0D7F)
]

# Standard ISCII / Unicode relative offset mapping to Latin phonetic characters
INDIC_OFFSET_MAP = {
    0x01: "n", 0x02: "m", 0x03: "h",
    # Independent vowels
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "r", 0x0C: "l",
    0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
    # Consonants (velar, palatal, retroflex, dental, labial)
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    # Sonorants, fricatives
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # Modifiers & Matras (dependent vowel signs)
    0x3C: "", 0x3D: "",
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "r", 0x44: "r",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4D: "",
    0x55: "", 0x56: "ai", 0x57: "au",
    # Additional letters & Persian borrowings
    0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
    # Digits
    0x66: "0", 0x67: "1", 0x68: "2", 0x69: "3", 0x6A: "4", 0x6B: "5", 0x6C: "6", 0x6D: "7", 0x6E: "8", 0x6F: "9"
}

def has_indic_script(text: str) -> bool:
    """Checks whether the text contains characters from any Indic Unicode block."""
    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        for base in SCRIPT_BASES:
            if base <= cp < base + 0x80:
                return True
    return False

def transliterate_indic(text: str) -> str:
    """
    Deterministically transliterates Brahmic Indic scripts to standard Latin phonetics.
    Returns transliterated Latin string if Indic characters were present, else empty string.
    """
    if not text:
        return ""
    out = []
    has_indic = False
    for ch in text:
        cp = ord(ch)
        matched = False
        for base in SCRIPT_BASES:
            if base <= cp < base + 0x80:
                has_indic = True
                offset = cp - base
                if offset in INDIC_OFFSET_MAP:
                    out.append(INDIC_OFFSET_MAP[offset])
                    matched = True
                    break
        if not matched:
            out.append(ch)
    return "".join(out) if has_indic else ""

def get_consonant_skeleton(text: str) -> str:
    """Extracts consonant skeleton from Latin/normalized text for phonetic resilience."""
    if not text:
        return ""
    vowels = set("aeiou")
    return "".join(c for c in text.lower() if c.isalpha() and c not in vowels)
