"""
Pairwise similarity features between a Source1 record and a
Source2/Source3 candidate. These feed the classifier in model.py.
"""
from rapidfuzz import fuzz

from normalize import (
    normalize_name,
    normalize_address,
    name_tokens,
    address_numeric_tokens,
)


def _jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


FEATURE_COLUMNS = [
    "name_exact_match",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_numeric_jaccard",
    "name_len_diff",
    "country_match",
]


def pair_features(name_a, addr_a, country_a, name_b, addr_b, country_b) -> dict:
    """Returns a dict of features for one (S1 record, candidate) pair.
    Keys always match FEATURE_COLUMNS."""
    norm_name_a, norm_name_b = normalize_name(name_a), normalize_name(name_b)
    norm_addr_a, norm_addr_b = normalize_address(addr_a), normalize_address(addr_b)

    tokens_a, tokens_b = name_tokens(norm_name_a), name_tokens(norm_name_b)
    num_a, num_b = address_numeric_tokens(norm_addr_a), address_numeric_tokens(norm_addr_b)

    return {
        "name_exact_match": float(norm_name_a == norm_name_b),
        "name_levenshtein_ratio": fuzz.ratio(norm_name_a, norm_name_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_token_jaccard": _jaccard(tokens_a, tokens_b),
        "addr_levenshtein_ratio": fuzz.ratio(norm_addr_a, norm_addr_b) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0,
        "addr_numeric_jaccard": _jaccard(num_a, num_b),
        "name_len_diff": float(abs(len(norm_name_a) - len(norm_name_b))),
        "country_match": float(str(country_a).strip().lower() == str(country_b).strip().lower()),
    }
