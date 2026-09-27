"""
Phase 8: High-Leverage Feature Engineering Engine
Amazon ML Challenge 2026 - Business Entity Resolution

Engineers 19 targeted features tackling the remaining error modes:
- IDF-weighted token discrimination (rare token match vs generic collision)
- Postal / PIN code agreement vs contradiction
- Directional containment (short vs verbose addresses and names)
- URL core matching
- Empty-address conditional confidence & penalty
"""
import os, sys, math, time, re
from collections import defaultdict, Counter
import numpy as np

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

def compute_corpus_token_idf(cached_entities):
    """
    Computes smooth IDF for all name tokens across the candidate universe:
    IDF(t) = ln((N + 1) / (DF(t) + 1)) + 1
    """
    df = Counter()
    n_docs = len(cached_entities)
    for e in cached_entities.values():
        for t in e["toks_n"]:
            df[t] += 1
    
    idf = {}
    default_idf = math.log((n_docs + 1) / 1.0) + 1.0
    for t, count in df.items():
        idf[t] = math.log((n_docs + 1) / (count + 1)) + 1.0
    return idf, default_idf

def extract_advanced_features_19(e1, e2, base_31, idf_dict, default_idf):
    """
    Extracts 19 high-leverage features for pair (e1, e2).
    base_31: array of 31 base features (for ratio_name, empty_addr, etc.)
    Returns list of 19 floats.
    """
    feats = [0.0] * 19

    # 1. PIN code features (0 to 3)
    pin1 = e1.get("pincode", "")
    pin2 = e2.get("pincode", "")
    if pin1 and pin2:
        feats[0] = 1.0 if pin1 == pin2 else 0.0              # pin_exact_match
        feats[1] = 1.0 if pin1[:3] == pin2[:3] else 0.0      # pin_prefix3_match
        feats[2] = 1.0 if pin1 != pin2 else 0.0              # pin_conflict
        feats[3] = 1.0                                       # both_have_pin

    # 2. Address directional containment (4 to 6)
    t_a1 = e1["toks_a"]
    t_a2 = e2["toks_a"]
    overlap_a = len(t_a1 & t_a2)
    c_s1_in_s2_a = overlap_a / (len(t_a1) + 1e-5) if t_a1 else 0.0
    c_s2_in_s1_a = overlap_a / (len(t_a2) + 1e-5) if t_a2 else 0.0
    feats[4] = c_s1_in_s2_a                                  # addr_containment_s1_in_s2
    feats[5] = c_s2_in_s1_a                                  # addr_containment_s2_in_s1
    feats[6] = max(c_s1_in_s2_a, c_s2_in_s1_a)               # addr_containment_max

    # 3. Name directional containment (7 to 8)
    t_n1 = e1["toks_n"]
    t_n2 = e2["toks_n"]
    overlap_n = t_n1 & t_n2
    c_s1_in_s2_n = len(overlap_n) / (len(t_n1) + 1e-5) if t_n1 else 0.0
    c_s2_in_s1_n = len(overlap_n) / (len(t_n2) + 1e-5) if t_n2 else 0.0
    feats[7] = c_s1_in_s2_n                                  # name_containment_s1_in_s2
    feats[8] = c_s2_in_s1_n                                  # name_containment_s2_in_s1

    # 4. IDF-weighted token discrimination (9 to 14)
    union_n = t_n1 | t_n2
    sum_shared_idf = 0.0
    max_shared_idf = 0.0
    for t in overlap_n:
        val = idf_dict.get(t, default_idf)
        sum_shared_idf += val
        if val > max_shared_idf:
            max_shared_idf = val

    sum_union_idf = 0.0
    for t in union_n:
        sum_union_idf += idf_dict.get(t, default_idf)

    unshared_n = t_n1 ^ t_n2
    max_unshared_idf = 0.0
    for t in unshared_n:
        val = idf_dict.get(t, default_idf)
        if val > max_unshared_idf:
            max_unshared_idf = val

    feats[9] = sum_shared_idf                                # idf_overlap_sum
    feats[10] = sum_shared_idf / (sum_union_idf + 1e-5) if sum_union_idf > 0 else 0.0 # idf_weighted_jaccard
    feats[11] = max_shared_idf                               # max_shared_token_idf
    feats[12] = max_unshared_idf                             # max_unshared_token_idf
    feats[13] = 1.0 if max_shared_idf >= 7.0 else 0.0       # rare_token_match_flag
    feats[14] = 1.0 if max_unshared_idf >= 7.0 and max_shared_idf < 7.0 else 0.0 # rare_token_conflict_flag

    # 5. URL core matching (15 to 16)
    u1 = e1.get("url_core", "")
    u2 = e2.get("url_core", "")
    norm_n1 = e1["norm_n"]
    norm_n2 = e2["norm_n"]
    feats[15] = 1.0 if u1 and u2 and u1 == u2 else 0.0       # url_core_match
    feats[16] = 1.0 if (u1 and len(u1) >= 4 and u1 in norm_n2) or (u2 and len(u2) >= 4 and u2 in norm_n1) else 0.0 # url_core_in_name

    # 6. Empty address conditional protection & penalty (17 to 18)
    empty_addr = base_31[19]
    ratio_name = base_31[1]
    # Penalty: moderate name similarity with empty address (often a collision)
    feats[17] = 1.0 if empty_addr and 65.0 <= ratio_name < 92.0 else 0.0
    # Boost: extremely high name similarity + rare token match with empty address
    feats[18] = 1.0 if empty_addr and ratio_name >= 94.0 and max_shared_idf >= 5.5 else 0.0

    return feats
