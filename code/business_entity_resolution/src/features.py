"""
High-Performance Pairwise Feature Engineering for Business Entity Resolution.
Extracts 26 rich C-accelerated similarity and interaction features using RapidFuzz.
"""
from rapidfuzz import fuzz
from normalization import (
    normalize_name,
    normalize_address,
    extract_name_aliases,
    extract_clean_numbers,
)

FEATURE_COLUMNS = [
    # Name features (10)
    "name_exact",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_len_diff",
    "name_len_ratio",
    "alias_match",
    
    # Address features (10)
    "addr_exact",
    "addr_fuzz_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_token_jaccard",
    "addr_numeric_jaccard",
    "addr_numeric_overlap_count",
    "addr_len_diff",
    "addr_is_empty_either",
    
    # Cross & Meta features (6)
    "country_match",
    "is_source3",
    "name_addr_mult",
    "name_addr_min",
    "name_addr_max",
    "name_addr_mean",
]

NAME_STOPWORDS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp",
    "co", "company", "and", "the", "of", "in", "for", "center", "centre", "group",
    "services", "solutions", "enterprises", "associates", "holdings", "management",
    "international", "global", "industries", "systems", "technologies", "tech", "dba"
}

ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "court", "ct", "way", "street", "road",
    "avenue", "boulevard", "drive", "lane", "apt", "suite", "ste", "fl", "floor",
    "bldg", "building", "near", "opp", "opposite", "behind", "phase", "sector",
    "plot", "shop", "no", "number", "us", "usa", "india", "null"
}

def extract_pairwise_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country, cand_id):
    """Computes exact 26-element feature vector for a pair."""
    norm_n1, norm_n2 = normalize_name(s1_name), normalize_name(cand_name)
    norm_a1, norm_a2 = normalize_address(s1_addr), normalize_address(cand_addr)
    
    # Name tokens
    toks_n1 = set(t for t in norm_n1.split() if t not in NAME_STOPWORDS)
    toks_n2 = set(t for t in norm_n2.split() if t not in NAME_STOPWORDS)
    name_union = toks_n1 | toks_n2
    name_inter = toks_n1 & toks_n2
    name_jaccard = len(name_inter) / len(name_union) if name_union else 0.0
    
    # RapidFuzz name similarities
    ratio_n = fuzz.ratio(norm_n1, norm_n2) / 100.0
    tsort_n = fuzz.token_sort_ratio(norm_n1, norm_n2) / 100.0
    tset_n = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0
    partial_n = fuzz.partial_ratio(norm_n1, norm_n2) / 100.0
    
    # Alias check
    aliases2 = extract_name_aliases(cand_name)
    alias_match = 0.0
    if len(aliases2) > 1:
        for a in aliases2:
            if fuzz.token_sort_ratio(norm_n1, normalize_name(a)) >= 85:
                alias_match = 1.0
                break
                
    # Address tokens
    toks_a1 = set(t for t in norm_a1.split() if t not in ADDR_STOPWORDS)
    toks_a2 = set(t for t in norm_a2.split() if t not in ADDR_STOPWORDS)
    addr_union = toks_a1 | toks_a2
    addr_inter = toks_a1 & toks_a2
    addr_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0
    
    # Address numbers
    nums_1 = set(extract_clean_numbers(norm_a1))
    nums_2 = set(extract_clean_numbers(norm_a2))
    num_union = nums_1 | nums_2
    num_inter = nums_1 & nums_2
    num_jaccard = len(num_inter) / len(num_union) if num_union else 0.0
    
    # RapidFuzz address similarities
    empty_addr = float(len(norm_a1) == 0 or len(norm_a2) == 0)
    if empty_addr:
        ratio_a = 0.0
        tsort_a = 0.0
        tset_a = 0.0
        partial_a = 0.0
    else:
        ratio_a = fuzz.ratio(norm_a1, norm_a2) / 100.0
        tsort_a = fuzz.token_sort_ratio(norm_a1, norm_a2) / 100.0
        tset_a = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0
        partial_a = fuzz.partial_ratio(norm_a1, norm_a2) / 100.0
        
    c1 = str(s1_country).strip().upper()
    c2 = str(cand_country).strip().upper()
    country_match = float(c1 == c2 and c1 != "")
    is_s3 = float(str(cand_id).startswith("S3-"))
    
    len_diff_n = abs(len(norm_n1) - len(norm_n2))
    max_len_n = max(len(norm_n1), len(norm_n2), 1)
    len_ratio_n = 1.0 - (len_diff_n / max_len_n)
    
    len_diff_a = abs(len(norm_a1) - len(norm_a2))
    
    # Interaction terms
    mult = tsort_n * (tsort_a if not empty_addr else tsort_n)
    min_sim = min(tsort_n, tsort_a) if not empty_addr else tsort_n
    max_sim = max(tsort_n, tsort_a)
    mean_sim = (tsort_n + tsort_a) / 2.0 if not empty_addr else tsort_n
    
    return [
        float(norm_n1 == norm_n2 and len(norm_n1) > 0),
        ratio_n,
        tsort_n,
        tset_n,
        partial_n,
        name_jaccard,
        float(len(name_inter)),
        float(len_diff_n),
        len_ratio_n,
        alias_match,
        
        float(norm_a1 == norm_a2 and len(norm_a1) > 0),
        ratio_a,
        tsort_a,
        tset_a,
        partial_a,
        addr_jaccard,
        num_jaccard,
        float(len(num_inter)),
        float(len_diff_a),
        empty_addr,
        
        country_match,
        is_s3,
        mult,
        min_sim,
        max_sim,
        mean_sim,
    ]

def extract_pairwise_features_prenorm(
    norm_n1, norm_a1, c1, toks_n1, toks_a1, nums_1,
    norm_n2, norm_a2, c2, toks_n2, toks_a2, nums_2,
    cand_id
):
    """Ultra-fast feature extraction using pre-normalized and pre-tokenized inputs."""
    name_union = toks_n1 | toks_n2
    name_inter = toks_n1 & toks_n2
    name_jaccard = len(name_inter) / len(name_union) if name_union else 0.0
    
    ratio_n = fuzz.ratio(norm_n1, norm_n2) / 100.0
    tsort_n = fuzz.token_sort_ratio(norm_n1, norm_n2) / 100.0
    tset_n = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0
    partial_n = fuzz.partial_ratio(norm_n1, norm_n2) / 100.0
    alias_match = 0.0
    
    addr_union = toks_a1 | toks_a2
    addr_inter = toks_a1 & toks_a2
    addr_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0
    
    num_union = nums_1 | nums_2
    num_inter = nums_1 & nums_2
    num_jaccard = len(num_inter) / len(num_union) if num_union else 0.0
    
    empty_addr = float(len(norm_a1) == 0 or len(norm_a2) == 0)
    if empty_addr:
        ratio_a = 0.0
        tsort_a = 0.0
        tset_a = 0.0
        partial_a = 0.0
    else:
        ratio_a = fuzz.ratio(norm_a1, norm_a2) / 100.0
        tsort_a = fuzz.token_sort_ratio(norm_a1, norm_a2) / 100.0
        tset_a = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0
        partial_a = fuzz.partial_ratio(norm_a1, norm_a2) / 100.0
        
    country_match = float(c1 == c2 and c1 != "")
    is_s3 = float(str(cand_id).startswith("S3-"))
    
    len_diff_n = abs(len(norm_n1) - len(norm_n2))
    max_len_n = max(len(norm_n1), len(norm_n2), 1)
    len_ratio_n = 1.0 - (len_diff_n / max_len_n)
    len_diff_a = abs(len(norm_a1) - len(norm_a2))
    
    mult = tsort_n * (tsort_a if not empty_addr else tsort_n)
    min_sim = min(tsort_n, tsort_a) if not empty_addr else tsort_n
    max_sim = max(tsort_n, tsort_a)
    mean_sim = (tsort_n + tsort_a) / 2.0 if not empty_addr else tsort_n
    
    return [
        float(norm_n1 == norm_n2 and len(norm_n1) > 0),
        ratio_n,
        tsort_n,
        tset_n,
        partial_n,
        name_jaccard,
        float(len(name_inter)),
        float(len_diff_n),
        len_ratio_n,
        alias_match,
        
        float(norm_a1 == norm_a2 and len(norm_a1) > 0),
        ratio_a,
        tsort_a,
        tset_a,
        partial_a,
        addr_jaccard,
        num_jaccard,
        float(len(num_inter)),
        float(len_diff_a),
        empty_addr,
        
        country_match,
        is_s3,
        mult,
        min_sim,
        max_sim,
        mean_sim,
    ]
