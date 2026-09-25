"""
High-Recall Multi-Stage Composite Blocking for Business Entity Resolution.
Combines multiple orthogonal blocking strategies:
1. Exact normalized name
2. URL domain core
3. Content name 2-token signatures
4. Rare content name tokens
5. Composite address keys (number + street/locality word)
6. Composite name-address keys (name token + address number)
7. Distinctive address word pairs
8. Business alias keys (formerly, aka, dba)
"""
from collections import defaultdict
import re
from normalization import (
    normalize_name,
    normalize_address,
    extract_name_aliases,
    extract_url_core,
    extract_clean_numbers,
)
from config import Config

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

def extract_blocking_keys(name, address, country=""):
    """Generates a rich set of blocking keys for an entity."""
    norm_n = normalize_name(name)
    norm_a = normalize_address(address)
    
    n_tokens = norm_n.split()
    content_n = [t for t in n_tokens if len(t) >= 2 and t not in NAME_STOPWORDS]
    
    url_core = extract_url_core(norm_n)
    addr_nums = extract_clean_numbers(norm_a)
    addr_words = [t for t in norm_a.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS]
    
    keys = set()
    
    # 1. Exact Name
    if norm_n and len(norm_n) >= 4:
        keys.add(("exact_name", norm_n))
        
    # 2. URL Domain
    if url_core and len(url_core) >= 4:
        keys.add(("url_core", url_core))
        
    # 3. Content Name 2-Token Signatures
    if len(content_n) >= 2:
        keys.add(("name_pair", tuple(sorted([content_n[0], content_n[1]]))))
        if len(content_n) >= 3:
            keys.add(("name_pair", tuple(sorted([content_n[0], content_n[2]]))))
            
    # 4. Content Name Tokens
    for t in content_n[:2]:
        if len(t) >= 3:
            keys.add(("name_tok", t))
            
    # 5. Composite Address Keys: (number, street/locality word)
    if addr_nums and addr_words:
        for num in addr_nums[:2]:
            for w in addr_words[:2]:
                keys.add(("num_word", num, w))
                
    # 6. Composite Name-Address Keys: (name token, number)
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]:
                keys.add(("name_num", nt, num))
                
    # 7. Distinctive Address Word Pairs
    if len(addr_words) >= 2:
        keys.add(("addr_pair", tuple(sorted([addr_words[0], addr_words[1]]))))
        if len(addr_words) >= 3:
            keys.add(("addr_pair", tuple(sorted([addr_words[0], addr_words[2]]))))
            
    # 8. Alias Keys
    aliases = extract_name_aliases(name)
    if len(aliases) > 1:
        for alias in aliases[1:]:
            norm_alias = normalize_name(alias)
            a_toks = [t for t in norm_alias.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            if len(a_toks) >= 2:
                keys.add(("name_pair", tuple(sorted([a_toks[0], a_toks[1]]))))
            for at in a_toks[:2]:
                keys.add(("name_tok", at))
                
    return keys

def build_blocking_index(df_or_iterable, id_col="entity_id", name_col="business_name", addr_col="business_address", country_col="country"):
    """Constructs inverted index from blocking key -> list of entity IDs."""
    index = defaultdict(list)
    
    if hasattr(df_or_iterable, "itertuples"):
        for row in df_or_iterable.itertuples():
            eid = getattr(row, id_col)
            name = getattr(row, name_col, "")
            addr = getattr(row, addr_col, "")
            country = getattr(row, country_col, "")
            for key in extract_blocking_keys(name, addr, country):
                index[key].append(eid)
    else:
        for row in df_or_iterable:
            eid = row[id_col]
            name = row.get(name_col, "")
            addr = row.get(addr_col, "")
            country = row.get(country_col, "")
            for key in extract_blocking_keys(name, addr, country):
                index[key].append(eid)
                
    return index

def generate_candidates_for_s1(s1_row, index_s2, index_s3, max_name_cap=None, max_comp_cap=None, max_cands=None):
    """Generates candidate matches from S2 and S3 for a single S1 record."""
    if max_name_cap is None:
        max_name_cap = Config.MAX_NAME_TOK_POSTINGS
    if max_comp_cap is None:
        max_comp_cap = Config.MAX_COMPOSITE_POSTINGS
    if max_cands is None:
        max_cands = Config.MAX_CANDIDATES_PER_S1
        
    keys = extract_blocking_keys(
        s1_row.get("business_name", ""),
        s1_row.get("business_address", ""),
        s1_row.get("country", "")
    )
    
    candidates = set()
    for k in keys:
        cap = max_name_cap if k[0] == "name_tok" else max_comp_cap
        p2 = index_s2.get(k, [])
        p3 = index_s3.get(k, [])
        if len(p2) <= cap:
            candidates.update(p2)
        if len(p3) <= cap:
            candidates.update(p3)
        if len(candidates) >= max_cands:
            break
            
    return candidates

def blocking_recall(candidates_dict, ground_truth_dict):
    """
    Measures blocking recall and identifies missed true pairs.
    ground_truth_dict: {s1_id: set(matched_ids)}
    candidates_dict: {s1_id: set(candidate_ids)}
    """
    total_true = 0
    total_found = 0
    missed = []
    
    for s1_id, true_set in ground_truth_dict.items():
        cand_set = candidates_dict.get(s1_id, set())
        for match_id in true_set:
            total_true += 1
            if match_id in cand_set:
                total_found += 1
            else:
                missed.append((s1_id, match_id))
                
    recall = total_found / total_true if total_true else 1.0
    return recall, missed
