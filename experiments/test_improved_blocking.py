"""
Compare Baseline Blocking vs Optimized Multi-Pass Dynamic Blocking on 10,000 entities.
Evaluates Candidate Recall, Total Candidate Pairs, and Average Candidates per S1.
"""
import os, sys, time
from collections import defaultdict, Counter
import unicodedata
import pandas as pd
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")

repo_root = os.path.abspath(".")
sys.path.insert(0, os.path.join(repo_root, "code", "business_entity_resolution", "src"))
from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import (
    normalize_name,
    normalize_address,
    extract_name_aliases,
    extract_url_core,
    extract_clean_numbers,
    NAME_ABBREVIATIONS,
    ADDRESS_ABBREVIATIONS
)
from blocking import extract_blocking_keys as baseline_extract_keys

# Indic Unicode mapping
INDIC_MAP = {
    0x01: "n", 0x02: "m", 0x03: "h",
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "r", 0x0C: "l",
    0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3C: "", 0x3D: "",
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "r", 0x44: "r",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4D: "",
    0x55: "", 0x56: "ai", 0x57: "au",
    0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
    0x66: "0", 0x67: "1", 0x68: "2", 0x69: "3", 0x6A: "4", 0x6B: "5", 0x6C: "6", 0x6D: "7", 0x6E: "8", 0x6F: "9"
}

SCRIPT_BASES = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

def transliterate_indic(text):
    if not text: return ""
    out = []
    has_indic = False
    for ch in text:
        cp = ord(ch)
        matched = False
        for base in SCRIPT_BASES:
            if base <= cp < base + 0x80:
                has_indic = True
                offset = cp - base
                if offset in INDIC_MAP:
                    out.append(INDIC_MAP[offset])
                    matched = True
                    break
        if not matched:
            out.append(ch)
    return "".join(out) if has_indic else ""

NAME_STOPWORDS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp",
    "co", "company", "and", "the", "of", "in", "for", "center", "centre", "group",
    "services", "solutions", "enterprises", "associates", "holdings", "management",
    "international", "global", "industries", "systems", "technologies", "tech", "dba",
    "praivet", "treding" # phonetic transliterations of common suffixes
}

ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "court", "ct", "way", "street", "road",
    "avenue", "boulevard", "drive", "lane", "apt", "suite", "ste", "fl", "floor",
    "bldg", "building", "near", "opp", "opposite", "behind", "phase", "sector",
    "plot", "shop", "no", "number", "us", "usa", "india", "null", "ground", "first",
    "second", "third", "c/o", "flats", "nagar", "dist", "district", "state"
}

def extract_optimized_keys_by_priority(name, address, country=""):
    """
    Returns ordered passes of blocking keys:
    Pass 1: Exact name, URL core, High-precision name token pairs
    Pass 2: Transliterated name pairs / tokens
    Pass 3: Name token + Address number composite keys
    Pass 4: Distinctive address word + Address number composite keys
    Pass 5: Single distinctive name tokens
    Pass 6: Distinctive address pairs
    """
    # Normalize name: replace hyphens/slashes with space
    clean_n = name.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    
    clean_a = address.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    
    n_tokens = norm_n.split()
    content_n = [t for t in n_tokens if len(t) >= 2 and t not in NAME_STOPWORDS]
    
    url_core = extract_url_core(norm_n)
    addr_nums = extract_clean_numbers(norm_a)
    
    # Pure non-digit address tokens, length >= 4, not generic stopwords
    addr_words = [t for t in norm_a.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and not any(c.isdigit() for c in t)]
    
    pass1_keys = []
    # 1. Exact Name
    if norm_n and len(norm_n) >= 4:
        pass1_keys.append(("exact_name", norm_n))
    # 2. URL Domain
    if url_core and len(url_core) >= 4:
        pass1_keys.append(("url_core", url_core))
    # 3. Content Name Pairs
    if len(content_n) >= 2:
        pass1_keys.append(("name_pair", tuple(sorted([content_n[0], content_n[1]]))))
        if len(content_n) >= 3:
            pass1_keys.append(("name_pair", tuple(sorted([content_n[0], content_n[2]]))))
            pass1_keys.append(("name_pair", tuple(sorted([content_n[1], content_n[2]]))))

    # Pass 2: Transliterated keys (if Indic script present)
    pass2_keys = []
    trans_n = transliterate_indic(name)
    if trans_n:
        norm_tn = normalize_name(trans_n.replace("-", " ").replace("/", " "))
        t_tokens = [t for t in norm_tn.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
        if len(t_tokens) >= 2:
            pass2_keys.append(("name_pair", tuple(sorted([t_tokens[0], t_tokens[1]]))))
        for t in t_tokens[:2]:
            pass2_keys.append(("name_tok", t))

    # Pass 3: Name token + Address number composite keys
    pass3_keys = []
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]:
                pass3_keys.append(("name_num", nt, num))

    # Pass 4: Address Number + Distinctive Address Word
    pass4_keys = []
    if addr_nums and addr_words:
        for num in addr_nums[:3]:
            for w in addr_words[:3]:
                pass4_keys.append(("num_word", num, w))

    # Pass 5: Single distinctive name tokens
    pass5_keys = []
    for t in content_n[:2]:
        if len(t) >= 4:
            pass5_keys.append(("name_tok", t))

    # Pass 6: Address word pairs
    pass6_keys = []
    if len(addr_words) >= 2:
        pass6_keys.append(("addr_pair", tuple(sorted([addr_words[0], addr_words[1]]))))
        if len(addr_words) >= 3:
            pass6_keys.append(("addr_pair", tuple(sorted([addr_words[0], addr_words[2]]))))

    # Aliases
    aliases = extract_name_aliases(name)
    if len(aliases) > 1:
        for alias in aliases[1:]:
            norm_alias = normalize_name(alias.replace("-", " ").replace("/", " "))
            a_toks = [t for t in norm_alias.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            if len(a_toks) >= 2:
                pass1_keys.append(("name_pair", tuple(sorted([a_toks[0], a_toks[1]]))))
            for at in a_toks[:2]:
                pass5_keys.append(("name_tok", at))

    return [pass1_keys, pass2_keys, pass3_keys, pass4_keys, pass5_keys, pass6_keys]

def main():
    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    s1_needed = set(gt_dict.keys())
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"): needed_s2.add(m)
            elif m.startswith("S3-"): needed_s3.add(m)

    s1_records = {}
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in s1_needed:
                s1_records[parts[0]] = dict(zip(header, parts))
                if len(s1_records) == len(s1_needed): break

    def load_pool(path, needed_set, max_pool=100000):
        pool = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                if eid in needed_set or i < max_pool:
                    pool[eid] = dict(zip(header, parts))
        return pool

    print(f"Loading pool...")
    s2_dict = load_pool(paths["train_s2"], needed_s2)
    s3_dict = load_pool(paths["train_s3"], needed_s3)
    print(f"Loaded: S1={len(s1_records)}, S2={len(s2_dict)}, S3={len(s3_dict)}")

    # 1. Build Inverted Index with Optimized Keys
    idx_s2 = defaultdict(list)
    for eid, row in s2_dict.items():
        ordered_passes = extract_optimized_keys_by_priority(row["business_name"], row["business_address"], row.get("country", ""))
        for pass_keys in ordered_passes:
            for k in pass_keys:
                idx_s2[k].append(eid)

    idx_s3 = defaultdict(list)
    for eid, row in s3_dict.items():
        ordered_passes = extract_optimized_keys_by_priority(row["business_name"], row["business_address"], row.get("country", ""))
        for pass_keys in ordered_passes:
            for k in pass_keys:
                idx_s3[k].append(eid)

    print(f"Index built: S2 keys={len(idx_s2):,}, S3 keys={len(idx_s3):,}")

    # Evaluate Candidate Generation
    total_true = sum(len(m) for m in gt_dict.values())
    
    # Test configurations
    for max_cands in [80, 100, 120, 150]:
        found_true = 0
        cand_counts = []
        
        for s1_id, s1_row in s1_records.items():
            true_set = gt_dict.get(s1_id, set())
            ordered_passes = extract_optimized_keys_by_priority(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
            
            cands = set()
            for p_idx, pass_keys in enumerate(ordered_passes):
                cap = 150 if p_idx == 4 else 400 # pass 5 is single name_tok
                for k in pass_keys:
                    p2 = idx_s2.get(k, [])
                    p3 = idx_s3.get(k, [])
                    if len(p2) <= cap: cands.update(p2)
                    if len(p3) <= cap: cands.update(p3)
                    if len(cands) >= max_cands:
                        break
                if len(cands) >= max_cands:
                    break
                    
            cand_counts.append(len(cands))
            found_true += len(true_set & cands)
            
        recall = found_true / total_true * 100
        avg_cands = sum(cand_counts) / len(cand_counts)
        sorted_c = sorted(cand_counts)
        p50 = sorted_c[int(len(sorted_c)*0.50)]
        p95 = sorted_c[int(len(sorted_c)*0.95)]
        p99 = sorted_c[int(len(sorted_c)*0.99)]
        print(f"Max Cands={max_cands}: Recall={recall:.2f}% ({found_true}/{total_true}), Avg={avg_cands:.1f}, P50={p50}, P95={p95}, P99={p99}")

if __name__ == "__main__":
    main()
