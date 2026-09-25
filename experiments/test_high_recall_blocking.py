"""
High-Recall Composite Blocking Benchmark for Amazon ML Challenge 2026.
Tests multi-signal composite blocking combining:
- Normalized exact name
- Cleaned content token pairs
- Low-frequency individual content tokens
- URL domain tokens
- Composite (house_number, street/city_word)
- Composite (name_token, house_number)
- Cleaned address postal codes
"""
import os
import sys
import time
import re
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address, name_tokens, address_numeric_tokens

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

URL_REGEX = re.compile(r"([a-zA-Z0-9\-]+)\.(?:com|in|org|net|co|io|biz|info|gov|edu)", re.I)

def get_composite_keys(raw_name, raw_addr, country=""):
    norm_n = normalize_name(raw_name)
    norm_a = normalize_address(raw_addr)
    
    # 1. Name tokens
    all_n_tokens = norm_n.split()
    content_n_tokens = [t for t in all_n_tokens if len(t) >= 3 and t not in NAME_STOPWORDS]
    
    # URL core
    m = URL_REGEX.search(norm_n)
    url_core = m.group(1).lower() if m else None
    
    # 2. Address tokens
    all_a_tokens = norm_a.split()
    addr_nums = [t.lstrip("0") or "0" for t in all_a_tokens if t.isdigit() and len(t) <= 7]
    addr_words = [t for t in all_a_tokens if len(t) >= 3 and not t.isdigit() and t not in ADDR_STOPWORDS]
    
    keys = set()
    
    # Name Keys
    if norm_n and len(norm_n) >= 4:
        keys.add(("exact_name", norm_n))
        
    if url_core and len(url_core) >= 4:
        keys.add(("url_core", url_core))
        
    if len(content_n_tokens) >= 2:
        keys.add(("name_pair", tuple(sorted([content_n_tokens[0], content_n_tokens[1]]))))
        if len(content_n_tokens) >= 3:
            keys.add(("name_pair", tuple(sorted([content_n_tokens[0], content_n_tokens[2]]))))
            
    for t in content_n_tokens[:3]:
        keys.add(("name_tok", t))
        
    # Address Composite Keys
    if addr_nums and addr_words:
        for num in addr_nums[:2]:
            for w in addr_words[:3]:
                keys.add(("num_word", num, w))
                
    # Name + Address Composite Keys (Very high recall for noise in either field)
    if content_n_tokens and addr_nums:
        for nt in content_n_tokens[:2]:
            for num in addr_nums[:2]:
                keys.add(("name_num", nt, num))
                
    return keys

def run():
    print("=== HIGH-RECALL COMPOSITE BLOCKING EVALUATION ===")
    t0 = time.time()
    
    s1_path = os.path.join(REPO_ROOT, "dataset", "train", "train_source1.tsv")
    gt_path = os.path.join(REPO_ROOT, "dataset", "train", "train_ground_truth.tsv")
    s2_path = os.path.join(REPO_ROOT, "dataset", "train", "train_source2.tsv")
    s3_path = os.path.join(REPO_ROOT, "dataset", "train", "train_source3.tsv")
    
    NUM_S1 = 10000
    gt_df = pd.read_csv(gt_path, sep="\t", nrows=NUM_S1, dtype=str, keep_default_na=False)
    s1_df = pd.read_csv(s1_path, sep="\t", nrows=NUM_S1, dtype=str, keep_default_na=False)
    
    gt_dict = {}
    needed_s2, needed_s3 = set(), set()
    total_true = 0
    
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
        gt_dict[s1_id] = set(matches)
        total_true += len(matches)
        for m in matches:
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)
                
    print(f"Loaded {len(s1_df)} S1 entities with {total_true} true matches.")
    
    # Load candidate pool
    def load_pool(path, needed_set, pool_size=120000):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                row_dict = dict(zip(header, parts))
                eid = row_dict["entity_id"]
                if eid in needed_set or i < pool_size:
                    records.append(row_dict)
        return pd.DataFrame(records)
        
    print("Loading target pool (S2 & S3)...")
    s2_pool = load_pool(s2_path, needed_s2)
    s3_pool = load_pool(s3_path, needed_s3)
    print(f"Target pool sizes: S2={len(s2_pool)}, S3={len(s3_pool)} ({time.time()-t0:.2f}s)")
    
    # Build Inverted Indices
    print("Building composite inverted indices...")
    t1 = time.time()
    def build_idx(df):
        idx = defaultdict(list)
        for _, r in df.iterrows():
            keys = get_composite_keys(r["business_name"], r["business_address"], r.get("country", ""))
            eid = r["entity_id"]
            for k in keys:
                idx[k].append(eid)
        return idx
        
    idx2 = build_idx(s2_pool)
    idx3 = build_idx(s3_pool)
    print(f"Indices built in {time.time()-t1:.2f}s. Unique keys: S2={len(idx2):,}, S3={len(idx3):,}")
    
    # Test different max posting limits for individual name tokens vs composite keys
    # Composite keys (num_word, name_num, name_pair) are very specific, so higher posting cap is fine.
    # Single name_tok can be capped at 150 to keep candidates lean.
    caps = [
        {"name_tok": 150, "other": 300},
        {"name_tok": 300, "other": 500},
        {"name_tok": 500, "other": 1000},
    ]
    
    for cap_cfg in caps:
        name_tok_cap = cap_cfg["name_tok"]
        other_cap = cap_cfg["other"]
        
        t_eval = time.time()
        total_cands = 0
        found_true = 0
        
        for _, r in s1_df.iterrows():
            s1_id = r["entity_id"]
            true_matches = gt_dict.get(s1_id, set())
            keys = get_composite_keys(r["business_name"], r["business_address"], r.get("country", ""))
            
            cand_set = set()
            for k in keys:
                max_allowed = name_tok_cap if k[0] == "name_tok" else other_cap
                p2 = idx2.get(k, [])
                p3 = idx3.get(k, [])
                if len(p2) <= max_allowed:
                    cand_set.update(p2)
                if len(p3) <= max_allowed:
                    cand_set.update(p3)
                    
            total_cands += len(cand_set)
            for tm in true_matches:
                if tm in cand_set:
                    found_true += 1
                    
        recall = found_true / total_true if total_true else 0
        avg_cands = total_cands / len(s1_df)
        print(f"\n[CONFIG] name_tok_cap={name_tok_cap}, other_cap={other_cap}:")
        print(f"  Recall:               {recall*100:.2f}% ({found_true}/{total_true})")
        print(f"  Average Cands per S1: {avg_cands:.1f}")
        print(f"  Total Candidates:     {total_cands:,}")
        print(f"  Eval time:            {time.time()-t_eval:.2f}s")

if __name__ == "__main__":
    run()
