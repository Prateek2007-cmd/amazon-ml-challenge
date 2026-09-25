"""
Candidate Generation and Blocking Experimentation for Amazon ML Challenge 2026.
Compares blocking strategies, measures Candidate Recall, Average Candidates per S1,
and analyzes missed true matches.
"""
import os
import sys
import time
import re
import unicodedata
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address, name_tokens, address_numeric_tokens

# ----------------- STOPWORDS & PATTERNS -----------------
COMMON_NAME_TERMS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp",
    "co", "company", "and", "the", "of", "in", "for", "center", "centre", "group",
    "services", "solutions", "enterprises", "associates", "holdings", "management",
    "international", "global", "industries", "systems", "technologies", "tech"
}

URL_REGEX = re.compile(r"([a-zA-Z0-9\-]+)\.(?:com|in|org|net|co|io|biz|info|gov|edu)", re.I)
PIN_INDIA_REGEX = re.compile(r"\b([1-9][0-9]{5})\b")
ZIP_US_REGEX = re.compile(r"\b([0-9]{5})(?:-[0-9]{4})?\b")

def clean_name_advanced(raw_name):
    norm = normalize_name(raw_name)
    # Check for URL patterns (e.g., maurewilliamscolombier.com)
    m = URL_REGEX.search(norm)
    url_core = m.group(1) if m else None
    return norm, url_core

def extract_address_keys(raw_addr, country):
    if not raw_addr:
        return []
    norm_addr = normalize_address(raw_addr)
    keys = []
    
    # Extract postal code if present
    c = str(country).strip().upper()
    if c == "INDIA":
        pins = PIN_INDIA_REGEX.findall(raw_addr)
        for p in pins:
            keys.append(("pin_in", p))
    elif c == "US":
        zips = ZIP_US_REGEX.findall(raw_addr)
        for z in zips:
            keys.append(("zip_us", z))
            
    # Extract numeric tokens
    nums = sorted(list(address_numeric_tokens(norm_addr)))
    # Leading house number / street number
    if nums:
        # If there are 2 or more numbers (e.g. house number + pin), strong key
        if len(nums) >= 2:
            keys.append(("num_pair", nums[0], nums[1]))
        keys.append(("num_lead", nums[0]))
        
    return keys

# ----------------- STRATEGY 1: BASELINE BLOCKING -----------------
def get_baseline_keys(norm_name):
    tokens = sorted(name_tokens(norm_name))
    keys = set()
    if tokens:
        keys.add(("tok0", tokens[0]))
        if len(tokens) >= 2:
            keys.add(("tok2", tokens[0], tokens[1]))
        prefix = "".join(tokens)[:3]
        if prefix:
            keys.add(("prefix3", prefix))
    return keys

# ----------------- STRATEGY 2: MULTI-STAGE BLOCKING -----------------
def get_multistage_keys(raw_name, raw_addr, country):
    norm_name, url_core = clean_name_advanced(raw_name)
    tokens = [t for t in norm_name.split() if len(t) >= 2]
    content_tokens = [t for t in tokens if t not in COMMON_NAME_TERMS]
    
    keys = set()
    c = str(country).strip().upper()
    
    # 1. Exact normalized name
    if norm_name:
        keys.add(("exact_name", norm_name))
        
    # 2. URL core domain
    if url_core and len(url_core) >= 3:
        keys.add(("url_core", url_core))
        
    # 3. Content tokens (first, second, and sorted 2-token signature)
    if content_tokens:
        keys.add(("c_tok0", content_tokens[0]))
        if len(content_tokens) >= 2:
            keys.add(("c_tok_pair", tuple(sorted([content_tokens[0], content_tokens[1]]))))
            keys.add(("c_tok1", content_tokens[1]))
    elif tokens:
        keys.add(("tok0", tokens[0]))
        if len(tokens) >= 2:
            keys.add(("tok_pair", tuple(sorted([tokens[0], tokens[1]]))))
            
    # 4. Character prefix (4 chars)
    condensed = "".join(content_tokens or tokens)
    if len(condensed) >= 4:
        keys.add(("prefix4", condensed[:4]))
        
    # 5. Combined Name Token + Postal / House Number (Ultra-high precision & recall!)
    addr_keys = extract_address_keys(raw_addr, country)
    top_name_tok = content_tokens[0] if content_tokens else (tokens[0] if tokens else None)
    
    for a_type, *a_vals in addr_keys:
        if a_type in ("pin_in", "zip_us"):
            keys.add((a_type, a_vals[0]))
            if top_name_tok:
                keys.add(("name_pin", top_name_tok, a_vals[0]))
        elif a_type == "num_pair" and top_name_tok:
            keys.add(("name_nums", top_name_tok, a_vals[0], a_vals[1]))
            
    return keys

# ----------------- RUN EXPERIMENT -----------------
def evaluate_blocking():
    print("=== LOADING VALIDATION EXPERIMENT SUBSET ===")
    t0 = time.time()
    
    # Read first 15,000 S1 entities
    s1_path = os.path.join(REPO_ROOT, "dataset", "train", "train_source1.tsv")
    gt_path = os.path.join(REPO_ROOT, "dataset", "train", "train_ground_truth.tsv")
    s2_path = os.path.join(REPO_ROOT, "dataset", "train", "train_source2.tsv")
    s3_path = os.path.join(REPO_ROOT, "dataset", "train", "train_source3.tsv")
    
    gt_df = pd.read_csv(gt_path, sep="\t", nrows=10000, dtype=str, keep_default_na=False)
    s1_df = pd.read_csv(s1_path, sep="\t", nrows=10000, dtype=str, keep_default_na=False)
    
    gt_dict = {}
    needed_s2 = set()
    needed_s3 = set()
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
    print(f"Needed target matches: {len(needed_s2)} S2, {len(needed_s3)} S3.")
    
    # Load 150,000 rows of S2 and S3, plus any true targets
    print("Loading S2 and S3 candidate pool (including all true targets)...")
    def load_pool(path, needed_set, pool_size=150000):
        records = []
        found_needed = set()
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                row_dict = dict(zip(header, parts))
                eid = row_dict["entity_id"]
                if eid in needed_set:
                    found_needed.add(eid)
                    records.append(row_dict)
                elif i < pool_size:
                    records.append(row_dict)
        return pd.DataFrame(records)
        
    s2_pool = load_pool(s2_path, needed_s2, 100000)
    s3_pool = load_pool(s3_path, needed_s3, 100000)
    print(f"Pool size: S2={len(s2_pool)}, S3={len(s3_pool)} (loaded in {time.time()-t0:.2f}s)")
    
    # ---------------- TEST 1: BASELINE BLOCKING ----------------
    print("\n--- [EXPERIMENT 1] BASELINE BLOCKING ---")
    t1 = time.time()
    def build_idx(df, key_fn):
        idx = defaultdict(list)
        for _, r in df.iterrows():
            keys = key_fn(r)
            for k in keys:
                idx[k].append(r["entity_id"])
        return idx
        
    idx2_base = build_idx(s2_pool, lambda r: get_baseline_keys(normalize_name(r["business_name"])))
    idx3_base = build_idx(s3_pool, lambda r: get_baseline_keys(normalize_name(r["business_name"])))
    
    # Max posting cutoff to avoid huge blowups
    MAX_POST = 150
    base_cands = {}
    base_cand_count = 0
    found_true_base = 0
    missed_base = []
    
    for _, r in s1_df.iterrows():
        s1_id = r["entity_id"]
        true_set = gt_dict.get(s1_id, set())
        keys = get_baseline_keys(normalize_name(r["business_name"]))
        cands = set()
        for k in keys:
            p2 = idx2_base.get(k, [])
            p3 = idx3_base.get(k, [])
            if len(p2) <= MAX_POST:
                cands.update(p2)
            if len(p3) <= MAX_POST:
                cands.update(p3)
        base_cands[s1_id] = cands
        base_cand_count += len(cands)
        for tm in true_set:
            if tm in cands:
                found_true_base += 1
            else:
                missed_base.append((s1_id, tm))
                
    base_recall = found_true_base / total_true if total_true else 0
    avg_base_cands = base_cand_count / len(s1_df)
    print(f"Baseline Recall:        {base_recall*100:.2f}% ({found_true_base}/{total_true})")
    print(f"Average Cands per S1:   {avg_base_cands:.2f}")
    print(f"Elapsed:                {time.time()-t1:.2f}s")
    
    # ---------------- TEST 2: MULTI-STAGE BLOCKING ----------------
    print("\n--- [EXPERIMENT 2] ADVANCED MULTI-STAGE BLOCKING ---")
    t2 = time.time()
    idx2_multi = build_idx(s2_pool, lambda r: get_multistage_keys(r["business_name"], r["business_address"], r.get("country", "")))
    idx3_multi = build_idx(s3_pool, lambda r: get_multistage_keys(r["business_name"], r["business_address"], r.get("country", "")))
    
    multi_cands = {}
    multi_cand_count = 0
    found_true_multi = 0
    missed_multi = []
    
    for _, r in s1_df.iterrows():
        s1_id = r["entity_id"]
        true_set = gt_dict.get(s1_id, set())
        keys = get_multistage_keys(r["business_name"], r["business_address"], r.get("country", ""))
        cands = set()
        for k in keys:
            p2 = idx2_multi.get(k, [])
            p3 = idx3_multi.get(k, [])
            # If key is pin-code alone, it has many records, so only include if pin + something or posting < 100
            if k[0] in ("pin_in", "zip_us") and len(p2) + len(p3) > 80:
                continue
            if len(p2) <= MAX_POST:
                cands.update(p2)
            if len(p3) <= MAX_POST:
                cands.update(p3)
        multi_cands[s1_id] = cands
        multi_cand_count += len(cands)
        for tm in true_set:
            if tm in cands:
                found_true_multi += 1
            else:
                missed_multi.append((s1_id, tm))
                
    multi_recall = found_true_multi / total_true if total_true else 0
    avg_multi_cands = multi_cand_count / len(s1_df)
    print(f"Multi-Stage Recall:     {multi_recall*100:.2f}% ({found_true_multi}/{total_true})")
    print(f"Average Cands per S1:   {avg_multi_cands:.2f}")
    print(f"Recall Gain:            +{(multi_recall - base_recall)*100:.2f}%")
    print(f"Elapsed:                {time.time()-t2:.2f}s")
    
    # Inspect a few missed pairs to see remaining gaps
    print("\n--- ANALYSIS OF MISSED PAIRS IN MULTI-STAGE ---")
    s1_lookup = s1_df.set_index("entity_id")
    s2_lookup = s2_pool.set_index("entity_id")
    s3_lookup = s3_pool.set_index("entity_id")
    
    for i, (s1_id, m_id) in enumerate(missed_multi[:5]):
        s1_row = s1_lookup.loc[s1_id] if s1_id in s1_lookup.index else {}
        t_row = s2_lookup.loc[m_id] if m_id in s2_lookup.index else (s3_lookup.loc[m_id] if m_id in s3_lookup.index else {})
        print(f"Missed Pair #{i+1}: S1={s1_id} -> {m_id}")
        print(f"  S1:   Name='{s1_row.get('business_name')}', Addr='{s1_row.get('business_address')}'")
        print(f"  Tgt:  Name='{t_row.get('business_name')}', Addr='{t_row.get('business_address')}'")

if __name__ == "__main__":
    evaluate_blocking()
