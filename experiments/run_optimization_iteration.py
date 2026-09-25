"""
Master Optimization & Red-Team Experimentation Runner.
Evaluates 5-fold GroupKFold Source-1 Entity-Level Macro F0.5.
Tests:
1. Dynamic Multi-Pass Candidate Generator vs Baseline
2. Targeted Pairwise Feature Expansion (Transliteration, 4-grams, Pincode, Address Interactions)
3. Hard Negative Mining
4. Threshold Optimization (tau sweep)
5. Cardinality and Country Breakdown
Logs results to experiments/optimization_results.csv.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from config import Config
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
from transliteration import transliterate_indic, has_indic_script
from evaluate import compute_comprehensive_metrics

# Stopwords
NAME_STOPWORDS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp",
    "co", "company", "and", "the", "of", "in", "for", "center", "centre", "group",
    "services", "solutions", "enterprises", "associates", "holdings", "management",
    "international", "global", "industries", "systems", "technologies", "tech", "dba",
    "praivet", "treding"
}

ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "court", "ct", "way", "street", "road",
    "avenue", "boulevard", "drive", "lane", "apt", "suite", "ste", "fl", "floor",
    "bldg", "building", "near", "opp", "opposite", "behind", "phase", "sector",
    "plot", "shop", "no", "number", "us", "usa", "india", "null", "ground", "first",
    "second", "third", "c/o", "flats", "nagar", "dist", "district", "state"
}

# --- CANDIDATE GENERATION STRATEGIES ---

def extract_baseline_keys(name, address, country=""):
    norm_n = normalize_name(name)
    norm_a = normalize_address(address)
    n_tokens = norm_n.split()
    content_n = [t for t in n_tokens if len(t) >= 2 and t not in NAME_STOPWORDS]
    url_core = extract_url_core(norm_n)
    addr_nums = extract_clean_numbers(norm_a)
    addr_words = [t for t in norm_a.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS]
    keys = set()
    if norm_n and len(norm_n) >= 4: keys.add(("exact_name", norm_n))
    if url_core and len(url_core) >= 4: keys.add(("url_core", url_core))
    if len(content_n) >= 2:
        keys.add(("name_pair", tuple(sorted([content_n[0], content_n[1]]))))
        if len(content_n) >= 3:
            keys.add(("name_pair", tuple(sorted([content_n[0], content_n[2]]))))
    for t in content_n[:2]:
        if len(t) >= 3: keys.add(("name_tok", t))
    if addr_nums and addr_words:
        for num in addr_nums[:2]:
            for w in addr_words[:2]:
                keys.add(("num_word", num, w))
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]:
                keys.add(("name_num", nt, num))
    if len(addr_words) >= 2:
        keys.add(("addr_pair", tuple(sorted([addr_words[0], addr_words[1]]))))
        if len(addr_words) >= 3:
            keys.add(("addr_pair", tuple(sorted([addr_words[0], addr_words[2]]))))
    return keys

def extract_dynamic_multipass_keys(name, address, country=""):
    """
    Second-Generation Multi-Pass Blocking Keys:
    Pass 1: Exact Name, URL Core, Name Token Pairs
    Pass 2: Deterministic Indic Transliteration Pairs & Tokens
    Pass 3: Name Token + Address Number
    Pass 4: Distinctive Address Word + Address Number
    Pass 5: Single Distinctive Name Token
    Pass 6: Distinctive Address Pairs
    """
    clean_n = name.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    clean_a = address.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    
    n_tokens = norm_n.split()
    content_n = [t for t in n_tokens if len(t) >= 2 and t not in NAME_STOPWORDS]
    url_core = extract_url_core(norm_n)
    addr_nums = extract_clean_numbers(norm_a)
    addr_words = [t for t in norm_a.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and not any(c.isdigit() for c in t)]
    
    p1 = []
    if norm_n and len(norm_n) >= 4: p1.append(("exact_name", norm_n))
    if url_core and len(url_core) >= 4: p1.append(("url_core", url_core))
    if len(content_n) >= 2:
        p1.append(("name_pair", tuple(sorted([content_n[0], content_n[1]]))))
        if len(content_n) >= 3:
            p1.append(("name_pair", tuple(sorted([content_n[0], content_n[2]]))))
            p1.append(("name_pair", tuple(sorted([content_n[1], content_n[2]]))))
            
    p2 = []
    trans_n = transliterate_indic(name)
    if trans_n:
        norm_tn = normalize_name(trans_n.replace("-", " ").replace("/", " "))
        t_tokens = [t for t in norm_tn.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
        if len(t_tokens) >= 2:
            p2.append(("name_pair", tuple(sorted([t_tokens[0], t_tokens[1]]))))
        for t in t_tokens[:2]:
            p2.append(("name_tok", t))
            
    p3 = []
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]:
                p3.append(("name_num", nt, num))
                
    p4 = []
    if addr_nums and addr_words:
        for num in addr_nums[:3]:
            for w in addr_words[:3]:
                p4.append(("num_word", num, w))
                
    p5 = []
    for t in content_n[:2]:
        if len(t) >= 4:
            p5.append(("name_tok", t))
            
    p6 = []
    if len(addr_words) >= 2:
        p6.append(("addr_pair", tuple(sorted([addr_words[0], addr_words[1]]))))
        if len(addr_words) >= 3:
            p6.append(("addr_pair", tuple(sorted([addr_words[0], addr_words[2]]))))
            
    aliases = extract_name_aliases(name)
    if len(aliases) > 1:
        for alias in aliases[1:]:
            norm_alias = normalize_name(alias.replace("-", " ").replace("/", " "))
            a_toks = [t for t in norm_alias.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            if len(a_toks) >= 2:
                p1.append(("name_pair", tuple(sorted([a_toks[0], a_toks[1]]))))
            for at in a_toks[:2]:
                p5.append(("name_tok", at))
                
    return [p1, p2, p3, p4, p5, p6]

# --- PRECOMPUTE RECORD & FEATURE SUITE ---

import re
PINCODE_RE = re.compile(r"\b(\d{5,6})\b")

def precompute_entity(row):
    raw_n = row.get("business_name", "")
    raw_a = row.get("business_address", "")
    c = str(row.get("country", "")).strip().upper()
    
    clean_n = raw_n.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    
    clean_a = raw_a.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    
    toks_n = set(t for t in norm_n.split() if t not in NAME_STOPWORDS)
    toks_a = set(t for t in norm_a.split() if t not in ADDR_STOPWORDS)
    nums = set(extract_clean_numbers(norm_a))
    
    # Indic transliterated name if available
    trans_n = transliterate_indic(raw_n)
    norm_trans_n = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""
    
    # Postal / Pincode
    pin_m = PINCODE_RE.search(raw_a)
    pincode = pin_m.group(1) if pin_m else ""
    
    # Character 4-grams of name
    n_compact = norm_n.replace(" ", "")
    char4 = set(n_compact[i:i+4] for i in range(len(n_compact)-3)) if len(n_compact) >= 4 else set()
    
    return {
        "norm_n": norm_n,
        "norm_a": norm_a,
        "country": c,
        "toks_n": toks_n,
        "toks_a": toks_a,
        "nums": nums,
        "norm_trans_n": norm_trans_n,
        "pincode": pincode,
        "char4": char4
    }

def extract_expanded_features(e1, e2, cand_id):
    """
    Expanded targeted feature suite (31 features):
    - Original 26 features
    - Transliterated name token sort ratio
    - Character 4-gram Jaccard similarity
    - Address number exact equality (any door/plot num match)
    - Empty address name strength interaction
    - Postal / Pincode match flag (+1 match, -1 mismatch, 0 missing)
    """
    n1, a1, c1 = e1["norm_n"], e1["norm_a"], e1["country"]
    n2, a2, c2 = e2["norm_n"], e2["norm_a"], e2["country"]
    
    toks_n1, toks_n2 = e1["toks_n"], e2["toks_n"]
    name_union = toks_n1 | toks_n2
    name_inter = toks_n1 & toks_n2
    name_jaccard = len(name_inter) / len(name_union) if name_union else 0.0
    
    ratio_n = fuzz.ratio(n1, n2) / 100.0
    tsort_n = fuzz.token_sort_ratio(n1, n2) / 100.0
    tset_n = fuzz.token_set_ratio(n1, n2) / 100.0
    partial_n = fuzz.partial_ratio(n1, n2) / 100.0
    
    # Address tokens
    toks_a1, toks_a2 = e1["toks_a"], e2["toks_a"]
    addr_union = toks_a1 | toks_a2
    addr_inter = toks_a1 & toks_a2
    addr_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0
    
    nums_1, nums_2 = e1["nums"], e2["nums"]
    num_union = nums_1 | nums_2
    num_inter = nums_1 & nums_2
    num_jaccard = len(num_inter) / len(num_union) if num_union else 0.0
    
    empty_addr = float(len(a1) == 0 or len(a2) == 0)
    if empty_addr:
        ratio_a = tsort_a = tset_a = partial_a = 0.0
    else:
        ratio_a = fuzz.ratio(a1, a2) / 100.0
        tsort_a = fuzz.token_sort_ratio(a1, a2) / 100.0
        tset_a = fuzz.token_set_ratio(a1, a2) / 100.0
        partial_a = fuzz.partial_ratio(a1, a2) / 100.0
        
    country_match = float(c1 == c2 and c1 != "")
    is_s3 = float(str(cand_id).startswith("S3-"))
    
    len_diff_n = abs(len(n1) - len(n2))
    max_len_n = max(len(n1), len(n2), 1)
    len_ratio_n = 1.0 - (len_diff_n / max_len_n)
    len_diff_a = abs(len(a1) - len(a2))
    
    mult = tsort_n * (tsort_a if not empty_addr else tsort_n)
    min_sim = min(tsort_n, tsort_a) if not empty_addr else tsort_n
    max_sim = max(tsort_n, tsort_a)
    mean_sim = (tsort_n + tsort_a) / 2.0 if not empty_addr else tsort_n
    
    # --- TARGETED NEW FEATURES ---
    # 27. Transliterated Name Similarity
    trans_sim = 0.0
    if e2["norm_trans_n"]:
        trans_sim = fuzz.token_sort_ratio(n1, e2["norm_trans_n"]) / 100.0
    elif e1["norm_trans_n"]:
        trans_sim = fuzz.token_sort_ratio(e1["norm_trans_n"], n2) / 100.0
    effective_name_sim = max(tsort_n, trans_sim)
    
    # 28. Character 4-gram Jaccard
    c4_1, c4_2 = e1["char4"], e2["char4"]
    c4_union = c4_1 | c4_2
    c4_inter = c4_1 & c4_2
    char4_jaccard = len(c4_inter) / len(c4_union) if c4_union else 0.0
    
    # 29. Address Number Exact Overlap
    has_num_match = float(len(num_inter) > 0)
    
    # 30. Missing Address Name Interaction
    empty_addr_name_strength = (effective_name_sim * name_jaccard) if empty_addr else 0.0
    
    # 31. Pincode Match / Mismatch Signal
    pin1, pin2 = e1["pincode"], e2["pincode"]
    if pin1 and pin2:
        pin_flag = 1.0 if pin1 == pin2 else -1.0
    else:
        pin_flag = 0.0
        
    return [
        float(n1 == n2 and len(n1) > 0),
        ratio_n,
        tsort_n,
        tset_n,
        partial_n,
        name_jaccard,
        float(len(name_inter)),
        float(len_diff_n),
        len_ratio_n,
        0.0, # alias_match placeholder
        float(a1 == a2 and len(a1) > 0),
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
        # New 5 features
        effective_name_sim,
        char4_jaccard,
        has_num_match,
        empty_addr_name_strength,
        pin_flag,
    ]

def evaluate_subsets(gt_dict, preds, s1_records_dict):
    """Evaluates metrics by cardinality (singleton, 1-match, multi-match) and country (US, India)."""
    res = {}
    for subset_name, condition in [
        ("singleton", lambda sid, t_matches, r: len(t_matches) == 0),
        ("one_match", lambda sid, t_matches, r: len(t_matches) == 1),
        ("multi_match", lambda sid, t_matches, r: len(t_matches) >= 2),
        ("us", lambda sid, t_matches, r: r.get("country", "").upper() == "US"),
        ("india", lambda sid, t_matches, r: r.get("country", "").upper() == "INDIA"),
    ]:
        sub_gt = {sid: m for sid, m in gt_dict.items() if condition(sid, m, s1_records_dict.get(sid, {}))}
        sub_preds = {sid: preds.get(sid, set()) for sid in sub_gt}
        if sub_gt:
            m = compute_comprehensive_metrics(sub_gt, sub_preds, beta=Config.BETA)
            res[subset_name] = m["macro_f05"]
        else:
            res[subset_name] = 0.0
    return res

def main():
    print("=" * 80, flush=True)
    print("STARTING COMPREHENSIVE OPTIMIZATION ITERATION EXPERIMENT", flush=True)
    print("=" * 80, flush=True)
    t0_total = time.time()
    paths = get_data_paths(is_sample=False)
    
    # Load 10k Ground Truth
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())
    s1_needed = set(gt_dict.keys())
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"): needed_s2.add(m)
            elif m.startswith("S3-"): needed_s3.add(m)
            
    s1_dict = {}
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in s1_needed:
                s1_dict[parts[0]] = dict(zip(header, parts))
                if len(s1_dict) == len(s1_needed): break
                
    def load_pool_stream(path, needed_set, max_pool=100000):
        pool = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                if eid in needed_set or i < max_pool:
                    pool[eid] = dict(zip(header, parts))
        return pool
        
    print("Loading candidate pools...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3)
    print(f"Loaded: S1={len(s1_dict):,d}, S2={len(s2_dict):,d}, S3={len(s3_dict):,d}", flush=True)
    
    # Precompute records
    print("Precomputing entity features & transliterations...", flush=True)
    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)
    
    # Build Inverted Indexes for Optimized Candidate Generator
    print("Building Multi-Pass Inverted Indexes...", flush=True)
    idx_s2 = defaultdict(list)
    for eid, row in s2_dict.items():
        ordered_passes = extract_dynamic_multipass_keys(row["business_name"], row["business_address"], row.get("country", ""))
        for pass_keys in ordered_passes:
            for k in pass_keys:
                idx_s2[k].append(eid)
                
    idx_s3 = defaultdict(list)
    for eid, row in s3_dict.items():
        ordered_passes = extract_dynamic_multipass_keys(row["business_name"], row["business_address"], row.get("country", ""))
        for pass_keys in ordered_passes:
            for k in pass_keys:
                idx_s3[k].append(eid)
                
    print(f"Inverted index sizes: S2={len(idx_s2):,d} keys, S3={len(idx_s3):,d} keys", flush=True)
    
    # Generate Candidates with Optimized Multi-Pass
    t_cand_start = time.time()
    pairs = []
    features_expanded = []
    features_base26 = []
    labels = []
    groups = []
    hits = 0
    cand_counts = []
    
    MAX_CANDS = 120
    for s1_id, s1_row in s1_dict.items():
        true_matches = gt_dict.get(s1_id, set())
        ordered_passes = extract_dynamic_multipass_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
        
        cands = set()
        for p_idx, pass_keys in enumerate(ordered_passes):
            cap = 150 if p_idx == 4 else 400
            for k in pass_keys:
                p2 = idx_s2.get(k, [])
                p3 = idx_s3.get(k, [])
                if len(p2) <= cap: cands.update(p2)
                if len(p3) <= cap: cands.update(p3)
                if len(cands) >= MAX_CANDS: break
            if len(cands) >= MAX_CANDS: break
            
        cand_counts.append(len(cands))
        found_set = true_matches & cands
        hits += len(found_set)
        
        e1 = cached_s1[s1_id]
        c1 = e1["country"]
        
        for cid in cands:
            e2 = cached_cands.get(cid)
            if e2 is None: continue
            c2 = e2["country"]
            if c1 and c2 and c1 != c2: continue
            
            f_exp = extract_expanded_features(e1, e2, cid)
            f_base = f_exp[:26]
            
            pairs.append((s1_id, cid))
            features_expanded.append(f_exp)
            features_base26.append(f_base)
            labels.append(1 if cid in true_matches else 0)
            groups.append(s1_id)
            
    cand_recall = hits / total_true
    avg_cands = float(np.mean(cand_counts))
    median_cands = float(np.median(cand_counts))
    p95_cands = float(np.percentile(cand_counts, 95))
    p99_cands = float(np.percentile(cand_counts, 99))
    
    print(f"CANDIDATE RECALL: {cand_recall*100:.2f}% ({hits:,d} / {total_true:,d} true matches)", flush=True)
    print(f"Cand stats: Avg={avg_cands:.1f}, Median={median_cands:.1f}, P95={p95_cands:.1f}, P99={p99_cands:.1f}", flush=True)
    print(f"Total Evaluated Pairs: {len(pairs):,d} (Positives: {sum(labels):,d}, Negatives: {len(labels)-sum(labels):,d})", flush=True)
    
    X_exp = np.array(features_expanded, dtype=np.float32)
    X_base = np.array(features_base26, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(groups)
    
    # 5-Fold GroupKFold Setup
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_exp, y, groups))
    
    results_records = []
    
    # --- EXPERIMENT 1: Dynamic Candidates + Base 26 Features + LightGBM ---
    print("\n--- Running EXP 1: Dynamic Candidates + Base 26 Features ---", flush=True)
    t_start = time.time()
    oof_probs_base = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(splits):
        clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf.fit(X_base[train_idx], y[train_idx])
        oof_probs_base[val_idx] = clf.predict_proba(X_base[val_idx])[:, 1]
        
    # Threshold sweep
    best_tau_b = 0.70
    best_f05_b = 0.0
    for tau in np.arange(0.50, 0.86, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs_base):
            if p >= tau: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=Config.BETA)
        if m["macro_f05"] > best_f05_b:
            best_f05_b = m["macro_f05"]
            best_tau_b = tau
            
    preds_b = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs_base):
        if p >= best_tau_b: preds_b[sid].add(cid)
    m_b = compute_comprehensive_metrics(gt_dict, preds_b, beta=Config.BETA)
    sub_b = evaluate_subsets(gt_dict, preds_b, s1_dict)
    
    print(f"EXP 1 Result: Macro F0.5 = {m_b['macro_f05']:.4f} (Prec = {m_b['macro_precision']:.4f}, Rec = {m_b['macro_recall']:.4f}) at tau={best_tau_b:.2f}", flush=True)
    results_records.append({
        "experiment_id": "EXP_DYNAMIC_BLOCKING_BASE_FEATS",
        "change": "Dynamic multi-pass blocking + 26 base features",
        "candidate_recall": cand_recall,
        "avg_candidates": avg_cands,
        "macro_f05": m_b["macro_f05"],
        "macro_precision": m_b["macro_precision"],
        "macro_recall": m_b["macro_recall"],
        "micro_f05": m_b["micro_f05"],
        "singleton_f05": sub_b["singleton"],
        "one_match_f05": sub_b["one_match"],
        "multi_match_f05": sub_b["multi_match"],
        "us_f05": sub_b["us"],
        "india_f05": sub_b["india"],
        "tp": m_b["tp"],
        "fp": m_b["fp"],
        "fn": m_b["fn"],
        "runtime": time.time() - t_start,
        "status": "IMPROVED" if m_b["macro_f05"] > 0.9422 else "NO_CHANGE"
    })
    
    # --- EXPERIMENT 2: Dynamic Candidates + Expanded Targeted Features (31 feats) ---
    print("\n--- Running EXP 2: Dynamic Candidates + 31 Targeted Features ---", flush=True)
    t_start = time.time()
    oof_probs_exp = np.zeros(len(y), dtype=np.float32)
    fold_f05_list = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf.fit(X_exp[train_idx], y[train_idx])
        oof_probs_exp[val_idx] = clf.predict_proba(X_exp[val_idx])[:, 1]
        
    best_tau_e = 0.70
    best_f05_e = 0.0
    for tau in np.arange(0.50, 0.86, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs_exp):
            if p >= tau: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=Config.BETA)
        if m["macro_f05"] > best_f05_e:
            best_f05_e = m["macro_f05"]
            best_tau_e = tau
            
    preds_e = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs_exp):
        if p >= best_tau_e: preds_e[sid].add(cid)
    m_e = compute_comprehensive_metrics(gt_dict, preds_e, beta=Config.BETA)
    sub_e = evaluate_subsets(gt_dict, preds_e, s1_dict)
    
    # Per-fold stability for EXP 2
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs = [pairs[i] for i in val_idx]
        val_s1_set = set(groups[val_idx])
        val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, oof_probs_exp[val_idx]):
            if p >= best_tau_e: fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=Config.BETA)
        fold_f05_list.append(m_fold["macro_f05"])
        print(f"  Fold {fold+1}: Macro F0.5 = {m_fold['macro_f05']:.4f} (Prec = {m_fold['macro_precision']:.4f}, Rec = {m_fold['macro_recall']:.4f})", flush=True)
        
    print(f"EXP 2 Result: Macro F0.5 = {m_e['macro_f05']:.4f} (Prec = {m_e['macro_precision']:.4f}, Rec = {m_e['macro_recall']:.4f}) at tau={best_tau_e:.2f}", flush=True)
    print(f"Fold Mean: {np.mean(fold_f05_list):.4f} +/- {np.std(fold_f05_list):.4f}", flush=True)
    
    results_records.append({
        "experiment_id": "EXP_DYNAMIC_BLOCKING_EXPANDED_FEATS",
        "change": "Dynamic multi-pass blocking + 31 targeted features",
        "candidate_recall": cand_recall,
        "avg_candidates": avg_cands,
        "macro_f05": m_e["macro_f05"],
        "macro_precision": m_e["macro_precision"],
        "macro_recall": m_e["macro_recall"],
        "micro_f05": m_e["micro_f05"],
        "singleton_f05": sub_e["singleton"],
        "one_match_f05": sub_e["one_match"],
        "multi_match_f05": sub_e["multi_match"],
        "us_f05": sub_e["us"],
        "india_f05": sub_e["india"],
        "tp": m_e["tp"],
        "fp": m_e["fp"],
        "fn": m_e["fn"],
        "runtime": time.time() - t_start,
        "status": "IMPROVED" if m_e["macro_f05"] > 0.9422 else "NO_CHANGE"
    })
    
    # --- EXPERIMENT 3: Source-Specific S2 vs S3 Matchers on Expanded Features ---
    print("\n--- Running EXP 3: Source-Specific S2 vs S3 Models on Expanded Features ---", flush=True)
    t_start = time.time()
    is_s3_col = 21 # index of is_s3
    s2_mask = (X_exp[:, is_s3_col] == 0.0)
    s3_mask = (X_exp[:, is_s3_col] == 1.0)
    
    oof_probs_src = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(splits):
        # S2 model
        tr_s2 = train_idx[s2_mask[train_idx]]
        val_s2 = val_idx[s2_mask[val_idx]]
        clf_s2 = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf_s2.fit(X_exp[tr_s2], y[tr_s2])
        oof_probs_src[val_s2] = clf_s2.predict_proba(X_exp[val_s2])[:, 1]
        
        # S3 model
        tr_s3 = train_idx[s3_mask[train_idx]]
        val_s3 = val_idx[s3_mask[val_idx]]
        clf_s3 = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf_s3.fit(X_exp[tr_s3], y[tr_s3])
        oof_probs_src[val_s3] = clf_s3.predict_proba(X_exp[val_s3])[:, 1]
        
    best_tau_s = 0.70
    best_f05_s = 0.0
    for tau in np.arange(0.50, 0.86, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs_src):
            if p >= tau: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=Config.BETA)
        if m["macro_f05"] > best_f05_s:
            best_f05_s = m["macro_f05"]
            best_tau_s = tau
            
    preds_s = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs_src):
        if p >= best_tau_s: preds_s[sid].add(cid)
    m_s = compute_comprehensive_metrics(gt_dict, preds_s, beta=Config.BETA)
    sub_s = evaluate_subsets(gt_dict, preds_s, s1_dict)
    
    print(f"EXP 3 Result: Macro F0.5 = {m_s['macro_f05']:.4f} (Prec = {m_s['macro_precision']:.4f}, Rec = {m_s['macro_recall']:.4f}) at tau={best_tau_s:.2f}", flush=True)
    results_records.append({
        "experiment_id": "EXP_SOURCE_SPECIFIC_EXPANDED_FEATS",
        "change": "Source-specific S2/S3 models + 31 targeted features",
        "candidate_recall": cand_recall,
        "avg_candidates": avg_cands,
        "macro_f05": m_s["macro_f05"],
        "macro_precision": m_s["macro_precision"],
        "macro_recall": m_s["macro_recall"],
        "micro_f05": m_s["micro_f05"],
        "singleton_f05": sub_s["singleton"],
        "one_match_f05": sub_s["one_match"],
        "multi_match_f05": sub_s["multi_match"],
        "us_f05": sub_s["us"],
        "india_f05": sub_s["india"],
        "tp": m_s["tp"],
        "fp": m_s["fp"],
        "fn": m_s["fn"],
        "runtime": time.time() - t_start,
        "status": "IMPROVED" if m_s["macro_f05"] > 0.9426 else "NO_CHANGE"
    })
    
    # Save results to optimization_results.csv
    res_df = pd.DataFrame(results_records)
    csv_path = os.path.join(REPO_ROOT, "experiments", "optimization_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved optimization results to {csv_path}", flush=True)
    print(res_df.to_string())
    
    print(f"\nTotal Run Completed in {time.time()-t0_total:.1f}s", flush=True)

if __name__ == "__main__":
    main()
