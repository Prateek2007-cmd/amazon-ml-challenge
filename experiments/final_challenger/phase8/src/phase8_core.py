"""
Phase 8: Master Optimization Engine — Pushing Business Entity Resolution to 0.985+ / 0.99
Amazon ML Challenge 2026 - Business Entity Resolution

Absolute Rules:
- Frozen champion (0.9579) is completely untouched.
- All code and artifacts live strictly inside experiments/final_challenger/phase8/
- No test inference, no external data, no test ground truth.
- 5-Fold GroupKFold CV by S1 entity.
- Metric: Entity-Level Macro F0.5.
"""
import os, sys, time, re, gc
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize
import lightgbm as lgb
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from config import Config
from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import (
    normalize_name, normalize_address, extract_name_aliases,
    extract_url_core, extract_clean_numbers,
)
from transliteration import transliterate_indic, has_indic_script
from evaluate import compute_comprehensive_metrics, compute_entity_metrics

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
    nums_list = extract_clean_numbers(norm_a)
    nums = set(nums_list)
    trans_n = transliterate_indic(raw_n)
    norm_trans_n = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""
    pin_m = PINCODE_RE.search(raw_a)
    pincode = pin_m.group(1) if pin_m else ""
    n_compact = norm_n.replace(" ", "")
    char4 = set(n_compact[i:i+4] for i in range(len(n_compact)-3)) if len(n_compact) >= 4 else set()
    url_core = extract_url_core(norm_n)
    return {
        "raw_n": raw_n, "raw_a": raw_a, "country": c,
        "norm_n": norm_n, "norm_a": norm_a,
        "toks_n": toks_n, "toks_a": toks_a,
        "nums": nums, "nums_list": nums_list,
        "norm_trans_n": norm_trans_n,
        "pincode": pincode, "char4": char4,
        "url_core": url_core,
    }

def extract_champion_keys(name, address, country=""):
    clean_n = name.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    clean_a = address.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    content_n = [t for t in norm_n.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
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
        if len(t_tokens) >= 2: p2.append(("name_pair", tuple(sorted([t_tokens[0], t_tokens[1]]))))
        for t in t_tokens[:2]: p2.append(("name_tok", t))
    p3 = []
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]: p3.append(("name_num", nt, num))
    p4 = []
    if addr_nums and addr_words:
        for num in addr_nums[:3]:
            for w in addr_words[:3]: p4.append(("num_word", num, w))
    p5 = []
    for t in content_n[:2]:
        if len(t) >= 4: p5.append(("name_tok", t))
    p6 = []
    if len(addr_words) >= 2:
        p6.append(("addr_pair", tuple(sorted([addr_words[0], addr_words[1]]))))
        if len(addr_words) >= 3: p6.append(("addr_pair", tuple(sorted([addr_words[0], addr_words[2]]))))
    aliases = extract_name_aliases(name)
    if len(aliases) > 1:
        for alias in aliases[1:]:
            norm_alias = normalize_name(alias.replace("-", " ").replace("/", " "))
            a_toks = [t for t in norm_alias.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            if len(a_toks) >= 2: p1.append(("name_pair", tuple(sorted([a_toks[0], a_toks[1]]))))
            for at in a_toks[:2]: p5.append(("name_tok", at))
    return [p1, p2, p3, p4, p5, p6]

def extract_features_31(e1, e2, cand_id):
    n1, a1, c1 = e1["norm_n"], e1["norm_a"], e1["country"]
    n2, a2, c2 = e2["norm_n"], e2["norm_a"], e2["country"]
    toks_n1, toks_n2 = e1["toks_n"], e2["toks_n"]
    toks_a1, toks_a2 = e1["toks_a"], e2["toks_a"]
    nums_1, nums_2 = e1["nums"], e2["nums"]
    name_union = toks_n1 | toks_n2
    name_inter = toks_n1 & toks_n2
    name_jaccard = len(name_inter) / len(name_union) if name_union else 0.0
    ratio_n = fuzz.ratio(n1, n2) / 100.0
    tsort_n = fuzz.token_sort_ratio(n1, n2) / 100.0
    tset_n = fuzz.token_set_ratio(n1, n2) / 100.0
    partial_n = fuzz.partial_ratio(n1, n2) / 100.0
    addr_union = toks_a1 | toks_a2
    addr_inter = toks_a1 & toks_a2
    addr_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0
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
    trans_sim = 0.0
    if e2["norm_trans_n"]:
        trans_sim = fuzz.token_sort_ratio(n1, e2["norm_trans_n"]) / 100.0
    elif e1["norm_trans_n"]:
        trans_sim = fuzz.token_sort_ratio(e1["norm_trans_n"], n2) / 100.0
    effective_name_sim = max(tsort_n, trans_sim)
    c4_union = e1["char4"] | e2["char4"]
    c4_inter = e1["char4"] & e2["char4"]
    char4_jaccard = len(c4_inter) / len(c4_union) if c4_union else 0.0
    has_num_match = float(len(num_inter) > 0)
    empty_addr_name_strength = (effective_name_sim * name_jaccard) if empty_addr else 0.0
    pin1, pin2 = e1["pincode"], e2["pincode"]
    pin_flag = 1.0 if (pin1 and pin2 and pin1 == pin2) else (-1.0 if (pin1 and pin2 and pin1 != pin2) else 0.0)
    return [
        float(n1 == n2 and len(n1) > 0), ratio_n, tsort_n, tset_n, partial_n,
        name_jaccard, float(len(name_inter)), float(len_diff_n), len_ratio_n, 0.0,
        float(a1 == a2 and len(a1) > 0), ratio_a, tsort_a, tset_a, partial_a,
        addr_jaccard, num_jaccard, float(len(num_inter)), float(len_diff_a), empty_addr,
        country_match, is_s3, mult, min_sim, max_sim, mean_sim,
        effective_name_sim, char4_jaccard, has_num_match, empty_addr_name_strength, pin_flag
    ]

def build_rep_text(row):
    name = row.get("business_name", "")
    addr = row.get("business_address", "")
    norm_n = normalize_name(name.replace("-", " ").replace("/", " "))
    norm_a = normalize_address(addr.replace("-", " ").replace("/", " "))
    return f"{norm_n} {norm_a}"

def precompute_semantic_index(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries):
    print("Building country-partitioned TF-IDF semantic indexes...", flush=True)
    all_cand_ids = list(s2_dict.keys()) + list(s3_dict.keys())
    country_to_cand_ids = defaultdict(list)
    for eid in all_cand_ids:
        country_to_cand_ids[cand_countries[eid]].append(eid)
    sem_top_records = defaultdict(dict)
    K_MAX = 30
    COS_MIN = 0.20
    for country, c_cand_ids in country_to_cand_ids.items():
        if len(c_cand_ids) < 5: continue
        c_s1_ids = [sid for sid in s1_list if s1_countries[sid] == country]
        if not c_s1_ids: continue
        cand_corpus = [build_rep_text(s2_dict[eid] if eid.startswith("S2-") else s3_dict[eid]) for eid in c_cand_ids]
        s1_corpus = [build_rep_text(s1_dict[sid]) for sid in c_s1_ids]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
        vectorizer.fit(cand_corpus + s1_corpus)
        X_cand = sklearn_normalize(vectorizer.transform(cand_corpus), norm='l2')
        X_s1 = sklearn_normalize(vectorizer.transform(s1_corpus), norm='l2')
        CHUNK = 500
        for start in range(0, len(c_s1_ids), CHUNK):
            end = min(start + CHUNK, len(c_s1_ids))
            chunk_ids = c_s1_ids[start:end]
            sim_matrix = (X_s1[start:end] @ X_cand.T).toarray()
            for local_i, s1_id in enumerate(chunk_ids):
                sims = sim_matrix[local_i]
                if len(sims) > K_MAX:
                    top_idx = np.argpartition(sims, -K_MAX)[-K_MAX:]
                else:
                    top_idx = np.arange(len(sims))
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                for rank_0, idx in enumerate(top_idx[:K_MAX]):
                    score = float(sims[idx])
                    if score >= COS_MIN:
                        cid = c_cand_ids[idx]
                        sem_top_records[s1_id][cid] = (score, rank_0 + 1)
        del X_cand, X_s1, sim_matrix, vectorizer
        gc.collect()
    print(f"Precomputed semantic retrieval for {len(sem_top_records):,d} S1 entities.", flush=True)
    return sem_top_records

def evaluate_comprehensive_full(gt_dict, preds, s1_dict, splits, pairs, groups_arr, oof_probs, best_tau):
    m_all = compute_comprehensive_metrics(gt_dict, preds, beta=0.5)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs = [pairs[i] for i in val_idx]
        val_s1_set = set(groups_arr[val_idx])
        val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, oof_probs[val_idx]):
            if p >= best_tau: fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=0.5)
        fold_scores.append(m_fold["macro_f05"])

    zero_gt = {sid: m for sid, m in gt_dict.items() if len(m) == 0}
    zero_preds = {sid: preds.get(sid, set()) for sid in zero_gt}
    m_zero = compute_comprehensive_metrics(zero_gt, zero_preds, beta=0.5)

    one_gt = {sid: m for sid, m in gt_dict.items() if len(m) == 1}
    one_preds = {sid: preds.get(sid, set()) for sid in one_gt}
    m_one = compute_comprehensive_metrics(one_gt, one_preds, beta=0.5)

    multi_gt = {sid: m for sid, m in gt_dict.items() if len(m) >= 2}
    multi_preds = {sid: preds.get(sid, set()) for sid in multi_gt}
    m_multi = compute_comprehensive_metrics(multi_gt, multi_preds, beta=0.5)

    us_gt = {sid: m for sid, m in gt_dict.items() if s1_dict[sid].get("country", "").upper() == "US"}
    us_preds = {sid: preds.get(sid, set()) for sid in us_gt}
    m_us = compute_comprehensive_metrics(us_gt, us_preds, beta=0.5)

    india_gt = {sid: m for sid, m in gt_dict.items() if s1_dict[sid].get("country", "").upper() == "INDIA"}
    india_preds = {sid: preds.get(sid, set()) for sid in india_gt}
    m_india = compute_comprehensive_metrics(india_gt, india_preds, beta=0.5)

    s2_gt = {sid: {m for m in true_m if m.startswith("S2-")} for sid, true_m in gt_dict.items()}
    s2_preds = {sid: {m for m in preds.get(sid, set()) if m.startswith("S2-")} for sid in gt_dict}
    m_s2 = compute_comprehensive_metrics(s2_gt, s2_preds, beta=0.5)

    s3_gt = {sid: {m for m in true_m if m.startswith("S3-")} for sid, true_m in gt_dict.items()}
    s3_preds = {sid: {m for m in preds.get(sid, set()) if m.startswith("S3-")} for sid in gt_dict}
    m_s3 = compute_comprehensive_metrics(s3_gt, s3_preds, beta=0.5)

    return {
        "macro_f05": m_all["macro_f05"],
        "precision": m_all["macro_precision"],
        "recall": m_all["macro_recall"],
        "tp": m_all["tp"],
        "fp": m_all["fp"],
        "fn": m_all["fn"],
        "zero": m_zero["macro_f05"],
        "one": m_one["macro_f05"],
        "multi": m_multi["macro_f05"],
        "us": m_us["macro_f05"],
        "india": m_india["macro_f05"],
        "s2": m_s2["macro_f05"],
        "s3": m_s3["macro_f05"],
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
    }
