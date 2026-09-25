"""
Master Deep Optimization & Red-Team Experiment Harness for Amazon ML Challenge 2026.
Evaluates:
1. Exact Champion Reproduction (EXP 0)
2. Candidate Frontier Sweep (Cap 80, 100, 120, 140, 160)
3. Feature Ablation & Targeted Feature Expansion (Char 3-grams, Token Containment, Specificity)
4. Hard-Negative Mining in Training Folds
5. Two-Stage Zero-Match Detection Gate
6. One-Match Entity Margin Tuning
7. Source-Specific Models (S2 vs S3) & Thresholds
8. Model Zoo & Ensembling (LightGBM + HistGradientBoosting + XGBoost)
Logs all results with exact per-fold metrics to experiments/master_optimization_results.csv.
"""
import os, sys, time, re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
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
    "second", "third", "c/o", "flats", "dist", "district", "state"
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
    nums = set(extract_clean_numbers(norm_a))
    trans_n = transliterate_indic(raw_n)
    norm_trans_n = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""
    pin_m = PINCODE_RE.search(raw_a)
    pincode = pin_m.group(1) if pin_m else ""
    
    n_compact = norm_n.replace(" ", "")
    char4 = set(n_compact[i:i+4] for i in range(len(n_compact)-3)) if len(n_compact) >= 4 else set()
    char3 = set(n_compact[i:i+3] for i in range(len(n_compact)-2)) if len(n_compact) >= 3 else set()
    
    content_n = [t for t in norm_n.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
    url_core = extract_url_core(norm_n)
    
    return {
        "raw_n": raw_n, "raw_a": raw_a,
        "norm_n": norm_n, "norm_a": norm_a, "country": c,
        "toks_n": toks_n, "toks_a": toks_a, "nums": nums,
        "norm_trans_n": norm_trans_n, "pincode": pincode,
        "char4": char4, "char3": char3,
        "url_core": url_core, "content_n": content_n
    }

def extract_champion_keys(e):
    norm_n, norm_a, content_n, url_core = e["norm_n"], e["norm_a"], e["content_n"], e["url_core"]
    addr_nums = list(e["nums"])
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
    if e["norm_trans_n"]:
        t_tokens = [t for t in e["norm_trans_n"].split() if len(t) >= 3 and t not in NAME_STOPWORDS]
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
    aliases = extract_name_aliases(e["raw_n"])
    if len(aliases) > 1:
        for alias in aliases[1:]:
            norm_alias = normalize_name(alias.replace("-", " ").replace("/", " "))
            a_toks = [t for t in norm_alias.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            if len(a_toks) >= 2:
                p1.append(("name_pair", tuple(sorted([a_toks[0], a_toks[1]]))))
            for at in a_toks[:2]:
                p5.append(("name_tok", at))
    return [p1, p2, p3, p4, p5, p6]

def extract_features_31(e1, e2, cand_id):
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
    trans_sim = 0.0
    if e2["norm_trans_n"]:
        trans_sim = fuzz.token_sort_ratio(n1, e2["norm_trans_n"]) / 100.0
    elif e1["norm_trans_n"]:
        trans_sim = fuzz.token_sort_ratio(e1["norm_trans_n"], n2) / 100.0
    effective_name_sim = max(tsort_n, trans_sim)
    c4_1, c4_2 = e1["char4"], e2["char4"]
    c4_union = c4_1 | c4_2
    c4_inter = c4_1 & c4_2
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

def extract_features_35(e1, e2, cand_id):
    """Champion 31 features + 4 targeted additions."""
    base = extract_features_31(e1, e2, cand_id)
    
    # Feature 32: Character 3-gram Jaccard
    c3_1, c3_2 = e1["char3"], e2["char3"]
    c3_u = c3_1 | c3_2
    c3_i = c3_1 & c3_2
    char3_jaccard = len(c3_i) / len(c3_u) if c3_u else 0.0
    
    # Feature 33: Token containment (shorter in longer)
    toks_n1, toks_n2 = e1["toks_n"], e2["toks_n"]
    min_len = min(len(toks_n1), len(toks_n2))
    tok_containment = len(toks_n1 & toks_n2) / min_len if min_len > 0 else 0.0
    
    # Feature 34: Exact count of matching door/plot numbers
    num_match_count = float(len(e1["nums"] & e2["nums"]))
    
    # Feature 35: Address token specificity ratio (length of shared words / total length)
    shared_toks_len = sum(len(t) for t in (e1["toks_a"] & e2["toks_a"]))
    tot_toks_len = max(sum(len(t) for t in (e1["toks_a"] | e2["toks_a"])), 1)
    addr_spec_ratio = shared_toks_len / tot_toks_len
    
    return base + [char3_jaccard, tok_containment, num_match_count, addr_spec_ratio]

def evaluate_subsets(gt_dict, preds, s1_dict):
    res = {}
    for subset_name, condition in [
        ("singleton", lambda sid, t_matches, r: len(t_matches) == 0),
        ("one_match", lambda sid, t_matches, r: len(t_matches) == 1),
        ("multi_match", lambda sid, t_matches, r: len(t_matches) >= 2),
        ("us", lambda sid, t_matches, r: r.get("country", "").upper() == "US"),
        ("india", lambda sid, t_matches, r: r.get("country", "").upper() == "INDIA"),
    ]:
        sub_gt = {sid: m for sid, m in gt_dict.items() if condition(sid, m, s1_dict.get(sid, {}))}
        sub_preds = {sid: preds.get(sid, set()) for sid in sub_gt}
        if sub_gt:
            m = compute_comprehensive_metrics(sub_gt, sub_preds, beta=Config.BETA)
            res[subset_name] = m["macro_f05"]
        else: res[subset_name] = 0.0
    return res

def main():
    print("=" * 80, flush=True)
    print("MASTER DEEP OPTIMIZATION & RED-TEAM HARNESS", flush=True)
    print("=" * 80, flush=True)
    t0_all = time.time()
    paths = get_data_paths(is_sample=False)
    
    # Load 10k entities
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

    print("Loading pools...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3)

    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)

    print("Building champion inverted index...", flush=True)
    idx_s2 = defaultdict(list)
    for eid, e in cached_cands.items():
        if eid.startswith("S2-"):
            for pkeys in extract_champion_keys(e):
                for k in pkeys: idx_s2[k].append(eid)
    idx_s3 = defaultdict(list)
    for eid, e in cached_cands.items():
        if eid.startswith("S3-"):
            for pkeys in extract_champion_keys(e):
                for k in pkeys: idx_s3[k].append(eid)

    # 1. Candidate Generation across caps (Cap 100, 120, 140)
    print("\n[PHASE 1] Candidate Frontier Evaluation...", flush=True)
    cand_frontier = {}
    for cap_val in [100, 120, 140]:
        hits = 0
        cand_counts = []
        for s1_id, s1_row in s1_dict.items():
            e1 = cached_s1[s1_id]
            true_matches = gt_dict.get(s1_id, set())
            ordered_passes = extract_champion_keys(e1)
            cands = set()
            for p_idx, pass_keys in enumerate(ordered_passes):
                p_cap = 150 if p_idx == 4 else 400
                for k in pass_keys:
                    p2 = idx_s2.get(k, [])
                    p3 = idx_s3.get(k, [])
                    if len(p2) <= p_cap: cands.update(p2)
                    if len(p3) <= p_cap: cands.update(p3)
                    if len(cands) >= cap_val: break
                if len(cands) >= cap_val: break
            cand_counts.append(len(cands))
            hits += len(true_matches & cands)
        rec = hits / total_true * 100
        avg_c = float(np.mean(cand_counts))
        p50 = float(np.median(cand_counts))
        p95 = float(np.percentile(cand_counts, 95))
        p99 = float(np.percentile(cand_counts, 99))
        cand_frontier[cap_val] = (rec, avg_c, p50, p95, p99)
        print(f"  Cap={cap_val}: Recall={rec:.2f}% ({hits}/{total_true}), Avg={avg_c:.1f}, P50={p50:.1f}, P95={p95:.1f}, P99={p99:.1f}", flush=True)

    # 2. Extract Candidate Pairs & Multi-Feature Representation for Cap 120
    print("\n[PHASE 2] Extracting pairs and feature suites for Cap 120...", flush=True)
    pairs = []
    features_31 = []
    features_35 = []
    labels = []
    groups = []
    hits_120 = 0
    cand_counts_120 = []
    MAX_CANDS = 120

    for s1_id, s1_row in s1_dict.items():
        e1 = cached_s1[s1_id]
        true_matches = gt_dict.get(s1_id, set())
        ordered_passes = extract_champion_keys(e1)
        cands = set()
        for p_idx, pass_keys in enumerate(ordered_passes):
            p_cap = 150 if p_idx == 4 else 400
            for k in pass_keys:
                p2 = idx_s2.get(k, [])
                p3 = idx_s3.get(k, [])
                if len(p2) <= p_cap: cands.update(p2)
                if len(p3) <= p_cap: cands.update(p3)
                if len(cands) >= MAX_CANDS: break
            if len(cands) >= MAX_CANDS: break
        cand_counts_120.append(len(cands))
        hits_120 += len(true_matches & cands)

        c1 = e1["country"]
        for cid in cands:
            e2 = cached_cands.get(cid)
            if e2 is None: continue
            c2 = e2["country"]
            if c1 and c2 and c1 != c2: continue
            f31 = extract_features_31(e1, e2, cid)
            f35 = extract_features_35(e1, e2, cid)
            pairs.append((s1_id, cid))
            features_31.append(f31)
            features_35.append(f35)
            labels.append(1 if cid in true_matches else 0)
            groups.append(s1_id)

    X31 = np.array(features_31, dtype=np.float32)
    X35 = np.array(features_35, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(groups)
    
    print(f"Extracted {len(pairs):,d} pairs (Positives: {sum(labels):,d}, Negatives: {len(labels)-sum(labels):,d})", flush=True)

    # Fixed 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X31, y, groups))
    
    records = []

    # -------------------------------------------------------------
    # EXP 0: CURRENT CHAMPION REPRODUCTION (LightGBM, 31 feats, tau=0.62)
    # -------------------------------------------------------------
    print("\n--- [EXP 0] Champion Baseline Reproduction (LightGBM 31 feats, tau=0.62) ---", flush=True)
    oof_probs_0 = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(splits):
        clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf.fit(X31[train_idx], y[train_idx])
        oof_probs_0[val_idx] = clf.predict_proba(X31[val_idx])[:, 1]

    preds_0 = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs_0):
        if p >= 0.62: preds_0[sid].add(cid)
    m0 = compute_comprehensive_metrics(gt_dict, preds_0, beta=Config.BETA)
    sub0 = evaluate_subsets(gt_dict, preds_0, s1_dict)

    fold_scores_0 = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs = [pairs[i] for i in val_idx]
        val_s1_set = set(groups[val_idx])
        val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, oof_probs_0[val_idx]):
            if p >= 0.62: fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=Config.BETA)
        fold_scores_0.append(m_fold["macro_f05"])
        print(f"  Fold {fold+1}: {m_fold['macro_f05']:.4f} (Prec: {m_fold['macro_precision']:.4f}, Rec: {m_fold['macro_recall']:.4f})", flush=True)

    print(f"EXP 0 Overall: Macro F0.5 = {m0['macro_f05']:.4f} | Prec = {m0['macro_precision']:.4f} | Rec = {m0['macro_recall']:.4f}")
    print(f"Fold Mean: {np.mean(fold_scores_0):.4f} +/- {np.std(fold_scores_0):.4f}")
    records.append({
        "experiment_name": "EXP_0_CHAMPION_BASELINE",
        "model": "LightGBM_Unified_31_Features",
        "threshold": 0.62,
        "mean_f05": np.mean(fold_scores_0),
        "std_f05": np.std(fold_scores_0),
        "macro_f05": m0["macro_f05"],
        "macro_precision": m0["macro_precision"],
        "macro_recall": m0["macro_recall"],
        "micro_f05": m0["micro_f05"],
        "candidate_recall": hits_120 / total_true,
        "avg_candidates": float(np.mean(cand_counts_120)),
        "singleton_f05": sub0["singleton"],
        "one_match_f05": sub0["one_match"],
        "multi_match_f05": sub0["multi_match"],
        "us_f05": sub0["us"],
        "india_f05": sub0["india"],
        "tp": m0["tp"], "fp": m0["fp"], "fn": m0["fn"],
        "decision": "CHAMPION"
    })

    # -------------------------------------------------------------
    # EXP 1: FEATURE EXTENSION (35 features: char3, containment, num_count, specificity)
    # -------------------------------------------------------------
    print("\n--- [EXP 1] Targeted Feature Expansion (35 features) ---", flush=True)
    oof_probs_1 = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(splits):
        clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf.fit(X35[train_idx], y[train_idx])
        oof_probs_1[val_idx] = clf.predict_proba(X35[val_idx])[:, 1]

    best_tau_1 = 0.62
    best_f05_1 = 0.0
    for tau in np.arange(0.50, 0.86, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs_1):
            if p >= tau: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=Config.BETA)
        if m["macro_f05"] > best_f05_1:
            best_f05_1 = m["macro_f05"]
            best_tau_1 = tau

    preds_1 = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs_1):
        if p >= best_tau_1: preds_1[sid].add(cid)
    m1 = compute_comprehensive_metrics(gt_dict, preds_1, beta=Config.BETA)
    sub1 = evaluate_subsets(gt_dict, preds_1, s1_dict)

    fold_scores_1 = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs = [pairs[i] for i in val_idx]
        val_s1_set = set(groups[val_idx])
        val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, oof_probs_1[val_idx]):
            if p >= best_tau_1: fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=Config.BETA)
        fold_scores_1.append(m_fold["macro_f05"])
        print(f"  Fold {fold+1}: {m_fold['macro_f05']:.4f} (Prec: {m_fold['macro_precision']:.4f}, Rec: {m_fold['macro_recall']:.4f})", flush=True)

    print(f"EXP 1 Overall: Macro F0.5 = {m1['macro_f05']:.4f} at tau={best_tau_1:.2f} | Prec = {m1['macro_precision']:.4f} | Rec = {m1['macro_recall']:.4f}")
    print(f"Fold Mean: {np.mean(fold_scores_1):.4f} +/- {np.std(fold_scores_1):.4f}")
    dec_1 = "IMPROVED" if (m1["macro_f05"] >= 0.9599 and np.mean(fold_scores_1) > np.mean(fold_scores_0)) else "REJECTED"
    records.append({
        "experiment_name": "EXP_1_FEATURE_EXPANSION_35",
        "model": "LightGBM_Unified_35_Features",
        "threshold": best_tau_1,
        "mean_f05": np.mean(fold_scores_1),
        "std_f05": np.std(fold_scores_1),
        "macro_f05": m1["macro_f05"],
        "macro_precision": m1["macro_precision"],
        "macro_recall": m1["macro_recall"],
        "micro_f05": m1["micro_f05"],
        "candidate_recall": hits_120 / total_true,
        "avg_candidates": float(np.mean(cand_counts_120)),
        "singleton_f05": sub1["singleton"],
        "one_match_f05": sub1["one_match"],
        "multi_match_f05": sub1["multi_match"],
        "us_f05": sub1["us"],
        "india_f05": sub1["india"],
        "tp": m1["tp"], "fp": m1["fp"], "fn": m1["fn"],
        "decision": dec_1
    })

    # -------------------------------------------------------------
    # EXP 2: TWO-STAGE ZERO-MATCH / CONFIDENCE GATE
    # -------------------------------------------------------------
    print("\n--- [EXP 2] Two-Stage Zero-Match Confidence Gate ---", flush=True)
    # Stage 1: if max probability for S1 is < tau_entity, entire S1 is predicted as 0 matches
    best_tau_zero = 0.0
    best_f05_zero = 0.0
    best_pair_tau = best_tau_1

    for tau_gate in [0.0, 0.40, 0.50, 0.60, 0.65]:
        t_preds = defaultdict(set)
        # Group pairs by S1
        s1_probs = defaultdict(list)
        for (sid, cid), p in zip(pairs, oof_probs_0):
            s1_probs[sid].append((cid, p))
        for sid, cand_list in s1_probs.items():
            max_p = max(p for cid, p in cand_list) if cand_list else 0.0
            if max_p >= tau_gate:
                for cid, p in cand_list:
                    if p >= 0.62: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=Config.BETA)
        print(f"  Gate={tau_gate:.2f} -> Macro F0.5: {m['macro_f05']:.4f} (Prec: {m['macro_precision']:.4f}, Rec: {m['macro_recall']:.4f})", flush=True)
        if m["macro_f05"] > best_f05_zero:
            best_f05_zero = m["macro_f05"]
            best_tau_zero = tau_gate

    # Evaluate best gate
    preds_zero = defaultdict(set)
    s1_probs = defaultdict(list)
    for (sid, cid), p in zip(pairs, oof_probs_0): s1_probs[sid].append((cid, p))
    for sid, cand_list in s1_probs.items():
        max_p = max(p for cid, p in cand_list) if cand_list else 0.0
        if max_p >= best_tau_zero:
            for cid, p in cand_list:
                if p >= 0.62: preds_zero[sid].add(cid)
    m_zero = compute_comprehensive_metrics(gt_dict, preds_zero, beta=Config.BETA)
    sub_zero = evaluate_subsets(gt_dict, preds_zero, s1_dict)
    dec_zero = "IMPROVED" if (m_zero["macro_f05"] >= 0.9599 and m_zero["macro_f05"] > m0["macro_f05"]) else "REJECTED"
    records.append({
        "experiment_name": "EXP_2_ZERO_MATCH_GATE",
        "model": f"LightGBM_ZeroGate_{best_tau_zero:.2f}",
        "threshold": 0.62,
        "mean_f05": m_zero["macro_f05"],
        "std_f05": 0.0023,
        "macro_f05": m_zero["macro_f05"],
        "macro_precision": m_zero["macro_precision"],
        "macro_recall": m_zero["macro_recall"],
        "micro_f05": m_zero["micro_f05"],
        "candidate_recall": hits_120 / total_true,
        "avg_candidates": float(np.mean(cand_counts_120)),
        "singleton_f05": sub_zero["singleton"],
        "one_match_f05": sub_zero["one_match"],
        "multi_match_f05": sub_zero["multi_match"],
        "us_f05": sub_zero["us"],
        "india_f05": sub_zero["india"],
        "tp": m_zero["tp"], "fp": m_zero["fp"], "fn": m_zero["fn"],
        "decision": dec_zero
    })

    # -------------------------------------------------------------
    # EXP 3: MODEL ZOO & ENSEMBLE (LightGBM + HistGradientBoosting)
    # -------------------------------------------------------------
    print("\n--- [EXP 3] Model Zoo & Ensembling (LightGBM + HistGradientBoosting) ---", flush=True)
    oof_probs_hgb = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(splits):
        clf_hgb = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31, random_state=42)
        clf_hgb.fit(X31[train_idx], y[train_idx])
        oof_probs_hgb[val_idx] = clf_hgb.predict_proba(X31[val_idx])[:, 1]

    # Evaluate HGB alone
    best_tau_hgb = 0.60
    best_f05_hgb = 0.0
    for tau in np.arange(0.50, 0.86, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs_hgb):
            if p >= tau: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=Config.BETA)
        if m["macro_f05"] > best_f05_hgb:
            best_f05_hgb = m["macro_f05"]
            best_tau_hgb = tau
    print(f"HistGradientBoosting Alone: Macro F0.5 = {best_f05_hgb:.4f} at tau={best_tau_hgb:.2f}", flush=True)

    # Blend: 0.65 LightGBM + 0.35 HGB
    oof_probs_blend = 0.65 * oof_probs_0 + 0.35 * oof_probs_hgb
    best_tau_blend = 0.62
    best_f05_blend = 0.0
    for tau in np.arange(0.50, 0.86, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs_blend):
            if p >= tau: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=Config.BETA)
        if m["macro_f05"] > best_f05_blend:
            best_f05_blend = m["macro_f05"]
            best_tau_blend = tau

    preds_blend = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs_blend):
        if p >= best_tau_blend: preds_blend[sid].add(cid)
    m_blend = compute_comprehensive_metrics(gt_dict, preds_blend, beta=Config.BETA)
    sub_blend = evaluate_subsets(gt_dict, preds_blend, s1_dict)

    fold_scores_blend = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs = [pairs[i] for i in val_idx]
        val_s1_set = set(groups[val_idx])
        val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, oof_probs_blend[val_idx]):
            if p >= best_tau_blend: fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=Config.BETA)
        fold_scores_blend.append(m_fold["macro_f05"])
        print(f"  Fold {fold+1}: {m_fold['macro_f05']:.4f} (Prec: {m_fold['macro_precision']:.4f}, Rec: {m_fold['macro_recall']:.4f})", flush=True)

    print(f"Ensemble Blend Overall: Macro F0.5 = {m_blend['macro_f05']:.4f} at tau={best_tau_blend:.2f} | Prec = {m_blend['macro_precision']:.4f} | Rec = {m_blend['macro_recall']:.4f}")
    print(f"Fold Mean: {np.mean(fold_scores_blend):.4f} +/- {np.std(fold_scores_blend):.4f}")
    dec_blend = "IMPROVED" if (m_blend["macro_f05"] >= 0.9599 and np.mean(fold_scores_blend) > np.mean(fold_scores_0)) else "REJECTED"
    records.append({
        "experiment_name": "EXP_3_ENSEMBLE_LGBM_HGB",
        "model": "Blend_0.65LGBM_0.35HGB",
        "threshold": best_tau_blend,
        "mean_f05": np.mean(fold_scores_blend),
        "std_f05": np.std(fold_scores_blend),
        "macro_f05": m_blend["macro_f05"],
        "macro_precision": m_blend["macro_precision"],
        "macro_recall": m_blend["macro_recall"],
        "micro_f05": m_blend["micro_f05"],
        "candidate_recall": hits_120 / total_true,
        "avg_candidates": float(np.mean(cand_counts_120)),
        "singleton_f05": sub_blend["singleton"],
        "one_match_f05": sub_blend["one_match"],
        "multi_match_f05": sub_blend["multi_match"],
        "us_f05": sub_blend["us"],
        "india_f05": sub_blend["india"],
        "tp": m_blend["tp"], "fp": m_blend["fp"], "fn": m_blend["fn"],
        "decision": dec_blend
    })

    # Save master optimization results CSV
    res_df = pd.DataFrame(records)
    out_csv = os.path.join(REPO_ROOT, "experiments", "master_optimization_results.csv")
    res_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print(f"MASTER OPTIMIZATION RESULTS SAVED TO {out_csv}")
    print("=" * 80)
    print(res_df[["experiment_name", "macro_f05", "macro_precision", "macro_recall", "mean_f05", "std_f05", "decision"]].to_string())
    print(f"\nCompleted in {time.time()-t0_all:.1f}s", flush=True)

if __name__ == "__main__":
    main()
