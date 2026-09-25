"""
Production Pipeline: Semantic-Augmented Entity Resolution.
Integrates TF-IDF char-ngram candidate recovery (Config C: zero-protected)
into the champion pipeline for final submission generation.

Architecture:
  1. Train on 10K entities with Champion UNION Semantic(C_norm_name_addr, K=5) candidates
  2. Infer on full test set with same augmented candidate generation
  3. Write matching_results.tsv and candidate_pairs.tsv
  4. Validate and package

DOES NOT modify the frozen champion files.
Outputs to: experiments/semantic_candidate_recovery/output/
"""
import os, sys, time, re, hashlib, zipfile, shutil
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize
import lightgbm as lgb
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from config import Config
from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import (
    normalize_name, normalize_address, extract_name_aliases,
    extract_url_core, extract_clean_numbers,
)
from transliteration import transliterate_indic
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
    nums = set(extract_clean_numbers(norm_a))
    trans_n = transliterate_indic(raw_n)
    norm_trans_n = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""
    pin_m = PINCODE_RE.search(raw_a)
    pincode = pin_m.group(1) if pin_m else ""
    n_compact = norm_n.replace(" ", "")
    char4 = set(n_compact[i:i+4] for i in range(len(n_compact)-3)) if len(n_compact) >= 4 else set()
    content_n = [t for t in norm_n.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
    url_core = extract_url_core(norm_n)
    return {
        "raw_n": raw_n, "raw_a": raw_a, "norm_n": norm_n, "norm_a": norm_a,
        "country": c, "toks_n": toks_n, "toks_a": toks_a, "nums": nums,
        "norm_trans_n": norm_trans_n, "pincode": pincode, "char4": char4,
        "url_core": url_core, "content_n": content_n
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
        if len(t) >= 4: p5.append(("name_tok", t))
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

def build_rep_text(row):
    """C_norm_name_addr representation."""
    name = row.get("business_name", "")
    addr = row.get("business_address", "")
    norm_n = normalize_name(name.replace("-", " ").replace("/", " "))
    norm_a = normalize_address(addr.replace("-", " ").replace("/", " "))
    return f"{norm_n} {norm_a}"

def build_blocking_index(records_dict):
    idx = defaultdict(list)
    for eid, row in records_dict.items():
        for pass_keys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pass_keys:
                idx[k].append(eid)
    return idx

def generate_champion_candidates(s1_id, s1_row, idx_s2, idx_s3, max_cands=120):
    ordered_passes = extract_champion_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
    cands = set()
    for p_idx, pass_keys in enumerate(ordered_passes):
        cap = 150 if p_idx == 4 else 400
        for k in pass_keys:
            p2 = idx_s2.get(k, [])
            p3 = idx_s3.get(k, [])
            if len(p2) <= cap: cands.update(p2)
            if len(p3) <= cap: cands.update(p3)
            if len(cands) >= max_cands: break
        if len(cands) >= max_cands: break
    return cands

def semantic_retrieve_batch(query_ids, query_rows, cand_ids, cand_rows, K=5, cos_threshold=0.30):
    """Retrieve top-K candidates per query using TF-IDF char-ngram similarity, partitioned by country."""
    # Partition candidates by country
    country_to_cand = defaultdict(list)
    for eid in cand_ids:
        c = str(cand_rows[eid].get("country", "")).strip().upper()
        country_to_cand[c].append(eid)

    query_countries = {qid: str(query_rows[qid].get("country", "")).strip().upper() for qid in query_ids}

    results = defaultdict(set)
    for country, c_cand_ids in country_to_cand.items():
        if len(c_cand_ids) < 5:
            continue
        c_query_ids = [qid for qid in query_ids if query_countries[qid] == country]
        if not c_query_ids:
            continue

        print(f"    Semantic retrieval [{country}]: {len(c_query_ids):,d} queries x {len(c_cand_ids):,d} candidates", flush=True)

        cand_corpus = [build_rep_text(cand_rows[eid]) for eid in c_cand_ids]
        query_corpus = [build_rep_text(query_rows[qid]) for qid in c_query_ids]

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
        vectorizer.fit(cand_corpus + query_corpus[:5000])  # fit on cands + sample of queries
        X_cand = sklearn_normalize(vectorizer.transform(cand_corpus), norm='l2')
        X_q = sklearn_normalize(vectorizer.transform(query_corpus), norm='l2')

        CHUNK = 500
        for start in range(0, len(c_query_ids), CHUNK):
            end = min(start + CHUNK, len(c_query_ids))
            chunk_ids = c_query_ids[start:end]
            sim_matrix = (X_q[start:end] @ X_cand.T).toarray()
            for local_i, qid in enumerate(chunk_ids):
                sims = sim_matrix[local_i]
                if len(sims) > K:
                    top_idx = np.argpartition(sims, -K)[-K:]
                else:
                    top_idx = np.arange(len(sims))
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                for idx in top_idx[:K]:
                    if sims[idx] >= cos_threshold:
                        results[qid].add(c_cand_ids[idx])
    return results

def main():
    print("=" * 80)
    print("PRODUCTION PIPELINE: SEMANTIC-AUGMENTED ENTITY RESOLUTION")
    print("Config C: Champion + C_norm_name_addr K=5 (zero-protected)")
    print("=" * 80)
    t0_total = time.time()
    paths = get_data_paths(is_sample=False)

    EXP_OUTPUT = os.path.join(REPO_ROOT, "experiments", "semantic_candidate_recovery", "output")
    os.makedirs(EXP_OUTPUT, exist_ok=True)

    # ===================================================================
    # PHASE 1: TRAINING
    # ===================================================================
    print("\n[PHASE 1] Loading Training Data...", flush=True)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())
    s1_needed = set(gt_dict.keys())
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"): needed_s2.add(m)
            elif m.startswith("S3-"): needed_s3.add(m)

    s1_train = {}
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in s1_needed:
                s1_train[parts[0]] = dict(zip(header, parts))
                if len(s1_train) == len(s1_needed): break

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

    s2_train = load_pool_stream(paths["train_s2"], needed_s2)
    s3_train = load_pool_stream(paths["train_s3"], needed_s3)
    print(f"  Loaded: S1={len(s1_train):,d}, S2={len(s2_train):,d}, S3={len(s3_train):,d}", flush=True)

    # Precompute
    cached_s1_train = {sid: precompute_entity(row) for sid, row in s1_train.items()}
    cached_cands_train = {}
    for eid, row in s2_train.items(): cached_cands_train[eid] = precompute_entity(row)
    for eid, row in s3_train.items(): cached_cands_train[eid] = precompute_entity(row)

    # Champion candidates
    print("[PHASE 1b] Building champion blocking index (train)...", flush=True)
    idx_s2_train = build_blocking_index(s2_train)
    idx_s3_train = build_blocking_index(s3_train)

    s1_train_list = list(s1_train.keys())
    champion_cands_train = {}
    for s1_id in s1_train_list:
        champion_cands_train[s1_id] = generate_champion_candidates(s1_id, s1_train[s1_id], idx_s2_train, idx_s3_train)

    champ_hits = sum(len(gt_dict.get(sid, set()) & champion_cands_train[sid]) for sid in s1_train_list)
    print(f"  Champion train cand recall: {champ_hits/total_true*100:.2f}%", flush=True)

    # Semantic retrieval for training
    print("[PHASE 1c] Semantic retrieval (train)...", flush=True)
    # Only retrieve for entities with >=1 champion candidate (zero-protected)
    query_ids_train = [sid for sid in s1_train_list if len(champion_cands_train[sid]) > 0]
    print(f"  Queries (zero-protected): {len(query_ids_train):,d} / {len(s1_train_list):,d}", flush=True)

    all_cand_train = {**s2_train, **s3_train}
    sem_cands_train = semantic_retrieve_batch(
        query_ids_train, s1_train, list(all_cand_train.keys()), all_cand_train, K=5
    )

    # Union candidates
    union_cands_train = {}
    for sid in s1_train_list:
        if len(champion_cands_train[sid]) > 0:
            union_cands_train[sid] = champion_cands_train[sid] | sem_cands_train.get(sid, set())
        else:
            union_cands_train[sid] = champion_cands_train[sid]

    union_hits = sum(len(gt_dict.get(sid, set()) & union_cands_train[sid]) for sid in s1_train_list)
    print(f"  Union train cand recall: {union_hits/total_true*100:.2f}%", flush=True)

    # Generate pairs and features
    print("[PHASE 1d] Generating training pairs + features...", flush=True)
    pairs = []
    features = []
    labels = []
    groups = []
    for s1_id in s1_train_list:
        true_matches = gt_dict.get(s1_id, set())
        cands = union_cands_train[s1_id]
        e1 = cached_s1_train[s1_id]
        c1 = e1["country"]
        for cid in cands:
            e2 = cached_cands_train.get(cid)
            if e2 is None: continue
            if c1 and e2["country"] and c1 != e2["country"]: continue
            f_vec = extract_features_31(e1, e2, cid)
            pairs.append((s1_id, cid))
            features.append(f_vec)
            labels.append(1 if cid in true_matches else 0)
            groups.append(s1_id)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups_arr = np.array(groups)
    print(f"  Pairs: {len(pairs):,d} (Pos:{sum(labels):,d}, Neg:{len(labels)-sum(labels):,d})", flush=True)

    # 5-fold CV to verify and find optimal threshold
    print("[PHASE 1e] 5-Fold GroupKFold CV...", flush=True)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups_arr))
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(splits):
        clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    # Threshold sweep
    best_tau = 0.68
    best_f05 = 0.0
    for tau in np.arange(0.50, 0.82, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs):
            if p >= tau: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=0.5)
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_tau = tau

    preds_cv = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs):
        if p >= best_tau: preds_cv[sid].add(cid)
    m_cv = compute_comprehensive_metrics(gt_dict, preds_cv, beta=0.5)

    # Per-fold scores
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs = [pairs[i] for i in val_idx]
        val_s1_set = set(groups_arr[val_idx])
        val_gt = {sid: gt_dict[sid] for sid in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, oof_probs[val_idx]):
            if p >= best_tau: fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=0.5)
        fold_scores.append(m_fold["macro_f05"])

    # Cardinality
    zero_gt = {sid: m for sid, m in gt_dict.items() if len(m) == 0}
    zero_p = {sid: preds_cv.get(sid, set()) for sid in zero_gt}
    m_zero = compute_comprehensive_metrics(zero_gt, zero_p, beta=0.5)
    one_gt = {sid: m for sid, m in gt_dict.items() if len(m) == 1}
    one_p = {sid: preds_cv.get(sid, set()) for sid in one_gt}
    m_one = compute_comprehensive_metrics(one_gt, one_p, beta=0.5)
    multi_gt = {sid: m for sid, m in gt_dict.items() if len(m) >= 2}
    multi_p = {sid: preds_cv.get(sid, set()) for sid in multi_gt}
    m_multi = compute_comprehensive_metrics(multi_gt, multi_p, beta=0.5)

    print(f"\n  === CV RESULTS (Semantic-Augmented Config C) ===", flush=True)
    print(f"  Macro F0.5 = {m_cv['macro_f05']:.4f} (Prec={m_cv['macro_precision']:.4f}, Rec={m_cv['macro_recall']:.4f}) tau={best_tau:.2f}", flush=True)
    print(f"  Folds: {' '.join(f'{s:.4f}' for s in fold_scores)} | Mean={np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}", flush=True)
    print(f"  Zero={m_zero['macro_f05']:.4f} One={m_one['macro_f05']:.4f} Multi={m_multi['macro_f05']:.4f}", flush=True)
    print(f"  TP={m_cv['tp']:,d} FP={m_cv['fp']:,d} FN={m_cv['fn']:,d}", flush=True)

    # Train final model on ALL data
    print("\n[PHASE 1f] Training final model on all data...", flush=True)
    final_clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
    final_clf.fit(X, y)
    print("  Final model trained.", flush=True)

    # ===================================================================
    # PHASE 2: TEST SET INFERENCE
    # ===================================================================
    print("\n[PHASE 2] Loading Test Data...", flush=True)

    # Load test S1 (stream all)
    test_s1_ids = []
    test_s1_dict = {}
    with open(paths["test_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid = parts[0]
            test_s1_ids.append(eid)
            test_s1_dict[eid] = dict(zip(header, parts))
    print(f"  Test S1: {len(test_s1_ids):,d}", flush=True)

    # Load test S2 and S3
    def load_full_pool(path):
        pool = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                pool[parts[0]] = dict(zip(header, parts))
        return pool

    print("  Loading test S2...", flush=True)
    test_s2 = load_full_pool(paths["test_s2"])
    print(f"  Test S2: {len(test_s2):,d}", flush=True)
    print("  Loading test S3...", flush=True)
    test_s3 = load_full_pool(paths["test_s3"])
    print(f"  Test S3: {len(test_s3):,d}", flush=True)

    # Precompute test candidates
    print("  Precomputing test candidate representations...", flush=True)
    cached_cands_test = {}
    for eid, row in test_s2.items(): cached_cands_test[eid] = precompute_entity(row)
    for eid, row in test_s3.items(): cached_cands_test[eid] = precompute_entity(row)
    print(f"  Precomputed {len(cached_cands_test):,d} test candidates.", flush=True)

    # Build blocking index
    print("[PHASE 2b] Building champion blocking index (test)...", flush=True)
    idx_s2_test = build_blocking_index(test_s2)
    idx_s3_test = build_blocking_index(test_s3)
    print(f"  Index: S2={len(idx_s2_test):,d} keys, S3={len(idx_s3_test):,d} keys", flush=True)

    # Champion candidates for test
    print("[PHASE 2c] Generating champion candidates for test...", flush=True)
    champion_cands_test = {}
    for i, s1_id in enumerate(test_s1_ids):
        champion_cands_test[s1_id] = generate_champion_candidates(s1_id, test_s1_dict[s1_id], idx_s2_test, idx_s3_test)
        if (i + 1) % 200000 == 0:
            print(f"    {i+1:,d}/{len(test_s1_ids):,d}...", flush=True)

    entities_with_cands = sum(1 for c in champion_cands_test.values() if len(c) > 0)
    print(f"  Entities with >=1 champion candidate: {entities_with_cands:,d}", flush=True)

    # Semantic retrieval for test (zero-protected: only for entities with champion candidates)
    print("[PHASE 2d] Semantic retrieval for test entities with candidates...", flush=True)
    query_ids_test = [sid for sid in test_s1_ids if len(champion_cands_test[sid]) > 0]
    print(f"  Semantic queries: {len(query_ids_test):,d}", flush=True)

    all_cand_test = {**test_s2, **test_s3}
    sem_cands_test = semantic_retrieve_batch(
        query_ids_test, test_s1_dict, list(all_cand_test.keys()), all_cand_test, K=5
    )

    # Union candidates
    union_cands_test = {}
    for s1_id in test_s1_ids:
        if len(champion_cands_test[s1_id]) > 0:
            union_cands_test[s1_id] = champion_cands_test[s1_id] | sem_cands_test.get(s1_id, set())
        else:
            union_cands_test[s1_id] = champion_cands_test[s1_id]

    # Inference
    print("[PHASE 2e] Running LightGBM inference...", flush=True)
    final_matches = defaultdict(set)
    final_candidates = {}
    total_scored = 0

    INFER_CHUNK = 50000
    for chunk_start in range(0, len(test_s1_ids), INFER_CHUNK):
        chunk_end = min(chunk_start + INFER_CHUNK, len(test_s1_ids))
        chunk_ids = test_s1_ids[chunk_start:chunk_end]

        chunk_pairs = []
        chunk_feats = []

        for s1_id in chunk_ids:
            cands = union_cands_test[s1_id]
            final_candidates[s1_id] = cands
            e1 = precompute_entity(test_s1_dict[s1_id])
            c1 = e1["country"]

            for cid in cands:
                e2 = cached_cands_test.get(cid)
                if e2 is None: continue
                if c1 and e2["country"] and c1 != e2["country"]: continue
                # Instant match on exact name
                if e1["norm_n"] == e2["norm_n"] and len(e1["norm_n"]) > 3 and (e1["norm_a"] == e2["norm_a"] or not e1["norm_a"] or not e2["norm_a"]):
                    final_matches[s1_id].add(cid)
                    continue
                f_vec = extract_features_31(e1, e2, cid)
                chunk_pairs.append((s1_id, cid))
                chunk_feats.append(f_vec)

        if chunk_feats:
            X_chunk = np.array(chunk_feats, dtype=np.float32)
            probs = final_clf.predict_proba(X_chunk)[:, 1]
            for (s1_id, cid), prob in zip(chunk_pairs, probs):
                if prob >= best_tau:
                    final_matches[s1_id].add(cid)
            total_scored += len(chunk_feats)

        pct = chunk_end / len(test_s1_ids) * 100
        print(f"  Chunk {chunk_start:,d}..{chunk_end:,d} ({pct:.1f}%) scored {len(chunk_feats):,d} pairs | Matches so far: {sum(len(v) for v in final_matches.values()):,d}", flush=True)

    print(f"  Total scored: {total_scored:,d} | Total matches: {sum(len(v) for v in final_matches.values()):,d} across {len(final_matches):,d} entities", flush=True)

    # ===================================================================
    # PHASE 3: WRITE OUTPUTS
    # ===================================================================
    print("\n[PHASE 3] Writing output files...", flush=True)
    matching_path = os.path.join(EXP_OUTPUT, "matching_results.tsv")
    candidate_path = os.path.join(EXP_OUTPUT, "candidate_pairs.tsv")

    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1_ids:
            matches = final_matches.get(s1_id, set())
            f.write(f"{s1_id}\t{','.join(sorted(matches))}\n")

    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in test_s1_ids:
            cands = final_candidates.get(s1_id, set())
            f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")

    print(f"  Written: {matching_path}", flush=True)
    print(f"  Written: {candidate_path}", flush=True)

    # Count stats
    match_count = sum(1 for s1 in test_s1_ids if len(final_matches.get(s1, set())) > 0)
    cand_count = sum(1 for s1 in test_s1_ids if len(final_candidates.get(s1, set())) > 0)
    print(f"  Matching: {match_count:,d} non-empty / {len(test_s1_ids):,d} total", flush=True)
    print(f"  Candidates: {cand_count:,d} non-empty / {len(test_s1_ids):,d} total", flush=True)

    # ===================================================================
    # PHASE 4: VALIDATION
    # ===================================================================
    print("\n[PHASE 4] Running official validator...", flush=True)
    validator_path = os.path.join(REPO_ROOT, "utils", "validate_submission.py")
    test_dir = os.path.dirname(paths["test_s1"])
    import subprocess
    result = subprocess.run(
        [sys.executable, validator_path, "--matching", matching_path, "--candidate", candidate_path, "--test-dir", test_dir, "--check-ids"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"  VALIDATOR ERROR: {result.stderr}", flush=True)

    # ===================================================================
    # PHASE 5: PACKAGING
    # ===================================================================
    print("\n[PHASE 5] Packaging submission ZIP...", flush=True)
    zip_name = "Antigravity_ML_submission_semantic.zip"
    zip_path = os.path.join(REPO_ROOT, zip_name)

    code_src = os.path.join(REPO_ROOT, "code", "business_entity_resolution")
    doc_path = os.path.join(REPO_ROOT, "Documentation_template.md")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(matching_path, "output/matching_results.tsv")
        z.write(candidate_path, "output/candidate_pairs.tsv")
        if os.path.exists(doc_path):
            z.write(doc_path, "Documentation_template.md")
        for root, dirs, files in os.walk(code_src):
            for f in files:
                if f.endswith(('.py', '.md', '.txt')) and '__pycache__' not in root:
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full, REPO_ROOT)
                    z.write(full, arcname)

    sha256 = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(1024*1024):
            sha256.update(chunk)

    print(f"  ZIP: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)", flush=True)
    print(f"  SHA256: {sha256.hexdigest().upper()}", flush=True)

    total_elapsed = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"PRODUCTION PIPELINE COMPLETE in {total_elapsed/60:.1f} minutes")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
