"""
Downstream LightGBM Evaluation on Semantic-Augmented Candidate Sets.
Tests whether candidate recovery from TF-IDF char-ngram retrieval
actually improves Macro F0.5 when scored by the existing 31-feature LightGBM.

Configurations tested:
  A) Champion only (control)
  B) Champion + C_norm_name_addr K=5
  C) Champion + C_norm_name_addr K=5 (zero-match-protected: skip semantic for card=0 champion entities)
  D) Champion + C_norm_name_addr K=10
  E) Champion + A_name_only K=5

All use the SAME 31-feature LightGBM and threshold sweep.
5-fold GroupKFold entity-level CV with full cardinality/country breakdown.
"""
import os, sys, time, re
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
        "raw_n": raw_n, "raw_a": raw_a,
        "norm_n": norm_n, "norm_a": norm_a, "country": c,
        "toks_n": toks_n, "toks_a": toks_a, "nums": nums,
        "norm_trans_n": norm_trans_n, "pincode": pincode,
        "char4": char4, "url_core": url_core, "content_n": content_n
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

def build_rep_text(row, rep_type):
    name = row.get("business_name", "")
    addr = row.get("business_address", "")
    norm_n = normalize_name(name.replace("-", " ").replace("/", " "))
    norm_a = normalize_address(addr.replace("-", " ").replace("/", " "))
    trans_n = transliterate_indic(name)
    norm_tn = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""
    if rep_type == "A_name_only":
        return norm_n
    elif rep_type == "C_norm_name_addr":
        return f"{norm_n} {norm_a}"
    return norm_n

def semantic_retrieve_per_country(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries, rep_type, K):
    """Build TF-IDF index per country and retrieve top-K candidates."""
    all_cand_ids = list(s2_dict.keys()) + list(s3_dict.keys())
    country_to_cand_ids = defaultdict(list)
    for eid in all_cand_ids:
        country_to_cand_ids[cand_countries[eid]].append(eid)

    sem_cands = defaultdict(set)
    for country, c_cand_ids in country_to_cand_ids.items():
        if len(c_cand_ids) < 5:
            continue
        c_s1_ids = [sid for sid in s1_list if s1_countries[sid] == country]
        if not c_s1_ids:
            continue

        cand_corpus = [build_rep_text(s2_dict[eid] if eid.startswith("S2-") else s3_dict[eid], rep_type) for eid in c_cand_ids]
        s1_corpus = [build_rep_text(s1_dict[sid], rep_type) for sid in c_s1_ids]

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
                if len(sims) > K:
                    top_idx = np.argpartition(sims, -K)[-K:]
                else:
                    top_idx = np.arange(len(sims))
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                for idx in top_idx[:K]:
                    if sims[idx] >= 0.30:
                        sem_cands[s1_id].add(c_cand_ids[idx])
    return sem_cands

def evaluate_config(config_name, candidate_sets, gt_dict, s1_dict, cached_s1, cached_cands, s2_dict, s3_dict):
    """Run 5-fold GroupKFold entity-level CV with 31-feature LightGBM on given candidate sets."""
    print(f"\n--- EVALUATING CONFIG: {config_name} ---", flush=True)
    t0 = time.time()

    # Generate pairs
    pairs = []
    features = []
    labels = []
    groups = []
    s1_list = list(s1_dict.keys())

    total_true = sum(len(m) for m in gt_dict.values())
    hits = 0
    cand_counts = []

    for s1_id in s1_list:
        true_matches = gt_dict.get(s1_id, set())
        cands = candidate_sets.get(s1_id, set())
        cand_counts.append(len(cands))
        hits += len(true_matches & cands)

        e1 = cached_s1[s1_id]
        c1 = e1["country"]
        for cid in cands:
            e2 = cached_cands.get(cid)
            if e2 is None:
                continue
            c2 = e2["country"]
            if c1 and c2 and c1 != c2:
                continue
            f_vec = extract_features_31(e1, e2, cid)
            pairs.append((s1_id, cid))
            features.append(f_vec)
            labels.append(1 if cid in true_matches else 0)
            groups.append(s1_id)

    cand_recall = hits / total_true
    avg_cands = np.mean(cand_counts)
    print(f"  CandRecall: {cand_recall*100:.2f}% | AvgCands: {avg_cands:.1f} | Pairs: {len(pairs):,d} (Pos:{sum(labels):,d}, Neg:{len(labels)-sum(labels):,d})", flush=True)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups_arr = np.array(groups)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups_arr))
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(splits):
        clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    # Threshold sweep
    best_tau = 0.62
    best_f05 = 0.0
    for tau in np.arange(0.50, 0.82, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs):
            if p >= tau:
                t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=0.5)
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_tau = tau

    # Final predictions at best tau
    preds = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs):
        if p >= best_tau:
            preds[sid].add(cid)
    m_all = compute_comprehensive_metrics(gt_dict, preds, beta=0.5)

    # Per-fold scores
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs_list = [pairs[i] for i in val_idx]
        val_s1_set = set(groups_arr[val_idx])
        val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs_list, oof_probs[val_idx]):
            if p >= best_tau:
                fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=0.5)
        fold_scores.append(m_fold["macro_f05"])

    # Cardinality breakdown
    zero_gt = {sid: m for sid, m in gt_dict.items() if len(m) == 0}
    zero_preds = {sid: preds.get(sid, set()) for sid in zero_gt}
    m_zero = compute_comprehensive_metrics(zero_gt, zero_preds, beta=0.5)

    one_gt = {sid: m for sid, m in gt_dict.items() if len(m) == 1}
    one_preds = {sid: preds.get(sid, set()) for sid in one_gt}
    m_one = compute_comprehensive_metrics(one_gt, one_preds, beta=0.5)

    multi_gt = {sid: m for sid, m in gt_dict.items() if len(m) >= 2}
    multi_preds = {sid: preds.get(sid, set()) for sid in multi_gt}
    m_multi = compute_comprehensive_metrics(multi_gt, multi_preds, beta=0.5)

    # Country
    us_gt = {sid: m for sid, m in gt_dict.items() if s1_dict.get(sid, {}).get("country", "").upper() == "US"}
    us_preds = {sid: preds.get(sid, set()) for sid in us_gt}
    m_us = compute_comprehensive_metrics(us_gt, us_preds, beta=0.5)

    india_gt = {sid: m for sid, m in gt_dict.items() if s1_dict.get(sid, {}).get("country", "").upper() == "INDIA"}
    india_preds = {sid: preds.get(sid, set()) for sid in india_gt}
    m_india = compute_comprehensive_metrics(india_gt, india_preds, beta=0.5)

    elapsed = time.time() - t0

    result = {
        "config": config_name,
        "cand_recall": cand_recall,
        "avg_cands": avg_cands,
        "best_tau": best_tau,
        "macro_f05": m_all["macro_f05"],
        "macro_prec": m_all["macro_precision"],
        "macro_rec": m_all["macro_recall"],
        "micro_f05": m_all["micro_f05"],
        "zero_f05": m_zero["macro_f05"],
        "one_f05": m_one["macro_f05"],
        "multi_f05": m_multi["macro_f05"],
        "us_f05": m_us["macro_f05"],
        "india_f05": m_india["macro_f05"],
        "tp": m_all["tp"],
        "fp": m_all["fp"],
        "fn": m_all["fn"],
        "fold_scores": fold_scores,
        "fold_mean": np.mean(fold_scores),
        "fold_std": np.std(fold_scores),
        "runtime": elapsed,
    }

    print(f"  Macro F0.5 = {result['macro_f05']:.4f} (Prec={result['macro_prec']:.4f}, Rec={result['macro_rec']:.4f}) tau={best_tau:.2f}", flush=True)
    print(f"  Folds: {' '.join(f'{s:.4f}' for s in fold_scores)} | Mean={result['fold_mean']:.4f} +/- {result['fold_std']:.4f}", flush=True)
    print(f"  Zero={result['zero_f05']:.4f} One={result['one_f05']:.4f} Multi={result['multi_f05']:.4f} US={result['us_f05']:.4f} India={result['india_f05']:.4f}", flush=True)
    print(f"  TP={result['tp']:,d} FP={result['fp']:,d} FN={result['fn']:,d} | {elapsed:.1f}s", flush=True)

    return result

def main():
    print("=" * 80)
    print("DOWNSTREAM LIGHTGBM EVALUATION ON SEMANTIC-AUGMENTED CANDIDATES")
    print("=" * 80)
    t0_total = time.time()
    paths = get_data_paths(is_sample=False)

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

    # Precompute entities
    print("Precomputing entities...", flush=True)
    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)

    # Champion candidate sets
    print("Building champion candidate sets...", flush=True)
    idx_s2 = defaultdict(list)
    for eid, row in s2_dict.items():
        for pass_keys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pass_keys: idx_s2[k].append(eid)
    idx_s3 = defaultdict(list)
    for eid, row in s3_dict.items():
        for pass_keys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pass_keys: idx_s3[k].append(eid)

    s1_list = list(s1_dict.keys())
    MAX_CANDS = 120
    champion_cands = {}
    for s1_id in s1_list:
        s1_row = s1_dict[s1_id]
        ordered_passes = extract_champion_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
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
        champion_cands[s1_id] = cands

    s1_countries = {sid: str(s1_dict[sid].get("country", "")).strip().upper() for sid in s1_list}
    cand_countries = {}
    for eid, row in s2_dict.items(): cand_countries[eid] = str(row.get("country", "")).strip().upper()
    for eid, row in s3_dict.items(): cand_countries[eid] = str(row.get("country", "")).strip().upper()

    # ===== CONFIG A: Champion only (CONTROL) =====
    result_A = evaluate_config("A_champion_only", champion_cands, gt_dict, s1_dict, cached_s1, cached_cands, s2_dict, s3_dict)

    # ===== CONFIG B: Champion + C_norm_name_addr K=5 =====
    print("\n[Building semantic candidates: C_norm_name_addr K=5]...", flush=True)
    sem_C5 = semantic_retrieve_per_country(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries, "C_norm_name_addr", K=5)
    union_C5 = {sid: champion_cands[sid] | sem_C5.get(sid, set()) for sid in s1_list}
    result_B = evaluate_config("B_champ+C_K5", union_C5, gt_dict, s1_dict, cached_s1, cached_cands, s2_dict, s3_dict)

    # ===== CONFIG C: Champion + C_norm_name_addr K=5 (zero-match-protected) =====
    # Only add semantic candidates for S1 entities where champion found >= 1 candidate
    union_C5_protected = {}
    for sid in s1_list:
        if len(champion_cands[sid]) > 0:
            union_C5_protected[sid] = champion_cands[sid] | sem_C5.get(sid, set())
        else:
            union_C5_protected[sid] = champion_cands[sid]
    result_C = evaluate_config("C_champ+C_K5_zeroProtected", union_C5_protected, gt_dict, s1_dict, cached_s1, cached_cands, s2_dict, s3_dict)

    # ===== CONFIG D: Champion + C_norm_name_addr K=10 =====
    print("\n[Building semantic candidates: C_norm_name_addr K=10]...", flush=True)
    sem_C10 = semantic_retrieve_per_country(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries, "C_norm_name_addr", K=10)
    union_C10 = {sid: champion_cands[sid] | sem_C10.get(sid, set()) for sid in s1_list}
    result_D = evaluate_config("D_champ+C_K10", union_C10, gt_dict, s1_dict, cached_s1, cached_cands, s2_dict, s3_dict)

    # ===== CONFIG E: Champion + A_name_only K=5 =====
    print("\n[Building semantic candidates: A_name_only K=5]...", flush=True)
    sem_A5 = semantic_retrieve_per_country(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries, "A_name_only", K=5)
    union_A5 = {sid: champion_cands[sid] | sem_A5.get(sid, set()) for sid in s1_list}
    result_E = evaluate_config("E_champ+A_K5", union_A5, gt_dict, s1_dict, cached_s1, cached_cands, s2_dict, s3_dict)

    # Summary table
    all_results = [result_A, result_B, result_C, result_D, result_E]
    print("\n" + "=" * 120)
    print("FINAL ABLATION SUMMARY TABLE")
    print("=" * 120)
    header = f"{'Config':<30} {'F0.5':>6} {'Prec':>6} {'Rec':>6} {'CandRec':>8} {'AvgCand':>8} {'tau':>5} {'Zero':>6} {'One':>6} {'Multi':>6} {'US':>6} {'India':>6} {'FoldMean':>9} {'FoldStd':>8} {'TP':>6} {'FP':>4} {'FN':>5}"
    print(header, flush=True)
    print("-" * 120, flush=True)
    for r in all_results:
        line = f"{r['config']:<30} {r['macro_f05']:.4f} {r['macro_prec']:.4f} {r['macro_rec']:.4f} {r['cand_recall']*100:>7.2f}% {r['avg_cands']:>7.1f} {r['best_tau']:.2f} {r['zero_f05']:.4f} {r['one_f05']:.4f} {r['multi_f05']:.4f} {r['us_f05']:.4f} {r['india_f05']:.4f} {r['fold_mean']:.4f}+/-{r['fold_std']:.4f} {r['tp']:>5d} {r['fp']:>4d} {r['fn']:>5d}"
        print(line, flush=True)
    print("=" * 120, flush=True)

    # Decision
    champion_f05 = result_A["macro_f05"]
    best_exp = max(all_results[1:], key=lambda r: r["macro_f05"])
    delta = best_exp["macro_f05"] - champion_f05

    print(f"\nChampion Macro F0.5: {champion_f05:.4f}")
    print(f"Best Experimental:  {best_exp['config']} -> {best_exp['macro_f05']:.4f} (Delta: {delta:+.4f})")

    if delta >= 0.002 and best_exp["fold_std"] <= 0.005 and best_exp["zero_f05"] >= result_A["zero_f05"] - 0.02:
        print("\n>>> CANDIDATE CHAMPION: Robust improvement detected. Requires independent verification. <<<")
    else:
        if delta < 0.002:
            print(f"\n>>> REJECTED: Delta {delta:+.4f} below +0.002 threshold. <<<")
        if best_exp["fold_std"] > 0.005:
            print(f">>> REJECTED: Fold std {best_exp['fold_std']:.4f} too high. <<<")
        if best_exp["zero_f05"] < result_A["zero_f05"] - 0.02:
            print(f">>> REJECTED: Zero-match F0.5 degradation {best_exp['zero_f05']:.4f} vs {result_A['zero_f05']:.4f}. <<<")
        print("\n>>> NO ROBUST IMPROVEMENT FOUND. CHAMPION RETAINED. <<<")

    # Save results
    out_dir = os.path.join(REPO_ROOT, "experiments", "semantic_candidate_recovery", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(out_dir, "downstream_ablation_results.csv"), index=False)
    print(f"\nResults saved. Total runtime: {time.time()-t0_total:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
