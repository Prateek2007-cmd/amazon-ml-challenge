"""
Phase 3 & Phase 4: Joint K / Cosine Frontier Sweep.
Workspace: experiments/final_challenger/
Non-negotiable rule: Frozen champion is untouched.

Evaluates:
  Phase 3 (K Sweep): K in [3, 5, 8, 10, 15, 20, 30] at cosine >= 0.30, tau = 0.68
  Phase 4 (Cosine Sweep): Cosine in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45] at K = 5
  Joint Sweep: Top K values evaluated across cosine thresholds
  Secondary Tau Sweep: tau in [0.55 .. 0.78] for top configurations

Outputs metrics to:
  experiments/final_challenger/artifacts/phase34_frontier_results.csv
  experiments/final_challenger/reports/phase34_frontier_report.md
"""
import os, sys, time, re, gc, psutil
from collections import defaultdict
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

def get_ram_str():
    mem = psutil.virtual_memory()
    return f"{mem.used / (1024**3):.2f} GB ({mem.percent}%)"

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

def build_rep_text(row):
    name = row.get("business_name", "")
    addr = row.get("business_address", "")
    norm_n = normalize_name(name.replace("-", " ").replace("/", " "))
    norm_a = normalize_address(addr.replace("-", " ").replace("/", " "))
    return f"{norm_n} {norm_a}"

def precompute_top30_semantic(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries):
    print("Precomputing top-30 semantic candidate similarities (cosine >= 0.20)...", flush=True)
    all_cand_ids = list(s2_dict.keys()) + list(s3_dict.keys())
    country_to_cand_ids = defaultdict(list)
    for eid in all_cand_ids:
        country_to_cand_ids[cand_countries[eid]].append(eid)

    sem_top30 = defaultdict(list)
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
                for idx in top_idx[:K_MAX]:
                    score = float(sims[idx])
                    if score >= COS_MIN:
                        sem_top30[s1_id].append((c_cand_ids[idx], score))

        del X_cand, X_s1, sim_matrix, vectorizer
        gc.collect()

    print(f"Top-30 semantic candidates precomputed for {len(sem_top30):,d} entities.", flush=True)
    return sem_top30

def extract_semantic_candidates(sem_top30, s1_list, K, cos_threshold):
    sem_cands = defaultdict(set)
    for sid in s1_list:
        items = sem_top30.get(sid, [])
        for cid, score in items[:K]:
            if score >= cos_threshold:
                sem_cands[sid].add(cid)
    return sem_cands

def run_configuration_evaluation(
    K, cos_thresh, fixed_tau, candidate_sets, gt_dict, s1_dict,
    cached_s1, cached_cands, feature_cache, gkf_splits_dict,
    sweep_secondary_tau=False
):
    t0 = time.time()
    s1_list = list(s1_dict.keys())
    total_true = sum(len(m) for m in gt_dict.values())
    hits = 0
    cand_counts = []
    pairs = []
    labels = []
    groups = []

    for s1_id in s1_list:
        true_matches = gt_dict.get(s1_id, set())
        cands = candidate_sets.get(s1_id, set())
        cand_counts.append(len(cands))
        hits += len(true_matches & cands)

        e1 = cached_s1[s1_id]
        c1 = e1["country"]
        for cid in cands:
            e2 = cached_cands.get(cid)
            if e2 is None: continue
            c2 = e2["country"]
            if c1 and c2 and c1 != c2: continue
            pairs.append((s1_id, cid))
            labels.append(1 if cid in true_matches else 0)
            groups.append(s1_id)

    cand_recall = hits / total_true
    avg_cands = float(np.mean(cand_counts))
    p50_cands = float(np.percentile(cand_counts, 50))
    p95_cands = float(np.percentile(cand_counts, 95))
    p99_cands = float(np.percentile(cand_counts, 99))
    max_cands = int(np.max(cand_counts))

    # Build feature matrix from cache
    features = [feature_cache[p] for p in pairs]
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

    # Threshold determination
    best_tau = fixed_tau
    if sweep_secondary_tau:
        best_f05 = 0.0
        tau_candidates = [0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78]
        for tau in tau_candidates:
            t_preds = defaultdict(set)
            for (sid, cid), p in zip(pairs, oof_probs):
                if p >= tau: t_preds[sid].add(cid)
            m = compute_comprehensive_metrics(gt_dict, t_preds, beta=0.5)
            if m["macro_f05"] > best_f05:
                best_f05 = m["macro_f05"]
                best_tau = round(float(tau), 2)

    # Compute comprehensive metrics at best_tau
    preds = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs):
        if p >= best_tau: preds[sid].add(cid)

    m_all = compute_comprehensive_metrics(gt_dict, preds, beta=0.5)

    # Per-fold validation metrics
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

    # Zero-match safety & contamination audit
    zero_gt = {sid: m for sid, m in gt_dict.items() if len(m) == 0}
    zero_preds = {sid: preds.get(sid, set()) for sid in zero_gt}
    m_zero = compute_comprehensive_metrics(zero_gt, zero_preds, beta=0.5)
    zero_true_empty = sum(1 for sid, p_set in zero_preds.items() if len(p_set) == 0)
    zero_contaminated = sum(1 for sid, p_set in zero_preds.items() if len(p_set) > 0)

    # Cardinality metrics
    one_gt = {sid: m for sid, m in gt_dict.items() if len(m) == 1}
    one_preds = {sid: preds.get(sid, set()) for sid in one_gt}
    m_one = compute_comprehensive_metrics(one_gt, one_preds, beta=0.5)

    multi_gt = {sid: m for sid, m in gt_dict.items() if len(m) >= 2}
    multi_preds = {sid: preds.get(sid, set()) for sid in multi_gt}
    m_multi = compute_comprehensive_metrics(multi_gt, multi_preds, beta=0.5)

    # Country metrics
    us_gt = {sid: m for sid, m in gt_dict.items() if s1_dict[sid].get("country", "").upper() == "US"}
    us_preds = {sid: preds.get(sid, set()) for sid in us_gt}
    m_us = compute_comprehensive_metrics(us_gt, us_preds, beta=0.5)

    india_gt = {sid: m for sid, m in gt_dict.items() if s1_dict[sid].get("country", "").upper() == "INDIA"}
    india_preds = {sid: preds.get(sid, set()) for sid in india_gt}
    m_india = compute_comprehensive_metrics(india_gt, india_preds, beta=0.5)

    # Source metrics
    s2_gt = {sid: {m for m in true_m if m.startswith("S2-")} for sid, true_m in gt_dict.items()}
    s2_preds = {sid: {m for m in preds.get(sid, set()) if m.startswith("S2-")} for sid in gt_dict}
    m_s2 = compute_comprehensive_metrics(s2_gt, s2_preds, beta=0.5)

    s3_gt = {sid: {m for m in true_m if m.startswith("S3-")} for sid, true_m in gt_dict.items()}
    s3_preds = {sid: {m for m in preds.get(sid, set()) if m.startswith("S3-")} for sid in gt_dict}
    m_s3 = compute_comprehensive_metrics(s3_gt, s3_preds, beta=0.5)

    elapsed = time.time() - t0
    ram_usage = get_ram_str()

    res = {
        "K": K,
        "Cosine": cos_thresh,
        "Matcher_Tau": best_tau,
        "Macro_F05": m_all["macro_f05"],
        "Precision": m_all["macro_precision"],
        "Recall": m_all["macro_recall"],
        "Candidate_Recall": cand_recall,
        "Avg_Candidates": avg_cands,
        "P50": p50_cands,
        "P95": p95_cands,
        "P99": p99_cands,
        "Max_Candidates": max_cands,
        "Zero": m_zero["macro_f05"],
        "Zero_True_Empty": zero_true_empty,
        "Zero_Contaminated": zero_contaminated,
        "One": m_one["macro_f05"],
        "Multi": m_multi["macro_f05"],
        "US": m_us["macro_f05"],
        "India": m_india["macro_f05"],
        "S2": m_s2["macro_f05"],
        "S3": m_s3["macro_f05"],
        "Fold_Mean": float(np.mean(fold_scores)),
        "Fold_Std": float(np.std(fold_scores)),
        "Fold_Scores": " ".join(f"{s:.4f}" for s in fold_scores),
        "TP": m_all["tp"],
        "FP": m_all["fp"],
        "FN": m_all["fn"],
        "Runtime_s": elapsed,
        "Memory": ram_usage,
    }

    print(f"[K={K:>2}, Cos={cos_thresh:.2f}, tau={best_tau:.2f}] F0.5={res['Macro_F05']:.4f} (P={res['Precision']:.4f}, R={res['Recall']:.4f}) | CandRec={cand_recall*100:.2f}% | Avg={avg_cands:.1f} (P99={p99_cands:.0f}) | Zero={res['Zero']:.4f} (Contam={zero_contaminated}) | TP={res['TP']:,d} FP={res['FP']:,d} | {elapsed:.1f}s", flush=True)

    del X, y, features, pairs, oof_probs
    gc.collect()

    return res

def main():
    print("=" * 80)
    print("PHASE 3 & PHASE 4: JOINT K / COSINE FRONTIER SWEEP HARNESS")
    print("=" * 80)
    t0_start = time.time()
    paths = get_data_paths(is_sample=False)

    print("Loading 10,000 validation ground truth entities...", flush=True)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
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

    print("Loading candidate pools (S2 & S3, max_pool=100k)...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2, max_pool=100000)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3, max_pool=100000)

    print("Precomputing entity feature attributes...", flush=True)
    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)

    print("Building champion candidate sets with 6-pass prioritized blocking (MAX_CANDS=120 early-break)...", flush=True)
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

    # Precompute top-30 semantic candidates per country
    sem_top30 = precompute_top30_semantic(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries)

    # Precompute feature cache for all possible pairs across all configs (champion + sem_top30 with cos >= 0.20)
    print("Collecting all unique pairs across candidate generators to build feature cache...", flush=True)
    all_unique_pairs = set()
    for sid in s1_list:
        c1 = cached_s1[sid]["country"]
        # Champion candidates
        for cid in champion_cands[sid]:
            e2 = cached_cands.get(cid)
            if e2 and (not c1 or not e2["country"] or c1 == e2["country"]):
                all_unique_pairs.add((sid, cid))
        # Top-30 semantic candidates
        if len(champion_cands[sid]) > 0: # Zero-protection condition
            for cid, score in sem_top30.get(sid, []):
                e2 = cached_cands.get(cid)
                if e2 and (not c1 or not e2["country"] or c1 == e2["country"]):
                    all_unique_pairs.add((sid, cid))

    print(f"Total unique pairs to pre-extract features for: {len(all_unique_pairs):,d}", flush=True)
    t_feat0 = time.time()
    feature_cache = {}
    for sid, cid in all_unique_pairs:
        feature_cache[(sid, cid)] = extract_features_31(cached_s1[sid], cached_cands[cid], cid)
    print(f"Feature extraction completed in {time.time() - t_feat0:.1f}s. Cache size: {len(feature_cache):,d} vectors.", flush=True)

    results = []

    # ==============================================================
    # PHASE 3: K SWEEP (K in [3, 5, 8, 10, 15, 20, 30] at Cosine 0.30)
    # ==============================================================
    print("\n" + "=" * 80)
    print("EXECUTING PHASE 3: K SWEEP (Cosine = 0.30, Tau = 0.68)")
    print("=" * 80)
    k_values = [3, 5, 8, 10, 15, 20, 30]
    phase3_results = {}

    for K in k_values:
        sem_cands_k = extract_semantic_candidates(sem_top30, s1_list, K=K, cos_threshold=0.30)
        challenger_cands_k = {}
        for sid in s1_list:
            if len(champion_cands[sid]) > 0:
                challenger_cands_k[sid] = champion_cands[sid] | sem_cands_k.get(sid, set())
            else:
                challenger_cands_k[sid] = champion_cands[sid]

        res = run_configuration_evaluation(
            K=K, cos_thresh=0.30, fixed_tau=0.68, candidate_sets=challenger_cands_k,
            gt_dict=gt_dict, s1_dict=s1_dict, cached_s1=cached_s1, cached_cands=cached_cands,
            feature_cache=feature_cache, gkf_splits_dict=None, sweep_secondary_tau=False
        )
        results.append(res)
        phase3_results[K] = res

    # Sort Phase 3 by Macro F0.5 to find top 3 K values
    sorted_k = sorted(k_values, key=lambda k: phase3_results[k]["Macro_F05"], reverse=True)
    top_3_k = sorted_k[:3]
    print(f"\nTop 3 K values by Macro F0.5: {top_3_k}", flush=True)

    # ==============================================================
    # PHASE 4: COSINE SWEEP (Cosine in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45] at K = 5)
    # ==============================================================
    print("\n" + "=" * 80)
    print("EXECUTING PHASE 4: COSINE SWEEP (K = 5, Tau = 0.68)")
    print("=" * 80)
    cos_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    phase4_results = {}

    for cos_t in cos_thresholds:
        if cos_t == 0.30:
            # Already evaluated in Phase 3
            phase4_results[cos_t] = phase3_results[5]
            continue

        sem_cands_cos = extract_semantic_candidates(sem_top30, s1_list, K=5, cos_threshold=cos_t)
        challenger_cands_cos = {}
        for sid in s1_list:
            if len(champion_cands[sid]) > 0:
                challenger_cands_cos[sid] = champion_cands[sid] | sem_cands_cos.get(sid, set())
            else:
                challenger_cands_cos[sid] = champion_cands[sid]

        res = run_configuration_evaluation(
            K=5, cos_thresh=cos_t, fixed_tau=0.68, candidate_sets=challenger_cands_cos,
            gt_dict=gt_dict, s1_dict=s1_dict, cached_s1=cached_s1, cached_cands=cached_cands,
            feature_cache=feature_cache, gkf_splits_dict=None, sweep_secondary_tau=False
        )
        results.append(res)
        phase4_results[cos_t] = res

    # ==============================================================
    # REDUCED JOINT SWEEP (Top 3 K values across other strong Cosine thresholds)
    # ==============================================================
    print("\n" + "=" * 80)
    print("EXECUTING REDUCED JOINT SWEEP (Top K values across Cosine Grid)")
    print("=" * 80)
    joint_evaluated = set((r["K"], r["Cosine"]) for r in results)

    for k_val in top_3_k:
        for cos_t in [0.25, 0.35, 0.40]:
            if (k_val, cos_t) in joint_evaluated: continue
            sem_cands_joint = extract_semantic_candidates(sem_top30, s1_list, K=k_val, cos_threshold=cos_t)
            cands_joint = {}
            for sid in s1_list:
                if len(champion_cands[sid]) > 0:
                    cands_joint[sid] = champion_cands[sid] | sem_cands_joint.get(sid, set())
                else:
                    cands_joint[sid] = champion_cands[sid]

            res = run_configuration_evaluation(
                K=k_val, cos_thresh=cos_t, fixed_tau=0.68, candidate_sets=cands_joint,
                gt_dict=gt_dict, s1_dict=s1_dict, cached_s1=cached_s1, cached_cands=cached_cands,
                feature_cache=feature_cache, gkf_splits_dict=None, sweep_secondary_tau=False
            )
            results.append(res)
            joint_evaluated.add((k_val, cos_t))

    # ==============================================================
    # SECONDARY THRESHOLD (TAU) OPTIMIZATION FOR TOP 3 CONFIGURATIONS
    # ==============================================================
    print("\n" + "=" * 80)
    print("EXECUTING SECONDARY MATCHER THRESHOLD (TAU) OPTIMIZATION FOR TOP CONFIGS")
    print("=" * 80)
    top_configs = sorted(results, key=lambda r: r["Macro_F05"], reverse=True)[:3]

    for cfg in top_configs:
        k_val = cfg["K"]
        cos_t = cfg["Cosine"]
        print(f"Re-evaluating top config (K={k_val}, Cosine={cos_t:.2f}) with full tau sweep...", flush=True)

        sem_cands_top = extract_semantic_candidates(sem_top30, s1_list, K=k_val, cos_threshold=cos_t)
        cands_top = {}
        for sid in s1_list:
            if len(champion_cands[sid]) > 0:
                cands_top[sid] = champion_cands[sid] | sem_cands_top.get(sid, set())
            else:
                cands_top[sid] = champion_cands[sid]

        res_opt_tau = run_configuration_evaluation(
            K=k_val, cos_thresh=cos_t, fixed_tau=0.68, candidate_sets=cands_top,
            gt_dict=gt_dict, s1_dict=s1_dict, cached_s1=cached_s1, cached_cands=cached_cands,
            feature_cache=feature_cache, gkf_splits_dict=None, sweep_secondary_tau=True
        )
        if res_opt_tau["Matcher_Tau"] != 0.68:
            results.append(res_opt_tau)

    # Save CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by="Macro_F05", ascending=False).reset_index(drop=True)
    artifacts_dir = os.path.join(REPO_ROOT, "experiments", "final_challenger", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    csv_path = os.path.join(artifacts_dir, "phase34_frontier_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV results to: {csv_path}", flush=True)

    # Marginal Analysis table for K sweep
    marginal_rows = []
    for i in range(len(k_values) - 1):
        k_prev, k_curr = k_values[i], k_values[i+1]
        r_prev, r_curr = phase3_results[k_prev], phase3_results[k_curr]
        marginal_rows.append({
            "Step": f"K{k_prev} -> K{k_curr}",
            "Delta_Cand_Recall": f"{r_curr['Candidate_Recall']*100 - r_prev['Candidate_Recall']*100:+.2f}%",
            "Delta_Avg_Cands": f"{r_curr['Avg_Candidates'] - r_prev['Avg_Candidates']:+.1f}",
            "Delta_TP": f"{r_curr['TP'] - r_prev['TP']:+d}",
            "Delta_FP": f"{r_curr['FP'] - r_prev['FP']:+d}",
            "Delta_Macro_F05": f"{r_curr['Macro_F05'] - r_prev['Macro_F05']:+.4f}",
        })
    df_marginal = pd.DataFrame(marginal_rows)

    # Build Markdown Report
    reports_dir = os.path.join(REPO_ROOT, "experiments", "final_challenger", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "phase34_frontier_report.md")

    best_cfg = df.iloc[0]
    best_cand_rec_cfg = df.sort_values(by="Candidate_Recall", ascending=False).iloc[0]
    best_prec_cfg = df.sort_values(by="Precision", ascending=False).iloc[0]
    best_rec_cfg = df.sort_values(by="Recall", ascending=False).iloc[0]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3 & 4: Joint K / Cosine Frontier Sweep Report\n\n")
        f.write(f"**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Sweep Runtime:** {time.time() - t0_start:.1f}s\n")
        f.write(f"**Validation Dataset:** 10,000 Source-1 Entities, 5-Fold GroupKFold CV, 31 Features\n\n")

        f.write("## 1. Primary Leaderboard (Sorted by Macro F0.5 Descending)\n\n")
        f.write("| Rank | K | Cosine | Tau | Macro F0.5 | Precision | Recall | Cand Recall | Avg Cands | P50 | P95 | P99 | Zero | One | Multi | US | India | S2 | S3 | TP | FP | FN | Runtime |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rank, r in enumerate(df.itertuples(), start=1):
            f.write(f"| {rank} | {r.K} | {r.Cosine:.2f} | {r.Matcher_Tau:.2f} | **{r.Macro_F05:.4f}** | {r.Precision:.4f} | {r.Recall:.4f} | {r.Candidate_Recall*100:.2f}% | {r.Avg_Candidates:.1f} | {r.P50:.0f} | {r.P95:.0f} | {r.P99:.0f} | {r.Zero:.4f} | {r.One:.4f} | {r.Multi:.4f} | {r.US:.4f} | {r.India:.4f} | {r.S2:.4f} | {r.S3:.4f} | {r.TP:,d} | {r.FP:,d} | {r.FN:,d} | {r.Runtime_s:.1f}s |\n")

        f.write("\n## 2. Marginal Analysis: Impact of Increasing K (at Cosine = 0.30)\n\n")
        f.write("| Step | Delta Cand Recall | Delta Avg Candidates | Delta TP | Delta FP | Delta Macro F0.5 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for m in marginal_rows:
            f.write(f"| {m['Step']} | {m['Delta_Cand_Recall']} | {m['Delta_Avg_Cands']} | {m['Delta_TP']} | {m['Delta_FP']} | {m['Delta_Macro_F05']} |\n")

        f.write("\n## 3. Zero-Match Contamination Audit\n\n")
        f.write("Evaluates whether expanding semantic retrieval introduces false positive predictions into true zero-match entities:\n\n")
        f.write("| Configuration (K, Cosine, Tau) | Zero-Match F0.5 | True Zero -> Empty (Safe) | True Zero -> NonEmpty (Contaminated) | Contamination Rate |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for r in df.itertuples():
            tot_zero = r.Zero_True_Empty + r.Zero_Contaminated
            contam_rate = (r.Zero_Contaminated / tot_zero * 100) if tot_zero > 0 else 0.0
            f.write(f"| K={r.K}, Cos={r.Cosine:.2f}, Tau={r.Matcher_Tau:.2f} | {r.Zero:.4f} | {r.Zero_True_Empty:,d} | {r.Zero_Contaminated:,d} | {contam_rate:.2f}% |\n")

        f.write("\n## 4. Cross-Fold Stability for Top 3 Configurations\n\n")
        for rank, r in enumerate(df.iloc[:3].itertuples(), start=1):
            f.write(f"### Rank {rank}: K={r.K}, Cosine={r.Cosine:.2f}, Tau={r.Matcher_Tau:.2f}\n")
            f.write(f"- **Macro F0.5:** {r.Macro_F05:.4f}\n")
            f.write(f"- **5-Fold Mean +/- Std:** {r.Fold_Mean:.4f} +/- {r.Fold_Std:.4f}\n")
            f.write(f"- **Individual Folds:** `{r.Fold_Scores}`\n\n")

        f.write("## 5. Decision Summary & Benchmark Comparison\n\n")
        f.write(f"**CURRENT CHAMPION**\n")
        f.write(f"Macro F0.5 = 0.9566 (Prec = 0.9738, Rec = 0.9232, CandRec = 94.97%)\n\n")

        f.write(f"**CURRENT CHALLENGER (Phase 1 Baseline)**\n")
        f.write(f"K = 5\ncosine = 0.30\ntau = 0.68\nMacro F0.5 = 0.9696\n\n")

        f.write(f"**BEST FRONTIER CONFIGURATION**\n")
        f.write(f"K = {best_cfg['K']}\ncosine = {best_cfg['Cosine']:.2f}\ntau = {best_cfg['Matcher_Tau']:.2f}\nMacro F0.5 = {best_cfg['Macro_F05']:.4f}\n\n")

        f.write(f"**IMPROVEMENT OVER CHAMPION**\n")
        f.write(f"{best_cfg['Macro_F05'] - 0.9566:+.4f} (from 0.9566 to {best_cfg['Macro_F05']:.4f})\n\n")

        f.write(f"**IMPROVEMENT OVER CURRENT CHALLENGER**\n")
        f.write(f"{best_cfg['Macro_F05'] - 0.9696:+.4f} (from 0.9696 to {best_cfg['Macro_F05']:.4f})\n\n")

        f.write(f"**CANDIDATE RECALL**\n")
        f.write(f"{best_cfg['Candidate_Recall']*100:.2f}%\n\n")

        f.write(f"**PRECISION**\n")
        f.write(f"{best_cfg['Precision']:.4f}\n\n")

        f.write(f"**RECALL**\n")
        f.write(f"{best_cfg['Recall']:.4f}\n\n")

        f.write(f"**ZERO-MATCH IMPACT**\n")
        f.write(f"Zero F0.5 = {best_cfg['Zero']:.4f} ({best_cfg['Zero_Contaminated']} contaminated entities out of {best_cfg['Zero_True_Empty'] + best_cfg['Zero_Contaminated']})\n\n")

        f.write(f"**5-FOLD STABILITY**\n")
        f.write(f"{best_cfg['Fold_Mean']:.4f} +/- {best_cfg['Fold_Std']:.4f}\n\n")

        f.write(f"**RECOMMENDED PHASE 5 CONFIGURATIONS**\n")
        for i, r in enumerate(df.iloc[:3].itertuples(), start=1):
            f.write(f"{i}. K={r.K}, Cosine={r.Cosine:.2f}, Tau={r.Matcher_Tau:.2f} (Macro F0.5 = {r.Macro_F05:.4f}, CandRec = {r.Candidate_Recall*100:.2f}%)\n")

    print(f"\nSaved Markdown report to: {report_path}", flush=True)
    print(f"Sweep complete in {time.time() - t0_start:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
