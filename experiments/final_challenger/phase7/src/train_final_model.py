"""
Train Final 44-Feature Model E (Config P5) on Full Training Candidate Universe.
Amazon ML Challenge 2026 - Business Entity Resolution (Phase 7)

Safety:
- Frozen champion untouched.
- Output artifacts saved strictly under experiments/final_challenger/phase7/artifacts/
"""
import os, sys, time, re, gc, joblib
from collections import defaultdict
import numpy as np
import pandas as pd
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
from transliteration import transliterate_indic

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
    content_n = [t for t in norm_n.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
    url_core = extract_url_core(norm_n)
    return {
        "raw_n": raw_n, "raw_a": raw_a,
        "norm_n": norm_n, "norm_a": norm_a, "country": c,
        "toks_n": toks_n, "toks_a": toks_a, "nums": nums,
        "nums_list": nums_list,
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

def extract_44_features_single(e1, e2, cid, base_feat, sem_info):
    """
    Extracts the exact 44 Model E features:
    - 31 base features
    - 2 semantic features (sem_cos, 1/sem_rk)
    - 5 interaction features
    - 6 conflict defense features
    """
    if sem_info and sem_info[1] <= 10 and sem_info[0] >= 0.30:
        sem_cos = float(sem_info[0])
        sem_rk_feat = 1.0 / float(sem_info[1])
    else:
        sem_cos = 0.0
        sem_rk_feat = 0.0

    tsort_n = base_feat[2]
    tsort_a = base_feat[12]
    empty_addr = base_feat[19]
    nums_1 = e1["nums"]
    nums_2 = e2["nums"]
    list_nums1 = e1["nums_list"]
    list_nums2 = e2["nums_list"]

    # 5 Interactions
    f_name_x_addr = tsort_n * (tsort_a if not empty_addr else 0.0)
    f_sem_cos_x_addr = sem_cos * (tsort_a if not empty_addr else 0.0)
    f_sem_cos_x_name = sem_cos * tsort_n
    f_sem_rk_x_addr = sem_rk_feat * (tsort_a if not empty_addr else 0.0)
    f_sem_rk_x_name = sem_rk_feat * tsort_n

    # 6 Conflict defense features
    f_exact_house_num_match = float(len(list_nums1) > 0 and len(list_nums2) > 0 and list_nums1[0] == list_nums2[0])
    num_inter = nums_1 & nums_2
    f_addr_num_conflict = float(len(nums_1) > 0 and len(nums_2) > 0 and len(num_inter) == 0)
    f_shared_bldg_diff_biz = float(len(num_inter) > 0) * (1.0 - tsort_n)
    f_same_name_diff_addr = (tsort_n * (1.0 - tsort_a)) if not empty_addr else 0.0
    f_name_high_addr_low = float(tsort_n >= 0.85 and tsort_a < 0.40 and not empty_addr)
    f_name_high_addr_conflict = float(tsort_n >= 0.80 and f_addr_num_conflict == 1.0)

    # 31 base + 2 semantic + 5 interaction + 6 conflict = 44 features
    return (
        base_feat +
        [sem_cos, sem_rk_feat] +
        [f_name_x_addr, f_sem_cos_x_addr, f_sem_cos_x_name, f_sem_rk_x_addr, f_sem_rk_x_name] +
        [f_exact_house_num_match, f_addr_num_conflict, f_shared_bldg_diff_biz,
         f_same_name_diff_addr, f_name_high_addr_low, f_name_high_addr_conflict]
    )

def main():
    print("=" * 80)
    print("TRAINING FINAL 44-FEATURE MODEL E (CONFIG P5)")
    print("Amazon ML Challenge 2026 - Phase 7 Final Model Fitting")
    print("=" * 80)
    t0_start = time.time()
    paths = get_data_paths(is_sample=False)

    art_dir = os.path.join(REPO_ROOT, "experiments", "final_challenger", "phase7", "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    print("Loading 10,000 ground truth training entities...", flush=True)
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

    print("Loading candidate pools (S2 & S3, max_pool=100k)...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2, max_pool=100000)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3, max_pool=100000)

    print("Precomputing entity feature attributes...", flush=True)
    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)

    print("Building champion candidate sets with 6-pass prioritized blocking (MAX_CANDS=120)...", flush=True)
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

    print("Building country-partitioned TF-IDF semantic indexes...", flush=True)
    all_cand_ids = list(s2_dict.keys()) + list(s3_dict.keys())
    country_to_cand_ids = defaultdict(list)
    for eid in all_cand_ids:
        country_to_cand_ids[cand_countries[eid]].append(eid)

    sem_top_records = defaultdict(dict)
    K_MAX = 10
    COS_MIN = 0.30

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

    print("Building union candidate sets (Zero-Protected)...", flush=True)
    cands_D = defaultdict(set)
    for sid in s1_list:
        cands_D[sid] = set(champion_cands[sid])
        if len(champion_cands[sid]) > 0:
            for cid, (score, rank) in sem_top_records.get(sid, {}).items():
                if rank <= 10 and score >= 0.30:
                    cands_D[sid].add(cid)

    chal_pairs = []
    labels = []
    hits = 0
    for sid in s1_list:
        t_set = gt_dict.get(sid, set())
        c_set = cands_D.get(sid, set())
        hits += len(t_set & c_set)
        c1 = cached_s1[sid]["country"]
        for cid in c_set:
            e2 = cached_cands.get(cid)
            if e2 and (not c1 or not e2["country"] or c1 == e2["country"]):
                chal_pairs.append((sid, cid))
                labels.append(1 if cid in t_set else 0)

    print(f"Candidate Universe: {len(chal_pairs):,d} pairs | Cand Recall: {hits/total_true*100:.2f}% ({hits:,d}/{total_true:,d})", flush=True)

    print("Extracting 44 Model E features...", flush=True)
    t0_feat = time.time()
    X_list = []
    for sid, cid in chal_pairs:
        base_f = extract_features_31(cached_s1[sid], cached_cands[cid], cid)
        sem_info = sem_top_records.get(sid, {}).get(cid)
        feat_44 = extract_44_features_single(cached_s1[sid], cached_cands[cid], cid, base_f, sem_info)
        X_list.append(feat_44)

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)
    print(f"Features extracted in {time.time() - t0_feat:.1f}s. Shape: {X_train.shape}", flush=True)
    print(f"Positives: {np.sum(y_train):,d} | Negatives: {len(y_train) - np.sum(y_train):,d}")

    # Final LightGBM Parameters (Config P5)
    p5_params = {
        **Config.LGBM_PARAMS,
        "max_depth": 9,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "min_child_samples": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    print("\nFitting final LightGBM model (P5)...", flush=True)
    t0_fit = time.time()
    clf = lgb.LGBMClassifier(**p5_params)
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0_fit
    print(f"Model fitted in {fit_time:.1f}s.", flush=True)

    model_path = os.path.join(art_dir, "final_model_p5.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved final model to: {model_path} ({os.path.getsize(model_path):,d} bytes)", flush=True)

    # Save training metadata
    train_meta = {
        "training_entities_s1": len(s1_list),
        "training_candidate_pool": len(cached_cands),
        "candidate_generation": "Champion 6-Pass Union Semantic (K=10, cosine>=0.30, zero-protected)",
        "training_rows": len(y_train),
        "positive_pairs": int(np.sum(y_train)),
        "negative_pairs": int(len(y_train) - np.sum(y_train)),
        "num_features": X_train.shape[1],
        "model_parameters": p5_params,
        "threshold_tau": 0.72,
        "fit_time_seconds": fit_time,
        "total_runtime_seconds": time.time() - t0_start
    }
    meta_path = os.path.join(art_dir, "training_metadata.json")
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)
    print(f"Saved training metadata to: {meta_path}", flush=True)
    print("\nTRAINING COMPLETE SUCCESSFULLY!")

if __name__ == "__main__":
    main()
