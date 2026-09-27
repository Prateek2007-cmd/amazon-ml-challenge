"""
Phase 8: Master Optimization Sprint Runner
Pushing Business Entity Resolution toward 0.985+ / 0.990 Macro F0.5
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
PHASE8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import (
    normalize_name, normalize_address, extract_name_aliases,
    extract_url_core, extract_clean_numbers,
)
from transliteration import transliterate_indic, has_indic_script
from evaluate import compute_comprehensive_metrics, compute_entity_metrics
from phase8_core import (
    NAME_STOPWORDS, ADDR_STOPWORDS, PINCODE_RE,
    precompute_entity, extract_champion_keys, extract_features_31,
    build_rep_text, precompute_semantic_index, evaluate_comprehensive_full
)

def log_experiment(res_dict, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame([res_dict])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)
    print(f"Logged experiment [{res_dict['experiment_id']}] to {csv_path}", flush=True)

def main():
    print("=" * 80)
    print("PHASE 8: MASTER OPTIMIZATION SPRINT (TARGET 0.985+ / 0.990)")
    print("Amazon ML Challenge 2026 - Business Entity Resolution")
    print("=" * 80)
    t0_start = time.time()
    paths = get_data_paths(is_sample=False)
    csv_results_path = os.path.join(PHASE8_DIR, "experiment_results.csv")

    # 1. Load Ground Truth and Entities
    print("Loading 10,000 validation ground truth entities...", flush=True)
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

    # Precompute semantic retrieval data
    sem_top_records = precompute_semantic_index(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries)

    # Config D: Champion UNION Semantic (K=10, cos >= 0.30) WITH zero protection
    cands_D = defaultdict(set)
    for sid in s1_list:
        cands_D[sid] = set(champion_cands[sid])
        if len(champion_cands[sid]) > 0:
            for cid, (score, rank) in sem_top_records.get(sid, {}).items():
                if rank <= 10 and score >= 0.30:
                    cands_D[sid].add(cid)

    chal_pairs = []
    labels = []
    groups = []
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
                groups.append(sid)

    cand_rec = hits / total_true
    y_chal = np.array(labels, dtype=np.int32)
    groups_chal = np.array(groups)
    print(f"Candidate Universe: {len(chal_pairs):,d} pairs | Cand Recall: {cand_rec*100:.2f}% ({hits:,d}/{total_true:,d})", flush=True)

    # Pre-extract base 31 features
    print("Pre-extracting features for candidate pairs...", flush=True)
    t0_feat = time.time()
    feature_cache_31 = {}
    extra_sem_features = {}
    for sid, cid in chal_pairs:
        feature_cache_31[(sid, cid)] = extract_features_31(cached_s1[sid], cached_cands[cid], cid)
        sem_info = sem_top_records.get(sid, {}).get(cid)
        if sem_info and sem_info[1] <= 10 and sem_info[0] >= 0.30:
            sem_cos, sem_rk = sem_info[0], sem_info[1]
            sem_rk_feat = 1.0 / sem_rk
        else:
            sem_cos, sem_rk_feat = 0.0, 0.0
        extra_sem_features[(sid, cid)] = (sem_cos, sem_rk_feat)

    X_base = np.array([feature_cache_31[p] for p in chal_pairs], dtype=np.float32)
    X_sem_extra = np.array([extra_sem_features[p] for p in chal_pairs], dtype=np.float32)

    # Extract 11 Phase 6 interaction & conflict features (Model E)
    model_e_extras = []
    for (sid, cid), base_feat in zip(chal_pairs, X_base):
        e1 = cached_s1[sid]
        e2 = cached_cands[cid]
        tsort_n = base_feat[2]
        tsort_a = base_feat[12]
        empty_addr = base_feat[19]
        nums_1 = e1["nums"]
        nums_2 = e2["nums"]
        list_nums1 = e1["nums_list"]
        list_nums2 = e2["nums_list"]
        sem_cos, sem_rk_feat = extra_sem_features[(sid, cid)]

        f_name_x_addr = tsort_n * (tsort_a if not empty_addr else 0.0)
        f_sem_cos_x_addr = sem_cos * (tsort_a if not empty_addr else 0.0)
        f_sem_cos_x_name = sem_cos * tsort_n
        f_sem_rk_x_addr = sem_rk_feat * (tsort_a if not empty_addr else 0.0)
        f_sem_rk_x_name = sem_rk_feat * tsort_n

        f_exact_house_num_match = float(len(list_nums1) > 0 and len(list_nums2) > 0 and list_nums1[0] == list_nums2[0])
        num_inter = nums_1 & nums_2
        f_addr_num_conflict = float(len(nums_1) > 0 and len(nums_2) > 0 and len(num_inter) == 0)
        f_shared_bldg_diff_biz = float(len(num_inter) > 0) * (1.0 - tsort_n)
        f_same_name_diff_addr = (tsort_n * (1.0 - tsort_a)) if not empty_addr else 0.0
        f_name_high_addr_low = float(tsort_n >= 0.85 and tsort_a < 0.40 and not empty_addr)
        f_name_high_addr_conflict = float(tsort_n >= 0.80 and f_addr_num_conflict == 1.0)

        model_e_extras.append([
            f_name_x_addr, f_sem_cos_x_addr, f_sem_cos_x_name, f_sem_rk_x_addr, f_sem_rk_x_name,
            f_exact_house_num_match, f_addr_num_conflict, f_shared_bldg_diff_biz,
            f_same_name_diff_addr, f_name_high_addr_low, f_name_high_addr_conflict
        ])

    X_model_e = np.hstack([X_base, X_sem_extra, np.array(model_e_extras, dtype=np.float32)])
    print(f"Model E 44-feature matrix ready: {X_model_e.shape}", flush=True)

    gkf = GroupKFold(n_splits=5)
    splits_chal = list(gkf.split(X_model_e, y_chal, groups_chal))

    # ==============================================================
    # STEP 1: REPRODUCE PHASE 6 P5 BASELINE EXACTLY
    # ==============================================================
    print("\n" + "=" * 80)
    print("STEP 1: REPRODUCING PHASE 6 P5 BASELINE (Target Macro F0.5 ~= 0.9724)")
    print("=" * 80)
    p5_params = {**Config.LGBM_PARAMS, "num_leaves": 63, "max_depth": 9, "learning_rate": 0.03, "n_estimators": 400}
    oof_p_p5 = np.zeros(len(y_chal), dtype=np.float32)
    t0_p5 = time.time()
    for fold, (train_idx, val_idx) in enumerate(splits_chal):
        clf = lgb.LGBMClassifier(**p5_params)
        clf.fit(X_model_e[train_idx], y_chal[train_idx])
        oof_p_p5[val_idx] = clf.predict_proba(X_model_e[val_idx])[:, 1]

    preds_p5 = defaultdict(set)
    for (sid, cid), p in zip(chal_pairs, oof_p_p5):
        if p >= 0.72: preds_p5[sid].add(cid)

    m_p5 = evaluate_comprehensive_full(gt_dict, preds_p5, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p_p5, 0.72)
    print(f"Phase 6 P5 Reproduction Result:")
    print(f"  Macro F0.5:       {m_p5['macro_f05']:.4f} (Target: 0.9724)")
    print(f"  Precision:        {m_p5['precision']:.4f} (Target: 0.9828)")
    print(f"  Recall:           {m_p5['recall']:.4f} (Target: 0.9505)")
    print(f"  5-Fold Mean:      {m_p5['fold_mean']:.4f} +/- {m_p5['fold_std']:.4f}")
    print(f"  Fold Scores:      {[round(x, 4) for x in m_p5['fold_scores']]}")
    print(f"  TP: {m_p5['tp']:,d} | FP: {m_p5['fp']:,d} | FN: {m_p5['fn']:,d} in {time.time() - t0_p5:.1f}s", flush=True)

    assert abs(m_p5['macro_f05'] - 0.9724) < 0.0005, f"Reproduction mismatch: {m_p5['macro_f05']}"
    print(">>> VERIFIED: Exact Phase 6 P5 reproduction confirmed! Proceeding to Phase 8 Attack.", flush=True)

    log_experiment({
        "experiment_id": "EXP01_Phase6_P5_Reproduction",
        "candidate_config": "Champ_UNION_Sem_K10_Cos30_ZeroProt",
        "feature_count": 44,
        "model_config": "LGBM_depth9_leaves63_lr003_est400",
        "threshold": 0.72,
        "macro_f05": m_p5["macro_f05"],
        "precision": m_p5["precision"],
        "recall": m_p5["recall"],
        "fold_mean": m_p5["fold_mean"],
        "fold_std": m_p5["fold_std"],
        "candidate_recall": cand_rec,
        "tp": m_p5["tp"],
        "fp": m_p5["fp"],
        "fn": m_p5["fn"],
        "zero_match_score": m_p5["zero"],
        "one_match_score": m_p5["one"],
        "multi_match_score": m_p5["multi"],
        "US_score": m_p5["us"],
        "India_score": m_p5["india"],
        "runtime": round(time.time() - t0_p5, 1),
        "notes": "Exact verified reproduction of Phase 6 P5 benchmark baseline"
    }, csv_results_path)

if __name__ == "__main__":
    main()
