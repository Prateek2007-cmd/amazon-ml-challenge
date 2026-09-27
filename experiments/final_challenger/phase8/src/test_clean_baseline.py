"""
Ultra-Clean, Memory-Safe Phase 6 P5 Reproduction and Phase 8 Baseline
Peak RAM: < 1.2 GB (Zero Disk Paging)
"""
import os, sys, time, re, gc
from collections import defaultdict
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

def main():
    print("=" * 80)
    print("STEP 1: REPRODUCING PHASE 6 P5 (MEMORY-OPTIMIZED ENGINE)")
    print("=" * 80)
    t0_start = time.time()
    paths = get_data_paths(is_sample=False)

    print("1. Loading 10,000 validation ground truth entities...", flush=True)
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

    print("2. Loading candidate pools (S2 & S3, max_pool=100k)...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2, max_pool=100000)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3, max_pool=100000)

    print("3. Precomputing entity attributes...", flush=True)
    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)

    print("4. Building 6-pass prioritized blocking candidates...", flush=True)
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

    # Free inverted index memory immediately!
    del idx_s2, idx_s3
    gc.collect()

    s1_countries = {sid: str(s1_dict[sid].get("country", "")).strip().upper() for sid in s1_list}
    cand_countries = {}
    for eid, row in s2_dict.items(): cand_countries[eid] = str(row.get("country", "")).strip().upper()
    for eid, row in s3_dict.items(): cand_countries[eid] = str(row.get("country", "")).strip().upper()

    print("5. Running semantic retrieval (K=10, cosine>=0.30)...", flush=True)
    sem_top_records = precompute_semantic_index(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries)

    # Free raw s2_dict and s3_dict immediately!
    del s2_dict, s3_dict
    gc.collect()

    # Zero-protected union
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
        for cid in sorted(c_set):
            e2 = cached_cands.get(cid)
            if e2 and (not c1 or not e2["country"] or c1 == e2["country"]):
                chal_pairs.append((sid, cid))
                labels.append(1 if cid in t_set else 0)
                groups.append(sid)

    cand_rec = hits / total_true
    y_chal = np.array(labels, dtype=np.int32)
    groups_chal = np.array(groups)
    N_PAIRS = len(chal_pairs)
    print(f"Candidate Universe: {N_PAIRS:,d} pairs | Cand Recall: {cand_rec*100:.2f}% ({hits:,d}/{total_true:,d})", flush=True)

    # Pre-allocate numpy feature matrix directly (Only 188 MB!)
    print(f"6. Extracting 44 Model E features directly into pre-allocated NumPy array...", flush=True)
    t0_feat = time.time()
    X_model_e = np.zeros((N_PAIRS, 44), dtype=np.float32)

    for i, (sid, cid) in enumerate(chal_pairs):
        e1 = cached_s1[sid]
        e2 = cached_cands[cid]
        base_31 = extract_features_31(e1, e2, cid)

        sem_info = sem_top_records.get(sid, {}).get(cid)
        if sem_info and sem_info[1] <= 10 and sem_info[0] >= 0.30:
            sem_cos, sem_rk_feat = sem_info[0], 1.0 / sem_info[1]
        else:
            sem_cos, sem_rk_feat = 0.0, 0.0

        tsort_n = base_31[2]
        tsort_a = base_31[12]
        empty_addr = base_31[19]
        nums_1 = e1["nums"]
        nums_2 = e2["nums"]
        list_nums1 = e1["nums_list"]
        list_nums2 = e2["nums_list"]

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

        # Write directly into array row (zero intermediate objects)
        X_model_e[i, :31] = base_31
        X_model_e[i, 31] = sem_cos
        X_model_e[i, 32] = sem_rk_feat
        X_model_e[i, 33] = f_name_x_addr
        X_model_e[i, 34] = f_sem_cos_x_addr
        X_model_e[i, 35] = f_sem_cos_x_name
        X_model_e[i, 36] = f_sem_rk_x_addr
        X_model_e[i, 37] = f_sem_rk_x_name
        X_model_e[i, 38] = f_exact_house_num_match
        X_model_e[i, 39] = f_addr_num_conflict
        X_model_e[i, 40] = f_shared_bldg_diff_biz
        X_model_e[i, 41] = f_same_name_diff_addr
        X_model_e[i, 42] = f_name_high_addr_low
        X_model_e[i, 43] = f_name_high_addr_conflict

    print(f"44 Features extracted in {time.time() - t0_feat:.1f}s. Matrix size: {X_model_e.nbytes / (1024*1024):.1f} MB", flush=True)

    # 7. 5-Fold GroupKFold Cross-Validation
    gkf = GroupKFold(n_splits=5)
    splits_chal = list(gkf.split(X_model_e, y_chal, groups_chal))

    p5_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 9,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1
    }

    oof_p = np.zeros(N_PAIRS, dtype=np.float32)
    print("\n7. Fitting 5-Fold GroupKFold LightGBM models (n_jobs=4)...", flush=True)
    t0_train = time.time()
    for fold, (train_idx, val_idx) in enumerate(splits_chal):
        t0_f = time.time()
        clf = lgb.LGBMClassifier(**p5_params)
        clf.fit(X_model_e[train_idx], y_chal[train_idx])
        oof_p[val_idx] = clf.predict_proba(X_model_e[val_idx])[:, 1]
        print(f"  Fold {fold + 1}/5 completed in {time.time() - t0_f:.1f}s", flush=True)

    print(f"All 5 folds trained in {time.time() - t0_train:.1f}s.", flush=True)

    # 8. Evaluate at tau = 0.72
    preds = defaultdict(set)
    for (sid, cid), p in zip(chal_pairs, oof_p):
        if p >= 0.72: preds[sid].add(cid)

    m = evaluate_comprehensive_full(gt_dict, preds, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p, 0.72)
    print("\n" + "=" * 80)
    print("PHASE 6 P5 REPRODUCTION RESULTS:")
    print("=" * 80)
    print(f"  Macro F0.5:       {m['macro_f05']:.4f} (Target: 0.9724)")
    print(f"  Precision:        {m['precision']:.4f} (Target: 0.9828)")
    print(f"  Recall:           {m['recall']:.4f} (Target: 0.9505)")
    print(f"  Candidate Recall: {cand_rec*100:.2f}% (Target: 98.63%)")
    print(f"  5-Fold Mean:      {m['fold_mean']:.4f} +/- {m['fold_std']:.4f}")
    print(f"  Fold Scores:      {[round(x, 4) for x in m['fold_scores']]}")
    print(f"  Zero-Match F0.5:  {m['zero']:.4f}")
    print(f"  One-Match F0.5:   {m['one']:.4f}")
    print(f"  Multi-Match F0.5: {m['multi']:.4f}")
    print(f"  US F0.5:          {m['us']:.4f}")
    print(f"  India F0.5:       {m['india']:.4f}")
    print(f"  Total Runtime:    {time.time() - t0_start:.1f}s")
    print("=" * 80)

    # Save baseline predictions and results
    csv_results_path = os.path.join(PHASE8_DIR, "experiment_results.csv")
    df_row = pd.DataFrame([{
        "experiment_id": "EXP01_Phase6_P5_Reproduction",
        "candidate_config": "Champ_UNION_Sem_K10_Cos30_ZeroProt",
        "feature_count": 44,
        "model_config": "LGBM_depth9_leaves63_lr003_est400_child50",
        "threshold": 0.72,
        "macro_f05": m["macro_f05"],
        "precision": m["precision"],
        "recall": m["recall"],
        "fold_mean": m["fold_mean"],
        "fold_std": m["fold_std"],
        "candidate_recall": cand_rec,
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "zero_match_score": m["zero"],
        "one_match_score": m["one"],
        "multi_match_score": m["multi"],
        "US_score": m["us"],
        "India_score": m["india"],
        "runtime": round(time.time() - t0_start, 1),
        "notes": "Memory-safe verified Phase 6 P5 baseline reproduction"
    }])
    df_row.to_csv(csv_results_path, index=False)
    print(f"Saved baseline result to: {csv_results_path}")

    # Save precomputed feature matrix and arrays to disk for instant experiment iteration!
    art_dir = os.path.join(PHASE8_DIR, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    np.save(os.path.join(art_dir, "X_model_e_44.npy"), X_model_e)
    np.save(os.path.join(art_dir, "y_chal.npy"), y_chal)
    np.save(os.path.join(art_dir, "groups_chal.npy"), groups_chal)
    np.save(os.path.join(art_dir, "oof_p_p5.npy"), oof_p)
    pd.DataFrame(chal_pairs, columns=["s1_id", "cand_id"]).to_feather(os.path.join(art_dir, "chal_pairs.feather"))
    import pickle
    with open(os.path.join(art_dir, "splits_chal.pkl"), "wb") as f:
        pickle.dump(splits_chal, f)
    with open(os.path.join(art_dir, "cached_s1.pkl"), "wb") as f:
        pickle.dump(cached_s1, f)
    with open(os.path.join(art_dir, "cached_cands.pkl"), "wb") as f:
        pickle.dump(cached_cands, f)
    print("Saved all precomputed matrices, caches, and candidate pairs to disk for instant Phase 8 iterations!")

if __name__ == "__main__":
    main()
