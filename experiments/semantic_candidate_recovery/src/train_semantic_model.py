"""
Train and Save Semantic LightGBM Model.
Uses exact verified Config C:
- 10,000 S1 training subsample
- Champion candidates UNION Semantic C_norm_name_addr K=5 (zero-protected)
- 31 targeted features
- Verifies 5-fold CV Macro F0.5 ~ 0.9700 and tau = 0.68
- Saves final trained model to experiments/semantic_candidate_recovery/models/semantic_lgbm_model.joblib
"""
import os, sys, time, re
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize
import lightgbm as lgb
import joblib
import psutil

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from config import Config
from data_loader import get_data_paths, parse_ground_truth_fast
from evaluate import compute_comprehensive_metrics

sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "semantic_candidate_recovery", "src"))
from production_pipeline import (
    precompute_entity, extract_champion_keys, extract_features_31,
    build_rep_text, build_blocking_index, generate_champion_candidates,
    semantic_retrieve_batch
)

def get_ram():
    return f"{psutil.virtual_memory().available / (1024**3):.2f} GB avail ({psutil.virtual_memory().percent}% used)"

def main():
    print("=" * 80)
    print("TRAINING SEMANTIC LIGHTGBM MODEL (CONFIG C)")
    print("=" * 80)
    print("Initial RAM:", get_ram())
    t0 = time.time()

    paths = get_data_paths(is_sample=False)
    models_dir = os.path.join(REPO_ROOT, "experiments", "semantic_candidate_recovery", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "semantic_lgbm_model.joblib")

    # 1. Load 10k Training Data
    print("\n[STEP 1] Loading 10k Ground Truth and Subsample...", flush=True)
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

    print("  Loading S2 and S3 train pools...", flush=True)
    s2_train = load_pool_stream(paths["train_s2"], needed_s2)
    s3_train = load_pool_stream(paths["train_s3"], needed_s3)
    print(f"  Loaded: S1={len(s1_train):,d}, S2={len(s2_train):,d}, S3={len(s3_train):,d}", flush=True)
    print("  RAM:", get_ram())

    # Precompute representations
    print("\n[STEP 2] Precomputing representations...", flush=True)
    cached_s1_train = {sid: precompute_entity(row) for sid, row in s1_train.items()}
    cached_cands_train = {}
    for eid, row in s2_train.items(): cached_cands_train[eid] = precompute_entity(row)
    for eid, row in s3_train.items(): cached_cands_train[eid] = precompute_entity(row)

    # Champion candidate sets
    print("\n[STEP 3] Champion blocking...", flush=True)
    idx_s2_train = build_blocking_index(s2_train)
    idx_s3_train = build_blocking_index(s3_train)
    s1_train_list = list(s1_train.keys())
    champion_cands_train = {}
    for s1_id in s1_train_list:
        champion_cands_train[s1_id] = generate_champion_candidates(s1_id, s1_train[s1_id], idx_s2_train, idx_s3_train)
    champ_hits = sum(len(gt_dict.get(sid, set()) & champion_cands_train[sid]) for sid in s1_train_list)
    print(f"  Champion candidate recall: {champ_hits/total_true*100:.2f}%", flush=True)

    # Semantic retrieval (zero-protected: only query for entities with >= 1 champion candidate)
    print("\n[STEP 4] Semantic retrieval (zero-protected)...", flush=True)
    query_ids_train = [sid for sid in s1_train_list if len(champion_cands_train[sid]) > 0]
    all_cand_train = {**s2_train, **s3_train}
    sem_cands_train = semantic_retrieve_batch(
        query_ids_train, s1_train, list(all_cand_train.keys()), all_cand_train, K=5, cos_threshold=0.30
    )

    # Union candidates
    union_cands_train = {}
    for sid in s1_train_list:
        if len(champion_cands_train[sid]) > 0:
            union_cands_train[sid] = champion_cands_train[sid] | sem_cands_train.get(sid, set())
        else:
            union_cands_train[sid] = champion_cands_train[sid]

    union_hits = sum(len(gt_dict.get(sid, set()) & union_cands_train[sid]) for sid in s1_train_list)
    union_cand_recall = union_hits / total_true
    print(f"  Union candidate recall: {union_cand_recall*100:.2f}% (Expected ~98.05%)", flush=True)

    # Generate pairs and 31 features
    print("\n[STEP 5] Extracting 31 features...", flush=True)
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
    print(f"  Total pairs: {len(pairs):,d} (Pos: {sum(labels):,d}, Neg: {len(labels)-sum(labels):,d})", flush=True)

    # 5-fold CV verification
    print("\n[STEP 6] 5-Fold GroupKFold Cross-Validation...", flush=True)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups_arr))
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(splits):
        clf_fold = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf_fold.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = clf_fold.predict_proba(X[val_idx])[:, 1]

    # Evaluate at tau = 0.68
    tau = 0.68
    preds_cv = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs):
        if p >= tau: preds_cv[sid].add(cid)
    m_cv = compute_comprehensive_metrics(gt_dict, preds_cv, beta=0.5)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs = [pairs[i] for i in val_idx]
        val_s1_set = set(groups_arr[val_idx])
        val_gt = {sid: gt_dict[sid] for sid in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, oof_probs[val_idx]):
            if p >= tau: fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=0.5)
        fold_scores.append(m_fold["macro_f05"])

    print(f"  CV Macro F0.5 at tau={tau:.2f}: {m_cv['macro_f05']:.4f} (Prec: {m_cv['macro_precision']:.4f}, Rec: {m_cv['macro_recall']:.4f})", flush=True)
    print(f"  Fold scores: {[round(s, 4) for s in fold_scores]} | Mean: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}", flush=True)

    # Train final model on ALL 10k data
    print("\n[STEP 7] Training Final Model on all 10k data...", flush=True)
    final_clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
    final_clf.fit(X, y)

    # Save model and metadata
    joblib.dump(final_clf, model_save_path)
    print(f"  Model saved to: {model_save_path} ({os.path.getsize(model_save_path):,d} bytes)", flush=True)

    meta = {
        "tau": tau,
        "features": 31,
        "union_cand_recall": union_cand_recall,
        "cv_macro_f05": m_cv["macro_f05"],
        "cv_precision": m_cv["macro_precision"],
        "cv_recall": m_cv["macro_recall"],
        "cv_fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
    }
    meta_path = os.path.join(models_dir, "model_meta.joblib")
    joblib.dump(meta, meta_path)
    print(f"  Metadata saved to: {meta_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\nModel training & verification complete in {elapsed:.1f}s ({elapsed/60:.2f} min).")
    print("Final RAM:", get_ram())

if __name__ == "__main__":
    main()
