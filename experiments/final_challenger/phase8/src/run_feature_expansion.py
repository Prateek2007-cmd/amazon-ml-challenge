"""
Phase 8: High-Leverage Feature Expansion Sprint
Amazon ML Challenge 2026 - Business Entity Resolution

Trains 5-Fold GroupKFold LightGBM on 63 features (44 Model E + 19 targeted discrimination features).
Evaluates comprehensive metrics at multiple thresholds.
"""
import os, sys, time, pickle, gc
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PHASE8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(PHASE8_DIR, "artifacts")
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import get_data_paths, parse_ground_truth_fast
from phase8_core import evaluate_comprehensive_full
from feature_engine import compute_corpus_token_idf, extract_advanced_features_19

def main():
    print("=" * 80)
    print("PHASE 8: HIGH-LEVERAGE FEATURE EXPANSION SPRINT (44 -> 63 FEATURES)")
    print("=" * 80)
    t0_start = time.time()

    # 1. Load precomputed baseline artifacts
    print("1. Loading cached artifacts from disk...", flush=True)
    X_base = np.load(os.path.join(ART_DIR, "X_model_e_44.npy"))
    y_chal = np.load(os.path.join(ART_DIR, "y_chal.npy"))
    groups_chal = np.load(os.path.join(ART_DIR, "groups_chal.npy"))
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    chal_pairs = list(zip(df_pairs["s1_id"], df_pairs["cand_id"]))

    with open(os.path.join(ART_DIR, "splits_chal.pkl"), "rb") as f:
        splits_chal = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_s1.pkl"), "rb") as f:
        cached_s1 = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_cands.pkl"), "rb") as f:
        cached_cands = pickle.load(f)

    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())
    s1_dict = {}
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in cached_s1:
                s1_dict[parts[0]] = dict(zip(header, parts))

    N_PAIRS = len(chal_pairs)
    print(f"Loaded {N_PAIRS:,d} candidate pairs and {X_base.shape[1]} base features.", flush=True)

    # 2. Compute corpus token IDF
    print("2. Computing smooth token IDF across candidate corpus...", flush=True)
    all_entities = {**cached_s1, **cached_cands}
    idf_dict, default_idf = compute_corpus_token_idf(all_entities)
    print(f"Vocabulary: {len(idf_dict):,d} distinct tokens indexed. Default IDF: {default_idf:.2f}", flush=True)

    # 3. Extract 19 targeted features
    print("3. Extracting 19 high-leverage features directly into NumPy matrix...", flush=True)
    t0_feat = time.time()
    X_new = np.zeros((N_PAIRS, 19), dtype=np.float32)

    for i, (sid, cid) in enumerate(chal_pairs):
        e1 = cached_s1[sid]
        e2 = cached_cands[cid]
        base_31 = X_base[i, :31]
        X_new[i] = extract_advanced_features_19(e1, e2, base_31, idf_dict, default_idf)

    print(f"19 Features extracted in {time.time() - t0_feat:.1f}s. Matrix size: {X_new.nbytes / 1024**2:.1f} MB", flush=True)

    # 4. Combine into 63-feature matrix
    X_combined = np.hstack([X_base, X_new])
    print(f"Combined Feature Matrix shape: {X_combined.shape} ({X_combined.nbytes / 1024**2:.1f} MB)", flush=True)
    np.save(os.path.join(ART_DIR, "X_model_63.npy"), X_combined)

    # 5. Train 5-Fold GroupKFold LightGBM
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
    print("\n4. Fitting 5-Fold GroupKFold LightGBM on 63 features (n_jobs=4)...", flush=True)
    t0_train = time.time()
    feature_importances = np.zeros(X_combined.shape[1], dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(splits_chal):
        t0_f = time.time()
        clf = lgb.LGBMClassifier(**p5_params)
        clf.fit(X_combined[train_idx], y_chal[train_idx])
        oof_p[val_idx] = clf.predict_proba(X_combined[val_idx])[:, 1]
        feature_importances += clf.feature_importances_
        print(f"  Fold {fold + 1}/5 completed in {time.time() - t0_f:.1f}s", flush=True)

    print(f"All 5 folds trained in {time.time() - t0_train:.1f}s.", flush=True)
    np.save(os.path.join(ART_DIR, "oof_p_63.npy"), oof_p)

    # 6. Threshold Sweep on OOF probabilities
    print("\n5. Running threshold sweep [0.66, 0.80] on 63-feature OOF probabilities...", flush=True)
    tau_list = [0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    best_tau = 0.72
    best_macro = 0.0
    best_m = None

    results_table = []
    for tau in tau_list:
        preds = defaultdict(set)
        for (sid, cid), p in zip(chal_pairs, oof_p):
            if p >= tau: preds[sid].add(cid)
        m = evaluate_comprehensive_full(gt_dict, preds, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p, tau)
        print(f"  Tau={tau:.2f} -> Macro F0.5={m['macro_f05']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | 5-Fold={m['fold_mean']:.4f} +/- {m['fold_std']:.4f} | FP={m['fp']} | FN={m['fn']}")
        results_table.append((tau, m))
        if m['macro_f05'] > best_macro:
            best_macro = m['macro_f05']
            best_tau = tau
            best_m = m

    print("\n" + "=" * 80)
    print(f"BEST 63-FEATURE RESULT (Tau={best_tau:.2f}):")
    print("=" * 80)
    print(f"  Macro F0.5:       {best_m['macro_f05']:.4f} (Baseline: 0.9719)")
    print(f"  Precision:        {best_m['precision']:.4f} (Baseline: 0.9826)")
    print(f"  Recall:           {best_m['recall']:.4f} (Baseline: 0.9495)")
    print(f"  5-Fold Mean:      {best_m['fold_mean']:.4f} +/- {best_m['fold_std']:.4f}")
    print(f"  Fold Scores:      {[round(x, 4) for x in best_m['fold_scores']]}")
    print(f"  Zero-Match F0.5:  {best_m['zero']:.4f}")
    print(f"  One-Match F0.5:   {best_m['one']:.4f}")
    print(f"  Multi-Match F0.5: {best_m['multi']:.4f}")
    print(f"  US F0.5:          {best_m['us']:.4f}")
    print(f"  India F0.5:       {best_m['india']:.4f}")
    print(f"  Total TP:         {best_m['tp']:,d} | FP: {best_m['fp']:,d} | FN: {best_m['fn']:,d}")
    print(f"  Total Runtime:    {time.time() - t0_start:.1f}s")
    print("=" * 80)

    # 7. Append to experiment_results.csv
    csv_results_path = os.path.join(PHASE8_DIR, "experiment_results.csv")
    df_row = pd.DataFrame([{
        "experiment_id": f"EXP02_Features_63_Tau{int(best_tau*100)}",
        "candidate_config": "Champ_UNION_Sem_K10_Cos30_ZeroProt",
        "feature_count": 63,
        "model_config": "LGBM_depth9_leaves63_lr003_est400_child50",
        "threshold": best_tau,
        "macro_f05": best_m["macro_f05"],
        "precision": best_m["precision"],
        "recall": best_m["recall"],
        "fold_mean": best_m["fold_mean"],
        "fold_std": best_m["fold_std"],
        "candidate_recall": 0.986323,
        "tp": best_m["tp"],
        "fp": best_m["fp"],
        "fn": best_m["fn"],
        "zero_match_score": best_m["zero"],
        "one_match_score": best_m["one"],
        "multi_match_score": best_m["multi"],
        "US_score": best_m["us"],
        "India_score": best_m["india"],
        "runtime": round(time.time() - t0_start, 1),
        "notes": f"Added 19 targeted features (IDF, PIN, Containment, URL, Empty-Addr). Best tau={best_tau:.2f}"
    }])
    df_row.to_csv(csv_results_path, mode="a", header=False, index=False)
    print(f"Appended EXP02 result to: {csv_results_path}")

if __name__ == "__main__":
    main()
