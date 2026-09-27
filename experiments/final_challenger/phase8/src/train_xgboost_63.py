"""
Phase 8: XGBoost 5-Fold Training on 63 Features
Amazon ML Challenge 2026 - Business Entity Resolution
"""
import os, sys, time, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PHASE8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(PHASE8_DIR, "artifacts")
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import get_data_paths, parse_ground_truth_fast
from phase8_core import evaluate_comprehensive_full

def main():
    print("=" * 80)
    print("PHASE 8: XGBOOST 5-FOLD TRAINING ON 63 FEATURES (HIST METHOD)")
    print("=" * 80)
    t0_start = time.time()

    print("1. Loading artifacts from disk...", flush=True)
    X_63 = np.load(os.path.join(ART_DIR, "X_model_63.npy"))
    y_chal = np.load(os.path.join(ART_DIR, "y_chal.npy"))
    groups_chal = np.load(os.path.join(ART_DIR, "groups_chal.npy"))
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    chal_pairs = list(zip(df_pairs["s1_id"], df_pairs["cand_id"]))

    with open(os.path.join(ART_DIR, "splits_chal.pkl"), "rb") as f:
        splits_chal = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_s1.pkl"), "rb") as f:
        cached_s1 = pickle.load(f)

    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    s1_dict = {}
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in cached_s1:
                s1_dict[parts[0]] = dict(zip(header, parts))

    N_PAIRS = len(chal_pairs)
    print(f"Loaded {N_PAIRS:,d} pairs with {X_63.shape[1]} features.", flush=True)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 8,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 4,
        "verbosity": 0
    }

    oof_p = np.zeros(N_PAIRS, dtype=np.float32)
    print("\n2. Fitting 5-Fold GroupKFold XGBoost models (tree_method=hist)...", flush=True)
    t0_train = time.time()

    for fold, (train_idx, val_idx) in enumerate(splits_chal):
        t0_f = time.time()
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_63[train_idx], y_chal[train_idx])
        oof_p[val_idx] = clf.predict_proba(X_63[val_idx])[:, 1]
        print(f"  Fold {fold + 1}/5 completed in {time.time() - t0_f:.1f}s", flush=True)

    print(f"All 5 folds trained in {time.time() - t0_train:.1f}s.", flush=True)
    np.save(os.path.join(ART_DIR, "oof_p_xgboost_63.npy"), oof_p)

    # Threshold sweep
    print("\n3. Running threshold sweep [0.66, 0.80] on XGBoost OOF probabilities...", flush=True)
    tau_list = [0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    best_tau = 0.72
    best_macro = 0.0
    best_m = None

    for tau in tau_list:
        preds = defaultdict(set)
        for (sid, cid), p in zip(chal_pairs, oof_p):
            if p >= tau: preds[sid].add(cid)
        m = evaluate_comprehensive_full(gt_dict, preds, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p, tau)
        print(f"  Tau={tau:.2f} -> Macro F0.5={m['macro_f05']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | 5-Fold={m['fold_mean']:.4f} +/- {m['fold_std']:.4f} | FP={m['fp']} | FN={m['fn']}")
        if m['macro_f05'] > best_macro:
            best_macro = m['macro_f05']
            best_tau = tau
            best_m = m

    print("\n" + "=" * 80)
    print(f"BEST XGBOOST 63-FEATURE RESULT (Tau={best_tau:.2f}):")
    print("=" * 80)
    print(f"  Macro F0.5:       {best_m['macro_f05']:.4f}")
    print(f"  Precision:        {best_m['precision']:.4f}")
    print(f"  Recall:           {best_m['recall']:.4f}")
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

    # Append to experiment_results.csv
    csv_results_path = os.path.join(PHASE8_DIR, "experiment_results.csv")
    df_row = pd.DataFrame([{
        "experiment_id": f"EXP05_XGBoost_63_Tau{int(best_tau*100)}",
        "candidate_config": "Champ_UNION_Sem_K10_Cos30_ZeroProt",
        "feature_count": 63,
        "model_config": "XGBoost_depth8_lr003_est400_hist",
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
        "notes": f"XGBoost 5-fold hist model on 63 features. Best tau={best_tau:.2f}"
    }])
    df_row.to_csv(csv_results_path, mode="a", header=False, index=False)
    print(f"Appended EXP05 result to: {csv_results_path}")

if __name__ == "__main__":
    main()
