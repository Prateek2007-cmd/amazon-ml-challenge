"""
Phase 8: LightGBM + CatBoost Ensemble & Threshold Optimization
Amazon ML Challenge 2026 - Business Entity Resolution

Blends LightGBM and CatBoost OOF probabilities across 5 folds and sweeps weights & thresholds.
"""
import os, sys, time, pickle
from collections import defaultdict
import numpy as np
import pandas as pd

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
    print("PHASE 8: LIGHTGBM + CATBOOST ENSEMBLE BLEND SPRINT")
    print("=" * 80)
    t0_start = time.time()

    print("1. Loading artifacts and model OOF predictions...", flush=True)
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    chal_pairs = list(zip(df_pairs["s1_id"], df_pairs["cand_id"]))
    oof_p_lgb = np.load(os.path.join(ART_DIR, "oof_p_63.npy"))
    oof_p_cb = np.load(os.path.join(ART_DIR, "oof_p_catboost_63.npy"))
    groups_chal = np.load(os.path.join(ART_DIR, "groups_chal.npy"))

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

    # Evaluate correlation between model predictions
    corr = np.corrcoef(oof_p_lgb, oof_p_cb)[0, 1]
    print(f"Pearson Correlation between LightGBM and CatBoost: {corr:.4f}", flush=True)

    # Grid search blend weights and thresholds
    print("\n2. Grid search across blend weights and thresholds...", flush=True)
    weights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    taus = [0.68, 0.70, 0.72, 0.74, 0.76, 0.78]

    best_macro = 0.0
    best_config = None
    best_m = None
    best_blend_p = None

    for w in weights:
        p_blend = w * oof_p_lgb + (1.0 - w) * oof_p_cb
        for tau in taus:
            preds = defaultdict(set)
            for (sid, cid), p in zip(chal_pairs, p_blend):
                if p >= tau: preds[sid].add(cid)
            m = evaluate_comprehensive_full(gt_dict, preds, s1_dict, splits_chal, chal_pairs, groups_chal, p_blend, tau)
            is_better = m['macro_f05'] > best_macro
            mark = " *** NEW BEST ***" if is_better else ""
            print(f"  w_LGB={w:.1f} w_CB={1-w:.1f} Tau={tau:.2f} -> Macro F0.5={m['macro_f05']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | 5-Fold={m['fold_mean']:.4f} +/- {m['fold_std']:.4f} | FP={m['fp']} | FN={m['fn']}{mark}")
            if is_better:
                best_macro = m['macro_f05']
                best_config = (w, tau)
                best_m = m
                best_blend_p = p_blend

    print("\n" + "=" * 80)
    print(f"BEST ENSEMBLE RESULT: w_LGB={best_config[0]:.2f}, w_CB={1-best_config[0]:.2f}, Tau={best_config[1]:.2f}")
    print("=" * 80)
    print(f"  Macro F0.5:       {best_m['macro_f05']:.4f} (LGBM alone: 0.9784, Baseline: 0.9723)")
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

    # Save blended OOF predictions
    np.save(os.path.join(ART_DIR, "oof_p_ensemble_blend.npy"), best_blend_p)

    # Append to experiment_results.csv
    csv_results_path = os.path.join(PHASE8_DIR, "experiment_results.csv")
    df_row = pd.DataFrame([{
        "experiment_id": f"EXP04_Ensemble_LGB{int(best_config[0]*100)}_CB{int((1-best_config[0])*100)}_Tau{int(best_config[1]*100)}",
        "candidate_config": "Champ_UNION_Sem_K10_Cos30_ZeroProt",
        "feature_count": 63,
        "model_config": f"Ensemble_LGBM_{best_config[0]:.1f}+CatBoost_{1-best_config[0]:.1f}",
        "threshold": best_config[1],
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
        "notes": f"Ensemble blend of LightGBM + CatBoost on 63 features. Best w={best_config[0]:.1f}, tau={best_config[1]:.2f}"
    }])
    df_row.to_csv(csv_results_path, mode="a", header=False, index=False)
    print(f"Appended EXP04 result to: {csv_results_path}")

if __name__ == "__main__":
    main()
