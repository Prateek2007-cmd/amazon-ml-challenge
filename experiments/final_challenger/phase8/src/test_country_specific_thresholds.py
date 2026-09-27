"""
Phase 8: Country-Specific Threshold Optimization & Slice Maximization
Amazon ML Challenge 2026 - Business Entity Resolution
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
from evaluate import compute_comprehensive_metrics

def main():
    print("Loading data for country threshold grid search...", flush=True)
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    chal_pairs = list(zip(df_pairs["s1_id"], df_pairs["cand_id"]))
    oof_p_63 = np.load(os.path.join(ART_DIR, "oof_p_63.npy"))
    p_xgb = np.load(os.path.join(ART_DIR, "oof_p_xgboost_63.npy"))
    p_cb = np.load(os.path.join(ART_DIR, "oof_p_catboost_63.npy"))

    with open(os.path.join(ART_DIR, "cached_s1.pkl"), "rb") as f: cached_s1 = pickle.load(f)

    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)

    # Let's test single best model (LightGBM) and best ensemble (0.5 LGB + 0.2 CB + 0.3 XGB)
    p_blend = 0.50 * oof_p_63 + 0.20 * p_cb + 0.30 * p_xgb

    models_to_test = [
        ("LightGBM_63", oof_p_63),
        ("TriEnsemble_63", p_blend)
    ]

    for model_name, p_arr in models_to_test:
        print(f"\n=======================================================")
        print(f"GRID SEARCH FOR {model_name}:")
        print(f"=======================================================")
        
        best_macro = 0.0
        best_tau_us = 0.74
        best_tau_in = 0.74
        best_m = None

        tau_us_list = [0.72, 0.74, 0.76, 0.78, 0.80]
        tau_in_list = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74]

        for tau_us in tau_us_list:
            for tau_in in tau_in_list:
                preds = defaultdict(set)
                for (sid, cid), p in zip(chal_pairs, p_arr):
                    c = cached_s1.get(sid, {}).get("country", "US")
                    tau = tau_in if c == "INDIA" else tau_us
                    if p >= tau:
                        preds[sid].add(cid)
                m = compute_comprehensive_metrics(gt_dict, preds, beta=0.5)
                macro = m["macro_f05"]
                if macro > best_macro:
                    best_macro = macro
                    best_tau_us = tau_us
                    best_tau_in = tau_in
                    best_m = m

        print(f"OPTIMAL COUNTRY THRESHOLDS for {model_name}:")
        print(f"  Tau_US = {best_tau_us:.2f} | Tau_India = {best_tau_in:.2f}")
        print(f"  Macro F0.5: {best_m['macro_f05']:.4f} (Global uniform tau was 0.9784)")
        print(f"  Precision:  {best_m['macro_precision']:.4f}")
        print(f"  Recall:     {best_m['macro_recall']:.4f}")
        print(f"  TP: {best_m['tp']:,d} | FP: {best_m['fp']:,d} | FN: {best_m['fn']:,d}")

if __name__ == "__main__":
    main()
