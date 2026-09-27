"""
Test Address-Only Collision Veto (Different Business in Same Building)
Amazon ML Challenge 2026 - Business Entity Resolution
"""
import os, sys, pickle
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
    print("Loading artifacts...", flush=True)
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    chal_pairs = list(zip(df_pairs["s1_id"], df_pairs["cand_id"]))
    X_base = np.load(os.path.join(ART_DIR, "X_model_e_44.npy"))
    oof_p_63 = np.load(os.path.join(ART_DIR, "oof_p_63.npy"))
    y_chal = np.load(os.path.join(ART_DIR, "y_chal.npy"))
    groups_chal = np.load(os.path.join(ART_DIR, "groups_chal.npy"))

    with open(os.path.join(ART_DIR, "splits_chal.pkl"), "rb") as f:
        splits_chal = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_s1.pkl"), "rb") as f:
        cached_s1 = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_cands.pkl"), "rb") as f:
        cached_cands = pickle.load(f)

    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    s1_dict = {}
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in cached_s1:
                s1_dict[parts[0]] = dict(zip(header, parts))

    # Base metrics at tau = 0.74
    base_preds = defaultdict(set)
    for (sid, cid), p in zip(chal_pairs, oof_p_63):
        if p >= 0.74: base_preds[sid].add(cid)

    m_base = evaluate_comprehensive_full(gt_dict, base_preds, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p_63, 0.74)
    print(f"BASELINE (Tau=0.74): Macro F0.5={m_base['macro_f05']:.4f} | Prec={m_base['precision']:.4f} | Rec={m_base['recall']:.4f} | FP={m_base['fp']} | FN={m_base['fn']}")

    # Let's check false positives vs true positives by name similarity
    # base_31[1] is ratio_name, base_31[2] is token_sort_name, base_31[3] is token_set_name
    ratio_names = X_base[:, 1]
    tsort_names = X_base[:, 2]
    tset_names = X_base[:, 3]
    phone_matches = X_base[:, 21]
    url_matches = X_base[:, 23]

    print("\nAnalyzing Name Similarity Distribution of Predictions (p >= 0.74):")
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        # Count FPs and TPs below this name similarity threshold
        mask_below = (ratio_names < thresh) & (tsort_names < thresh)
        fps_below = np.sum((oof_p_63 >= 0.74) & (y_chal == 0) & mask_below)
        tps_below = np.sum((oof_p_63 >= 0.74) & (y_chal == 1) & mask_below)
        print(f"  Name Sim < {thresh:.2f}: FPs = {fps_below:3d} | TPs = {tps_below:3d}")

    # Sweep veto threshold
    print("\nTesting Name-Veto on Predictions (p >= 0.74):")
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50]:
        preds_veto = defaultdict(set)
        for (sid, cid), p, r_n, ts_n in zip(chal_pairs, oof_p_63, ratio_names, tsort_names):
            if p >= 0.74:
                # If name similarity is below thresh, veto!
                if r_n < thresh and ts_n < thresh:
                    continue
                preds_veto[sid].add(cid)
        m = evaluate_comprehensive_full(gt_dict, preds_veto, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p_63, 0.74)
        print(f"  Veto Thresh={thresh:.2f} -> Macro F0.5={m['macro_f05']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | FP={m['fp']} | FN={m['fn']}")

if __name__ == "__main__":
    main()
