"""
Analyze the 289 False Positives of the 63-Feature Model
Amazon ML Challenge 2026 - Business Entity Resolution
"""
import os, sys, pickle
from collections import defaultdict
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PHASE8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(PHASE8_DIR, "artifacts")
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from data_loader import get_data_paths, parse_ground_truth_fast

def main():
    print("Loading artifacts...", flush=True)
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    chal_pairs = list(zip(df_pairs["s1_id"], df_pairs["cand_id"]))
    oof_p_63 = np.load(os.path.join(ART_DIR, "oof_p_63.npy"))
    y_chal = np.load(os.path.join(ART_DIR, "y_chal.npy"))

    with open(os.path.join(ART_DIR, "cached_s1.pkl"), "rb") as f:
        cached_s1 = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_cands.pkl"), "rb") as f:
        cached_cands = pickle.load(f)

    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)

    # Find false positives at tau = 0.74
    fps = []
    for i, ((sid, cid), p, y) in enumerate(zip(chal_pairs, oof_p_63, y_chal)):
        if p >= 0.74 and y == 0:
            e1 = cached_s1.get(sid, {})
            e2 = cached_cands.get(cid, {})
            t_set = gt_dict.get(sid, set())
            fps.append({
                "s1_id": sid,
                "cand_id": cid,
                "prob": p,
                "s1_has_true_matches": len(t_set) > 0,
                "s1_num_true": len(t_set),
                "s1_name": e1.get("norm_n", ""),
                "cand_name": e2.get("norm_n", ""),
                "s1_addr": e1.get("norm_a", "")[:40],
                "cand_addr": e2.get("norm_a", "")[:40],
                "s1_pin": e1.get("pincode", ""),
                "cand_pin": e2.get("pincode", ""),
                "country": e1.get("country", ""),
            })

    print(f"\nTOTAL FALSE POSITIVES at tau=0.74: {len(fps)}")
    fp_zero_match = [x for x in fps if not x["s1_has_true_matches"]]
    fp_positive_match = [x for x in fps if x["s1_has_true_matches"]]
    print(f"  False positives on ZERO-MATCH entities: {len(fp_zero_match)} ({len(fp_zero_match)/len(fps)*100:.1f}%)")
    print(f"  False positives on ENTITIES WITH TRUE MATCHES: {len(fp_positive_match)} ({len(fp_positive_match)/len(fps)*100:.1f}%)")

    # Check pin conflict among false positives
    pin_conflicts = [x for x in fps if x["s1_pin"] and x["cand_pin"] and x["s1_pin"] != x["cand_pin"]]
    print(f"  False positives with conflicting PIN codes: {len(pin_conflicts)}")

    # Check probability distribution of FPs
    print("\nProbability Distribution of False Positives:")
    for low, high in [(0.74, 0.76), (0.76, 0.78), (0.78, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.00)]:
        cnt = sum(1 for x in fps if low <= x["prob"] < high)
        print(f"  [{low:.2f}, {high:.2f}): {cnt}")

    print("\nSample False Positives (First 15):")
    for x in fps[:15]:
        print(f"P={x['prob']:.3f} | S1 has GT: {x['s1_has_true_matches']} | S1: [{x['s1_name']}] | CAND: [{x['cand_name']}] | Addr1: [{x['s1_addr']}] vs Addr2: [{x['cand_addr']}] | PINs: {x['s1_pin']} vs {x['cand_pin']}")

if __name__ == "__main__":
    main()
