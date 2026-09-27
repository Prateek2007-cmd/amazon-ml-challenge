"""
Analyze Candidate Misses & Recovery Frontier
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

from data_loader import get_data_paths, parse_ground_truth_fast

def main():
    print("Loading data for candidate miss analysis...", flush=True)
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    existing_pairs = set(zip(df_pairs["s1_id"], df_pairs["cand_id"]))

    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())

    missed_gt = []
    for sid, matches in gt_dict.items():
        for cid in matches:
            if (sid, cid) not in existing_pairs:
                missed_gt.append((sid, cid))

    print(f"Total True Pairs: {total_true:,d}")
    print(f"Retrieved Pairs:  {len(existing_pairs):,d}")
    print(f"Missed GT Pairs:  {len(missed_gt):,d} ({len(missed_gt)/total_true*100:.2f}%)")

    # Load caches
    with open(os.path.join(ART_DIR, "cached_s1.pkl"), "rb") as f: cached_s1 = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_cands.pkl"), "rb") as f: cached_cands = pickle.load(f)

    # Inspect missed pairs
    country_misses = Counter = defaultdict(int)
    for sid, cid in missed_gt:
        e1 = cached_s1.get(sid, {})
        country_misses[e1.get("country", "UNKNOWN")] += 1

    print("\nMissed GT by Country:")
    for c, cnt in country_misses.items():
        print(f"  {c}: {cnt}")

    print("\nFirst 20 Missed GT Pairs:")
    for sid, cid in missed_gt[:20]:
        e1 = cached_s1.get(sid, {})
        e2 = cached_cands.get(cid, {})
        n1 = e1.get("norm_n", "")
        n2 = e2.get("norm_n", "") if e2 else "NOT_IN_POOL"
        a1 = e1.get("norm_a", "")[:35]
        a2 = e2.get("norm_a", "")[:35] if e2 else ""
        c1 = e1.get("country", "")
        c2 = e2.get("country", "") if e2 else ""
        print(f"S1: [{n1}] | CAND: [{n2}] | Addr1: [{a1}] vs Addr2: [{a2}] | Country: {c1} vs {c2}")

if __name__ == "__main__":
    main()
