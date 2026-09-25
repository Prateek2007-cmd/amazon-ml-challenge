"""
Test lean candidate generation: max 12 candidates per S1.
Measure recall, candidate count, and feature extraction speed.
"""
import sys, time
sys.path.insert(0, "code/business_entity_resolution/src")
from data_loader import get_data_paths, load_training_subsample
from blocking import build_blocking_index, blocking_recall, extract_blocking_keys
from config import Config

paths = get_data_paths(is_sample=False)
s1_df, s2_dict, s3_dict, gt_dict = load_training_subsample(paths, sample_size=5000, pool_size=100000)

print("Building indices...")
idx2 = build_blocking_index(s2_dict.values())
idx3 = build_blocking_index(s3_dict.values())

for max_cands in [8, 12, 16, 25]:
    t0 = time.time()
    cands_dict = {}
    total_pairs = 0
    
    for _, r in s1_df.iterrows():
        s1_id = r["entity_id"]
        keys = extract_blocking_keys(r["business_name"], r["business_address"], r.get("country", ""))
        
        cands = set()
        # Pass 1: High precision keys first (exact_name, name_pair, name_num, num_word, addr_pair)
        for k in keys:
            if k[0] in ("exact_name", "url_core", "name_pair", "name_num", "num_word", "addr_pair"):
                p2 = idx2.get(k, [])
                p3 = idx3.get(k, [])
                if len(p2) <= 300: cands.update(p2)
                if len(p3) <= 300: cands.update(p3)
                if len(cands) >= max_cands:
                    break
                    
        # Pass 2: If still under 4 candidates, add rare name_tok
        if len(cands) < 4:
            for k in keys:
                if k[0] == "name_tok":
                    p2 = idx2.get(k, [])
                    p3 = idx3.get(k, [])
                    if len(p2) <= 80: cands.update(p2)
                    if len(p3) <= 80: cands.update(p3)
                    if len(cands) >= max_cands:
                        break
                        
        cands_dict[s1_id] = set(list(cands)[:max_cands])
        total_pairs += len(cands_dict[s1_id])
        
    rec, missed = blocking_recall(cands_dict, gt_dict)
    avg_c = total_pairs / len(s1_df)
    print(f"Max Cands = {max_cands:>2}: Recall = {rec*100:.2f}% | Avg Cands = {avg_c:.1f} | Pairs = {total_pairs:,} ({time.time()-t0:.2f}s)")
