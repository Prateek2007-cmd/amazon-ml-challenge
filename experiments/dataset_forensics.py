"""
Phase 1: Full Workspace Forensics
Thoroughly inspects train and test datasets without exhausting RAM.
"""
import os
import sys
import pandas as pd
from collections import Counter

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

def inspect_tsv_streaming(path, name):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n--- Forensics on {name} ({size_mb:.2f} MB) ---")
    
    total_rows = 0
    missing_counts = Counter()
    countries = Counter()
    seen_ids = set()
    dup_ids = 0
    name_lens = []
    addr_lens = []
    
    chunksize = 200000
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False):
        total_rows += len(chunk)
        for col in chunk.columns:
            empty_in_col = (chunk[col] == "").sum()
            missing_counts[col] += empty_in_col
            
        if "entity_id" in chunk.columns:
            for eid in chunk["entity_id"]:
                if eid in seen_ids:
                    dup_ids += 1
                else:
                    seen_ids.add(eid)
                    
        if "country" in chunk.columns:
            countries.update(chunk["country"].value_counts().to_dict())
            
        if len(name_lens) < 50000:
            if "business_name" in chunk.columns:
                name_lens.extend(chunk["business_name"].str.len().tolist()[:5000])
            if "business_address" in chunk.columns:
                addr_lens.extend(chunk["business_address"].str.len().tolist()[:5000])
                
    print(f"Total Rows: {total_rows:,d}")
    print(f"Unique IDs: {len(seen_ids):,d} | Duplicate IDs: {dup_ids}")
    print(f"Missingness (empty strings): {dict(missing_counts)}")
    print(f"Country distribution: {dict(countries)}")
    if name_lens:
        print(f"Business Name length (sample mean): {sum(name_lens)/len(name_lens):.1f} chars (max: {max(name_lens)})")
    if addr_lens:
        print(f"Business Address length (sample mean): {sum(addr_lens)/len(addr_lens):.1f} chars (max: {max(addr_lens)})")
    return total_rows, dict(countries)

def inspect_ground_truth(path):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n--- Forensics on train_ground_truth.tsv ({size_mb:.2f} MB) ---")
    total_rows = 0
    cardinality_counts = Counter()
    total_matches = 0
    match_sources = Counter()
    
    chunksize = 200000
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False):
        total_rows += len(chunk)
        for val in chunk["matched_entity_ids"]:
            val_str = val.strip()
            if not val_str:
                cardinality_counts[0] += 1
            else:
                ids = val_str.split(",")
                card = len(ids)
                cardinality_counts[card if card <= 5 else "6+"] += 1
                total_matches += card
                for mid in ids:
                    prefix = mid[:3]
                    match_sources[prefix] += 1
                    
    print(f"Total S1 entities in Ground Truth: {total_rows:,d}")
    print(f"Total true match pairs: {total_matches:,d} (avg: {total_matches/total_rows:.2f} matches/S1)")
    print(f"Cardinality breakdown:")
    for card, cnt in sorted(cardinality_counts.items(), key=lambda x: str(x[0])):
        print(f"  {card} matches: {cnt:,d} ({cnt/total_rows*100:.2f}%)")
    print(f"Matches by source prefix: {dict(match_sources)}")

if __name__ == "__main__":
    inspect_tsv_streaming(os.path.join(TRAIN_DIR, "train_source1.tsv"), "train_source1.tsv")
    inspect_tsv_streaming(os.path.join(TRAIN_DIR, "train_source2.tsv"), "train_source2.tsv")
    inspect_tsv_streaming(os.path.join(TRAIN_DIR, "train_source3.tsv"), "train_source3.tsv")
    inspect_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    inspect_tsv_streaming(os.path.join(TEST_DIR, "test_source1.tsv"), "test_source1.tsv")
    inspect_tsv_streaming(os.path.join(TEST_DIR, "test_source2.tsv"), "test_source2.tsv")
    inspect_tsv_streaming(os.path.join(TEST_DIR, "test_source3.tsv"), "test_source3.tsv")
