"""
Ultra-fast Dataset Forensics and Data Auditing for Amazon ML Challenge 2026.
Optimized for multi-gigabyte datasets with instant buffered line counting,
vectorized pandas analytics, and immediate stdout flushing.
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from collections import Counter

# Ensure instant unbuffered output with UTF-8
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def fast_count_lines(path):
    """Counts newlines using 1MB binary chunks - runs at ~1 GB/s."""
    lines = 0
    with open(path, "rb") as f:
        while True:
            buffer = f.read(1024 * 1024)
            if not buffer:
                break
            lines += buffer.count(b"\n")
    return lines

def run_audit(data_dir=None):
    start_total = time.time()
    if data_dir is None:
        data_dir = os.path.join(REPO_ROOT, "dataset")
    
    print(f"=== COMMENCING ULTRA-FAST DATA FORENSICS ===", flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    file_map = {
        "train_source1.tsv": os.path.join(train_dir, "train_source1.tsv"),
        "train_source2.tsv": os.path.join(train_dir, "train_source2.tsv"),
        "train_source3.tsv": os.path.join(train_dir, "train_source3.tsv"),
        "train_ground_truth.tsv": os.path.join(train_dir, "train_ground_truth.tsv"),
        "test_source1.tsv": os.path.join(test_dir, "test_source1.tsv"),
        "test_source2.tsv": os.path.join(test_dir, "test_source2.tsv"),
        "test_source3.tsv": os.path.join(test_dir, "test_source3.tsv"),
    }
    
    file_stats = {}
    print("\n[STEP 1] Fast file size and row count audit:", flush=True)
    for fname, path in file_map.items():
        if not os.path.exists(path):
            print(f"  [MISSING] {fname}: {path}", flush=True)
            continue
        t0 = time.time()
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        raw_lines = fast_count_lines(path)
        data_rows = max(0, raw_lines - 1)  # subtract header
        
        # Read header
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            first_line = f.readline().rstrip("\n").split("\t")
            
        elapsed = time.time() - t0
        print(f"  {fname:<25} | {size_mb:>8.2f} MB | {data_rows:>10,d} rows | cols: {header} ({elapsed:.2f}s)", flush=True)
        file_stats[fname] = {"rows": data_rows, "cols": header, "sample_row": first_line, "path": path}

    # STEP 2: Ground truth analysis
    gt_path = file_map["train_ground_truth.tsv"]
    if os.path.exists(gt_path):
        print("\n[STEP 2] Analyzing train_ground_truth.tsv...", flush=True)
        t0 = time.time()
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        print(f"  Loaded {len(gt_df):,d} ground truth rows in {time.time()-t0:.2f}s", flush=True)
        
        # Fast vectorized processing of matches
        matched_series = gt_df["matched_entity_ids"].values
        match_lens = np.array([len([x for x in m.split(",") if x]) if m else 0 for m in matched_series])
        
        c = Counter(match_lens)
        total_gt_entities = len(gt_df)
        total_matches = int(match_lens.sum())
        singletons = c[0]
        
        print(f"  Total S1 Entities in GT: {total_gt_entities:,d}", flush=True)
        print(f"  Total True Matches:      {total_matches:,d}", flush=True)
        print(f"  Singletons (0 matches):  {singletons:,d} ({singletons / total_gt_entities * 100:.2f}%)", flush=True)
        print(f"  Distribution of match cardinality:", flush=True)
        for num_m in sorted(c.keys())[:10]:
            print(f"    {num_m} matches: {c[num_m]:,d} entities ({c[num_m] / total_gt_entities * 100:.2f}%)", flush=True)
        if len(c) > 10:
            remaining = sum(c[k] for k in sorted(c.keys())[10:])
            print(f"    >9 matches: {remaining:,d} entities", flush=True)

    # STEP 3: Inspect Column Values and Noise Patterns
    print("\n[STEP 3] Inspecting schema, nulls, countries, string lengths across sources...", flush=True)
    for fname in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        path = file_map[fname]
        if not os.path.exists(path):
            continue
        df_sample = pd.read_csv(path, sep="\t", nrows=20000, dtype=str, keep_default_na=False)
        countries = df_sample["country"].value_counts().to_dict() if "country" in df_sample.columns else {}
        name_lens = df_sample["business_name"].str.len() if "business_name" in df_sample.columns else []
        addr_lens = df_sample["business_address"].str.len() if "business_address" in df_sample.columns else []
        
        avg_name_len = name_lens.mean() if len(name_lens) else 0
        avg_addr_len = addr_lens.mean() if len(addr_lens) else 0
        print(f"  {fname:<22}: Countries (20k sample): {countries} | Avg Name Len: {avg_name_len:.1f} | Avg Addr Len: {avg_addr_len:.1f}", flush=True)

    # STEP 4: Inspect real true match pairs to understand noise
    print("\n[STEP 4] Concrete True Match Examples (Ground Truth Pairs):", flush=True)
    gt_sample = pd.read_csv(gt_path, sep="\t", nrows=100, dtype=str, keep_default_na=False)
    non_empty = gt_sample[gt_sample["matched_entity_ids"] != ""].head(5)
    s1_needed = set(non_empty["source1_entity_id"])
    all_m = []
    for m in non_empty["matched_entity_ids"]:
        all_m.extend(m.split(","))
    s2_needed = set(x for x in all_m if x.startswith("S2-"))
    s3_needed = set(x for x in all_m if x.startswith("S3-"))

    def fast_lookup_file(path, id_set):
        res = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts[0] in id_set:
                    res[parts[0]] = dict(zip(header, parts))
                    if len(res) == len(id_set):
                        break
        return res

    print("  Loading matching records for sample GT pairs...", flush=True)
    s1_records = fast_lookup_file(file_map["train_source1.tsv"], s1_needed)
    s2_records = fast_lookup_file(file_map["train_source2.tsv"], s2_needed)
    s3_records = fast_lookup_file(file_map["train_source3.tsv"], s3_needed)

    for _, row in non_empty.iterrows():
        s1_id = row["source1_entity_id"]
        matches = row["matched_entity_ids"].split(",")
        s1 = s1_records.get(s1_id, {})
        print(f"\n=== S1 Entity: {s1_id} [{s1.get('country', '')}] ===", flush=True)
        print(f"  Name:    {s1.get('business_name', '')}", flush=True)
        print(f"  Address: {s1.get('business_address', '')}", flush=True)
        for m in matches:
            target = s2_records.get(m, {}) if m.startswith("S2-") else s3_records.get(m, {})
            print(f"  -> MATCH {m} [{target.get('country', '')}]", flush=True)
            print(f"     Name:    {target.get('business_name', '')}", flush=True)
            print(f"     Address: {target.get('business_address', '')}", flush=True)

    print(f"\n=== FORENSICS COMPLETE in {time.time()-start_total:.2f}s ===", flush=True)

if __name__ == "__main__":
    run_audit()
