"""
High-Performance Streaming and IO Utilities for Business Entity Resolution.
Designed for multi-gigabyte datasets: minimal RAM overhead, zero DataFrame bloat,
instant line-by-line streaming.
"""
import os
import pandas as pd
from collections import defaultdict
from config import Config

def get_data_paths(data_dir=None, is_sample=False):
    """Resolves data paths whether running on sample or full dataset."""
    if data_dir is None:
        data_dir = Config.SAMPLE_DIR if is_sample else Config.DATA_DIR
        
    flat_layout = is_sample or os.path.exists(os.path.join(data_dir, "train_source1.tsv"))
    
    def p(split, name):
        return os.path.join(data_dir, name) if flat_layout else os.path.join(data_dir, split, name)
        
    return {
        "train_s1": p("train", "train_source1.tsv"),
        "train_s2": p("train", "train_source2.tsv"),
        "train_s3": p("train", "train_source3.tsv"),
        "train_gt": p("train", "train_ground_truth.tsv"),
        "test_s1": p("test", "test_source1.tsv"),
        "test_s2": p("test", "test_source2.tsv"),
        "test_s3": p("test", "test_source3.tsv"),
    }

def load_tsv(path, nrows=None):
    """Loads TSV safely with all columns as strings and empty string for NaN."""
    return pd.read_csv(path, sep="\t", nrows=nrows, dtype=str, keep_default_na=False)

def parse_ground_truth_fast(gt_path, nrows=None):
    """
    Ultra-fast line-by-line ground truth parser.
    Parses into {s1_id: set(matched_ids)} in milliseconds.
    """
    gt_dict = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline()
        count = 0
        for line in f:
            line_str = line.rstrip("\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            s1_id = parts[0]
            matches = set(x.strip() for x in parts[1].split(",") if x.strip()) if len(parts) > 1 else set()
            gt_dict[s1_id] = matches
            count += 1
            if nrows is not None and count >= nrows:
                break
    return gt_dict

def load_training_subsample(paths, sample_size=15000, pool_size=100000):
    """
    Loads an aligned training subsample without loading multi-gigabyte files into RAM.
    Returns:
        s1_df: DataFrame of sample S1 entities
        s2_dict: {entity_id: dict(row)} candidate pool for S2
        s3_dict: {entity_id: dict(row)} candidate pool for S3
        gt_dict: {s1_id: set(matched_ids)} for sample S1 entities
    """
    print(f"Loading ground truth for {sample_size:,d} entities...", flush=True)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=sample_size)
    s1_needed = set(gt_dict.keys())
    
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)
                
    print(f"Finding {len(s1_needed):,d} S1 records from train_source1.tsv...", flush=True)
    s1_records = []
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in s1_needed:
                s1_records.append(dict(zip(header, parts)))
                if len(s1_records) == len(s1_needed):
                    break
    s1_df = pd.DataFrame(s1_records)
    
    def load_pool_stream(path, needed_set, max_pool):
        pool = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                if eid in needed_set or i < max_pool:
                    pool[eid] = dict(zip(header, parts))
        return pool
        
    print(f"Loading target pools: S2 ({pool_size:,d} + true targets), S3 ({pool_size:,d} + true targets)...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2, pool_size)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3, pool_size)
    
    return s1_df, s2_dict, s3_dict, gt_dict
