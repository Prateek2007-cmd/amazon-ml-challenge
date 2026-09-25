"""
Test memory footprint of test inverted index.
"""
import os
import sys
import psutil
import time
import pandas as pd
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from blocking import extract_blocking_keys

def check_mem():
    mem = psutil.virtual_memory()
    return f"Avail: {mem.available/1e9:.2f} GB ({mem.percent}%)"

print("Initial RAM:", check_mem())

# Test indexing 500,000 rows of test S2
t0 = time.time()
idx = defaultdict(list)
count = 0
with open("dataset/test/test_source2.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\n").split("\t")
        eid, name, addr = parts[0], parts[1], parts[2]
        c = parts[3] if len(parts) > 3 else ""
        keys = extract_blocking_keys(name, addr, c)
        for k in keys:
            idx[k].append(eid)
        count += 1
        if count >= 500000:
            break

print(f"Indexed {count:,d} records in {time.time()-t0:.2f}s | Unique keys: {len(idx):,d} | RAM: {check_mem()}")
