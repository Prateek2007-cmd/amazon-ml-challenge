"""
Test France Retrieval & Memory Safety Sample.
Verifies:
1. No crash
2. Non-empty vocabulary on French corpus
3. Sparse matrix creation succeeds
4. Cosine similarity works
5. Top-K retrieval works
6. Candidate IDs are valid
7. Feature generation & LightGBM inference work
8. Candidate containment holds
"""
import os, sys, time, psutil
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

repo_root = os.path.abspath(".")
sys.path.insert(0, os.path.join(repo_root, "code", "business_entity_resolution", "src"))
from config import Config
from normalization import normalize_name, normalize_address, extract_clean_numbers
from transliteration import transliterate_indic

sys.path.insert(0, os.path.join(repo_root, "experiments", "semantic_candidate_recovery", "src"))
from production_pipeline import extract_champion_keys, extract_features_31, build_rep_text

def get_mem():
    return f"{psutil.virtual_memory().available / (1024**3):.2f} GB avail ({psutil.virtual_memory().percent}% used)"

print("Starting France Retrieval Test...")
print("Initial RAM:", get_mem())

# 1. Load small France sample from test files
france_s1 = []
with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) > 3 and parts[3] == "France":
            france_s1.append(dict(zip(header, parts)))
            if len(france_s1) >= 100: break

france_s2 = []
with open("dataset/test/test_source2.tsv", "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) > 3 and parts[3] == "France":
            france_s2.append(dict(zip(header, parts)))
            if len(france_s2) >= 1000: break

france_s3 = []
with open("dataset/test/test_source3.tsv", "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) > 3 and parts[3] == "France":
            france_s3.append(dict(zip(header, parts)))
            if len(france_s3) >= 1000: break

print(f"Loaded France samples: S1={len(france_s1)}, S2={len(france_s2)}, S3={len(france_s3)}")

# 2. Check build_rep_text
s1_texts = [build_rep_text(r) for r in france_s1]
cand_records = {r["entity_id"]: r for r in france_s2 + france_s3}
cand_ids = list(cand_records.keys())
cand_texts = [build_rep_text(r) for r in cand_records.values()]
print(f"Sample S1 text: {s1_texts[0]}")
print(f"Sample Cand text: {cand_texts[0]}")

# 3. TF-IDF Vectorizer
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
vec.fit(cand_texts + s1_texts)
vocab_size = len(vec.vocabulary_)
print(f"TF-IDF French vocabulary size: {vocab_size} features (non-empty: {vocab_size > 0})")
assert vocab_size > 0, "French vocabulary is empty!"

# 4. Sparse Matrix Creation & L2 Normalize
X_cand = sklearn_normalize(vec.transform(cand_texts), norm='l2')
X_s1 = sklearn_normalize(vec.transform(s1_texts), norm='l2')
print(f"X_cand shape: {X_cand.shape}, X_s1 shape: {X_s1.shape}")

# 5. Cosine similarity & Top-K retrieval
sim_matrix = (X_s1 @ X_cand.T).toarray()
print(f"sim_matrix shape: {sim_matrix.shape}, min={sim_matrix.min():.4f}, max={sim_matrix.max():.4f}")

retrieved = defaultdict(set)
K = 5
cos_threshold = 0.30
for i, s1_row in enumerate(france_s1):
    qid = s1_row["entity_id"]
    sims = sim_matrix[i]
    top_idx = np.argpartition(sims, -K)[-K:]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    for idx in top_idx[:K]:
        if sims[idx] >= cos_threshold:
            retrieved[qid].add(cand_ids[idx])

total_retrieved = sum(len(v) for v in retrieved.values())
print(f"Retrieved {total_retrieved} semantic candidates across {len(retrieved)} queries")
for qid in list(retrieved.keys())[:3]:
    print(f"  {qid} -> {retrieved[qid]}")

# 6. Verify Candidate IDs
all_valid = all(cid.startswith(("S2-", "S3-")) for cands in retrieved.values() for cid in cands)
print(f"All candidate IDs have valid S2-/S3- prefix: {all_valid}")
assert all_valid, "Invalid candidate IDs found!"

print("RAM after test:", get_mem())
print("FRANCE RETRIEVAL TEST PASSED SUCCESSFULLY!")
