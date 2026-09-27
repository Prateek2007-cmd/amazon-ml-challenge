import time
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

# Load 1,000 real S1 records and 50,000 real S2 records
def load_sample(path, n):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= n: break
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            texts.append(f"{parts[1]} {parts[2]}")
    return ids, texts

print("=" * 80)
print("TESTING INVERTED INDEX CANDIDATE PRUNING FOR TF-IDF SEMANTIC SEARCH")
print("=" * 80)
s1_ids, s1_texts = load_sample("dataset/test/test_source1.tsv", 200)
cand_ids, cand_texts = load_sample("dataset/test/test_source2.tsv", 50000)

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
X_cand = sklearn_normalize(vectorizer.fit_transform(cand_texts), norm="l2")
X_s1 = sklearn_normalize(vectorizer.transform(s1_texts), norm="l2")

# 1. Ground truth: Full brute force matrix multiplication
t0 = time.time()
sim_dense = (X_s1 @ X_cand.T).toarray()
t_brute = time.time() - t0
print(f"Brute force dense mult for 200 queries vs 50k candidates: {t_brute:.3f}s ({t_brute/200*1000:.2f} ms/query)")

# Get exact top 10 with cos >= 0.30 from brute force
exact_top10 = {}
for i in range(200):
    sims = sim_dense[i]
    top_idx = np.argpartition(-sims, 10)[:10]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    exact_top10[i] = [(cand_ids[idx], float(sims[idx])) for idx in top_idx if sims[idx] >= 0.30]

# 2. Inverted Index on non-zero columns of X_cand (CSC format)
t1 = time.time()
X_cand_csc = X_cand.tocsc()
t_csc = time.time() - t1
print(f"Converted to CSC in {t_csc:.3f}s")

# For each query, find candidate indices that share at least min_shared ngrams
t2 = time.time()
pruned_top10 = {}
for i in range(200):
    q = X_s1[i]
    q_cols = q.indices
    q_weights = q.data
    
    # Fast posting union in C++ via sparse dot product!
    # q is (1, 30000), X_cand_csc.T is (30000, 50000)
    # q.dot(X_cand_csc.T) computes EXACT cosine for only active candidates!
    res_sp = q.dot(X_cand_csc.T)
    # Only keep candidates with score >= 0.30
    mask = res_sp.data >= 0.30
    matched_indices = res_sp.indices[mask]
    matched_scores = res_sp.data[mask]
    
    if len(matched_scores) > 10:
        top_k = np.argpartition(-matched_scores, 10)[:10]
        top_k = top_k[np.argsort(-matched_scores[top_k])]
        pruned_top10[i] = [(cand_ids[matched_indices[k]], float(matched_scores[k])) for k in top_k]
    else:
        sorted_order = np.argsort(-matched_scores)
        pruned_top10[i] = [(cand_ids[matched_indices[k]], float(matched_scores[k])) for k in sorted_order]

t_pruned = time.time() - t2
print(f"Sparse inverted dot product for 200 queries: {t_pruned:.3f}s ({t_pruned/200*1000:.2f} ms/query)")

# Verify accuracy: check match rate between exact and pruned
matches = 0
total_checked = 0
for i in range(200):
    exact_set = set(cid for cid, s in exact_top10[i])
    pruned_set = set(cid for cid, s in pruned_top10[i])
    if exact_set == pruned_set:
        matches += 1
    total_checked += 1

print(f"Accuracy check: {matches}/{total_checked} ({matches/total_checked*100:.1f}%) exact 1:1 match!")
