import time, psutil
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

print("=" * 80)
print("TESTING SPARSE-SPARSE BATCH MULTIPLICATION")
print("=" * 80)

# Load 500 real S1 records and 50,000 real S2 records
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

s1_ids, s1_texts = load_sample("dataset/test/test_source1.tsv", 500)
cand_ids, cand_texts = load_sample("dataset/test/test_source2.tsv", 50000)

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
X_cand = sklearn_normalize(vectorizer.fit_transform(cand_texts), norm="l2")
X_s1 = sklearn_normalize(vectorizer.transform(s1_texts), norm="l2")

print(f"X_cand: {X_cand.shape}, nnz={X_cand.nnz:,d}")
print(f"X_s1: {X_s1.shape}, nnz={X_s1.nnz:,d}")

# 1. Sparse-sparse multiplication
t0 = time.time()
# X_s1 is (500, 30000) CSR, X_cand.T is (30000, 50000) CSC
X_cand_csc_T = X_cand.T.tocsc()
sim_sparse = X_s1 @ X_cand_csc_T
t_sp = time.time() - t0
print(f"Sparse-sparse matmul for 500 queries vs 50k candidates: {t_sp:.3f}s ({t_sp/500*1000:.2f} ms/query)")
print(f"sim_sparse shape: {sim_sparse.shape}, nnz: {sim_sparse.nnz:,d}, RAM: {sim_sparse.data.nbytes/(1024*1024):.1f} MB")

# Now extract top 10 with score >= 0.30
t1 = time.time()
top10_results = {}
for i in range(500):
    row = sim_sparse.getrow(i)
    # Filter >= 0.30
    mask = row.data >= 0.30
    if np.any(mask):
        scores = row.data[mask]
        indices = row.indices[mask]
        if len(scores) > 10:
            top_k = np.argpartition(-scores, 10)[:10]
            top_k = top_k[np.argsort(-scores[top_k])]
            top10_results[i] = [(cand_ids[indices[k]], float(scores[k]), rank_0 + 1) for rank_0, k in enumerate(top_k)]
        else:
            sorted_order = np.argsort(-scores)
            top10_results[i] = [(cand_ids[indices[k]], float(scores[k]), rank_0 + 1) for rank_0, k in enumerate(sorted_order)]

t_filter = time.time() - t1
print(f"Filtering top 10 took {t_filter:.3f}s ({t_filter/500*1000:.2f} ms/query)")
print(f"Total time per query: {(t_sp + t_filter)/500*1000:.2f} ms")
