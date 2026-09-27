import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

print("=" * 80)
print("TESTING PRUNED TF-IDF FOR FAST SEMANTIC RETRIEVAL")
print("=" * 80)

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

# Baseline vectorizer from Phase 6:
v_base = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
X_cand_base = sklearn_normalize(v_base.fit_transform(cand_texts), norm="l2")
X_s1_base = sklearn_normalize(v_base.transform(s1_texts), norm="l2")

# Get exact top 10 with cos >= 0.30 from baseline
t0 = time.time()
sim_dense = (X_s1_base[:200] @ X_cand_base.T).toarray()
exact_top10 = {}
for i in range(200):
    sims = sim_dense[i]
    top_idx = np.argpartition(-sims, 10)[:10]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    exact_top10[i] = [cand_ids[idx] for idx in top_idx if sims[idx] >= 0.30]

print(f"Brute force baseline computed in {time.time()-t0:.2f}s")

# Test with max_df=0.25 (removes omnipresent boilerplate 3-grams that cause dense bloat)
v_pruned = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.25, max_features=30000, sublinear_tf=True)
X_cand_p = sklearn_normalize(v_pruned.fit_transform(cand_texts), norm="l2")
X_s1_p = sklearn_normalize(v_pruned.transform(s1_texts), norm="l2")

t1 = time.time()
sim_p = (X_s1_p[:200] @ X_cand_p.T).toarray()
pruned_top10 = {}
for i in range(200):
    sims = sim_p[i]
    top_idx = np.argpartition(-sims, 10)[:10]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    pruned_top10[i] = [cand_ids[idx] for idx in top_idx if sims[idx] >= 0.30]

# Check overlap
overlap_count = 0
total_exact = 0
for i in range(200):
    e_set = set(exact_top10[i])
    p_set = set(pruned_top10[i])
    total_exact += len(e_set)
    overlap_count += len(e_set & p_set)

print(f"Exact candidates: {total_exact}, Pruned overlap: {overlap_count} ({overlap_count/max(total_exact,1)*100:.1f}%)")
