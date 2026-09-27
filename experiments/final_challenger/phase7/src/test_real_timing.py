import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

# Load 1,000 real S1 records and 50,000 real S2 records (France or India)
def load_sample(path, n):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= n: break
            parts = line.rstrip("\n").split("\t")
            texts.append(f"{parts[1]} {parts[2]}")
    return texts

print("Loading real test strings...")
s1_texts = load_sample("dataset/test/test_source1.tsv", 1000)
cand_texts = load_sample("dataset/test/test_source2.tsv", 100000)

print(f"Loaded {len(s1_texts)} S1 and {len(cand_texts)} S2 texts.")
t0 = time.time()
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
X_cand = sklearn_normalize(vectorizer.fit_transform(cand_texts), norm="l2")
X_s1 = sklearn_normalize(vectorizer.transform(s1_texts), norm="l2")
print(f"Vectorized in {time.time()-t0:.2f}s | Cand shape: {X_cand.shape}")

# Test chunk dot product with real data
CHUNK = 100
t1 = time.time()
sim_mat = (X_s1[:CHUNK] @ X_cand.T).toarray()
t_dense = time.time() - t1
print(f"Dense matrix mult for {CHUNK} real queries vs 100k real candidates: {t_dense:.3f}s ({t_dense/CHUNK*1000:.2f} ms/query)")

# Filter top 10 with cosine >= 0.30
t2 = time.time()
for row_i in range(CHUNK):
    sims = sim_mat[row_i]
    top_idx = np.argpartition(-sims, 10)[:10]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
print(f"Top 10 filter took {time.time()-t2:.3f}s ({((time.time()-t2)/CHUNK)*1000:.2f} ms/query)")
