import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

sample_words = ["amazon", "logistics", "india", "pvt", "ltd", "enterprises", "solutions", "mumbai", "delhi", "road", "street", "industrial", "area", "phase", "plot"]
np.random.seed(42)
cand_texts = [" ".join(np.random.choice(sample_words, size=6)) for _ in range(500000)]
s1_texts = [" ".join(np.random.choice(sample_words, size=6)) for _ in range(1000)]

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
X_cand_csr = vectorizer.fit_transform(cand_texts)
X_cand_csr = sklearn_normalize(X_cand_csr, norm="l2")
X_cand_csc = X_cand_csr.tocsc()

X_s1 = sklearn_normalize(vectorizer.transform(s1_texts), norm="l2")

print("Testing single-row sparse dot product:")
t0 = time.time()
N_TEST = 100
for i in range(N_TEST):
    q = X_s1[i]
    res = q.dot(X_cand_csc.T)
t_single = (time.time() - t0) / N_TEST
print(f"Single sparse dot product: {t_single*1000:.3f} ms per query")

print("Testing chunk sparse dot product (batch 500):")
t1 = time.time()
q_batch = X_s1[:500]
res_batch = q_batch.dot(X_cand_csc.T)
t_batch = time.time() - t1
print(f"Batch 500 sparse dot product: {t_batch:.3f}s ({t_batch/500*1000:.3f} ms per query) | Nonzeros: {res_batch.nnz:,d}")

# Finding top K with cosine >= 0.30 from sparse result:
t2 = time.time()
for row_i in range(500):
    row_sp = res_batch.getrow(row_i)
    # only candidates with score >= 0.30
    mask = row_sp.data >= 0.30
    c_indices = row_sp.indices[mask]
    c_scores = row_sp.data[mask]
    if len(c_scores) > 10:
        top_k_idx = np.argpartition(-c_scores, 10)[:10]
        top_k_idx = top_k_idx[np.argsort(-c_scores[top_k_idx])]
print(f"Filtering top 10 for 500 rows took {time.time()-t2:.3f}s")
