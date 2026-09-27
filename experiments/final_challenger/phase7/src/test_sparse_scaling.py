import time, psutil
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

print("=" * 80)
print("TESTING SPARSE TF-IDF INVERTED RETRIEVAL SCALING")
print("=" * 80)

# Simulate 1.4M candidates (France scale)
N_CAND = 1400000
N_FEAT = 30000
NNZ_PER_ROW = 30

print(f"Creating sparse matrix for {N_CAND:,d} candidates...")
# Generate random sparse matrix with typical character n-gram sparsity
# In reality, character n-grams have skewed frequency (Zipf's law)
np.random.seed(42)
cols_sample = np.random.zipf(1.5, size=N_CAND * NNZ_PER_ROW) % N_FEAT
rows = np.repeat(np.arange(N_CAND, dtype=np.int32), NNZ_PER_ROW)
data = np.random.rand(N_CAND * NNZ_PER_ROW).astype(np.float32)

X_cand = sp.csr_matrix((data, (rows, cols_sample)), shape=(N_CAND, N_FEAT))
X_cand = sklearn_normalize(X_cand, norm="l2")
print(f"CSR matrix created: {X_cand.shape}, nnz={X_cand.nnz:,d}, RAM={X_cand.data.nbytes/(1024*1024):.1f} MB")

t0 = time.time()
# Convert to CSC transpose for inverted column traversal
# Notice: (X_cand).T is a CSC matrix if X_cand is CSR!
# In scipy: csr.T is csc! Zero copy or near instant!
X_cand_T = X_cand.T.tocsc()
print(f"Transposed to CSC in {time.time()-t0:.2f}s")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# Test queries
N_QUERIES = 100
q_cols = np.random.choice(N_FEAT, size=(N_QUERIES, NNZ_PER_ROW))
q_rows = np.repeat(np.arange(N_QUERIES, dtype=np.int32), NNZ_PER_ROW)
q_data = np.random.rand(N_QUERIES * NNZ_PER_ROW).astype(np.float32)
Q = sp.csr_matrix((q_data, (q_rows, q_cols.ravel())), shape=(N_QUERIES, N_FEAT))
Q = sklearn_normalize(Q, norm="l2")

print(f"\nTesting 100 single-query sparse dot products...")
t_start = time.time()
hits_total = 0
for i in range(N_QUERIES):
    q = Q[i]
    # Sparse dot product
    res = q.dot(X_cand_T)
    # Filter >= 0.30
    mask = res.data >= 0.30
    if np.any(mask):
        scores = res.data[mask]
        indices = res.indices[mask]
        hits_total += len(scores)

elapsed = time.time() - t_start
print(f"Completed {N_QUERIES} queries in {elapsed:.3f}s ({elapsed/N_QUERIES*1000:.2f} ms/query)!")
print(f"Extrapolated: 10,000 queries = {elapsed/N_QUERIES*10000:.1f}s")
