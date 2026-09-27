import time
import numpy as np
import scipy.sparse as sp

# 1.43M candidates (France)
N_CAND = 1434993
N_FEAT = 30000
NNZ_PER_ROW = 100

print("Generating synthetic sparse candidate matrix...")
t0 = time.time()
rows = np.repeat(np.arange(N_CAND, dtype=np.int32), NNZ_PER_ROW)
cols = np.random.randint(0, N_FEAT, size=N_CAND * NNZ_PER_ROW, dtype=np.int32)
data = np.ones(N_CAND * NNZ_PER_ROW, dtype=np.float32)
X_cand = sp.csr_matrix((data, (rows, cols)), shape=(N_CAND, N_FEAT))
# X_cand_T in CSR format: shape (30000, 1434993)
X_cand_T = X_cand.T.tocsr()
print(f"X_cand_T CSR created in {time.time()-t0:.2f}s")

# Test 100 queries with sparse dot product
N_Q = 100
q_rows = np.repeat(np.arange(N_Q, dtype=np.int32), NNZ_PER_ROW)
q_cols = np.random.randint(0, N_FEAT, size=N_Q * NNZ_PER_ROW, dtype=np.int32)
q_data = np.ones(N_Q * NNZ_PER_ROW, dtype=np.float32)
X_q = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(N_Q, N_FEAT))

# Approach A: Iterate each query, q.dot(X_cand_T)
t1 = time.time()
hits = 0
for i in range(N_Q):
    q = X_q.getrow(i)
    res_sp = q.dot(X_cand_T)
    mask = res_sp.data >= 0.30
    matched_idx = res_sp.indices[mask]
    matched_scores = res_sp.data[mask]
    if len(matched_scores) > 10:
        top_k = np.argpartition(-matched_scores, 10)[:10]
    hits += len(matched_scores)
t_iter = time.time() - t1
print(f"Approach A (Iterative sparse dot): {t_iter:.3f}s ({t_iter/N_Q*1000:.2f} ms/query)")
