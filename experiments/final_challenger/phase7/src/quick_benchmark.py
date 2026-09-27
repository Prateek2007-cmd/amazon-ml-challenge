import time
import numpy as np
import scipy.sparse as sp

# Test 50 queries vs 1.43M candidates
N_CAND = 1434993
N_FEAT = 30000
NNZ_PER_ROW = 100

print("Generating synthetic sparse candidate matrix...")
t0 = time.time()
rows = np.repeat(np.arange(N_CAND, dtype=np.int32), NNZ_PER_ROW)
cols = np.random.randint(0, N_FEAT, size=N_CAND * NNZ_PER_ROW, dtype=np.int32)
data = np.ones(N_CAND * NNZ_PER_ROW, dtype=np.float32)
X_cand = sp.csr_matrix((data, (rows, cols)), shape=(N_CAND, N_FEAT))
X_cand_T = X_cand.T.tocsr()
print(f"Cand matrix created in {time.time()-t0:.2f}s | NNZ: {X_cand.nnz:,d}")

# 50 queries
N_Q = 50
q_rows = np.repeat(np.arange(N_Q, dtype=np.int32), NNZ_PER_ROW)
q_cols = np.random.randint(0, N_FEAT, size=N_Q * NNZ_PER_ROW, dtype=np.int32)
q_data = np.ones(N_Q * NNZ_PER_ROW, dtype=np.float32)
X_q = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(N_Q, N_FEAT))

t1 = time.time()
res = (X_q @ X_cand_T).toarray()
t_mult = time.time() - t1
print(f"Matmul 50 queries vs 1.43M: {t_mult:.3f}s ({t_mult/N_Q*1000:.2f} ms/query)")

t2 = time.time()
for row_i in range(N_Q):
    sims = res[row_i]
    top_idx = np.argpartition(-sims, 10)[:10]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
t_topk = time.time() - t2
print(f"Top 10 filter: {t_topk:.3f}s ({t_topk/N_Q*1000:.2f} ms/query)")
