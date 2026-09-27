import time, psutil
import numpy as np
import scipy.sparse as sp

print("=" * 80)
print("TESTING BLAS MULTI-CORE SCALING ON 1.4M CANDIDATES")
print("=" * 80)

N_CAND = 1400000
N_FEAT = 30000
NNZ_PER_ROW = 25

print("Allocating sparse candidate matrix (1.4M rows)...")
rows = np.repeat(np.arange(N_CAND, dtype=np.int32), NNZ_PER_ROW)
cols = np.random.randint(0, N_FEAT, size=N_CAND * NNZ_PER_ROW, dtype=np.int32)
data = np.random.rand(N_CAND * NNZ_PER_ROW).astype(np.float32)

X_cand = sp.csr_matrix((data, (rows, cols)), shape=(N_CAND, N_FEAT), dtype=np.float32)
# Normalize rows
row_sums = np.sqrt(np.asarray(X_cand.power(2).sum(axis=1)).ravel())
row_sums[row_sums == 0] = 1.0
X_cand = sp.diags(1.0 / row_sums) @ X_cand

print(f"X_cand created: {X_cand.shape}, RAM: {X_cand.data.nbytes/(1024*1024):.1f} MB")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# Test 500 queries
N_Q = 500
q_rows = np.repeat(np.arange(N_Q, dtype=np.int32), NNZ_PER_ROW)
q_cols = np.random.randint(0, N_FEAT, size=N_Q * NNZ_PER_ROW, dtype=np.int32)
q_data = np.random.rand(N_Q * NNZ_PER_ROW).astype(np.float32)
X_q = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(N_Q, N_FEAT), dtype=np.float32)

# Convert X_cand to CSC for fast column slicing or multiply
t0 = time.time()
# Multiply batch of 50 queries at a time to keep memory under 300 MB
BATCH = 50
total_mult_time = 0.0
total_filter_time = 0.0

for b_start in range(0, N_Q, BATCH):
    b_end = min(b_start + BATCH, N_Q)
    t_b0 = time.time()
    sim_batch = (X_q[b_start:b_end] @ X_cand.T).toarray()
    total_mult_time += time.time() - t_b0

    t_f0 = time.time()
    for local_i in range(b_end - b_start):
        sims = sim_batch[local_i]
        top_idx = np.argpartition(-sims, 10)[:10]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        _ = [(top_idx[k], float(sims[top_idx[k]])) for k in range(10) if sims[top_idx[k]] >= 0.30]
    total_filter_time += time.time() - t_f0
    del sim_batch

total_time = total_mult_time + total_filter_time
print(f"Processed {N_Q} queries vs 1.4M candidates in {total_time:.2f}s ({total_time/N_Q*1000:.2f} ms/query)!")
print(f"  - Matmul time: {total_mult_time:.2f}s ({total_mult_time/N_Q*1000:.2f} ms/query)")
print(f"  - Filter time: {total_filter_time:.2f}s ({total_filter_time/N_Q*1000:.2f} ms/query)")
print(f"Extrapolated for 100,000 queries: {total_time/N_Q*100000/60:.1f} minutes")
