"""
Benchmark chunked candidate matrix multiplication.
Compares memory and time.
"""
import time, psutil
import numpy as np
from scipy import sparse

def get_ram():
    return f"{psutil.virtual_memory().available / (1024**3):.2f} GB avail"

print("Benchmarking chunked candidate dot product...")
print("Initial RAM:", get_ram())

# Create simulated sparse TF-IDF matrices
n_features = 30000
n_cands_block = 200000
n_queries = 200

# Random sparse CSR matrices with density ~0.003 (similar to char 3-5 ngrams)
print(f"Creating sparse matrices: {n_queries} queries x {n_features} features, and {n_cands_block} candidates...")
X_q = sparse.random(n_queries, n_features, density=0.003, format="csr", dtype=np.float32)
X_cand_block = sparse.random(n_cands_block, n_features, density=0.003, format="csr", dtype=np.float32)

t0 = time.time()
# Multiply: (200 x 30000) @ (30000 x 200000) -> (200 x 200000)
sim_matrix = (X_q @ X_cand_block.T).toarray()
t_mult = time.time() - t0
print(f"Dot product + toarray took: {t_mult:.3f}s | Shape: {sim_matrix.shape} | RAM: {get_ram()}")

t0 = time.time()
K = 5
for i in range(n_queries):
    row = sim_matrix[i]
    top_idx = np.argpartition(row, -K)[-K:]
t_topk = time.time() - t0
print(f"Top-K extraction for {n_queries} queries took: {t_topk*1000:.2f}ms")
print(f"Total time for {n_queries} queries against {n_cands_block:,d} candidates: {(t_mult + t_topk):.3f}s")
