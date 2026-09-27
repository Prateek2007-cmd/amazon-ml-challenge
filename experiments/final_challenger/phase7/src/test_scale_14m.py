import time, psutil
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

print("Testing 1.4M candidates (France scale)...")
# Generate 1.4M random sparse vectors
N_CAND = 1400000
N_FEAT = 30000
# create realistic sparse matrix with 25 nonzeros per row
rows = np.repeat(np.arange(N_CAND), 25)
cols = np.random.randint(0, N_FEAT, size=N_CAND * 25)
data = np.random.rand(N_CAND * 25).astype(np.float32)
X_cand = sp.csr_matrix((data, (rows, cols)), shape=(N_CAND, N_FEAT))
X_cand = sklearn_normalize(X_cand, norm="l2")
print(f"Created X_cand {X_cand.shape} with {X_cand.nnz:,d} nonzeros ({X_cand.data.nbytes/(1024*1024):.1f} MB)")
print(f"RAM available: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# Test 100 queries
N_Q = 100
q_rows = np.repeat(np.arange(N_Q), 25)
q_cols = np.random.randint(0, N_FEAT, size=N_Q * 25)
q_data = np.random.rand(N_Q * 25).astype(np.float32)
X_q = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(N_Q, N_FEAT))
X_q = sklearn_normalize(X_q, norm="l2")

t0 = time.time()
res = (X_q @ X_cand.T).toarray()
t_mult = time.time() - t0
print(f"Multiplied {N_Q} queries vs 1.4M candidates in {t_mult:.2f}s ({t_mult/N_Q*1000:.2f} ms/query) | Result shape: {res.shape} ({res.nbytes/(1024*1024):.1f} MB)")
