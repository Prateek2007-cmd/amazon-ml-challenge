import time, psutil, os, gc
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

def print_ram(tag=""):
    proc = psutil.Process()
    ram_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"[{tag}] RAM: {ram_mb:.1f} MB", flush=True)

print("=" * 80)
print("BENCHMARKING SEMANTIC SCALING & MEMORY SAFETY")
print("=" * 80)
print_ram("Start")

# Test with 500,000 candidate texts (representative of a country or chunk)
print("Generating 500,000 sample normalized strings...")
sample_words = ["amazon", "logistics", "india", "pvt", "ltd", "enterprises", "solutions", "mumbai", "delhi", "road", "street", "industrial", "area", "phase", "plot"]
np.random.seed(42)
cand_texts = [" ".join(np.random.choice(sample_words, size=6)) for _ in range(500000)]
s1_texts = [" ".join(np.random.choice(sample_words, size=6)) for _ in range(1000)]
print_ram("After text generation")

t0 = time.time()
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
X_cand = vectorizer.fit_transform(cand_texts)
X_cand = sklearn_normalize(X_cand, norm="l2")
print(f"Vectorized 500k texts in {time.time()-t0:.2f}s | Shape: {X_cand.shape} | Nonzeros: {X_cand.nnz:,d}")
print_ram("After X_cand")

X_s1 = sklearn_normalize(vectorizer.transform(s1_texts), norm="l2")

# Test batch multiplication
batch_sizes = [20, 50, 100]
for b in batch_sizes:
    t1 = time.time()
    b_mat = (X_s1[:b] @ X_cand.T).toarray()
    print(f"Batch size {b}: {time.time()-t1:.3f}s | b_mat shape: {b_mat.shape} | {b_mat.nbytes/(1024*1024):.1f} MB")
    print_ram(f"Batch {b}")
    del b_mat

print("Benchmark complete!")
