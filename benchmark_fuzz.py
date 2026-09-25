"""
Benchmark of cached vs non-cached feature extraction.
"""
import time
from rapidfuzz import fuzz

# Uncached:
s1_n = "Blue Network Corporation"
s1_a = "177 1st Avenue, Nashville, TN"
c_n = "The Blue Netw0rk Corporation"
c_a = "177 1TH AVENUE, NASHVILLE, TN"

t0 = time.time()
for _ in range(50000):
    n1 = s1_n.lower()
    n2 = c_n.lower()
    r = fuzz.token_sort_ratio(n1, n2)
print(f"50,000 fuzz comparisons completed in {time.time()-t0:.2f}s")
