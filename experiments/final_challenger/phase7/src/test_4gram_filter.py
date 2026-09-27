import time
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

print("=" * 80)
print("TESTING INVERTED 4-GRAM FILTER VS BRUTE FORCE COSINE >= 0.30")
print("=" * 80)

def load_sample(path, n):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= n: break
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            texts.append(f"{parts[1]} {parts[2]}")
    return ids, texts

s1_ids, s1_texts = load_sample("dataset/test/test_source1.tsv", 500)
cand_ids, cand_texts = load_sample("dataset/test/test_source2.tsv", 50000)

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
X_cand = sklearn_normalize(vectorizer.fit_transform(cand_texts), norm="l2")
X_s1 = sklearn_normalize(vectorizer.transform(s1_texts), norm="l2")

# 1. Full Brute Force (Ground Truth)
t0 = time.time()
sim_dense = (X_s1[:200] @ X_cand.T).toarray()
exact_matches = defaultdict(dict)
for i in range(200):
    sims = sim_dense[i]
    top_idx = np.argpartition(-sims, 10)[:10]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    for rank_0, idx in enumerate(top_idx[:10]):
        if sims[idx] >= 0.30:
            exact_matches[i][cand_ids[idx]] = (float(sims[idx]), rank_0 + 1)

total_exact = sum(len(m) for m in exact_matches.values())
print(f"Brute force: {total_exact} candidate matches >= 0.30 found in {time.time()-t0:.2f}s")

# 2. Build lightweight char-4gram inverted index
t1 = time.time()
idx_4gram = defaultdict(list)
for i, text in enumerate(cand_texts):
    # Extract unique char 4-grams from clean text
    compact = text.lower().replace(" ", "")
    ngrams = set(compact[j:j+4] for j in range(len(compact)-3))
    for ng in ngrams:
        idx_4gram[ng].append(i)
print(f"Built 4-gram index in {time.time()-t1:.2f}s ({len(idx_4gram):,d} keys)")

# 3. Query via 4-gram index
t2 = time.time()
recovered_exact = 0
filtered_matches = defaultdict(dict)

for i in range(200):
    q_text = s1_texts[i]
    q_compact = q_text.lower().replace(" ", "")
    q_ngrams = set(q_compact[j:j+4] for j in range(len(q_compact)-3))
    
    # Union of postings
    cand_indices = set()
    for ng in q_ngrams:
        postings = idx_4gram.get(ng, [])
        if len(postings) <= 2000: # cap common n-grams
            cand_indices.update(postings)
    
    if cand_indices:
        cand_idx_arr = np.array(list(cand_indices), dtype=np.int32)
        # Dot product ONLY against active candidate subset!
        q_vec = X_s1[i]
        sub_X_cand = X_cand[cand_idx_arr]
        scores = (q_vec @ sub_X_cand.T).toarray().ravel()
        
        mask = scores >= 0.30
        if np.any(mask):
            matched_sub_idx = np.where(mask)[0]
            matched_scores = scores[matched_sub_idx]
            order = np.argsort(-matched_scores)[:10]
            for rank_0, o_idx in enumerate(order):
                real_idx = cand_idx_arr[matched_sub_idx[o_idx]]
                filtered_matches[i][cand_ids[real_idx]] = (float(matched_scores[o_idx]), rank_0 + 1)

t_query = time.time() - t2
print(f"Queried 200 via 4-gram index in {t_query:.3f}s ({t_query/200*1000:.2f} ms/query)!")

# Check accuracy
captured = 0
for i in range(200):
    for cid in exact_matches[i]:
        if cid in filtered_matches[i]:
            captured += 1

print(f"Recall vs Brute Force: {captured}/{total_exact} ({captured/max(total_exact,1)*100:.2f}%)!")
