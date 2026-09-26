import os, sys, psutil, time
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

repo_root = os.path.abspath(".")
SRC_DIR = os.path.join(repo_root, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)
from normalization import normalize_name, normalize_address
sys.path.insert(0, os.path.join(repo_root, "experiments", "semantic_candidate_recovery", "src"))
from production_pipeline import extract_champion_keys, build_rep_text

def get_ram():
    return f"{psutil.virtual_memory().available / (1024**3):.2f} GB avail ({psutil.virtual_memory().percent}% used)"

print("Testing US candidate indexing & TF-IDF memory...")
print("Initial RAM:", get_ram())
t0 = time.time()

# Let's count how many US candidates there are
# We can load US candidates from S2 and S3
cands = []
for p in ("dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"):
    with open(p, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 3 and parts[3] == "US":
                cands.append((parts[0], parts[1], parts[2]))
                if len(cands) >= 500000: break
    if len(cands) >= 500000: break

print(f"Loaded {len(cands):,d} US candidates in {time.time()-t0:.2f}s | RAM: {get_ram()}")

# Test building TF-IDF on these 500k
texts = [f"{normalize_name(n)} {normalize_address(a)}" for _, n, a in cands]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
X = sklearn_normalize(vec.fit_transform(texts), norm='l2')
print(f"TF-IDF matrix shape: {X.shape}, RAM: {get_ram()}")
print(f"Done in {time.time()-t0:.2f}s")
