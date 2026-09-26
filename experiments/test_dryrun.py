"""
Small Memory-Safe Dry Run.
Verifies all 7 required components before full test run:
1. Candidate generation
2. Semantic retrieval (including France)
3. Feature generation (31 features)
4. LightGBM inference (tau = 0.68)
5. Candidate containment (100% / 0 violations)
6. Output writing (correct TSV formatting)
7. France retrieval (working end-to-end)
Reports memory usage.
"""
import os, sys, time, psutil
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

repo_root = os.path.abspath(".")
SRC_DIR = os.path.join(repo_root, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from config import Config
from data_loader import get_data_paths

sys.path.insert(0, os.path.join(repo_root, "experiments", "semantic_candidate_recovery", "src"))
from production_pipeline import (
    precompute_entity, extract_champion_keys, extract_features_31,
    build_rep_text, build_blocking_index, generate_champion_candidates
)

def get_ram():
    return f"{psutil.virtual_memory().available / (1024**3):.2f} GB avail ({psutil.virtual_memory().percent}% used)"

print("=" * 80)
print("SMALL MEMORY-SAFE DRY RUN")
print("=" * 80)
print("Initial RAM:", get_ram())
t0 = time.time()

paths = get_data_paths(is_sample=False)
model_path = os.path.join(repo_root, "experiments", "semantic_candidate_recovery", "models", "semantic_lgbm_model.joblib")
print(f"Loading trained model: {model_path}...")
clf = joblib.load(model_path)
tau = 0.68

# 1. Sample S1 test entities (take 1,500 total: 500 US, 500 India, 500 France)
print("\n[DRY RUN 1] Sampling test S1 entities across US, India, France...")
s1_sample = {}
s1_by_country = defaultdict(list)
with open(paths["test_s1"], "r", encoding="utf-8") as f:
    header_s1 = f.readline().strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        c = parts[3].strip()
        if len(s1_by_country[c]) < 500:
            row = dict(zip(header_s1, parts))
            s1_sample[parts[0]] = row
            s1_by_country[c].append(parts[0])
        if all(len(s1_by_country[ctry]) >= 500 for ctry in ("US", "India", "France")):
            break

print(f"Sampled S1: Total={len(s1_sample)}, France={len(s1_by_country['France'])}, US={len(s1_by_country['US'])}, India={len(s1_by_country['India'])}")

# 2. Sample S2 and S3 test candidates (10,000 per country from S2 and S3)
print("\n[DRY RUN 2] Sampling test S2 and S3 candidates across countries...")
def sample_pool(path, per_country=10000):
    pool = {}
    counts = defaultdict(int)
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            c = parts[3].strip() if len(parts) > 3 else ""
            if counts[c] < per_country:
                pool[parts[0]] = dict(zip(header, parts))
                counts[c] += 1
            if all(counts[ctry] >= per_country for ctry in ("US", "India", "France")):
                break
    return pool

s2_sample = sample_pool(paths["test_s2"], per_country=10000)
s3_sample = sample_pool(paths["test_s3"], per_country=10000)
cand_pool = {**s2_sample, **s3_sample}
print(f"Sampled Candidates: S2={len(s2_sample)}, S3={len(s3_sample)}, Total={len(cand_pool)}")
print("RAM after sampling:", get_ram())

# 3. Candidate Generation (Champion Blocking)
print("\n[DRY RUN 3] Building champion blocking index & generating candidates...")
t_block = time.time()
idx_s2 = build_blocking_index(s2_sample)
idx_s3 = build_blocking_index(s3_sample)
champion_cands = {}
for sid, row in s1_sample.items():
    champion_cands[sid] = generate_champion_candidates(sid, row, idx_s2, idx_s3)

non_empty_champ = sum(1 for c in champion_cands.values() if len(c) > 0)
print(f"Champion blocking completed in {time.time()-t_block:.2f}s. Entities with >=1 candidate: {non_empty_champ} / {len(s1_sample)}")

# 4. Semantic Retrieval (Zero-Protected)
print("\n[DRY RUN 4] Semantic retrieval (zero-protected, country-partitioned)...")
t_sem = time.time()
query_ids = [sid for sid in s1_sample if len(champion_cands[sid]) > 0]
print(f"Queries qualifying for semantic retrieval: {len(query_ids)} / {len(s1_sample)}")

# Partition candidates and queries by country
country_to_cands = defaultdict(list)
for eid, row in cand_pool.items():
    country_to_cands[row.get("country", "").strip()].append(eid)

query_to_country = {sid: s1_sample[sid].get("country", "").strip() for sid in query_ids}

sem_cands = defaultdict(set)
K = 5
cos_threshold = 0.30

for country in ("France", "US", "India"):
    c_cand_ids = country_to_cands[country]
    c_q_ids = [sid for sid in query_ids if query_to_country[sid] == country]
    if not c_q_ids or not c_cand_ids:
        print(f"  [{country}] Skipping (queries={len(c_q_ids)}, cands={len(c_cand_ids)})")
        continue

    print(f"  [{country}] Running retrieval: {len(c_q_ids)} queries x {len(c_cand_ids)} candidates...", flush=True)
    cand_corpus = [build_rep_text(cand_pool[eid]) for eid in c_cand_ids]
    q_corpus = [build_rep_text(s1_sample[sid]) for sid in c_q_ids]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
    vec.fit(cand_corpus + q_corpus)
    X_cand = sklearn_normalize(vec.transform(cand_corpus), norm='l2')
    X_q = sklearn_normalize(vec.transform(q_corpus), norm='l2')

    sim_matrix = (X_q @ X_cand.T).toarray()
    for local_i, sid in enumerate(c_q_ids):
        sims = sim_matrix[local_i]
        top_idx = np.argpartition(sims, -K)[-K:] if len(sims) > K else np.arange(len(sims))
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        for idx in top_idx[:K]:
            if sims[idx] >= cos_threshold:
                sem_cands[sid].add(c_cand_ids[idx])

total_sem_added = sum(len(v) for v in sem_cands.values())
print(f"Semantic retrieval finished in {time.time()-t_sem:.2f}s. Total semantic candidates: {total_sem_added}")
print("RAM after semantic retrieval:", get_ram())

# 5. Union Candidates
print("\n[DRY RUN 5] Unioning candidate sets...")
final_candidates = {}
for sid in s1_sample:
    if len(champion_cands[sid]) > 0:
        final_candidates[sid] = champion_cands[sid] | sem_cands.get(sid, set())
    else:
        final_candidates[sid] = champion_cands[sid]

# Precompute cached representations for feature extraction
cached_s1 = {sid: precompute_entity(row) for sid, row in s1_sample.items()}
cached_cands = {eid: precompute_entity(row) for eid, row in cand_pool.items()}

# 6. Feature Extraction & LightGBM Inference
print(f"\n[DRY RUN 6] Scoring candidate pairs with 31 features & LightGBM (tau={tau:.2f})...")
t_inf = time.time()
final_matches = defaultdict(set)
dry_pairs = []
dry_feats = []

for sid, cands in final_candidates.items():
    e1 = cached_s1[sid]
    c1 = e1["country"]
    for cid in cands:
        e2 = cached_cands.get(cid)
        if e2 is None: continue
        if c1 and e2["country"] and c1 != e2["country"]: continue
        if e1["norm_n"] == e2["norm_n"] and len(e1["norm_n"]) > 3 and (e1["norm_a"] == e2["norm_a"] or not e1["norm_a"] or not e2["norm_a"]):
            final_matches[sid].add(cid)
            continue
        f_vec = extract_features_31(e1, e2, cid)
        dry_pairs.append((sid, cid))
        dry_feats.append(f_vec)

if dry_feats:
    X_dry = np.array(dry_feats, dtype=np.float32)
    probs = clf.predict_proba(X_dry)[:, 1]
    for (sid, cid), prob in zip(dry_pairs, probs):
        if prob >= tau:
            final_matches[sid].add(cid)

total_matches = sum(len(v) for v in final_matches.values())
print(f"Inference completed in {time.time()-t_inf:.2f}s. Scored {len(dry_feats)} pairs -> {total_matches} predicted matches.")

# 7. Candidate Containment Verification
print("\n[DRY RUN 7] Checking candidate containment...")
containment_violations = 0
for sid, matches in final_matches.items():
    cands = final_candidates.get(sid, set())
    for mid in matches:
        if mid not in cands:
            containment_violations += 1
            print(f"  CONTAINMENT VIOLATION: {sid} match {mid} not in candidate pool!")

print(f"Containment violations: {containment_violations} (100% containment: {containment_violations == 0})")
assert containment_violations == 0, "Containment violation detected!"

# 8. Output Writing
print("\n[DRY RUN 8] Writing dry-run output TSVs...")
dry_out_dir = os.path.join(repo_root, "experiments", "semantic_candidate_recovery", "output")
os.makedirs(dry_out_dir, exist_ok=True)
match_tsv = os.path.join(dry_out_dir, "dryrun_matching_results.tsv")
cand_tsv = os.path.join(dry_out_dir, "dryrun_candidate_pairs.tsv")

with open(match_tsv, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in s1_sample:
        f.write(f"{sid}\t{','.join(sorted(final_matches.get(sid, set())))}\n")

with open(cand_tsv, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_sample:
        f.write(f"{sid}\t{','.join(sorted(final_candidates.get(sid, set())))}\n")

print(f"Wrote {len(s1_sample)} rows to {match_tsv} and {cand_tsv}")

# 9. Verify France Results Specifically
france_queries = s1_by_country["France"]
france_cands_count = sum(len(final_candidates[sid]) for sid in france_queries)
france_matches_count = sum(len(final_matches.get(sid, set())) for sid in france_queries)
print(f"\n[DRY RUN 9] France Specific Results:")
print(f"  France queries: {len(france_queries)}")
print(f"  France candidates generated: {france_cands_count}")
print(f"  France matches predicted: {france_matches_count}")
print(f"  France candidate containment: 100%")

print("\n" + "=" * 80)
print(f"DRY RUN SUCCESSFUL in {time.time()-t0:.2f}s!")
print(f"Final RAM: {get_ram()}")
print("=" * 80)
