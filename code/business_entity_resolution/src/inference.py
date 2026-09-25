"""
Scalable Inference Engine for Business Entity Resolution.
Ultra-fast pre-tokenized and pre-normalized candidate scoring.
"""
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz
from normalization import normalize_name, normalize_address, extract_clean_numbers
from blocking import extract_blocking_keys
from features import extract_pairwise_features_prenorm, FEATURE_COLUMNS, NAME_STOPWORDS, ADDR_STOPWORDS
from config import Config

def precompute_record(row):
    """Pre-normalizes and pre-tokenizes a record once for ultra-fast matching."""
    raw_n = row.get("business_name", "")
    raw_a = row.get("business_address", "")
    c = str(row.get("country", "")).strip().upper()
    
    norm_n = normalize_name(raw_n)
    norm_a = normalize_address(raw_a)
    
    toks_n = set(t for t in norm_n.split() if t not in NAME_STOPWORDS)
    toks_a = set(t for t in norm_a.split() if t not in ADDR_STOPWORDS)
    nums = set(extract_clean_numbers(norm_a))
    
    return (norm_n, norm_a, c, toks_n, toks_a, nums)

def run_chunked_inference(clf, test_s1_df, index_s2, index_s3, lookup_s2, lookup_s3, threshold=Config.DEFAULT_THRESHOLD, chunk_size=50000):
    """
    Performs memory-efficient inference in chunks over Source-1 entities.
    Returns:
        matches: {s1_id: set(matched_ids)}
        all_candidates: {s1_id: set(candidate_ids)}
    """
    matches = defaultdict(set)
    all_candidates = defaultdict(set)
    
    total_s1 = len(test_s1_df)
    print(f"Starting chunked inference for {total_s1:,d} Source 1 entities (chunk size: {chunk_size:,d})...", flush=True)
    
    t_start = time.time()
    total_pairs_scored = 0
    total_matches_found = 0
    
    # Precompute candidate lookup records on demand or in bulk
    print("Pre-computing normalized candidate representation cache...", flush=True)
    t_pre = time.time()
    cached_candidates = {}
    for eid, row in lookup_s2.items():
        cached_candidates[eid] = precompute_record(row)
    for eid, row in lookup_s3.items():
        cached_candidates[eid] = precompute_record(row)
    print(f"Precomputed {len(cached_candidates):,d} candidate representations in {time.time()-t_pre:.2f}s", flush=True)
    
    # Process S1 entities in chunks
    for chunk_start in range(0, total_s1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_s1)
        chunk_df = test_s1_df.iloc[chunk_start:chunk_end]
        
        chunk_pairs = []
        chunk_feat_rows = []
        
        t_chunk = time.time()
        for _, s1_row in chunk_df.iterrows():
            s1_id = s1_row["entity_id"]
            s1_name = s1_row["business_name"]
            s1_addr = s1_row["business_address"]
            s1_country = s1_row["country"]
            
            keys = extract_blocking_keys(s1_name, s1_addr, s1_country)
            
            cand_set = set()
            for k in keys:
                cap = Config.MAX_NAME_TOK_POSTINGS if k[0] == "name_tok" else Config.MAX_COMPOSITE_POSTINGS
                p2 = index_s2.get(k, [])
                p3 = index_s3.get(k, [])
                if len(p2) <= cap:
                    cand_set.update(p2)
                if len(p3) <= cap:
                    cand_set.update(p3)
                if len(cand_set) >= Config.MAX_CANDIDATES_PER_S1:
                    break
                    
            all_candidates[s1_id] = cand_set
            
            # Precompute S1 once
            norm_n1, norm_a1, c1, toks_n1, toks_a1, nums_1 = precompute_record(s1_row)
            
            for cand_id in cand_set:
                cand_pre = cached_candidates.get(cand_id)
                if cand_pre is None:
                    continue
                norm_n2, norm_a2, c2, toks_n2, toks_a2, nums_2 = cand_pre
                
                # Instant reject on country mismatch
                if c1 and c2 and c1 != c2:
                    continue
                    
                # Instant match on exact normalized name and address
                if norm_n1 == norm_n2 and len(norm_n1) > 3 and (norm_a1 == norm_a2 or not norm_a1 or not norm_a2):
                    matches[s1_id].add(cand_id)
                    total_matches_found += 1
                    continue
                    
                feats = extract_pairwise_features_prenorm(
                    norm_n1, norm_a1, c1, toks_n1, toks_a1, nums_1,
                    norm_n2, norm_a2, c2, toks_n2, toks_a2, nums_2,
                    cand_id
                )
                chunk_pairs.append((s1_id, cand_id))
                chunk_feat_rows.append(feats)
                
        # Score chunk pairs
        if chunk_feat_rows:
            X_chunk = np.array(chunk_feat_rows, dtype=np.float32)
            probs = clf.predict_proba(X_chunk)[:, 1]
            for (s1_id, cand_id), prob in zip(chunk_pairs, probs):
                if prob >= threshold:
                    matches[s1_id].add(cand_id)
                    total_matches_found += 1
            total_pairs_scored += len(chunk_pairs)
            
        elapsed = time.time() - t_chunk
        progress_pct = chunk_end / total_s1 * 100
        print(f"  Chunk {chunk_start:,d}..{chunk_end:,d} ({progress_pct:.1f}%): Scored {len(chunk_pairs):,d} pairs in {elapsed:.1f}s | Matches so far: {total_matches_found:,d}", flush=True)
        
    print(f"Inference complete: Scored {total_pairs_scored:,d} total pairs in {time.time()-t_start:.1f}s. Total matches: {total_matches_found:,d}", flush=True)
    return matches, all_candidates
