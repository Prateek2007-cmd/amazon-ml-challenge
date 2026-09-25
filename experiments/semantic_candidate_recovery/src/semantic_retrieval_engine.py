"""
Semantic Retrieval Engine for Candidate Recovery — Optimized Version.
Uses batch sparse matrix multiplication for fast cosine retrieval.
Evaluates local, offline TF-IDF char-ngram retrieval across representations.
Measures Candidate Recall, Candidate Growth, Hard Negatives, Zero-Match Contamination.
"""
import os
import sys
import time
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from config import Config
from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import (
    normalize_name, normalize_address, extract_name_aliases,
    extract_url_core, extract_clean_numbers,
)
from transliteration import transliterate_indic

NAME_STOPWORDS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp",
    "co", "company", "and", "the", "of", "in", "for", "center", "centre", "group",
    "services", "solutions", "enterprises", "associates", "holdings", "management",
    "international", "global", "industries", "systems", "technologies", "tech", "dba",
    "praivet", "treding"
}

ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "court", "ct", "way", "street", "road",
    "avenue", "boulevard", "drive", "lane", "apt", "suite", "ste", "fl", "floor",
    "bldg", "building", "near", "opp", "opposite", "behind", "phase", "sector",
    "plot", "shop", "no", "number", "us", "usa", "india", "null", "ground", "first",
    "second", "third", "c/o", "flats", "nagar", "dist", "district", "state"
}

def extract_champion_keys(name, address, country=""):
    clean_n = name.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    clean_a = address.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    content_n = [t for t in norm_n.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
    url_core = extract_url_core(norm_n)
    addr_nums = extract_clean_numbers(norm_a)
    addr_words = [t for t in norm_a.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and not any(c.isdigit() for c in t)]

    p1 = []
    if norm_n and len(norm_n) >= 4: p1.append(("exact_name", norm_n))
    if url_core and len(url_core) >= 4: p1.append(("url_core", url_core))
    if len(content_n) >= 2:
        p1.append(("name_pair", tuple(sorted([content_n[0], content_n[1]]))))
        if len(content_n) >= 3:
            p1.append(("name_pair", tuple(sorted([content_n[0], content_n[2]]))))
            p1.append(("name_pair", tuple(sorted([content_n[1], content_n[2]]))))
    p2 = []
    trans_n = transliterate_indic(name)
    if trans_n:
        norm_tn = normalize_name(trans_n.replace("-", " ").replace("/", " "))
        t_tokens = [t for t in norm_tn.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
        if len(t_tokens) >= 2:
            p2.append(("name_pair", tuple(sorted([t_tokens[0], t_tokens[1]]))))
        for t in t_tokens[:2]:
            p2.append(("name_tok", t))
    p3 = []
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]:
                p3.append(("name_num", nt, num))
    p4 = []
    if addr_nums and addr_words:
        for num in addr_nums[:3]:
            for w in addr_words[:3]:
                p4.append(("num_word", num, w))
    p5 = []
    for t in content_n[:2]:
        if len(t) >= 4:
            p5.append(("name_tok", t))
    p6 = []
    if len(addr_words) >= 2:
        p6.append(("addr_pair", tuple(sorted([addr_words[0], addr_words[1]]))))
        if len(addr_words) >= 3:
            p6.append(("addr_pair", tuple(sorted([addr_words[0], addr_words[2]]))))
    aliases = extract_name_aliases(name)
    if len(aliases) > 1:
        for alias in aliases[1:]:
            norm_alias = normalize_name(alias.replace("-", " ").replace("/", " "))
            a_toks = [t for t in norm_alias.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            if len(a_toks) >= 2:
                p1.append(("name_pair", tuple(sorted([a_toks[0], a_toks[1]]))))
            for at in a_toks[:2]:
                p5.append(("name_tok", at))
    return [p1, p2, p3, p4, p5, p6]

def build_rep_text(row, rep_type):
    name = row.get("business_name", "")
    addr = row.get("business_address", "")
    norm_n = normalize_name(name.replace("-", " ").replace("/", " "))
    norm_a = normalize_address(addr.replace("-", " ").replace("/", " "))
    trans_n = transliterate_indic(name)
    norm_tn = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""

    if rep_type == "A_name_only":
        return norm_n
    elif rep_type == "C_norm_name_addr":
        return f"{norm_n} {norm_a}"
    elif rep_type == "D_trans_name_addr":
        eff_n = f"{norm_n} {norm_tn}" if norm_tn else norm_n
        return f"{eff_n} {norm_a}"
    return norm_n

def main():
    print("=" * 80)
    print("SEMANTIC CANDIDATE RETRIEVAL (OPTIMIZED BATCH SPARSE)")
    print("=" * 80)
    t0 = time.time()
    paths = get_data_paths(is_sample=False)

    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())
    s1_needed = set(gt_dict.keys())
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"): needed_s2.add(m)
            elif m.startswith("S3-"): needed_s3.add(m)

    print(f"GT: {len(gt_dict):,d} S1 entities, {total_true:,d} true matches.", flush=True)

    s1_dict = {}
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in s1_needed:
                s1_dict[parts[0]] = dict(zip(header, parts))
                if len(s1_dict) == len(s1_needed): break

    def load_pool_stream(path, needed_set, max_pool=100000):
        pool = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                if eid in needed_set or i < max_pool:
                    pool[eid] = dict(zip(header, parts))
        return pool

    print("Loading pools...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3)

    # Build champion candidate sets
    print("Building champion candidate sets...", flush=True)
    idx_s2 = defaultdict(list)
    for eid, row in s2_dict.items():
        for pass_keys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pass_keys: idx_s2[k].append(eid)
    idx_s3 = defaultdict(list)
    for eid, row in s3_dict.items():
        for pass_keys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pass_keys: idx_s3[k].append(eid)

    MAX_CANDS = 120
    s1_list = list(s1_dict.keys())
    champion_cands = {}
    champ_hits = 0

    # Build country lookup for S1
    s1_countries = {}
    for s1_id, s1_row in s1_dict.items():
        s1_countries[s1_id] = str(s1_row.get("country", "")).strip().upper()

    # Build country lookup for candidates
    cand_countries = {}
    for eid, row in s2_dict.items():
        cand_countries[eid] = str(row.get("country", "")).strip().upper()
    for eid, row in s3_dict.items():
        cand_countries[eid] = str(row.get("country", "")).strip().upper()

    for s1_id in s1_list:
        s1_row = s1_dict[s1_id]
        true_matches = gt_dict.get(s1_id, set())
        ordered_passes = extract_champion_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
        cands = set()
        for p_idx, pass_keys in enumerate(ordered_passes):
            cap = 150 if p_idx == 4 else 400
            for k in pass_keys:
                p2 = idx_s2.get(k, [])
                p3 = idx_s3.get(k, [])
                if len(p2) <= cap: cands.update(p2)
                if len(p3) <= cap: cands.update(p3)
                if len(cands) >= MAX_CANDS: break
            if len(cands) >= MAX_CANDS: break
        champion_cands[s1_id] = cands
        champ_hits += len(true_matches & cands)

    base_recall = champ_hits / total_true
    base_avg_cands = sum(len(c) for c in champion_cands.values()) / len(champion_cands)
    print(f"Champion: Recall={base_recall*100:.2f}%, AvgCands={base_avg_cands:.1f}", flush=True)

    zero_match_s1 = set(sid for sid, m in gt_dict.items() if len(m) == 0)
    print(f"Zero-Match (card=0): {len(zero_match_s1):,d}", flush=True)

    # Group candidates by country
    country_to_cand_ids = defaultdict(list)
    all_cand_ids = list(s2_dict.keys()) + list(s3_dict.keys())
    for eid in all_cand_ids:
        c = cand_countries[eid]
        country_to_cand_ids[c].append(eid)

    representations = ["A_name_only", "C_norm_name_addr", "D_trans_name_addr"]
    K_grid = [5, 10, 20, 30]
    results_summary = []
    hard_negatives_list = []

    for rep in representations:
        print(f"\n{'='*60}", flush=True)
        print(f"REPRESENTATION: {rep}", flush=True)
        print(f"{'='*60}", flush=True)
        t_rep = time.time()

        # Process per-country (US and India are the main ones)
        # Build vectorizer and retrieve per country in batch
        countries = sorted(country_to_cand_ids.keys())
        print(f"  Countries: {[f'{c}({len(country_to_cand_ids[c])})' for c in countries]}", flush=True)

        # For each K, accumulate results
        # Pre-allocate: for each s1_id, store retrieved cand sets per K
        sem_cands_by_K = {K: defaultdict(set) for K in K_grid}

        for country in countries:
            c_cand_ids = country_to_cand_ids[country]
            if len(c_cand_ids) < 5:
                continue

            # S1 entities in this country
            c_s1_ids = [sid for sid in s1_list if s1_countries[sid] == country]
            if not c_s1_ids:
                continue

            print(f"  Country {country}: {len(c_cand_ids):,d} cands, {len(c_s1_ids):,d} S1s", flush=True)

            # Build corpus
            cand_corpus = [build_rep_text(s2_dict[eid] if eid.startswith("S2-") else s3_dict[eid], rep) for eid in c_cand_ids]
            s1_corpus = [build_rep_text(s1_dict[sid], rep) for sid in c_s1_ids]

            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )
            vectorizer.fit(cand_corpus + s1_corpus)
            X_cand = vectorizer.transform(cand_corpus)
            X_s1 = vectorizer.transform(s1_corpus)

            # L2 normalize for cosine similarity
            X_cand = sklearn_normalize(X_cand, norm='l2')
            X_s1 = sklearn_normalize(X_s1, norm='l2')

            max_K = max(K_grid)

            # Process S1 in chunks to avoid huge dense matrices
            CHUNK = 500
            for start in range(0, len(c_s1_ids), CHUNK):
                end = min(start + CHUNK, len(c_s1_ids))
                chunk_s1_ids = c_s1_ids[start:end]
                X_chunk = X_s1[start:end]

                # Sparse dot product -> dense similarity matrix (chunk_size x n_cands)
                sim_matrix = (X_chunk @ X_cand.T).toarray()

                for local_i, s1_id in enumerate(chunk_s1_ids):
                    sims = sim_matrix[local_i]

                    # Get top max_K indices
                    if len(sims) > max_K:
                        top_idx = np.argpartition(sims, -max_K)[-max_K:]
                    else:
                        top_idx = np.arange(len(sims))

                    # Sort by descending similarity
                    top_idx = top_idx[np.argsort(-sims[top_idx])]

                    for K in K_grid:
                        k_idx = top_idx[:K]
                        retrieved = set()
                        for idx in k_idx:
                            if sims[idx] >= 0.30:  # cosine gate
                                retrieved.add(c_cand_ids[idx])
                        sem_cands_by_K[K][s1_id] |= retrieved

            print(f"    Done {country} in {time.time()-t_rep:.1f}s", flush=True)

        # Now evaluate each K
        for K in K_grid:
            union_hits = 0
            semantic_only_hits = 0
            cand_lengths = []
            zero_match_contaminated = 0
            zero_match_cands_added = 0

            for s1_id in s1_list:
                true_matches = gt_dict.get(s1_id, set())
                champ_set = champion_cands[s1_id]
                sem_set = sem_cands_by_K[K].get(s1_id, set())
                union_set = champ_set | sem_set
                cand_lengths.append(len(union_set))

                union_found = true_matches & union_set
                union_hits += len(union_found)

                new_hits = (true_matches & sem_set) - champ_set
                semantic_only_hits += len(new_hits)

                if s1_id in zero_match_s1:
                    added = sem_set - champ_set
                    if added:
                        zero_match_contaminated += 1
                        zero_match_cands_added += len(added)

            union_recall = union_hits / total_true
            avg_cands = np.mean(cand_lengths)
            p95_cands = np.percentile(cand_lengths, 95)
            p99_cands = np.percentile(cand_lengths, 99)
            rec_delta = (union_recall - base_recall) * 100

            print(f"  [{rep} K={K:>2}] Recall: {union_recall*100:.2f}% ({rec_delta:+.2f}%, +{semantic_only_hits} new TPs) | AvgCands: {avg_cands:.1f} (+{avg_cands-base_avg_cands:.1f}) | P95: {p95_cands:.0f} P99: {p99_cands:.0f} | ZeroCont: {zero_match_contaminated}/{len(zero_match_s1)} ({zero_match_contaminated/len(zero_match_s1)*100:.1f}%)", flush=True)

            results_summary.append({
                "representation": rep,
                "K": K,
                "union_recall_pct": union_recall * 100,
                "recall_delta_pct": rec_delta,
                "incremental_TPs": semantic_only_hits,
                "avg_cands": avg_cands,
                "cands_added": avg_cands - base_avg_cands,
                "p95_cands": p95_cands,
                "p99_cands": p99_cands,
                "zero_match_contaminated": zero_match_contaminated,
                "zero_match_pct": zero_match_contaminated / len(zero_match_s1) * 100,
            })

            # Sample hard negatives for K=10
            if K == 10:
                count = 0
                for s1_id in s1_list[:500]:
                    true_matches = gt_dict.get(s1_id, set())
                    champ_set = champion_cands[s1_id]
                    sem_set = sem_cands_by_K[K].get(s1_id, set())
                    for cid in (sem_set - true_matches - champ_set):
                        s1_r = s1_dict[s1_id]
                        cr = s2_dict.get(cid) or s3_dict.get(cid)
                        if cr:
                            hard_negatives_list.append({
                                "s1_id": s1_id, "cand_id": cid, "rep": rep, "K": K,
                                "s1_name": s1_r["business_name"], "s1_addr": s1_r["business_address"],
                                "cand_name": cr["business_name"], "cand_addr": cr["business_address"],
                            })
                            count += 1
                            if count >= 20: break
                    if count >= 20: break

    # Save results
    out_dir = os.path.join(REPO_ROOT, "experiments", "semantic_candidate_recovery", "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    df_pareto = pd.DataFrame(results_summary)
    pareto_path = os.path.join(out_dir, "pareto_results.csv")
    df_pareto.to_csv(pareto_path, index=False)
    print(f"\nPareto results saved: {pareto_path}", flush=True)

    hn_path = os.path.join(out_dir, "semantic_hard_negatives.tsv")
    df_hn = pd.DataFrame(hard_negatives_list)
    if len(df_hn) > 0:
        df_hn.to_csv(hn_path, sep="\t", index=False)
    print(f"Hard negatives saved: {hn_path}", flush=True)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
