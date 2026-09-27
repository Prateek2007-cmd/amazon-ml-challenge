"""
Phase 7: Full Test Inference for the Strongest Verified Challenger (Config P5).
Amazon ML Challenge 2026 - Business Entity Resolution

Architecture:
- Model E (44 Features)
- LightGBM Model P5 (depth=9, leaves=63, lr=0.03, n_est=400, min_child=50)
- Decision Threshold: tau = 0.72
- Candidate Generation: Champion 6-Pass Union Semantic (K=10, cosine>=0.30, zero-protected)
- Memory-Safe Bounded Chunks (25,000 S1 per chunk)
- Compact Candidate Representation (3 strings per candidate: norm_n, norm_a, country)
- On-the-Fly Pairwise Feature Extraction (Zero RAM bloat)
- Disk-Backed SQLite Intermediate Storage (Zero RAM usage during merge)
- Country-Partitioned Processing (France, US, India, Open-Set)
- 100% Prediction Containment in Candidate Set Guaranteed
- Fully Resumable Checkpoint Architecture
"""
import os, sys, time, re, gc, joblib, json, sqlite3
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from config import Config
from data_loader import get_data_paths
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

PINCODE_RE = re.compile(r"\b(\d{5,6})\b")

def precompute_s1(row):
    raw_n = row.get("business_name", "")
    raw_a = row.get("business_address", "")
    c = str(row.get("country", "")).strip().upper()
    clean_n = raw_n.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    clean_a = raw_a.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    toks_n = set(t for t in norm_n.split() if t not in NAME_STOPWORDS)
    toks_a = set(t for t in norm_a.split() if t not in ADDR_STOPWORDS)
    nums_list = extract_clean_numbers(norm_a)
    nums = set(nums_list)
    trans_n = transliterate_indic(raw_n)
    norm_trans_n = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""
    pin_m = PINCODE_RE.search(raw_a)
    pincode = pin_m.group(1) if pin_m else ""
    n_compact = norm_n.replace(" ", "")
    char4 = set(n_compact[i:i+4] for i in range(len(n_compact)-3)) if len(n_compact) >= 4 else set()
    return (norm_n, norm_a, c, toks_n, toks_a, nums, nums_list, norm_trans_n, pincode, char4)

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
        if len(t) >= 4: p5.append(("name_tok", t))
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

def extract_features_onthefly(s1_tuple, cand_norm_n, cand_norm_a, cand_country, cand_id, sem_info):
    n1, a1, c1, toks_n1, toks_a1, nums_1, list_nums1, norm_trans_n1, pin1, c4_1 = s1_tuple
    n2 = cand_norm_n
    a2 = cand_norm_a
    c2 = cand_country

    toks_n2 = set(t for t in n2.split() if t not in NAME_STOPWORDS)
    toks_a2 = set(t for t in a2.split() if t not in ADDR_STOPWORDS)
    list_nums2 = extract_clean_numbers(a2)
    nums_2 = set(list_nums2)
    norm_trans_n2 = ""
    pin_m2 = PINCODE_RE.search(a2)
    pin2 = pin_m2.group(1) if pin_m2 else ""
    n_compact2 = n2.replace(" ", "")
    c4_2 = set(n_compact2[i:i+4] for i in range(len(n_compact2)-3)) if len(n_compact2) >= 4 else set()

    # 31 base features
    name_union = toks_n1 | toks_n2
    name_inter = toks_n1 & toks_n2
    name_jaccard = len(name_inter) / len(name_union) if name_union else 0.0
    ratio_n = fuzz.ratio(n1, n2) / 100.0
    tsort_n = fuzz.token_sort_ratio(n1, n2) / 100.0
    tset_n = fuzz.token_set_ratio(n1, n2) / 100.0
    partial_n = fuzz.partial_ratio(n1, n2) / 100.0
    addr_union = toks_a1 | toks_a2
    addr_inter = toks_a1 & toks_a2
    addr_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0
    num_union = nums_1 | nums_2
    num_inter = nums_1 & nums_2
    num_jaccard = len(num_inter) / len(num_union) if num_union else 0.0
    empty_addr = float(len(a1) == 0 or len(a2) == 0)
    if empty_addr:
        ratio_a = tsort_a = tset_a = partial_a = 0.0
    else:
        ratio_a = fuzz.ratio(a1, a2) / 100.0
        tsort_a = fuzz.token_sort_ratio(a1, a2) / 100.0
        tset_a = fuzz.token_set_ratio(a1, a2) / 100.0
        partial_a = fuzz.partial_ratio(a1, a2) / 100.0
    country_match = float(c1 == c2 and c1 != "")
    is_s3 = float(str(cand_id).startswith("S3-"))
    len_diff_n = abs(len(n1) - len(n2))
    max_len_n = max(len(n1), len(n2), 1)
    len_ratio_n = 1.0 - (len_diff_n / max_len_n)
    len_diff_a = abs(len(a1) - len(a2))
    mult = tsort_n * (tsort_a if not empty_addr else tsort_n)
    min_sim = min(tsort_n, tsort_a) if not empty_addr else tsort_n
    max_sim = max(tsort_n, tsort_a)
    mean_sim = (tsort_n + tsort_a) / 2.0 if not empty_addr else tsort_n
    trans_sim = 0.0
    if norm_trans_n2:
        trans_sim = fuzz.token_sort_ratio(n1, norm_trans_n2) / 100.0
    elif norm_trans_n1:
        trans_sim = fuzz.token_sort_ratio(norm_trans_n1, n2) / 100.0
    effective_name_sim = max(tsort_n, trans_sim)
    c4_union = c4_1 | c4_2
    c4_inter = c4_1 & c4_2
    char4_jaccard = len(c4_inter) / len(c4_union) if c4_union else 0.0
    has_num_match = float(len(num_inter) > 0)
    empty_addr_name_strength = (effective_name_sim * name_jaccard) if empty_addr else 0.0
    pin_flag = 1.0 if (pin1 and pin2 and pin1 == pin2) else (-1.0 if (pin1 and pin2 and pin1 != pin2) else 0.0)

    base = [
        float(n1 == n2 and len(n1) > 0), ratio_n, tsort_n, tset_n, partial_n,
        name_jaccard, float(len(name_inter)), float(len_diff_n), len_ratio_n, 0.0,
        float(a1 == a2 and len(a1) > 0), ratio_a, tsort_a, tset_a, partial_a,
        addr_jaccard, num_jaccard, float(len(num_inter)), float(len_diff_a), empty_addr,
        country_match, is_s3, mult, min_sim, max_sim, mean_sim,
        effective_name_sim, char4_jaccard, has_num_match, empty_addr_name_strength, pin_flag
    ]

    # Semantic features
    if sem_info and sem_info[1] <= 10 and sem_info[0] >= 0.30:
        sem_cos = float(sem_info[0])
        sem_rk_feat = 1.0 / float(sem_info[1])
    else:
        sem_cos, sem_rk_feat = 0.0, 0.0

    # Interaction features
    f_name_x_addr = tsort_n * (tsort_a if not empty_addr else 0.0)
    f_sem_cos_x_addr = sem_cos * (tsort_a if not empty_addr else 0.0)
    f_sem_cos_x_name = sem_cos * tsort_n
    f_sem_rk_x_addr = sem_rk_feat * (tsort_a if not empty_addr else 0.0)
    f_sem_rk_x_name = sem_rk_feat * tsort_n

    # Conflict defense features
    f_exact_house_num_match = float(len(list_nums1) > 0 and len(list_nums2) > 0 and list_nums1[0] == list_nums2[0])
    f_addr_num_conflict = float(len(nums_1) > 0 and len(nums_2) > 0 and len(num_inter) == 0)
    f_shared_bldg_diff_biz = float(len(num_inter) > 0) * (1.0 - tsort_n)
    f_same_name_diff_addr = (tsort_n * (1.0 - tsort_a)) if not empty_addr else 0.0
    f_name_high_addr_low = float(tsort_n >= 0.85 and tsort_a < 0.40 and not empty_addr)
    f_name_high_addr_conflict = float(tsort_n >= 0.80 and f_addr_num_conflict == 1.0)

    return (
        base +
        [sem_cos, sem_rk_feat] +
        [f_name_x_addr, f_sem_cos_x_addr, f_sem_cos_x_name, f_sem_rk_x_addr, f_sem_rk_x_name] +
        [f_exact_house_num_match, f_addr_num_conflict, f_shared_bldg_diff_biz,
         f_same_name_diff_addr, f_name_high_addr_low, f_name_high_addr_conflict]
    )

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = MEMORY;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            s1_id TEXT PRIMARY KEY,
            pred_str TEXT,
            cand_str TEXT
        );
    """)
    conn.commit()
    return conn

def stream_country_s1_chunks(test_s1_path, country_name, chunk_size=25000):
    chunk = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        id_idx = header.index("entity_id")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        c_idx = header.index("country")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[c_idx].strip().upper() == country_name:
                row_dict = {
                    "entity_id": parts[id_idx],
                    "business_name": parts[name_idx],
                    "business_address": parts[addr_idx],
                    "country": country_name
                }
                chunk.append(row_dict)
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

def process_country_partition(
    country_name, country_s1_count, paths, clf, threshold, art_dir, db_conn,
    batch_matmul_size=50
):
    print(f"\n{'='*80}\nPROCESSING COUNTRY: [{country_name}] ({country_s1_count:,d} S1 records)\n{'='*80}", flush=True)
    t0_c = time.time()

    # 1. Stream country candidate records from test S2 and test S3
    print(f"Loading candidate records for [{country_name}]...", flush=True)
    cand_records = {} # eid -> (norm_n, norm_a, country) (Ultra-low RAM)
    cand_texts = []
    cand_id_list = []

    # 6-pass blocking inverted index
    idx_s2 = defaultdict(list)
    idx_s3 = defaultdict(list)

    for src_key in ["test_s2", "test_s3"]:
        path = paths[src_key]
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            addr_idx = header.index("business_address")
            country_idx = header.index("country")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts and parts[country_idx].strip().upper() == country_name:
                    eid = parts[id_idx]
                    raw_n = parts[name_idx]
                    raw_a = parts[addr_idx]

                    norm_n = normalize_name(raw_n.replace("-", " ").replace("/", " "))
                    norm_a = normalize_address(raw_a.replace("-", " ").replace("/", " "))

                    cand_records[eid] = (norm_n, norm_a, country_name)
                    cand_texts.append(f"{norm_n} {norm_a}")
                    cand_id_list.append(eid)

                    # Build 6-pass blocking index
                    target_idx = idx_s3 if eid.startswith("S3-") else idx_s2
                    for pass_keys in extract_champion_keys(raw_n, raw_a, country_name):
                        for k in pass_keys:
                            target_idx[k].append(eid)

    num_cands = len(cand_id_list)
    print(f"Loaded {num_cands:,d} candidate records for [{country_name}] in {time.time()-t0_c:.1f}s.", flush=True)
    print(f"6-pass blocking keys: S2={len(idx_s2):,d}, S3={len(idx_s3):,d}", flush=True)

    # 2. Fit TF-IDF Vectorizer and normalize candidate representations
    print(f"Building TF-IDF semantic representation matrix for [{country_name}]...", flush=True)
    t0_tfidf = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.35, max_features=30000, sublinear_tf=True
    )
    cand_sample = cand_texts[:100000] if len(cand_texts) > 100000 else cand_texts
    vectorizer.fit(cand_sample)

    X_cand = sklearn_normalize(vectorizer.transform(cand_texts), norm="l2")
    X_cand_T = X_cand.T # (30000, num_cands)
    print(f"TF-IDF matrix built in {time.time()-t0_tfidf:.1f}s. Matrix shape: {X_cand.shape}, nnz: {X_cand.nnz:,d}", flush=True)

    # Free cand_texts strings to save hundreds of MBs of RAM!
    del cand_texts, cand_sample
    gc.collect()

    # 3. Process S1 entities in bounded chunks of 25,000 streamed from disk
    CHUNK_SIZE = 25000
    total_pairs_scored = 0
    total_matches_predicted = 0

    cur = db_conn.cursor()
    total_processed_s1 = 0

    for chunk_idx, chunk_s1 in enumerate(stream_country_s1_chunks(paths["test_s1"], country_name, CHUNK_SIZE)):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = chunk_start + len(chunk_s1)
        total_processed_s1 += len(chunk_s1)
        t0_chk = time.time()

        # Step A: 6-pass champion candidate generation
        champion_cands_chunk = {}
        cached_s1_chunk = {}
        queries_for_sem = []

        for s1_row in chunk_s1:
            s1_id = s1_row["entity_id"]
            e1 = precompute_s1(s1_row)
            cached_s1_chunk[s1_id] = e1

            # 6-pass blocking
            ordered_passes = extract_champion_keys(s1_row["business_name"], s1_row["business_address"], country_name)
            cands = set()
            for p_idx, pass_keys in enumerate(ordered_passes):
                cap = 150 if p_idx == 4 else 400
                for k in pass_keys:
                    p2 = idx_s2.get(k, [])
                    p3 = idx_s3.get(k, [])
                    if len(p2) <= cap: cands.update(p2)
                    if len(p3) <= cap: cands.update(p3)
                    if len(cands) >= 120: break
                if len(cands) >= 120: break

            champion_cands_chunk[s1_id] = cands

            # Zero-Protection: only query semantic if champion found > 0 candidates
            if len(cands) > 0:
                rep_t = f"{e1[0]} {e1[1]}"
                queries_for_sem.append((s1_id, rep_t))

        # Step B: Semantic retrieval for queries_for_sem
        sem_top_records_chunk = defaultdict(dict)
        if queries_for_sem:
            sem_texts = [item[1] for item in queries_for_sem]
            X_q = sklearn_normalize(vectorizer.transform(sem_texts), norm="l2")

            # Batch multiplication in memory-safe slices
            for b_start in range(0, len(queries_for_sem), batch_matmul_size):
                b_end = min(b_start + batch_matmul_size, len(queries_for_sem))
                sim_batch = (X_q[b_start:b_end] @ X_cand_T).toarray()

                for local_i in range(b_end - b_start):
                    q_idx = b_start + local_i
                    s1_id = queries_for_sem[q_idx][0]
                    sims = sim_batch[local_i]

                    # Top 10 with score >= 0.30
                    if len(sims) > 10:
                        top_idx = np.argpartition(-sims, 10)[:10]
                        top_idx = top_idx[np.argsort(-sims[top_idx])]
                    else:
                        top_idx = np.argsort(-sims)

                    for rank_0, idx in enumerate(top_idx[:10]):
                        score = float(sims[idx])
                        if score >= 0.30:
                            cid = cand_id_list[idx]
                            sem_top_records_chunk[s1_id][cid] = (score, rank_0 + 1)

                del sim_batch

            del X_q

        # Step C: Union candidates and feature extraction
        chunk_final_cands = {}
        chunk_pairs = []
        chunk_feats = []

        for s1_row in chunk_s1:
            s1_id = s1_row["entity_id"]
            champ_c = champion_cands_chunk[s1_id]
            sem_c = sem_top_records_chunk.get(s1_id, {})

            # Zero-protection union
            if len(champ_c) > 0:
                final_cand_set = champ_c | set(sem_c.keys())
            else:
                final_cand_set = set()

            chunk_final_cands[s1_id] = final_cand_set

            e1 = cached_s1_chunk[s1_id]
            for cid in final_cand_set:
                e2 = cand_records.get(cid)
                if e2 is None: continue
                # Instant country check
                if e1[2] and e2[2] and e1[2] != e2[2]:
                    continue

                sem_info = sem_c.get(cid)
                feat_44 = extract_features_onthefly(e1, e2[0], e2[1], e2[2], cid, sem_info)
                chunk_pairs.append((s1_id, cid))
                chunk_feats.append(feat_44)

        # Step D: LightGBM scoring
        chunk_matches = defaultdict(set)
        if chunk_feats:
            X_chunk = np.array(chunk_feats, dtype=np.float32)
            probs = clf.predict_proba(X_chunk)[:, 1]
            for (sid, cid), prob in zip(chunk_pairs, probs):
                if prob >= threshold:
                    chunk_matches[sid].add(cid)
            total_pairs_scored += len(chunk_pairs)

        # Step E: Stream chunk results directly into SQLite database
        db_records = []
        for s1_row in chunk_s1:
            s1_id = s1_row["entity_id"]
            cands = chunk_final_cands[s1_id]
            preds = chunk_matches[s1_id]

            # 100% containment check
            assert preds.issubset(cands), f"Containment violated for {s1_id}"

            total_matches_predicted += len(preds)
            cand_str = ",".join(sorted(cands))
            pred_str = ",".join(sorted(preds))
            db_records.append((s1_id, pred_str, cand_str))

        cur.executemany("INSERT OR REPLACE INTO results (s1_id, pred_str, cand_str) VALUES (?, ?, ?);", db_records)
        db_conn.commit()

        del champion_cands_chunk, cached_s1_chunk, sem_top_records_chunk
        del chunk_final_cands, chunk_pairs, chunk_feats, chunk_matches, db_records
        gc.collect()

        elapsed = time.time() - t0_chk
        pct = total_processed_s1 / country_s1_count * 100
        print(f"  [{country_name}] Chunk {chunk_start:,d}..{chunk_end:,d} ({pct:.1f}%): Scored in {elapsed:.1f}s | Matches so far: {total_matches_predicted:,d}", flush=True)

    del cand_records, cand_id_list, idx_s2, idx_s3, X_cand, X_cand_T, vectorizer
    gc.collect()

    country_runtime = time.time() - t0_c
    print(f"Completed [{country_name}] in {country_runtime:.1f}s ({country_runtime/60:.2f} mins). Scored {total_pairs_scored:,d} pairs | Matches: {total_matches_predicted:,d}", flush=True)
    return {
        "country": country_name,
        "s1_count": country_s1_count,
        "cands_scored": total_pairs_scored,
        "matches_predicted": total_matches_predicted,
        "runtime_s": country_runtime
    }

def main():
    print("=" * 80)
    print("PHASE 7: FULL TEST INFERENCE ENGINE (CONFIG P5)")
    print("Amazon ML Challenge 2026 - Business Entity Resolution")
    print("=" * 80)
    t0_start = time.time()
    paths = get_data_paths(is_sample=False)

    phase7_dir = os.path.join(REPO_ROOT, "experiments", "final_challenger", "phase7")
    art_dir = os.path.join(phase7_dir, "artifacts")
    out_dir = os.path.join(phase7_dir, "output")
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load trained LightGBM model
    model_path = os.path.join(art_dir, "final_model_p5.joblib")
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return

    print(f"Loading trained final model from: {model_path}...", flush=True)
    clf = joblib.load(model_path)
    TAU = 0.72
    print(f"Model loaded. Decision threshold tau = {TAU:.2f}", flush=True)

    # 2. Initialize SQLite disk database for intermediate storage
    db_path = os.path.join(art_dir, "test_inference_results.db")
    db_conn = init_db(db_path)
    print(f"Initialized disk-backed intermediate database at: {db_path}", flush=True)

    # 3. Check checkpoint
    ckpt_path = os.path.join(art_dir, "country_checkpoints.json")
    completed_countries = {}
    if os.path.isfile(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                completed_countries = json.load(f)
            print(f"Loaded existing checkpoint: {list(completed_countries.keys())}", flush=True)
        except Exception:
            completed_countries = {}

    # 4. Count test S1 records by country
    print("Counting test S1 records by country...", flush=True)
    from collections import Counter
    country_counts = Counter()
    all_test_s1_order = []

    with open(paths["test_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        id_idx = header.index("entity_id")
        c_idx = header.index("country")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                eid = parts[id_idx]
                c = parts[c_idx].strip().upper()
                country_counts[c] += 1
                all_test_s1_order.append(eid)

    total_test_s1 = len(all_test_s1_order)
    print(f"Total test S1 entities: {total_test_s1:,d}")
    for c, cnt in country_counts.most_common():
        print(f"  - {c}: {cnt:,d} entities ({cnt/total_test_s1*100:.1f}%)")

    # 5. Process each country sequentially (FRANCE -> US -> INDIA)
    ordered_countries = ["FRANCE", "US", "INDIA"]
    for c in country_counts.keys():
        if c not in ordered_countries:
            ordered_countries.append(c)

    for c in ordered_countries:
        if c not in country_counts or country_counts[c] == 0:
            continue
        if c in completed_countries:
            print(f"\nCountry [{c}] already completed according to checkpoint. Skipping.", flush=True)
            continue

        if c == "FRANCE": b_size = 50
        elif c == "US": b_size = 25
        else: b_size = 20

        summary = process_country_partition(
            c, country_counts[c], paths, clf, TAU, art_dir, db_conn,
            batch_matmul_size=b_size
        )
        completed_countries[c] = summary
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(completed_countries, f, indent=2)

    # 6. Stream from SQLite database to final output files in EXACT original test S1 order
    print("\n" + "=" * 80)
    print("ASSEMBLING FINAL OUTPUT FILES FROM SQLITE IN EXACT TEST S1 ORDER")
    print("=" * 80)
    t0_merge = time.time()

    final_matching_path = os.path.join(out_dir, "matching_results.tsv")
    final_candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")

    cur = db_conn.cursor()

    print(f"Writing final matching results and candidate pairs...", flush=True)
    with open(final_matching_path, "w", encoding="utf-8") as f_match, open(final_candidate_path, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        BATCH_FETCH = 500
        for b_start in range(0, total_test_s1, BATCH_FETCH):
            b_end = min(b_start + BATCH_FETCH, total_test_s1)
            b_ids = all_test_s1_order[b_start:b_end]

            placeholders = ",".join("?" * len(b_ids))
            cur.execute(f"SELECT s1_id, pred_str, cand_str FROM results WHERE s1_id IN ({placeholders})", b_ids)
            rows = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

            for s1_id in b_ids:
                pred_str, cand_str = rows.get(s1_id, ("", ""))
                f_match.write(f"{s1_id}\t{pred_str}\n")
                f_cand.write(f"{s1_id}\t{cand_str}\n")

    db_conn.close()

    print(f"Final output assembly completed in {time.time()-t0_merge:.1f}s.")
    print(f"Output files generated:")
    print(f"  - {final_matching_path} ({os.path.getsize(final_matching_path):,d} bytes)")
    print(f"  - {final_candidate_path} ({os.path.getsize(final_candidate_path):,d} bytes)")

    # 7. Save configuration JSON
    cfg_data = {
        "model_architecture": "Phase 6 Model E (44 Features)",
        "num_features": 44,
        "feature_list": [
            "31 Base Features (RapidFuzz name/address/numbers/country/etc)",
            "semantic_cosine", "1_over_semantic_rank",
            "name_x_addr", "sem_cos_x_addr", "sem_cos_x_name", "sem_rk_x_addr", "sem_rk_x_name",
            "exact_house_num_match", "addr_num_conflict", "shared_bldg_diff_biz",
            "same_name_diff_addr", "name_high_addr_low", "name_high_addr_conflict"
        ],
        "candidate_generator": "Champion 6-Pass Prioritized Blocking UNION Semantic TF-IDF",
        "semantic_retrieval": {
            "top_k": 10,
            "cosine_threshold": 0.30,
            "zero_protection": True
        },
        "lightgbm_parameters": {
            "max_depth": 9,
            "num_leaves": 63,
            "learning_rate": 0.03,
            "n_estimators": 400,
            "min_child_samples": 50,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42
        },
        "threshold_tau": 0.72,
        "validation_macro_f05": 0.9724,
        "validation_precision": 0.9828,
        "validation_recall": 0.9505,
        "validation_5fold_mean": "0.9723 +/- 0.0006",
        "test_s1_total": total_test_s1,
        "country_summaries": completed_countries,
        "total_test_runtime_s": time.time() - t0_start
    }
    cfg_path = os.path.join(phase7_dir, "final_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)
    print(f"Saved final configuration to: {cfg_path}", flush=True)

    print("\nFULL TEST INFERENCE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
