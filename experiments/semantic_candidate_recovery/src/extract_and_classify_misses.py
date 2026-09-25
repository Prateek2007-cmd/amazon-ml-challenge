"""
Phase 1 of Semantic Candidate Recovery Experiment:
Extract and Classify True Candidate-Generation Misses under the 0.9579 Champion.
"""
import os
import sys
import time
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from config import Config
from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import (
    normalize_name,
    normalize_address,
    extract_name_aliases,
    extract_url_core,
    extract_clean_numbers,
    NAME_ABBREVIATIONS,
    ADDRESS_ABBREVIATIONS,
)
from transliteration import transliterate_indic, has_indic_script

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

def precompute_entity(row):
    raw_n = row.get("business_name", "")
    raw_a = row.get("business_address", "")
    c = str(row.get("country", "")).strip().upper()
    clean_n = raw_n.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    clean_a = raw_a.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    toks_n = set(t for t in norm_n.split() if t not in NAME_STOPWORDS)
    toks_a = set(t for t in norm_a.split() if t not in ADDR_STOPWORDS)
    nums = extract_clean_numbers(norm_a)
    trans_n = transliterate_indic(raw_n)
    norm_trans_n = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""
    pin_m = PINCODE_RE.search(raw_a)
    pincode = pin_m.group(1) if pin_m else ""
    n_compact = norm_n.replace(" ", "")
    char4 = set(n_compact[i:i+4] for i in range(len(n_compact)-3)) if len(n_compact) >= 4 else set()
    content_n = [t for t in norm_n.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
    url_core = extract_url_core(norm_n)
    return {
        "raw_n": raw_n, "raw_a": raw_a,
        "norm_n": norm_n, "norm_a": norm_a, "country": c,
        "toks_n": toks_n, "toks_a": toks_a, "nums": nums,
        "norm_trans_n": norm_trans_n, "pincode": pincode,
        "char4": char4, "url_core": url_core, "content_n": content_n
    }

def extract_dynamic_multipass_keys(name, address, country=""):
    clean_n = name.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    clean_a = address.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    n_tokens = norm_n.split()
    content_n = [t for t in n_tokens if len(t) >= 2 and t not in NAME_STOPWORDS]
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

def classify_miss(s1, tgt):
    n1, n2 = s1["norm_n"], tgt["norm_n"]
    raw_n1, raw_n2 = s1["raw_n"], tgt["raw_n"]
    a1, a2 = s1["norm_a"], tgt["norm_a"]
    raw_a1, raw_a2 = s1["raw_a"], tgt["raw_a"]
    c1, c2 = s1["country"], tgt["country"]
    
    # 1. Cross-script / Transliteration
    has_indic_1 = has_indic_script(raw_n1)
    has_indic_2 = has_indic_script(raw_n2)
    if has_indic_1 or has_indic_2:
        if s1["norm_trans_n"] or tgt["norm_trans_n"]:
            return "A_cross_script_transliteration", "Indic script transliteration divergence"
            
    # 2. Domain / Website Variation
    u1, u2 = s1["url_core"], tgt["url_core"]
    if (u1 or u2) and (u1 in n2 or u2 in n1 or (u1 and u2 and u1 == u2)):
        return "D_domain_website_variation", "Domain/website URL vs business name"
        
    # 3. DBA / Brand vs Legal Entity
    aliases_1 = extract_name_aliases(raw_n1)
    aliases_2 = extract_name_aliases(raw_n2)
    if len(aliases_1) > 1 or len(aliases_2) > 1:
        return "B_dba_brand_vs_legal", "DBA or trade name slash/hyphen separator"
        
    # Acronyms (e.g. initials)
    words1 = [w for w in n1.split() if w not in NAME_STOPWORDS]
    words2 = [w for w in n2.split() if w not in NAME_STOPWORDS]
    if len(words1) >= 2 and len(words2) == 1:
        acronym1 = "".join(w[0] for w in words1)
        if acronym1 == words2[0]:
            return "K_acronym", f"Acronym match: {acronym1} vs {words2[0]}"
    elif len(words2) >= 2 and len(words1) == 1:
        acronym2 = "".join(w[0] for w in words2)
        if acronym2 == words1[0]:
            return "K_acronym", f"Acronym match: {words1[0]} vs {acronym2}"
            
    # Very short names
    if (len(n1) <= 3 or len(n2) <= 3) and n1 != n2:
        return "L_very_short_names", "Ultra-short name tokens (<4 chars)"
        
    # Token overlaps
    t1, t2 = s1["toks_n"], tgt["toks_n"]
    common_tokens = t1 & t2
    name_sim = fuzz.token_sort_ratio(n1, n2)
    
    # 5. Legal Suffix Divergence (e.g. one has pvt ltd, one has llc, core matches)
    core1 = " ".join(sorted(list(t1)))
    core2 = " ".join(sorted(list(t2)))
    if core1 == core2 and len(core1) > 0:
        return "E_legal_suffix_divergence", "Core name identical, legal suffix divergence"
        
    # 7. Weak or Missing Address
    if not a1 or not a2 or len(a1) < 5 or len(a2) < 5:
        if common_tokens:
            return "G_weak_or_missing_address", "One or both addresses missing/empty with weak name overlap"
        else:
            return "G_weak_or_missing_address", "Missing address and unshared name tokens"
            
    # Address numbers & granularity
    num1 = set(s1["nums"])
    num2 = set(tgt["nums"])
    addr_sim = fuzz.token_sort_ratio(a1, a2)
    
    if len(common_tokens) == 0:
        if name_sim < 30 and addr_sim > 70:
            return "B_dba_brand_vs_legal", "Completely different name at identical physical address (likely DBA/brand mismatch)"
        elif name_sim < 30 and addr_sim < 40:
            return "C_parent_vs_brand_or_unrelated", "Completely disjoint name and address (no apparent lexical overlap)"
        else:
            return "F_missing_business_name_tokens", "Disjoint name tokens with partial address similarity"
            
    if name_sim > 70 and addr_sim < 40:
        return "H_address_granularity_or_branch", "High name similarity but disjoint address (different branch/location or granularity)"
        
    if fuzz.ratio(n1, n2) > 60 and name_sim > 65:
        return "J_typographical_corruption", f"Spelling typo / phonetic variation ({n1} vs {n2})"
        
    if num1 and num2 and not (num1 & num2):
        return "H_address_granularity_or_branch", "Conflicting street/door numbers"
        
    return "N_other", f"Complex residual: name_sim={name_sim}, addr_sim={addr_sim}, common={common_tokens}"

def main():
    print("=" * 80)
    print("EXTRACTING & CLASSIFYING CANDIDATE MISSES (0.9579 CHAMPION)")
    print("=" * 80)
    t0 = time.time()
    paths = get_data_paths(is_sample=False)
    
    # 1. Load Ground Truth
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())
    s1_needed = set(gt_dict.keys())
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"): needed_s2.add(m)
            elif m.startswith("S3-"): needed_s3.add(m)
            
    print(f"Ground Truth loaded: {len(gt_dict):,d} S1 entities, {total_true:,d} total true matches.", flush=True)
    
    # 2. Load Records
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

    print("Loading candidate pools...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3)
    
    # 3. Precompute entities
    print("Precomputing representations...", flush=True)
    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)
    
    # 4. Build inverted indexes
    print("Building champion inverted indexes...", flush=True)
    idx_s2 = defaultdict(list)
    for eid, row in s2_dict.items():
        ordered_passes = extract_dynamic_multipass_keys(row["business_name"], row["business_address"], row.get("country", ""))
        for pass_keys in ordered_passes:
            for k in pass_keys: idx_s2[k].append(eid)
            
    idx_s3 = defaultdict(list)
    for eid, row in s3_dict.items():
        ordered_passes = extract_dynamic_multipass_keys(row["business_name"], row["business_address"], row.get("country", ""))
        for pass_keys in ordered_passes:
            for k in pass_keys: idx_s3[k].append(eid)
            
    # 5. Extract candidates and find misses
    print("Running champion candidate generator...", flush=True)
    MAX_CANDS = 120
    hits = 0
    missed_pairs = []
    
    for s1_id, s1_row in s1_dict.items():
        true_matches = gt_dict.get(s1_id, set())
        ordered_passes = extract_dynamic_multipass_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
        
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
            
        found = true_matches & cands
        hits += len(found)
        missed = true_matches - cands
        
        for tgt_id in missed:
            tgt_row = s2_dict.get(tgt_id) or s3_dict.get(tgt_id)
            if tgt_row is None:
                continue
            e1 = cached_s1[s1_id]
            e2 = cached_cands[tgt_id]
            category, reason = classify_miss(e1, e2)
            src = "S2" if tgt_id.startswith("S2-") else "S3"
            missed_pairs.append({
                "s1_id": s1_id,
                "source": src,
                "true_entity_id": tgt_id,
                "business_name_s1": s1_row["business_name"],
                "business_name_target": tgt_row["business_name"],
                "address_s1": s1_row["business_address"],
                "address_target": tgt_row["business_address"],
                "country_s1": s1_row.get("country", ""),
                "country_target": tgt_row.get("country", ""),
                "miss_category": category,
                "miss_reason_metadata": reason
            })
            
    cand_recall = hits / total_true
    print(f"\nChampion Candidate Recall: {cand_recall*100:.2f}% ({hits:,d} hits / {total_true:,d} total true matches)")
    print(f"Total True Misses: {len(missed_pairs):,d} missed true pairs.")
    
    # 6. Save missed pairs TSV
    out_dir = os.path.join(REPO_ROOT, "experiments", "semantic_candidate_recovery", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "missed_true_pairs.tsv")
    
    df_miss = pd.DataFrame(missed_pairs)
    df_miss.to_csv(out_path, sep="\t", index=False)
    print(f"Saved missed true pairs to: {out_path}", flush=True)
    
    # 7. Print Taxonomy Summary
    print("\n" + "=" * 80)
    print("EMPIRICAL CANDIDATE MISS TAXONOMY (0.9579 CHAMPION)")
    print("=" * 80)
    cat_counts = Counter(df_miss["miss_category"])
    for cat, count in cat_counts.most_common():
        pct = (count / len(df_miss)) * 100
        print(f"  {cat:<35} : {count:>5,d} ({pct:>5.1f}%)")
        
    print("\n--- SAMPLE MISSES FROM TOP CATEGORIES ---")
    for cat, count in cat_counts.most_common(5):
        print(f"\n[Category: {cat} (Total: {count})]")
        samples = df_miss[df_miss["miss_category"] == cat].head(3)
        for _, r in samples.iterrows():
            print(f"  S1 ({r['s1_id']}): '{r['business_name_s1']}' | '{r['address_s1']}' [{r['country_s1']}]")
            print(f"  Tgt({r['true_entity_id']}): '{r['business_name_target']}' | '{r['address_target']}' [{r['country_target']}]")
            print(f"  Reason: {r['miss_reason_metadata']}")
            print("  -")

    # 8. Forensic Evaluation of Recoverability
    print("\n" + "=" * 80)
    print("FORENSIC RECOVERABILITY ANALYSIS (RESEARCH QUESTION)")
    print("=" * 80)
    name_sims = []
    addr_sims = []
    both_low = 0
    recoverable_semantic = 0
    for _, r in df_miss.iterrows():
        ns = fuzz.token_sort_ratio(str(r["business_name_s1"]), str(r["business_name_target"]))
        as_ = fuzz.token_sort_ratio(str(r["address_s1"]), str(r["address_target"]))
        name_sims.append(ns)
        addr_sims.append(as_)
        if ns < 30 and as_ < 30:
            both_low += 1
        if (ns >= 50 or as_ >= 50) or (ns >= 35 and as_ >= 35):
            recoverable_semantic += 1
            
    print(f"Total Misses: {len(df_miss):,d}")
    print(f"  Mean Name TokenSortRatio:    {np.mean(name_sims):.1f}")
    print(f"  Mean Address TokenSortRatio: {np.mean(addr_sims):.1f}")
    print(f"  Completely Disjoint (<30 name & <30 addr): {both_low:,d} ({both_low/len(df_miss)*100:.1f}%) -> Zero textual evidence")
    print(f"  Potential Semantic/Lexical Signal (ns>=50 or as>=50): {recoverable_semantic:,d} ({recoverable_semantic/len(df_miss)*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
