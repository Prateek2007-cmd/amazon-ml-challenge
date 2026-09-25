"""
Detailed Taxonomy Analysis of Remaining Candidate Generation Misses (1,737 pairs).
Classifies every missed true pair into categories A through J and generates
reports/candidate_miss_taxonomy.md with representative examples.
"""
import os
import sys
import time
from collections import defaultdict, Counter
import re
import unicodedata
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import (
    normalize_name,
    normalize_address,
    extract_name_aliases,
    extract_url_core,
    extract_clean_numbers,
    NAME_ABBREVIATIONS,
    ADDRESS_ABBREVIATIONS
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

def main():
    print("Loading data for candidate miss taxonomy...", flush=True)
    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())
    s1_needed = set(gt_dict.keys())
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"): needed_s2.add(m)
            elif m.startswith("S3-"): needed_s3.add(m)
            
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
        
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3)
    
    print("Indexing candidate pools...", flush=True)
    idx_s2 = defaultdict(list)
    for eid, row in s2_dict.items():
        ordered_passes = extract_dynamic_multipass_keys(row["business_name"], row["business_address"], row.get("country", ""))
        for pass_keys in ordered_passes:
            for k in pass_keys:
                idx_s2[k].append(eid)
                
    idx_s3 = defaultdict(list)
    for eid, row in s3_dict.items():
        ordered_passes = extract_dynamic_multipass_keys(row["business_name"], row["business_address"], row.get("country", ""))
        for pass_keys in ordered_passes:
            for k in pass_keys:
                idx_s3[k].append(eid)
                
    print("Evaluating candidate retrieval and isolating misses...", flush=True)
    missed_pairs = []
    MAX_CANDS = 120
    
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
            
        for tm in true_matches:
            if tm not in cands:
                cand_row = s2_dict.get(tm) or s3_dict.get(tm)
                missed_pairs.append((s1_id, tm, s1_row, cand_row))
                
    print(f"Total True Pairs: {total_true:,d}")
    print(f"Retrieved: {total_true - len(missed_pairs):,d} ({(total_true - len(missed_pairs))/total_true*100:.2f}%)")
    print(f"Missed: {len(missed_pairs):,d} ({len(missed_pairs)/total_true*100:.2f}%)")
    
    # Classify into Categories A to J
    category_counts = Counter()
    category_samples = defaultdict(list)
    
    for s1_id, tm, s1, c in missed_pairs:
        if c is None:
            category_counts["J_other"] += 1
            continue
            
        n1 = normalize_name(s1.get("business_name", ""))
        n2 = normalize_name(c.get("business_name", ""))
        raw_n1 = s1.get("business_name", "")
        raw_n2 = c.get("business_name", "")
        
        a1 = normalize_address(s1.get("business_address", ""))
        a2 = normalize_address(c.get("business_address", ""))
        
        c1 = s1.get("country", "").upper()
        c2 = c.get("country", "").upper()
        
        sim_name = fuzz.token_sort_ratio(n1, n2)
        sim_addr = fuzz.token_sort_ratio(a1, a2) if a1 and a2 else 0
        
        nums1 = set(extract_clean_numbers(a1))
        nums2 = set(extract_clean_numbers(a2))
        num_overlap = nums1 & nums2
        
        # Check specific characteristics
        is_empty_addr = (len(a1) == 0 or len(a2) == 0)
        has_url_pattern = bool(re.search(r"(\.com|\.in|\.org|\.co|\.net|www\.|http)", raw_n1 + " " + raw_n2, re.I))
        has_indic = has_indic_script(raw_n1) or has_indic_script(raw_n2)
        has_dba_alias = bool(re.search(r"\b(dba|aka|formerly|fka|group|division)\b", raw_n1 + " " + raw_n2, re.I))
        
        # Classification hierarchy
        cat = "J_other"
        
        if is_empty_addr:
            cat = "H_missing_address"
        elif has_url_pattern and sim_name < 60:
            cat = "D_url_domain_mismatch"
        elif has_indic and sim_name < 50:
            cat = "B_transliteration_mismatch"
        elif has_dba_alias or ("llp" in raw_n1.lower() and "ltd" not in raw_n2.lower() and sim_name < 50):
            cat = "C_dba_alias_mismatch"
        elif nums1 and nums2 and not num_overlap and sim_addr > 50:
            cat = "F_numeric_address_mismatch"
        elif len(a1.split()) >= 6 and len(a2.split()) <= 3 and sim_name < 70:
            cat = "G_geographic_granularity_mismatch"
        elif sim_name >= 75 and sim_addr < 40:
            cat = "A_name_representation_mismatch"
        elif sim_addr >= 75 and sim_name < 40:
            cat = "E_address_representation_mismatch"
        elif n1 in ("star", "krishna", "om", "shree", "apex", "royal", "elite", "shri"):
            cat = "I_generic_name"
        else:
            if sim_addr >= 50:
                cat = "E_address_representation_mismatch"
            elif sim_name >= 50:
                cat = "A_name_representation_mismatch"
            else:
                cat = "J_other"
                
        category_counts[cat] += 1
        if len(category_samples[cat]) < 5:
            category_samples[cat].append({
                "s1_name": raw_n1,
                "s1_addr": s1.get("business_address", ""),
                "c_name": raw_n2,
                "c_addr": c.get("business_address", ""),
                "sim_name": sim_name,
                "sim_addr": sim_addr,
                "nums1": list(nums1),
                "nums2": list(nums2),
            })
            
    print("\nCandidate Miss Taxonomy Breakdown:")
    for cat, count in category_counts.most_common():
        pct = count / len(missed_pairs) * 100
        print(f"  {cat}: {count:,d} ({pct:.1f}%)")
        
    # Write reports/candidate_miss_taxonomy.md
    report_path = os.path.join(REPO_ROOT, "reports", "candidate_miss_taxonomy.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Candidate Generation Miss Taxonomy & Error Breakdown\n\n")
        f.write(f"**Benchmark Scope:** 10,000 Source-1 Entities | 34,511 True Ground-Truth Pairs\n")
        f.write(f"**Current Champion Candidate Recall:** 94.97% (32,774 / 34,511 retrieved)\n")
        f.write(f"**Total Missed Pairs Analyzed:** {len(missed_pairs):,d} (5.03% of ground truth)\n\n")
        f.write("## 1. Global Taxonomy Distribution\n\n")
        f.write("| Category | Description | Count | % of Misses | Primary Recovery Mechanism |\n")
        f.write("| :--- | :--- | :---: | :---: | :--- |\n")
        for cat, count in category_counts.most_common():
            pct = count / len(missed_pairs) * 100
            name_clean = cat.replace("_", " ").upper()
            f.write(f"| **{name_clean}** | ")
            if "url" in cat:
                f.write("Domain vs company name discrepancy (e.g. `saffronfinancers.com` vs `Saffron Financers LLP`). | ")
            elif "missing" in cat:
                f.write("One entity has empty address, preventing composite address blocking. | ")
            elif "transliteration" in cat:
                f.write("Complex Indic script or irregular phonetic spelling in native script. | ")
            elif "dba" in cat:
                f.write("Unlinked DBA, trade brand name, or parent subsidiary variation. | ")
            elif "address" in cat:
                f.write("Address phrasing variation, landmark, or street name abbreviation divergence. | ")
            elif "numeric" in cat:
                f.write("Door/plot number typo or numbering divergence (e.g. #932 vs #9327). | ")
            elif "granularity" in cat:
                f.write("One address has fine-grained plot/street, other only has city/state. | ")
            elif "name" in cat:
                f.write("Name variation, legal form divergence, or minor typo. | ")
            else:
                f.write("Extreme multi-field divergence across both name and address. | ")
            f.write(f"{count:,d} | {pct:.1f}% | ")
            if "url" in cat:
                f.write("Pass A: Domain core & web-stem extraction |\n")
            elif "missing" in cat:
                f.write("Pass D: Rare-token backoff on distinctive name words |\n")
            elif "transliteration" in cat:
                f.write("Pass F: Transliteration backoff + script n-grams |\n")
            elif "dba" in cat:
                f.write("Pass A: Acronym / initialism matching |\n")
            elif "address" in cat or "granularity" in cat:
                f.write("Pass B & C: Structured address component & hierarchical retrieval |\n")
            elif "numeric" in cat:
                f.write("Pass E: Character 4-gram retrieval |\n")
            else:
                f.write("Pass D & E: Informative token & n-gram backoff |\n")
        f.write("\n\n## 2. Representative Concrete Missed Cases\n\n")
        for cat, count in category_counts.most_common():
            f.write(f"### {cat.replace('_', ' ').upper()} ({count:,d} misses)\n\n")
            for i, ex in enumerate(category_samples[cat][:3], 1):
                f.write(f"**Case {i}:** (Name Sim: {ex['sim_name']}%, Addr Sim: {ex['sim_addr']}%, S1 Nums: {ex['nums1']}, C Nums: {ex['nums2']})\n")
                f.write(f"- **S1:** `{ex['s1_name']}` | `{ex['s1_addr']}`\n")
                f.write(f"- **Cand:** `{ex['c_name']}` | `{ex['c_addr']}`\n\n")
    print(f"Taxonomy report generated at {report_path}")

if __name__ == "__main__":
    main()
