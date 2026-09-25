"""
Systematic Recovery Pass Ablation Benchmark on 10,000 entities.
Evaluates incremental candidate recall, average candidates, and new true matches recovered:
- Baseline Champion (94.97%, 111.3 avg cands)
- Pass A: Domain / Concatenated core retrieval
- Pass B & C: Length-ranked address hierarchy
- Pass D: Distinctive rare name token backoff for missing address
- Pass F: Phonetic transliteration normalization (x->ks, ph->f)
Writes reports/recovery_pass_ablation.md.
"""
import os, sys, time
from collections import defaultdict, Counter
import re
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

repo_root = os.path.abspath(".")
sys.path.insert(0, os.path.join(repo_root, "code", "business_entity_resolution", "src"))
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
    "second", "third", "c/o", "flats", "dist", "district", "state"
}

def clean_phonetic(text):
    """Normalizes frequent English-Indic phonetic divergences: x->ks, ph->f."""
    if not text: return ""
    return text.replace("x", "ks").replace("ph", "f").replace("ee", "i").replace("oo", "u")

def extract_keys_with_recovery_passes(name, address, country="", enabled_passes=None):
    if enabled_passes is None:
        enabled_passes = {"base"}
        
    clean_n = name.replace("-", " ").replace("/", " ")
    norm_n = normalize_name(clean_n)
    clean_a = address.replace("-", " ").replace("/", " ")
    norm_a = normalize_address(clean_a)
    
    n_tokens = norm_n.split()
    content_n = [t for t in n_tokens if len(t) >= 2 and t not in NAME_STOPWORDS]
    url_core = extract_url_core(norm_n)
    addr_nums = extract_clean_numbers(norm_a)
    
    # Address words
    raw_addr_words = [t for t in norm_a.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and not any(c.isdigit() for c in t)]
    # Normal order
    addr_words_pos = list(dict.fromkeys(raw_addr_words))
    # Length-ranked (most distinctive words first)
    addr_words_ranked = sorted(addr_words_pos, key=lambda w: (-len(w), w))
    
    # Passes:
    # 1. Exact & High-Precision Name / URL
    p1 = []
    if norm_n and len(norm_n) >= 4: p1.append(("exact_name", norm_n))
    if url_core and len(url_core) >= 4: p1.append(("url_core", url_core))
    
    # Pass A: Domain / Concatenated Core
    if "pass_a" in enabled_passes:
        if len(content_n) >= 2:
            concat_stem = "".join(content_n[:2])
            if len(concat_stem) >= 6:
                p1.append(("url_core", concat_stem))
                
    if len(content_n) >= 2:
        p1.append(("name_pair", tuple(sorted([content_n[0], content_n[1]]))))
        if len(content_n) >= 3:
            p1.append(("name_pair", tuple(sorted([content_n[0], content_n[2]]))))
            p1.append(("name_pair", tuple(sorted([content_n[1], content_n[2]]))))
            
    # 2. Transliteration (with Pass F phonetic normalization if enabled)
    p2 = []
    trans_n = transliterate_indic(name)
    if trans_n:
        norm_tn = normalize_name(trans_n.replace("-", " ").replace("/", " "))
        if "pass_f" in enabled_passes:
            norm_tn = clean_phonetic(norm_tn)
        t_tokens = [t for t in norm_tn.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
        if len(t_tokens) >= 2:
            p2.append(("name_pair", tuple(sorted([t_tokens[0], t_tokens[1]]))))
        for t in t_tokens[:2]:
            p2.append(("name_tok", t))
    elif "pass_f" in enabled_passes and any(ch in norm_n for ch in ("x", "ph")):
        # Latin name with x or ph -> phonetic pair
        ph_norm = clean_phonetic(norm_n)
        ph_toks = [t for t in ph_norm.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
        if len(ph_toks) >= 2:
            p2.append(("name_pair", tuple(sorted([ph_toks[0], ph_toks[1]]))))

    # 3. Name token + Address Number
    p3 = []
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]:
                p3.append(("name_num", nt, num))
                
    # 4. Address Number + Address Word (Pass B: Length-ranked distinctive words)
    p4 = []
    words_for_p4 = addr_words_ranked[:3] if "pass_b" in enabled_passes else addr_words_pos[:3]
    if addr_nums and words_for_p4:
        for num in addr_nums[:3]:
            for w in words_for_p4:
                p4.append(("num_word", num, w))
                
    # 5. Distinctive Name Tokens (Pass D: Informative rare token fallback)
    p5 = []
    for t in content_n[:2]:
        if len(t) >= 4:
            p5.append(("name_tok", t))
    if "pass_d" in enabled_passes and not addr_nums and len(content_n) >= 1:
        # Entity with no address numbers -> add longest content token
        longest_t = max(content_n, key=len)
        if len(longest_t) >= 6:
            p5.append(("name_tok", longest_t))

    # 6. Address pairs (Pass C: Length-ranked address pairs)
    p6 = []
    words_for_p6 = addr_words_ranked if "pass_c" in enabled_passes else addr_words_pos
    if len(words_for_p6) >= 2:
        p6.append(("addr_pair", tuple(sorted([words_for_p6[0], words_for_p6[1]]))))
        if len(words_for_p6) >= 3:
            p6.append(("addr_pair", tuple(sorted([words_for_p6[0], words_for_p6[2]]))))

    return [p1, p2, p3, p4, p5, p6]

def main():
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

    print("Loading pools...", flush=True)
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3)

    pass_configs = [
        ("Baseline Champion", {"base"}),
        ("+ Pass A (Domain Concatenation)", {"base", "pass_a"}),
        ("+ Pass B (Distinctive Addr Words)", {"base", "pass_a", "pass_b"}),
        ("+ Pass C (Distinctive Addr Pairs)", {"base", "pass_a", "pass_b", "pass_c"}),
        ("+ Pass D (Missing Addr Backoff)", {"base", "pass_a", "pass_b", "pass_c", "pass_d"}),
        ("+ Pass F (Phonetic Transliteration)", {"base", "pass_a", "pass_b", "pass_c", "pass_d", "pass_f"}),
    ]

    results = []
    MAX_CANDS = 120

    for name, passes in pass_configs:
        t0 = time.time()
        idx_s2 = defaultdict(list)
        for eid, row in s2_dict.items():
            for pass_keys in extract_keys_with_recovery_passes(row["business_name"], row["business_address"], row.get("country", ""), passes):
                for k in pass_keys: idx_s2[k].append(eid)
                
        idx_s3 = defaultdict(list)
        for eid, row in s3_dict.items():
            for pass_keys in extract_keys_with_recovery_passes(row["business_name"], row["business_address"], row.get("country", ""), passes):
                for k in pass_keys: idx_s3[k].append(eid)

        hits = 0
        cand_counts = []
        for s1_id, s1_row in s1_dict.items():
            true_set = gt_dict.get(s1_id, set())
            ordered_passes = extract_keys_with_recovery_passes(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""), passes)
            
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
                
            cand_counts.append(len(cands))
            hits += len(true_set & cands)

        recall = hits / total_true * 100
        avg_c = float(np.mean(cand_counts))
        med_c = float(np.median(cand_counts))
        p95_c = float(np.percentile(cand_counts, 95))
        p99_c = float(np.percentile(cand_counts, 99))
        elapsed = time.time() - t0
        
        print(f"[{name}] Recall: {recall:.2f}% ({hits}/{total_true}) | Avg: {avg_c:.1f}, Med: {med_c:.1f}, P95: {p95_c:.1f}, P99: {p99_c:.1f} ({elapsed:.1f}s)", flush=True)
        results.append({
            "pass_name": name,
            "recall": recall,
            "hits": hits,
            "misses": total_true - hits,
            "avg_cands": avg_c,
            "med_cands": med_c,
            "p95": p95_c,
            "p99": p99_c,
            "runtime": elapsed
        })

    # Write reports/recovery_pass_ablation.md
    report_path = os.path.join(repo_root, "reports", "recovery_pass_ablation.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Incremental Recovery Pass Ablation Report\n\n")
        f.write(f"**Benchmark Scope:** 10,000 Source-1 Entities | 34,511 Ground-Truth True Pairs\n")
        f.write(f"**Candidate Cap Constraint:** Max Cands = 120 per Source-1 Entity\n\n")
        f.write("## 1. Step-by-Step Cumulative Ablation Results\n\n")
        f.write("| Recovery Pass Configuration | Candidate Recall | True Matches Recovered | Missed Matches | Avg Candidates / S1 | Median | P95 | P99 | Incremental Gain |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        prev_hits = 0
        for i, r in enumerate(results):
            inc = f"+{r['hits'] - prev_hits:,d} matches" if i > 0 else "-"
            f.write(f"| **{r['pass_name']}** | **{r['recall']:.2f}%** | {r['hits']:,d} | {r['misses']:,d} | {r['avg_cands']:.1f} | {r['med_cands']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | {inc} |\n")
            prev_hits = r["hits"]
            
        f.write("\n\n## 2. Pass Analysis & Architectural Rationale\n\n")
        f.write("1. **Pass A (Domain Concatenation):**\n")
        f.write("   - Automatically bridges unsegmented web domain business names (e.g. `saffronfinancers.com`) with tokenized legal entity names (`Saffron Financers LLP`).\n\n")
        f.write("2. **Pass B & C (Length-Ranked Distinctive Address Components):**\n")
        f.write("   - Sorting address words by length descending ensures unique locality and building tokens are prioritized over generic positional prefixes (`lane`, `road`, `street`).\n\n")
        f.write("3. **Pass D (Missing Address Backoff):**\n")
        f.write("   - For entities with null addresses, recovers distinctive name tokens of length >= 6 rather than dropping out of blocking.\n\n")
        f.write("4. **Pass F (Phonetic Transliteration Normalization):**\n")
        f.write("   - Normalizes regular phonetic divergences between English spellings (`x`, `ph`) and Indic phonetic representations (`ks`, `f`).\n")

    print(f"Report written to {report_path}", flush=True)

if __name__ == "__main__":
    main()
