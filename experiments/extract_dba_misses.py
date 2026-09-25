"""
Extract and analyze exact DBA / Brand / Website / Acronym candidate misses
from the 10,000-entity validation ground truth under the 0.9579 champion.
Clusters misses into concrete subtypes A through J and generates the detailed analysis table.
"""
import os, sys, time, re
from collections import defaultdict, Counter
import pandas as pd
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

def extract_champion_keys(name, address, country=""):
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
    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
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

    print("Building champion index...", flush=True)
    idx_s2 = defaultdict(list)
    for eid, row in s2_dict.items():
        for pkeys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pkeys: idx_s2[k].append(eid)
    idx_s3 = defaultdict(list)
    for eid, row in s3_dict.items():
        for pkeys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pkeys: idx_s3[k].append(eid)

    print("Isolating candidate misses...", flush=True)
    missed_pairs = []
    MAX_CANDS = 120

    for s1_id, s1_row in s1_dict.items():
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

        for tm in true_matches:
            if tm not in cands:
                cand_row = s2_dict.get(tm) or s3_dict.get(tm)
                missed_pairs.append((s1_id, tm, s1_row, cand_row))

    print(f"Total missed true pairs: {len(missed_pairs)}", flush=True)

    # Filter & cluster DBA / Brand / Website / Acronym misses
    dba_misses = []
    subtype_counts = Counter()

    for s1_id, tm, s1, c in missed_pairs:
        if c is None: continue
        raw_n1 = s1.get("business_name", "")
        raw_n2 = c.get("business_name", "")
        raw_a1 = s1.get("business_address", "")
        raw_a2 = c.get("business_address", "")
        c1 = s1.get("country", "")
        c2 = c.get("country", "")

        n1 = normalize_name(raw_n1)
        n2 = normalize_name(raw_n2)
        a1 = normalize_address(raw_a1)
        a2 = normalize_address(raw_a2)

        sim_n = fuzz.token_sort_ratio(n1, n2)
        sim_a = fuzz.token_sort_ratio(a1, a2) if a1 and a2 else 0

        # Subtype clustering:
        # A: DBA vs legal entity (e.g. DBA keywords, aka, formerly)
        # B: Brand vs parent company (short brand in one, corporate entity in other)
        # C: Website/domain variation (contains .com, .in, www., http)
        # D: Website redirect / brand-domain unsegmented (e.g. pushorizonpapercom vs PUS Horizon Paper)
        # E: Abbreviated brand / Acronym (e.g. ABC vs Asian Business Corporation)
        # F: Missing business name tokens (only 1 generic word shared)
        # G: Empty/weak address (one address empty or <10 chars)
        # H: Brand appears only in address
        # I: Legal suffix dominating
        # J: Cross-script DBA representation

        is_domain = bool(re.search(r"(\.com|\.in|\.org|\.co|\.net|www\.|http|@)", raw_n1 + " " + raw_n2, re.I))
        is_dba = bool(re.search(r"\b(dba|aka|formerly|fka|trade|trading as|operating as)\b", raw_n1 + " " + raw_n2, re.I))
        is_empty_addr = (len(a1) == 0 or len(a2) == 0 or len(a1) < 10 or len(a2) < 10)
        is_indic = has_indic_script(raw_n1) or has_indic_script(raw_n2)

        subtype = "Other_Unclustered"
        why = "Multi-field lexical variation"
        recovery_mech = "None"

        if is_domain:
            if sim_a >= 50 or (set(extract_clean_numbers(a1)) & set(extract_clean_numbers(a2))):
                subtype = "C_website_domain_variation"
                why = "Candidate name is a website/domain with unsegmented words while S1 is legal entity"
                recovery_mech = "Pass A: Strip TLD + token stem concatenation + address anchor"
            else:
                subtype = "D_website_redirect_weak_address"
                why = "Domain name present but address is missing or disparate"
                recovery_mech = "Strict domain core exact match + country agreement"
        elif is_dba:
            subtype = "A_dba_vs_legal_entity"
            why = "DBA keyword present separating trade brand from legal parent"
            recovery_mech = "Explicit DBA token splitting + alias indexing"
        elif is_indic:
            subtype = "J_cross_script_dba"
            why = "One entity in native Indic script, other in English with different brand phrasing"
            recovery_mech = "Indic transliteration + consonant skeleton + address key"
        elif is_empty_addr:
            subtype = "G_empty_or_weak_address"
            why = "Address is empty or severely truncated, preventing composite address blocking"
            recovery_mech = "Distinctive multi-word name signature fallback (min length 10)"
        elif len(n1.split()) == 1 or len(n2.split()) == 1:
            # Check acronym
            toks_long = (n1 if len(n1) > len(n2) else n2).split()
            tok_short = (n2 if len(n1) > len(n2) else n1).replace(" ", "")
            acronym = "".join(t[0] for t in toks_long if t not in NAME_STOPWORDS)
            if acronym and tok_short.lower() == acronym.lower():
                subtype = "E_abbreviated_brand_acronym"
                why = "Entity name is an acronym/initialism of the full corporate name"
                recovery_mech = "Deterministic acronym generation + postal/address anchor"
            else:
                subtype = "B_brand_vs_parent_company"
                why = "Short distinctive brand name vs multi-word parent company name"
                recovery_mech = "Single distinctive brand token + address door/plot number"
        elif sim_a >= 60 and sim_n < 50:
            subtype = "F_missing_business_name_tokens"
            why = "Address matches strongly but business names have disparate commercial branding"
            recovery_mech = "High-precision address anchor (plot number + 2 distinctive street words)"
        else:
            subtype = "I_legal_suffix_dominating"
            why = "Lexical divergence across commercial branding"
            recovery_mech = "Address component intersection"

        subtype_counts[subtype] += 1
        dba_misses.append({
            "s1_id": s1_id,
            "true_matched_source": "S2" if tm.startswith("S2-") else "S3",
            "true_entity_id": tm,
            "business_name_s1": raw_n1,
            "business_name_s2_or_s3": raw_n2,
            "address_s1": raw_a1,
            "address_s2_or_s3": raw_a2,
            "country": c1 or c2,
            "current_candidate_status": "MISSED_BY_BLOCKING",
            "subtype": subtype,
            "why_missed": why,
            "proposed_recovery_mechanism": recovery_mech
        })

    print(f"\nDBA / Brand / Website Misses Subtype Breakdown ({len(dba_misses)} total):")
    for st, count in subtype_counts.most_common():
        print(f"  {st}: {count} ({count/len(dba_misses)*100:.1f}%)")

    # Save detailed CSV and print table snippet
    df_dba = pd.DataFrame(dba_misses)
    df_dba.to_csv(os.path.join(REPO_ROOT, "experiments", "dba_misses_detailed.csv"), index=False)
    print("\nSaved detailed table to experiments/dba_misses_detailed.csv", flush=True)

if __name__ == "__main__":
    main()
