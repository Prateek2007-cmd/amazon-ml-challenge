"""
Phase 8: Comprehensive India Recovery Test (Address Keys + Clean Indic Transliteration)
Amazon ML Challenge 2026 - Business Entity Resolution
"""
import os, sys, time, pickle, re
from collections import defaultdict
import pandas as pd

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PHASE8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(PHASE8_DIR, "artifacts")
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import normalize_name, normalize_address, extract_clean_numbers
from transliteration import transliterate_indic, has_indic_script

PINCODE_RE = re.compile(r"\b(\d{6})\b")
NAME_STOPWORDS = {"pvt", "ltd", "limited", "private", "inc", "llp", "services", "enterprises"}

def clean_indic_text(text):
    if not has_indic_script(text): return text
    words = text.split()
    joined_words = []
    curr_seg = []
    for w in words:
        if len(w) == 1 and has_indic_script(w):
            curr_seg.append(w)
        else:
            if curr_seg:
                joined_words.append("".join(curr_seg))
                curr_seg = []
            joined_words.append(w)
    if curr_seg: joined_words.append("".join(curr_seg))
    return transliterate_indic(" ".join(joined_words))

def extract_comprehensive_india_keys(e):
    keys = []
    norm_a = e.get("norm_a", "")
    raw_a = e.get("raw_a", "")
    raw_n = e.get("raw_n", "")
    norm_n = e.get("norm_n", "")

    nums = extract_clean_numbers(norm_a)
    pin_m = PINCODE_RE.search(raw_a)
    pin = pin_m.group(1) if pin_m else ""

    # 1. PIN + Number
    if pin and nums:
        for n in nums[:2]: keys.append(("in_pin_num", pin, n))

    # 2. Number pair
    if nums and len(nums) >= 2:
        keys.append(("in_num_pair", nums[0], nums[1]))

    # 3. Clean Indic Transliteration tokens
    if has_indic_script(raw_n):
        trans = clean_indic_text(raw_n)
        norm_t = normalize_name(trans)
        t_tokens = [t for t in norm_t.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
        for t in t_tokens[:2]:
            keys.append(("indic_name_tok", t))
            if nums:
                for n in nums[:2]: keys.append(("indic_name_num", t, n))

    # 4. English S1 matching indic transliteration tokens
    if not has_indic_script(raw_n):
        s1_tokens = [t for t in norm_n.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
        for t in s1_tokens[:2]:
            keys.append(("indic_name_tok", t))
            if nums:
                for n in nums[:2]: keys.append(("indic_name_num", t, n))

    return keys

def main():
    print("Testing Comprehensive India Recovery...", flush=True)
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    existing_pairs = set(zip(df_pairs["s1_id"], df_pairs["cand_id"]))

    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())

    with open(os.path.join(ART_DIR, "cached_s1.pkl"), "rb") as f: cached_s1 = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_cands.pkl"), "rb") as f: cached_cands = pickle.load(f)

    missed_gt = []
    for sid, matches in gt_dict.items():
        for cid in matches:
            if (sid, cid) not in existing_pairs: missed_gt.append((sid, cid))

    india_misses = [(s, c) for s, c in missed_gt if cached_s1.get(s, {}).get("country") == "INDIA"]
    print(f"Total Missed GT pairs: {len(missed_gt)} (India: {len(india_misses)})")

    # Index candidate keys
    idx = defaultdict(list)
    for cid, e in cached_cands.items():
        if e.get("country") == "INDIA":
            for k in extract_comprehensive_india_keys(e):
                idx[k].append(cid)

    # Test retrieval on India misses
    recovered = 0
    total_cands_added = 0
    for sid, cid in india_misses:
        e1 = cached_s1.get(sid, {})
        keys1 = extract_comprehensive_india_keys(e1)
        found = False
        retrieved_set = set()
        for k in keys1:
            pool = idx.get(k, [])
            if len(pool) <= 80:  # reasonable bucket cap
                retrieved_set.update(pool)
                if cid in pool: found = True
        if found: recovered += 1
        total_cands_added += len(retrieved_set)

    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE INDIA RECOVERY RESULTS:")
    print("=" * 80)
    print(f"  India Misses Recovered: {recovered:,d} / {len(india_misses):,d} ({recovered/len(india_misses)*100:.1f}%)")
    print(f"  Average Cands per Entity: {total_cands_added / max(len(india_misses), 1):.1f}")
    new_cand_recall = (total_true - len(missed_gt) + recovered) / total_true
    print(f"  Projected Candidate Recall: {new_cand_recall*100:.2f}% (Up from 98.63%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
