"""
Test Indic Script Transliteration & Address Blocking Recovery
Amazon ML Challenge 2026 - Business Entity Resolution
"""
import os, sys, time, pickle, re
from collections import defaultdict
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PHASE8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(PHASE8_DIR, "artifacts")
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from transliteration import transliterate_indic, has_indic_script

from data_loader import get_data_paths, parse_ground_truth_fast
from normalization import normalize_name, normalize_address, extract_clean_numbers

def clean_indic_text(text):
    """
    Cleans segmented Indic characters by joining isolated single letters,
    then transliterates to Latin.
    """
    if not has_indic_script(text):
        return text
    # Remove artificial spaces between single Indic characters
    # e.g. "ह ट ल" -> "हटल", "ए स ए स" -> "एसएस"
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
    if curr_seg:
        joined_words.append("".join(curr_seg))
    cleaned = " ".join(joined_words)
    return transliterate_indic(cleaned)

def main():
    print("Testing clean_indic_text on missed candidates...")
    test_cases = [
        "एसएस फ ड पर इव ट ल म ट ड",
        "ह टल ए टरपर इज ज ल म ट ड",
        "र म म ल ज सट कस पर इव ट ल म ट ड",
        "स वन सन पर डय सर पर इव ट ल म ट ड",
        "ডর ম কনसটর কশন पर ইভ ট ল म ট ড",
        "পর ম यर फ उ ड शन ल म ट ड",
        "र यल स रय म न जम ट"
    ]
    for tc in test_cases:
        print(f"RAW: [{tc}] -> TRANSLIT: [{clean_indic_text(tc)}]")

    # Check recovery on missed GT pairs
    print("\nLoading missed GT pairs...")
    df_pairs = pd.read_feather(os.path.join(ART_DIR, "chal_pairs.feather"))
    existing_pairs = set(zip(df_pairs["s1_id"], df_pairs["cand_id"]))

    paths = get_data_paths(is_sample=False)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)

    with open(os.path.join(ART_DIR, "cached_s1.pkl"), "rb") as f: cached_s1 = pickle.load(f)
    with open(os.path.join(ART_DIR, "cached_cands.pkl"), "rb") as f: cached_cands = pickle.load(f)

    missed_gt = []
    for sid, matches in gt_dict.items():
        for cid in matches:
            if (sid, cid) not in existing_pairs:
                missed_gt.append((sid, cid))

    print(f"Total Missed GT pairs: {len(missed_gt)}")

    # Check address number overlap among missed pairs
    num_match_count = 0
    indic_script_count = 0
    both_count = 0

    for sid, cid in missed_gt:
        e1 = cached_s1.get(sid, {})
        e2 = cached_cands.get(cid, {})
        if not e2: continue
        raw_n2 = e2.get("raw_n", "")
        nums1 = e1.get("nums", set())
        nums2 = e2.get("nums", set())
        has_num = len(nums1 & nums2) > 0
        has_indic = has_indic_script(raw_n2)

        if has_num: num_match_count += 1
        if has_indic: indic_script_count += 1
        if has_num or has_indic: both_count += 1

    print(f"Missed pairs with matching address numbers: {num_match_count} ({num_match_count/len(missed_gt)*100:.1f}%)")
    print(f"Missed pairs with Indic script names:       {indic_script_count} ({indic_script_count/len(missed_gt)*100:.1f}%)")
    print(f"Missed pairs recoverable via Num or Indic:  {both_count} ({both_count/len(missed_gt)*100:.1f}%)")

if __name__ == "__main__":
    import pandas as pd
    main()
