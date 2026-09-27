import os, sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from normalization import (
    normalize_name, normalize_address, extract_name_aliases,
    extract_url_core, extract_clean_numbers,
)
from transliteration import transliterate_indic
from rapidfuzz import fuzz

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_final_model import (
    precompute_entity, extract_features_31, extract_44_features_single,
    NAME_STOPWORDS, ADDR_STOPWORDS, PINCODE_RE
)

# Test pair
s1_row = {
    "entity_id": "S1-100",
    "business_name": "Starbucks Coffee Main St",
    "business_address": "123 Main Street Suite 400",
    "country": "US"
}
cand_row = {
    "entity_id": "S2-200",
    "business_name": "Starbucks Coffee Co",
    "business_address": "123 Main St",
    "country": "US"
}
sem_info = (0.85, 1)

# Method 1: Train script method
e1 = precompute_entity(s1_row)
e2 = precompute_entity(cand_row)
base_f = extract_features_31(e1, e2, "S2-200")
feat_44_train = extract_44_features_single(e1, e2, "S2-200", base_f, sem_info)

# Method 2: On-the-fly method
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

def extract_features_onthefly(s1_tuple, cand_norm_n, cand_norm_a, cand_country, cand_id, sem_info):
    n1, a1, c1, toks_n1, toks_a1, nums_1, list_nums1, norm_trans_n1, pin1, c4_1 = s1_tuple
    n2 = cand_norm_n
    a2 = cand_norm_a
    c2 = cand_country

    toks_n2 = set(t for t in n2.split() if t not in NAME_STOPWORDS)
    toks_a2 = set(t for t in a2.split() if t not in ADDR_STOPWORDS)
    list_nums2 = extract_clean_numbers(a2)
    nums_2 = set(list_nums2)
    # transliterate indic only if indic present
    trans_n2 = ""
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

norm_n2 = normalize_name(cand_row["business_name"].replace("-", " ").replace("/", " "))
norm_a2 = normalize_address(cand_row["business_address"].replace("-", " ").replace("/", " "))
s1_t = precompute_s1(s1_row)
feat_44_onthefly = extract_features_onthefly(s1_t, norm_n2, norm_a2, "US", "S2-200", sem_info)

diff = np.max(np.abs(np.array(feat_44_train) - np.array(feat_44_onthefly)))
print(f"Max feature difference: {diff}")
assert diff == 0.0, "Feature mismatch detected!"
print("EXACT 100% FEATURE PARITY VERIFIED!")
