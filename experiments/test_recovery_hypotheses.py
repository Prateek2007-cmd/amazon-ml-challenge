import sys, os
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")

repo_root = os.path.abspath(".")
sys.path.insert(0, os.path.join(repo_root, "code", "business_entity_resolution", "src"))
from normalization import normalize_name, normalize_address, extract_clean_numbers
from transliteration import transliterate_indic

# Test 1: Saffron Financers LLP vs saffronfinancers.com
s1_n = "Saffron Financers LLP"
c_n = "saffronfinancers.com"
norm_s1 = normalize_name(s1_n)
toks_s1 = [t for t in norm_s1.split() if t not in ("llp", "pvt", "ltd")]
concat_s1 = "".join(toks_s1)
print(f"S1: {s1_n} -> norm: {norm_s1} -> concat: {concat_s1}")
print(f"Cand: {c_n} -> core: {c_n.split('.')[0]}")
print(f"Match? {concat_s1 == c_n.split('.')[0]}")

# Test 2: Great Impex Private Limited vs గ్రేట్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్
s1_n2 = "Great Impex Private Limited"
c_n2 = "గ్రేట్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్"
trans_c2 = transliterate_indic(c_n2)
print(f"\nS1: {s1_n2}")
print(f"Cand: {c_n2} -> Trans: {trans_c2}")
# Replace 'x' with 'ks' in S1
norm_s1_ks = normalize_name(s1_n2).replace("x", "ks")
print(f"S1 with x->ks: {norm_s1_ks}")
print(f"Fuzz similarity: {fuzz.token_sort_ratio(norm_s1_ks, trans_c2)}%")

# Test 3: Address word sets across full address
s1_a = "6-2-101/5/C, Telangana, Hyderabad, Secunderabad, Lane Beside Centralview Apt New Bhoiguda"
c_a = "6-2-101/5/C, Lane Beside Centralview Apt New Bhoiguda, Secunderabad, Hyderabad, TG"

def get_distinctive_addr_words(addr):
    norm = normalize_address(addr.replace("-", " ").replace("/", " "))
    STOP = {"st", "rd", "ave", "blvd", "dr", "ln", "court", "ct", "way", "street", "road",
            "avenue", "boulevard", "drive", "lane", "apt", "suite", "ste", "fl", "floor",
            "bldg", "building", "near", "opp", "opposite", "behind", "phase", "sector",
            "plot", "shop", "no", "number", "us", "usa", "india", "null", "ground", "first",
            "second", "third", "c/o", "flats", "dist", "district", "state"}
    words = [t for t in norm.split() if len(t) >= 4 and not t.isdigit() and t not in STOP and not any(c.isdigit() for c in t)]
    return list(dict.fromkeys(words))

w1 = get_distinctive_addr_words(s1_a)
w2 = get_distinctive_addr_words(c_a)
print(f"\nS1 distinctive addr words: {w1}")
print(f"Cand distinctive addr words: {w2}")
print(f"Common addr words: {set(w1) & set(w2)}")
