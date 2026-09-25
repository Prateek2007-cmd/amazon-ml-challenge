"""
Properly aligned candidate blocking evaluation.
Picks 5,000 S1 entities and evaluates blocking against THEIR exact ground truth.
"""
import os, sys, time, re
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
from collections import defaultdict
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from normalize import normalize_name, normalize_address

# Load 5,000 GT entries
print("Loading 5,000 GT entries...")
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=5000, dtype=str, keep_default_na=False)
s1_needed = set(gt_df["source1_entity_id"])
gt_dict = {}
needed_s2, needed_s3 = set(), set()
total_true = 0

for _, row in gt_df.iterrows():
    s1_id = row["source1_entity_id"]
    matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
    gt_dict[s1_id] = set(matches)
    total_true += len(matches)
    for m in matches:
        if m.startswith("S2-"):
            needed_s2.add(m)
        elif m.startswith("S3-"):
            needed_s3.add(m)

print(f"Entities: {len(s1_needed)}, True matches: {total_true}")

# Load the exact S1 records
print("Loading exact S1 records...")
s1_records = []
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if parts[0] in s1_needed:
            s1_records.append(dict(zip(header, parts)))
            if len(s1_records) == len(s1_needed):
                break
s1_df = pd.DataFrame(s1_records)
print(f"Loaded {len(s1_df)} S1 records.")

# Load candidate pool: 100k rows from S2 and S3 + all needed true targets
def load_pool(path, needed_set, pool_size=100000):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            eid = parts[0]
            if eid in needed_set or i < pool_size:
                records.append(dict(zip(header, parts)))
    return pd.DataFrame(records)

print("Loading S2 and S3 pool...")
s2_pool = load_pool("dataset/train/train_source2.tsv", needed_s2)
s3_pool = load_pool("dataset/train/train_source3.tsv", needed_s3)
print(f"Pool size: S2={len(s2_pool)}, S3={len(s3_pool)}")

recall_base = 0.6916
avg_cands_base = 2432.14

# --- MULTI-STAGE COMPOSITE BLOCKING ---
NAME_STOPWORDS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp",
    "co", "company", "and", "the", "of", "in", "for", "center", "centre", "group",
    "services", "solutions", "enterprises", "associates", "holdings", "management",
    "international", "global", "industries", "systems", "technologies", "tech", "dba", "llc"
}

ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "court", "ct", "way", "street", "road",
    "avenue", "boulevard", "drive", "lane", "apt", "suite", "ste", "fl", "floor",
    "bldg", "building", "near", "opp", "opposite", "behind", "phase", "sector",
    "plot", "shop", "no", "number", "us", "usa", "india", "null"
}

URL_REGEX = re.compile(r"([a-zA-Z0-9\-]+)\.(?:com|in|org|net|co|io|biz|info|gov|edu)", re.I)
NUM_CLEAN_RE = re.compile(r"^([0-9]+)[a-zA-Z]?$")

def get_smart_keys(name, addr, country=""):
    norm_n = normalize_name(name)
    norm_a = normalize_address(addr)
    
    # Also strip '#' or '@' handles (e.g. #sjace -> sj ace)
    norm_n_clean = norm_n.replace("#", " ").replace("@", " ")
    
    n_tokens = norm_n_clean.split()
    # Allow 2-char tokens if not in stopwords
    content_n = [t for t in n_tokens if len(t) >= 2 and t not in NAME_STOPWORDS]
    
    m = URL_REGEX.search(norm_n)
    url_core = m.group(1).lower() if m else None
    
    a_tokens = norm_a.split()
    addr_nums = []
    for t in a_tokens:
        clean_num = NUM_CLEAN_RE.match(t)
        if clean_num:
            val = clean_num.group(1).lstrip("0") or "0"
            if len(val) <= 7:
                addr_nums.append(val)
                
    addr_words = [t for t in a_tokens if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS]
    
    keys = set()
    # 1. Exact normalized name
    if norm_n and len(norm_n) >= 4:
        keys.add(("exact_name", norm_n))
        
    # 2. URL core domain
    if url_core and len(url_core) >= 4:
        keys.add(("url_core", url_core))
        
    # 3. 2-token name signature (high precision)
    if len(content_n) >= 2:
        keys.add(("name_pair", tuple(sorted([content_n[0], content_n[1]]))))
        if len(content_n) >= 3:
            keys.add(("name_pair", tuple(sorted([content_n[0], content_n[2]]))))
            
    # 4. Content name tokens (first 2 if len >= 3, or if len == 2 only with another token)
    for t in content_n[:2]:
        if len(t) >= 3:
            keys.add(("name_tok", t))
        
    # 5. Composite address keys (number + street/locality word)
    if addr_nums and addr_words:
        for num in addr_nums[:2]:
            for w in addr_words[:2]:
                keys.add(("num_word", num, w))
                
    # 6. Composite name token + address number
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]:
                keys.add(("name_num", nt, num))
                
    # 7. Distinctive address word pair (for when name is transliterated / brand new and address has no number)
    if len(addr_words) >= 2:
        keys.add(("addr_pair", tuple(sorted([addr_words[0], addr_words[1]]))))
        if len(addr_words) >= 3:
            keys.add(("addr_pair", tuple(sorted([addr_words[0], addr_words[2]]))))
                
    return keys

print("\n--- BUILDING SMART COMPOSITE INVERTED INDICES ---")
t_idx = time.time()
def build_smart_index(df):
    idx = defaultdict(list)
    for _, r in df.iterrows():
        keys = get_smart_keys(r["business_name"], r["business_address"], r.get("country", ""))
        eid = r["entity_id"]
        for k in keys:
            idx[k].append(eid)
    return idx

smart_idx2 = build_smart_index(s2_pool)
smart_idx3 = build_smart_index(s3_pool)
print(f"Smart indices built in {time.time()-t_idx:.2f}s")

# Evaluate smart candidate generation with posting caps
# Name_tok cap at 200, composite keys cap at 500
t_gen = time.time()
smart_cands = {}
smart_cand_count = 0
smart_found = 0
smart_missed = []

for _, r in s1_df.iterrows():
    s1_id = r["entity_id"]
    true_set = gt_dict.get(s1_id, set())
    keys = get_smart_keys(r["business_name"], r["business_address"], r.get("country", ""))
    
    cand_set = set()
    for k in keys:
        cap = 150 if k[0] == "name_tok" else 400
        p2 = smart_idx2.get(k, [])
        p3 = smart_idx3.get(k, [])
        if len(p2) <= cap:
            cand_set.update(p2)
        if len(p3) <= cap:
            cand_set.update(p3)
            
    smart_cands[s1_id] = cand_set
    smart_cand_count += len(cand_set)
    for tm in true_set:
        if tm in cand_set:
            smart_found += 1
        else:
            smart_missed.append((s1_id, tm))

smart_recall = smart_found / total_true if total_true else 0
avg_smart_cands = smart_cand_count / len(s1_df)

print(f"\n=== SMART COMPOSITE BLOCKING RESULTS ===")
print(f"Recall:               {smart_recall*100:.2f}% ({smart_found}/{total_true})")
print(f"Average Cands per S1: {avg_smart_cands:.2f}")
print(f"Missed true matches:  {len(smart_missed)}")
print(f"\n--- DIAGNOSING MISSED PAIRS ---")
s1_map = {r['entity_id']: r for r in s1_records}
s2_map = s2_pool.set_index('entity_id')
s3_map = s3_pool.set_index('entity_id')

for i, (s1_id, m_id) in enumerate(smart_missed[:6]):
    s1 = s1_map.get(s1_id, {})
    tgt = s2_map.loc[m_id].to_dict() if m_id in s2_map.index else s3_map.loc[m_id].to_dict()
    print(f"Missed #{i+1}: {s1_id} -> {m_id}")
    print(f"  S1:  Name: '{s1.get('business_name')}', Addr: '{s1.get('business_address')}'")
    print(f"  TGT: Name: '{tgt.get('business_name')}', Addr: '{tgt.get('business_address')}'")

