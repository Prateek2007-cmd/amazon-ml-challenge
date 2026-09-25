"""
Comprehensive Feature Engineering, Model Training, Cross-Validation,
and F0.5 Threshold Optimization for Amazon ML Challenge 2026.
"""
import os
import sys
import time
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address, name_tokens, address_numeric_tokens
from evaluate import macro_f_beta

NAME_STOPWORDS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp",
    "co", "company", "and", "the", "of", "in", "for", "center", "centre", "group",
    "services", "solutions", "enterprises", "associates", "holdings", "management",
    "international", "global", "industries", "systems", "technologies", "tech", "dba"
}

ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "court", "ct", "way", "street", "road",
    "avenue", "boulevard", "drive", "lane", "apt", "suite", "ste", "fl", "floor",
    "bldg", "building", "near", "opp", "opposite", "behind", "phase", "sector",
    "plot", "shop", "no", "number", "us", "usa", "india", "null"
}

URL_REGEX = re.compile(r"([a-zA-Z0-9\-]+)\.(?:com|in|org|net|co|io|biz|info|gov|edu)", re.I)
NUM_CLEAN_RE = re.compile(r"^([0-9]+)[a-zA-Z]?$")
ALIAS_SPLIT_RE = re.compile(r"\b(?:aka|dba|d/b/a|formerly|f/k/a|fka)\b", re.I)

def extract_name_aliases(raw_name):
    """Splits 'Parent dba Subsidiary' or 'New formerly Old'."""
    parts = ALIAS_SPLIT_RE.split(str(raw_name))
    return [p.strip() for p in parts if p.strip()]

def get_smart_keys(name, addr, country=""):
    norm_n = normalize_name(name)
    norm_a = normalize_address(addr)
    
    norm_n_clean = norm_n.replace("#", " ").replace("@", " ")
    n_tokens = norm_n_clean.split()
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
    if norm_n and len(norm_n) >= 4:
        keys.add(("exact_name", norm_n))
        
    if url_core and len(url_core) >= 4:
        keys.add(("url_core", url_core))
        
    if len(content_n) >= 2:
        keys.add(("name_pair", tuple(sorted([content_n[0], content_n[1]]))))
        if len(content_n) >= 3:
            keys.add(("name_pair", tuple(sorted([content_n[0], content_n[2]]))))
            
    for t in content_n[:2]:
        if len(t) >= 3:
            keys.add(("name_tok", t))
        
    if addr_nums and addr_words:
        for num in addr_nums[:2]:
            for w in addr_words[:2]:
                keys.add(("num_word", num, w))
                
    if content_n and addr_nums:
        for nt in content_n[:2]:
            for num in addr_nums[:2]:
                keys.add(("name_num", nt, num))
                
    if len(addr_words) >= 2:
        keys.add(("addr_pair", tuple(sorted([addr_words[0], addr_words[1]]))))
        if len(addr_words) >= 3:
            keys.add(("addr_pair", tuple(sorted([addr_words[0], addr_words[2]]))))
            
    # Also index aliases if present
    aliases = extract_name_aliases(name)
    if len(aliases) > 1:
        for alias in aliases[1:]:
            norm_alias = normalize_name(alias)
            a_toks = [t for t in norm_alias.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            if len(a_toks) >= 2:
                keys.add(("name_pair", tuple(sorted([a_toks[0], a_toks[1]]))))
            for at in a_toks[:2]:
                keys.add(("name_tok", at))
                
    return keys

# ----------------- PAIRWISE FEATURE EXTRACTION -----------------
FEATURE_NAMES = [
    # Name features
    "name_exact",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_len_diff",
    "name_len_ratio",
    "alias_match",
    
    # Address features
    "addr_exact",
    "addr_fuzz_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_token_jaccard",
    "addr_numeric_jaccard",
    "addr_numeric_overlap_count",
    "addr_len_diff",
    "addr_is_empty_either",
    
    # Cross & Meta features
    "country_match",
    "is_source3",
    "name_addr_mult",
    "name_addr_min",
    "name_addr_max",
    "name_addr_mean",
]

def char_ngrams(s, n=3):
    return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set([s]) if s else set()

def extract_pairwise_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country, cand_id):
    norm_n1, norm_n2 = normalize_name(s1_name), normalize_name(cand_name)
    norm_a1, norm_a2 = normalize_address(s1_addr), normalize_address(cand_addr)
    
    # Name tokens
    toks_n1 = set(t for t in norm_n1.split() if t not in NAME_STOPWORDS)
    toks_n2 = set(t for t in norm_n2.split() if t not in NAME_STOPWORDS)
    name_union = toks_n1 | toks_n2
    name_inter = toks_n1 & toks_n2
    name_jaccard = len(name_inter) / len(name_union) if name_union else 0.0
    
    # RapidFuzz name similarities
    ratio_n = fuzz.ratio(norm_n1, norm_n2) / 100.0
    tsort_n = fuzz.token_sort_ratio(norm_n1, norm_n2) / 100.0
    tset_n = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0
    partial_n = fuzz.partial_ratio(norm_n1, norm_n2) / 100.0
    
    # Alias check
    aliases2 = extract_name_aliases(cand_name)
    alias_match = 0.0
    if len(aliases2) > 1:
        for a in aliases2:
            if fuzz.token_sort_ratio(norm_n1, normalize_name(a)) >= 85:
                alias_match = 1.0
                break
                
    # Address tokens
    toks_a1 = set(t for t in norm_a1.split() if t not in ADDR_STOPWORDS)
    toks_a2 = set(t for t in norm_a2.split() if t not in ADDR_STOPWORDS)
    addr_union = toks_a1 | toks_a2
    addr_inter = toks_a1 & toks_a2
    addr_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0
    
    # Address numbers
    nums_1 = set(t.lstrip("0") or "0" for t in norm_a1.split() if t.isdigit() and len(t) <= 7)
    nums_2 = set(t.lstrip("0") or "0" for t in norm_a2.split() if t.isdigit() and len(t) <= 7)
    num_union = nums_1 | nums_2
    num_inter = nums_1 & nums_2
    num_jaccard = len(num_inter) / len(num_union) if num_union else 0.0
    
    # RapidFuzz address similarities
    empty_addr = float(len(norm_a1) == 0 or len(norm_a2) == 0)
    if empty_addr:
        ratio_a = 0.0
        tsort_a = 0.0
        tset_a = 0.0
        partial_a = 0.0
    else:
        ratio_a = fuzz.ratio(norm_a1, norm_a2) / 100.0
        tsort_a = fuzz.token_sort_ratio(norm_a1, norm_a2) / 100.0
        tset_a = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0
        partial_a = fuzz.partial_ratio(norm_a1, norm_a2) / 100.0
        
    c1 = str(s1_country).strip().upper()
    c2 = str(cand_country).strip().upper()
    country_match = float(c1 == c2 and c1 != "")
    is_s3 = float(str(cand_id).startswith("S3-"))
    
    len_diff_n = abs(len(norm_n1) - len(norm_n2))
    max_len_n = max(len(norm_n1), len(norm_n2), 1)
    len_ratio_n = 1.0 - (len_diff_n / max_len_n)
    
    len_diff_a = abs(len(norm_a1) - len(norm_a2))
    
    # Interaction terms
    mult = tsort_n * (tsort_a if not empty_addr else tsort_n)
    min_sim = min(tsort_n, tsort_a) if not empty_addr else tsort_n
    max_sim = max(tsort_n, tsort_a)
    mean_sim = (tsort_n + tsort_a) / 2.0 if not empty_addr else tsort_n
    
    return [
        float(norm_n1 == norm_n2 and len(norm_n1) > 0),
        ratio_n,
        tsort_n,
        tset_n,
        partial_n,
        name_jaccard,
        float(len(name_inter)),
        float(len_diff_n),
        len_ratio_n,
        alias_match,
        
        float(norm_a1 == norm_a2 and len(norm_a1) > 0),
        ratio_a,
        tsort_a,
        tset_a,
        partial_a,
        addr_jaccard,
        num_jaccard,
        float(len(num_inter)),
        float(len_diff_a),
        empty_addr,
        
        country_match,
        is_s3,
        mult,
        min_sim,
        max_sim,
        mean_sim,
    ]

# ----------------- MAIN EXPERIMENT RUNNER -----------------
def run_experiments():
    print("=== COMMENCING SYSTEMATIC ML EXPERIMENTATION ===")
    t0 = time.time()
    
    # Load 5,000 S1 validation set
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
                
    # Load S1 records
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
    
    # Load pool
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
        
    s2_pool = load_pool("dataset/train/train_source2.tsv", needed_s2)
    s3_pool = load_pool("dataset/train/train_source3.tsv", needed_s3)
    
    # Build Inverted index
    print("Building composite inverted index...")
    def build_idx(df):
        idx = defaultdict(list)
        for _, r in df.iterrows():
            keys = get_smart_keys(r["business_name"], r["business_address"], r.get("country", ""))
            eid = r["entity_id"]
            for k in keys:
                idx[k].append(eid)
        return idx
        
    idx2 = build_idx(s2_pool)
    idx3 = build_idx(s3_pool)
    
    # Build candidate pairs
    print("Generating candidate pairs...")
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    s2_dict = s2_pool.set_index("entity_id").to_dict("index")
    s3_dict = s3_pool.set_index("entity_id").to_dict("index")
    
    pairs = []
    features_list = []
    labels = []
    groups = []
    
    cand_recall_hits = 0
    t_feat_start = time.time()
    
    for s1_id, s1_row in s1_dict.items():
        true_matches = gt_dict.get(s1_id, set())
        keys = get_smart_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
        
        cand_set = set()
        for k in keys:
            cap = 150 if k[0] == "name_tok" else 400
            p2 = idx2.get(k, [])
            p3 = idx3.get(k, [])
            if len(p2) <= cap:
                cand_set.update(p2)
            if len(p3) <= cap:
                cand_set.update(p3)
                
        for tm in true_matches:
            if tm in cand_set:
                cand_recall_hits += 1
                
        for cand_id in cand_set:
            cand_row = s2_dict.get(cand_id) or s3_dict.get(cand_id)
            if cand_row is None:
                continue
            feats = extract_pairwise_features(
                s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""),
                cand_row["business_name"], cand_row["business_address"], cand_row.get("country", ""),
                cand_id
            )
            is_match = 1 if cand_id in true_matches else 0
            pairs.append((s1_id, cand_id))
            features_list.append(feats)
            labels.append(is_match)
            groups.append(s1_id)
            
    print(f"Generated {len(pairs):,d} candidate pairs ({cand_recall_hits}/{total_true} true matches = {cand_recall_hits/total_true*100:.2f}% blocking recall)")
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"Positives: {pos_count:,d} | Negatives (Hard Blocking Negatives): {neg_count:,d} (Ratio 1:{neg_count/pos_count:.1f})")
    print(f"Features extracted in {time.time()-t_feat_start:.2f}s")
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(groups)
    
    # ----------------- ENTITY-LEVEL 5-FOLD CROSS VALIDATION -----------------
    print("\n=== RUNNING 5-FOLD ENTITY-LEVEL CROSS-VALIDATION ===")
    gkf = GroupKFold(n_splits=5)
    
    models = {
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_depth=7, learning_rate=0.08, max_iter=150, random_state=42),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, max_depth=7, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    }
    
    exp_results = []
    
    for m_name, m_builder in models.items():
        print(f"\n>>> Evaluating Model: {m_name} across 5 entity-level folds...")
        t_model_start = time.time()
        
        oof_probs = np.zeros(len(y), dtype=np.float32)
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            clf = m_builder()
            clf.fit(X_train, y_train)
            val_probs = clf.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_probs
            
        print(f"  Training 5 folds complete in {time.time()-t_model_start:.2f}s")
        
        # Fine-grained threshold sweep directly on OOF predictions for F0.5
        best_t, best_f05 = 0.5, -1.0
        best_prec, best_rec = 0.0, 0.0
        best_tp, best_fp, best_fn = 0, 0, 0
        
        thresholds = np.arange(0.20, 0.96, 0.02)
        for t in thresholds:
            t = round(float(t), 2)
            preds = defaultdict(set)
            for (s1_id, cand_id), p in zip(pairs, oof_probs):
                if p >= t:
                    preds[s1_id].add(cand_id)
                    
            # Ensure every S1 in ground truth is present
            for s1_id in gt_dict:
                if s1_id not in preds:
                    preds[s1_id] = set()
                    
            # Calculate macro F0.5
            f05 = macro_f_beta(gt_dict, preds, beta=0.5)
            
            # Global TP, FP, FN
            tp, fp, fn = 0, 0, 0
            for s1_id, true_set in gt_dict.items():
                p_set = preds.get(s1_id, set())
                tp += len(true_set & p_set)
                fp += len(p_set - true_set)
                fn += len(true_set - p_set)
                
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if f05 > best_f05:
                best_f05 = f05
                best_t = t
                best_prec = prec
                best_rec = rec
                best_tp, best_fp, best_fn = tp, fp, fn
                
        print(f"  [RESULT] Best Threshold: {best_t:.2f} | OOF Macro F0.5: {best_f05:.4f} | Precision: {best_prec:.4f} | Recall: {best_rec:.4f} (TP={best_tp}, FP={best_fp}, FN={best_fn})")
        exp_results.append({
            "model": m_name,
            "best_threshold": best_t,
            "oof_macro_f05": best_f05,
            "precision": best_prec,
            "recall": best_rec,
            "tp": best_tp,
            "fp": best_fp,
            "fn": best_fn,
            "features_count": len(FEATURE_NAMES)
        })
        
    res_df = pd.DataFrame(exp_results)
    exp_csv_path = os.path.join(REPO_ROOT, "experiments", "results.csv")
    res_df.to_csv(exp_csv_path, index=False)
    print(f"\n=== EXPERIMENT SUMMARY SAVED TO {exp_csv_path} ===")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_experiments()
