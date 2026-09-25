"""
Strict-Gated DBA / Brand / Website Recovery Experiment.
Enforces: STRONG DOMAIN/BRAND SIGNAL + INDEPENDENT SUPPORTING ADDRESS/COUNTRY SIGNAL.
Measures:
1. True pairs recovered vs new false candidates (hard negative rate)
2. Singleton & Zero-match impact
3. 5-Fold GroupKFold Macro F0.5 vs 0.9579 Champion
"""
import os, sys, time, re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from rapidfuzz import fuzz

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    ADDRESS_ABBREVIATIONS
)
from transliteration import transliterate_indic, has_indic_script
from evaluate import compute_comprehensive_metrics

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
    nums = set(extract_clean_numbers(norm_a))
    trans_n = transliterate_indic(raw_n)
    norm_trans_n = normalize_name(trans_n.replace("-", " ").replace("/", " ")) if trans_n else ""
    pin_m = PINCODE_RE.search(raw_a)
    pincode = pin_m.group(1) if pin_m else ""
    n_compact = norm_n.replace(" ", "")
    char4 = set(n_compact[i:i+4] for i in range(len(n_compact)-3)) if len(n_compact) >= 4 else set()
    
    # Domain / concatenated stems
    content_n = [t for t in norm_n.split() if len(t) >= 2 and t not in NAME_STOPWORDS]
    concat_stem = "".join(content_n[:2]) if len(content_n) >= 2 and len("".join(content_n[:2])) >= 6 else ""
    url_core = extract_url_core(norm_n)
    
    return {
        "raw_n": raw_n, "raw_a": raw_a,
        "norm_n": norm_n, "norm_a": norm_a, "country": c,
        "toks_n": toks_n, "toks_a": toks_a, "nums": nums,
        "norm_trans_n": norm_trans_n, "pincode": pincode, "char4": char4,
        "concat_stem": concat_stem, "url_core": url_core, "content_n": content_n
    }

def extract_champion_keys(e):
    norm_n, norm_a, content_n, url_core = e["norm_n"], e["norm_a"], e["content_n"], e["url_core"]
    addr_nums = list(e["nums"])
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
    if e["norm_trans_n"]:
        t_tokens = [t for t in e["norm_trans_n"].split() if len(t) >= 3 and t not in NAME_STOPWORDS]
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
    aliases = extract_name_aliases(e["raw_n"])
    if len(aliases) > 1:
        for alias in aliases[1:]:
            norm_alias = normalize_name(alias.replace("-", " ").replace("/", " "))
            a_toks = [t for t in norm_alias.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            if len(a_toks) >= 2:
                p1.append(("name_pair", tuple(sorted([a_toks[0], a_toks[1]]))))
            for at in a_toks[:2]:
                p5.append(("name_tok", at))
    return [p1, p2, p3, p4, p5, p6]

def extract_expanded_features(e1, e2, cand_id):
    n1, a1, c1 = e1["norm_n"], e1["norm_a"], e1["country"]
    n2, a2, c2 = e2["norm_n"], e2["norm_a"], e2["country"]
    toks_n1, toks_n2 = e1["toks_n"], e2["toks_n"]
    name_union = toks_n1 | toks_n2
    name_inter = toks_n1 & toks_n2
    name_jaccard = len(name_inter) / len(name_union) if name_union else 0.0
    ratio_n = fuzz.ratio(n1, n2) / 100.0
    tsort_n = fuzz.token_sort_ratio(n1, n2) / 100.0
    tset_n = fuzz.token_set_ratio(n1, n2) / 100.0
    partial_n = fuzz.partial_ratio(n1, n2) / 100.0
    toks_a1, toks_a2 = e1["toks_a"], e2["toks_a"]
    addr_union = toks_a1 | toks_a2
    addr_inter = toks_a1 & toks_a2
    addr_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0
    nums_1, nums_2 = e1["nums"], e2["nums"]
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
    if e2["norm_trans_n"]:
        trans_sim = fuzz.token_sort_ratio(n1, e2["norm_trans_n"]) / 100.0
    elif e1["norm_trans_n"]:
        trans_sim = fuzz.token_sort_ratio(e1["norm_trans_n"], n2) / 100.0
    effective_name_sim = max(tsort_n, trans_sim)
    c4_1, c4_2 = e1["char4"], e2["char4"]
    c4_union = c4_1 | c4_2
    c4_inter = c4_1 & c4_2
    char4_jaccard = len(c4_inter) / len(c4_union) if c4_union else 0.0
    has_num_match = float(len(num_inter) > 0)
    empty_addr_name_strength = (effective_name_sim * name_jaccard) if empty_addr else 0.0
    pin1, pin2 = e1["pincode"], e2["pincode"]
    pin_flag = 1.0 if (pin1 and pin2 and pin1 == pin2) else (-1.0 if (pin1 and pin2 and pin1 != pin2) else 0.0)
    return [
        float(n1 == n2 and len(n1) > 0), ratio_n, tsort_n, tset_n, partial_n,
        name_jaccard, float(len(name_inter)), float(len_diff_n), len_ratio_n, 0.0,
        float(a1 == a2 and len(a1) > 0), ratio_a, tsort_a, tset_a, partial_a,
        addr_jaccard, num_jaccard, float(len(num_inter)), float(len_diff_a), empty_addr,
        country_match, is_s3, mult, min_sim, max_sim, mean_sim,
        effective_name_sim, char4_jaccard, has_num_match, empty_addr_name_strength, pin_flag
    ]

def evaluate_subsets(gt_dict, preds, s1_dict):
    res = {}
    for subset_name, condition in [
        ("singleton", lambda sid, t_matches, r: len(t_matches) == 0),
        ("one_match", lambda sid, t_matches, r: len(t_matches) == 1),
        ("multi_match", lambda sid, t_matches, r: len(t_matches) >= 2),
        ("us", lambda sid, t_matches, r: r.get("country", "").upper() == "US"),
        ("india", lambda sid, t_matches, r: r.get("country", "").upper() == "INDIA"),
    ]:
        sub_gt = {sid: m for sid, m in gt_dict.items() if condition(sid, m, s1_dict.get(sid, {}))}
        sub_preds = {sid: preds.get(sid, set()) for sid in sub_gt}
        if sub_gt:
            m = compute_comprehensive_metrics(sub_gt, sub_preds, beta=Config.BETA)
            res[subset_name] = m["macro_f05"]
        else: res[subset_name] = 0.0
    return res

def main():
    print("=" * 80, flush=True)
    print("STRICT-GATED DBA / BRAND RECOVERY EXPERIMENT", flush=True)
    print("=" * 80, flush=True)
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

    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)

    # 1. Build Base Inverted Index
    print("Building base inverted indices...", flush=True)
    idx_s2_base = defaultdict(list)
    for eid, e in cached_cands.items():
        if eid.startswith("S2-"):
            for pkeys in extract_champion_keys(e):
                for k in pkeys: idx_s2_base[k].append(eid)
    idx_s3_base = defaultdict(list)
    for eid, e in cached_cands.items():
        if eid.startswith("S3-"):
            for pkeys in extract_champion_keys(e):
                for k in pkeys: idx_s3_base[k].append(eid)

    # 2. Build Strict DBA Inverted Index: mapping domain_core / concat_stem -> list of IDs
    idx_dba_s2 = defaultdict(list)
    idx_dba_s3 = defaultdict(list)
    for eid, e in cached_cands.items():
        target_idx = idx_dba_s2 if eid.startswith("S2-") else idx_dba_s3
        if e["url_core"] and len(e["url_core"]) >= 6:
            target_idx[e["url_core"]].append(eid)
        if e["concat_stem"] and len(e["concat_stem"]) >= 8:
            target_idx[e["concat_stem"]].append(eid)

    print(f"DBA index keys: S2={len(idx_dba_s2):,d}, S3={len(idx_dba_s3):,d}", flush=True)

    # 3. Candidate Generation with Strict Gate
    print("Generating candidates with Strict-Gated DBA recovery...", flush=True)
    pairs = []
    features_list = []
    labels = []
    groups = []
    hits = 0
    cand_counts = []
    new_dba_cands_total = 0
    new_dba_true_recovered = 0
    new_dba_false_added = 0
    zero_match_new_cands = 0

    MAX_CANDS = 120

    for s1_id, s1_row in s1_dict.items():
        e1 = cached_s1[s1_id]
        true_matches = gt_dict.get(s1_id, set())
        ordered_passes = extract_champion_keys(e1)

        # Baseline candidates
        base_cands = set()
        for p_idx, pass_keys in enumerate(ordered_passes):
            cap = 150 if p_idx == 4 else 400
            for k in pass_keys:
                p2 = idx_s2_base.get(k, [])
                p3 = idx_s3_base.get(k, [])
                if len(p2) <= cap: base_cands.update(p2)
                if len(p3) <= cap: base_cands.update(p3)
                if len(base_cands) >= MAX_CANDS: break
            if len(base_cands) >= MAX_CANDS: break

        final_cands = set(base_cands)

        # STRICT DBA RECOVERY PASS
        # Look for domain/stem matches
        dba_candidates_pool = set()
        stems_to_probe = []
        if e1["concat_stem"]: stems_to_probe.append(e1["concat_stem"])
        if e1["url_core"]: stems_to_probe.append(e1["url_core"])

        for stem in stems_to_probe:
            p2 = idx_dba_s2.get(stem, [])
            p3 = idx_dba_s3.get(stem, [])
            # Only consider if posting list is small (<= 5 distinct entities, highly distinctive)
            if len(p2) <= 5: dba_candidates_pool.update(p2)
            if len(p3) <= 5: dba_candidates_pool.update(p3)

        # STRICT GATE: must have independent supporting evidence
        c1 = e1["country"]
        nums1 = e1["nums"]
        pin1 = e1["pincode"]
        toks_a1 = e1["toks_a"]

        for cand_id in dba_candidates_pool:
            if cand_id in final_cands: continue # already retrieved
            e2 = cached_cands.get(cand_id)
            if e2 is None: continue
            c2 = e2["country"]
            if c1 and c2 and c1 != c2: continue # country mismatch

            nums2 = e2["nums"]
            pin2 = e2["pincode"]
            toks_a2 = e2["toks_a"]

            # Independent supporting signals
            has_num_match = bool(nums1 & nums2)
            has_pin_match = bool(pin1 and pin2 and pin1 == pin2)
            has_distinctive_addr_overlap = bool(toks_a1 & toks_a2 and any(len(t) >= 5 for t in (toks_a1 & toks_a2)))
            addr_sim = fuzz.token_sort_ratio(e1["norm_a"], e2["norm_a"]) if (e1["norm_a"] and e2["norm_a"]) else 0

            # GATE: Pass if at least ONE strong supporting signal exists
            passed_gate = False
            if has_num_match or has_pin_match or has_distinctive_addr_overlap or addr_sim >= 40:
                passed_gate = True
            elif (not e1["norm_a"] or not e2["norm_a"]) and len(stem) >= 12:
                # Extreme empty address: only accept if domain stem is very long (>= 12 chars, e.g. saffronfinancers)
                passed_gate = True

            if passed_gate:
                final_cands.add(cand_id)
                new_dba_cands_total += 1
                if cand_id in true_matches:
                    new_dba_true_recovered += 1
                else:
                    new_dba_false_added += 1
                if len(true_matches) == 0:
                    zero_match_new_cands += 1

        cand_counts.append(len(final_cands))
        hits += len(true_matches & final_cands)

        for cid in final_cands:
            e2 = cached_cands.get(cid)
            if e2 is None: continue
            c2 = e2["country"]
            if c1 and c2 and c1 != c2: continue
            f_vec = extract_expanded_features(e1, e2, cid)
            pairs.append((s1_id, cid))
            features_list.append(f_vec)
            labels.append(1 if cid in true_matches else 0)
            groups.append(s1_id)

    cand_recall = hits / total_true
    avg_c = float(np.mean(cand_counts))
    med_c = float(np.median(cand_counts))
    p95_c = float(np.percentile(cand_counts, 95))
    p99_c = float(np.percentile(cand_counts, 99))

    print(f"\n--- DBA RECOVERY HARD NEGATIVE AUDIT ---")
    print(f"Total True Pairs: {total_true:,d} | Retrieved: {hits:,d} ({cand_recall*100:.2f}%)")
    print(f"New DBA candidates added: {new_dba_cands_total:,d}")
    print(f"  -> True pairs recovered: {new_dba_true_recovered:,d}")
    print(f"  -> False candidates added (Hard Negatives): {new_dba_false_added:,d}")
    print(f"  -> False candidate rate: {new_dba_false_added/max(new_dba_cands_total,1)*100:.1f}%")
    print(f"Zero-match entities affected: {zero_match_new_cands:,d}")
    print(f"Candidate counts: Avg={avg_c:.1f}, Med={med_c:.1f}, P95={p95_c:.1f}, P99={p99_c:.1f}")
    print(f"Total Evaluated Pairs: {len(pairs):,d}")

    # 4. Train 5-fold LightGBM Matcher
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(groups)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups))
    oof_probs = np.zeros(len(y), dtype=np.float32)

    print("\nTraining 5-fold LightGBM matchers...", flush=True)
    for fold, (train_idx, val_idx) in enumerate(splits):
        clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    # Dense threshold sweep [0.50 to 0.85]
    print("\nSweeping threshold for optimal Macro F0.5...", flush=True)
    best_tau = 0.62
    best_f05 = 0.0
    for tau in np.arange(0.50, 0.86, 0.02):
        t_preds = defaultdict(set)
        for (sid, cid), p in zip(pairs, oof_probs):
            if p >= tau: t_preds[sid].add(cid)
        m = compute_comprehensive_metrics(gt_dict, t_preds, beta=Config.BETA)
        print(f"  tau={tau:.2f} -> Macro F0.5: {m['macro_f05']:.4f} (Prec: {m['macro_precision']:.4f}, Rec: {m['macro_recall']:.4f})", flush=True)
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_tau = tau

    preds_opt = defaultdict(set)
    for (sid, cid), p in zip(pairs, oof_probs):
        if p >= best_tau: preds_opt[sid].add(cid)
    m_opt = compute_comprehensive_metrics(gt_dict, preds_opt, beta=Config.BETA)
    sub_opt = evaluate_subsets(gt_dict, preds_opt, s1_dict)

    fold_f05 = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        val_pairs = [pairs[i] for i in val_idx]
        val_s1_set = set(groups[val_idx])
        val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_set}
        fold_p = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, oof_probs[val_idx]):
            if p >= best_tau: fold_p[sid].add(cid)
        m_fold = compute_comprehensive_metrics(val_gt, fold_p, beta=Config.BETA)
        fold_f05.append(m_fold["macro_f05"])
        print(f"  Fold {fold+1}: Macro F0.5 = {m_fold['macro_f05']:.4f} (Prec = {m_fold['macro_precision']:.4f}, Rec = {m_fold['macro_recall']:.4f})", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"STRICT DBA OVERALL OOF Macro F0.5: {m_opt['macro_f05']:.4f} at tau={best_tau:.2f}", flush=True)
    print(f"Macro Precision: {m_opt['macro_precision']:.4f} | Macro Recall: {m_opt['macro_recall']:.4f}", flush=True)
    print(f"Micro F0.5: {m_opt['micro_f05']:.4f}", flush=True)
    print(f"Fold Mean: {np.mean(fold_f05):.4f} +/- {np.std(fold_f05):.4f}", flush=True)
    print(f"Cardinality: Singleton={sub_opt['singleton']:.4f}, 1-Match={sub_opt['one_match']:.4f}, Multi={sub_opt['multi_match']:.4f}", flush=True)
    print(f"Country: US={sub_opt['us']:.4f}, India={sub_opt['india']:.4f}", flush=True)
    print(f"Confusion: TP={m_opt['tp']:,d}, FP={m_opt['fp']:,d}, FN={m_opt['fn']:,d}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
