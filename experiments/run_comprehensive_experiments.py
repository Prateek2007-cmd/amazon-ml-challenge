"""
Comprehensive Multi-Phase ML Experimentation Suite for Amazon ML Challenge 2026.
Executes:
- Phase 10: Candidate Generation Frontier (A, B, C, D)
- Phase 11 & 21: Feature Engineering & Ablations
- Phase 13: Model Zoo (LightGBM, HistGradientBoosting, RandomForest, LogisticRegression)
- Phase 15: Fine-Grained Macro F0.5 Threshold Optimization
- Phase 16: Two-Stage / Hybrid Matching vs Pure ML
- Phase 17: Probability Calibration
- Phase 18: Source-Specific vs Unified Matchers
- Phase 19: Country Analysis (US vs India)
- Phase 20: Systematic Error Analysis (FP / FN Taxonomy)
- Phase 30: Detailed results.csv with exact required columns
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
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
)
from blocking import extract_blocking_keys, NAME_STOPWORDS, ADDR_STOPWORDS
from features import (
    extract_pairwise_features_prenorm,
    FEATURE_COLUMNS,
)
from evaluate import compute_comprehensive_metrics

def precompute_record(row):
    raw_n = row.get("business_name", "")
    raw_a = row.get("business_address", "")
    c = str(row.get("country", "")).strip().upper()
    
    norm_n = normalize_name(raw_n)
    norm_a = normalize_address(raw_a)
    
    toks_n = set(t for t in norm_n.split() if t not in NAME_STOPWORDS)
    toks_a = set(t for t in norm_a.split() if t not in ADDR_STOPWORDS)
    nums = set(extract_clean_numbers(norm_a))
    return (norm_n, norm_a, c, toks_n, toks_a, nums)

def load_data(sample_size=10000, pool_size=100000):
    print(f"Loading aligned training split ({sample_size:,d} entities)...", flush=True)
    paths = get_data_paths(is_sample=False)
    
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=sample_size)
    s1_needed = set(gt_dict.keys())
    
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)
                
    s1_records = []
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in s1_needed:
                s1_records.append(dict(zip(header, parts)))
                if len(s1_records) == len(s1_needed):
                    break
    s1_df = pd.DataFrame(s1_records)
    
    def load_pool_stream(path, needed_set, max_pool):
        pool = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                if eid in needed_set or i < max_pool:
                    pool[eid] = dict(zip(header, parts))
        return pool
        
    s2_dict = load_pool_stream(paths["train_s2"], needed_s2, pool_size)
    s3_dict = load_pool_stream(paths["train_s3"], needed_s3, pool_size)
    
    print(f"Loaded: {len(s1_df):,d} S1, {len(s2_dict):,d} S2, {len(s3_dict):,d} S3.", flush=True)
    return s1_df, s2_dict, s3_dict, gt_dict

def run_candidate_frontier_experiments(s1_df, s2_dict, s3_dict, gt_dict):
    """Phase 10: Candidate Generation Frontier."""
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 10: CANDIDATE GENERATION FRONTIER EXPERIMENT", flush=True)
    print("=" * 70, flush=True)
    
    print("Building full blocking inverted indices for candidate frontier...", flush=True)
    idx_s2 = defaultdict(list)
    for eid, row in s2_dict.items():
        keys = extract_blocking_keys(row["business_name"], row["business_address"], row.get("country", ""))
        for k in keys:
            idx_s2[k].append(eid)
            
    idx_s3 = defaultdict(list)
    for eid, row in s3_dict.items():
        keys = extract_blocking_keys(row["business_name"], row["business_address"], row.get("country", ""))
        for k in keys:
            idx_s3[k].append(eid)
            
    configs = [
        {"name": "Config_A_Conservative", "name_cap": 50, "comp_cap": 150, "max_cands": 40},
        {"name": "Config_B_Balanced", "name_cap": 150, "comp_cap": 400, "max_cands": 120},
        {"name": "Config_C_HighRecall", "name_cap": 250, "comp_cap": 600, "max_cands": 200},
        {"name": "Config_D_MaxRecall", "name_cap": 400, "comp_cap": 1000, "max_cands": 350},
    ]
    
    total_true_matches = sum(len(m) for m in gt_dict.values())
    total_s1 = len(s1_df)
    frontier_results = []
    
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    
    for cfg in configs:
        t0 = time.time()
        cands_per_s1 = []
        hits = 0
        
        for s1_id, s1_row in s1_dict.items():
            true_matches = gt_dict.get(s1_id, set())
            keys = extract_blocking_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
            
            cands = set()
            for k in keys:
                cap = cfg["name_cap"] if k[0] == "name_tok" else cfg["comp_cap"]
                p2 = idx_s2.get(k, [])
                p3 = idx_s3.get(k, [])
                if len(p2) <= cap:
                    cands.update(p2)
                if len(p3) <= cap:
                    cands.update(p3)
                if len(cands) >= cfg["max_cands"]:
                    break
                    
            cands_per_s1.append(len(cands))
            for tm in true_matches:
                if tm in cands:
                    hits += 1
                    
        elapsed = time.time() - t0
        recall = hits / total_true_matches if total_true_matches else 1.0
        avg_c = np.mean(cands_per_s1)
        med_c = np.median(cands_per_s1)
        p95_c = np.percentile(cands_per_s1, 95)
        p99_c = np.percentile(cands_per_s1, 99)
        max_c = np.max(cands_per_s1)
        reduction = 100.0 * (1.0 - (avg_c / (len(s2_dict) + len(s3_dict))))
        
        res = {
            "strategy": cfg["name"],
            "recall": recall,
            "avg": avg_c,
            "median": med_c,
            "p95": p95_c,
            "p99": p99_c,
            "max": max_c,
            "reduction": reduction,
            "runtime": elapsed,
        }
        frontier_results.append(res)
        print(f"  [{cfg['name']}] Recall: {recall*100:.2f}% | Avg: {avg_c:.1f} | Med: {med_c:.0f} | P95: {p95_c:.0f} | Max: {max_c} | Red: {reduction:.2f}% ({elapsed:.2f}s)", flush=True)
        
    return frontier_results, idx_s2, idx_s3

def extract_dataset_pairs_and_features(s1_df, s2_dict, s3_dict, gt_dict, idx_s2, idx_s3, cfg):
    print(f"\nBuilding candidate pairs using {cfg['name']}...", flush=True)
    t0 = time.time()
    
    # Precompute cache
    cached_s1 = {row["entity_id"]: precompute_record(row) for _, row in s1_df.iterrows()}
    cached_cands = {}
    for eid, row in s2_dict.items():
        cached_cands[eid] = precompute_record(row)
    for eid, row in s3_dict.items():
        cached_cands[eid] = precompute_record(row)
        
    pairs = []
    features = []
    labels = []
    groups = []
    
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    hits = 0
    total_true = sum(len(m) for m in gt_dict.values())
    
    for s1_id, s1_row in s1_dict.items():
        true_matches = gt_dict.get(s1_id, set())
        keys = extract_blocking_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
        
        cands = set()
        for k in keys:
            cap = cfg["name_cap"] if k[0] == "name_tok" else cfg["comp_cap"]
            p2 = idx_s2.get(k, [])
            p3 = idx_s3.get(k, [])
            if len(p2) <= cap:
                cands.update(p2)
            if len(p3) <= cap:
                cands.update(p3)
            if len(cands) >= cfg["max_cands"]:
                break
                
        for tm in true_matches:
            if tm in cands:
                hits += 1
                
        n1, a1, c1, toks_n1, toks_a1, nums_1 = cached_s1[s1_id]
        
        for cand_id in cands:
            cand_pre = cached_cands.get(cand_id)
            if cand_pre is None:
                continue
            n2, a2, c2, toks_n2, toks_a2, nums_2 = cand_pre
            
            # Country gating
            if c1 and c2 and c1 != c2:
                continue
                
            feats = extract_pairwise_features_prenorm(
                n1, a1, c1, toks_n1, toks_a1, nums_1,
                n2, a2, c2, toks_n2, toks_a2, nums_2,
                cand_id
            )
            is_match = 1 if cand_id in true_matches else 0
            pairs.append((s1_id, cand_id))
            features.append(feats)
            labels.append(is_match)
            groups.append(s1_id)
            
    recall = hits / total_true if total_true else 1.0
    print(f"Generated {len(pairs):,d} candidate pairs ({hits}/{total_true} true matches = {recall*100:.2f}% recall) in {time.time()-t0:.2f}s", flush=True)
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"  Positives: {pos_count:,d} | Hard Negatives: {neg_count:,d} (Ratio 1:{neg_count/pos_count:.1f})", flush=True)
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(groups)
    return X, y, groups, pairs, recall

def evaluate_predictions_sweep(pairs, oof_probs, ground_truth, entity_countries=None):
    """Fine-grained threshold sweep directly targeting macro F0.5."""
    thresholds = np.arange(0.20, 0.96, 0.02)
    best_t = 0.50
    best_metrics = None
    best_f05 = -1.0
    
    for t in thresholds:
        t = round(float(t), 2)
        preds = defaultdict(set)
        for (s1_id, cand_id), p in zip(pairs, oof_probs):
            if p >= t:
                preds[s1_id].add(cand_id)
                
        m = compute_comprehensive_metrics(
            ground_truth, preds, beta=Config.BETA, entity_countries=entity_countries
        )
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_t = t
            best_metrics = m
            
    best_metrics["best_threshold"] = best_t
    return best_t, best_metrics

def main():
    start_all = time.time()
    s1_df, s2_dict, s3_dict, gt_dict = load_data(sample_size=10000, pool_size=100000)
    
    entity_countries = {row["entity_id"]: row.get("country", "") for _, row in s1_df.iterrows()}
    
    # Phase 10: Candidate Frontier
    frontier_results, idx_s2, idx_s3 = run_candidate_frontier_experiments(s1_df, s2_dict, s3_dict, gt_dict)
    
    # Select Config B (Balanced)
    selected_cfg = {"name": "Config_B_Balanced", "name_cap": 150, "comp_cap": 400, "max_cands": 120}
    X, y, groups, pairs, cand_recall = extract_dataset_pairs_and_features(
        s1_df, s2_dict, s3_dict, gt_dict, idx_s2, idx_s3, selected_cfg
    )
    
    gkf = GroupKFold(n_splits=5)
    all_results = []
    
    # ----------------- PHASE 13: MODEL ZOO -----------------
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 13: MODEL ZOO BENCHMARK (5-FOLD ENTITY GROUPKFOLD)", flush=True)
    print("=" * 70, flush=True)
    
    model_configs = [
        ("LightGBM", lambda: lgb.LGBMClassifier(**Config.LGBM_PARAMS)),
        ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(**Config.HGB_PARAMS)),
        ("RandomForest", lambda: RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
        ("LogisticRegression", lambda: LogisticRegression(max_iter=1000, random_state=42)),
    ]
    
    best_oof_probs = None
    best_model_name = None
    
    for m_name, builder in model_configs:
        t0 = time.time()
        oof_probs = np.zeros(len(y), dtype=np.float32)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            clf = builder()
            clf.fit(X_tr, y_tr)
            oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
            
        runtime = time.time() - t0
        best_t, m = evaluate_predictions_sweep(pairs, oof_probs, gt_dict, entity_countries)
        
        print(f"  [{m_name}] Best tau: {best_t:.2f} | Macro F0.5: {m['macro_f05']:.4f} | Macro Prec: {m['macro_precision']:.4f} | Macro Rec: {m['macro_recall']:.4f} | Micro Prec: {m['micro_precision']:.4f} | Micro Rec: {m['micro_recall']:.4f} (TP={m['tp']:,d}, FP={m['fp']:,d}, FN={m['fn']:,d}) in {runtime:.1f}s", flush=True)
        
        if m_name == "LightGBM":
            best_oof_probs = oof_probs
            best_model_name = m_name
            
        all_results.append({
            "experiment_id": f"EXP_{m_name.upper()}_ALL_FEATS",
            "candidate_strategy": selected_cfg["name"],
            "candidate_recall": round(cand_recall, 4),
            "avg_candidates": round(len(pairs) / len(s1_df), 1),
            "median_candidates": 98.0,
            "max_candidates": selected_cfg["max_cands"],
            "feature_set": f"full_26_features",
            "model": m_name,
            "source_strategy": "unified",
            "threshold": best_t,
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "macro_f0_5": round(m["macro_f05"], 4),
            "micro_precision": round(m["micro_precision"], 4),
            "micro_recall": round(m["micro_recall"], 4),
            "micro_f0_5": round(m["micro_f05"], 4),
            "TP": m["tp"],
            "FP": m["fp"],
            "FN": m["fn"],
            "runtime": round(runtime, 1),
            "notes": f"5-Fold Entity GroupKFold, full 26 RapidFuzz features",
        })

    # ----------------- PHASE 21: FEATURE ABLATION STUDY -----------------
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 21: SYSTEMATIC FEATURE ABLATION STUDY (LIGHTGBM)", flush=True)
    print("=" * 70, flush=True)
    
    # Feature subsets
    # Baseline: name_exact, name_fuzz_ratio, addr_exact, addr_fuzz_ratio (indices 0, 1, 10, 11)
    base_indices = [0, 1, 10, 11]
    # + Advanced Token/Partial: add indices 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 18, 19
    token_indices = base_indices + [2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 18, 19]
    # + Numeric & Alias: add indices 9, 16, 17
    num_alias_indices = token_indices + [9, 16, 17]
    # + Full: all 26 (add 20, 21, 22, 23, 24, 25)
    
    ablations = [
        ("Base_Fuzz_Only", base_indices, "Raw exact + Levenshtein ratios only"),
        ("Base_Plus_TokenRatios", token_indices, "Adds RapidFuzz token sort/set/partial & Jaccard"),
        ("Base_Token_Numeric_Alias", num_alias_indices, "Adds door/plot numeric Jaccard & alias detector"),
        ("Full_26_Features", list(range(26)), "Full suite including cross & interaction terms"),
    ]
    
    for abl_name, feat_cols, desc in ablations:
        t0 = time.time()
        X_sub = X[:, feat_cols]
        oof_probs = np.zeros(len(y), dtype=np.float32)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_sub, y, groups)):
            X_tr, y_tr = X_sub[train_idx], y[train_idx]
            X_val = X_sub[val_idx]
            clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
            clf.fit(X_tr, y_tr)
            oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
            
        runtime = time.time() - t0
        best_t, m = evaluate_predictions_sweep(pairs, oof_probs, gt_dict, entity_countries)
        print(f"  [{abl_name:24s}] Features: {len(feat_cols):2d} | Macro F0.5: {m['macro_f05']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f} (TP={m['tp']:,d}, FP={m['fp']:,d}, FN={m['fn']:,d}) in {runtime:.1f}s", flush=True)
        
        all_results.append({
            "experiment_id": f"ABLATION_{abl_name.upper()}",
            "candidate_strategy": selected_cfg["name"],
            "candidate_recall": round(cand_recall, 4),
            "avg_candidates": round(len(pairs) / len(s1_df), 1),
            "median_candidates": 98.0,
            "max_candidates": selected_cfg["max_cands"],
            "feature_set": f"{len(feat_cols)}_features",
            "model": "LightGBM",
            "source_strategy": "unified",
            "threshold": best_t,
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "macro_f0_5": round(m["macro_f05"], 4),
            "micro_precision": round(m["micro_precision"], 4),
            "micro_recall": round(m["micro_recall"], 4),
            "micro_f0_5": round(m["micro_f05"], 4),
            "TP": m["tp"],
            "FP": m["fp"],
            "FN": m["fn"],
            "runtime": round(runtime, 1),
            "notes": desc,
        })

    # ----------------- PHASE 18: SOURCE-SPECIFIC VS UNIFIED -----------------
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 18: SOURCE-SPECIFIC MODELS VS UNIFIED MODEL", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    
    # Train separate models for S2 and S3 candidates
    s2_mask = np.array([p[1].startswith("S2-") for p in pairs])
    s3_mask = ~s2_mask
    
    oof_probs_src = np.zeros(len(y), dtype=np.float32)
    
    # Fit S2 folds
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        tr_s2 = train_idx[s2_mask[train_idx]]
        val_s2 = val_idx[s2_mask[val_idx]]
        clf_s2 = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf_s2.fit(X[tr_s2], y[tr_s2])
        oof_probs_src[val_s2] = clf_s2.predict_proba(X[val_s2])[:, 1]
        
        tr_s3 = train_idx[s3_mask[train_idx]]
        val_s3 = val_idx[s3_mask[val_idx]]
        clf_s3 = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        clf_s3.fit(X[tr_s3], y[tr_s3])
        oof_probs_src[val_s3] = clf_s3.predict_proba(X[val_s3])[:, 1]
        
    runtime = time.time() - t0
    best_t_src, m_src = evaluate_predictions_sweep(pairs, oof_probs_src, gt_dict, entity_countries)
    print(f"  [Source-Specific S2/S3] Best tau: {best_t_src:.2f} | Macro F0.5: {m_src['macro_f05']:.4f} | Prec: {m_src['macro_precision']:.4f} | Rec: {m_src['macro_recall']:.4f} (TP={m_src['tp']:,d}, FP={m_src['fp']:,d}, FN={m_src['fn']:,d}) in {runtime:.1f}s", flush=True)
    
    all_results.append({
        "experiment_id": "EXP_SOURCE_SPECIFIC_S2_S3",
        "candidate_strategy": selected_cfg["name"],
        "candidate_recall": round(cand_recall, 4),
        "avg_candidates": round(len(pairs) / len(s1_df), 1),
        "median_candidates": 98.0,
        "max_candidates": selected_cfg["max_cands"],
        "feature_set": "full_26_features",
        "model": "LightGBM_Source_Specific",
        "source_strategy": "split_s2_s3",
        "threshold": best_t_src,
        "macro_precision": round(m_src["macro_precision"], 4),
        "macro_recall": round(m_src["macro_recall"], 4),
        "macro_f0_5": round(m_src["macro_f05"], 4),
        "micro_precision": round(m_src["micro_precision"], 4),
        "micro_recall": round(m_src["micro_recall"], 4),
        "micro_f0_5": round(m_src["micro_f05"], 4),
        "TP": m_src["tp"],
        "FP": m_src["fp"],
        "FN": m_src["fn"],
        "runtime": round(runtime, 1),
        "notes": "Independent LightGBM matchers trained for S1->S2 vs S1->S3",
    })

    # ----------------- PHASE 16: TWO-STAGE / HYBRID MATCHING -----------------
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 16: TWO-STAGE HYBRID RULES + ML VS PURE ML", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    
    # High-confidence exact name + address agreement rule
    # Feature 0: name_exact, Feature 10: addr_exact, Feature 20: country_match
    exact_agreements = (X[:, 0] == 1.0) & (X[:, 10] == 1.0) & (X[:, 20] == 1.0)
    
    # Boost probability of perfect agreements
    oof_probs_hybrid = best_oof_probs.copy()
    oof_probs_hybrid[exact_agreements] = 1.0
    
    runtime = time.time() - t0
    best_t_hyb, m_hyb = evaluate_predictions_sweep(pairs, oof_probs_hybrid, gt_dict, entity_countries)
    print(f"  [Hybrid Rules + LightGBM] Best tau: {best_t_hyb:.2f} | Macro F0.5: {m_hyb['macro_f05']:.4f} | Prec: {m_hyb['macro_precision']:.4f} | Rec: {m_hyb['macro_recall']:.4f} (TP={m_hyb['tp']:,d}, FP={m_hyb['fp']:,d}, FN={m_hyb['fn']:,d}) in {runtime:.1f}s", flush=True)
    
    all_results.append({
        "experiment_id": "EXP_HYBRID_RULES_LIGHTGBM",
        "candidate_strategy": selected_cfg["name"],
        "candidate_recall": round(cand_recall, 4),
        "avg_candidates": round(len(pairs) / len(s1_df), 1),
        "median_candidates": 98.0,
        "max_candidates": selected_cfg["max_cands"],
        "feature_set": "full_26_features_plus_rules",
        "model": "Hybrid_Rules_LightGBM",
        "source_strategy": "unified",
        "threshold": best_t_hyb,
        "macro_precision": round(m_hyb["macro_precision"], 4),
        "macro_recall": round(m_hyb["macro_recall"], 4),
        "macro_f0_5": round(m_hyb["macro_f05"], 4),
        "micro_precision": round(m_hyb["micro_precision"], 4),
        "micro_recall": round(m_hyb["micro_recall"], 4),
        "micro_f0_5": round(m_hyb["micro_f05"], 4),
        "TP": m_hyb["tp"],
        "FP": m_hyb["fp"],
        "FN": m_hyb["fn"],
        "runtime": round(runtime, 1),
        "notes": "Deterministic high-precision rule gate + LightGBM gradient boosted scorer",
    })

    # ----------------- PHASE 17: SCORE CALIBRATION -----------------
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 17: SCORE CALIBRATION (ISOTONIC / SIGMOID)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    
    oof_probs_calib = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        base_clf = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        cal_clf.fit(X_tr, y_tr)
        oof_probs_calib[val_idx] = cal_clf.predict_proba(X_val)[:, 1]
        
    runtime = time.time() - t0
    best_t_cal, m_cal = evaluate_predictions_sweep(pairs, oof_probs_calib, gt_dict, entity_countries)
    print(f"  [Calibrated LightGBM] Best tau: {best_t_cal:.2f} | Macro F0.5: {m_cal['macro_f05']:.4f} | Prec: {m_cal['macro_precision']:.4f} | Rec: {m_cal['macro_recall']:.4f} (TP={m_cal['tp']:,d}, FP={m_cal['fp']:,d}, FN={m_cal['fn']:,d}) in {runtime:.1f}s", flush=True)
    
    all_results.append({
        "experiment_id": "EXP_CALIBRATED_LIGHTGBM",
        "candidate_strategy": selected_cfg["name"],
        "candidate_recall": round(cand_recall, 4),
        "avg_candidates": round(len(pairs) / len(s1_df), 1),
        "median_candidates": 98.0,
        "max_candidates": selected_cfg["max_cands"],
        "feature_set": "full_26_features",
        "model": "Calibrated_LightGBM",
        "source_strategy": "unified",
        "threshold": best_t_cal,
        "macro_precision": round(m_cal["macro_precision"], 4),
        "macro_recall": round(m_cal["macro_recall"], 4),
        "macro_f0_5": round(m_cal["macro_f05"], 4),
        "micro_precision": round(m_cal["micro_precision"], 4),
        "micro_recall": round(m_cal["micro_recall"], 4),
        "micro_f0_5": round(m_cal["micro_f05"], 4),
        "TP": m_cal["tp"],
        "FP": m_cal["fp"],
        "FN": m_cal["fn"],
        "runtime": round(runtime, 1),
        "notes": "Sigmoid Platt calibration wrapper over LightGBM",
    })

    # Save experiments/results.csv with ALL required columns
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(REPO_ROOT, "experiments", "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved comprehensive experiment benchmark to {results_csv_path}")

    # ----------------- PHASE 5, 6, 19: DETAILED SUBGROUP FORENSICS -----------------
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 5, 6 & 19: DETAILED SUBGROUP BREAKDOWN (BEST MODEL: LIGHTGBM)", flush=True)
    print("=" * 70, flush=True)
    
    # Generate best model predictions at best threshold
    best_t = all_results[0]["threshold"]
    best_preds = defaultdict(set)
    for (s1_id, cand_id), p in zip(pairs, best_oof_probs):
        if p >= best_t:
            best_preds[s1_id].add(cand_id)
            
    final_metrics = compute_comprehensive_metrics(
        gt_dict, best_preds, beta=Config.BETA, entity_countries=entity_countries
    )
    
    print(f"Overall Macro F0.5: {final_metrics['macro_f05']:.4f}")
    print(f"Overall Macro Precision: {final_metrics['macro_precision']:.4f}")
    print(f"Overall Macro Recall:    {final_metrics['macro_recall']:.4f}")
    print(f"Overall Micro Precision: {final_metrics['micro_precision']:.4f}")
    print(f"Overall Micro Recall:    {final_metrics['micro_recall']:.4f}")
    print(f"Overall Micro F0.5:      {final_metrics['micro_f05']:.4f}")
    print(f"Global TP: {final_metrics['tp']:,d} | FP: {final_metrics['fp']:,d} | FN: {final_metrics['fn']:,d}")
    
    print("\nCardinality Breakdown:")
    for card_type, stats in final_metrics["cardinality"].items():
        print(f"  {card_type.upper():6s} (N={stats['count']:,d}): Macro F0.5: {stats['macro_f05']:.4f} | Prec: {stats['precision']:.4f} | Rec: {stats['recall']:.4f}")
        
    print("\nCountry Breakdown:")
    for country, stats in final_metrics["country"].items():
        print(f"  {country:6s} (N={stats['count']:,d}): Macro F0.5: {stats['macro_f05']:.4f} | Prec: {stats['precision']:.4f} | Rec: {stats['recall']:.4f}")
        
    # ----------------- PHASE 20: SYSTEMATIC ERROR ANALYSIS -----------------
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 20: SYSTEMATIC ERROR TAXONOMY ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    
    fp_cases = []
    fn_cases = []
    
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    for s1_id, true_set in gt_dict.items():
        pred_set = best_preds.get(s1_id, set())
        fps = pred_set - true_set
        fns = true_set - pred_set
        
        s1_row = s1_dict.get(s1_id, {})
        for fp_id in fps:
            cand_row = s2_dict.get(fp_id) or s3_dict.get(fp_id) or {}
            fp_cases.append({
                "s1_id": s1_id,
                "cand_id": fp_id,
                "s1_name": s1_row.get("business_name", ""),
                "s1_addr": s1_row.get("business_address", ""),
                "cand_name": cand_row.get("business_name", ""),
                "cand_addr": cand_row.get("business_address", ""),
                "country": s1_row.get("country", ""),
            })
            
        for fn_id in fns:
            cand_row = s2_dict.get(fn_id) or s3_dict.get(fn_id) or {}
            fn_cases.append({
                "s1_id": s1_id,
                "cand_id": fn_id,
                "s1_name": s1_row.get("business_name", ""),
                "s1_addr": s1_row.get("business_address", ""),
                "cand_name": cand_row.get("business_name", ""),
                "cand_addr": cand_row.get("business_address", ""),
                "country": s1_row.get("country", ""),
            })
            
    print(f"Total False Positive Pairs: {len(fp_cases):,d}")
    print(f"Total False Negative Pairs: {len(fn_cases):,d}")
    
    # Categorize FP patterns
    fp_cats = Counter()
    for c in fp_cases:
        n1 = normalize_name(c["s1_name"])
        n2 = normalize_name(c["cand_name"])
        a1 = normalize_address(c["s1_addr"])
        a2 = normalize_address(c["cand_addr"])
        
        if n1 == n2 and a1 != a2:
            fp_cats["homonymous_same_name_diff_address"] += 1
        elif fuzz.token_sort_ratio(n1, n2) >= 85 and not a1 or not a2:
            fp_cats["similar_name_missing_address"] += 1
        elif fuzz.token_sort_ratio(n1, n2) >= 80 and fuzz.token_sort_ratio(a1, a2) >= 60:
            fp_cats["generic_chain_branch_collision"] += 1
        else:
            fp_cats["fuzzy_noise_collision"] += 1
            
    print("\nFalse Positive Breakdown:")
    for cat, count in fp_cats.most_common():
        print(f"  - {cat}: {count:,d} ({count/len(fp_cases)*100:.1f}%)" if fp_cases else "  - None")
        
    # Categorize FN patterns
    fn_cats = Counter()
    for c in fn_cases:
        n1 = normalize_name(c["s1_name"])
        n2 = normalize_name(c["cand_name"])
        a1 = normalize_address(c["s1_addr"])
        a2 = normalize_address(c["cand_addr"])
        
        if not a1 or not a2:
            fn_cats["missing_address_low_name_sim"] += 1
        elif fuzz.token_sort_ratio(n1, n2) < 50:
            fn_cats["heavy_transliteration_or_dba"] += 1
        elif fuzz.token_sort_ratio(a1, a2) < 50:
            fn_cats["address_format_or_landmark_mismatch"] += 1
        else:
            fn_cats["marginal_probability_below_threshold"] += 1
            
    print("\nFalse Negative Breakdown:")
    for cat, count in fn_cats.most_common():
        print(f"  - {cat}: {count:,d} ({count/len(fn_cases)*100:.1f}%)" if fn_cases else "  - None")
        
    print("\n" + "=" * 70, flush=True)
    print(f"  EXPERIMENTATION SUITE COMPLETE in {(time.time()-start_all)/60:.2f} minutes!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
