"""
End-to-End Autonomous Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Wired end-to-end:
1. Fast streaming data loading (zero RAM bloat)
2. Multi-representation normalization
3. High-recall composite blocking (candidate generation)
4. Pairwise feature engineering (26 C-accelerated RapidFuzz features)
5. Model training (LightGBM / HistGradientBoosting)
6. Out-of-fold threshold optimization for F0.5
7. Scalable chunked inference on test set
8. Output generation and strict formatting
9. Official submission validation
10. Final ZIP packaging
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import get_data_paths, load_tsv, parse_ground_truth_fast, load_training_subsample
from normalization import normalize_name, normalize_address
from blocking import build_blocking_index, blocking_recall, extract_blocking_keys
from features import extract_pairwise_features, FEATURE_COLUMNS
from models import build_classifier, train_matcher, sweep_threshold_f05, save_model
from validation import run_entity_level_cv
from inference import run_chunked_inference
from submission import write_results_tsv, run_official_validator, package_submission_zip

def parse_args():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default=None, help="Path to dataset root folder")
    parser.add_argument("--output-dir", default=Config.OUTPUT_DIR, help="Path to output folder")
    parser.add_argument("--sample", action="store_true", help="Run smoke test on tiny sample fixture")
    parser.add_argument("--train-sample-size", type=int, default=10000, help="Number of S1 training entities for model fitting (default: 10,000)")
    parser.add_argument("--test-sample-size", type=int, default=None, help="Optional sample size for test inference (default: all test entities)")
    parser.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "hist_gradient_boosting"], help="Model type to train")
    parser.add_argument("--team-name", default="Antigravity_ML", help="Team name for final zip package")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 70, flush=True)
    print("  AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE", flush=True)
    print("  Mode: MAXIMUM COMPETITION MODE (Metric: Macro F_0.5)", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Resolve paths
    paths = get_data_paths(args.data_dir, is_sample=args.sample)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[PHASE 1] Resolving Dataset Paths from {os.path.dirname(paths['train_s1'])}...", flush=True)
    
    # 2. Load training data
    if args.sample:
        print("  Running in SAMPLE MODE on synthetic fixture...", flush=True)
        s1_train_df = load_tsv(paths["train_s1"])
        s2_train_df = load_tsv(paths["train_s2"])
        s3_train_df = load_tsv(paths["train_s3"])
        gt_train = parse_ground_truth_fast(paths["train_gt"])
        s2_dict = s2_train_df.set_index("entity_id").to_dict("index")
        s3_dict = s3_train_df.set_index("entity_id").to_dict("index")
    else:
        print(f"\n[PHASE 2] Loading Aligned Training Subsample ({args.train_sample_size:,d} entities)...", flush=True)
        s1_train_df, s2_dict, s3_dict, gt_train = load_training_subsample(
            paths, sample_size=args.train_sample_size, pool_size=100000
        )
        
    print(f"  Training Pool: {len(s1_train_df):,d} S1 entities | {len(s2_dict):,d} S2 | {len(s3_dict):,d} S3", flush=True)
    
    # 3. Candidate Generation (Blocking)
    print("\n[PHASE 3] Building Multi-Stage Composite Inverted Indices...", flush=True)
    t_idx = time.time()
    idx_s2_train = build_blocking_index(s2_dict.values())
    idx_s3_train = build_blocking_index(s3_dict.values())
    print(f"  Inverted indices built in {time.time()-t_idx:.2f}s (S2 keys: {len(idx_s2_train):,d}, S3 keys: {len(idx_s3_train):,d})", flush=True)
    
    print("  Generating candidate pairs for training entities...", flush=True)
    t_cand = time.time()
    train_cands = {}
    train_pairs = []
    features_list = []
    labels = []
    groups = []
    
    s1_dict = s1_train_df.set_index("entity_id").to_dict("index")
    
    for s1_id, s1_row in s1_dict.items():
        true_matches = gt_train.get(s1_id, set())
        keys = extract_blocking_keys(s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""))
        
        cands = set()
        for k in keys:
            cap = Config.MAX_NAME_TOK_POSTINGS if k[0] == "name_tok" else Config.MAX_COMPOSITE_POSTINGS
            p2 = idx_s2_train.get(k, [])
            p3 = idx_s3_train.get(k, [])
            if len(p2) <= cap:
                cands.update(p2)
            if len(p3) <= cap:
                cands.update(p3)
            if len(cands) >= Config.MAX_CANDIDATES_PER_S1:
                break
                
        train_cands[s1_id] = cands
        
        for cand_id in cands:
            cand_row = s2_dict.get(cand_id) or s3_dict.get(cand_id)
            if cand_row is None:
                continue
            feats = extract_pairwise_features(
                s1_row["business_name"], s1_row["business_address"], s1_row.get("country", ""),
                cand_row["business_name"], cand_row["business_address"], cand_row.get("country", ""),
                cand_id
            )
            is_match = 1 if cand_id in true_matches else 0
            train_pairs.append((s1_id, cand_id))
            features_list.append(feats)
            labels.append(is_match)
            groups.append(s1_id)
            
    recall, missed = blocking_recall(train_cands, gt_train)
    avg_cands = len(train_pairs) / len(s1_train_df) if len(s1_train_df) else 0
    print(f"  [BLOCKING METRIC] Candidate Recall: {recall*100:.2f}% | Avg Candidates/S1: {avg_cands:.1f} ({len(train_pairs):,d} pairs generated in {time.time()-t_cand:.2f}s)", flush=True)
    
    # 4. Model Training & Cross-Validation
    print(f"\n[PHASE 4] Entity-Level Cross-Validation & Threshold Optimization ({args.model_type})...", flush=True)
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(groups)
    
    if len(s1_train_df) >= 10:
        oof_probs, best_threshold, best_f05, cv_metrics = run_entity_level_cv(
            lambda: build_classifier(args.model_type),
            X, y, groups, train_pairs, gt_train, n_splits=5
        )
        print(f"  [5-FOLD CV RESULT]", flush=True)
        print(f"    Best Threshold: {best_threshold:.2f}", flush=True)
        print(f"    Macro F0.5:     {best_f05:.4f}", flush=True)
        print(f"    Precision:      {cv_metrics['precision']:.4f}", flush=True)
        print(f"    Recall:         {cv_metrics['recall']:.4f}", flush=True)
        print(f"    TP: {cv_metrics['tp']:,d} | FP: {cv_metrics['fp']:,d} | FN: {cv_metrics['fn']:,d}", flush=True)
    else:
        best_threshold = 0.50
        best_f05 = 1.0
        
    print(f"\n[PHASE 5] Training Final Matcher on Full Training Subsample...", flush=True)
    clf = train_matcher(X, y, model_type=args.model_type)
    model_save_path = os.path.join(output_dir, "entity_resolver_model.joblib")
    save_model(clf, model_save_path)
    print(f"  Model saved to {model_save_path}", flush=True)
    
    # 5. Test Set Inference
    print("\n[PHASE 6] Loading Test Set & Executing Inference...", flush=True)
    if args.sample:
        test_s1 = load_tsv(paths["test_s1"])
        test_s2 = load_tsv(paths["test_s2"])
        test_s3 = load_tsv(paths["test_s3"])
        idx_s2_test = build_blocking_index(test_s2)
        idx_s3_test = build_blocking_index(test_s3)
        lookup_s2 = test_s2.set_index("entity_id").to_dict("index")
        lookup_s3 = test_s3.set_index("entity_id").to_dict("index")
    else:
        if args.test_sample_size is not None:
            print(f"  Reading sample of {args.test_sample_size:,d} test entities...", flush=True)
            test_s1 = load_tsv(paths["test_s1"], nrows=args.test_sample_size)
            # For test sample, build index over first 150k of test S2/S3
            test_s2 = load_tsv(paths["test_s2"], nrows=150000)
            test_s3 = load_tsv(paths["test_s3"], nrows=150000)
        else:
            print(f"  Reading ALL test entities from test_source1.tsv...", flush=True)
            test_s1 = load_tsv(paths["test_s1"])
            test_s2 = load_tsv(paths["test_s2"])
            test_s3 = load_tsv(paths["test_s3"])
            
        print(f"  Building Test Inverted Indices (S2: {len(test_s2):,d} records, S3: {len(test_s3):,d} records)...", flush=True)
        t_tidx = time.time()
        idx_s2_test = build_blocking_index(test_s2)
        idx_s3_test = build_blocking_index(test_s3)
        print(f"  Test indices built in {time.time()-t_tidx:.2f}s", flush=True)
        lookup_s2 = test_s2.set_index("entity_id").to_dict("index")
        lookup_s3 = test_s3.set_index("entity_id").to_dict("index")
        
    test_matches, test_candidates = run_chunked_inference(
        clf, test_s1, idx_s2_test, idx_s3_test,
        lookup_s2, lookup_s3,
        threshold=best_threshold,
        chunk_size=50000
    )
    
    # 6. Writing Outputs
    print("\n[PHASE 7] Writing Output TSVs...", flush=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    test_s1_source = paths["test_s1"] if not args.sample else list(test_s1["entity_id"])
    write_results_tsv(matching_path, test_s1_source, test_matches, "matched_entity_ids")
    write_results_tsv(candidate_path, test_s1_source, test_candidates, "candidate_entity_ids")
    
    # 7. Official Validation
    print("\n[PHASE 8] Executing Official Submission Validator...", flush=True)
    test_dir = os.path.dirname(paths["test_s1"])
    val_passed = run_official_validator(matching_path, candidate_path, test_dir)
    print(f"  Validation status: {'PASSED' if val_passed else 'WARNINGS/ISSUES'}", flush=True)
    
    # 8. Final Packaging
    print("\n[PHASE 9] Packaging Submission Package...", flush=True)
    zip_path = package_submission_zip(team_name=args.team_name)
    
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print(f"  PIPELINE COMPLETE in {total_elapsed/60:.2f} minutes ({total_elapsed:.1f}s)!", flush=True)
    print(f"  Matching Results: {matching_path}", flush=True)
    print(f"  Candidate Pairs:  {candidate_path}", flush=True)
    print(f"  Submission ZIP:   {zip_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
