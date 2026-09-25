"""
Central Configuration for Amazon ML Challenge 2026: Business Entity Resolution.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class Config:
    REPO_ROOT = REPO_ROOT
    # Directories
    DATA_DIR = os.path.join(REPO_ROOT, "dataset")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
    OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
    EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
    
    # Output Files
    MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    RESULTS_CSV = os.path.join(EXPERIMENTS_DIR, "results.csv")
    
    # Random Seed
    RANDOM_SEED = 42
    
    # Blocking Parameters
    MAX_NAME_TOK_POSTINGS = 180
    MAX_COMPOSITE_POSTINGS = 450
    MAX_CANDIDATES_PER_S1 = 80
    
    # Model & Metric Parameters
    BETA = 0.5  # F_0.5 metric (precision heavily weighted)
    DEFAULT_THRESHOLD = 0.62  # Optimized via 5-Fold Entity-Level CV
    
    # Fast Pre-filter Thresholds
    PREFILTER_NAME_SIM_CUTOFF = 0.35
    PREFILTER_ADDR_SIM_CUTOFF = 0.35
    
    # Model Hyperparameters (LightGBM)
    LGBM_PARAMS = {
        "n_estimators": 200,
        "learning_rate": 0.07,
        "max_depth": 7,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    # Model Hyperparameters (HistGradientBoosting fallback)
    HGB_PARAMS = {
        "max_depth": 7,
        "learning_rate": 0.07,
        "max_iter": 200,
        "random_state": RANDOM_SEED,
    }
