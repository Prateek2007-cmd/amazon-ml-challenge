"""
Cross-Validation and Evaluation System for Business Entity Resolution.
Implements Source-1 Entity-Level GroupKFold to strictly prevent data leakage.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from evaluate import macro_f_beta, compute_comprehensive_metrics
from config import Config

def run_entity_level_cv(clf_builder, X, y, groups, pairs, ground_truth, entity_countries=None, n_splits=5):
    """
    Performs out-of-fold entity-level cross-validation and threshold optimization.
    Returns:
        oof_probs: array of out-of-fold prediction probabilities
        best_threshold: threshold maximizing macro F0.5
        best_f05: highest macro F0.5 score
        metrics: dict of macro/micro metrics, cardinality, and country breakdowns
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = clf_builder()
        clf.fit(X_train, y_train)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
        
    # Sweep threshold on OOF predictions directly targeting Macro F0.5
    thresholds = np.arange(0.20, 0.96, 0.02)
    best_t = Config.DEFAULT_THRESHOLD
    best_metrics = None
    best_f05 = -1.0
    
    for t in thresholds:
        t = round(float(t), 2)
        preds = defaultdict(set)
        for (s1_id, cand_id), p in zip(pairs, oof_probs):
            if p >= t:
                preds[s1_id].add(cand_id)
                
        metrics = compute_comprehensive_metrics(
            ground_truth, preds, beta=Config.BETA, entity_countries=entity_countries
        )
        
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_t = t
            best_metrics = metrics
            
    best_metrics["best_threshold"] = best_t
    best_metrics["oof_macro_f05"] = best_f05
    # Backwards compatibility keys
    best_metrics["precision"] = best_metrics["micro_precision"]
    best_metrics["recall"] = best_metrics["micro_recall"]
    return oof_probs, best_t, best_f05, best_metrics

