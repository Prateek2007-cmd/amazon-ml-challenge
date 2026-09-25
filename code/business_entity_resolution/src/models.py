"""
Classification Models and Precision-First Decision System for Business Entity Resolution.
Trained specifically against hard negative blocking pairs and optimized for F0.5.
"""
import os
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from evaluate import macro_f_beta
from config import Config

def build_classifier(model_type="lightgbm", random_state=Config.RANDOM_SEED):
    """Instantiates the specified gradient boosting model with competition parameters."""
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(**Config.LGBM_PARAMS)
    elif model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**Config.HGB_PARAMS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_matcher(X_train, y_train, model_type="lightgbm"):
    """Fits the classifier on candidate pairs and returns the trained model."""
    clf = build_classifier(model_type=model_type)
    clf.fit(X_train, y_train)
    return clf

def sweep_threshold_f05(clf, X_val, val_pairs, ground_truth, min_t=0.20, max_t=0.96, step=0.02):
    """
    Fine-grained threshold sweep directly maximizing macro F_0.5 on validation split.
    val_pairs: list of (s1_id, cand_id) row-aligned with X_val
    ground_truth: {s1_id: set(matched_ids)} covering all S1 validation entities.
    """
    probs = clf.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    
    thresholds = np.arange(min_t, max_t, step)
    best_t = Config.DEFAULT_THRESHOLD
    best_f05 = -1.0
    best_prec, best_rec = 0.0, 0.0
    
    results = []
    
    for t in thresholds:
        t = round(float(t), 2)
        preds = {}
        for (s1_id, cand_id), p in zip(val_pairs, probs):
            if p >= t:
                preds.setdefault(s1_id, set()).add(cand_id)
                
        # Ensure singletons have an entry
        for s1_id in ground_truth:
            preds.setdefault(s1_id, set())
            
        score = macro_f_beta(ground_truth, preds, beta=Config.BETA)
        results.append((t, score))
        
        if score > best_f05:
            best_f05 = score
            best_t = t
            
    return best_t, best_f05, results

def save_model(clf, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)

def load_model(path):
    return joblib.load(path)
