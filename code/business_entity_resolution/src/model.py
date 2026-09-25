"""
Pairwise match classifier + threshold selection tuned for F0.5 --
not F1, not accuracy. Since precision counts 2x in the scoring
formula, the F0.5-optimal threshold is usually stricter than 0.5.

Uses sklearn's HistGradientBoostingClassifier by default: ships with
scikit-learn, no extra native build deps to fight with mid-hackathon,
and comfortably satisfies the "MIT/Apache-licensed, <=8B params" final
model constraint. Swap in XGBoost/LightGBM via `build_classifier()` if
you prefer -- everything else here is classifier-agnostic.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import macro_f_beta


def build_classifier(random_state=42):
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        random_state=random_state,
    )


def train(X_train, y_train):
    clf = build_classifier()
    clf.fit(X_train, y_train)
    return clf


def sweep_threshold(clf, X_val, val_pairs, ground_truth, thresholds=None):
    """val_pairs: list of (s1_id, cand_id), row-aligned with X_val.
    ground_truth: dict s1_id -> set of true matched ids (should cover
    every S1 id in the validation split, including singletons).

    Returns (best_threshold, best_f05, [(threshold, f05), ...]).
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.96, 0.02)

    probs = clf.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])

    results = []
    best_threshold, best_f05 = 0.5, -1.0
    for t in thresholds:
        predictions = {}
        for (s1_id, cand_id), p in zip(val_pairs, probs):
            if p >= t:
                predictions.setdefault(s1_id, set()).add(cand_id)
        for s1_id in ground_truth:  # ensure singletons get an (empty) entry
            predictions.setdefault(s1_id, set())

        score = macro_f_beta(ground_truth, predictions, beta=0.5)
        results.append((float(t), score))
        if score > best_f05:
            best_f05, best_threshold = score, float(t)

    return best_threshold, best_f05, results
