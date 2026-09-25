"""
Exact implementation of the challenge's scoring metric: macro-averaged
F_0.5 per Source1 entity. Use this to self-score your validation split
before ever touching the leaderboard.

Verified against the problem statement's worked example:
  predicted [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812]
  -> precision 0.667, recall 1.0, F0.5 = 0.714
"""


def f_beta_for_entity(true_set, pred_set, beta=0.5) -> float:
    """F_beta for one Source1 entity.

    Special cases (matches the challenge's stated scoring):
      - true_set empty, pred_set empty      -> 1.0 (correct singleton)
      - true_set empty, pred_set non-empty  -> 0.0 (false merge -- no
        partial credit on a singleton)
      - true_set non-empty, pred_set empty  -> 0.0 (missed everything)
    """
    if not true_set and not pred_set:
        return 1.0
    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set) if true_set else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f_beta(ground_truth: dict, predictions: dict, beta=0.5) -> float:
    """ground_truth, predictions: dict s1_id -> set of matched ids.

    Every s1_id in ground_truth is scored even if `predictions` has no
    entry for it (treated as an empty prediction) -- mirrors the real
    leaderboard, where a missing entity is not silently skipped.
    """
    if not ground_truth:
        return 0.0
    scores = [
        f_beta_for_entity(true_set, predictions.get(s1_id, set()), beta=beta)
        for s1_id, true_set in ground_truth.items()
    ]
    return sum(scores) / len(scores)


if __name__ == "__main__":
    # Sanity check against the problem statement's worked example.
    truth = {"S1-00001": {"S2-00047", "S3-00812"}}
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    score = macro_f_beta(truth, pred)
    expected = 0.714
    print(f"Worked example score: {score:.3f} (expected ~{expected})")
    assert abs(score - expected) < 0.001, "evaluate.py does not match the spec's example!"
    print("evaluate.py matches the problem statement's worked example.")
