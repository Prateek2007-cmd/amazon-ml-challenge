"""
Exact implementation of the challenge's scoring metric: macro-averaged
F_0.5 per Source1 entity. Use this to self-score your validation split
before ever touching the leaderboard.

Verified against the problem statement's worked example:
  predicted [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812]
  -> precision 0.667, recall 1.0, F0.5 = 0.714
"""


def compute_entity_metrics(true_set, pred_set, beta=0.5):
    """
    Returns (f_beta, precision, recall, tp, fp, fn) for one Source1 entity.
    Exact match to official challenge specification.
    """
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if not pred_set:
        return 0.0, 0.0, 0.0, 0, 0, len(true_set)
    if not true_set:
        # True is empty, but we predicted matches -> false merge
        return 0.0, 0.0, 0.0, 0, len(pred_set), 0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if precision == 0.0 and recall == 0.0:
        return 0.0, 0.0, 0.0, tp, fp, fn

    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    f_beta = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
    return f_beta, precision, recall, tp, fp, fn

def macro_f_beta(ground_truth: dict, predictions: dict, beta=0.5) -> float:
    """Computes official macro F0.5 over all entities in ground_truth."""
    if not ground_truth:
        return 0.0
    scores = [
        compute_entity_metrics(true_set, predictions.get(s1_id, set()), beta=beta)[0]
        for s1_id, true_set in ground_truth.items()
    ]
    return sum(scores) / len(scores)

def compute_comprehensive_metrics(ground_truth: dict, predictions: dict, beta=0.5, entity_countries: dict = None):

    """
    Computes official macro F0.5 alongside macro precision, macro recall,
    micro metrics, and subpopulation breakdowns (cardinality, country).
    """
    if not ground_truth:
        return {}

    entity_f_scores = []
    entity_precisions = []
    entity_recalls = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Subgroup trackers
    card_groups = {"zero": [], "one": [], "multi": []}
    country_groups = {}

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f_score, prec, rec, tp, fp, fn = compute_entity_metrics(true_set, pred_set, beta=beta)

        entity_f_scores.append(f_score)
        entity_precisions.append(prec)
        entity_recalls.append(rec)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Cardinality group
        card = len(true_set)
        if card == 0:
            c_tag = "zero"
        elif card == 1:
            c_tag = "one"
        else:
            c_tag = "multi"
        card_groups[c_tag].append((f_score, prec, rec))

        # Country group
        if entity_countries and s1_id in entity_countries:
            country = entity_countries[s1_id]
            if country not in country_groups:
                country_groups[country] = []
            country_groups[country].append((f_score, prec, rec))

    macro_f05 = sum(entity_f_scores) / len(entity_f_scores)
    macro_prec = sum(entity_precisions) / len(entity_precisions)
    macro_rec = sum(entity_recalls) / len(entity_recalls)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    beta_sq = beta * beta
    micro_denom = beta_sq * micro_prec + micro_rec
    micro_f05 = (1 + beta_sq) * micro_prec * micro_rec / micro_denom if micro_denom > 0 else 0.0

    def summarize_group(tuples_list):
        if not tuples_list:
            return {"count": 0, "macro_f05": 0.0, "precision": 0.0, "recall": 0.0}
        n = len(tuples_list)
        return {
            "count": n,
            "macro_f05": sum(x[0] for x in tuples_list) / n,
            "precision": sum(x[1] for x in tuples_list) / n,
            "recall": sum(x[2] for x in tuples_list) / n,
        }

    card_summary = {k: summarize_group(v) for k, v in card_groups.items()}
    country_summary = {k: summarize_group(v) for k, v in country_groups.items()}

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f05": micro_f05,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "cardinality": card_summary,
        "country": country_summary,
    }

if __name__ == "__main__":
    # Sanity check against the problem statement's worked example.
    truth = {"S1-00001": {"S2-00047", "S3-00812"}}
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    score = macro_f_beta(truth, pred)
    expected = 0.714
    print(f"Worked example score: {score:.3f} (expected ~{expected})")
    assert abs(score - expected) < 0.001, "evaluate.py does not match the spec's example!"
    
    # Test zero-match behavior
    truth_zero = {"S1-00002": set()}
    pred_zero_correct = {"S1-00002": set()}
    pred_zero_wrong = {"S1-00002": {"S2-00001"}}
    assert macro_f_beta(truth_zero, pred_zero_correct) == 1.0
    assert macro_f_beta(truth_zero, pred_zero_wrong) == 0.0
    print("evaluate.py matches official singleton/zero-match specification!")

