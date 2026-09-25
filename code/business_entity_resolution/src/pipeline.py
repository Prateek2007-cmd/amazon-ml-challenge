"""
End-to-end pipeline: load -> normalize -> block -> features ->
train -> threshold sweep -> predict -> write outputs.

Run from anywhere, from this `src/` directory:
    python3 pipeline.py                         # uses dataset/train + dataset/test
    python3 pipeline.py --data-dir dataset/sample   # smoke test on the tiny fixture

Expects, under --data-dir (default: dataset, resolved relative to the
repo root):
    train/train_source1.tsv
    train/train_source2.tsv
    train/train_source3.tsv
    train/train_ground_truth.tsv
    test/test_source1.tsv
    test/test_source2.tsv
    test/test_source3.tsv

--data-dir dataset/sample expects the same 7 filenames directly inside
it (no train/ or test/ subfolders) -- see dataset/sample/ itself.

Writes, under --output-dir (default: output/):
    matching_results.tsv
    candidate_pairs.tsv
"""
import argparse
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import (
    build_blocking_index,
    generate_candidates,
    merge_candidate_sets,
    blocking_recall,
)
from features import pair_features, FEATURE_COLUMNS
from model import train as train_model, sweep_threshold

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def parse_ground_truth(gt_df):
    gt = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "")
        gt[s1_id] = set(x for x in str(raw).split(",") if x)
    return gt


def build_candidates_for(s1_df, s2_df, s3_df):
    idx2 = build_blocking_index(s2_df)
    idx3 = build_blocking_index(s3_df)
    cand2 = generate_candidates(s1_df, idx2)
    cand3 = generate_candidates(s1_df, idx3)
    return merge_candidate_sets(cand2, cand3)


def build_lookup(*dfs):
    lookup = {}
    for df in dfs:
        for _, row in df.iterrows():
            lookup[row["entity_id"]] = row
    return lookup


def build_feature_rows(s1_df, candidates, lookup, ground_truth=None):
    """Returns (pairs, X, y). y is None if ground_truth is None."""
    pairs, feat_rows = [], []
    labels = [] if ground_truth is not None else None

    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        cand_ids = candidates.get(s1_id, set())
        true_set = ground_truth.get(s1_id, set()) if ground_truth is not None else set()
        for cand_id in cand_ids:
            cand_row = lookup.get(cand_id)
            if cand_row is None:
                continue
            feats = pair_features(
                s1_row["business_name"], s1_row["business_address"], s1_row["country"],
                cand_row["business_name"], cand_row["business_address"], cand_row["country"],
            )
            pairs.append((s1_id, cand_id))
            feat_rows.append(feats)
            if labels is not None:
                labels.append(1 if cand_id in true_set else 0)

    X = pd.DataFrame(feat_rows, columns=FEATURE_COLUMNS).fillna(0.0)
    return pairs, X, labels


def write_id_list_tsv(path, s1_ids, id_lists, id_col_name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for s1_id in s1_ids:
            ids = sorted(id_lists.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(ids)}\n")


def resolve_paths(data_dir, output_dir):
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(REPO_ROOT, data_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    return data_dir, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset",
                         help="folder containing train/ and test/ subfolders "
                              "(or, for the sample fixture, the 7 TSVs directly)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--sample", action="store_true",
                         help="shorthand for --data-dir dataset/sample, flat layout")
    args = parser.parse_args()

    data_dir = "dataset/sample" if args.sample else args.data_dir
    data_dir, output_dir = resolve_paths(data_dir, args.output_dir)
    flat_layout = args.sample or os.path.exists(os.path.join(data_dir, "train_source1.tsv"))

    def p(split, name):
        return os.path.join(data_dir, name) if flat_layout else os.path.join(data_dir, split, name)

    print(f"Data dir: {data_dir}  (flat layout: {flat_layout})")

    print("Loading training data...")
    train_s1 = load_tsv(p("train", "train_source1.tsv"))
    train_s2 = load_tsv(p("train", "train_source2.tsv"))
    train_s3 = load_tsv(p("train", "train_source3.tsv"))
    ground_truth = parse_ground_truth(load_tsv(p("train", "train_ground_truth.tsv")))

    print("Building training candidates (blocking)...")
    train_candidates = build_candidates_for(train_s1, train_s2, train_s3)

    recall, missed = blocking_recall(train_candidates, ground_truth)
    print(f"Blocking recall on training set: {recall:.4f} ({len(missed)} true matches missed)")
    if recall < 0.9:
        print("WARNING: blocking recall is low -- widen your blocking keys "
              "in blocking.py before trusting the classifier's numbers.")

    print("Building feature rows for training pairs...")
    train_lookup = build_lookup(train_s2, train_s3)
    pairs, X, y = build_feature_rows(train_s1, train_candidates, train_lookup, ground_truth)
    pos = sum(y) if y else 0
    print(f"{len(pairs)} candidate pairs, {pos} positive, {len(pairs) - pos} negative")

    print("Splitting Source1 entities into train/val (80/20)...")
    s1_ids = list(train_s1["entity_id"])
    random.Random(42).shuffle(s1_ids)
    split_point = max(1, int(len(s1_ids) * 0.8))
    train_ids = set(s1_ids[:split_point])
    val_ids = set(s1_ids[split_point:]) or set(s1_ids[-1:])  # guarantee non-empty val

    train_mask = [pid in train_ids for pid, _ in pairs]
    val_mask = [pid in val_ids for pid, _ in pairs]

    X_train = X[train_mask].reset_index(drop=True)
    y_train = [label for label, m in zip(y, train_mask) if m]
    X_val = X[val_mask].reset_index(drop=True)
    val_pairs = [pr for pr, m in zip(pairs, val_mask) if m]
    val_gt = {s1_id: matches for s1_id, matches in ground_truth.items() if s1_id in val_ids}

    print("Training classifier...")
    clf = train_model(X_train, y_train)

    print("Sweeping threshold for best F0.5 on validation split...")
    best_t, best_f05, _ = sweep_threshold(clf, X_val, val_pairs, val_gt)
    print(f"Best threshold: {best_t:.2f} -> validation F0.5: {best_f05:.4f}")

    print("Loading test data...")
    test_s1 = load_tsv(p("test", "test_source1.tsv"))
    test_s2 = load_tsv(p("test", "test_source2.tsv"))
    test_s3 = load_tsv(p("test", "test_source3.tsv"))

    print("Building test candidates (blocking)...")
    test_candidates = build_candidates_for(test_s1, test_s2, test_s3)

    print("Building feature rows for test pairs...")
    test_lookup = build_lookup(test_s2, test_s3)
    test_pairs, X_test, _ = build_feature_rows(test_s1, test_candidates, test_lookup)

    print("Scoring test pairs and applying threshold...")
    matches = {}
    if len(X_test):
        test_probs = clf.predict_proba(X_test)[:, 1]
        for (s1_id, cand_id), prob in zip(test_pairs, test_probs):
            if prob >= best_t:
                matches.setdefault(s1_id, set()).add(cand_id)

    test_s1_ids = list(test_s1["entity_id"])

    print("Writing outputs...")
    write_id_list_tsv(
        os.path.join(output_dir, "matching_results.tsv"),
        test_s1_ids, matches, "matched_entity_ids",
    )
    write_id_list_tsv(
        os.path.join(output_dir, "candidate_pairs.tsv"),
        test_s1_ids, test_candidates, "candidate_entity_ids",
    )
    print(f"Done. See {output_dir}/matching_results.tsv and {output_dir}/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
