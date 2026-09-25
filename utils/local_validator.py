"""
NOT the official challenge validator.

This is a lightweight local sanity check implementing the rules
stated in the problem statement, so you can catch obvious formatting
bugs before running the *real* utils/validate_submission.py that ships
in the official challenge kit (student_resource/) -- that one is
authoritative; this one is just a fast local proxy for it.

Checks:
  - every Source1 test entity appears exactly once
  - no rows referencing Source1 ids outside the test set
  - no duplicate source1_entity_id rows
  - matched IDs only reference S2-/S3- prefixed ids that exist in the
    provided test source files
  - no duplicate ids within a single row's id list
  - (if --candidate is given) every matched id also appears as a
    candidate for that entity

Usage (mirrors the official script's arguments):
    python3 utils/local_validator.py \\
        --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir dataset/test
"""
import argparse
import csv
import sys


def load_id_list_tsv(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            id_list = row[1].split(",") if len(row) > 1 and row[1] else []
            rows[s1_id] = id_list
    return rows


def load_valid_ids(*paths):
    valid = set()
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row:
                    valid.add(row[0])
    return valid


def load_test_s1_ids(path):
    ids = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                ids.append(row[0])
    return ids


def check(matching_path, test_s1_path, test_s2_path, test_s3_path, candidate_path=None):
    issues = []

    with open(matching_path, newline="", encoding="utf-8") as f:
        raw_rows = [row[0] for row in csv.reader(f, delimiter="\t") if row][1:]
    if len(raw_rows) != len(set(raw_rows)):
        issues.append("duplicate source1_entity_id rows in matching_results.tsv")

    matching = load_id_list_tsv(matching_path)
    expected_s1 = load_test_s1_ids(test_s1_path)
    expected_set = set(expected_s1)
    valid_ids = load_valid_ids(test_s2_path, test_s3_path)

    if len(expected_s1) != len(expected_set):
        issues.append("test_source1.tsv itself has duplicate entity_ids -- check your data")

    missing = expected_set - set(matching.keys())
    if missing:
        issues.append(f"{len(missing)} Source1 test entities missing from submission "
                       f"(e.g. {sorted(missing)[:5]})")

    extra = set(matching.keys()) - expected_set
    if extra:
        issues.append(f"{len(extra)} rows reference Source1 ids not in the test set "
                       f"(e.g. {sorted(extra)[:5]})")

    for s1_id, id_list in matching.items():
        if len(id_list) != len(set(id_list)):
            issues.append(f"{s1_id}: duplicate ids within its own matched list")
        for mid in id_list:
            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                issues.append(f"{s1_id}: '{mid}' is not a Source2/Source3 id")
            elif mid not in valid_ids:
                issues.append(f"{s1_id}: '{mid}' does not exist in the test set")

    if candidate_path:
        candidates = load_id_list_tsv(candidate_path)
        for s1_id, id_list in matching.items():
            cand_set = set(candidates.get(s1_id, []))
            for mid in id_list:
                if mid not in cand_set:
                    issues.append(
                        f"{s1_id}: matched id '{mid}' never appeared as a candidate "
                        f"-- your final matches should be a subset of candidate_pairs.tsv"
                    )

    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--test-dir", required=True, help="folder containing test_source1/2/3.tsv")
    args = parser.parse_args()

    issues = check(
        args.matching,
        f"{args.test_dir}/test_source1.tsv",
        f"{args.test_dir}/test_source2.tsv",
        f"{args.test_dir}/test_source3.tsv",
        candidate_path=args.candidate,
    )

    if issues:
        print(f"FAIL -- {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        sys.exit(1)

    print("PASS (local checks only -- still run the OFFICIAL utils/validate_submission.py "
          "from your challenge kit before submitting; this script is a fast local proxy, not the authority)")
    sys.exit(0)


if __name__ == "__main__":
    main()
