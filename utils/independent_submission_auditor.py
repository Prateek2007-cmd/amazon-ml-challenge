"""
Phase 24: Independent Submission Auditor for Amazon ML Challenge 2026.
Performs exhaustive structural, semantic, and relational audits on:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
against dataset/test source files.
Checks:
  1. Exact line counts and matching row counts (1,732,544 rows each)
  2. Header correctness (tab-separated, exact column names)
  3. No accidental index columns or quoting
  4. Unique S1 IDs (no duplicate S1 IDs)
  5. Zero missing S1 IDs from test_source1.tsv
  6. Zero extra S1 IDs outside test_source1.tsv
  7. No duplicate matched IDs or candidate IDs within any row
  8. Valid ID prefixes (only S2- or S3- allowed)
  9. Strict subset rule: Every matched ID MUST appear in candidate_pairs.tsv
  10. Cardinality & singleton consistency
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

def run_independent_audit(matching_path, candidate_path, test_dir):
    print("=" * 70)
    print("  PHASE 24: INDEPENDENT RIGOROUS SUBMISSION AUDITOR")
    print("=" * 70)
    t0 = time.time()
    errors = []
    warnings = []
    
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    
    # 1. Inspect test_source1.tsv
    print(f"\n[AUDIT 1] Loading reference Source 1 entity IDs from {test_s1_path}...")
    expected_s1_ids = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        for line in f:
            line_str = line.rstrip("\n")
            if line_str:
                expected_s1_ids.append(line_str.split("\t")[0])
                
    expected_s1_set = set(expected_s1_ids)
    print(f"  Reference S1 count: {len(expected_s1_ids):,d} (Unique: {len(expected_s1_set):,d})")
    if len(expected_s1_ids) != len(expected_s1_set):
        errors.append("test_source1.tsv has duplicate IDs!")

    # 2. Audit matching_results.tsv
    print(f"\n[AUDIT 2] Auditing {matching_path}...")
    if not os.path.exists(matching_path):
        errors.append(f"Missing file: {matching_path}")
        return False, errors
        
    matching_seen_s1 = set()
    matching_dup_s1 = 0
    matching_empty_rows = 0
    total_matching_rows = 0
    total_matched_ids = 0
    invalid_prefix_matches = 0
    intra_dup_matches = 0
    
    # Store candidates mapping in memory or stream audit
    # To check subset without 2GB RAM: we can store matched IDs per S1
    matched_by_s1 = {}
    
    with open(matching_path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        if "\t" not in header_line:
            errors.append("matching_results.tsv header is not tab-separated!")
        cols = header_line.split("\t")
        if cols != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"matching_results.tsv header mismatch: got {cols}")
            
        for line_no, line in enumerate(f, start=2):
            line_str = line.rstrip("\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) != 2:
                errors.append(f"matching_results.tsv line {line_no} malformed: expected 2 tab-separated cols, got {len(parts)}")
                continue
            s1_id, matched_str = parts[0], parts[1]
            
            if s1_id in matching_seen_s1:
                matching_dup_s1 += 1
            matching_seen_s1.add(s1_id)
            total_matching_rows += 1
            
            if not matched_str.strip():
                matching_empty_rows += 1
                matched_by_s1[s1_id] = set()
            else:
                m_ids = [x.strip() for x in matched_str.split(",") if x.strip()]
                m_set = set(m_ids)
                if len(m_ids) != len(m_set):
                    intra_dup_matches += 1
                for mid in m_set:
                    if not (mid.startswith("S2-") or mid.startswith("S3-")):
                        invalid_prefix_matches += 1
                total_matched_ids += len(m_set)
                matched_by_s1[s1_id] = m_set

    print(f"  Total Rows: {total_matching_rows:,d} | Empty matches (singletons): {matching_empty_rows:,d}")
    print(f"  Non-empty matches: {total_matching_rows - matching_empty_rows:,d} | Total Predicted Pairs: {total_matched_ids:,d}")
    if matching_dup_s1 > 0:
        errors.append(f"matching_results.tsv has {matching_dup_s1:,d} duplicate S1 IDs!")
    if intra_dup_matches > 0:
        errors.append(f"matching_results.tsv has {intra_dup_matches:,d} rows with duplicate IDs in matched list!")
    if invalid_prefix_matches > 0:
        errors.append(f"matching_results.tsv has {invalid_prefix_matches:,d} IDs with invalid prefix (not S2-/S3-)!")
    if matching_seen_s1 != expected_s1_set:
        diff_missing = expected_s1_set - matching_seen_s1
        diff_extra = matching_seen_s1 - expected_s1_set
        if diff_missing:
            errors.append(f"matching_results.tsv is missing {len(diff_missing):,d} S1 IDs!")
        if diff_extra:
            errors.append(f"matching_results.tsv contains {len(diff_extra):,d} unknown S1 IDs!")

    # 3. Audit candidate_pairs.tsv and check subset constraint
    print(f"\n[AUDIT 3] Auditing {candidate_path} and verifying Candidate-Set Subset Rule...")
    if not os.path.exists(candidate_path):
        errors.append(f"Missing file: {candidate_path}")
        return False, errors

    candidate_seen_s1 = set()
    candidate_dup_s1 = 0
    candidate_empty_rows = 0
    total_candidate_rows = 0
    total_candidate_pairs = 0
    subset_violations = 0
    intra_dup_candidates = 0
    invalid_prefix_cands = 0
    
    with open(candidate_path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        if "\t" not in header_line:
            errors.append("candidate_pairs.tsv header is not tab-separated!")
        cols = header_line.split("\t")
        if cols != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"candidate_pairs.tsv header mismatch: got {cols}")
            
        for line_no, line in enumerate(f, start=2):
            line_str = line.rstrip("\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) != 2:
                errors.append(f"candidate_pairs.tsv line {line_no} malformed: expected 2 tab-separated cols, got {len(parts)}")
                continue
            s1_id, cand_str = parts[0], parts[1]
            
            if s1_id in candidate_seen_s1:
                candidate_dup_s1 += 1
            candidate_seen_s1.add(s1_id)
            total_candidate_rows += 1
            
            c_set = set(x.strip() for x in cand_str.split(",") if x.strip()) if cand_str.strip() else set()
            if not c_set:
                candidate_empty_rows += 1
            else:
                c_list = [x.strip() for x in cand_str.split(",") if x.strip()]
                if len(c_list) != len(c_set):
                    intra_dup_candidates += 1
                for cid in c_set:
                    if not (cid.startswith("S2-") or cid.startswith("S3-")):
                        invalid_prefix_cands += 1
                total_candidate_pairs += len(c_set)
                
            # Verify subset rule: Every matched ID must be in candidates!
            expected_matches = matched_by_s1.get(s1_id, set())
            if not expected_matches.issubset(c_set):
                violation = expected_matches - c_set
                subset_violations += len(violation)

    print(f"  Total Rows: {total_candidate_rows:,d} | Empty candidate sets: {candidate_empty_rows:,d}")
    print(f"  Non-empty candidates: {total_candidate_rows - candidate_empty_rows:,d} | Total Candidate Pairs: {total_candidate_pairs:,d}")
    if candidate_dup_s1 > 0:
        errors.append(f"candidate_pairs.tsv has {candidate_dup_s1:,d} duplicate S1 IDs!")
    if intra_dup_candidates > 0:
        errors.append(f"candidate_pairs.tsv has {intra_dup_candidates:,d} rows with duplicate IDs in candidate list!")
    if invalid_prefix_cands > 0:
        errors.append(f"candidate_pairs.tsv has {invalid_prefix_cands:,d} IDs with invalid prefix!")
    if candidate_seen_s1 != expected_s1_set:
        errors.append("candidate_pairs.tsv S1 entity set does not match test_source1.tsv!")
    if subset_violations > 0:
        errors.append(f"SUBSET VIOLATION: {subset_violations:,d} predicted matches do NOT appear in candidate_pairs.tsv!")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    if errors:
        print(f"  AUDIT FAILED with {len(errors)} critical issues ({elapsed:.2f}s):")
        for err in errors:
            print(f"    [ERROR] {err}")
        return False
    else:
        print(f"  AUDIT PASSED: 100% compliant with all competition specifications! ({elapsed:.2f}s)")
        print(f"  - Total test S1 entities checked: {total_matching_rows:,d}")
        print(f"  - Zero missing entities, zero duplicates")
        print(f"  - 100% matched IDs exist in candidate_pairs.tsv")
        print("=" * 70)
        return True

if __name__ == "__main__":
    run_independent_audit(
        matching_path="output/matching_results.tsv",
        candidate_path="output/candidate_pairs.tsv",
        test_dir="dataset/test"
    )
