import os
import hashlib

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

files_to_check = [
    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
    "Antigravity_ML_submission.zip",
    "Antigravity_ML_submission_optimized.zip",
    "champion_backup_09579/output/matching_results.tsv",
    "champion_backup_09579/output/candidate_pairs.tsv",
]

print("=" * 80)
print("FROZEN CHAMPION ARTIFACT SAFETY AUDIT (BEFORE PHASE 7)")
print("=" * 80)
for f in files_to_check:
    if os.path.exists(f):
        size = os.path.getsize(f)
        h = sha256_file(f)
        print(f"Path:   {f}")
        print(f"Size:   {size:,d} bytes ({size / (1024*1024):.2f} MB)")
        print(f"SHA256: {h}")
        print("-" * 80)
    else:
        print(f"Path:   {f} -> NOT FOUND")

# Also check test and train sizes
print("\nDATASET LINE COUNTS:")
for path in [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
]:
    if os.path.exists(path):
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f) - 1
        print(f"{path:40s}: {lines:8,d} rows ({size / (1024*1024):6.2f} MB)")
