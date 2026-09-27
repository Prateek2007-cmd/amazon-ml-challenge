"""
Package Phase 7 Submission ZIP: Antigravity_ML_submission_phase7.zip
Amazon ML Challenge 2026 - Business Entity Resolution

Ensures strict competition structure:
output/
  matching_results.tsv
  candidate_pairs.tsv
code/business_entity_resolution/src/
  [source files]
README.md
requirements.txt
Documentation_template.md
"""
import os, sys, zipfile, hashlib

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def package_submission():
    print("=" * 80)
    print("PACKAGING PHASE 7 FINAL SUBMISSION ZIP")
    print("=" * 80)

    phase7_dir = os.path.join(REPO_ROOT, "experiments", "final_challenger", "phase7")
    out_dir = os.path.join(phase7_dir, "output")
    matching_path = os.path.join(out_dir, "matching_results.tsv")
    candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")

    if not os.path.isfile(matching_path) or not os.path.isfile(candidate_path):
        print("ERROR: Output files matching_results.tsv or candidate_pairs.tsv do not exist yet!")
        return None

    zip_path = os.path.join(phase7_dir, "Antigravity_ML_submission_phase7.zip")
    if os.path.isfile(zip_path):
        os.remove(zip_path)

    print(f"Creating {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write(matching_path, "output/matching_results.tsv")
        zf.write(candidate_path, "output/candidate_pairs.tsv")
        print("  Added output/matching_results.tsv and output/candidate_pairs.tsv")

        # 2. Source code
        code_src_dir = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
        for root, dirs, files in os.walk(code_src_dir):
            if "__pycache__" in root:
                continue
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, REPO_ROOT)
                zf.write(full_path, rel_path)
        print("  Added code/business_entity_resolution/src/")

        # 3. Documentation and requirements
        for fname in ["README.md", "requirements.txt", "Documentation_template.md"]:
            fpath = os.path.join(REPO_ROOT, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
                print(f"  Added {fname}")

    zip_size = os.path.getsize(zip_path)
    zip_sha = compute_sha256(zip_path)
    print(f"\nPhase 7 Package Created: {zip_path}")
    print(f"Size: {zip_size:,d} bytes ({zip_size/(1024*1024):.2f} MB)")
    print(f"SHA256: {zip_sha}")

    # Verify contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"\nArchive contains {len(namelist)} files:")
        for name in namelist[:10]:
            print(f"  - {name}")
        if len(namelist) > 10:
            print(f"  ... and {len(namelist)-10} more files")

    return zip_path, zip_sha, zip_size

if __name__ == "__main__":
    package_submission()
