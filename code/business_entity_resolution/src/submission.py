"""
Submission Generation, Formatting, and Packaging for Amazon ML Challenge 2026.
Generates matching_results.tsv, candidate_pairs.tsv, runs official validation,
and packages the final submission zip.
"""
import os
import zipfile
import subprocess
import sys
from config import Config

def write_results_tsv(path, s1_id_source, mapping_dict, col_name):
    """
    Writes a tab-separated results file.
    path: output filepath
    s1_id_source: either a list of IDs or path to test_source1.tsv
    mapping_dict: {s1_id: set(matched_or_candidate_ids)}
    col_name: 'matched_entity_ids' or 'candidate_entity_ids'
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out_f:
        out_f.write(f"source1_entity_id\t{col_name}\n")
        
        # If path provided, stream all test entities to ensure zero missing entities
        if isinstance(s1_id_source, str) and os.path.exists(s1_id_source):
            count = 0
            with open(s1_id_source, "r", encoding="utf-8") as in_f:
                in_f.readline()  # skip header
                for line in in_f:
                    line_str = line.rstrip("\n")
                    if not line_str:
                        continue
                    s1_id = line_str.split("\t")[0]
                    ids = sorted(list(mapping_dict.get(s1_id, set())))
                    out_f.write(f"{s1_id}\t{','.join(ids)}\n")
                    count += 1
            print(f"Wrote {count:,d} complete test rows to {path}")
        else:
            for s1_id in s1_id_source:
                ids = sorted(list(mapping_dict.get(s1_id, set())))
                out_f.write(f"{s1_id}\t{','.join(ids)}\n")
            print(f"Wrote {len(s1_id_source):,d} rows to {path}")

def run_official_validator(matching_path, candidate_path, test_dir):
    """Executes the official challenge validator."""
    validator_script = os.path.join(Config.REPO_ROOT, "utils", "validate_submission.py")
    if not os.path.exists(validator_script):
        print(f"Warning: Validator script not found at {validator_script}")
        return False
        
    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]
    print(f"\nRunning official validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0

def package_submission_zip(team_name="Antigravity_ML", zip_path=None):
    """
    Creates the final competition submission ZIP with required structure:
    output/
        matching_results.tsv
        candidate_pairs.tsv
    code/
        business_entity_resolution/
            src/
            README.md
            requirements.txt
    Documentation_template.md
    """
    if zip_path is None:
        zip_path = os.path.join(Config.REPO_ROOT, f"{team_name}_submission.zip")
        
    print(f"\nPackaging final submission ZIP: {zip_path}")
    repo_root = Config.REPO_ROOT
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        for fname in ("matching_results.tsv", "candidate_pairs.tsv"):
            fpath = os.path.join(repo_root, "output", fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=os.path.join("output", fname))
            else:
                print(f"Warning: Output file {fpath} missing for zip.")
                
        # 2. Documentation template
        doc_path = os.path.join(repo_root, "Documentation_template.md")
        if os.path.exists(doc_path):
            zf.write(doc_path, arcname="Documentation_template.md")
            
        # 3. Code files
        code_base = os.path.join(repo_root, "code", "business_entity_resolution")
        for root, dirs, files in os.walk(code_base):
            if "__pycache__" in root or ".git" in root or "checkpoints" in root:
                continue
            for file in files:
                if file.endswith((".pyc", ".pyo", ".pyd")):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_root)
                zf.write(full_path, arcname=rel_path)
                
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Successfully packaged {zip_path} ({zip_size_mb:.2f} MB).")
    return zip_path
