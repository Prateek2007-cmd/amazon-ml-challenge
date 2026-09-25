import sys, os
sys.path.insert(0, "code/business_entity_resolution/src")
from submission import package_submission_zip, run_official_validator
from config import Config

val_ok = run_official_validator(Config.MATCHING_OUTPUT, Config.CANDIDATE_OUTPUT, Config.TEST_DIR)
print(f"Official Validator returned: {'PASS' if val_ok else 'FAIL'}")
zip_path = package_submission_zip(team_name="Antigravity_ML")
print(f"Final Package created at: {zip_path}")
