import os, sys
from collections import Counter
import unicodedata

repo_root = os.path.abspath(".")
sys.path.insert(0, os.path.join(repo_root, "code", "business_entity_resolution", "src"))
from data_loader import get_data_paths

def detect_script(text):
    if not text:
        return "empty"
    scripts = Counter()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            script = name.split()[0] if name else "UNKNOWN"
            scripts[script] += 1
    if not scripts:
        return "non_alpha"
    return scripts.most_common(1)[0][0]

paths = get_data_paths(is_sample=False)

for src_name, p in [
    ("train_s1", paths["train_s1"]), ("train_s2", paths["train_s2"]), ("train_s3", paths["train_s3"]),
    ("test_s1", paths["test_s1"]), ("test_s2", paths["test_s2"]), ("test_s3", paths["test_s3"])
]:
    c = Counter()
    with open(p, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        name_idx = header.index("business_name")
        for i, line in enumerate(f):
            if i >= 50000: break
            parts = line.rstrip("\n").split("\t")
            if len(parts) > name_idx:
                c[detect_script(parts[name_idx])] += 1
    print(f"\n{src_name} top scripts (first 50k):")
    for s, cnt in c.most_common(6):
        print(f"  {s}: {cnt} ({cnt/500:.1f}%)")

