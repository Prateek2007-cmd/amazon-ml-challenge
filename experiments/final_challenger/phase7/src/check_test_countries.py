import pandas as pd
from collections import Counter

for name, path in [
    ("test_s1", "dataset/test/test_source1.tsv"),
    ("test_s2", "dataset/test/test_source2.tsv"),
    ("test_s3", "dataset/test/test_source3.tsv"),
]:
    countries = Counter()
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        c_idx = header.index("country")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            c = parts[c_idx].strip().upper() if len(parts) > c_idx else ""
            countries[c] += 1
    print(f"\n{name} country counts:")
    for c, cnt in countries.most_common():
        print(f"  {c:15s}: {cnt:9,d}")
