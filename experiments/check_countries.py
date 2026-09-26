import os

for name, p in [('test_s1', 'dataset/test/test_source1.tsv'), ('test_s2', 'dataset/test/test_source2.tsv'), ('test_s3', 'dataset/test/test_source3.tsv')]:
    countries = set()
    empty = 0
    total = 0
    with open(p, encoding='utf-8') as f:
        f.readline()
        for line in f:
            total += 1
            parts = line.strip().split('\t')
            c = parts[3].strip() if len(parts) > 3 else ''
            if not c: empty += 1
            else: countries.add(c)
    print(name, f"total={total:,d}, countries={countries}, empty={empty:,d}")
