import os
from itertools import islice

countries_order = []
with open('dataset/test/test_source1.tsv', encoding='utf-8') as f:
    f.readline()
    prev_c = None
    runs = []
    current_run_len = 0
    for line in f:
        c = line.strip().split('\t')[3]
        if c != prev_c:
            if prev_c is not None:
                runs.append((prev_c, current_run_len))
            prev_c = c
            current_run_len = 1
        else:
            current_run_len += 1
    runs.append((prev_c, current_run_len))

print(f"Number of country blocks: {len(runs)}")
for c, count in runs[:10]:
    print(f"  {c}: {count:,d}")
