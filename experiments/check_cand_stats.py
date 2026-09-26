import os

for path in ['output/candidate_pairs.tsv', 'champion_backup_09579/candidate_pairs.tsv', 'output/matching_results.tsv', 'champion_backup_09579/matching_results.tsv']:
    if os.path.exists(path):
        non_empty = 0
        total = 0
        with open(path, encoding='utf-8') as f:
            f.readline()
            for line in f:
                total += 1
                parts = line.strip().split('\t')
                if len(parts) > 1 and parts[1]:
                    non_empty += 1
        print(f"{path}: size={os.path.getsize(path):,d}, total={total:,d}, non_empty={non_empty:,d}")
