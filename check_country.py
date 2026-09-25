import pandas as pd
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=5000, dtype=str).set_index('entity_id')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=50000, dtype=str).set_index('entity_id')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=50000, dtype=str).set_index('entity_id')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=5000, dtype=str, keep_default_na=False)

diff_country = 0
total_checked = 0
for _, r in gt.iterrows():
    s1_id = r['source1_entity_id']
    if s1_id not in s1.index:
        continue
    c1 = s1.loc[s1_id, 'country']
    matches = [m.strip() for m in r['matched_entity_ids'].split(',') if m.strip()]
    for m in matches:
        target = s2.loc[m, 'country'] if m in s2.index else (s3.loc[m, 'country'] if m in s3.index else None)
        if target is not None:
            total_checked += 1
            if target != c1:
                diff_country += 1

print(f'Total true matches checked: {total_checked}')
print(f'Different country count: {diff_country}')
