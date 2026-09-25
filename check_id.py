import pandas as pd
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=10)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10)
print('S1 head IDs:', list(s1['entity_id']))
print('GT head IDs:', list(gt['source1_entity_id']))

# Check intersection
s1_10k = set(pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=10000)['entity_id'])
gt_10k = set(pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)['source1_entity_id'])
print(f'Intersection size: {len(s1_10k & gt_10k)} / 10000')
