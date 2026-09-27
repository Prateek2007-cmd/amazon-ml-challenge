import pandas as pd

df = pd.read_csv('experiments/final_challenger/artifacts/error_ledger.csv')
print('Error ledger shape:', df.shape)
print('\nError types:')
print(df['error_type'].value_counts())

fn_matcher = df[df['error_type'] == 'FN_MATCHER_REJECTED']
print('\nMatcher misses count:', len(fn_matcher))

bins = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
cats = pd.cut(fn_matcher['model_probability'], bins=bins)
print('\nMatcher Miss Probability Bands:')
print(cats.value_counts().sort_index())

print('\nFN Matcher stats:')
print('Mean name_sim:', fn_matcher['name_similarity'].mean())
print('Mean addr_sim:', fn_matcher['address_similarity'].mean())
print('Mean semantic_cosine:', fn_matcher['semantic_cosine'].mean())

fp = df[df['error_type'] == 'FP_FALSE_POSITIVE_MERGE']
print('\nFP stats (count = %d):' % len(fp))
print('Mean name_sim:', fp['name_similarity'].mean())
print('Mean addr_sim:', fp['address_similarity'].mean())
print('Mean semantic_cosine:', fp['semantic_cosine'].mean())
print('Mean model_probability:', fp['model_probability'].mean())
