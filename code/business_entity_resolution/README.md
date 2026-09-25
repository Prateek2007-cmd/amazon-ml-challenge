# Business Entity Resolution — pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv`
end to end: normalize → block (candidate generation) → pairwise
features → classifier → F0.5-tuned threshold → output.

## Setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

## Smoke test (no real data needed)

Confirms the pipeline itself works, using a tiny synthetic fixture at
`dataset/sample/` — six S1 entities, deliberate name/address noise,
and one non-training-set country (France) to exercise the open-set
country handling.

```bash
cd src
python3 pipeline.py --sample
```

Should finish in a couple seconds and print a blocking recall and
validation F0.5, both near 1.0 on this tiny fixture (not representative
of real-data performance — it's a correctness check, not a benchmark).

## Real run

Drop the official challenge files into, relative to the repo root:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

Then, from `src/`:

```bash
python3 pipeline.py
```

Watch the printed **blocking recall** number first — if it's
meaningfully below ~0.9, widen the blocking keys in `blocking.py`
before trusting anything downstream; a true match that never becomes
a candidate can't be recovered by the classifier, whatever else you
improve.

Outputs land in `output/matching_results.tsv` and
`output/candidate_pairs.tsv` at the repo root.

## Validate before submitting

```bash
cd ../../..   # repo root
python3 utils/local_validator.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

This is **not** the official validator — it's a fast local proxy for
the same rules. Still run the real `utils/validate_submission.py` from
the official challenge kit before every leaderboard upload.

## Module map

| File | Responsibility |
|---|---|
| `normalize.py` | Name/address cleanup — accent folding, legal-suffix and street-type canonicalization. Pattern-based, not a fixed country vocabulary lookup. |
| `blocking.py` | Candidate generation (token-based blocking keys) + `blocking_recall()` to measure your recall ceiling. |
| `features.py` | Pairwise similarity features (Levenshtein, token-sort, Jaccard, numeric-token overlap, country match). |
| `model.py` | `HistGradientBoostingClassifier` training + `sweep_threshold()`, which picks the threshold that maximizes **F0.5** specifically, not F1 or accuracy. |
| `evaluate.py` | The exact scoring metric from the problem statement — verified against its own worked example (`python3 evaluate.py`). |
| `pipeline.py` | Wires all of the above together; the only file you run directly. |

## Extending this

Everything here is a working baseline, not a finished solution:

- **Blocking** (`blocking.py`): the three key types (`tok0`, `tok2`,
  `prefix3`) are a starting point. Consider character n-grams or
  MinHash/LSH if candidate sets get too large on the real data, or
  additional keys if recall is below target.
- **Features** (`features.py`): add TF-IDF cosine similarity, phonetic
  encodings (Soundex/Metaphone), or address-component-level parsing
  once you've looked at real noise patterns.
- **Model** (`model.py`): swap in XGBoost/LightGBM by changing
  `build_classifier()` — the training/threshold code doesn't care
  which classifier it wraps, as long as it exposes `.fit()` and
  `.predict_proba()`.
