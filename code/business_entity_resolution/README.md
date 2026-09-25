# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

State-of-the-art Entity Resolution system for matching noisy business records across Source 1 (master/deduped reference), Source 2 (noisy), and Source 3 (noisy). Specifically engineered and optimized for the competition's precision-weighted **macro $F_{0.5}$** metric.

---

## Performance Summary

| Architecture / Model | Candidate Recall | Best $\tau$ | Out-of-Fold Macro $F_{0.5}$ | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-Stage Composite + LightGBM** | **95.99%** | **0.62** | **0.9567** | **97.91%** | **92.52%** |
| Multi-Stage Composite + HistGradientBoosting | 95.99% | 0.68 | 0.9557 | 0.9811 | 0.9203 |
| Multi-Stage Composite + RandomForest | 95.99% | 0.46 | 0.9522 | 0.9754 | 0.9246 |
| Multi-Stage Composite + LogisticRegression | 95.99% | 0.68 | 0.9444 | 0.9821 | 0.8958 |
| Naive Baseline Blocking | 69.16% | 0.50 | ~0.72 | ~74% | ~66% |

---

## Quickstart

### 1. Installation

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Smoke Test on Sample Fixture

```bash
cd src
python pipeline.py --sample
```
Runs end-to-end on `dataset/sample/`, validates outputs, and packages the submission ZIP in < 5 seconds.

### 3. Full Production Run on Competition Dataset

```bash
cd src
python pipeline.py --train-sample-size 15000
```

This single command:
1. Loads aligned training subsamples with zero memory bloat.
2. Builds the high-recall composite inverted index.
3. Extracts 26 C-accelerated RapidFuzz pairwise features.
4. Performs 5-Fold Entity-Level Cross-Validation (GroupKFold).
5. Sweeps decision thresholds over $\tau \in [0.20, 0.96]$ optimizing macro $F_{0.5}$.
6. Trains the final LightGBM model.
7. Executes chunked inference on the test set.
8. Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` ensuring all 1,732,544 test $S_1$ entities are present.
9. Runs the official submission validator `utils/validate_submission.py`.
10. Packages the complete reproducible `Antigravity_ML_submission.zip`.

---

## Code Architecture

```
code/business_entity_resolution/src/
├── config.py           # Global paths, hyperparameters, and blocking thresholds
├── data_loader.py      # Streaming TSV reader and fast ground-truth parser
├── normalization.py   # Multi-representation string normalizer (accents, legal suffixes, ordinals)
├── blocking.py         # Multi-stage composite blocking index and recall metrics
├── features.py         # 26 C-accelerated RapidFuzz pairwise similarity features
├── models.py           # LightGBM / HistGradientBoosting training and F0.5 threshold sweep
├── validation.py       # 5-fold entity-level GroupKFold CV evaluator
├── inference.py        # Scalable chunked inference engine with cached representations
├── submission.py       # Output TSV writer, official validator runner, and ZIP packager
├── audit.py            # Dataset forensics and schema inspection utility
├── evaluate.py         # Exact competition scoring metric implementation
└── pipeline.py         # Master CLI entry point orchestrating the entire lifecycle
```

---

## Validation & Verification

Always verify outputs before submission:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Exit code `0` confirms complete compliance (`PASS — no blocking issues found. Safe to submit.`).
