# Comprehensive Audit of the Current Business Entity Resolution System
**Amazon ML Challenge 2026**
**Date:** September 26, 2026
**Auditor:** Senior ML Competition Engineer & System Auditor

---

## A. Current System Architecture

The existing pipeline (`code/business_entity_resolution/src/`) comprises a modular, two-stage entity resolution system:
1. **Candidate Generation (Blocking):** An inverted-index retrieval engine constructing orthogonal posting lists over exact names, 2-token content signatures, composite address keys (numeric house token + street word), cross-field keys (name token + house number), distinctive address pairs, and extracted business aliases (`dba`, `aka`, `formerly`).
2. **Pairwise Feature Engineering:** A 26-dimensional RapidFuzz-accelerated pairwise similarity vector evaluating string, token, numeric, and cross-field interactions in microseconds.
3. **Classification & Scoring Engine:** LightGBM Gradient Boosted Decision Trees trained against hard negatives generated directly by the candidate generator.
4. **Metric-Aligned Decision Threshold:** Threshold search directly targeting macro $F_{0.5}$ across all Source-1 entities.
5. **Streaming Inference & Packaging:** Zero-RAM-bloat line-by-line TSV output generators passing strict formatting, schema, and subset constraints.

```
+----------------------------------------------------------------------------------------------------+
|                                    END-TO-END EXECUTION GRAPH                                      |
+----------------------------------------------------------------------------------------------------+
|  [Train / Test S1, S2, S3]                                                                         |
|            |                                                                                       |
|            v                                                                                       |
|  [Multi-Representation Normalization] (accents, legal suffixes, ordinals, door/plot numbers)       |
|            |                                                                                       |
|            v                                                                                       |
|  [Composite Inverted Index Blocking] (exact_name, name_pair, num_word, name_num, addr_pair, alias)  |
|            |                                                                                       |
|            v                                                                                       |
|  [Candidate Pairs Retrieval] (Posting caps: 150 name, 400 comp; max 120 cands/S1)                  |
|            |                                                                                       |
|            v                                                                                       |
|  [26 RapidFuzz C-Accelerated Features] (Levenshtein, token sort/set, numeric Jaccard, cross-terms)|
|            |                                                                                       |
|            v                                                                                       |
|  [LightGBM Model (Unified or S2/S3 Split)]                                                         |
|            |                                                                                       |
|            v                                                                                       |
|  [Out-of-Fold Threshold Optimization on Macro F0.5] (tau in [0.20, 0.96], best tau = 0.70 - 0.72)   |
|            |                                                                                       |
|            v                                                                                       |
|  [Streamed Inference Engine] -> matching_results.tsv (1.73M rows) & candidate_pairs.tsv (1.73M rows) |
+----------------------------------------------------------------------------------------------------+
```

---

## B. Data Flow & Input Schemas
- **Reference Table ($S_1$):** `entity_id` (`S1-XXXX`), `business_name`, `business_address`, `country`.
- **Target Tables ($S_2, S_3$):** `entity_id` (`S2-XXXX` or `S3-XXXX`), `business_name`, `business_address`, `country`.
- **Ground Truth:** `source1_entity_id`, `matched_entity_ids` (comma-separated $S_2$ and $S_3$ IDs).
- **ID Alignment Trap in Training:** Handled correctly in `data_loader.py` via `load_training_subsample()` which loads ground truth first and indexes corresponding $S_1$, $S_2$, $S_3$ entities by explicit entity ID lookup rather than naive row slicing.

---

## C. Candidate Generation Flow
- Extracted blocking keys:
  - `exact_name`: NFKD normalized lowercase string.
  - `url_core`: Domain token extracted via regex (`dprobst.com` $\to$ `dprobst`).
  - `name_pair`: Sorted 2-token signature of content words.
  - `name_tok`: Rare individual name tokens.
  - `num_word`: `(clean_house_number, street_word)` composite tuple.
  - `name_num`: `(name_token, clean_house_number)` cross-field tuple.
  - `addr_pair`: Sorted 2-token address signatures.
  - `alias_keys`: Tokens extracted after alias delimiters (`aka`, `dba`, `formerly`).
- Candidate retrieval caps:
  - `MAX_NAME_TOK_POSTINGS` = 150
  - `MAX_COMPOSITE_POSTINGS` = 400
  - `MAX_CANDIDATES_PER_S1` = 120
- Output candidate set: Union of retrieved candidates up to 120 per $S_1$ entity.

---

## D. Feature Engineering (26 Features)
1. **Name Features (10):** `name_exact`, `name_fuzz_ratio`, `name_token_sort_ratio`, `name_token_set_ratio`, `name_partial_ratio`, `name_token_jaccard`, `name_token_overlap_count`, `name_len_diff`, `name_len_ratio`, `alias_match`.
2. **Address Features (10):** `addr_exact`, `addr_fuzz_ratio`, `addr_token_sort_ratio`, `addr_token_set_ratio`, `addr_partial_ratio`, `addr_token_jaccard`, `addr_numeric_jaccard`, `addr_numeric_overlap_count`, `addr_len_diff`, `addr_is_empty_either`.
3. **Cross & Meta Features (6):** `country_match`, `is_source3`, `name_addr_mult`, `name_addr_min`, `name_addr_max`, `name_addr_mean`.

---

## E. Model Training
- **Model Architecture:** LightGBM Gradient Boosted Decision Trees (`n_estimators=150-200`, `learning_rate=0.07-0.08`, `max_depth=7`, `num_leaves=31`, `subsample=0.85`, `colsample_bytree=0.85`).
- **Negative Ratio:** Natural hard negative ratio produced by candidate generation (~1:27 positive to hard negative).
- **Loss:** Binary log-loss, with threshold post-processing directly targeting macro $F_{0.5}$.

---

## F. Validation Methodology
- **Strategy:** 5-Fold GroupKFold grouped strictly by Source-1 entity ID (`groups = groups`). No entity appears in both training and validation folds.
- **Metric Formulation:** Exact macro-averaged $F_{0.5}$ over all Source-1 entities:
  - Correct singletons (true=[], pred=[]) score 1.0.
  - False merges on singletons (true=[], pred!=[]) score 0.0.
  - Missed entities (true!=[], pred=[]) score 0.0.

---

## G. Threshold Optimization
- Out-of-fold probability predictions are collected across all 5 folds.
- Fine-grained grid search: $\tau \in [0.20, 0.96]$ in steps of $0.02$.
- Criterion: Maximizes macro $F_{0.5}$ across all evaluation entities.
- Best verified threshold: $\tau = 0.70$ (unified), $\tau = 0.72$ (source-specific).

---

## H. Inference Engine
- Uses pre-normalized candidate representations cached in memory.
- Instant country mismatch rejection: if both records specify country and $c_1 \neq c_2$, skip pair immediately.
- Fast C-accelerated pre-tokenized feature computation.
- Threshold gating at optimal $\tau$.

---

## I. Submission Construction & Validation
- Emits `matching_results.tsv` and `candidate_pairs.tsv` with required columns.
- All $S_1$ entities from `test_source1.tsv` are guaranteed to appear exactly once.
- Verified: All matched IDs are guaranteed subsets of candidate IDs.
- Validated via official `utils/validate_submission.py` (Exit 0, PASS).

---

## J. Potential Leakage Risks
1. **Candidate Index:** Built on target tables ($S_2, S_3$). In production, candidate indexing does not use ground truth. In validation, the target pool must not contain validation-only information.
2. **Frequency & Rarity Counts:** If calculated globally across train + validation, leakage could occur. Currently, stop words are static and token posting caps are local to the index.
3. **Threshold Selection:** Must be performed on Out-of-Fold (OOF) predictions rather than training predictions. Currently implemented cleanly on OOF predictions.

---

## K. Computational Bottlenecks
- **RAM Capacity:** Available RAM is ~4.5 GB out of 16.85 GB. Monolithic in-memory indexing of 10M records consumes ~18 GB and will trigger out-of-memory errors if not handled with streaming or chunked processing.
- **Pairwise Scoring:** RapidFuzz is highly optimized in C, scoring ~25,000 pairs/sec per core. With 887k pairs in CV, feature extraction takes 40s.

---

## L. Current Reproducibility Risks
- Random seeds are pinned to 42 across LightGBM, GroupKFold, and scikit-learn.
- Dependencies are pinned in `requirements.txt`.
- Codebase is modular and executable via single command CLI.
