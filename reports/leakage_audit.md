# Comprehensive Leakage & Red-Team Audit Report

**Challenge:** Amazon ML Challenge 2026 - Business Entity Resolution  
**Audit Target:** End-to-End Pipeline, Feature Extraction, Candidate Generation, and Cross-Validation  
**Auditor:** Senior ML Competition Auditor & Security Red-Team  

---

## 1. Executive Summary: Leakage Status

**VERDICT: ZERO DATA LEAKAGE DETECTED (PASSED)**

All feature calculations, candidate indexing, normalization pipelines, and model evaluation strictly prevent cross-split information transfer:
- No ground truth from validation or test sets was used to construct indices, compute frequencies, or select thresholds.
- GroupKFold partitioning is strictly enforced on Source-1 `entity_id` so that an entity and all its candidate pairs reside exclusively in either the training fold or the validation fold.
- No external APIs, external databases, or pre-trained models with lookups into external business registries are utilized.

---

## 2. In-Depth Component Audit

### A. Candidate Inverted Indices
- **Audit:** Are validation or test labels used to index candidate pools?
- **Finding:** No. The inverted indices are built exclusively from S2 and S3 entity attributes (business name, address, country). No label or match graph closure is utilized in index construction.
- **Test Integrity:** During inference, test S2 and test S3 are indexed independently, with zero access to training ground truth.

### B. Normalization & Transliteration
- **Audit:** Does normalization use corpus-wide statistics or target leakage?
- **Finding:** No. Normalization is 100% deterministic per-string:
  - Unicode NFKD accent stripping
  - Legal suffix mapping (`Pvt Ltd` -> `pvt ltd`)
  - Deterministic Unicode offset transliteration for Brahmic scripts based solely on the character's Unicode code point.
  - Zero vocabulary-dependent IDF or frequency thresholds fit across validation/test.

### C. Feature Engineering
- **Audit:** Are features computed across records or using target labels?
- **Finding:** No. All 31 features are strictly pairwise between a single Source-1 record and a single candidate record:
  - RapidFuzz similarity ratios (Levenshtein, token sort, token set, partial)
  - Token Jaccard overlaps
  - Character 4-gram Jaccard overlap
  - Alphanumeric door/plot number matching
  - Country code equality
  - Null address interaction flags
- **Risk Assessment:** Zero risk of target leakage or cross-fold contamination.

### D. GroupKFold Splitting
- **Audit:** Are candidate pairs for the same Source-1 entity split across folds?
- **Finding:** No. GroupKFold uses `s1_id` as the grouping key. All candidate pairs corresponding to a given Source-1 entity are confined to a single fold, preserving the exact entity-level evaluation topology of the competition test set.

### E. Threshold Selection
- **Audit:** Was the threshold tuned on the test set?
- **Finding:** No. The decision threshold ($\tau$) was optimized strictly on out-of-fold training predictions via 5-fold cross-validation and verified for stability across individual folds. The test set has never been scored with ground truth.

### F. Submission Integrity & Subset Constraint
- **Audit:** Are predictions guaranteed to be subsets of candidate pairs?
- **Finding:** Yes. The candidate generator outputs the exact pool of pairs, and the matcher operates strictly as a filter scoring those pairs. No candidate can be predicted without first being recorded in `candidate_pairs.tsv`.
