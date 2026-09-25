# Incremental Recovery Pass Ablation Report

**Benchmark Scope:** 10,000 Source-1 Entities | 34,511 Ground-Truth True Pairs
**Candidate Cap Constraint:** Max Cands = 120 per Source-1 Entity

## 1. Step-by-Step Cumulative Ablation Results

| Recovery Pass Configuration | Candidate Recall | True Matches Recovered | Missed Matches | Avg Candidates / S1 | Median | P95 | P99 | Incremental Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Champion** | **94.91%** | 32,755 | 1,756 | 115.6 | 74.0 | 401.0 | 535.0 | - |
| **+ Pass A (Domain Concatenation)** | **94.78%** | 32,708 | 1,803 | 114.4 | 74.0 | 387.1 | 497.0 | +-47 matches |
| **+ Pass B (Distinctive Addr Words)** | **95.51%** | 32,963 | 1,548 | 117.6 | 81.0 | 376.0 | 554.0 | +255 matches |
| **+ Pass C (Distinctive Addr Pairs)** | **95.44%** | 32,939 | 1,572 | 121.8 | 90.0 | 382.0 | 554.0 | +-24 matches |
| **+ Pass D (Missing Addr Backoff)** | **95.44%** | 32,939 | 1,572 | 120.2 | 88.0 | 382.0 | 554.0 | +0 matches |
| **+ Pass F (Phonetic Transliteration)** | **95.44%** | 32,938 | 1,573 | 120.4 | 89.0 | 382.0 | 554.0 | +-1 matches |


## 2. Pass Analysis & Architectural Rationale

1. **Pass A (Domain Concatenation):**
   - Automatically bridges unsegmented web domain business names (e.g. `saffronfinancers.com`) with tokenized legal entity names (`Saffron Financers LLP`).

2. **Pass B & C (Length-Ranked Distinctive Address Components):**
   - Sorting address words by length descending ensures unique locality and building tokens are prioritized over generic positional prefixes (`lane`, `road`, `street`).

3. **Pass D (Missing Address Backoff):**
   - For entities with null addresses, recovers distinctive name tokens of length >= 6 rather than dropping out of blocking.

4. **Pass F (Phonetic Transliteration Normalization):**
   - Normalizes regular phonetic divergences between English spellings (`x`, `ph`) and Indic phonetic representations (`ks`, `f`).
