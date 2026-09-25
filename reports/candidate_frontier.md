# Candidate Generation Frontier Analysis

**Challenge:** Amazon ML Challenge 2026 - Business Entity Resolution  
**Benchmark Pool:** 10,000 Source-1 Entities (34,511 True Ground-Truth Pairs across S2 & S3)  
**Evaluated Against:** Inverted Index over 116,329 S2 records + 117,525 S3 records  

---

## 1. Candidate Frontier Overview & Trade-off Analysis

Candidate generation forms the upper bound for any downstream matching model: any true entity pair missed at the blocking stage can never be recovered by the classifier.

However, the competition scoring evaluates the candidate set size: final predicted matches must be a subset of `candidate_pairs.tsv`, and generating an excessively large candidate pool risks lower candidate precision and higher computational footprint.

The goal is to **maximize true-match candidate recall while strictly controlling candidate budget (average and tail percentiles)**.

---

## 2. Experimental Candidate Frontier Results

| Configuration | Retrieval Passes Enabled | Max Cands Cap | Candidate Recall | True Matches Retrieved | Missed Matches | Avg Cands / S1 | Median (P50) | P95 | P99 | Search Space Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Config B)** | Unordered Set (Exact, Name pairs, Addr comp) | 120 | **91.96%** | 31,738 | 2,773 | 97.5 | 98 | 120 | 120 | 99.96% |
| **Frontier 1 (Strict)** | Multi-Pass Dynamic (Pass 1-6 + Translit) | 80 | **94.43%** | 32,588 | 1,923 | 104.0 | 71 | 325 | 491 | 99.95% |
| **Frontier 2 (Balanced)**| Multi-Pass Dynamic (Pass 1-6 + Translit) | 100 | **94.78%** | 32,709 | 1,802 | 108.1 | 71 | 343 | 498 | 99.95% |
| **Frontier 3 (Optimal)** | Multi-Pass Dynamic (Pass 1-6 + Translit) | 120 | **94.97%** | 32,774 | 1,737 | 111.3 | 71 | 365 | 507 | 99.95% |
| **Frontier 4 (High-Recall)**| Multi-Pass Dynamic (Pass 1-6 + Translit) | 150 | **95.36%** | 32,909 | 1,602 | 116.2 | 71 | 393 | 513 | 99.95% |

---

## 3. Key Architectural Innovations in Second-Generation Candidate System

1. **Deterministic Indic Script Transliteration (Pass 2):**
   - Transliterates Brahmic scripts (Devanagari, Telugu, Kannada, Bengali, Tamil, Gujarati, Malayalam, Odia, Gurmukhi) using pure Unicode block relative offsets.
   - Converts non-Latin S2/S3 names to Latin phonetic equivalents before extracting name tokens and pairs.
   - Recovers ~1,150 true pairs that were completely invisible to Latin S1 tokens.

2. **Strict Priority-Ordered Pass Evaluation:**
   - In the baseline, keys were stored in a Python `set`, causing arbitrary hash iteration order where high-cardinality address pairs filled the candidate cap before exact name keys were evaluated.
   - In the optimized engine, high-precision passes (Pass 1: Exact Name & URL Core; Pass 2: Transliteration; Pass 3: Name+Number) are evaluated strictly before generic address tokens.

3. **Noise-Filtered Address Tokens:**
   - Prevents composite house/door numbers (e.g. `J-3/299`, `6-2-101/5/C`) from contaminating address word tokens.
   - Filters out ubiquitous Indian address stopwords (`ground`, `flats`, `floor`, `bldg`, `c/o`, `beside`, `near`, `opp`, `nagar`, `district`) that generated posting lists exceeding cap thresholds.

4. **Dynamic Candidate Allocation:**
   - P50 (median) candidate count dropped from 98 to **71**, demonstrating that high-confidence entities resolve quickly with compact candidate lists, allowing budget allocation where true ambiguity exists.

---

## 4. Pareto Frontier Selection

- **Selected Candidate Configuration:** **Frontier 3 (Max Cands = 120)**
- **Candidate Recall:** **94.97% (+3.01% absolute improvement over baseline)**
- **Average Candidates:** 111.3 per S1
- **Median Candidates:** 71 per S1
- **Search Space Reduction:** > 99.95%
