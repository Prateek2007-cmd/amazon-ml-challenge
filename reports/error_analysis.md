# Error Breakdown & Bottleneck Analysis Report

## 1. Global Error Decomposition

| Error Type | Count | % of All True Matches | % of Total Missed Matches (FN) | Dominant Factor |
| :--- | :---: | :---: | :---: | :--- |
| **Candidate Generation Miss (Blocking FN)** | 2,773 | 8.04% | **72.1%** | True candidate never reached matcher |
| **Matcher Rejection (Matcher FN)** | 1,068 | 3.10% | **27.8%** | Scored below threshold (p < 0.70) |
| **Matcher False Positive (Matcher FP)** | 454 | 0.05% of pairs | N/A (Precision impact) | Deceptive negative accepted |

### Critical Bottleneck Insight:
- **72.2% of all missed matches are lost at the Candidate Generation stage.**
- The matcher itself has **98.57% precision** and **96.6% recall** on pairs that reach it.
- Therefore, expanding candidate generation with targeted, high-precision retrieval passes (address numbers, script transliteration, postal codes, and rare name prefixes) is the single highest-impact lever to increase Macro F0.5.

## 2. Taxonomy of Missed Candidate Pairs (Blocking FN)

| Missed Category | Count | % of Misses | Root Cause & Resolution Strategy |
| :--- | :---: | :---: | :--- |
| `non_ascii_script_in_name` | 1,151 | 41.5% | Indic or native non-Latin script in candidate name. Resolution: Pass F (transliteration / script-robust address-anchored retrieval). |
| `shared_address_number_but_missed_composite_key` | 673 | 24.3% | Entities share door/plot number but street words differ in spelling or order. Resolution: Pass C (relaxed numeric + locality/pincode key). |
| `high_name_similarity_but_missed_key` | 488 | 17.6% | High name similarity but token combinations hit posting cap or minor typo. Resolution: Pass B (character 4-gram / prefix retrieval). |
| `low_name_and_low_address_similarity` | 251 | 9.1% | Extreme lexical variation across both fields. |
| `empty_candidate_address_and_no_name_key` | 210 | 7.6% | Candidate has blank address. Resolution: Pass E (pure name distinctive token retrieval). |

## 3. Concrete Examples of Recoverable Pairs

**Case 1: [high_name_similarity_but_missed_key]** (Name Sim: 94.73684210526316, Addr Sim: 72.46376811594203, Num Overlap: ['0'])
- **S1:** `Spicer Star Environmental Services LLC` | `Fl 0, MO, Saint Louis, 9327 Atwood Drive`
- **Cand:** `Spicer-Star Environmental Seraices LLC` | `932 Atwood Dr, Fl 0, Stlouis, Missouri`

**Case 2: [non_ascii_script_in_name]** (Name Sim: 13.33333333333333, Addr Sim: 64.76190476190476, Num Overlap: [])
- **S1:** `Shakti Agro Limited` | `C/O Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, Orissa`
- **Cand:** `ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍` | `C/O GURNAV SINGH SALUJA, SADAR, Odisha`

**Case 3: [low_name_and_low_address_similarity]** (Name Sim: 9.523809523809524, Addr Sim: 87.07482993197279, Num Overlap: [])
- **S1:** `Shakti Agro Limited` | `C/O Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, Orissa`
- **Cand:** `Wexveo` | `Block F-264- C/o Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, OD`

**Case 4: [non_ascii_script_in_name]** (Name Sim: 12.5, Addr Sim: 98.36065573770492, Num Overlap: ['3', '299'])
- **S1:** `Vijay Trading Private Limited` | `J-3/299, Ground Floor Dda Flats Kalka Ji, New Delhi, South Delhi, Delhi`
- **Cand:** `विजय ट्रेडिंग प्राइवेट लिमिटेड` | `J-D/3/299, GROUND FLOOR DDA FLATS KALKA JI, NEW DELHI, SOUTH DELHI, Delhi`

**Case 5: [low_name_and_low_address_similarity]** (Name Sim: 43.90243902439024, Addr Sim: 77.77777777777779, Num Overlap: [])
- **S1:** `Saffron Financers LLP` | `47Akumaresa Puram Tiruchengode, Namakkal District, Salem, Tamil Nadu`
- **Cand:** `saffronfinancers.com` | `TN, Namakkal District, <NULL>, H.no 47Akumaresa Purma Tiruchengode`

**Case 6: [high_name_similarity_but_missed_key]** (Name Sim: 92.5925925925926, Addr Sim: 87.5, Num Overlap: ['411', '1'])
- **S1:** `United Consulting Private Limited` | `C 411 E, 1St Floor, Sushant Lok, Phase-1, Dlf Qe, Gurgaon, Haryana`
- **Cand:** `Sri united consulting private limited` | `Haryana, #5-412 C 411 E, 1ST FLOOR, SUSHANT LOK, PHASE-1, DLF QE`

**Case 7: [non_ascii_script_in_name]** (Name Sim: 13.33333333333333, Addr Sim: 95.59748427672956, Num Overlap: ['101', '2', '5', '6'])
- **S1:** `Great Impex Private Limited` | `6-2-101/5/C, Telangana, Hyderabad, Secunderabad, Lane Beside Centralview Apt New Bhoiguda`
- **Cand:** `గ్రేట్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్` | `6-2-101/5/C, Lane Beside Centralview Apt New Bhoiguda, Secunderabad, Hyderabad, TG`

**Case 8: [shared_address_number_but_missed_composite_key]** (Name Sim: 68.0, Addr Sim: 95.59748427672956, Num Overlap: ['101', '2', '5', '6'])
- **S1:** `Great Impex Private Limited` | `6-2-101/5/C, Telangana, Hyderabad, Secunderabad, Lane Beside Centralview Apt New Bhoiguda`
- **Cand:** `Smt *** Great Impex Private-Limited` | `6-2-101/5/C, Lane Beside Centralview Apt New Bhoiguda, Hyderabad, Secunderabad, TG`

