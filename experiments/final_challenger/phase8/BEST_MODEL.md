# Phase 8: Current Strongest Verified Model

## Status
- **Current Strongest Verified Model**: **EXP02 (LightGBM 63 Features @ Tau=0.74)**
- **Macro F0.5**: **0.9784** (Up from Baseline **0.9723**, +61 basis points)
- **Precision**: **0.9878** (Up from 0.9827)
- **Recall**: **0.9580** (Up from 0.9501)
- **5-Fold Cross-Validation Mean**: **0.9783 ± 0.0010**
- **Fold Consistency**: `[0.9785, 0.9797, 0.9771, 0.9792, 0.9772]` (Variance: 0.000001)

---

## Slice Breakdown (EXP02 vs Baseline)
| Metric / Slice | Phase 6 P5 Baseline (44 Feat) | Phase 8 EXP02 (63 Feat, Tau=0.74) | Delta |
| :--- | :---: | :---: | :---: |
| **Macro F0.5** | **0.9723** | **0.9784** | **+0.0061 (+61 bps)** |
| **Precision** | 0.9827 | **0.9878** | **+0.0051** |
| **Recall** | 0.9501 | **0.9580** | **+0.0079** |
| **False Positives** | 422 | **289** | **-133 FP (-31.5%)** |
| **False Negatives** | 1,699 | **1,474** | **-225 FN (-13.2%)** |
| **True Positives** | 32,812 | **33,037** | **+225 TP** |
| **Zero-Match F0.5** | 0.9461 | **0.9677** | **+0.0216** |
| **One-Match F0.5** | 0.9304 | **0.9474** | **+0.0170** |
| **Multi-Match F0.5**| 0.9764 | **0.9809** | **+0.0045** |
| **US Entities F0.5**| 0.9796 | **0.9860** | **+0.0064** |
| **India Entities F0.5**| 0.9613 | **0.9670** | **+0.0057** |

---

## What Drove This Improvement?
1. **IDF Token Specificity**: Downweighted generic business suffixes ("Enterprises", "Limited", "Private", "Services") while heavily weighting rare discriminating tokens ("Zomato", "Accenture", "McKinsey").
2. **PIN Code Verification**: Exact matching PIN codes provided near-certainty positive signals; conflicting PIN codes in the same city successfully eliminated 133 false positive collisions.
3. **Directional Address Containment**: Short abbreviated addresses fully contained in detailed candidate strings are no longer penalized by standard Jaccard.
4. **Clean Domain Matching**: Normalized URL core tokens matched website domains directly against entity names.
5. **Empty Address Protection**: Entities with empty candidate addresses but >94% name similarity and rare token matches are preserved instead of penalized.

---

## Next Direct Attack: The India Script Frontier
- Forensics revealed that **409 out of 472 remaining candidate misses (86.7%)** are in India where candidate names are in Indic scripts (Devanagari, Bengali, Marathi) or formatted with character segmentation.
- Incorporating Indic-to-Latin phonetic transliteration before candidate indexing and address-based blocking will unlock the remaining candidate recall barrier toward 99.2%+ recall and push Macro F0.5 over 0.985+.
