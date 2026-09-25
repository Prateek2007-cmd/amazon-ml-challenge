# Final Micro-Optimization & Candidate-Recall Red-Team Report

**Challenge:** Amazon ML Challenge 2026 - Business Entity Resolution  
**Evaluation Standard:** 5-Fold Source-1 Entity-Level GroupKFold Across 10,000 Entities  
**Objective:** Micro-optimization to test whether targeted candidate recovery can push Macro F0.5 beyond 0.9579 without precision regression.

---

## 1. Executive Summary & Verdict

**DECISION: NO ROBUST IMPROVEMENT FOUND — CURRENT 0.9579 CHAMPION RETAINED.**

The current champion (Unified LightGBM with 31 targeted features at $\tau = 0.62$) was subjected to extensive micro-optimization targeting the remaining 1,737 candidate-generation misses.
While Pass B (length-ranked distinctive address tokens) successfully increased true candidate recall from **94.97% to 95.71%** (+190 true positive matches caught), the additional hard negative pairs introduced into the matcher frontier caused false positives to rise from 556 to 705.
Because the competition metric is **Macro $F_{0.5}$** (which weights precision twice as heavily as recall), the precision decline from 0.9752 to 0.9719 caused Macro $F_{0.5}$ to settle at **0.9565** (below the 0.9579 champion).
In accordance with Rule 19 and Rule 24 ("A validated 0.9579 is better than an unstable 0.9580; if no robust improvement is found, keep the current 0.9579 system unchanged"), **the 0.9579 champion is retained as the definitive final submission.**

---

## 2. In-Depth Head-to-Head Comparison

| Metric / Dimension | Champion Baseline (Retained) | Micro-Optimization (Pass B) | Delta | Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Macro F0.5** | **0.9579** | 0.9565 | -0.0014 | Champion superior |
| **Macro Precision** | **0.9752** | 0.9719 | -0.0033 | Champion superior |
| **Macro Recall** | 0.9233 | **0.9279** | **+0.0046** | Pass B higher recall |
| **Micro F0.5** | **0.9703** | 0.9681 | -0.0022 | Champion superior |
| **Candidate Recall** | 94.97% (32,774 / 34,511) | **95.71%** (33,031 / 34,511) | **+0.74%** (+257 true pairs) | Pass B higher |
| **Average Candidates / S1** | **111.3** | 118.7 | +7.4 | Champion tighter |
| **Median Candidates (P50)**| **71.0** | 81.0 | +10.0 | Champion tighter |
| **P95 Candidates** | **365.0** | 376.0 | +11.0 | Comparable |
| **P99 Candidates** | **506.0** | 554.0 | +48.0 | Tail expanded |
| **Singleton F0.5** | **0.9497** | 0.9425 | -0.0072 | Champion superior |
| **One-Match F0.5** | **0.9039** | 0.8930 | -0.0109 | Champion superior |
| **Multi-Match F0.5** | **0.9616** | 0.9612 | -0.0004 | Neutral |
| **US F0.5** | **0.9717** | 0.9703 | -0.0014 | Champion superior |
| **India F0.5** | **0.9373** | 0.9360 | -0.0013 | Champion superior |
| **True Positives (TP)** | 31,860 | **32,050** | **+190 matches** | Pass B higher |
| **False Positives (FP)** | **556** | 705 | +149 FP | Champion cleaner |
| **False Negatives (FN)**| 2,651 | **2,461** | **-190 FN** | Pass B lower FN |
| **5-Fold Cross-Validation Std**| **0.0023** | 0.0048 | +0.0025 | Champion 2x more stable |

---

## 3. Cross-Fold Stability Breakdown

### Champion (Unified LightGBM, 31 Features, $\tau = 0.62$):
- Fold 1: **0.9578** (Prec: 0.9744, Rec: 0.9234)
- Fold 2: **0.9543** (Prec: 0.9712, Rec: 0.9229)
- Fold 3: **0.9603** (Prec: 0.9763, Rec: 0.9291)
- Fold 4: **0.9565** (Prec: 0.9744, Rec: 0.9210)
- Fold 5: **0.9604** (Prec: 0.9795, Rec: 0.9196)
- **Mean $\pm$ Std:** **0.9579 $\pm$ 0.0023**

### Pass B Micro-Optimization:
- Fold 1: 0.9572 (Prec: 0.9725, Rec: 0.9286)
- Fold 2: 0.9532 (Prec: 0.9672, Rec: 0.9283)
- Fold 3: 0.9494 (Prec: 0.9647, Rec: 0.9260)
- Fold 4: 0.9633 (Prec: 0.9787, Rec: 0.9312)
- Fold 5: 0.9593 (Prec: 0.9760, Rec: 0.9253)
- **Mean $\pm$ Std:** **0.9565 $\pm$ 0.0048**

---

## 4. Engineering Conclusion

The empirical test confirms that attempting to squeeze the last 0.74% of candidate recall via looser address word permutations brings in subtle hard-negative noise (similar locality names in dense urban districts like Mumbai and Delhi) that degrades precision on $F_{0.5}$.
Therefore, adhering to scientific discipline and the explicit mandate of Section 24 ("A validated 0.9579 is better than an unstable 0.9580"), **the 0.9579 champion is fully verified, frozen, and retained.**
