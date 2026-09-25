# Model Optimization & Feature Engineering Results

**Challenge:** Amazon ML Challenge 2026 - Business Entity Resolution  
**Baseline Verified Score:** Macro F0.5 = 0.9422 (Unified), 0.9426 (Source-Specific)  
**Evaluation Protocol:** 5-Fold Source-1 Entity-Level GroupKFold (10,000 Entities, 34,511 Ground-Truth Pairs)  

---

## 1. Summary of Optimization Experiments

| Experiment ID | Architectural Change | Candidate Recall | Avg Cands / S1 | Macro F0.5 | Macro Prec | Macro Rec | Micro F0.5 | Singleton F0.5 | 1-Match F0.5 | Multi-Match F0.5 | US F0.5 | India F0.5 | TP | FP | FN | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | Config B + 26 base features ($\tau=0.70$) | 91.96% | 97.5 | **0.9422** | 0.9687 | 0.8909 | 0.9645 | 0.9512 | 0.8654 | 0.9491 | 0.9662 | 0.9204 | 30,650 | 446 | 3,861 | BASELINE |
| **EXP_DYNAMIC_BLOCKING_BASE_FEATS** | Multi-pass dynamic blocking + 26 base features ($\tau=0.74$) | 94.97% | 111.3 | **0.9537** | 0.9749 | 0.9100 | 0.9691 | 0.9605 | 0.8793 | 0.9577 | 0.9706 | 0.9284 | 31,392 | 473 | 3,119 | **IMPROVED** |
| **EXP_DYNAMIC_BLOCKING_EXPANDED_FEATS** | Multi-pass dynamic blocking + 31 targeted features ($\tau=0.62$) | 94.97% | 111.3 | **0.9579** | 0.9752 | 0.9233 | 0.9703 | 0.9497 | 0.9039 | 0.9616 | 0.9717 | 0.9373 | 31,860 | 556 | 2,651 | **IMPROVED** |
| **EXP_SOURCE_SPECIFIC_EXPANDED_FEATS** | Source-specific S2/S3 models + 31 targeted features ($\tau=0.76$) | 94.97% | 111.3 | **0.9578** | 0.9784 | 0.9143 | 0.9716 | 0.9659 | 0.8967 | 0.9609 | 0.9714 | 0.9374 | 31,512 | 401 | 2,999 | **IMPROVED** |

---

## 2. 5-Fold GroupKFold Stability for Winning Model (EXP 2)

| Fold Number | Macro F0.5 | Macro Precision | Macro Recall | Micro F0.5 |
| :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 0.9578 | 0.9744 | 0.9234 | 0.9701 |
| Fold 2 | 0.9543 | 0.9712 | 0.9229 | 0.9688 |
| Fold 3 | 0.9603 | 0.9763 | 0.9291 | 0.9724 |
| Fold 4 | 0.9565 | 0.9744 | 0.9210 | 0.9695 |
| Fold 5 | 0.9604 | 0.9795 | 0.9196 | 0.9708 |
| **Mean +/- Std** | **0.9579 +/- 0.0023** | **0.9752** | **0.9233** | **0.9703** |

Cross-fold standard deviation is an exceptionally low **0.0023**, proving that the +0.0157 Macro F0.5 gain is globally stable and not driven by any single fortunate split.

---

## 3. Detailed Ablation Insights

1. **Candidate Retrieval Lever (+0.0115 Macro F0.5):**
   - Shifting from unordered set blocking to prioritized 6-pass retrieval with Indic transliteration and noise-filtered address keys directly boosted candidate recall from 91.96% to 94.97%.
   - With base features alone, this lifted Macro F0.5 from 0.9422 to 0.9537.

2. **Feature Expansion Lever (+0.0042 Macro F0.5):**
   - Adding 5 targeted features (Transliterated name similarity, Character 4-gram Jaccard, Door/plot exact number match, Empty-address name confidence interaction, and Postal/Pincode consistency flag) lifted Macro F0.5 from 0.9537 to 0.9579.
   - Most notably, Recall jumped from 0.9100 to 0.9233 while Precision remained elite at 0.9752.

3. **High-Precision Configuration (EXP 3):**
   - Source-specific models trained at threshold $\tau = 0.76$ achieve **0.9784 precision** with only 401 false positives across 1.02M pairs, and the highest singleton score (0.9659).
