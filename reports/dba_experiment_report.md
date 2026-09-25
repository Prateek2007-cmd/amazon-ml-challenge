# Final DBA / Brand / Website Recovery Experiment Report

**Challenge:** Amazon ML Challenge 2026 - Business Entity Resolution  
**Benchmark Scope:** 10,000 Source-1 Validation Entities | 34,511 True Ground-Truth Pairs  
**Frozen Champion Target:** Macro F0.5 = 0.9579 (Precision: 0.9752, Recall: 0.9233)  
**Strict Decision Threshold:** New Macro F0.5 >= 0.9590 across all 5 GroupKFold folds  

---

## 1. DBA / Brand / Website Miss Taxonomy

Empirical isolation of all 1,781 candidate-generation misses under the 0.9579 champion revealed the following concrete distribution:

| Subtype | Description | Count | % of Misses | Root Cause Analysis |
| :--- | :--- | :---: | :---: | :--- |
| **J_cross_script_dba** | Indic native script vs English brand phrasing | 726 | 40.8% | Native script phonetic divergence where address has differing word order |
| **C_website_domain_variation** | Unsegmented domain name vs legal entity | 358 | 20.1% | Concatenated domain stem (e.g. `saffronfinancers.com` vs `Saffron Financers LLP`) |
| **I_legal_suffix_dominating** | Legal form divergence with disparate branding | 332 | 18.6% | Multiple commercial trading brands under same holding entity |
| **B_brand_vs_parent_company** | Short brand name vs full parent corporate name | 182 | 10.2% | Single-token brand (e.g. `Wexveo`) vs legal entity (`Shakti Agro Limited`) |
| **G_empty_or_weak_address** | Blank or severely truncated address (< 10 chars) | 140 | 7.9% | Candidate has `<NULL>` or empty address, precluding composite address keys |
| **D_website_redirect_weak_address**| Domain present but address is missing / disparate | 17 | 1.0% | Domain name present but geographic coordinates diverge |
| **F_missing_business_name_tokens** | Address matches strongly but names have disparate branding | 14 | 0.8% | Commercial tenant variation |
| **A_dba_vs_legal_entity** | Explicit DBA keyword present (`dba`, `aka`, `formerly`) | 9 | 0.5% | Trade brand split |
| **E_abbreviated_brand_acronym** | Acronym / initialism vs full name | 3 | 0.2% | Unexpanded initialism |
| **Total** | | **1,781** | **100.0%** | |

*(Detailed record-by-record table saved in `experiments/dba_misses_detailed.csv`)*

---

## 2. Tested Recovery Mechanisms

1. **Pass A (Domain Concatenation & Stem Extraction):**
   - Stripping TLD (`.com`, `.in`, `.org`), lowercasing, and matching unsegmented stems (`saffronfinancers` $\leftrightarrow$ `saffron financers`).
2. **Pass B (Length-Ranked Distinctive Address Hierarchy):**
   - Sorting address tokens by length descending to prioritize distinctive locality tokens over generic prefixes.
3. **Strict Gate Formulation:**
   - Requiring STRONG DOMAIN/BRAND SIGNAL + at least one independent supporting signal (country agreement, door/plot number match, 5/6-digit postal code agreement, or distinctive address token overlap).

---

## 3. Empirical Results: Head-to-Head Comparison

| Metric | Champion (Retained) | Pass B (Looser Address) | Strict DBA Gate | Decision Rule Requirement | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro F0.5** | **0.9579** | 0.9565 | 0.9556 | **>= 0.9590** | **REJECTED** |
| **Macro Precision** | **0.9752** | 0.9719 | 0.9739 | > 0.9750 | Champion superior |
| **Macro Recall** | 0.9233 | **0.9279** | 0.9188 | - | Pass B higher recall |
| **Micro F0.5** | **0.9703** | 0.9681 | 0.9689 | - | Champion superior |
| **Candidate Recall** | 94.97% | **95.71%** | 94.69% | - | Pass B higher recall |
| **Average Candidates / S1** | **111.3** | 118.7 | 114.9 | <= 120 | All compliant |
| **Hard Negatives Introduced** | **Baseline** | +149 FP | +16 FP | Minimize | Champion cleanest |
| **5-Fold Cross-Validation Std**| **0.0023** | 0.0048 | 0.0045 | Low variance | Champion 2x more stable |
| **Singleton F0.5** | **0.9497** | 0.9425 | 0.9461 | Protect | Champion superior |
| **Zero-Match Impact** | **0 false merges** | +8 false merges | +3 false merges | 0 false merges | Champion superior |
| **One-Match F0.5** | **0.9039** | 0.8930 | 0.8985 | - | Champion superior |
| **Multi-Match F0.5** | **0.9616** | 0.9612 | 0.9596 | - | Champion superior |

---

## 4. 5-Fold GroupKFold Stability Breakdown

| Fold | Champion Baseline | Pass B Experiment | Strict DBA Gate |
| :---: | :---: | :---: | :---: |
| **Fold 1** | **0.9578** | 0.9572 | 0.9515 |
| **Fold 2** | **0.9543** | 0.9532 | 0.9615 |
| **Fold 3** | **0.9603** | 0.9494 | 0.9578 |
| **Fold 4** | **0.9565** | 0.9633 | 0.9491 |
| **Fold 5** | **0.9604** | 0.9593 | 0.9580 |
| **Mean +/- Std** | **0.9579 +/- 0.0023** | 0.9565 +/- 0.0048 | 0.9556 +/- 0.0045 |

---

## 5. Critical Engineering Findings

1. **Precision Penalty of Looser DBA / Address Keys:**
   - Attempting to force retrieval of extreme DBA/unlinked-brand pairs introduces subtle hard negatives (such as different retail branches or unrelated businesses sharing a dense metropolitan locality like Goregaon, Mumbai or Connaught Place, Delhi).
   - In a competition evaluated on **Macro $F_{0.5}$** ($\beta = 0.5$, precision weighted $2\times$ over recall), gaining 190 true matches at the cost of 149 false positives causes a net score DROP from 0.9579 to 0.9565.
2. **Current Champion Already Optimal on High-Confidence Domains:**
   - Pass 1 in the Champion already extracts `exact_name`, `url_core`, and high-precision token pairs. High-confidence domains (e.g. `pushorizonpaper.com`) are already captured.
   - The remaining unlinked DBA cases (like `Wexveo` $\leftrightarrow$ `Shakti Agro Limited`) share zero lexical name tokens and have weak address overlap; attempting to retrieve them without external data lookups inevitably degrades precision.
3. **Zero-Match and Singleton Protection:**
   - Looser candidate passes introduced false positive merges into true zero-match entities, reducing Singleton F0.5 from 0.9497 to 0.9425.
   - The frozen champion remains the optimal trade-off point between true recall and precision protection.

---

## 6. Final Decision & Status

**NO ROBUST DBA RECOVERY IMPROVEMENT FOUND — CURRENT 0.9579 CHAMPION RETAINED.**

In strict compliance with Rules 10, 11, and 12:
- The experimental methods failed to reach the required $\ge 0.9590$ bar.
- The existing **0.9579 champion** (Unified LightGBM, 31 features, $\tau = 0.62$, 5-fold mean $0.9579 \pm 0.0023$) remains frozen, verified, and untouched.
- Submission archive `Antigravity_ML_submission_optimized.zip` (18.69 MB) is 100% compliant with all official validation checks and is retained as the final competition deliverable.
