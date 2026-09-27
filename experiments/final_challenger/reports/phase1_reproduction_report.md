# Phase 1: Semantic Challenger Reproduction Report

**Execution Timestamp:** 2026-09-26 23:49:10
**Benchmark Scope:** 10,000 Entities, 5-Fold GroupKFold CV, 31 Features

## 1. Reproduction Results

| Metric | Frozen Champion (Control) | Semantic Challenger (K=5 ZeroProtected) | Target | Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Macro F0.5** | **0.9566** | **0.9696** | ≈ 0.9701 | **+0.0129** |
| **Macro Precision** | 0.9738 | **0.9808** | ≈ 0.9813 | +0.0070 |
| **Macro Recall** | 0.9232 | **0.9469** | ≈ 0.9472 | +0.0237 |
| **Candidate Recall** | 94.97% | **98.05%** | ≈ 98.05% | **+3.09%** |
| **Avg Candidates / S1** | 111.3 | 112.7 | ~112.7 | +1.4 |
| **P50 Candidates** | 71.0 | 72.0 | - | - |
| **P95 Candidates** | 365.0 | 366.0 | - | - |
| **P99 Candidates** | 506.0 | 507.0 | - | - |
| **Optimal Threshold (tau)** | 0.62 | 0.68 | 0.68 | - |
| **5-Fold Mean +/- Std** | 0.9566 +/- 0.0029 | **0.9695 +/- 0.0020** | 0.9700 +/- 0.0019 | - |
| **Zero-Match Score** | 0.9497 | 0.9461 | ≈ 0.9479 | -0.0036 |
| **One-Match Score** | 0.8942 | 0.9238 | - | +0.0296 |
| **Multi-Match Score** | 0.9608 | 0.9738 | - | +0.0130 |
| **US F0.5** | 0.9713 | 0.9786 | - | +0.0073 |
| **India F0.5** | 0.9347 | 0.9560 | - | +0.0213 |
| **S2 F0.5** | 0.9388 | 0.9595 | - | +0.0207 |
| **S3 F0.5** | 0.9460 | 0.9605 | - | +0.0144 |
| **True Positives (TP)** | 31,870 | 32,690 | - | +820 |
| **False Positives (FP)** | 603 | 509 | - | -94 |
| **False Negatives (FN)** | 2,641 | 1,821 | - | -820 |
| **Runtime** | 70.7s | 83.7s | - | - |

## 2. Phase 1 Reproduction Verdict

**SUCCESS:** Challenger achieved Macro F0.5 = **0.9696** and Candidate Recall = **98.05%**, matching target within empirical tolerance (diff = 0.0005). Proceeding to next challenger phases.
