"""
Phase 8: Entity Competition & Margin Calibration Engine
Amazon ML Challenge 2026 - Business Entity Resolution

Analyzes candidate competition within each S1 entity group to:
1. Recover near-miss true matches where top candidate is dominant but p slightly below tau.
2. Filter out noisy false positives caused by locality/branch collisions with PIN conflicts.
"""
import os, sys, time, pickle
from collections import defaultdict
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PHASE8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(PHASE8_DIR, "artifacts")
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import get_data_paths, parse_ground_truth_fast
from phase8_core import evaluate_comprehensive_full

def evaluate_predictions_entity(gt_dict, preds, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p, tau):
    return evaluate_comprehensive_full(gt_dict, preds, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p, tau)

def run_entity_competition_sweep(chal_pairs, oof_p, cached_s1, cached_cands, gt_dict, s1_dict, splits_chal, groups_chal, base_tau=0.72):
    """
    Sweeps margin rescue thresholds on top of base threshold predictions.
    """
    print("=" * 80)
    print("ENTITY COMPETITION & TOP-MARGIN CALIBRATION SWEEP")
    print(f"Base Threshold: {base_tau:.2f}")
    print("=" * 80)

    # Group candidate predictions by S1 entity
    entity_cand_probs = defaultdict(list)
    for (sid, cid), p in zip(chal_pairs, oof_p):
        entity_cand_probs[sid].append((cid, p))

    # Sort candidates per entity descending by probability
    entity_sorted = {}
    for sid, c_list in entity_cand_probs.items():
        entity_sorted[sid] = sorted(c_list, key=lambda x: x[1], reverse=True)

    # 1. Evaluate baseline at base_tau
    base_preds = defaultdict(set)
    for (sid, cid), p in zip(chal_pairs, oof_p):
        if p >= base_tau: base_preds[sid].add(cid)

    m_base = evaluate_predictions_entity(gt_dict, base_preds, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p, base_tau)
    print(f"BASELINE (Tau={base_tau:.2f}): Macro F0.5 = {m_base['macro_f05']:.4f} | Prec = {m_base['precision']:.4f} | Rec = {m_base['recall']:.4f} | FP = {m_base['fp']} | FN = {m_base['fn']}")

    # 2. Sweep rescue configurations
    best_macro = m_base["macro_f05"]
    best_config = None
    best_m = m_base

    tau_rescues = [0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    margins = [0.10, 0.15, 0.20, 0.25, 0.30]

    for tau_r in tau_rescues:
        for min_m in margins:
            preds = defaultdict(set)
            # Add base predictions
            for (sid, cid), p in zip(chal_pairs, oof_p):
                if p >= base_tau: preds[sid].add(cid)

            # Apply margin rescue ONLY for entities with 0 base predictions
            rescued_count = 0
            for sid, c_list in entity_sorted.items():
                if len(preds[sid]) == 0 and len(c_list) > 0:
                    top_cid, top_p = c_list[0]
                    sec_p = c_list[1][1] if len(c_list) > 1 else 0.0
                    margin = top_p - sec_p
                    if top_p >= tau_r and margin >= min_m:
                        e1 = cached_s1.get(sid, {})
                        e2 = cached_cands.get(top_cid, {})
                        pin1 = e1.get("pincode", "")
                        pin2 = e2.get("pincode", "")
                        # Reject if PIN contradiction
                        if pin1 and pin2 and pin1 != pin2:
                            continue
                        preds[sid].add(top_cid)
                        rescued_count += 1

            m = evaluate_predictions_entity(gt_dict, preds, s1_dict, splits_chal, chal_pairs, groups_chal, oof_p, base_tau)
            is_better = m["macro_f05"] > best_macro
            mark = " *** NEW BEST ***" if is_better else ""
            print(f"  Rescue Tau={tau_r:.2f} Margin={min_m:.2f} (rescued {rescued_count:4d}) -> Macro F0.5={m['macro_f05']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | FP={m['fp']} | FN={m['fn']}{mark}")

            if is_better:
                best_macro = m["macro_f05"]
                best_config = (tau_r, min_m, rescued_count)
                best_m = m

    print("\n" + "=" * 80)
    if best_config:
        print(f"BEST COMPETITION CONFIG: Rescue Tau={best_config[0]:.2f}, Min Margin={best_config[1]:.2f} ({best_config[2]} rescued)")
        print(f"  Macro F0.5:  {best_m['macro_f05']:.4f} (Baseline: {m_base['macro_f05']:.4f})")
        print(f"  Precision:   {best_m['precision']:.4f} (Baseline: {m_base['precision']:.4f})")
        print(f"  Recall:      {best_m['recall']:.4f} (Baseline: {m_base['recall']:.4f})")
        print(f"  5-Fold Mean: {best_m['fold_mean']:.4f} +/- {best_m['fold_std']:.4f}")
        print(f"  Zero-Match:  {best_m['zero']:.4f}")
        print(f"  One-Match:   {best_m['one']:.4f}")
        print(f"  Multi-Match: {best_m['multi']:.4f}")
        print(f"  US:          {best_m['us']:.4f}")
        print(f"  India:       {best_m['india']:.4f}")
        print(f"  TP: {best_m['tp']:,d} | FP: {best_m['fp']:,d} | FN: {best_m['fn']:,d}")
    else:
        print("Base threshold was already optimal; no rescue configuration improved Macro F0.5.")
    print("=" * 80)

    return best_config, best_m

if __name__ == "__main__":
    pass
