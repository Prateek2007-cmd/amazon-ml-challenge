import os, sys, time, re
from collections import defaultdict

BASE_DIR = r"c:\Users\Prateek\Downloads\ml\amazon-ml-challenge"
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
PHASE8_SRC = os.path.join(BASE_DIR, "experiments", "final_challenger", "phase8", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PHASE8_SRC)

from data_loader import get_data_paths, parse_ground_truth_fast
from phase8_core import (
    precompute_entity, extract_champion_keys, precompute_semantic_index
)

def main():
    paths = get_data_paths(is_sample=False)
    print("1. Loading 10,000 validation ground truth entities...", flush=True)
    gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
    total_true = sum(len(m) for m in gt_dict.values())
    s1_needed = set(gt_dict.keys())
    needed_s2, needed_s3 = set(), set()
    for matches in gt_dict.values():
        for m in matches:
            if m.startswith("S2-"): needed_s2.add(m)
            elif m.startswith("S3-"): needed_s3.add(m)

    s1_dict = {}
    with open(paths["train_s1"], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in s1_needed:
                s1_dict[parts[0]] = dict(zip(header, parts))
    s1_list = sorted(s1_dict.keys())

    s2_dict, s3_dict = {}, {}
    for path, d, needed in [(paths["train_s2"], s2_dict, needed_s2), (paths["train_s3"], s3_dict, needed_s3)]:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            cnt = 0
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                if eid in needed or cnt < 100000:
                    d[eid] = dict(zip(header, parts))
                    cnt += 1

    cached_s1 = {sid: precompute_entity(row) for sid, row in s1_dict.items()}
    cached_cands = {}
    for eid, row in s2_dict.items(): cached_cands[eid] = precompute_entity(row)
    for eid, row in s3_dict.items(): cached_cands[eid] = precompute_entity(row)

    idx_s2, idx_s3 = defaultdict(list), defaultdict(list)
    for eid, row in s2_dict.items():
        for pass_keys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pass_keys: idx_s2[k].append(eid)
    for eid, row in s3_dict.items():
        for pass_keys in extract_champion_keys(row["business_name"], row["business_address"], row.get("country", "")):
            for k in pass_keys: idx_s3[k].append(eid)

    MAX_CANDS = 120
    champion_cands = {}
    for s1_id in s1_list:
        e1 = cached_s1[s1_id]
        ordered_passes = [
            e1["pass1_keys"], e1["pass2_keys"], e1["pass3_keys"],
            e1["pass4_keys"], e1["pass5_keys"], e1["pass6_keys"]
        ]
        cands = set()
        for p_idx, pass_keys in enumerate(ordered_passes):
            cap = 150 if p_idx == 4 else 400
            for k in pass_keys:
                p2 = idx_s2.get(k, [])
                p3 = idx_s3.get(k, [])
                if len(p2) <= cap: cands.update(p2)
                if len(p3) <= cap: cands.update(p3)
                if len(cands) >= MAX_CANDS: break
            if len(cands) >= MAX_CANDS: break
        champion_cands[s1_id] = cands

    del idx_s2, idx_s3

    s1_countries = {sid: str(s1_dict[sid].get("country", "")).strip().upper() for sid in s1_list}
    cand_countries = {}
    for eid, row in s2_dict.items(): cand_countries[eid] = str(row.get("country", "")).strip().upper()
    for eid, row in s3_dict.items(): cand_countries[eid] = str(row.get("country", "")).strip().upper()

    print("Precomputing semantic index...", flush=True)
    sem_top_records = precompute_semantic_index(s1_dict, s1_list, s2_dict, s3_dict, s1_countries, cand_countries)

    # Union candidates
    cands_D = defaultdict(set)
    for sid in s1_list:
        cands_D[sid] = set(champion_cands[sid])
        if len(champion_cands[sid]) > 0:
            for cid, (score, rank) in sem_top_records.get(sid, {}).items():
                if rank <= 10 and score >= 0.30:
                    cands_D[sid].add(cid)

    # Misses
    misses = []
    for sid in s1_list:
        t_set = gt_dict.get(sid, set())
        c_set = cands_D.get(sid, set())
        for cid in t_set:
            if cid not in c_set:
                e1 = cached_s1[sid]
                e2 = cached_cands.get(cid)
                sem_res = sem_top_records.get(sid, {}).get(cid, (0.0, 999))
                misses.append({
                    "s1_id": sid,
                    "cand_id": cid,
                    "s1_name": e1["name_norm"],
                    "cand_name": e2["name_norm"] if e2 else "NOT_FOUND",
                    "s1_addr": e1["addr_norm"][:35],
                    "cand_addr": e2["addr_norm"][:35] if e2 else "",
                    "s1_country": e1["country"],
                    "cand_country": e2["country"] if e2 else "",
                    "sem_cos": sem_res[0],
                    "sem_rank": sem_res[1],
                })

    print(f"\n=======================================================")
    print(f"CANDIDATE MISSES FORENSICS: Total {len(misses)} missed pairs")
    print(f"=======================================================")
    
    in_sem_pool = [m for m in misses if m["sem_rank"] < 999]
    top_20 = [m for m in in_sem_pool if m["sem_rank"] <= 20]
    top_30 = [m for m in in_sem_pool if m["sem_rank"] <= 30]
    top_50 = [m for m in in_sem_pool if m["sem_rank"] <= 50]
    cos_ge_20 = [m for m in misses if m["sem_cos"] >= 0.20]
    cos_ge_25 = [m for m in misses if m["sem_cos"] >= 0.25]
    zero_champ = [m for m in misses if len(champion_cands[m["s1_id"]]) == 0]

    print(f"  Total Misses: {len(misses)}")
    print(f"  Zero-Champ S1 entities (blocking found 0): {len(zero_champ)}")
    print(f"  Present in TF-IDF index at all: {len(in_sem_pool)}")
    print(f"  Rank <= 20: {len(top_20)}")
    print(f"  Rank <= 30: {len(top_30)}")
    print(f"  Rank <= 50: {len(top_50)}")
    print(f"  Cosine >= 0.25: {len(cos_ge_25)}")
    print(f"  Cosine >= 0.20: {len(cos_ge_20)}")

    print("\nSample Misses (First 20):")
    for m in misses[:20]:
        print(f"S1: [{m['s1_name']}] | CAND: [{m['cand_name']}] | Sem rank: {m['sem_rank']} cos: {m['sem_cos']:.3f} | Addr1: [{m['s1_addr']}] vs Addr2: [{m['cand_addr']}]")

if __name__ == "__main__":
    main()
