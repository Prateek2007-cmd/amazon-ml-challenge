import sys, os
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code", "business_entity_resolution", "src"))
sys.path.insert(0, SRC_DIR)
from data_loader import get_data_paths, parse_ground_truth_fast

paths = get_data_paths(is_sample=False)
s1_check = {"S1-193682202", "S1-777597828", "S1-65263544", "S1-348059257", "S1-740600107"}
s1_data = {}
with open(paths["train_s1"], "r", encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[0] in s1_check:
            s1_data[p[0]] = dict(zip(header, p))

gt_dict = parse_ground_truth_fast(paths["train_gt"], nrows=10000)
needed_cand = set()
for sid in s1_check:
    needed_cand.update(gt_dict.get(sid, set()))

cand_data = {}
for src in ["train_s2", "train_s3"]:
    with open(paths[src], "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in needed_cand:
                cand_data[p[0]] = dict(zip(header, p))

for sid, s1 in s1_data.items():
    print(f"\n--- S1: {sid} ---")
    print(f"  Name: {s1['business_name']}")
    print(f"  Addr: {s1['business_address']}")
    print(f"  Country: {s1['country']}")
    true_matches = gt_dict.get(sid, set())
    print(f"  True matches ({len(true_matches)}):")
    for tid in true_matches:
        c = cand_data.get(tid, {})
        print(f"    [{tid}] Name: {c.get('business_name')} | Addr: {c.get('business_address')}")
