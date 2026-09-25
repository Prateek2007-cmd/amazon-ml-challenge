"""
Test blocking keys and posting sizes.
"""
import re
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))
from normalize import normalize_name, normalize_address

s1_name = "Blue Network Corporation"
s1_addr = "177 1st Avenue, Nashville, TN"
tgt_name = "The Blue Netw0rk Corporation"
tgt_addr = "177 1TH AVENUE, NASHVILLE, TN"

def get_keys(name, addr, country="US"):
    norm_n = normalize_name(name)
    norm_a = normalize_address(addr)
    
    # Clean noise like 'd/b/a', 'the', etc.
    name_tokens = [t for t in norm_n.split() if t not in ("the", "and", "inc", "corp", "llc", "ltd", "co", "pvt")]
    addr_tokens = [t for t in norm_a.split() if t not in ("st", "rd", "ave", "blvd", "dr", "ln", "apt", "ste")]
    addr_nums = [t for t in addr_tokens if t.isdigit()]
    addr_words = [t for t in addr_tokens if not t.isdigit() and len(t) >= 3]
    
    keys = set()
    # 1. Name 2-gram or individual non-stop tokens
    for t in name_tokens:
        if len(t) >= 3:
            keys.add(("name_tok", t))
    if len(name_tokens) >= 2:
        keys.add(("name_pair", tuple(sorted([name_tokens[0], name_tokens[1]]))))
        
    # 2. Address number + first address word (e.g. 177 + nashville, or 31 + floyd, 9308 + home)
    if addr_nums and addr_words:
        for num in addr_nums[:2]:
            for w in addr_words[:3]:
                keys.add(("num_word", num, w))
                
    # 3. Name token + Address number (e.g. blue + 177, novent + 9308, probst + 31)
    if name_tokens and addr_nums:
        for nt in name_tokens[:2]:
            if len(nt) >= 3:
                for num in addr_nums[:2]:
                    keys.add(("name_num", nt, num))
                    
    return keys

k1 = get_keys(s1_name, s1_addr)
k2 = get_keys(tgt_name, tgt_addr)
overlap = k1 & k2
print(f"S1 keys count: {len(k1)}")
print(f"Target keys count: {len(k2)}")
print(f"Shared keys ({len(overlap)}): {overlap}")
