"""
Candidate generation / blocking.

Goal: cheap, high-recall filtering so the pairwise classifier only has
to look at a plausible subset of Source2/Source3 records for each
Source1 entity, instead of the full cross product.

Blocking recall is the ceiling on your final score -- a true match
that never becomes a candidate can't be recovered downstream. Always
measure it (see `blocking_recall` below) on your own validation split
before trusting anything past this stage.
"""
from collections import defaultdict

from normalize import normalize_name, name_tokens


def _block_keys(normalized_name: str):
    """Blocking keys for one normalized name. A record only needs to
    match on ONE of these to become a candidate -- deliberately loose,
    trading precision (bigger candidate sets) for recall. Precision
    gets recovered later by the classifier + threshold."""
    tokens = sorted(name_tokens(normalized_name))
    keys = set()

    if tokens:
        keys.add(("tok0", tokens[0]))  # first token alone
        if len(tokens) >= 2:
            keys.add(("tok2", tokens[0], tokens[1]))  # sorted 2-token signature
        prefix = "".join(tokens)[:3]  # catches near-dupes with odd tokenization
        if prefix:
            keys.add(("prefix3", prefix))
    return keys


def build_blocking_index(df, id_col="entity_id", name_col="business_name"):
    """Inverted index: blocking key -> list of entity_ids in this df."""
    index = defaultdict(list)
    for _, row in df.iterrows():
        norm = normalize_name(row[name_col])
        for key in _block_keys(norm):
            index[key].append(row[id_col])
    return index


def generate_candidates(s1_df, other_index, name_col="business_name", id_col="entity_id"):
    """For every Source1 row, look up candidates from `other_index`
    (built over Source2 or Source3) via shared blocking keys."""
    candidates = {}
    for _, row in s1_df.iterrows():
        s1_id = row[id_col]
        norm = normalize_name(row[name_col])
        seen = set()
        for key in _block_keys(norm):
            seen.update(other_index.get(key, []))
        candidates[s1_id] = seen
    return candidates


def merge_candidate_sets(*candidate_dicts):
    """Combine per-source candidate dicts (e.g. from Source2 and
    Source3) into one dict keyed by S1 id -> combined candidate set."""
    merged = defaultdict(set)
    for cd in candidate_dicts:
        for s1_id, cand_set in cd.items():
            merged[s1_id] |= cand_set
    return merged


def blocking_recall(candidates, ground_truth):
    """ground_truth: dict s1_id -> set of true matched ids.
    Returns (recall, missed_pairs). This is the number to check before
    trusting anything downstream of blocking."""
    total_true = 0
    total_found = 0
    missed = []
    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        for match_id in true_set:
            total_true += 1
            if match_id in cand_set:
                total_found += 1
            else:
                missed.append((s1_id, match_id))
    recall = total_found / total_true if total_true else 1.0
    return recall, missed
