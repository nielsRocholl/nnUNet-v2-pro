"""Stratified batch sampling by (dataset, size_bin)."""
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def build_stratum_weights(
    strata: Dict[Tuple[str, str], List[str]],
    dataset_weights: Dict[str, float],
    size_bin_weights: Dict[str, float],
) -> Dict[Tuple[str, str], float]:
    datasets = {k[0] for k in strata}
    bins = {k[1] for k in strata}
    def_d = 1.0 / len(datasets) if datasets else 1.0
    def_s = 1.0 / len(bins) if bins else 1.0
    return {
        k: dataset_weights.get(k[0], def_d) * size_bin_weights.get(k[1], def_s)
        for k in strata
    }


def build_strata(case_stats: dict, tr_keys: list) -> Dict[Tuple[str, str], List[str]]:
    strata = defaultdict(list)
    tr_set = set(tr_keys)
    for case_id, info in case_stats.items():
        if case_id == "_metadata" or case_id not in tr_set:
            continue
        key = (info["dataset"], info["size_bin"])
        strata[key].append(case_id)
    return {k: v for k, v in strata.items() if v}


def sample_batch(
    strata: Dict[Tuple[str, str], List[str]],
    batch_size: int,
    weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[str]:
    if not strata:
        raise ValueError("strata is empty")
    stratum_keys = list(strata.keys())
    if weights is None:
        random.shuffle(stratum_keys)
        batch = []
        for key in stratum_keys:
            if len(batch) >= batch_size:
                break
            batch.append(random.choice(strata[key]))
        while len(batch) < batch_size:
            key = random.choice(stratum_keys)
            batch.append(random.choice(strata[key]))
    else:
        w = [weights.get(k, 1.0) for k in stratum_keys]
        total = sum(w)
        if total <= 0:
            probs = [1.0 / len(stratum_keys)] * len(stratum_keys)
        else:
            probs = [x / total for x in w]
        batch = []
        for _ in range(batch_size):
            key = random.choices(stratum_keys, weights=probs, k=1)[0]
            batch.append(random.choice(strata[key]))
    random.shuffle(batch)
    return batch[:batch_size]
