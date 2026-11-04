"""Vectorized arithmetic with the historical Python RNG draw sequence unchanged."""
import math
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np


def _group(values, clusters):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(float(value))
    return list(grouped.values())


@lru_cache(maxsize=16)
def _bootstrap_indices(count, seed, draws):
    rng = random.Random(seed)
    ids = range(count)
    result = np.fromiter((rng.choice(ids) for _ in range(draws * count)), dtype=np.int32,
                         count=draws * count).reshape(draws, count)
    result.flags.writeable = False
    return result


def target_cluster_bootstrap(values, clusters, *, seed=17, draws=2000):
    if len(values) != len(clusters) or not values:
        raise ValueError("Cluster bootstrap requires equal non-empty values and clusters")
    groups = _group(values, clusters)
    sums = np.array([math.fsum(g) for g in groups])
    sizes = np.array([len(g) for g in groups])
    idx = _bootstrap_indices(len(groups), seed, draws)
    estimates = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975], method="linear"))


@lru_cache(maxsize=16)
def _signs(count, seed, draws):
    rng = random.Random(seed)
    out = np.fromiter((1 if rng.getrandbits(1) else -1 for _ in range(draws * count)),
                      dtype=np.int8, count=draws * count).reshape(draws, count)
    out.flags.writeable = False
    return out


def _cluster_signflip_pvalue(values, clusters, *, seed, draws=20_000):
    groups = _group(values, clusters)
    contributions = [sum(g) for g in groups]
    observed = abs(sum(contributions))
    if observed == 0:
        return 1.0
    count = len(groups)
    if count <= 20:
        # Exactly the original exhaustive branch, including its tie tolerance.
        extreme = sum(abs(sum(v if mask & (1 << i) else -v for i, v in enumerate(contributions)))
                      >= observed - 1e-12 for mask in range(1 << count))
        return extreme / (1 << count)
    signs = _signs(count, seed, draws)
    values_array = np.array(contributions)
    extreme = 0
    for start in range(0, draws, 256):
        statistics = (signs[start:start + 256] * values_array).sum(axis=1)
        # Recheck threshold-near values with the original summation order.
        near = np.flatnonzero(np.abs(np.abs(statistics) - observed) < 1e-9)
        for j in near:
            statistics[j] = sum(v * int(s) for v, s in zip(contributions, signs[start + j]))
        extreme += int((np.abs(statistics) >= observed - 1e-12).sum())
    return (extreme + 1) / (draws + 1)
