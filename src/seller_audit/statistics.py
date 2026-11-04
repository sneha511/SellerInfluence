"""Frozen clustered inference and multiplicity corrections."""
from __future__ import annotations
import math
from collections import defaultdict
from statistics import mean, median
from typing import Hashable, Sequence
from .fast_statistics import target_cluster_bootstrap, _cluster_signflip_pvalue

def _two_sided_sign_pvalue(values: Sequence[float]) -> float:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(positive, negative) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_summary(values: Sequence[float], clusters: Sequence[Hashable], *, seed: int = 17,
                   draws: int = 2000, null_value: float = 0.0) -> dict[str, float | int]:
    if len(values) != len(clusters) or not values:
        raise ValueError("Paired summary requires equal non-empty values and clusters")
    numeric = [float(value) for value in values]
    centered = [value - float(null_value) for value in numeric]
    low, high = target_cluster_bootstrap(numeric, clusters, seed=seed, draws=draws)
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for value, cluster in zip(numeric, clusters):
        grouped[cluster].append(value)
    target_means = [mean(group) for group in grouped.values()]
    return {
        "n_instances": len(numeric), "n_target_clusters": len(grouped),
        "statistical_null": float(null_value),
        "instance_weighted_mean": mean(numeric),
        "instance_weighted_effect_from_null": mean(centered), "median": median(numeric),
        "target_cluster_bootstrap_ci95_low": low, "target_cluster_bootstrap_ci95_high": high,
        "target_cluster_signflip_pvalue": _cluster_signflip_pvalue(centered, clusters, seed=seed),
        "instance_sign_test_pvalue_exploratory": _two_sided_sign_pvalue(centered),
        "equal_target_sensitivity_mean": mean(target_means),
        "equal_target_sign_test_pvalue": _two_sided_sign_pvalue(
            [value - float(null_value) for value in target_means]
        ),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: float(p_values[index]))
    adjusted = [1.0] * count
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, min(1.0, float(p_values[index]) * count / rank))
        adjusted[index] = running
    return adjusted
