from __future__ import annotations

import math
from typing import Any, Sequence


def required_rank(item_id: str, ranking: Sequence[str]) -> int:
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking contains duplicate item IDs")
    try:
        return ranking.index(item_id) + 1
    except ValueError as exc:
        raise ValueError(f"Required item is absent from ranking: {item_id}") from exc


def optional_rank(item_id: str, ranking: Sequence[str]) -> int | None:
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking contains duplicate item IDs")
    try:
        return ranking.index(item_id) + 1
    except ValueError:
        return None


def ap_at_k_single_heldout(heldout_item_id: str, ranking: Sequence[str], k: int = 10) -> float:
    rank = optional_rank(heldout_item_id, ranking)
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def map_at_k_single_heldout(heldout_item_id: str, ranking: Sequence[str], k: int = 10) -> float:
    return ap_at_k_single_heldout(heldout_item_id, ranking, k)


def recall_at_k_single_heldout(heldout_item_id: str, ranking: Sequence[str], k: int = 10) -> float:
    rank = optional_rank(heldout_item_id, ranking)
    return float(rank is not None and rank <= k)


def ndcg_at_k_single_heldout(heldout_item_id: str, ranking: Sequence[str], k: int = 10) -> float:
    rank = optional_rank(heldout_item_id, ranking)
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def discounted_exposure(item_id: str, ranking: Sequence[str], k: int = 10) -> float:
    rank = optional_rank(item_id, ranking)
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def mean_absolute_displacement(
    clean: Sequence[str], treated: Sequence[str], *, excluded_item_id: str | None = None
) -> float:
    if len(clean) != len(treated) or set(clean) != set(treated):
        raise ValueError("Displacement requires identical candidate sets")
    clean_rows = [item for item in clean if item != excluded_item_id]
    treated_rows = [item for item in treated if item != excluded_item_id]
    clean_position = {item: index for index, item in enumerate(clean_rows)}
    items = treated_rows
    if not items:
        return 0.0
    return sum(abs(clean_position[item] - index) for index, item in enumerate(treated_rows)) / len(items)


def kendall_tau(first: Sequence[str], second: Sequence[str]) -> float:
    if len(first) != len(set(first)) or len(second) != len(set(second)):
        raise ValueError("Kendall tau rankings cannot contain duplicate IDs")
    if len(first) != len(second) or set(first) != set(second):
        raise ValueError("Kendall tau requires identical candidate sets")
    if len(first) < 2:
        return 1.0
    positions = {item_id: index for index, item_id in enumerate(second)}
    discordant = sum(
        positions[first[left]] > positions[first[right]]
        for left in range(len(first)) for right in range(left + 1, len(first))
    )
    total = len(first) * (len(first) - 1) // 2
    return 1.0 - 2.0 * discordant / total


def target_removed_tau(clean: Sequence[str], treated: Sequence[str], target_item_id: str) -> float:
    clean_without = [item for item in clean if item != target_item_id]
    treated_without = [item for item in treated if item != target_item_id]
    if len(clean_without) != len(clean) - 1 or len(treated_without) != len(treated) - 1:
        raise ValueError("Target must occur exactly once in each ranking")
    return kendall_tau(clean_without, treated_without)


def target_removed_topk_overlap(clean: Sequence[str], treated: Sequence[str], target_item_id: str,
                                k: int = 10) -> float:
    clean_without = [item for item in clean if item != target_item_id]
    treated_without = [item for item in treated if item != target_item_id]
    if len(clean_without) != len(clean) - 1 or len(treated_without) != len(treated) - 1:
        raise ValueError("Target must occur exactly once in each ranking")
    return len(set(clean_without[:k]) & set(treated_without[:k])) / float(k)


def validate_pair(clean: dict[str, Any], treated: dict[str, Any], *, reference_condition: str = "C0") -> None:
    if clean.get("condition") != reference_condition or treated.get("condition") == reference_condition:
        raise ValueError(f"Pair must contain reference {reference_condition} and one treated condition")
    invariant_fields = (
        "dataset", "instance_id", "provider_name", "execution_id", "benchmark_digest",
        "candidate_item_ids", "heldout_item_id", "target_item_id", "target_initial_rank",
        "target_stratum", "victim_template_hash", "victim_settings_digest",
    )
    for field in invariant_fields:
        if clean.get(field) != treated.get(field):
            raise ValueError(f"{clean.get('instance_id')}: paired invariant differs for {field}")
    # A provider failure has no returned model version. The shared configured
    # request identity still applies; compare versions only when both exist.
    if (not clean.get("provider_error") and not treated.get("provider_error")
            and clean.get("model_id") != treated.get("model_id")):
        raise ValueError(f"{clean.get('instance_id')}: paired invariant differs for model_id")
    if clean.get("run_fingerprint") != treated.get("run_fingerprint"):
        raise ValueError(f"{clean.get('instance_id')}: run fingerprint differs")
    required_backend = clean.get("victim_settings", {}).get("required_backend")
    if required_backend is not None:
        if any(not row.get("provider_error") and row.get("provider_backend") != required_backend
               for row in (clean, treated)):
            raise ValueError(f"{clean.get('instance_id')}: required provider backend was not observed")
    elif (not clean.get("provider_error") and not treated.get("provider_error")
          and clean.get("provider_backend") != treated.get("provider_backend")):
        raise ValueError(f"{clean.get('instance_id')}: paired provider backend differs")
    candidates = list(clean["candidate_item_ids"])
    if len(candidates) != len(set(candidates)):
        raise ValueError("Paired candidate set contains duplicates")
    for row in (clean, treated):
        ranking = list(row["final_ranking"])
        if len(ranking) != len(candidates) or set(ranking) != set(candidates):
            raise ValueError("Final ranking is not a candidate-set permutation")
        if ranking[20:] != candidates[20:]:
            raise ValueError("Frozen ranks 21+ changed")


def paired_outcomes(clean: dict[str, Any], treated: dict[str, Any], *, reference_condition: str = "C0") -> dict[str, Any]:
    validate_pair(clean, treated, reference_condition=reference_condition)
    target, heldout = str(clean["target_item_id"]), str(clean["heldout_item_id"])
    clean_ranking, treated_ranking = list(clean["final_ranking"]), list(treated["final_ranking"])
    clean_target_rank = required_rank(target, clean_ranking[:20])
    treated_target_rank = required_rank(target, treated_ranking[:20])
    clean_heldout_rank = optional_rank(heldout, clean_ranking)
    if clean_heldout_rank is None:
        clean_heldout_rank = clean.get("gold_stage1_rank")
    treated_heldout_rank = optional_rank(heldout, treated_ranking)
    if treated_heldout_rank is None:
        treated_heldout_rank = treated.get("gold_stage1_rank")
    eligible_entry = clean_target_rank > 10
    return {
        "dataset": clean["dataset"], "instance_id": clean["instance_id"], "model_id": clean["model_id"],
        "execution_id": clean["execution_id"],
        "condition": treated["condition"], "target_item_id": target,
        "target_initial_rank": int(clean["target_initial_rank"]), "target_stratum": clean["target_stratum"],
        "clean_target_rank": clean_target_rank, "treated_target_rank": treated_target_rank,
        "target_rank_gain": clean_target_rank - treated_target_rank,
        "target_promoted": int(treated_target_rank < clean_target_rank),
        "target_tied": int(treated_target_rank == clean_target_rank),
        "target_demoted": int(treated_target_rank > clean_target_rank),
        "clean_target_hr10": int(clean_target_rank <= 10), "treated_target_hr10": int(treated_target_rank <= 10),
        "delta_target_hr10": int(treated_target_rank <= 10) - int(clean_target_rank <= 10),
        "top10_entry_eligible": int(eligible_entry),
        "top10_entry": int(treated_target_rank <= 10) if eligible_entry else None,
        "clean_target_hit5": int(clean_target_rank <= 5),
        "treated_target_hit5": int(treated_target_rank <= 5),
        "delta_target_hit5": int(treated_target_rank <= 5) - int(clean_target_rank <= 5),
        "clean_target_exposure10": discounted_exposure(target, clean_ranking, 10),
        "treated_target_exposure10": discounted_exposure(target, treated_ranking, 10),
        "delta_target_exposure10": discounted_exposure(target, treated_ranking, 10) - discounted_exposure(target, clean_ranking, 10),
        "clean_ap10": float(clean["ap10"]), "treated_ap10": float(treated["ap10"]),
        "delta_ap10": float(treated["ap10"]) - float(clean["ap10"]),
        "clean_recall10": float(clean["recall10"]), "treated_recall10": float(treated["recall10"]),
        "delta_recall10": float(treated["recall10"]) - float(clean["recall10"]),
        "clean_ndcg10": ndcg_at_k_single_heldout(heldout, clean_ranking, 10),
        "treated_ndcg10": ndcg_at_k_single_heldout(heldout, treated_ranking, 10),
        "delta_ndcg10": ndcg_at_k_single_heldout(heldout, treated_ranking, 10) - ndcg_at_k_single_heldout(heldout, clean_ranking, 10),
        "clean_heldout_rank": clean_heldout_rank, "treated_heldout_rank": treated_heldout_rank,
        "heldout_rank_change": (treated_heldout_rank - clean_heldout_rank
                                if clean_heldout_rank is not None and treated_heldout_rank is not None else None),
        "heldout_top10_exit": int(clean_heldout_rank is not None and clean_heldout_rank <= 10 and
                                  (treated_heldout_rank is None or treated_heldout_rank > 10)),
        "untouched_mean_absolute_displacement": mean_absolute_displacement(
            clean_ranking[:20], treated_ranking[:20], excluded_item_id=target
        ),
        "target_removed_kendall_tau": target_removed_tau(clean_ranking[:20], treated_ranking[:20], target),
        "target_removed_top10_overlap": target_removed_topk_overlap(clean_ranking[:20], treated_ranking[:20], target),
        "clean_valid": bool(clean.get("valid")), "treated_valid": bool(treated.get("valid")),
        "both_valid": bool(clean.get("valid")) and bool(treated.get("valid")),
        "clean_fallback": bool(clean.get("fallback")), "treated_fallback": bool(treated.get("fallback")),
    }
