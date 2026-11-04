import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest

from seller_audit.cli import validate_observation
from seller_audit.data import Benchmark, digest
from seller_audit.metrics import (ap_at_k_single_heldout, recall_at_k_single_heldout,
    discounted_exposure, target_removed_tau, mean_absolute_displacement, paired_outcomes)
from seller_audit.statistics import paired_summary, benjamini_hochberg


def observation():
    return dict(condition="C0", replicate="main", ranking_indices=list(range(20)),
        parsed_scores={str(i): 20-i for i in range(20)}, valid=True, fallback=False,
        abstained=False, provider_error=False, api_embedded_error=False,
        refusal_recorded=False, refusal_text_heuristic=False, short_incomplete_output=False)


def test_valid_scores_and_stable_ties():
    r = observation()
    validate_observation(r)
    r["parsed_scores"] = [1]*20
    validate_observation(r)
    r["ranking_indices"] = list(reversed(range(20)))
    with pytest.raises(ValueError, match="Scores/output order"):
        validate_observation(r)


@pytest.mark.parametrize("permutation", [[0]*20, list(range(19)), list(range(21)), [True]+list(range(1,20))])
def test_reject_bad_permutation(permutation):
    r = observation() | {"ranking_indices": permutation}
    with pytest.raises(ValueError, match="permutation"):
        validate_observation(r)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_reject_bad_numeric_scores(value):
    r = observation()
    r["parsed_scores"]["0"] = value
    with pytest.raises(ValueError):
        validate_observation(r)


def test_invalid_output_uses_saved_fallback():
    r = observation() | dict(valid=False, fallback=True, parsed_scores={})
    validate_observation(r)
    with pytest.raises(ValueError):
        validate_observation(r | dict(fallback=False))
    with pytest.raises(ValueError):
        validate_observation(r | dict(ranking_indices=list(reversed(range(20)))))


def test_unavailable_is_not_valid():
    r = observation() | dict(provider_error=True, valid=False, parsed_scores={})
    validate_observation(r)
    with pytest.raises(ValueError):
        validate_observation(r | dict(valid=True))


@pytest.mark.parametrize("gold,ap,recall", [(1,1,1),(10,.1,1),(11,0,0),(77,0,0),(None,0,0)])
def test_hydrate_from_numeric_context_only(gold,ap,recall):
    b = object.__new__(Benchmark)
    b.contexts = {("d","u00001"):dict(target_cluster_id="t00001",target_initial_rank=20,
        target_stratum="rank_16_20",gold_stage1_rank=gold,cohort_origin="new_p2")}
    b.profiles = {"p":dict(settings={})}
    b.manifest = dict(scientific_digest="frozen")
    raw = observation() | dict(dataset="d",instance_id="u00001",model="m",profile_id="p")
    before = copy.deepcopy(raw)
    r = b.hydrate(raw)
    assert r["ap10"] == ap and r["recall10"] == recall
    assert r["final_ranking"][19] == "t00001" and raw == before
    assert r["prompt_hash"] == digest(["d","u00001","C0"])


def test_metrics_and_missing_gold():
    order = [str(i) for i in range(20)]
    assert ap_at_k_single_heldout("9",order) == .1
    assert recall_at_k_single_heldout("10",order) == 0
    assert ap_at_k_single_heldout("absent",order) == 0
    assert discounted_exposure("0",order) == 1
    promoted = [order[-1]] + order[:-1]
    assert target_removed_tau(order,promoted,"19") == 1
    assert mean_absolute_displacement(order,promoted,excluded_item_id="19") == 0


def test_pair_profile_guard_and_tail_scalar():
    ranking = [str(i) for i in range(20)]
    row = dict(condition="C0", dataset="d", instance_id="u", target_item_id="19", heldout_item_id="gold",
        target_initial_rank=20, target_stratum="rank_16_20", candidate_item_ids=ranking, final_ranking=ranking,
        model_id="m", execution_id="p", victim_settings_digest="p", run_fingerprint="p", victim_settings={},
        ap10=0., recall10=0., valid=True, fallback=False, gold_stage1_rank=77)
    other = row | dict(condition="C2", final_ranking=["19"] + ranking[:-1])
    pair = paired_outcomes(row,other)
    assert pair["target_rank_gain"] == 19
    assert pair["clean_heldout_rank"] == 77 and pair["treated_heldout_rank"] == 77
    assert pair["heldout_rank_change"] == 0 and pair["target_removed_kendall_tau"] == 1
    with pytest.raises(ValueError,match="victim_settings_digest"):
        paired_outcomes(row,other | dict(victim_settings_digest="different"))


def test_inference_zero_and_bh():
    s = paired_summary([0]*30, list(range(30)), draws=50)
    assert s["target_cluster_signflip_pvalue"] == 1
    assert s["target_cluster_bootstrap_ci95_low"] == 0
    assert benjamini_hochberg([.01,.04,.1]) == pytest.approx([.03,.06,.1])


def test_vectorized_inference_preserves_rng_sequence():
    from seller_audit.fast_statistics import _bootstrap_indices, _signs
    rng = random.Random(17)
    expected = [[rng.choice(list(range(23))) for _ in range(23)] for _ in range(80)]
    assert np.array_equal(expected,_bootstrap_indices(23,17,80))
    rng = random.Random(17)
    expected = [[1 if rng.getrandbits(1) else -1 for _ in range(23)] for _ in range(80)]
    assert np.array_equal(expected,_signs(23,17,80))


def test_inference_cluster_aliases_preserve_results():
    values, clusters = [1,1,-1,-1],["source_a","source_a","source_b","source_b"]
    original = paired_summary(values,clusters,draws=200)
    anonymized = paired_summary(values,["t00001","t00001","t00002","t00002"],draws=200)
    assert original == anonymized
    assert original["n_target_clusters"] == 2
    assert original["target_cluster_bootstrap_ci95_low"] == -1
    assert original["target_cluster_bootstrap_ci95_high"] == 1


def test_reject_full_benchmark_schema(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(schema_version="seller_audit_compact_v1")))
    with pytest.raises(ValueError, match="results-only"):
        Benchmark(tmp_path)


def test_installed_package_has_no_private_or_live_runtime():
    import seller_audit
    root = Path(seller_audit.__file__).parent
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from recsys" not in text and "import recsys" not in text
        assert "artifacts/seller_influence" not in text
        assert "import openai" not in text and "from openai" not in text
    assert not (root / "evaluation.py").exists()
