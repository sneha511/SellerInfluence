"""Offline validation and statistical reproduction of the numerical release."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from .data import Benchmark, CONDITIONS, STEERS, write_json


def validate_observation(raw):
    if raw["condition"] not in CONDITIONS or raw["replicate"] not in {"main", "r1", "r2"}:
        raise ValueError("Invalid condition/replicate")
    indices = raw["ranking_indices"]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(20)):
        raise ValueError("Invalid output permutation")
    for k in ("valid", "fallback", "abstained", "provider_error", "api_embedded_error",
              "refusal_recorded", "refusal_text_heuristic", "short_incomplete_output"):
        if type(raw[k]) is not bool:
            raise ValueError("Invalid outcome flag: " + k)
    scores = raw["parsed_scores"] or {}
    if isinstance(scores, list):
        scores = dict(enumerate(scores))
    elif isinstance(scores, dict):
        scores = {int(k): v for k, v in scores.items()}
    else:
        raise ValueError("Invalid score container")
    if any(i not in range(20) or type(v) not in (int, float) or not math.isfinite(v) for i, v in scores.items()):
        raise ValueError("Non-finite/invalid parsed score")
    if raw["provider_error"]:
        if raw["valid"]:
            raise ValueError("Unavailable response marked valid")
    elif raw["valid"]:
        if set(scores) != set(range(20)) or raw["fallback"]:
            raise ValueError("Valid response has missing scores or fallback")
        expected = sorted(range(20), key=lambda i: -scores[i])
        if indices != expected:
            raise ValueError("Scores/output order disagree (ties use original order)")
    elif not raw["fallback"] or indices != list(range(20)):
        raise ValueError("Malformed response did not preserve historical fallback")
    return scores


def validate(bundle):
    bundle.verify_files()
    if tuple(bundle.manifest["conditions"]) != CONDITIONS:
        raise ValueError("Condition set differs")
    for (dataset, instance), c in bundle.contexts.items():
        if not re.fullmatch(r"u\d{5}", instance) or not re.fullmatch(r"t\d{5}", c["target_cluster_id"]):
            raise ValueError("Expected opaque context/target identifiers")
        rank, gold = c["target_initial_rank"], c["gold_stage1_rank"]
        if type(rank) is not int or not 1 <= rank <= 20:
            raise ValueError("Invalid target rank")
        if gold is not None and (type(gold) is not int or not 1 <= gold <= 100 or gold == rank):
            raise ValueError("Invalid relevant-item position")
        lo = 1 + 5 * ((rank - 1) // 5)
        if c["target_stratum"] != f"rank_{lo}_{lo+4}" or c["cohort_origin"] not in {"new_p2", "reused_p1"}:
            raise ValueError("Invalid rank stratum/cohort")
        if (dataset, c["target_cluster_id"]) not in bundle.variants:
            raise ValueError("Missing target metadata")
    for (dataset, target), v in bundle.variants.items():
        if len(v["steers"]) != 3 or len(set(v["steers"])) != 3 or set(v["steers"]) - STEERS:
            raise ValueError("Expected three distinct registered steers")
        lengths = v["description_lengths"]
        if set(lengths) != set(CONDITIONS) or any(type(n) is not int or n < 1 for n in lengths.values()):
            raise ValueError("Invalid description-length metadata")
    seen, counts, memberships = set(), Counter(), {}
    failures, malformed = 0, 0
    for raw in bundle.observations():
        validate_observation(raw)
        key = tuple(raw[k] for k in ("model", "dataset", "instance_id", "condition", "replicate"))
        if key in seen:
            raise ValueError("Duplicate observation")
        seen.add(key)
        profile = bundle.profiles[raw["profile_id"]]
        if profile["model"] != raw["model"]:
            raise ValueError("Model/profile mismatch")
        expected_route = profile["settings"].get("extra_body", {}).get("provider", {}).get("only", ["openai"])[0]
        if raw["route"] != expected_route:
            raise ValueError("Recorded route/profile mismatch")
        group = key[:3]
        if group in memberships and memberships[group] != raw["profile_id"]:
            raise ValueError("Serving profile changed within a matched context")
        memberships[group] = raw["profile_id"]
        row = bundle.hydrate(raw)
        if any(abs(row[k] - raw[k]) > 1e-12 for k in ("ap10", "recall10")):
            raise ValueError("Saved utility disagrees with recorded permutation")
        failures += int(raw["provider_error"])
        malformed += int(not raw["provider_error"] and not raw["valid"])
        counts[raw["replicate"]] += 1
    if dict(counts) != bundle.manifest["observation_counts"]:
        raise ValueError("Observation count mismatch")
    for model in bundle.manifest["models"]:
        for dataset, instance in bundle.contexts:
            if any((model, dataset, instance, c, "main") not in seen for c in CONDITIONS):
                raise ValueError("Missing planned main outcome")
        roster = []
        for rep in ("r1", "r2"):
            members = {(d,i) for m,d,i,c,r in seen if m == model and r == rep and c == "C0"}
            if len(members) != 40 or any(bundle.contexts[d,i]["cohort_origin"] != "new_p2" for d,i in members):
                raise ValueError("Clean-repeat roster changed")
            roster.append(members)
        if roster[0] != roster[1]:
            raise ValueError("Repeats do not use identical contexts")
    if (len(bundle.contexts) != bundle.manifest["contexts"] or len(bundle.variants) != bundle.manifest["targets"]
        or sum(c["cohort_origin"] == "new_p2" for c in bundle.contexts.values()) != bundle.manifest["new_contexts"]):
        raise ValueError("Context/target/cohort counts differ")
    return dict(status="PASS", contexts=len(bundle.contexts), targets=len(bundle.variants),
                observations=len(seen), counts=dict(counts), unavailable=failures,
                returned_malformed=malformed, paid_calls=0)


def main():
    parser = argparse.ArgumentParser(description="Offline numerical reproduction; no network/API execution")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "reproduce", "plot"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("results_data"))
        p.add_argument("--out", type=Path, default=Path("output"))
        if name == "reproduce":
            p.add_argument("--inference", action="store_true", help="Recompute frozen bootstrap/sign-flip tests")
    args = parser.parse_args()
    if args.command == "plot":
        from .plots import plot
        plot(args.out)
        return
    bundle = Benchmark(args.data)
    if args.out.resolve().is_relative_to(bundle.root.resolve()):
        raise ValueError("Write outputs outside frozen data")
    result = validate(bundle)
    write_json(args.out / "validation.json", result)
    if args.command == "reproduce":
        from .analysis import reproduce
        result = reproduce(bundle, args.out, infer=args.inference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
