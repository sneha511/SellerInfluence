"""Recompute study results from saved observations, never from saved scorecards."""
import csv
from collections import defaultdict
from pathlib import Path

from .analysis_core import *
from .data import write_json
from .supplemental import supplemental


def make_pairs(rows):
    index = {(r["dataset"], r["instance_id"], r["condition"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate main observation")
    result = []
    for reference, treatment in CONTRASTS:
        for (dataset, instance, condition), row in sorted(index.items()):
            if condition != treatment:
                continue
            base = index[dataset, instance, reference]
            if row["provider_error"] or base["provider_error"]:
                continue
            pair = paired_outcomes(base, row, reference_condition=reference)
            pair.update(model=row["model"], reference=reference, cohort_origin=row["cohort_origin"], route=row["route"],
                        head_tau=kendall_tau(base["final_ranking"], row["final_ranking"]),
                        head_mad=mean_absolute_displacement(base["final_ranking"], row["final_ranking"]),
                        top10_overlap=len(set(base["final_ranking"][:10]) & set(row["final_ranking"][:10])) / 10)
            result.append(pair)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reproduce(bundle, out, *, infer=False):
    out = Path(out)
    if out.resolve().is_relative_to(bundle.root.resolve()):
        raise ValueError("Write generated results outside the frozen benchmark")
    out.mkdir(parents=True, exist_ok=True)
    main, repeats = [], []
    for raw in bundle.observations():
        row = bundle.hydrate(raw)
        (main if raw["replicate"] == "main" else repeats).append(row)
    keys = [(r["model"],r["dataset"],r["instance_id"],r["condition"],r["replicate"]) for r in main+repeats]
    if len(set(keys)) != len(keys) or len(main) != bundle.manifest["observation_counts"]["main"] or len(repeats) != 720:
        raise ValueError("Reproduction requires a complete, nonduplicated nine-profile main/repeat sweep")
    groups = defaultdict(list)
    for row in main:
        groups[row["model"]].append(row)
    pairs = [p for model in sorted(groups) for p in make_pairs(groups[model])]
    indexed = {(r["model"], r["dataset"], r["instance_id"], r["replicate"]): r
               for r in main + repeats if r["condition"] == "C0"}
    roster = sorted({(r["model"], r["dataset"], r["instance_id"]) for r in repeats})
    repeat_rows = []
    for model, dataset, instance in roster:
        for left, right in (("main", "r1"), ("main", "r2"), ("r1", "r2")):
            a, b = indexed[model, dataset, instance, left], indexed[model, dataset, instance, right]
            if a["provider_error"] or b["provider_error"]:
                continue
            pair = clean_repeat_pair(a, b)
            repeat_rows.append(pair | dict(model=model, reference=left, repeat=right,
                ranking_identical=a["final_ranking"] == b["final_ranking"],
                absolute_rank_change=abs(pair["target_rank_gain"]),
                head_tau=kendall_tau(a["final_ranking"], b["final_ranking"])))
    print(f"Recomputed {len(pairs)} paired outcomes; inference={infer}", flush=True)
    grouped = cohort_tables(main, pairs, infer=infer)
    tables = {name: [row for cohort in grouped.values() for row in cohort[name]]
              for name in ("condition_scorecard", "paired_scorecard", "paired_statistics", "both_valid_sensitivity", "utility_visibility")}
    tables.update(supplemental(bundle, main, pairs, repeat_rows, infer=infer))
    for rows in tables.values():
        for row in rows:
            row.pop("mean_reported_response_cost_usd", None)
    # Row-level derived outputs are generated, not distributed in the release.
    write_csv(out / "paired_instances.csv", pairs)
    write_csv(out / "clean_repeat_instances.csv", repeat_rows)
    for name, rows in tables.items():
        write_csv(out / (name + ".csv"), rows)
    write_json(out / "tables.json", tables)
    summary = dict(contexts=len(bundle.contexts), main_observations=len(main), repeats=len(repeats),
                   paired_observations=len(pairs), inference=infer, paid_calls=0)
    write_json(out / "reproduction.json", summary)
    return summary
