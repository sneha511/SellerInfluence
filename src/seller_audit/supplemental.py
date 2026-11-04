"""Frozen route, length, rank, steer, repeat and size-ladder sensitivities."""
from .analysis_core import *

def supplemental(bundle, all_rows, all_pairs, repeat_rows, *, infer):
    tables = {}
    route_groups=defaultdict(list)
    for cohort,origin in (("full",None),("new_only","new_p2")):
        for row in all_pairs:
            if row["model"] in ("qwen35_27b", "gemma3_4b") and row["reference"]=="C1" and (origin is None or row["cohort_origin"]==origin):
                route_groups[cohort,row["model"],row["dataset"],row["condition"],row["route"]].append(row)
    tables["route_sensitivity"]=[dict(zip(("cohort","model","dataset","condition","route"),key),n=len(group),
        **{metric:average(group,metric) for metric in EFFECTS}) for key,group in sorted(route_groups.items())]
    route_conditions=defaultdict(list)
    for cohort,origin in (("full",None),("new_only","new_p2")):
        for row in all_rows:
            if row["model"] in ("qwen35_27b", "gemma3_4b") and (origin is None or row["cohort_origin"]==origin):
                route_conditions[cohort,row["model"],row["dataset"],row["condition"],row["route"]].append(row)
    tables["route_condition_scorecard"]=[condition_summary(model,dataset,condition,group)|dict(cohort=cohort,route=route)
        for (cohort,model,dataset,condition,route),group in sorted(route_conditions.items())]
    reps = defaultdict(list)
    for r in repeat_rows:
        reps[r["model"], r["dataset"], r["reference"], r["repeat"]].append(r)
    tables["clean_repeat_scorecard"] = []
    for m in bundle.manifest["models"]:
        for d in bundle.manifest["datasets"]:
            for l,r in (("main","r1"),("main","r2"),("r1","r2")):
                g=reps.get((m,d,l,r),[])
                tables["clean_repeat_scorecard"].append(dict(model=m,dataset=d,reference=l,repeat=r,
                    planned_n=10,n=len(g),missing_pair_n=10-len(g),
                    **{k:average(g,k) for k in (*EFFECTS,"absolute_rank_change","ranking_identical","head_tau","both_valid")}))
    lengths = {(d, t, c): v["description_lengths"][c] for (d, t), v in bundle.variants.items() for c in ("C1", "C2", "C4")}
    strata, lengths_groups, steer_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    steer_index = {k: v["steers"] for k, v in bundle.variants.items()}
    for cohort, origin in (("full", None), ("new_only", "new_p2")):
        pairs = [r for r in all_pairs if origin is None or r["cohort_origin"] == origin]
        for r in pairs:
            k = (cohort, r["model"], r["dataset"], r["reference"], r["condition"])
            strata[(*k, r["target_stratum"])].append(r)
            if r["reference"] == "C1":
                lengths_groups[k].append(r)
                if r["condition"] == "C2":
                    for steer in steer_index[r["dataset"], r["target_item_id"]]:
                        steer_groups[(*k, steer)].append(r)
    def descriptive(groups, extra):
        return [dict(zip(("cohort", "model", "dataset", "reference", "condition", extra), k), n=len(g),
                     **{metric: average(g, metric) for metric in EFFECTS}) for k, g in sorted(groups.items())]
    tables["rank_stratum_scorecard"] = descriptive(strata, "target_stratum")
    tables["steer_descriptive"] = descriptive(steer_groups, "steer")
    tables["length_sensitivity"] = []
    for k, g in sorted(lengths_groups.items()):
        restricted = [r for r in g if .8 <= lengths[r["dataset"], r["target_item_id"], r["condition"]] / max(1, lengths[r["dataset"], r["target_item_id"], "C1"]) <= 1.2]
        tables["length_sensitivity"].append(dict(zip(("cohort", "model", "dataset", "reference", "condition"), k),
            full_n=len(g), restricted_n=len(restricted), **{metric: average(restricted, metric) for metric in EFFECTS}))
    ladders = []
    for cohort, origin in (("full", None), ("new_only", "new_p2")):
        index = {(r["model"], r["dataset"], r["instance_id"]): r for r in all_pairs if r["reference"] == "C1" and r["condition"] == "C2" and (origin is None or r["cohort_origin"] == origin)}
        local = []
        for low, high in LADDERS:
            for dataset in bundle.manifest["datasets"]:
                shared = [(d, i) for m, d, i in index if m == low and d == dataset and (high, d, i) in index]
                for metric in EFFECTS[:2]:
                    if shared and infer:
                        local.append(dict(cohort=cohort, low=low, high=high, dataset=dataset, metric=metric,
                            **paired_summary([index[high, d, i][metric] - index[low, d, i][metric] for d, i in shared],
                                             [index[low, d, i]["target_item_id"] for d, i in shared], seed=20260923)))
        if len(local) == 32:
            for r, q in zip(local, benjamini_hochberg([r["target_cluster_signflip_pvalue"] for r in local])):
                r.update(bh_qvalue=q, bh_family=cohort + "_exploratory_ladder_32")
        ladders.extend(local)
    tables["ladder_comparisons"] = ladders
    # Stage1 ceiling is computed independently for each explicitly named cohort.
    ceilings = []
    for cohort, origin in (("full", None), ("new_only", "new_p2"), ("prior_p1", "reused_p1")):
        for d in bundle.manifest["datasets"]:
            rows = [r for r in all_rows if r["model"] == "gpt41mini" and r["condition"] == "C0" and r["dataset"] == d and (origin is None or r["cohort_origin"] == origin)]
            ranks = [r["gold_stage1_rank"] for r in rows]
            ceilings.append(dict(cohort=cohort, dataset=d, n=len(rows), map10=mean(1/r if r and r <= 10 else 0 for r in ranks),
                recall10=mean(bool(r and r <= 10) for r in ranks), recall20_ceiling=mean(bool(r and r <= 20) for r in ranks),
                recall100=mean(r is not None for r in ranks), gold_missing=sum(r is None for r in ranks)))
    tables["stage1_ceiling"] = ceilings
    return tables
