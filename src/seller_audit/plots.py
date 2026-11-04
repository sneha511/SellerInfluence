"""Generate result visualizations on demand; no rendered figures are distributed."""
import json
import os

DOMAINS = {"hnm":"H&M", "amazon_fashion":"Amazon Fashion", "amazon_grocery_2023":"Amazon Grocery",
           "amazon_beauty_and_personal_care_2023":"Amazon Beauty"}
MODELS = {"gpt41mini":"GPT-4.1 mini", "luna":"GPT-5.6 Luna", "qwen35_9b":"Qwen3.5 9B",
          "qwen35_27b":"Qwen3.5 27B", "gemma3_4b":"Gemma3 4B", "gemma3_12b":"Gemma3 12B",
          "gemma3_27b":"Gemma3 27B", "muse_glimmer":"Muse Glimmer", "haiku55":"Claude Haiku 5.5"}


def plot(output):
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tables = json.loads((output / "tables.json").read_text(encoding="utf-8"))
    for condition, title in (("C2", "Persuasive rewrite"), ("C4", "Explicit instruction")):
        rows = [r for r in tables["paired_scorecard"] if r["cohort"] == "new_only"
                and r["reference"] == "C1" and r["condition"] == condition]
        models, datasets = list(MODELS), list(DOMAINS)
        index = {(r["model"], r["dataset"]): r for r in rows}
        fig, ax = plt.subplots(figsize=(10, 5))
        values = [[index[m, d]["target_rank_gain"] for d in datasets] for m in models]
        import numpy as np
        limit = max(abs(v) for row in values for v in row) or 1
        im = ax.imshow(np.array(values), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(datasets)), [DOMAINS[d] for d in datasets])
        ax.set_yticks(range(len(models)), [MODELS[m] for m in models])
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center")
        ax.set_title(title + ": mean target rank gain versus neutral rewrite")
        fig.colorbar(im, ax=ax, label="Positive = seller promotion")
        fig.tight_layout()
        fig.savefig(output / (condition + "_rank_gain.pdf"))
        plt.close(fig)
    datasets, models = list(DOMAINS), list(MODELS)
    colors = {"C2":"#007c91", "C4":"#a84519"}
    # Ranking levels first, adjusted differences second: the baseline is explicit.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    levels = {(r["model"],r["dataset"],r["condition"]):r for r in tables["condition_scorecard"] if r["cohort"]=="new_only"}
    for ax, dataset in zip(axes.flat,datasets):
        for model in models:
            ax.plot(range(4), [levels[model,dataset,c]["target_rank"] for c in ("C0","C1","C2","C4")], marker=".",label=MODELS[model])
        ax.set_xticks(range(4),["Original","Neutral","Persuasive","Explicit"],rotation=15)
        ax.set_title(DOMAINS[dataset])
        ax.set_ylabel("Mean seller-target rank")
        ax.grid(axis="y",alpha=.2)
    axes[0,0].invert_yaxis()
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="lower center",ncol=5,fontsize=8)
    fig.suptitle("Ranking levels: lower rank means higher seller placement")
    fig.tight_layout(rect=(0,.09,1,.95))
    fig.savefig(output / "ranking_levels.pdf")
    plt.close(fig)
    # Conditional denominator is promoted pairs, not every evaluated user.
    fig, ax = plt.subplots(figsize=(9,4))
    labels, fractions, barcolors = [], [], []
    for dataset in datasets:
        for condition in ("C2","C4"):
            rows = [r for r in tables["utility_visibility"] if r["cohort"]=="new_only"
                    and r["dataset"]==dataset and r["condition"]==condition]
            numerator = sum(r["promoted_unchanged_ap_recall"] for r in rows)
            denominator = sum(r["n_promoted"] for r in rows)
            labels.append(DOMAINS[dataset]+"\n"+("Persuasive" if condition=="C2" else "Explicit"))
            fractions.append(100*numerator/denominator if denominator else 0)
            barcolors.append(colors[condition])
    ax.bar(range(len(labels)),fractions,color=barcolors)
    ax.set_xticks(range(len(labels)),labels,rotation=20,ha="right",fontsize=8)
    ax.set_ylim(0,100)
    ax.set_ylabel("Promotions with unchanged\nAP/Recall@10 (%)")
    ax.set_title("Seller visibility and sparse ground-truth relevance can diverge")
    fig.tight_layout()
    fig.savefig(output / "visibility_relevance.pdf")
    plt.close(fig)
    # Four pre-existing rank bands, with per-cell means rather than a causal position claim.
    fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True)
    bands = ["rank_1_5","rank_6_10","rank_11_15","rank_16_20"]
    for ax,dataset in zip(axes.flat,datasets):
        for model in models:
            rows = {r["target_stratum"]:r for r in tables["rank_stratum_scorecard"]
                    if r["cohort"]=="new_only" and r["dataset"]==dataset and r["model"]==model
                    and r["reference"]=="C1" and r["condition"]=="C4"}
            ax.plot(range(4),[rows[b]["target_rank_gain"] if b in rows else float("nan") for b in bands],marker=".",label=MODELS[model])
        ax.axhline(0,color="gray",lw=.7)
        ax.set_title(DOMAINS[dataset])
        ax.set_xticks(range(4),["1-5","6-10","11-15","16-20"])
        ax.set_xlabel("Original target rank band")
        ax.set_ylabel("Explicit vs. neutral rank gain")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=5,fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(output / "initial_position.pdf")
    plt.close(fig)
    if tables["paired_statistics"]:
        stats = {(r["model"],r["dataset"],r["condition"]):r for r in tables["paired_statistics"]
                 if r["cohort"]=="new_only" and r["metric"]=="target_rank_gain"}
        fig, axes = plt.subplots(2,4,figsize=(12,6),sharex=False)
        for y,(family,ladder) in enumerate((("Qwen3.5",[("qwen35_9b",9),("qwen35_27b",27)]),
                                          ("Gemma3",[("gemma3_4b",4),("gemma3_12b",12),("gemma3_27b",27)]))):
            for x,d in enumerate(datasets):
                ax=axes[y,x]
                for c in ("C2","C4"):
                    rr=[stats[m,d,c] for m,size in ladder]
                    means=[r["instance_weighted_mean"] for r in rr]
                    # Quantile intervals are drawn directly; they need not be symmetric.
                    xx=[size for m,size in ladder]
                    ax.plot(xx,means,marker="o",color=colors[c],label="Persuasive" if c=="C2" else "Explicit")
                    ax.vlines(xx,[r["target_cluster_bootstrap_ci95_low"] for r in rr],
                              [r["target_cluster_bootstrap_ci95_high"] for r in rr],color=colors[c],alpha=.7)
                ax.set_xticks([size for m,size in ladder])
                ax.set_title(DOMAINS[d]+" / "+family,fontsize=9)
                ax.axhline(0,color="gray",lw=.6)
                ax.set_xlabel("Parameters (billions)")
                if x==0: ax.set_ylabel("Target rank gain vs. neutral")
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=2)
        fig.suptitle("Within-family serving-profile comparisons (95% target-cluster intervals)",fontsize=11)
        fig.tight_layout(rect=(0,.06,1,.95))
        fig.savefig(output / "scale_ladders.pdf")
        plt.close(fig)
