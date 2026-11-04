# Seller influence: reproducible evaluation results

Offline reproduction package for **Beyond Relevance: Auditing Seller Influence
in Personalized LLM Re-Ranking**. It provides per-context numerical outcomes
from the final nine-model study, with code to recompute the analyses.

## Quick start

Use Python 3.11 or later on a CPU machine. Extract `data.zip` into the repository
root so that `results_data/manifest.json` exists, then run:

```sh
python -m zipfile -e data.zip .
python -m venv .venv
# Activate: Windows .venv\Scripts\Activate.ps1; Unix source .venv/bin/activate
python -m pip install ".[test,plots]"
python -m pytest -q
seller-audit validate --data results_data --out output
seller-audit reproduce --data results_data --out output --inference
seller-audit plot --out output
```

For point estimates alone, omit `--inference`. All analysis commands operate
offline on supplied records; no API credentials or paid requests are needed.

## What you can reproduce

- Seller-target rank, discounted exposure, promotion/demotion and entry rates.
- MAP@10, Recall@10 and NDCG@10 using the recorded relevant item's position.
- Relative-order changes among unedited candidates and top-ten overlap.
- Paired original/neutral/persuasive/explicit comparisons, including the joint
  visibility--relevance counts (from generated `paired_instances.csv`).
- Target-cluster bootstrap intervals, paired sign-flip tests and the frozen
  multiple-testing corrections.
- Descriptive results by domain, model, starting rank, steer, description
  length, serving route, and development/evaluation cohort; clean repetitions
  and exploratory within-family size contrasts.

Metric values are computed from the saved scores, output permutations and
evaluation labels, not loaded from precomputed scorecards. Historical parser
and failure classifications are retained as metadata. Numerical validation
checks the scores, stable-tie ordering, fallback permutations, cohort coverage
and per-context relevance labels.

## Code layout

```text
.
|-- README.md
|-- RIGHTS.md                         # Provenance and full-benchmark access
|-- LICENSE                           # MIT: code
|-- LICENSE_DATA.md                   # CC BY 4.0: derived numerical records
|-- pyproject.toml
|-- .gitignore
|-- data.zip                          # Numerical records; tracked in Git
|-- src/seller_audit/
|   |-- __init__.py
|   |-- cli.py                        # validate / reproduce / plot
|   |-- data.py                       # Strict numeric schema and record loading
|   |-- metrics.py                    # Ranking, relevance and stability metrics
|   |-- analysis.py                   # Pair observations and write tables
|   |-- analysis_core.py              # Frozen estimands and cohort definitions
|   |-- statistics.py                 # Inference and multiplicity correction
|   |-- fast_statistics.py            # Deterministic numerical acceleration
|   |-- supplemental.py               # Supplementary analyses
|   `-- plots.py                      # Generate numeric plots
`-- tests/test_release.py
```

## Data layout

Extracting `data.zip` creates:

```text
results_data/
|-- manifest.json
|-- profiles.json
|-- LICENSE_DATA.md
|-- hnm/
|   |-- contexts.jsonl.gz
|   `-- target_metadata.jsonl.gz
|-- amazon_fashion/                   # Same two files
|-- amazon_grocery_2023/              # Same two files
|-- amazon_beauty_and_personal_care_2023/  # Same two files
`-- observations/                    # One .jsonl.gz per evaluated model
```

Each gzip-compressed JSON Lines file contains one JSON object per line.

| Record | Stored fields and purpose |
|---|---|
| Context | Opaque context/target-cluster IDs, initial target rank, relevant-item rank and cohort label; supports pairing and relevance evaluation |
| Target metadata | Three assigned persuasion steers and the four description lengths; supports clustered inference and descriptive analyses |
| Observation | Model/profile, condition, repeat, parsed numeric scores, zero-based candidate-index permutation, saved utility checks and failure/response diagnostics |
| Profile | Recorded model identifiers and inference/serving settings |

Candidate indices `0..19` refer to first-stage positions `1..20`, not source
product identifiers. Target-cluster IDs preserve repeated-target membership
across contexts. A null relevant-item rank means absent from the original
top 100; positions 21--100 stay fixed and are represented only by that scalar.
Description lengths are Unicode character counts of the actual frozen strings.

The numerical release covers **3,919 contexts**: **719 development** and
**3,200 evaluation** (800 per domain), with **2,800 distinct seller targets**.
It includes **141,084 main outcomes** and **720 additional original-description
repeats**. The primary evaluation uses `new_only` (3,200 contexts / 115,200 planned
outcomes); `prior_p1` and `full` retain development and combined views explicitly.
Development and evaluation contexts are disjoint, but some target products recur.

Conditions in the machine-readable records are `C0` (original), `C1` (neutral),
`C2` (persuasive), and `C4` (explicit instructions). Comparisons use only available
matched responses; malformed returned responses retain first-stage fallback.
Rank gain is reference rank minus treated rank, so positive means promotion.
Three jointly assigned steers are descriptive labels, not isolated causal effects.

## Outputs and access

Commands write CSV/JSON scorecards, row-level paired outcomes, statistical
summaries and six numeric plot PDFs to `output/`. Extracted data and generated
outputs are ignored by Git; `data.zip` is explicitly allowed.

For research access to the full description/context benchmark, contact the
authors through the paper or the venue's anonymous communication channel,
subject to source-data permissions. See `RIGHTS.md` for source attribution.
