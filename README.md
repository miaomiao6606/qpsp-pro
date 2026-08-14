# qPSP-Pro

Minimal code for the four-architecture qPSP-Pro simulation experiment.

## Install

```bash
python -m pip install .
```

## Included data

- [`data/pathways.json.gz`](data/pathways.json.gz): the 1,323-pathway analysis catalogue used in the paper
- [`data/pathways_metadata.tsv.gz`](data/pathways_metadata.tsv.gz): pathway source, collection, entry type and gene count
- [`data/reactome_fi_score075.tsv.gz`](data/reactome_fi_score075.tsv.gz): the 278,974 Reactome Functional Interaction edges used in the paper

Load the pathway catalogue and interaction network:

```python
import gzip
import json

import pandas as pd

with gzip.open("data/pathways.json.gz", "rt", encoding="utf-8") as handle:
    pathways = json.load(handle)

pathway_metadata = pd.read_csv("data/pathways_metadata.tsv.gz", sep="\t")
reactome_fi = pd.read_csv("data/reactome_fi_score075.tsv.gz", sep="\t")
```

The catalogue contains 761 parent pathways and 562 observed-protein subpathways from Reactome, KEGG Medicus, KEGG Legacy and Hallmark.

## Input

- `expression`: protein-by-sample abundance `DataFrame`
- `pathways`: `{pathway: [protein, ...]}`
- `groups`: `{sample: group}`
- `contrast`: `(test_group, reference_group)`
- `blocks`: `{sample: participant}` for repeated measurements
- `ppi`: interaction `DataFrame` with `gene1`, `gene2` and `score`

Protein identifiers are uppercased during matching. Rank quantiles are calculated from proteins observed across the contrast samples.

## Run qPSP-Pro

```python
import pandas as pd

from qpsp_pro import qpsp_pro

expression = pd.read_csv("expression.tsv", sep="\t", index_col=0)
sample_table = pd.read_csv("samples.tsv", sep="\t")
groups = dict(zip(sample_table["sample"], sample_table["group"]))
blocks = dict(zip(sample_table["sample"], sample_table["participant"]))

result = qpsp_pro(
    expression,
    pathways,
    groups,
    ("case", "control"),
    blocks=blocks,
    topology="weighted_degree",
    ppi=reactome_fi,
    ppi_threshold=0.75,
    topology_strength=0.5,
    tau=0.85,
    kappa=30,
    n_permutations=9999,
)

result.pathway_scores.to_csv("pathway_scores.tsv", sep="\t")
result.significance.to_csv("pathway_significance.tsv", sep="\t", index=False)
result.member_effects.to_csv("member_effects.tsv", sep="\t", index=False)
```

`pathway_scores` contains sample-level qPSP-Pro pathway scores. `significance` reports pathway-score, net-shift and directional-discordance permutation tests with Benjamini–Hochberg correction. `member_effects` reports member-protein contributions to the pathway-architecture summaries.

## Calculation

For each sample, abundance ranks are transformed to quantiles `(rank - 1) / (n - 1)`. The logistic kernel `sigmoid(kappa * (quantile - tau))` emphasizes the upper abundance range.

For topology weighting, qPSP-Pro extracts each pathway's internal Reactome FI subnetwork at confidence `>= 0.75`, calculates confidence-weighted degree, normalizes degree to mean one and applies `lambda = 0.5` shrinkage toward uniform weight. The pathway score is the kernel- and topology-weighted mean rank quantile.

The simulation also derives topology-weighted net shift, absolute shift, directional discordance and discordance fraction from protein-level between-group rank shifts. Repeated measurements are averaged by participant before group permutation.

## Simulated application

[`examples/simulate_expression.py`](examples/simulate_expression.py) loads the packaged pathway catalogue and Reactome FI network. It selects 40 nonoverlapping catalogue pathways with 10–20 Reactome-mapped proteins, five signal pathways, 120 background proteins, 16 participants per group and two visits per participant. It adds pathway-correlated abundance, participant random intercepts, sample offsets and abundance-dependent background missingness.

The four signal architectures are:

| Architecture | Group A signal |
|---|---|
| `coordinated_up` | every pathway member `+0.62` |
| `coordinated_down` | every pathway member `-0.62` |
| `cancellation` | first half `+0.76`, second half `-0.76` |
| `core_shift` | highest-degree member and its two strongest neighbours `+1.05` |

Run the simulation:

```bash
python examples/simulate_expression.py
```

The script generates expression values for each architecture, calls the compact simulation implementation and prints pathway recovery, effect size, net shift and discordance fraction. Each inferential test uses 9,999 permutations.
