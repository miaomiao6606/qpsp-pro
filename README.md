# qPSP-Pro

qPSP-Pro calculates sample-level pathway scores, between-group pathway tests,
and signed coherence summaries from a protein-by-sample abundance matrix.

## Install

Requires Python 3.10 or newer. From the repository directory:

```bash
python -m pip install .
```

## Prepare inputs

- `expression.tsv`: numeric abundances; first column contains unique protein
  symbols and remaining columns are samples. Missing values may be blank.
- `pathways.json`: a mapping such as `{"PATHWAY_A": ["GENE1", "GENE2", "GENE3"]}`.
  Supply your own pathway definitions.
- `samples.tsv`: columns `sample` and `group`; add `participant` for repeated
  measurements. Every expression sample must have a group assignment.
- An interaction TSV with `gene1`, `gene2`, `score` columns when using
  `topology="weighted_degree"`. Gzip-compressed TSV files are accepted.
  A public Reactome FI network is available at `data/reactome_fi_score075.tsv.gz`.

Protein symbols are stripped and uppercased. Use matching identifiers across
expression, pathways and interactions. Keep study inputs outside this repository.

## Run an independent-group comparison

```python
import json
import pandas as pd
from qpsp_pro import qpsp_pro

expression = pd.read_csv("/path/to/expression.tsv", sep="\t", index_col=0)
with open("/path/to/pathways.json", encoding="utf-8") as handle:
    pathways = json.load(handle)
samples = pd.read_csv("/path/to/samples.tsv", sep="\t")
groups = dict(zip(samples["sample"], samples["group"]))

result = qpsp_pro(
    expression=expression,
    pathways=pathways,
    groups=groups,
    contrast=("case", "control"),  # test group, reference group
    topology="weighted_degree",
    ppi_path="data/reactome_fi_score075.tsv.gz",
    ranking_universe="analysis_complete_case",
    kernel="logistic",
    tau=0.85,
    kappa=30,
    score_denominator="kernel_weighted",
    n_permutations=9999,
    seed=42,
)

result.pathway_scores.to_csv("/path/to/pathway_scores.tsv", sep="\t")
result.significance.to_csv("/path/to/significance.tsv", sep="\t", index=False)
result.member_effects.to_csv("/path/to/member_effects.tsv", sep="\t", index=False)
```

Use keyword arguments: the current API takes `contrast` before `groups`.
Interaction input uses `ppi_path`, rather than a `ppi` DataFrame.
For uniform weights, set `topology="uniform"` and omit `ppi_path`.

## Repeated measurements

Add these arguments to the call above:

```python
blocks = dict(zip(samples["sample"], samples["participant"]))
# qpsp_pro(..., design="clustered", blocks=blocks)
```

Set both `design="clustered"` and `blocks=blocks`. Scores and rank quantiles
are averaged within participant before testing. Each participant must belong
to one group. Paired within-participant contrasts are not supported.
`pathway_scores` retains sample-level columns.

## Options

| Argument | Values and behavior |
|---|---|
| `ranking_universe` | Default `"analysis_complete_case"` retains proteins observed in every input sample. `"contrast_complete_case"` fits on the two contrast groups only. `"per_sample"` and `"shared"` permit sample-specific ranking denominators and are sensitivity options when detection depth differs. |
| `ranking_min_coverage` | Detection fraction for the `"shared"` universe; default `0.5`. |
| `kernel` | `"logistic"` (default), `"measurement_calibrated"`, or `"identity"` (kernel ablation). |
| `tau`, `kappa` | Logistic threshold and steepness; defaults `0.85`, `30`. |
| `topology` | `"weighted_degree"` (default) or `"uniform"`. |
| `topology_strength` | Shrinkage from uniform to topology weights, between 0 and 1; default `0.5`. |
| `ppi_threshold` | Minimum retained interaction confidence; default `0.75`. |
| `score_denominator` | `"kernel_weighted"` (default), `"valid_topology"`, `"fixed_mapped"`, or `"complete_case"`. The alternatives change the score definition; `"complete_case"` divides by topology weights, not kernel weights. |
| `pathway_test` | `"permutation"` (default) or `"welch"`. |
| `scd` | `"permutation"` (default), `"descriptive"`, or `"none"`. |
| `scd_min_effect` | Minimum absolute net shift for practical net-shift classification; default `0.01`. Does not change p-values. |
| `n_permutations` | Default `9999`; exact enumeration when all assignments fit within this count. |
| `fdr_method`, `fdr_threshold` | `"BH"` or `"BY"`; default threshold `0.05`. |
| `hypothesis_families` | `"separate"` (default) corrects pathway-score tests separately from the combined net-shift and discordance tests; `"pooled"` combines all three. |
| `min_detected`, `min_coverage` | Required pathway proteins: maximum of `min_detected` and rounded-up coverage times pathway size; defaults `3`, `0.1`. |
| `min_group_units` | Minimum analysis units per group; default `3`. |
| `min_protein_obs_per_group` | Minimum protein observations per group for SCD; default `3`. |

For measurement calibration, pass quantile anchors and positive weights
estimated from your own technical repeats:

```python
# Illustrative values only; replace with your estimated anchors.
# qpsp_pro(...,
#     kernel="measurement_calibrated",
#     kernel_quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
#     kernel_weights=(0.1, 0.2, 0.4, 0.8, 1.0),
# )
```

## Read results

- `pathway_scores`: pathways by samples.
- `significance`: score differences (`score_diff`), Hedges' g, permutation
  p-values and adjusted q-values, net shift, absolute shift (`abs_shift`),
  discordance, discordance fraction and `scd_category`.
- `member_effects`: each protein's rank shift and weighted contribution to
  the signed decomposition.
- `kernel_weights`, `topology_weights`: weights used in scoring.
- `params`, `diagnostics`: settings, coverage, test status and permutation details.

Discordance inference columns are `p_discordant` and `q_discordant`.
Check `mapping_status`, `pathway_test_status` and `scd_test_status` before
interpreting missing statistics. Output tables can contain sample identifiers;
write them to a private analysis location.

## Run the synthetic example

All proteins, pathways, interactions and participants in this example are
artificial. It requires no study data or external downloads.

```bash
python examples/simulate_expression.py
# Faster demonstration:
python examples/simulate_expression.py --permutations 99
```

## Run tests

```bash
python -m pip install '.[test]'
python -m pytest -q
```
