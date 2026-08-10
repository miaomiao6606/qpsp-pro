# Data

## Pathway catalogue

`pathways.json.gz` contains the 1,323 pathways used in the paper analysis as `{pathway: [protein, ...]}`.

`pathways_metadata.tsv.gz` contains:

- `pathway`: analysis pathway identifier
- `entry_type`: `parent` or `observed_subpathway`
- `source_parent`: source pathway identifier
- `collection`: Reactome, KEGG Medicus, KEGG Legacy or Hallmark
- `n_genes`: pathway gene count

The catalogue comprises 761 parent pathways and 562 observed-protein subpathways.

## Reactome FI network

`reactome_fi_score075.tsv.gz` contains the paper analysis network with columns `gene1`, `gene2` and `score`.

- Confidence threshold: `score >= 0.75`
- Interactions: 278,974
- Genes: 14,034
- Score range: 0.89–1.00

qPSP-Pro extracts the internal subnetwork for each pathway, calculates confidence-weighted degree, normalizes the weights to mean one and applies `lambda = 0.5` shrinkage toward uniform weight.

Reactome distributes data and data-derived files under CC0. See the [Reactome download page](https://reactome.org/download-data) and [license](https://reactome.org/license).
