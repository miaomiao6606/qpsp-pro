# Interaction input

`reactome_fi_score075.tsv.gz` is a public Reactome Functional Interaction
network with `gene1`, `gene2` and `score` columns. Pass its path using
`ppi_path="data/reactome_fi_score075.tsv.gz"` and
`topology="weighted_degree"`.

Provide pathway memberships separately as a Python dictionary or JSON file.
The synthetic example generates its own artificial pathways and interactions.

Reactome distributes data and data-derived files under CC0. See the
[download page](https://reactome.org/download-data) and
[license](https://reactome.org/license).
