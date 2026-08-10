import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpsp_pro import qpsp_pro


root = Path(__file__).resolve().parents[1]
with gzip.open(root / "data/pathways.json.gz", "rt", encoding="utf-8") as handle:
    pathway_catalogue = json.load(handle)
ppi = pd.read_csv(root / "data/reactome_fi_score075.tsv.gz", sep="\t")
ppi = ppi.loc[ppi["score"] >= 0.75, ["gene1", "gene2", "score"]].copy()
ppi["gene1"] = ppi["gene1"].astype(str).str.upper()
ppi["gene2"] = ppi["gene2"].astype(str).str.upper()
edge_scores = {
    tuple(sorted((gene1, gene2))): float(score)
    for gene1, gene2, score in ppi.itertuples(index=False, name=None)
    if gene1 != gene2
}
ppi_genes = {gene for edge in edge_scores for gene in edge}

pathways = {}
core_members = {}
pathway_genes = []
used_genes = set()
for pathway, catalogue_members in pathway_catalogue.items():
    members = list(
        dict.fromkeys(
            str(gene).strip().upper()
            for gene in catalogue_members
            if str(gene).strip().upper() in ppi_genes
        )
    )
    if not 10 <= len(members) <= 20 or set(members) & used_genes:
        continue
    neighbors = {gene: [] for gene in members}
    weighted_degree = {gene: 0.0 for gene in members}
    for i, gene1 in enumerate(members):
        for gene2 in members[i + 1 :]:
            score = edge_scores.get(tuple(sorted((gene1, gene2))))
            if score is None:
                continue
            neighbors[gene1].append(gene2)
            neighbors[gene2].append(gene1)
            weighted_degree[gene1] += score
            weighted_degree[gene2] += score
    hub = max(members, key=lambda gene: (len(neighbors[gene]), weighted_degree[gene], gene))
    if len(neighbors[hub]) < 2:
        continue
    connected = sorted(
        neighbors[hub],
        key=lambda gene: (edge_scores[tuple(sorted((hub, gene)))], gene),
        reverse=True,
    )
    pathways[pathway] = members
    core_members[pathway] = [hub, connected[0], connected[1]]
    pathway_genes.extend(members)
    used_genes.update(members)
    if len(pathways) == 40:
        break

seed = 20260720
scenarios = ("coordinated_up", "coordinated_down", "cancellation", "core_shift")
n_background = 120
n_true = 5
n_participants_per_group = 16
n_visits = 2
background_genes = sorted(ppi_genes - used_genes)[:n_background]
genes = pathway_genes + background_genes
true_pathways = set(list(pathways)[:n_true])

summaries = []
for scenario_index, scenario in enumerate(scenarios):
    rng = np.random.default_rng(seed + scenario_index * 100_000)
    participants = [f"A{i:02d}" for i in range(n_participants_per_group)] + [
        f"B{i:02d}" for i in range(n_participants_per_group)
    ]
    samples = [
        f"{participant}_V{visit + 1}"
        for participant in participants
        for visit in range(n_visits)
    ]
    groups = {
        sample: ("A" if sample.startswith("A") else "B") for sample in samples
    }
    blocks = {sample: sample.split("_")[0] for sample in samples}

    values = rng.normal(0, 0.72, size=(len(genes), len(samples)))
    values += rng.normal(0, 0.20, size=(1, len(samples)))
    gene_position = {gene: i for i, gene in enumerate(genes)}

    for members in pathways.values():
        latent = rng.normal(0, 0.40, size=len(samples))
        values[[gene_position[gene] for gene in members]] += latent
    for participant in participants:
        columns = [i for i, sample in enumerate(samples) if blocks[sample] == participant]
        values[:, columns] += rng.normal(0, 0.28, size=(len(genes), 1))

    group_a_columns = [i for i, sample in enumerate(samples) if groups[sample] == "A"]
    for pathway in sorted(true_pathways):
        members = pathways[pathway]
        if scenario == "coordinated_up":
            shifts = {gene: 0.62 for gene in members}
        elif scenario == "coordinated_down":
            shifts = {gene: -0.62 for gene in members}
        elif scenario == "cancellation":
            shifts = {
                gene: (0.76 if i < len(members) // 2 else -0.76)
                for i, gene in enumerate(members)
            }
        else:
            shifts = {gene: 1.05 for gene in core_members[pathway]}
        for gene, shift in shifts.items():
            values[gene_position[gene], group_a_columns] += shift

    background_start = len(pathway_genes)
    detection_probability = 1 / (
        1 + np.exp(-(values[background_start:] + 1.5) * 2.0)
    )
    missing = rng.random(detection_probability.shape) > detection_probability
    values[background_start:][missing] = np.nan

    expression = pd.DataFrame(values, index=genes, columns=samples)
    result = qpsp_pro(
        expression,
        pathways,
        groups,
        ("A", "B"),
        blocks=blocks,
        topology="weighted_degree",
        ppi=ppi,
        topology_strength=0.5,
        ppi_threshold=0.75,
        tau=0.85,
        kappa=30,
        min_detected=3,
        min_coverage=0.1,
        n_permutations=499,
        seed=seed + scenario_index * 100_000,
    )

    statistics = result.significance.set_index("pathway")
    ranked = statistics["hedges_g"].abs().sort_values(ascending=False)
    true_statistics = statistics.loc[sorted(true_pathways)]
    summaries.append(
        {
            "scenario": scenario,
            "true_in_top5": int(
                sum(pathway in true_pathways for pathway in ranked.head(5).index)
            ),
            "median_true_abs_g": true_statistics["hedges_g"].abs().median(),
            "median_true_net_shift": true_statistics["net_shift"].median(),
            "median_true_discordance_fraction": true_statistics[
                "discordance_fraction"
            ].median(),
        }
    )

print(pd.DataFrame(summaries).to_string(index=False))
