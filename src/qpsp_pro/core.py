from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, comb

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import expit


@dataclass
class QpspProResult:
    pathway_scores: pd.DataFrame
    significance: pd.DataFrame
    member_effects: pd.DataFrame
    rank_quantiles: pd.DataFrame
    topology_weights: dict[str, dict[str, float]]
    params: dict


def _welch(data: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
    a = data[:, group_a]
    b = data[:, group_b]
    n_a = a.shape[1]
    n_b = b.shape[1]
    mean_a = a.mean(axis=1)
    mean_b = b.mean(axis=1)
    var_a = a.var(axis=1, ddof=1)
    var_b = b.var(axis=1, ddof=1)
    se2 = var_a / n_a + var_b / n_b
    diff = mean_a - mean_b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            se2 > 0,
            diff / np.sqrt(se2),
            np.where(diff == 0, 0.0, np.sign(diff) * np.inf),
        )


def _bh(p_values: np.ndarray):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite = np.isfinite(p_values)
    p = p_values[finite]
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    q = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0.0, 1.0)
    adjusted[finite] = restored
    return adjusted


def qpsp_pro(
    expression: pd.DataFrame,
    pathways: dict[str, list[str]],
    groups: dict[str, str],
    contrast: tuple[str, str],
    *,
    blocks: dict[str, str] | None = None,
    topology: str = "uniform",
    ppi: pd.DataFrame | None = None,
    topology_strength: float = 0.5,
    ppi_threshold: float = 0.75,
    tau: float = 0.85,
    kappa: float = 30.0,
    min_detected: int = 3,
    min_coverage: float = 0.1,
    min_group_units: int = 3,
    n_permutations: int = 9999,
    fdr_threshold: float = 0.05,
    seed: int = 42,
) -> QpspProResult:
    """Run qPSP-Pro on a protein-by-sample abundance matrix."""
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a pandas DataFrame")
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("expression row and column labels must be unique")
    if topology not in {"uniform", "weighted_degree"}:
        raise ValueError("topology must be 'uniform' or 'weighted_degree'")
    if topology == "weighted_degree" and ppi is None:
        raise ValueError("ppi is required for weighted_degree topology")
    if not 0 <= topology_strength <= 1:
        raise ValueError("topology_strength must be in [0, 1]")
    if min_detected < 2 or n_permutations < 1:
        raise ValueError("min_detected must be at least 2 and n_permutations positive")

    missing_groups = set(expression.columns) - set(groups)
    if missing_groups:
        raise ValueError(f"samples missing from groups: {sorted(missing_groups)}")

    test_group, reference_group = contrast
    samples = [
        sample
        for sample in expression.columns
        if groups[sample] in {test_group, reference_group}
    ]
    labels = pd.Series({sample: groups[sample] for sample in samples})
    if set(labels.index) != set(samples) or set(labels.values) != {
        test_group,
        reference_group,
    }:
        raise ValueError("both contrast groups must be present")

    values = expression.loc[:, samples].copy()
    values.index = pd.Index([str(gene).strip().upper() for gene in values.index])
    if not values.index.is_unique:
        raise ValueError("protein names must be unique after uppercasing")
    values = values.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("expression contains infinite values")
    values = values.dropna(axis=0, how="any")
    if len(values) < min_detected:
        raise ValueError("too few complete proteins for ranking")

    rank_quantiles = (values.rank(axis=0, method="average") - 1) / (len(values) - 1)
    kernel_weights = pd.DataFrame(
        expit(kappa * (rank_quantiles.to_numpy() - tau)),
        index=rank_quantiles.index,
        columns=rank_quantiles.columns,
    )

    normalized_pathways = {
        str(name): tuple(
            dict.fromkeys(
                str(gene).strip().upper()
                for gene in genes
                if str(gene).strip()
            )
        )
        for name, genes in pathways.items()
    }

    edge_scores: dict[tuple[str, str], float] = {}
    if topology == "weighted_degree":
        required_columns = {"gene1", "gene2", "score"}
        if not required_columns.issubset(ppi.columns):
            raise ValueError("ppi must contain gene1, gene2 and score columns")
        for gene1, gene2, score in ppi[["gene1", "gene2", "score"]].itertuples(
            index=False, name=None
        ):
            gene1 = str(gene1).strip().upper()
            gene2 = str(gene2).strip().upper()
            score = float(score)
            if gene1 != gene2 and score >= ppi_threshold:
                edge = tuple(sorted((gene1, gene2)))
                edge_scores[edge] = max(edge_scores.get(edge, 0.0), score)

    topology_weights: dict[str, dict[str, float]] = {}
    for name, genes in normalized_pathways.items():
        present = [gene for gene in genes if gene in rank_quantiles.index]
        required = max(min_detected, ceil(min_coverage * len(genes)))
        if len(present) < required:
            continue
        if topology == "uniform":
            topology_weights[name] = {gene: 1.0 for gene in present}
            continue
        centrality = {gene: 1.0 for gene in present}
        for i, gene1 in enumerate(present):
            for gene2 in present[i + 1 :]:
                score = edge_scores.get(tuple(sorted((gene1, gene2))), 0.0)
                centrality[gene1] += score
                centrality[gene2] += score
        scale = len(present) / sum(centrality.values())
        topology_weights[name] = {
            gene: 1 - topology_strength + topology_strength * centrality[gene] * scale
            for gene in present
        }

    if not topology_weights:
        raise ValueError("no pathway met the mapping threshold")

    pathway_scores = {}
    for name, weights in topology_weights.items():
        genes = list(weights)
        topology_vector = pd.Series(weights)
        weighted_kernel = kernel_weights.loc[genes].mul(topology_vector, axis=0)
        pathway_scores[name] = (
            weighted_kernel.mul(rank_quantiles.loc[genes]).sum(axis=0)
            / weighted_kernel.sum(axis=0)
        )
    pathway_scores = pd.DataFrame(pathway_scores).T
    pathway_scores.index.name = "pathway"

    if blocks is None:
        unit_quantiles = rank_quantiles
        unit_scores = pathway_scores
        unit_labels = labels.to_dict()
    else:
        missing_blocks = set(samples) - set(blocks)
        if missing_blocks:
            raise ValueError(f"samples missing from blocks: {sorted(missing_blocks)}")
        unit_labels = {}
        for sample in samples:
            block = blocks[sample]
            label = groups[sample]
            if block in unit_labels and unit_labels[block] != label:
                raise ValueError("each block must belong to one contrast group")
            unit_labels[block] = label
        block_index = pd.Series({sample: blocks[sample] for sample in samples})
        unit_quantiles = rank_quantiles.T.groupby(block_index).mean().T
        unit_scores = pathway_scores.T.groupby(block_index).mean().T

    units = list(unit_scores.columns)
    unit_group_values = np.array([unit_labels[unit] for unit in units])
    observed_a = unit_group_values == test_group
    observed_b = unit_group_values == reference_group
    n_a = int(observed_a.sum())
    n_b = int(observed_b.sum())
    if n_a < min_group_units or n_b < min_group_units:
        raise ValueError("too few analysis units in a contrast group")

    score_array = unit_scores.to_numpy(dtype=float)
    observed_t = _welch(score_array, observed_a, observed_b)
    score_diff = score_array[:, observed_a].mean(axis=1) - score_array[:, observed_b].mean(axis=1)

    genes = list(unit_quantiles.index)
    gene_positions = {gene: i for i, gene in enumerate(genes)}
    rows = []
    columns = []
    weights = []
    for row, name in enumerate(unit_scores.index):
        for gene, weight in topology_weights[name].items():
            rows.append(row)
            columns.append(gene_positions[gene])
            weights.append(weight)
    weight_matrix = csr_matrix(
        (weights, (rows, columns)),
        shape=(len(unit_scores), len(unit_quantiles)),
    )
    weight_sums = np.asarray(weight_matrix.sum(axis=1)).ravel()
    unit_quantile_array = unit_quantiles.loc[:, units].to_numpy(dtype=float)
    observed_delta = (
        unit_quantile_array[:, observed_a].mean(axis=1)
        - unit_quantile_array[:, observed_b].mean(axis=1)
    )
    net_shift = np.asarray(weight_matrix @ observed_delta).ravel() / weight_sums
    absolute_shift = np.asarray(weight_matrix @ np.abs(observed_delta)).ravel() / weight_sums
    discordance = np.maximum(absolute_shift - np.abs(net_shift), 0.0)
    discordance_fraction = np.divide(
        discordance,
        absolute_shift,
        out=np.full_like(discordance, np.nan),
        where=absolute_shift > 0,
    )

    exceed_score = np.zeros(len(unit_scores), dtype=int)
    exceed_net = np.zeros(len(unit_scores), dtype=int)
    exceed_discordance = np.zeros(len(unit_scores), dtype=int)
    possible_assignments = comb(len(units), n_a)
    exact = possible_assignments <= n_permutations
    rng = np.random.default_rng(seed)
    if exact:
        assignments = combinations(range(len(units)), n_a)
        draws = possible_assignments
    else:
        assignments = (
            rng.choice(len(units), n_a, replace=False)
            for _ in range(n_permutations)
        )
        draws = n_permutations

    for assignment in assignments:
        permuted_a = np.zeros(len(units), dtype=bool)
        permuted_a[list(assignment)] = True
        permuted_b = ~permuted_a
        permuted_t = _welch(score_array, permuted_a, permuted_b)
        exceed_score += np.abs(permuted_t) >= np.abs(observed_t)
        delta = (
            unit_quantile_array[:, permuted_a].mean(axis=1)
            - unit_quantile_array[:, permuted_b].mean(axis=1)
        )
        permuted_net = np.asarray(weight_matrix @ delta).ravel() / weight_sums
        permuted_absolute = np.asarray(weight_matrix @ np.abs(delta)).ravel() / weight_sums
        permuted_discordance = np.maximum(
            permuted_absolute - np.abs(permuted_net), 0.0
        )
        exceed_net += np.abs(permuted_net) >= np.abs(net_shift)
        exceed_discordance += permuted_discordance >= discordance

    correction = 0 if exact else 1
    denominator = draws if exact else draws + 1
    p_pathway = (exceed_score + correction) / denominator
    p_net = (exceed_net + correction) / denominator
    p_discordance = (exceed_discordance + correction) / denominator
    q_pathway = _bh(p_pathway)
    q_scd = _bh(np.concatenate([p_net, p_discordance]))
    q_net = q_scd[: len(p_net)]
    q_discordance = q_scd[len(p_net) :]

    hedges_g = []
    for row in score_array:
        a = row[observed_a]
        b = row[observed_b]
        pooled = np.sqrt(
            ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1))
            / (n_a + n_b - 2)
        )
        if pooled == 0:
            hedges_g.append(0.0 if a.mean() == b.mean() else np.nan)
        else:
            correction_g = 1 - 3 / (4 * (n_a + n_b - 2) - 1)
            hedges_g.append((a.mean() - b.mean()) / pooled * correction_g)

    architectures = []
    for net, qn, qd in zip(net_shift, q_net, q_discordance):
        if qd < fdr_threshold and qn >= fdr_threshold:
            architectures.append("cancellation")
        elif qd < fdr_threshold and qn < fdr_threshold:
            architectures.append("mixed")
        elif qn < fdr_threshold:
            architectures.append("coherent_higher" if net > 0 else "coherent_lower")
        else:
            architectures.append("not_significant")

    significance = pd.DataFrame(
        {
            "pathway": unit_scores.index,
            "n_pathway_genes": [len(normalized_pathways[name]) for name in unit_scores.index],
            "n_mapped": [len(topology_weights[name]) for name in unit_scores.index],
            "score_difference": score_diff,
            "hedges_g": hedges_g,
            "p_pathway": p_pathway,
            "q_pathway": q_pathway,
            "net_shift": net_shift,
            "absolute_shift": absolute_shift,
            "discordance": discordance,
            "discordance_fraction": discordance_fraction,
            "p_net_shift": p_net,
            "q_net_shift": q_net,
            "p_discordance": p_discordance,
            "q_discordance": q_discordance,
            "architecture": architectures,
        }
    )

    member_rows = []
    for name, pathway_weight_map in topology_weights.items():
        weight_sum = sum(pathway_weight_map.values())
        for gene, weight in pathway_weight_map.items():
            delta = observed_delta[gene_positions[gene]]
            normalized_weight = weight / weight_sum
            member_rows.append(
                {
                    "pathway": name,
                    "protein": gene,
                    "topology_weight": weight,
                    "delta_rank": delta,
                    "net_contribution": normalized_weight * delta,
                    "absolute_contribution": normalized_weight * abs(delta),
                }
            )

    return QpspProResult(
        pathway_scores=pathway_scores,
        significance=significance,
        member_effects=pd.DataFrame(member_rows),
        rank_quantiles=rank_quantiles,
        topology_weights=topology_weights,
        params={
            "contrast": contrast,
            "topology": topology,
            "topology_strength": topology_strength,
            "ppi_threshold": ppi_threshold,
            "tau": tau,
            "kappa": kappa,
            "n_permutations": n_permutations,
            "permutation_mode": "exact" if exact else "monte_carlo",
            "analysis_units": len(units),
            "seed": seed,
        },
    )
