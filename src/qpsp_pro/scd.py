"""Signed Coherence Decomposition (SCD) — §4 of the qPSP-Pro design."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# §4.1  Canonical N/M/U decomposition
# ---------------------------------------------------------------------------

def decompose_shift(N, M, tol=1e-12):
    scalar_input = np.ndim(N) == 0 and np.ndim(M) == 0

    N = np.asarray(N, dtype=float)
    M = np.asarray(M, dtype=float)

    abs_N_raw = np.abs(N)

    if np.any(abs_N_raw > M + tol):
        raise RuntimeError(
            "Numerical inconsistency: |N| exceeds M beyond tolerance")

    abs_N = np.minimum(abs_N_raw, M)
    U = M - abs_N

    with np.errstate(invalid='ignore', divide='ignore'):
        disc = np.where(M > 0, U / M, np.nan)

    if scalar_input:
        return float(U), float(disc)

    return U, disc


# ---------------------------------------------------------------------------
# §4.3  Group-level SCD
# ---------------------------------------------------------------------------

SCD_COLUMNS = [
    'pathway', 'n_pathway_genes', 'n_mapped', 'n_tested',
    'coverage_detected', 'coverage_scd',
    'net_shift', 'abs_shift', 'discordance', 'discordance_fraction',
]

MEMBER_EFFECT_COLUMNS = [
    'pathway', 'protein', 'topology_weight', 'normalized_topology_weight',
    'mean_q_test', 'mean_q_ref', 'delta_q', 'weighted_net_contribution',
    'weighted_abs_contribution', 'direction',
]


def compute_group_coherence(q_matrix, groups, pathways, topo_weights,
                            pathway_masks, contrast, dead_zone=0.05,
                            include_sign_metrics=False):
    group_a, group_b = contrast
    cols_a = [s for s in q_matrix.columns if groups.get(s) == group_a]
    cols_b = [s for s in q_matrix.columns if groups.get(s) == group_b]

    results = []
    for P_name in pathways:
        if P_name not in topo_weights or P_name not in pathway_masks:
            continue
        genes_deduped = pathways[P_name]
        mapped = list(topo_weights[P_name].keys())
        present = pathway_masks[P_name]

        if len(present) == 0:
            continue

        if not set(present).issubset(mapped):
            raise RuntimeError(
                f"SCD mask contains genes outside topology universe "
                f"for {P_name!r}")

        t = np.array([topo_weights[P_name][g] for g in present])

        mean_a = q_matrix.loc[present, cols_a].mean(axis=1).to_numpy()
        mean_b = q_matrix.loc[present, cols_b].mean(axis=1).to_numpy()
        deltas = mean_a - mean_b

        if not np.all(np.isfinite(deltas)):
            raise RuntimeError(
                f"Internal inconsistency: non-finite delta under fixed "
                f"SCD mask for pathway {P_name!r}")

        n_mapped = len(mapped)
        n_tested = len(present)
        t_sum = t.sum()

        N_val = (t * deltas).sum() / t_sum
        M_val = (t * np.abs(deltas)).sum() / t_sum
        U_val, disc_frac = decompose_shift(N_val, M_val)

        row = {
            'pathway': P_name,
            'n_pathway_genes': len(genes_deduped),
            'n_mapped': n_mapped,
            'n_tested': n_tested,
            'coverage_detected': n_mapped / len(genes_deduped) if len(genes_deduped) > 0 else 0.0,
            'coverage_scd': n_tested / len(genes_deduped) if len(genes_deduped) > 0 else 0.0,
            'net_shift': N_val,
            'abs_shift': M_val,
            'discordance': U_val,
            'discordance_fraction': disc_frac,
        }

        if include_sign_metrics:
            signs = np.zeros_like(deltas)
            signs[deltas > dead_zone] = 1
            signs[deltas < -dead_zone] = -1
            row['sign_coherence'] = abs((t * signs).sum()) / t_sum
            row['response_coverage'] = t[signs != 0].sum() / t_sum

        results.append(row)

    columns = SCD_COLUMNS[:]
    if include_sign_metrics:
        columns = columns + ['sign_coherence', 'response_coverage']
    return pd.DataFrame(results, columns=columns)


def compute_member_effects(q_matrix, groups, topo_weights, pathway_masks,
                           contrast, dead_zone=0.05):
    """Return the protein-level contributions underlying group-level SCD.

    Rows use the same fixed pathway masks and topology weights as SCD.  The
    normalized signed contributions therefore sum to ``net_shift`` and the
    normalized absolute contributions sum to ``abs_shift`` for each pathway.
    This table makes an opposing-change call directly auditable instead of
    exposing only pathway-level summary statistics.
    """
    group_a, group_b = contrast
    cols_a = [s for s in q_matrix.columns if groups.get(s) == group_a]
    cols_b = [s for s in q_matrix.columns if groups.get(s) == group_b]
    rows = []
    for pathway, present in pathway_masks.items():
        if pathway not in topo_weights or not present:
            continue
        weights = np.array(
            [topo_weights[pathway][protein] for protein in present],
            dtype=float,
        )
        weight_sum = weights.sum()
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue
        mean_a = q_matrix.loc[present, cols_a].mean(axis=1).to_numpy()
        mean_b = q_matrix.loc[present, cols_b].mean(axis=1).to_numpy()
        deltas = mean_a - mean_b
        if not np.all(np.isfinite(deltas)):
            raise RuntimeError(
                f"Internal inconsistency: non-finite member delta under "
                f"fixed SCD mask for pathway {pathway!r}")
        normalized = weights / weight_sum
        for protein, weight, norm_weight, a_value, b_value, delta in zip(
                present, weights, normalized, mean_a, mean_b, deltas):
            if delta > dead_zone:
                direction = 'higher_rank_in_test'
            elif delta < -dead_zone:
                direction = 'lower_rank_in_test'
            else:
                direction = 'within_dead_zone'
            rows.append({
                'pathway': pathway,
                'protein': protein,
                'topology_weight': weight,
                'normalized_topology_weight': norm_weight,
                'mean_q_test': a_value,
                'mean_q_ref': b_value,
                'delta_q': delta,
                'weighted_net_contribution': norm_weight * delta,
                'weighted_abs_contribution': norm_weight * abs(delta),
                'direction': direction,
            })
    return pd.DataFrame(rows, columns=MEMBER_EFFECT_COLUMNS)


# ---------------------------------------------------------------------------
# §5.3  SCD eligibility masks
# ---------------------------------------------------------------------------

def build_scd_eligible_mask(Q, n_a, n_b, min_protein_obs):
    if min_protein_obs > min(n_a, n_b):
        return None
    if isinstance(Q, pd.DataFrame):
        total_valid = Q.notna().sum(axis=1).to_numpy()
    else:
        total_valid = np.sum(np.isfinite(Q), axis=1)
    minimum_total_valid = max(n_a, n_b) + min_protein_obs
    return total_valid >= minimum_total_valid


def build_descriptive_scd_mask(Q, labels, group_a, group_b,
                                min_protein_obs):
    if isinstance(Q, pd.DataFrame):
        Q_arr = Q.to_numpy()
    else:
        Q_arr = Q
    labels = np.asarray(labels)
    idx_a = np.where(labels == group_a)[0]
    idx_b = np.where(labels == group_b)[0]
    count_a = np.sum(np.isfinite(Q_arr[:, idx_a]), axis=1)
    count_b = np.sum(np.isfinite(Q_arr[:, idx_b]), axis=1)
    return (count_a >= min_protein_obs) & (count_b >= min_protein_obs)


# ---------------------------------------------------------------------------
# §4.4  Sample-level SCD (standalone descriptive helper)
# ---------------------------------------------------------------------------

def compute_sample_scd(q_matrix, pathways, topo_weights,
                       reference_samples, min_detected, min_coverage,
                       min_protein_obs_per_group):
    reference_samples = list(reference_samples)
    if len(set(reference_samples)) != len(reference_samples):
        raise ValueError("reference_samples must not contain duplicates")
    missing_references = [s for s in reference_samples
                          if s not in q_matrix.columns]
    if missing_references:
        raise ValueError(
            f"reference_samples not found in q_matrix: "
            f"{missing_references[:5]}")
    if len(reference_samples) < min_protein_obs_per_group + 1:
        raise ValueError(
            f"Sample-level SCD requires at least "
            f"min_protein_obs_per_group + 1 = {min_protein_obs_per_group + 1} "
            f"reference samples, got {len(reference_samples)}.")

    ref_full = q_matrix[reference_samples].mean(axis=1)

    results = {k: {} for k in
               ['net_shift', 'abs_shift', 'discordance', 'discordance_fraction']}
    for P_name in pathways:
        if P_name not in topo_weights:
            continue
        genes_deduped = tuple(dict.fromkeys(
            str(g).strip().upper() for g in pathways[P_name]
            if g is not None and str(g).strip()))
        present = [g for g in genes_deduped if g in topo_weights[P_name]
                   and g in q_matrix.index]
        if not present:
            continue
        t = np.array([topo_weights[P_name][g] for g in present])

        row = {k: {} for k in results}
        for s in q_matrix.columns:
            if s in reference_samples:
                others = [r for r in reference_samples if r != s]
                ref = q_matrix.loc[present, others].mean(axis=1).values
            else:
                ref = ref_full[present].values

            q_s = q_matrix.loc[present, s].values

            if s in reference_samples:
                ref_count = q_matrix.loc[present, others].notna().sum(axis=1).values
            else:
                ref_count = q_matrix.loc[present, reference_samples].notna().sum(axis=1).values
            deltas = q_s - ref

            valid = (~np.isnan(deltas)) & (ref_count >= min_protein_obs_per_group)
            t_v = t[valid]
            d_v = deltas[valid]
            t_v_sum = t_v.sum()

            required = max(min_detected,
                           math.ceil(min_coverage * len(genes_deduped)))
            if t_v_sum == 0 or valid.sum() < required:
                for k in row:
                    row[k][s] = np.nan
                continue

            N_val = (t_v * d_v).sum() / t_v_sum
            M_val = (t_v * np.abs(d_v)).sum() / t_v_sum
            U_val, disc = decompose_shift(N_val, M_val)

            row['net_shift'][s] = N_val
            row['abs_shift'][s] = M_val
            row['discordance'][s] = float(U_val)
            row['discordance_fraction'][s] = float(disc)

        for k in results:
            results[k][P_name] = row[k]

    pathway_index = list(pathways.keys())
    return {k: pd.DataFrame(v).T.reindex(
                index=pathway_index, columns=q_matrix.columns)
            for k, v in results.items()}
