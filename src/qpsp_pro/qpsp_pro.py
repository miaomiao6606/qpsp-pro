"""qPSP-Pro: Quantitative Proteomic Set Profiling with continuous kernel
scoring, topology-weighted aggregation, signed coherence decomposition,
and permutation-based inference."""

from __future__ import annotations

import gzip
import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse
from scipy.interpolate import PchipInterpolator
from scipy.special import expit

from .scd import (
    decompose_shift,
    compute_group_coherence,
    compute_member_effects,
    build_scd_eligible_mask,
    build_descriptive_scd_mask,
)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class QpspProResult:
    pathway_scores: pd.DataFrame
    significance: pd.DataFrame
    topology_weights: dict[str, dict[str, float]]
    kernel_weights: pd.DataFrame
    params: dict
    diagnostics: dict
    member_effects: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# §9.2  Validation helpers
# ---------------------------------------------------------------------------

_VALID = {
    'kernel': {'measurement_calibrated', 'logistic', 'identity'},
    'score_denominator': {'valid_topology', 'kernel_weighted',
                          'fixed_mapped', 'complete_case'},
    'topology': {'weighted_degree', 'uniform'},
    'ranking_universe': {'per_sample', 'shared',
                         'contrast_complete_case', 'analysis_complete_case'},
    'design': {'independent', 'clustered'},
    'pathway_test': {'permutation', 'welch'},
    'scd': {'permutation', 'descriptive', 'none'},
    'fdr_method': {'BH', 'BY'},
    'hypothesis_families': {'separate', 'pooled'},
}


# Dataset-agnostic reference settings for studies without technical repeats.
# Measurement-calibrated kernels require user-supplied anchors.
DEFAULT_KERNEL = 'logistic'
DEFAULT_TAU = 0.85
DEFAULT_KAPPA = 30.0


def _validate_params(*, tau, kappa, topology_strength, ppi_threshold,
                     ranking_min_coverage, min_coverage, fdr_threshold,
                     n_permutations, min_detected, min_group_units,
                     min_protein_obs_per_group, scd_min_effect, contrast,
                     kernel_quantiles, kernel_weights,
                     **enum_params):
    if not (0 <= tau <= 1):
        raise ValueError("tau must be in [0, 1]")
    if not (kappa > 0):
        raise ValueError("kappa must be > 0")
    if not (0 <= topology_strength <= 1):
        raise ValueError("topology_strength must be in [0, 1]")
    if not (0 <= ppi_threshold <= 1):
        raise ValueError("ppi_threshold must be in [0, 1]")
    if not (0 <= ranking_min_coverage <= 1):
        raise ValueError("ranking_min_coverage must be in [0, 1]")
    if not (0 <= min_coverage <= 1):
        raise ValueError("min_coverage must be in [0, 1]")
    if not (0 < fdr_threshold < 1):
        raise ValueError("fdr_threshold must be in (0, 1)")
    if not (n_permutations >= 1):
        raise ValueError("n_permutations must be >= 1")
    if not (min_detected >= 1):
        raise ValueError("min_detected must be >= 1")
    if not (min_group_units >= 2):
        raise ValueError("min_group_units must be >= 2")
    if not (min_protein_obs_per_group >= 1):
        raise ValueError("min_protein_obs_per_group must be >= 1")
    if not (scd_min_effect >= 0):
        raise ValueError("scd_min_effect must be >= 0")
    if len(contrast) != 2:
        raise ValueError("contrast must be a 2-tuple")
    if contrast[0] == contrast[1]:
        raise ValueError("contrast groups must differ")
    for name, allowed in _VALID.items():
        val = enum_params.get(name)
        if val is not None and val not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
    kernel = enum_params.get('kernel')
    if kernel != 'measurement_calibrated':
        return
    if kernel_quantiles is None or kernel_weights is None:
        raise ValueError(
            "kernel_quantiles and kernel_weights are required when "
            "kernel='measurement_calibrated'; estimate them from technical "
            "repeats or use the default logistic kernel")
    if len(kernel_quantiles) != len(kernel_weights):
        raise ValueError(
            "kernel_quantiles and kernel_weights must have equal length")
    if len(kernel_quantiles) < 2:
        raise ValueError("measurement-calibrated kernel needs >=2 anchors")
    q_anchor = np.asarray(kernel_quantiles, dtype=float)
    w_anchor = np.asarray(kernel_weights, dtype=float)
    if not np.isfinite(q_anchor).all() or not np.isfinite(w_anchor).all():
        raise ValueError("kernel anchors must be finite")
    if not ((q_anchor >= 0).all() and (q_anchor <= 1).all()):
        raise ValueError("kernel_quantiles must lie in [0, 1]")
    if not np.all(np.diff(q_anchor) > 0):
        raise ValueError("kernel_quantiles must be strictly increasing")
    if not (w_anchor > 0).all():
        raise ValueError("kernel_weights must be strictly positive")


# ---------------------------------------------------------------------------
# §2.3  Expression score q  (fit / transform)
# ---------------------------------------------------------------------------

def fit_ranking_model(expression_fit, ranking_universe='per_sample',
                      ranking_min_coverage=0.5):
    if ranking_universe == 'shared':
        detection_rate = expression_fit.notna().mean(axis=1)
        universe = detection_rate[detection_rate >= ranking_min_coverage].index
    elif ranking_universe in {'contrast_complete_case',
                              'analysis_complete_case'}:
        universe = expression_fit.index[expression_fit.notna().all(axis=1)]
    else:
        universe = None
    return {
        'universe': universe,
        'ranking_universe': ranking_universe,
        'fit_samples': tuple(expression_fit.columns),
    }


def compute_expression_scores(expression, ranking_model):
    universe = ranking_model['universe']
    q = pd.DataFrame(np.nan, index=expression.index,
                     columns=expression.columns, dtype=float)

    if universe is not None:
        expr_subset = expression.reindex(index=universe)
    else:
        expr_subset = expression

    n_valid_per_sample = {}
    for sample in expr_subset.columns:
        col = expr_subset[sample]
        valid = col.dropna()
        n_valid = len(valid)
        n_valid_per_sample[sample] = n_valid
        if n_valid < len(expr_subset) / 2:
            warnings.warn(
                f"sample {sample!r} has more than 50% missing values in "
                "the ranking universe", RuntimeWarning)
        if n_valid == 0:
            continue
        if n_valid == 1:
            q.loc[valid.index, sample] = 0.5
            continue
        if valid.nunique() == 1:
            warnings.warn(
                f"sample {sample!r} has identical expression values for "
                "all ranked proteins; assigning q=0.5", RuntimeWarning)
            q.loc[valid.index, sample] = 0.5
            continue
        ranks = valid.rank(method='average', ascending=True)
        q.loc[valid.index, sample] = (ranks - 1) / (n_valid - 1)

    excluded_per_sample = {}
    if universe is not None:
        universe_set = set(universe)
        for sample in expression.columns:
            detected = set(expression[sample].dropna().index)
            excluded_per_sample[sample] = len(detected - universe_set)

    n_detected_raw_per_sample = {
        s: int(expression[s].notna().sum())
        for s in expression.columns
    }

    fixed_complete_case = ranking_model['ranking_universe'] in {
        'contrast_complete_case', 'analysis_complete_case'}
    if fixed_complete_case:
        validation_samples = [
            s for s in ranking_model.get('fit_samples', ()) if s in q.columns]
        validation_counts = {
            n_valid_per_sample[s] for s in validation_samples}
        if len(validation_counts) > 1:
            raise RuntimeError(
                "complete-case ranking failed to produce a fixed ranking "
                "denominator")

    fixed_for_all_scored = (
        fixed_complete_case
        and len(set(n_valid_per_sample.values())) <= 1)
    if (ranking_model['ranking_universe'] == 'contrast_complete_case'
            and not fixed_for_all_scored):
        warnings.warn(
            "contrast_complete_case fixes the denominator only for contrast "
            "samples; one or more out-of-contrast samples have missing values "
            "in that universe. Use ranking_universe='analysis_complete_case' "
            "for analyses that compare every scored group.", RuntimeWarning)

    return q, {
        'universe_size': len(universe) if universe is not None else None,
        'n_detected_raw_per_sample': n_detected_raw_per_sample,
        'n_ranked_per_sample': n_valid_per_sample,
        'n_excluded_by_universe': excluded_per_sample if excluded_per_sample else None,
        'ranking_denominator_fixed_for_contrast': fixed_complete_case,
        'ranking_denominator_fixed_for_all_scored_samples': fixed_for_all_scored,
    }


# ---------------------------------------------------------------------------
# §2.4  Detection-rate diagnostics
# ---------------------------------------------------------------------------

def check_detection_balance(counts_per_sample, groups, contrast,
                            blocks=None):
    group_a, group_b = contrast

    if blocks is not None:
        from collections import defaultdict
        unit_vals = defaultdict(list)
        for s, n in counts_per_sample.items():
            if s in blocks and groups.get(s) in {group_a, group_b}:
                unit_vals[(blocks[s], groups[s])].append(n)
        vals_a = [np.mean(v) for (pid, g), v in unit_vals.items()
                  if g == group_a]
        vals_b = [np.mean(v) for (pid, g), v in unit_vals.items()
                  if g == group_b]
    else:
        vals_a = [counts_per_sample[s] for s in groups
                  if groups[s] == group_a and s in counts_per_sample]
        vals_b = [counts_per_sample[s] for s in groups
                  if groups[s] == group_b and s in counts_per_sample]

    if len(vals_a) < 2 or len(vals_b) < 2:
        return None
    mean_a = float(np.mean(vals_a))
    mean_b = float(np.mean(vals_b))
    if np.var(vals_a, ddof=1) == 0 and np.var(vals_b, ddof=1) == 0:
        p = 1.0 if mean_a == mean_b else 0.0
    else:
        values = np.asarray([vals_a + vals_b], dtype=float)
        idx_a = np.arange(len(vals_a))
        idx_b = np.arange(len(vals_a), len(vals_a) + len(vals_b))
        t_value, df = welch_t_per_row(values, idx_a, idx_b, 2)
        p = float(welch_p_from_t(t_value, df)[0])
    return {
        'mean_detection_a': mean_a,
        'mean_detection_b': mean_b,
        'p_detection_imbalance': p,
        'warning': p < 0.01,
    }


# ---------------------------------------------------------------------------
# §2.5  Analysis-unit construction
# ---------------------------------------------------------------------------

def aggregate_to_participants(data, blocks, groups):
    unit_data = {}
    unit_groups_out = {}

    for participant in sorted(set(blocks.values())):
        samples = [s for s, p in blocks.items()
                   if p == participant and s in data.columns]
        if not samples:
            continue

        labels = {groups[s] for s in samples if s in groups}
        if len(labels) != 1:
            raise ValueError(
                f"Participant {participant} has samples in multiple groups: "
                f"{labels}. Cluster permutation requires one group per "
                f"participant. For within-participant contrasts, use "
                f"design='paired'.")

        unit_data[participant] = data[samples].mean(axis=1)
        unit_groups_out[participant] = next(iter(labels))

    unit_df = pd.DataFrame(unit_data)
    unit_df = unit_df.reindex(columns=sorted(unit_df.columns))
    return unit_df, unit_groups_out


# ---------------------------------------------------------------------------
# §3.1  Kernel
# ---------------------------------------------------------------------------

def _logistic_kernel(q, tau, kappa):
    return expit(kappa * (q - tau))


def compute_kernel_weights(
        q_matrix, kernel, tau, kappa,
        kernel_quantiles=None,
        kernel_weights=None):
    if kernel == 'identity':
        w = q_matrix.copy()
        w[q_matrix.notna()] = 1.0
        return w
    elif kernel == 'measurement_calibrated':
        q_anchor = np.asarray(kernel_quantiles, dtype=float)
        w_anchor = np.asarray(kernel_weights, dtype=float)
        interpolation = PchipInterpolator(
            q_anchor, np.log(w_anchor), extrapolate=False)
        values = q_matrix.to_numpy(dtype=float)
        clipped = np.clip(values, q_anchor[0], q_anchor[-1])
        weights = np.exp(interpolation(clipped))
        weights[~np.isfinite(values)] = np.nan
        return pd.DataFrame(
            weights, index=q_matrix.index, columns=q_matrix.columns)
    elif kernel == 'logistic':
        return q_matrix.apply(lambda col: _logistic_kernel(col, tau, kappa))
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")


# ---------------------------------------------------------------------------
# §3.2  Topology weights
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def load_ppi(path, score_threshold=0.75):
    if path is None:
        raise ValueError(
            "ppi_path is required when topology='weighted_degree'. "
            "Pass ppi_path or set topology='uniform'.")
    ppi = {}
    n_skipped = 0
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, mode="rt", encoding="utf-8") as f:
        try:
            header = next(f)
        except StopIteration:
            raise ValueError(f"PPI file is empty: {path}")
        sep = '\t' if '\t' in header else None
        cols = header.strip().split(sep)
        col_map = {c.strip(): i for i, c in enumerate(cols)}
        col_lower = {c.lower(): i for c, i in col_map.items()}
        gene1_idx = col_lower.get('gene1')
        gene2_idx = col_lower.get('gene2')
        score_idx = col_lower.get('score')
        if gene1_idx is None or gene2_idx is None or score_idx is None:
            missing = [c for c, v in [('Gene1', gene1_idx),
                       ('Gene2', gene2_idx), ('Score', score_idx)]
                       if v is None]
            raise ValueError(
                f"PPI file {path} missing required columns: {missing}. "
                f"Found: {cols}")

        for line_no, line in enumerate(f, start=2):
            parts = line.strip().split(sep)
            if len(parts) <= max(gene1_idx, gene2_idx, score_idx):
                n_skipped += 1
                continue
            gene1 = parts[gene1_idx].strip().upper()
            gene2 = parts[gene2_idx].strip().upper()
            if not gene1 or not gene2:
                n_skipped += 1
                continue
            if gene1 == gene2:
                continue
            try:
                score = float(parts[score_idx])
            except ValueError:
                raise ValueError(
                    f"Non-numeric score at {path}:{line_no}: "
                    f"{parts[score_idx]!r}")
            if not np.isfinite(score):
                raise ValueError(
                    f"Non-finite score at {path}:{line_no}: "
                    f"{parts[score_idx]!r}")
            if score < score_threshold:
                continue
            key = tuple(sorted((gene1, gene2)))
            ppi[key] = max(ppi.get(key, 0.0), score)
    if n_skipped > 0:
        warnings.warn(f"load_ppi: skipped {n_skipped} malformed rows in {path}")
    return ppi


def compute_topology_weights(pathways, detected, ppi, lam, topology,
                              min_detected, min_coverage):
    weights = {}
    weight_diagnostics = {}
    for P_name, P_genes in pathways.items():
        genes_deduped = P_genes

        if len(genes_deduped) == 0:
            weight_diagnostics[P_name] = {
                'n_pathway_genes': 0, 'mapping_status': 'empty_pathway'}
            continue

        present = [g for g in genes_deduped if g in detected]
        k = len(present)

        required_count = max(min_detected,
                             math.ceil(min_coverage * len(genes_deduped)))
        if k < required_count:
            weights[P_name] = {g: 1.0 for g in present}
            weight_diagnostics[P_name] = {
                'n_pathway_genes': len(genes_deduped),
                'n_mapped': k,
                'mapping_status': 'insufficient_mapping',
                'ppi_coverage': np.nan,
            }
            continue

        if topology == 'uniform':
            t = {g: 1.0 for g in present}
            ppi_cov = np.nan
        else:
            c = {g: 1.0 for g in present}
            for i, g in enumerate(present):
                for h in present[i + 1:]:
                    key = tuple(sorted((g, h)))
                    if key in ppi:
                        c[g] += ppi[key]
                        c[h] += ppi[key]

            c_sum = sum(c.values())
            t = {}
            for g in present:
                t_topo = c[g] * k / c_sum
                t[g] = (1 - lam) * 1.0 + lam * t_topo

            has_edge = sum(1 for g in present if c[g] > 1.0)
            ppi_cov = has_edge / k if k > 0 else 0.0

        weights[P_name] = t

        t_arr = np.array(list(t.values()))
        t_norm = t_arr / t_arr.sum()
        entropy = -np.sum(t_norm * np.log(t_norm + 1e-12))
        max_entropy = np.log(k) if k > 0 else 0
        weight_diagnostics[P_name] = {
            'n_pathway_genes': len(genes_deduped),
            'n_mapped': k,
            'max_weight': float(t_arr.max()),
            'weight_entropy': float(entropy),
            'effective_n': float(np.exp(entropy)),
            'max_entropy': float(max_entropy),
            'ppi_coverage': float(ppi_cov),
            'mapping_status': 'mapped',
        }

    return weights, weight_diagnostics


# ---------------------------------------------------------------------------
# §3.3  Pathway scoring
# ---------------------------------------------------------------------------

def score_pathways(q_matrix, w_matrix, topo_weights, pathways,
                   score_denominator, min_detected, min_coverage,
                   contrast_samples=None):
    pathway_names = list(pathways.keys())
    sample_names = list(q_matrix.columns)
    scores = pd.DataFrame(np.nan, index=pathway_names, columns=sample_names,
                          dtype=float)

    C_global = None
    if score_denominator == 'complete_case' and contrast_samples is not None:
        C_global = set()
        for g in q_matrix.index:
            if all(np.isfinite(q_matrix.loc[g, s])
                   for s in contrast_samples
                   if s in q_matrix.columns):
                C_global.add(g)

    for P_name in pathway_names:
        if P_name not in topo_weights:
            continue
        t = topo_weights[P_name]
        genes_deduped = pathways[P_name]
        G = [g for g in genes_deduped if g in t]

        if not G:
            continue

        required = max(min_detected,
                       math.ceil(min_coverage * len(genes_deduped)))

        if score_denominator == 'complete_case':
            C_P = [g for g in G if g in C_global] if C_global else []
            if len(C_P) < required:
                continue
            for s in sample_names:
                vals_q = []
                vals_w = []
                vals_t = []
                all_finite = True
                for g in C_P:
                    qv = q_matrix.loc[g, s] if g in q_matrix.index else np.nan
                    wv = w_matrix.loc[g, s] if g in w_matrix.index else np.nan
                    if not (np.isfinite(qv) and np.isfinite(wv)):
                        all_finite = False
                        break
                    vals_q.append(qv)
                    vals_w.append(wv)
                    vals_t.append(t[g])
                if not all_finite:
                    continue
                denom = sum(vals_t)
                if denom > 0:
                    scores.loc[P_name, s] = (
                        sum(ti * wi * qi for ti, wi, qi
                            in zip(vals_t, vals_w, vals_q)) / denom)
        else:
            for s in sample_names:
                vals_q = []
                vals_w = []
                vals_t = []
                for g in G:
                    qv = q_matrix.loc[g, s] if g in q_matrix.index else np.nan
                    wv = w_matrix.loc[g, s] if g in w_matrix.index else np.nan
                    if np.isfinite(qv) and np.isfinite(wv):
                        vals_q.append(qv)
                        vals_w.append(wv)
                        vals_t.append(t[g])

                if len(vals_q) < required:
                    continue

                numer = sum(ti * wi * qi for ti, wi, qi
                            in zip(vals_t, vals_w, vals_q))

                if score_denominator == 'valid_topology':
                    denom = sum(vals_t)
                elif score_denominator == 'kernel_weighted':
                    denom = sum(ti * wi for ti, wi
                                in zip(vals_t, vals_w))
                elif score_denominator == 'fixed_mapped':
                    denom = sum(t[g] for g in G)
                else:
                    denom = sum(vals_t)

                if denom > 0:
                    scores.loc[P_name, s] = numer / denom

    return scores


# ---------------------------------------------------------------------------
# §5.2  Welch t  (vectorized, NaN-aware)
# ---------------------------------------------------------------------------

def welch_t_per_row(data, idx_a, idx_b, min_group_units):
    a = data[:, idx_a]
    b = data[:, idx_b]
    n_a = np.sum(~np.isnan(a), axis=1)
    n_b = np.sum(~np.isnan(b), axis=1)

    sufficient = (n_a >= min_group_units) & (n_b >= min_group_units)
    t = np.full(data.shape[0], np.nan)
    df = np.full(data.shape[0], np.nan)

    if not sufficient.any():
        return t, df

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_a = np.nanmean(a[sufficient], axis=1)
        mean_b = np.nanmean(b[sufficient], axis=1)
        var_a = np.nanvar(a[sufficient], axis=1, ddof=1)
        var_b = np.nanvar(b[sufficient], axis=1, ddof=1)
        na_s = n_a[sufficient].astype(float)
        nb_s = n_b[sufficient].astype(float)
        se2 = var_a / na_s + var_b / nb_s
        se = np.sqrt(se2)

        df_num = se2 ** 2
        df_den = ((var_a / na_s) ** 2 / (na_s - 1) +
                  (var_b / nb_s) ** 2 / (nb_s - 1))
        df[sufficient] = np.where(df_den > 0, df_num / df_den, np.nan)

        diff = mean_a - mean_b
        t[sufficient] = np.where(
            se > 0, diff / se,
            np.where(diff == 0, 0.0, np.sign(diff) * np.inf))
    return t, df


def welch_p_from_t(t, df):
    p = np.full_like(t, np.nan)
    valid = np.isfinite(t) & np.isfinite(df)
    p[valid] = 2.0 * scipy.stats.t.sf(np.abs(t[valid]), df[valid])
    p[np.isinf(t) & ~np.isnan(t)] = 0.0
    p[t == 0] = 1.0
    return p


# ---------------------------------------------------------------------------
# §5.5  Hedges' g
# ---------------------------------------------------------------------------

def hedges_g(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan

    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b)
                         / (n_a + n_b - 2))

    if pooled_std == 0:
        if np.mean(a) == np.mean(b):
            return 0.0
        else:
            return np.nan

    d = (np.mean(a) - np.mean(b)) / pooled_std
    J = 1 - 3 / (4 * (n_a + n_b - 2) - 1)
    return d * J


# ---------------------------------------------------------------------------
# §5.1 / §5.2  Permutation engine  (pathway scores)
# ---------------------------------------------------------------------------

def _generate_assignment_batches(labels, n_group_a, mode, n_permutations,
                                 rng, batch_size=500):
    n = len(labels)
    if mode == 'exact':
        from itertools import combinations
        all_combos = list(combinations(range(n), n_group_a))
        for start in range(0, len(all_combos), batch_size):
            batch = all_combos[start:start + batch_size]
            A = np.zeros((n, len(batch)), dtype=float)
            for j, combo in enumerate(batch):
                A[list(combo), j] = 1.0
            yield A
    else:
        drawn = 0
        while drawn < n_permutations:
            this_batch = min(batch_size, n_permutations - drawn)
            A = np.zeros((n, this_batch), dtype=float)
            for j in range(this_batch):
                perm = rng.choice(n, n_group_a, replace=False)
                A[perm, j] = 1.0
            drawn += this_batch
            yield A


def _initialize_pathway_test(scores, idx_a, idx_b, min_group_units):
    t_obs, _ = welch_t_per_row(scores, idx_a, idx_b, min_group_units)
    state = {
        't_obs': t_obs,
        'exceed': np.zeros(scores.shape[0], dtype=np.int64),
        'valid_draws': np.zeros(scores.shape[0], dtype=np.int64),
        'scores': scores,
        'min_group_units': min_group_units,
    }
    return t_obs, state


def _update_pathway_counts(A, state):
    scores = state['scores']
    min_gu = state['min_group_units']
    t_obs = state['t_obs']

    Y0 = np.nan_to_num(scores, nan=0.0)
    R = np.isfinite(scores).astype(float)
    Y2 = Y0 ** 2
    Ac = 1 - A

    sum_a = Y0 @ A
    ss_a = Y2 @ A
    n_a = R @ A
    sum_b = Y0 @ Ac
    ss_b = Y2 @ Ac
    n_b = R @ Ac

    sufficient = (n_a >= min_gu) & (n_b >= min_gu)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_a = np.where(n_a > 0, sum_a / n_a, 0.0)
        var_a = np.maximum((ss_a - sum_a ** 2 / n_a) / (n_a - 1), 0.0)
        mean_b = np.where(n_b > 0, sum_b / n_b, 0.0)
        var_b = np.maximum((ss_b - sum_b ** 2 / n_b) / (n_b - 1), 0.0)
        se = np.sqrt(var_a / n_a + var_b / n_b)
        diff = mean_a - mean_b
        t_null = np.where(se > 0, diff / se,
                          np.where(diff == 0, 0.0, np.sign(diff) * np.inf))

    t_null = np.where(sufficient, t_null, np.nan)
    computable = ~np.isnan(t_null)

    state['exceed'] += np.sum(
        computable & (np.abs(t_null) >= np.abs(t_obs[:, None])), axis=1)
    state['valid_draws'] += np.sum(computable, axis=1)


def _finalize_pathway_pvalues(state, perm_mode, total_draws):
    t_obs = state['t_obs']
    exceed = state['exceed']
    valid_draws = state['valid_draws']

    if perm_mode == 'exact':
        p_values = np.where(valid_draws > 0,
                            exceed / valid_draws, np.nan)
    else:
        p_values = np.where(valid_draws > 0,
                            (exceed + 1) / (valid_draws + 1), np.nan)

    p_values[np.isnan(t_obs)] = np.nan

    return p_values, {
        'valid_draws': valid_draws,
        'total_draws': total_draws,
        'valid_draw_fraction': (
            valid_draws / total_draws
            if total_draws > 0 else np.full_like(valid_draws, np.nan,
                                                  dtype=float)
        ),
    }


# ---------------------------------------------------------------------------
# §5.3  SCD permutation
# ---------------------------------------------------------------------------

def _build_scd_weight_matrix(pathways, topo_weights, pathway_masks,
                              pathway_index, gene_index):
    n_pathways = len(pathway_index)
    n_genes = len(gene_index)
    gene_to_idx = {g: i for i, g in enumerate(gene_index)}

    rows, cols, vals = [], [], []
    for p_idx, P_name in enumerate(pathway_index):
        if P_name not in pathway_masks:
            continue
        for g in pathway_masks[P_name]:
            if g in gene_to_idx and P_name in topo_weights and g in topo_weights[P_name]:
                rows.append(p_idx)
                cols.append(gene_to_idx[g])
                vals.append(topo_weights[P_name][g])

    W = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_pathways, n_genes))
    return W


def _initialize_scd_test(unit_q_contrast, unit_groups, topo_weights,
                          pathways, pathway_masks, contrast,
                          pathway_index, dead_zone=0.05):
    coh_obs = compute_group_coherence(
        unit_q_contrast, unit_groups, pathways, topo_weights,
        pathway_masks, contrast, dead_zone)

    if coh_obs.empty:
        coh_obs = pd.DataFrame(columns=[
            'pathway', 'n_pathway_genes', 'n_mapped', 'n_tested',
            'coverage_detected', 'coverage_scd',
            'net_shift', 'abs_shift', 'discordance', 'discordance_fraction',
        ])

    coh_obs = coh_obs.set_index('pathway').reindex(pathway_index)

    N_obs = coh_obs['net_shift'].to_numpy()
    U_obs = coh_obs['discordance'].to_numpy()
    n_pathways = len(pathway_index)

    if np.all(np.isnan(N_obs)):
        return N_obs, U_obs, coh_obs, None

    gene_index = unit_q_contrast.index
    W = _build_scd_weight_matrix(pathways, topo_weights, pathway_masks,
                                  pathway_index, gene_index)
    denom = np.asarray(W.sum(axis=1)).ravel()
    denom[denom == 0] = np.nan

    state = {
        'N_obs': N_obs,
        'U_obs': U_obs,
        'exceed_n': np.zeros(n_pathways, dtype=np.int64),
        'exceed_u': np.zeros(n_pathways, dtype=np.int64),
        'W': W,
        'denom': denom[:, None],
        'Q0': np.nan_to_num(unit_q_contrast.to_numpy(), nan=0.0),
        'R': np.isfinite(unit_q_contrast.to_numpy()).astype(float),
    }
    return N_obs, U_obs, coh_obs, state


def _update_scd_counts(A, state):
    Q0 = state['Q0']
    R = state['R']
    W = state['W']
    denom = state['denom']
    Ac = 1 - A

    sum_a = Q0 @ A
    count_a = R @ A
    sum_b = Q0 @ Ac
    count_b = R @ Ac

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_a = np.where(count_a > 0, sum_a / count_a, 0.0)
        mean_b = np.where(count_b > 0, sum_b / count_b, 0.0)
    delta = mean_a - mean_b

    N_null = (W @ delta) / denom
    M_null = (W @ np.abs(delta)) / denom
    U_null, _ = decompose_shift(N_null, M_null)

    state['exceed_n'] += np.sum(
        np.abs(N_null) >= np.abs(state['N_obs'][:, None]), axis=1)
    state['exceed_u'] += np.sum(
        U_null >= state['U_obs'][:, None], axis=1)


def _finalize_scd_pvalues(state, perm_mode, total_draws):
    N_obs = state['N_obs']
    U_obs = state['U_obs']
    exceed_n = state['exceed_n']
    exceed_u = state['exceed_u']

    if perm_mode == 'exact':
        p_net = exceed_n / total_draws
        p_disc = exceed_u / total_draws
    else:
        p_net = (exceed_n + 1) / (total_draws + 1)
        p_disc = (exceed_u + 1) / (total_draws + 1)

    p_net = p_net.astype(float)
    p_disc = p_disc.astype(float)
    p_net[~np.isfinite(N_obs)] = np.nan
    p_disc[~np.isfinite(U_obs)] = np.nan

    return p_net, p_disc


# ---------------------------------------------------------------------------
# §5.4  FDR
# ---------------------------------------------------------------------------

def _fdr_correct(p_values, method='BH'):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return q

    # Benjamini-Hochberg/BY step-up adjustment, implemented locally so the
    # core package keeps the design-specified numpy/pandas/scipy dependency
    # footprint.  NaNs are excluded from the family size.
    p_finite = p[finite_mask]
    order = np.argsort(p_finite, kind='stable')
    ranked = p_finite[order]
    m = len(ranked)
    scale = m / np.arange(1, m + 1, dtype=float)
    if method == 'BY':
        scale *= np.sum(1.0 / np.arange(1, m + 1, dtype=float))
    adjusted_sorted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    q[finite_mask] = adjusted
    return q


# ---------------------------------------------------------------------------
# §5.6  Category assignment
# ---------------------------------------------------------------------------

def _assign_category(row, alpha, min_effect=0.01):
    if pd.isna(row.get('q_net_shift')) or pd.isna(row.get('q_discordant')):
        return np.nan

    sig_net = row['q_net_shift'] < alpha
    sig_disc = row['q_discordant'] < alpha
    practical_net = sig_net and abs(row['net_shift']) > min_effect

    if sig_disc and not practical_net:
        return 'significant_opposing_component'
    elif practical_net and not sig_disc:
        if row['net_shift'] > 0:
            return 'net_shift_higher_rank'
        else:
            return 'net_shift_lower_rank'
    elif sig_net and not practical_net and not sig_disc:
        return 'net_shift_negligible'
    elif practical_net and sig_disc:
        return 'mixed_with_net_shift'
    else:
        return 'not_significant'


# ---------------------------------------------------------------------------
# §9.1  Pathway normalization
# ---------------------------------------------------------------------------

def _normalize_pathways(pathways):
    normalized = {}
    for name, value in pathways.items():
        genes = value
        if (isinstance(genes, (str, bytes, dict))
                or not hasattr(genes, '__iter__')):
            raise ValueError(
                f"pathway {name!r} genes must be an iterable of symbols")
        normalized[name] = tuple(dict.fromkeys(
            str(g).strip().upper()
            for g in genes
            if g is not None and str(g).strip()
        ))
    return normalized


# ---------------------------------------------------------------------------
# §6.1  Main entry point
# ---------------------------------------------------------------------------

def qpsp_pro(
    expression: pd.DataFrame,
    pathways: dict[str, list[str]],
    contrast: tuple[str, str],
    groups: dict[str, str],
    ppi_path: str | None = None,
    ranking_universe: str = 'analysis_complete_case',
    ranking_min_coverage: float = 0.5,
    kernel: str = DEFAULT_KERNEL,
    tau: float = DEFAULT_TAU,
    kappa: float = DEFAULT_KAPPA,
    kernel_quantiles: tuple[float, ...] | None = None,
    kernel_weights: tuple[float, ...] | None = None,
    topology: str = 'weighted_degree',
    topology_strength: float = 0.5,
    ppi_threshold: float = 0.75,
    score_denominator: str = 'kernel_weighted',
    scd: str = 'permutation',
    scd_min_effect: float = 0.01,
    pathway_test: str = 'permutation',
    n_permutations: int = 9999,
    design: str = 'independent',
    blocks: dict[str, str] | None = None,
    fdr_method: str = 'BH',
    fdr_threshold: float = 0.05,
    hypothesis_families: str = 'separate',
    min_detected: int = 3,
    min_coverage: float = 0.1,
    min_group_units: int = 3,
    min_protein_obs_per_group: int = 3,
    seed: int = 42,
) -> QpspProResult:

    if design == 'paired':
        raise NotImplementedError(
            "design='paired' is planned for v2; v1 supports only "
            "independent and clustered between-unit contrasts")

    # -- Validate --
    _validate_params(
        tau=tau, kappa=kappa, topology_strength=topology_strength,
        ppi_threshold=ppi_threshold,
        ranking_min_coverage=ranking_min_coverage,
        min_coverage=min_coverage, fdr_threshold=fdr_threshold,
        n_permutations=n_permutations, min_detected=min_detected,
        min_group_units=min_group_units,
        min_protein_obs_per_group=min_protein_obs_per_group,
        scd_min_effect=scd_min_effect, contrast=contrast,
        kernel_quantiles=kernel_quantiles, kernel_weights=kernel_weights,
        kernel=kernel, score_denominator=score_denominator,
        topology=topology, ranking_universe=ranking_universe,
        design=design, pathway_test=pathway_test, scd=scd,
        fdr_method=fdr_method, hypothesis_families=hypothesis_families,
    )

    if topology == 'weighted_degree' and ppi_path is None:
        raise ValueError(
            "ppi_path is required when topology='weighted_degree'")

    if design == 'clustered' and blocks is None:
        raise ValueError("blocks is required when design='clustered'")

    if not isinstance(expression, pd.DataFrame):
        raise ValueError("expression must be a pandas DataFrame")

    if not expression.columns.is_unique:
        raise ValueError("expression must have unique column names (samples)")

    missing_groups = [s for s in expression.columns if s not in groups]
    if missing_groups:
        raise ValueError(
            "all expression samples must appear in groups; missing: "
            f"{missing_groups[:5]}")
    if design == 'clustered':
        missing_blocks = [s for s in expression.columns if s not in blocks]
        if missing_blocks:
            raise ValueError(
                "all expression samples must appear in blocks for "
                f"design='clustered'; missing: {missing_blocks[:5]}")

    group_a, group_b = contrast
    # Extra entries in ``groups`` must not make an absent contrast arm appear
    # present.  Inference is defined only by columns in the expression matrix.
    present_groups = {groups[sample] for sample in expression.columns}
    if group_a not in present_groups:
        raise ValueError(f"contrast group {group_a!r} not found in groups")
    if group_b not in present_groups:
        raise ValueError(f"contrast group {group_b!r} not found in groups")

    # Normalize expression gene symbols once, matching normalized pathways.
    # Work on a copy so the caller's DataFrame is never mutated.
    expression = expression.copy()
    expression.index = pd.Index(
        [str(g).strip().upper() for g in expression.index],
        name=expression.index.name,
    )
    if expression.index.duplicated().any():
        duplicates = expression.index[expression.index.duplicated()].unique()
        raise ValueError(
            "expression must have unique row names (proteins) after "
            f"uppercasing; duplicates: {list(duplicates[:5])}")

    try:
        expression_values = expression.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("expression values must be numeric") from exc
    if np.any(np.isinf(expression_values)):
        raise ValueError("expression contains ±inf values")
    # Ensure downstream pandas operations have numeric dtypes even when the
    # input used object columns containing numeric Python values.
    expression = pd.DataFrame(
        expression_values, index=expression.index, columns=expression.columns)

    # -- Normalize pathways --
    normalized_pathways = _normalize_pathways(pathways)

    # -- Contrast samples --
    fit_samples = [s for s in expression.columns
                   if groups.get(s) in {group_a, group_b}]
    ranking_fit_samples = (
        list(expression.columns)
        if ranking_universe == 'analysis_complete_case'
        else fit_samples)
    expression_fit = expression[ranking_fit_samples]

    # -- §2.3: Expression scores --
    ranking_model = fit_ranking_model(
        expression_fit, ranking_universe, ranking_min_coverage)
    if (ranking_universe in {'contrast_complete_case',
                             'analysis_complete_case'} and
            len(ranking_model['universe']) < min_detected):
        raise ValueError(
            "contrast_complete_case retained fewer proteins than "
            f"min_detected ({len(ranking_model['universe'])} < "
            f"{min_detected})")
    q_matrix, expr_diag = compute_expression_scores(expression, ranking_model)

    # -- §2.4: Detection balance --
    balance_blocks = blocks if design == 'clustered' else None
    raw_balance = check_detection_balance(
        expr_diag['n_detected_raw_per_sample'], groups, contrast,
        balance_blocks)
    ranked_balance = check_detection_balance(
        expr_diag['n_ranked_per_sample'], groups, contrast,
        balance_blocks)
    if (raw_balance is not None and raw_balance['warning'] and
            ranking_universe in {'per_sample', 'shared'}):
        warnings.warn(
            "Detection depth differs between contrast groups while "
            f"ranking_universe={ranking_universe!r} permits a variable "
            "ranking denominator. Treat this run as sensitivity-only and "
            "use a prespecified complete-case universe (the default is "
            "ranking_universe='analysis_complete_case') for confirmatory "
            "analysis.", RuntimeWarning)

    # -- Detected set --
    q_fit = q_matrix[ranking_fit_samples]
    detected = frozenset(
        g for g in q_fit.index
        if q_fit.loc[g].notna().any()
    )

    # -- §3.2: Topology weights --
    ppi = None
    if topology == 'weighted_degree':
        ppi = load_ppi(ppi_path, ppi_threshold)

    topo_weights, topo_diag = compute_topology_weights(
        normalized_pathways, detected, ppi, topology_strength,
        topology, min_detected, min_coverage)

    # -- §3.1: Kernel weights --
    w_matrix = compute_kernel_weights(
        q_matrix, kernel, tau, kappa,
        kernel_quantiles=kernel_quantiles,
        kernel_weights=kernel_weights)

    # -- §3.3: Pathway scoring --
    pathway_scores = score_pathways(
        q_matrix, w_matrix, topo_weights, normalized_pathways,
        score_denominator, min_detected, min_coverage,
        contrast_samples=fit_samples)

    # -- §2.5: Analysis units --
    if design == 'clustered':
        unit_scores, unit_groups = aggregate_to_participants(
            pathway_scores, blocks, groups)
        unit_q, _ = aggregate_to_participants(
            q_matrix, blocks, groups)
    else:
        unit_scores = pathway_scores
        unit_q = q_matrix
        unit_groups = groups

    analysis_units = tuple(sorted(
        set(unit_scores.columns)
        & set(unit_q.columns)
        & {u for u, g in unit_groups.items() if g in {group_a, group_b}}
    ))

    unit_scores_contrast = unit_scores.reindex(columns=list(analysis_units))
    unit_q_contrast = unit_q.reindex(columns=list(analysis_units))
    labels = np.array([unit_groups[u] for u in analysis_units])
    n_a = int(np.sum(labels == group_a))
    n_b = int(np.sum(labels == group_b))

    idx_a = np.where(labels == group_a)[0]
    idx_b = np.where(labels == group_b)[0]

    # -- Master pathway index --
    pathway_index = pd.Index(list(normalized_pathways.keys()), name='pathway')

    scores_arr = unit_scores_contrast.reindex(
        index=pathway_index).to_numpy()

    # -- §5: Statistical inference --
    K = math.comb(n_a + n_b, n_a)
    perm_mode = 'exact' if K <= n_permutations else 'monte_carlo'

    pathway_state = None
    scd_state = None
    coh_obs = None
    pw_diag = None
    p_pathway = np.full(len(pathway_index), np.nan)
    t_obs = np.full(len(pathway_index), np.nan)
    p_net = np.full(len(pathway_index), np.nan)
    p_disc = np.full(len(pathway_index), np.nan)
    N_obs = np.full(len(pathway_index), np.nan)
    U_obs = np.full(len(pathway_index), np.nan)

    can_test = n_a >= min_group_units and n_b >= min_group_units

    # -- Pathway test initialization --
    if can_test:
        if pathway_test == 'welch':
            t_obs, df_obs = welch_t_per_row(
                scores_arr, idx_a, idx_b, min_group_units)
            p_pathway = welch_p_from_t(t_obs, df_obs)
        elif pathway_test == 'permutation':
            t_obs, pathway_state = _initialize_pathway_test(
                scores_arr, idx_a, idx_b, min_group_units)

    # -- SCD initialization --
    scd_mask_type = 'none'
    pathway_masks = {}

    if scd != 'none' and (scd == 'descriptive' or can_test):
        if scd == 'permutation':
            mask_vec = build_scd_eligible_mask(
                unit_q_contrast.reindex(index=q_matrix.index).to_numpy(),
                n_a, n_b, min_protein_obs_per_group)
            scd_mask_type = 'permutation_safe'
            if mask_vec is not None:
                gene_list = list(unit_q_contrast.reindex(
                    index=q_matrix.index).index)
                gene_positions = {g: i for i, g in enumerate(gene_list)}
                for P_name in normalized_pathways:
                    if P_name not in topo_weights:
                        continue
                    mapped = list(topo_weights[P_name].keys())
                    present = [g for g in mapped
                               if g in gene_positions and
                               mask_vec[gene_positions[g]]]
                    required = max(
                        min_detected,
                        math.ceil(min_coverage *
                                  len(normalized_pathways[P_name])))
                    if len(present) >= required:
                        pathway_masks[P_name] = present

                N_obs, U_obs, coh_obs, scd_state = _initialize_scd_test(
                    unit_q_contrast.reindex(index=q_matrix.index),
                    {u: unit_groups[u] for u in analysis_units},
                    topo_weights, normalized_pathways, pathway_masks,
                    contrast, pathway_index)

        elif scd == 'descriptive':
            scd_mask_type = 'observed_group'
            q_arr = unit_q_contrast.reindex(index=q_matrix.index).to_numpy()
            desc_mask = build_descriptive_scd_mask(
                q_arr, labels, group_a, group_b,
                min_protein_obs_per_group)
            if desc_mask is not None:
                gene_list = list(unit_q_contrast.reindex(
                    index=q_matrix.index).index)
                gene_positions = {g: i for i, g in enumerate(gene_list)}
                for P_name in normalized_pathways:
                    if P_name not in topo_weights:
                        continue
                    mapped = list(topo_weights[P_name].keys())
                    present = [g for g in mapped
                               if g in gene_positions and
                               desc_mask[gene_positions[g]]]
                    required = max(
                        min_detected,
                        math.ceil(min_coverage *
                                  len(normalized_pathways[P_name])))
                    if len(present) >= required:
                        pathway_masks[P_name] = present

                coh_obs = compute_group_coherence(
                    unit_q_contrast.reindex(index=q_matrix.index),
                    {u: unit_groups[u] for u in analysis_units},
                    normalized_pathways, topo_weights, pathway_masks,
                    contrast)
                if not coh_obs.empty:
                    coh_obs = coh_obs.set_index('pathway').reindex(
                        pathway_index)
                    N_obs = coh_obs['net_shift'].to_numpy()
                    U_obs = coh_obs['discordance'].to_numpy()

    # -- Permutation loop --
    total_draws = 0
    if pathway_state is not None or scd_state is not None:
        rng = np.random.default_rng(seed)
        for A in _generate_assignment_batches(
            labels=labels, n_group_a=n_a, mode=perm_mode,
            n_permutations=n_permutations, rng=rng,
        ):
            total_draws += A.shape[1]
            if pathway_state is not None:
                _update_pathway_counts(A, pathway_state)
            if scd_state is not None:
                _update_scd_counts(A, scd_state)

    # -- Finalize p-values --
    if pathway_state is not None:
        p_pathway, pw_diag = _finalize_pathway_pvalues(
            pathway_state, perm_mode, total_draws)

    if scd_state is not None:
        p_net, p_disc = _finalize_scd_pvalues(
            scd_state, perm_mode, total_draws)

    # -- §5.4: FDR --
    if hypothesis_families == 'separate':
        q_pathway = _fdr_correct(p_pathway, fdr_method)
        if scd == 'permutation':
            p_scd_pool = np.concatenate([p_net, p_disc])
            q_scd_pool = _fdr_correct(p_scd_pool, fdr_method)
            q_net = q_scd_pool[:len(p_net)]
            q_disc = q_scd_pool[len(p_net):]
        else:
            q_net = np.full_like(p_net, np.nan)
            q_disc = np.full_like(p_disc, np.nan)
    else:
        p_all = np.concatenate([p_pathway, p_net, p_disc])
        q_all = _fdr_correct(p_all, fdr_method)
        n_pw = len(p_pathway)
        n_pn = len(p_net)
        q_pathway = q_all[:n_pw]
        q_net = q_all[n_pw:n_pw + n_pn]
        q_disc = q_all[n_pw + n_pn:]

    # -- Effect sizes & group means --
    hg = np.full(len(pathway_index), np.nan)
    mean_test = np.full(len(pathway_index), np.nan)
    mean_ref = np.full(len(pathway_index), np.nan)
    score_diff = np.full(len(pathway_index), np.nan)
    n_test_units = np.full(len(pathway_index), 0, dtype=int)
    n_ref_units = np.full(len(pathway_index), 0, dtype=int)

    for i, P_name in enumerate(pathway_index):
        row = scores_arr[i]
        a_vals = row[idx_a]
        b_vals = row[idx_b]
        fa = a_vals[np.isfinite(a_vals)]
        fb = b_vals[np.isfinite(b_vals)]
        n_test_units[i] = len(fa)
        n_ref_units[i] = len(fb)
        if len(fa) > 0:
            mean_test[i] = np.mean(fa)
        if len(fb) > 0:
            mean_ref[i] = np.mean(fb)
        if len(fa) > 0 and len(fb) > 0:
            score_diff[i] = np.mean(fa) - np.mean(fb)
        hg[i] = hedges_g(fa, fb)

    # -- Significance booleans --
    sig_pathway = np.where(np.isfinite(q_pathway),
                           q_pathway < fdr_threshold, False)
    sig_net = np.where(np.isfinite(q_net),
                       q_net < fdr_threshold, False)
    sig_disc = np.where(np.isfinite(q_disc),
                        q_disc < fdr_threshold, False)

    # -- Pathway test status --
    pathway_perm_mode_col = []
    pathway_test_status_col = []
    for i, P_name in enumerate(pathway_index):
        diag = topo_diag.get(P_name, {})
        ms = diag.get('mapping_status', 'mapped')
        if ms != 'mapped':
            pathway_perm_mode_col.append('insufficient')
            pathway_test_status_col.append('insufficient_mapping')
        elif not can_test:
            pathway_perm_mode_col.append('insufficient')
            pathway_test_status_col.append('insufficient_group_size')
        elif np.isnan(t_obs[i]):
            pathway_perm_mode_col.append('insufficient')
            pathway_test_status_col.append('insufficient_pathway_scores')
        elif (pathway_test == 'permutation' and pw_diag is not None and
              pw_diag['valid_draws'][i] == 0):
            pathway_perm_mode_col.append('insufficient')
            pathway_test_status_col.append(
                'insufficient_valid_permutations')
        else:
            if pathway_test == 'welch':
                pathway_perm_mode_col.append('welch')
            else:
                pathway_perm_mode_col.append(perm_mode)
            pathway_test_status_col.append('tested')

    # -- SCD test status --
    scd_perm_mode_col = []
    scd_test_status_col = []
    for i, P_name in enumerate(pathway_index):
        if scd == 'none':
            scd_perm_mode_col.append('none')
            scd_test_status_col.append('not_requested')
        elif scd == 'descriptive':
            scd_perm_mode_col.append('none')
            scd_test_status_col.append('descriptive')
        elif not can_test:
            scd_perm_mode_col.append('insufficient')
            scd_test_status_col.append('insufficient_group_size')
        elif scd_state is None:
            scd_perm_mode_col.append('insufficient')
            scd_test_status_col.append('insufficient_scd_mask')
        elif np.isnan(N_obs[i]):
            scd_perm_mode_col.append('insufficient')
            scd_test_status_col.append('insufficient_scd_mask')
        else:
            scd_perm_mode_col.append(perm_mode)
            scd_test_status_col.append('tested')

    # -- Mapping status --
    mapping_status_col = []
    n_pathway_genes_col = []
    n_mapped_col = []
    ppi_coverage_col = []
    for P_name in pathway_index:
        diag = topo_diag.get(P_name, {})
        mapping_status_col.append(diag.get('mapping_status', 'mapped'))
        n_pathway_genes_col.append(
            diag.get('n_pathway_genes', len(normalized_pathways.get(P_name, ()))))
        n_mapped_col.append(diag.get('n_mapped', 0))
        ppi_coverage_col.append(diag.get('ppi_coverage', np.nan))

    # -- SCD columns from coh_obs --
    net_shift_col = N_obs.copy()
    abs_shift_col = np.full(len(pathway_index), np.nan)
    discordance_col = U_obs.copy()
    disc_frac_col = np.full(len(pathway_index), np.nan)
    n_tested_col = np.full(len(pathway_index), 0, dtype=int)
    cov_detected_col = np.array([
        (n_mapped_col[i] / n_pathway_genes_col[i]
         if n_pathway_genes_col[i] > 0 else 0.0)
        for i in range(len(pathway_index))
    ], dtype=float)
    cov_scd_col = np.full(len(pathway_index), np.nan)

    if coh_obs is not None and isinstance(coh_obs, pd.DataFrame):
        if 'abs_shift' in coh_obs.columns:
            for i, P_name in enumerate(pathway_index):
                if P_name in coh_obs.index:
                    row = coh_obs.loc[P_name]
                    abs_shift_col[i] = row.get('abs_shift', np.nan)
                    disc_frac_col[i] = row.get('discordance_fraction', np.nan)
                    n_tested_col[i] = int(row.get('n_tested', 0)) if pd.notna(row.get('n_tested')) else 0
                    cov_detected_col[i] = row.get('coverage_detected', np.nan)
                    cov_scd_col[i] = row.get('coverage_scd', np.nan)

    member_effects = compute_member_effects(
        unit_q_contrast.reindex(index=q_matrix.index),
        {u: unit_groups[u] for u in analysis_units},
        topo_weights,
        pathway_masks,
        contrast,
    )

    # -- §5.6: Category --
    scd_category_col = []
    for i in range(len(pathway_index)):
        row_data = {
            'q_net_shift': q_net[i],
            'q_discordant': q_disc[i],
            'net_shift': net_shift_col[i],
        }
        scd_category_col.append(
            _assign_category(row_data, fdr_threshold, scd_min_effect))

    # -- Build significance DataFrame --
    sig_df = pd.DataFrame({
        'pathway': list(pathway_index),
        'n_pathway_genes': n_pathway_genes_col,
        'n_mapped': n_mapped_col,
        'n_tested': n_tested_col,
        'coverage_detected': cov_detected_col,
        'coverage_scd': cov_scd_col,
        'ppi_coverage': ppi_coverage_col,
        't_statistic': t_obs,
        'p_pathway': p_pathway,
        'q_pathway': q_pathway,
        'hedges_g': hg,
        'mean_score_test': mean_test,
        'mean_score_ref': mean_ref,
        'score_diff': score_diff,
        'n_test_units': n_test_units,
        'n_ref_units': n_ref_units,
        'sig_pathway': sig_pathway,
        'pathway_perm_mode': pathway_perm_mode_col,
        'net_shift': net_shift_col,
        'abs_shift': abs_shift_col,
        'discordance': discordance_col,
        'discordance_fraction': disc_frac_col,
        'p_net_shift': p_net,
        'q_net_shift': q_net,
        'p_discordant': p_disc,
        'q_discordant': q_disc,
        'sig_net_shift': sig_net,
        'sig_discordant': sig_disc,
        'scd_perm_mode': scd_perm_mode_col,
        'scd_category': scd_category_col,
        'mapping_status': mapping_status_col,
        'pathway_test_status': pathway_test_status_col,
        'scd_test_status': scd_test_status_col,
    })

    # -- Reindex pathway_scores to master index --
    pathway_scores = pathway_scores.reindex(index=pathway_index)

    params = dict(
        kernel=kernel, tau=tau, kappa=kappa,
        kernel_quantiles=(None if kernel_quantiles is None else
                          tuple(map(float, kernel_quantiles))),
        kernel_weights=(None if kernel_weights is None else
                        tuple(map(float, kernel_weights))),
        topology=topology, topology_strength=topology_strength,
        ppi_threshold=ppi_threshold, ppi_path=ppi_path,
        score_denominator=score_denominator,
        ranking_universe=ranking_universe,
        ranking_min_coverage=ranking_min_coverage,
        scd=scd, scd_min_effect=scd_min_effect,
        pathway_test=pathway_test, n_permutations=n_permutations,
        design=design, fdr_method=fdr_method,
        fdr_threshold=fdr_threshold,
        hypothesis_families=hypothesis_families,
        min_detected=min_detected, min_coverage=min_coverage,
        min_group_units=min_group_units,
        min_protein_obs_per_group=min_protein_obs_per_group,
        seed=seed,
        contrast=contrast,
    )

    diagnostics = {
        'ranking': expr_diag,
        'raw_detection_balance': raw_balance,
        'ranked_detection_balance': ranked_balance,
        'topology': topo_diag,
        'n_analysis_units': n_a + n_b,
        'n_group_a': n_a,
        'n_group_b': n_b,
        'perm_mode': perm_mode,
        'total_draws': total_draws,
        'n_possible_assignments': K,
        'n_sampled_draws': total_draws if perm_mode == 'monte_carlo' else 0,
        'minimum_attainable_p': (
            (min(1.0, (2.0 if n_a == n_b else 1.0) / K)
             if perm_mode == 'exact' and K > 0
             else 1.0 / (n_permutations + 1))
        ),
        'bh_single_hit_threshold': (
            fdr_threshold / max(1, int(np.isfinite(p_pathway).sum()))
        ),
        'scd_mask_type': scd_mask_type,
        'contrast_samples': list(fit_samples),
        'out_of_contrast_samples': [
            s for s in expression.columns if s not in fit_samples],
    }
    if pathway_state is not None:
        diagnostics['pathway_permutation'] = pw_diag
        diagnostics['pathway_permutation']['low_valid_draw_warning'] = (
            pw_diag['valid_draw_fraction'] < 0.5)

    return QpspProResult(
        pathway_scores=pathway_scores,
        significance=sig_df,
        member_effects=member_effects,
        topology_weights=topo_weights,
        kernel_weights=w_matrix,
        params=params,
        diagnostics=diagnostics,
    )
