"""Smoke tests for qpsp_pro — verify end-to-end pipeline runs."""

import gzip

import numpy as np
import pandas as pd
import pytest

from qpsp_pro import (
    DEFAULT_KAPPA,
    DEFAULT_KERNEL,
    DEFAULT_TAU,
    QpspProResult,
    qpsp_pro,
)
from qpsp_pro.qpsp_pro import compute_kernel_weights, load_ppi
from qpsp_pro.scd import decompose_shift


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_expression(n_proteins=50, n_samples=12, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i}" for i in range(n_proteins)]
    samples = [f"S{i}" for i in range(n_samples)]
    data = rng.normal(size=(n_proteins, n_samples))
    data[rng.random(size=data.shape) < 0.05] = np.nan
    return pd.DataFrame(data, index=genes, columns=samples)


def _make_pathways(n_proteins=50):
    return {
        'PATHWAY_A': [f"GENE{i}" for i in range(0, 15)],
        'PATHWAY_B': [f"GENE{i}" for i in range(10, 30)],
        'PATHWAY_C': [f"GENE{i}" for i in range(25, 45)],
        'PATHWAY_EMPTY': [],
        'PATHWAY_SMALL': [f"GENE{i}" for i in range(48, 50)],
    }


def _make_groups(n_samples=12):
    groups = {}
    for i in range(n_samples):
        groups[f"S{i}"] = 'case' if i < 6 else 'control'
    return groups


def _make_ppi_file(tmp_path, n_proteins=50):
    ppi_path = tmp_path / "test_ppi.txt"
    lines = ["Gene1\tGene2\tScore\n"]
    rng = np.random.default_rng(42)
    for i in range(n_proteins):
        for j in range(i + 1, min(i + 5, n_proteins)):
            score = rng.uniform(0.5, 1.0)
            lines.append(f"GENE{i}\tGENE{j}\t{score:.3f}\n")
    ppi_path.write_text("".join(lines))
    return str(ppi_path)


def test_load_ppi_reads_gzip(tmp_path):
    path = tmp_path / "test_ppi.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("gene1\tgene2\tscore\nA\tB\t0.90\nA\tC\t0.50\n")
    assert load_ppi(str(path), 0.75) == {("A", "B"): 0.9}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDecomposeShift:
    def test_all_same_direction(self):
        U, disc = decompose_shift(0.5, 0.5)
        assert U == pytest.approx(0.0)
        assert disc == pytest.approx(0.0)

    def test_full_cancellation(self):
        U, disc = decompose_shift(0.0, 0.4)
        assert U == pytest.approx(0.4)
        assert disc == pytest.approx(1.0)

    def test_partial(self):
        U, disc = decompose_shift(0.2, 0.6)
        assert U == pytest.approx(0.4)
        assert disc == pytest.approx(0.4 / 0.6)

    def test_zero_M(self):
        U, disc = decompose_shift(0.0, 0.0)
        assert U == pytest.approx(0.0)
        assert np.isnan(disc)

    def test_negative_N(self):
        U, disc = decompose_shift(-0.3, 0.5)
        assert U == pytest.approx(0.2)

    def test_scalar_returns_float(self):
        U, disc = decompose_shift(0.1, 0.3)
        assert isinstance(U, float)
        assert isinstance(disc, float)

    def test_array_input(self):
        N = np.array([0.5, 0.0, -0.2])
        M = np.array([0.5, 0.4, 0.6])
        U, disc = decompose_shift(N, M)
        assert U.shape == (3,)
        assert U[0] == pytest.approx(0.0)
        assert U[1] == pytest.approx(0.4)


def test_result_schema_keeps_member_effects_backward_compatible():
    result = QpspProResult(
        pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), {}, {})
    assert result.member_effects.empty


def test_measurement_calibrated_kernel_reproduces_supplied_anchors():
    quantiles = (0.1, 0.3, 0.5, 0.7, 0.9)
    anchor_weights = (0.1, 0.2, 0.4, 0.8, 1.0)
    q = pd.DataFrame(
        {"S": quantiles},
        index=[f"G{i}" for i in range(
            len(quantiles))])
    weights = compute_kernel_weights(
        q, "measurement_calibrated", tau=.85, kappa=30,
        kernel_quantiles=quantiles,
        kernel_weights=anchor_weights)
    assert weights["S"].to_numpy() == pytest.approx(
        anchor_weights)


def test_measurement_calibrated_kernel_requires_explicit_anchors():
    q = pd.DataFrame({"S": [0.1, 0.9]})
    with pytest.raises(ValueError, match="kernel_quantiles"):
        qpsp_pro(
            expression=_make_expression(), pathways=_make_pathways(),
            contrast=('case', 'control'), groups=_make_groups(),
            kernel='measurement_calibrated', topology='uniform',
            pathway_test='welch', scd='none')


def test_measurement_calibrated_kernel_clamps_and_accepts_custom_anchors():
    q = pd.DataFrame({"S": [-1.0, 0.0, 0.5, 1.0, 2.0, np.nan]})
    weights = compute_kernel_weights(
        q, "measurement_calibrated", tau=.85, kappa=30,
        kernel_quantiles=(0.25, 0.75), kernel_weights=(0.2, 1.0))
    assert weights["S"].iloc[:2].to_numpy() == pytest.approx([0.2, 0.2])
    assert weights["S"].iloc[-3:-1].to_numpy() == pytest.approx([1.0, 1.0])
    assert pd.isna(weights["S"].iloc[-1])


def test_public_defaults_use_dataset_agnostic_full_method_configuration():
    expr = _make_expression()
    result = qpsp_pro(
        expression=expr,
        pathways=_make_pathways(),
        contrast=('case', 'control'),
        groups=_make_groups(),
        topology='uniform',
        pathway_test='welch',
        scd='none',
    )
    assert result.params["kernel"] == DEFAULT_KERNEL == "logistic"
    assert result.params["tau"] == DEFAULT_TAU == pytest.approx(.85)
    assert result.params["kappa"] == DEFAULT_KAPPA == pytest.approx(30)
    assert result.params["score_denominator"] == "kernel_weighted"
    assert result.params["ranking_universe"] == "analysis_complete_case"
    assert result.params["n_permutations"] == 9_999


class TestWelchPipeline:
    def test_contrast_complete_case_fixes_ranking_denominator(self):
        expr = _make_expression()
        expr.loc['GENE0', 'S0'] = np.nan
        result = qpsp_pro(
            expression=expr, pathways=_make_pathways(),
            contrast=('case', 'control'), groups=_make_groups(),
            ranking_universe='contrast_complete_case',
            topology='uniform', pathway_test='welch', scd='none',
        )
        ranking = result.diagnostics['ranking']
        counts = ranking['n_ranked_per_sample']
        assert len(set(counts.values())) == 1
        assert ranking['ranking_denominator_fixed_for_contrast'] is True
        assert result.kernel_weights.loc['GENE0'].isna().all()

    def test_complete_case_is_fit_on_contrast_only(self):
        expr = _make_expression(n_samples=13)
        groups = _make_groups()
        groups['S12'] = 'other'
        expr.loc['GENE0', expr.columns[:12]] = np.arange(12, dtype=float)
        expr.loc['GENE0', 'S12'] = np.nan
        with pytest.warns(RuntimeWarning, match="out-of-contrast"):
            result = qpsp_pro(
                expression=expr,
                pathways={'P': ['GENE0', 'GENE1', 'GENE2']},
                contrast=('case', 'control'), groups=groups,
                ranking_universe='contrast_complete_case',
                topology='uniform', pathway_test='welch', scd='none',
            )
        assert result.kernel_weights.loc['GENE0', 'S0':'S11'].notna().all()
        assert pd.isna(result.kernel_weights.loc['GENE0', 'S12'])
        assert result.diagnostics['out_of_contrast_samples'] == ['S12']

    def test_analysis_complete_case_covers_every_scored_sample(self):
        expr = _make_expression(n_samples=13)
        groups = _make_groups()
        groups['S12'] = 'other'
        expr.loc['GENE0', 'S12'] = np.nan
        result = qpsp_pro(
            expression=expr, pathways={'P': ['GENE0', 'GENE1', 'GENE2']},
            contrast=('case', 'control'), groups=groups,
            ranking_universe='analysis_complete_case',
            topology='uniform', pathway_test='welch', scd='none',
        )
        ranking = result.diagnostics['ranking']
        assert ranking['ranking_denominator_fixed_for_all_scored_samples'] is True
        assert len(set(ranking['n_ranked_per_sample'].values())) == 1
        assert result.kernel_weights.loc['GENE0'].isna().all()

    def test_welch_runs(self, tmp_path):
        expr = _make_expression()
        pathways = _make_pathways()
        groups = _make_groups()

        result = qpsp_pro(
            expression=expr,
            pathways=pathways,
            contrast=('case', 'control'),
            groups=groups,
            topology='uniform',
            pathway_test='welch',
            scd='none',
        )
        assert isinstance(result, QpspProResult)
        assert result.significance.shape[0] == len(pathways)
        assert 'p_pathway' in result.significance.columns
        assert 'sig_pathway' in result.significance.columns

    def test_welch_with_descriptive_scd(self, tmp_path):
        expr = _make_expression()
        pathways = _make_pathways()
        groups = _make_groups()

        result = qpsp_pro(
            expression=expr,
            pathways=pathways,
            contrast=('case', 'control'),
            groups=groups,
            topology='uniform',
            pathway_test='welch',
            scd='descriptive',
        )
        sig = result.significance
        assert sig['net_shift'].notna().any()
        assert sig['p_net_shift'].isna().all()
        assert not result.member_effects.empty
        for pathway, members in result.member_effects.groupby('pathway'):
            summary = sig.set_index('pathway').loc[pathway]
            assert members['weighted_net_contribution'].sum() == \
                pytest.approx(summary['net_shift'])
            assert members['weighted_abs_contribution'].sum() == \
                pytest.approx(summary['abs_shift'])


class TestPermutationPipeline:
    def test_small_exact(self, tmp_path):
        expr = _make_expression(n_samples=6)
        pathways = {
            'P1': [f"GENE{i}" for i in range(0, 15)],
            'P2': [f"GENE{i}" for i in range(10, 30)],
        }
        groups = {f"S{i}": 'A' if i < 3 else 'B' for i in range(6)}

        result = qpsp_pro(
            expression=expr,
            pathways=pathways,
            contrast=('A', 'B'),
            groups=groups,
            topology='uniform',
            pathway_test='permutation',
            scd='permutation',
            n_permutations=1000,
            min_group_units=2,
        )
        assert result.diagnostics['perm_mode'] == 'exact'
        sig = result.significance
        assert sig['p_pathway'].notna().any()
        assert sig['p_net_shift'].notna().any()

    def test_with_topology(self, tmp_path):
        expr = _make_expression()
        pathways = _make_pathways()
        groups = _make_groups()
        ppi_path = _make_ppi_file(tmp_path)

        result = qpsp_pro(
            expression=expr,
            pathways=pathways,
            contrast=('case', 'control'),
            groups=groups,
            ppi_path=ppi_path,
            topology='weighted_degree',
            topology_strength=0.5,
            pathway_test='welch',
            scd='descriptive',
        )
        assert len(result.topology_weights) > 0
        found_variation = False
        for pw, tw in result.topology_weights.items():
            vals = list(tw.values())
            if len(vals) > 1 and not all(v == vals[0] for v in vals):
                found_variation = True
                break
        assert found_variation, "at least one pathway should have varying topology weights"


class TestClusteredDesign:
    def test_clustered(self, tmp_path):
        expr = _make_expression(n_samples=12)
        pathways = {'P1': [f"GENE{i}" for i in range(0, 20)]}
        groups = {}
        blocks = {}
        for i in range(12):
            groups[f"S{i}"] = 'case' if i < 6 else 'control'
            blocks[f"S{i}"] = f"PID{i // 2}"

        result = qpsp_pro(
            expression=expr,
            pathways=pathways,
            contrast=('case', 'control'),
            groups=groups,
            design='clustered',
            blocks=blocks,
            topology='uniform',
            pathway_test='welch',
            scd='none',
        )
        assert result.diagnostics['n_analysis_units'] == 6

    def test_paired_design_is_explicitly_deferred(self):
        with pytest.raises(NotImplementedError, match="planned for v2"):
            qpsp_pro(
                expression=_make_expression(), pathways=_make_pathways(),
                contrast=('case', 'control'), groups=_make_groups(),
                topology='uniform', design='paired',
            )


class TestEdgeCases:
    def test_expression_gene_symbols_are_normalized(self):
        expr = _make_expression()
        expr.index = [g.lower() for g in expr.index]
        result = qpsp_pro(
            expression=expr,
            pathways={'P': ['gene0', ' GENE1 ', 'Gene2']},
            contrast=('case', 'control'), groups=_make_groups(),
            topology='uniform', pathway_test='welch', scd='none',
            ranking_universe='per_sample', min_detected=3,
        )
        assert result.significance.loc[0, 'n_mapped'] == 3
        assert list(expr.index[:2]) == ['gene0', 'gene1']

    def test_case_colliding_gene_symbols_rejected(self):
        expr = _make_expression()
        expr.index = ['gene0', 'GENE0'] + list(expr.index[2:])
        with pytest.raises(ValueError, match='after uppercasing'):
            qpsp_pro(
                expression=expr, pathways={'P': ['GENE0']},
                contrast=('case', 'control'), groups=_make_groups(),
                topology='uniform', pathway_test='welch', scd='none',
            )

    def test_missing_group_assignment_rejected(self):
        expr = _make_expression()
        groups = _make_groups()
        del groups['S11']
        with pytest.raises(ValueError, match='must appear in groups'):
            qpsp_pro(
                expression=expr, pathways=_make_pathways(),
                contrast=('case', 'control'), groups=groups,
                topology='uniform', pathway_test='welch', scd='none',
            )

    def test_contrast_group_must_have_expression_columns(self):
        expr = _make_expression()
        groups = {sample: 'case' for sample in expr.columns}
        groups['NOT_A_COLUMN'] = 'control'
        with pytest.raises(ValueError, match="not found"):
            qpsp_pro(
                expression=expr, pathways=_make_pathways(),
                contrast=('case', 'control'), groups=groups,
                topology='uniform', pathway_test='welch', scd='none',
            )

    def test_insufficient_groups_do_not_run_scd_permutations(self):
        expr = _make_expression(n_samples=4)
        groups = {f'S{i}': 'A' if i < 2 else 'B' for i in range(4)}
        result = qpsp_pro(
            expression=expr,
            pathways={'P': [f'GENE{i}' for i in range(15)]},
            contrast=('A', 'B'), groups=groups, topology='uniform',
            pathway_test='permutation', scd='permutation',
            min_group_units=3, min_protein_obs_per_group=1,
            n_permutations=100,
        )
        assert result.diagnostics['total_draws'] == 0
        assert result.significance.loc[0, 'scd_test_status'] == \
            'insufficient_group_size'
        assert pd.isna(result.significance.loc[0, 'p_net_shift'])

    def test_scd_enforces_pathway_coverage(self):
        expr = _make_expression(n_proteins=10)
        # Leave only two genes eligible in each observed group while pathway
        # mapping itself remains sufficient.
        expr.loc[['GENE2', 'GENE3', 'GENE4'], expr.columns[:6]] = np.nan
        expr.loc[['GENE5', 'GENE6', 'GENE7'], expr.columns[6:]] = np.nan
        result = qpsp_pro(
            expression=expr,
            pathways={'P': [f'GENE{i}' for i in range(8)]},
            contrast=('case', 'control'), groups=_make_groups(),
            topology='uniform', pathway_test='welch', scd='descriptive',
            ranking_universe='per_sample',
            min_detected=3, min_protein_obs_per_group=3,
        )
        row = result.significance.iloc[0]
        assert row['n_mapped'] == 8
        assert row['n_tested'] == 0
        assert pd.isna(row['net_shift'])

    def test_coverage_detected_available_without_scd(self):
        result = qpsp_pro(
            expression=_make_expression(), pathways={'P': ['GENE0', 'NOPE']},
            contrast=('case', 'control'), groups=_make_groups(),
            topology='uniform', pathway_test='welch', scd='none',
            min_detected=1,
        )
        assert result.significance.loc[0, 'coverage_detected'] == 0.5

    def test_pathway_metadata_mapping_without_genes_is_rejected(self):
        with pytest.raises(ValueError, match='iterable of symbols'):
            qpsp_pro(
                expression=_make_expression(),
                pathways={'BAD': {'collection': 'H'}},
                contrast=('case', 'control'), groups=_make_groups(),
                topology='uniform', pathway_test='welch', scd='none',
            )

    def test_empty_pathway(self, tmp_path):
        expr = _make_expression()
        pathways = {'EMPTY': [], 'GOOD': [f"GENE{i}" for i in range(15)]}
        groups = _make_groups()

        result = qpsp_pro(
            expression=expr,
            pathways=pathways,
            contrast=('case', 'control'),
            groups=groups,
            topology='uniform',
            pathway_test='welch',
            scd='none',
        )
        sig = result.significance
        empty_row = sig[sig['pathway'] == 'EMPTY'].iloc[0]
        assert empty_row['mapping_status'] == 'empty_pathway'

    def test_identity_kernel(self, tmp_path):
        expr = _make_expression()
        pathways = _make_pathways()
        groups = _make_groups()

        result = qpsp_pro(
            expression=expr,
            pathways=pathways,
            contrast=('case', 'control'),
            groups=groups,
            kernel='identity',
            topology='uniform',
            pathway_test='welch',
            scd='none',
        )
        assert result.kernel_weights.notna().any().any()

    def test_invalid_params(self):
        expr = _make_expression()
        pathways = _make_pathways()
        groups = _make_groups()

        with pytest.raises(ValueError, match="tau"):
            qpsp_pro(expr, pathways, ('case', 'control'), groups,
                     tau=2.0, topology='uniform', pathway_test='welch',
                     scd='none')

        with pytest.raises(ValueError, match="contrast"):
            qpsp_pro(expr, pathways, ('case',), groups,
                     topology='uniform', pathway_test='welch', scd='none')

    def test_shared_universe(self, tmp_path):
        expr = _make_expression()
        pathways = _make_pathways()
        groups = _make_groups()

        result = qpsp_pro(
            expression=expr,
            pathways=pathways,
            contrast=('case', 'control'),
            groups=groups,
            ranking_universe='shared',
            ranking_min_coverage=0.8,
            topology='uniform',
            pathway_test='welch',
            scd='none',
        )
        assert isinstance(result, QpspProResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
