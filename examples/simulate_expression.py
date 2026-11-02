"""Run four synthetic signal architectures without any study data."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from qpsp_pro import qpsp_pro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--permutations', type=int, default=9999)
    args = parser.parse_args()
    genes = [f'PROTEIN{i:03d}' for i in range(160)]
    pathways = {f'PATHWAY{i}': genes[i * 10:(i + 1) * 10] for i in range(8)}
    participants = [f'{group}{i:02d}' for group in ('A', 'B') for i in range(8)]
    samples = [f'{person}_V{visit}' for person in participants for visit in (1, 2)]
    groups = {sample: sample[0] for sample in samples}
    blocks = {sample: sample.split('_')[0] for sample in samples}
    # Artificial star networks; no catalogue or cohort-derived membership.
    edges = [(members[0], gene, 0.9) for members in pathways.values()
             for gene in members[1:]]
    summaries = []
    with tempfile.TemporaryDirectory() as directory:
        ppi_path = Path(directory) / 'synthetic_ppi.tsv'
        pd.DataFrame(edges, columns=['gene1', 'gene2', 'score']).to_csv(
            ppi_path, sep='\t', index=False)
        for index, architecture in enumerate(
                ('coordinated_up', 'coordinated_down', 'cancellation', 'core_shift')):
            rng = np.random.default_rng(42 + index)
            values = rng.normal(0, 0.7, size=(len(genes), len(samples)))
            for person in participants:
                columns = [i for i, sample in enumerate(samples) if blocks[sample] == person]
                values[:, columns] += rng.normal(0, 0.3, size=(len(genes), 1))
            for members in pathways.values():
                positions = [genes.index(gene) for gene in members]
                values[positions] += rng.normal(0, 0.3, size=len(samples))
            case_columns = [i for i, sample in enumerate(samples) if groups[sample] == 'A']
            for members in list(pathways.values())[:2]:
                for position, gene in enumerate(members):
                    if architecture == 'coordinated_up':
                        shift = 0.8
                    elif architecture == 'coordinated_down':
                        shift = -0.8
                    elif architecture == 'cancellation':
                        shift = 0.9 if position < 5 else -0.9
                    else:
                        shift = 1.2 if position < 3 else 0.0
                    values[genes.index(gene), case_columns] += shift
            # Missingness is confined to synthetic background proteins.
            background = values[80:]
            background[rng.random(background.shape) < 0.02] = np.nan
            result = qpsp_pro(
                expression=pd.DataFrame(values, index=genes, columns=samples),
                pathways=pathways, groups=groups, contrast=('A', 'B'),
                ppi_path=str(ppi_path), topology='weighted_degree',
                design='clustered', blocks=blocks,
                ranking_universe='analysis_complete_case', kernel='logistic',
                score_denominator='kernel_weighted',
                n_permutations=args.permutations, seed=42 + index,
            )
            statistics = result.significance.set_index('pathway')
            true_statistics = statistics.loc[['PATHWAY0', 'PATHWAY1']]
            ranked = statistics.hedges_g.abs().nlargest(2).index
            summaries.append({
                'architecture': architecture,
                'true_in_top2': sum(name in {'PATHWAY0', 'PATHWAY1'} for name in ranked),
                'median_true_abs_g': true_statistics.hedges_g.abs().median(),
                'median_true_net_shift': true_statistics.net_shift.median(),
                'median_true_discordance_fraction': true_statistics.discordance_fraction.median(),
            })
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
