import pytest
import numpy as np
import m2.datasets as d
import m2.povel1985 as pe
from m2.datasets.fitch2007 import fitch_atime_dataset

povel1985 = {
    'examples': [
        d.seq_to_beats(1, seq)
        for seq in [  # Examples with clustering from Povel 1985
                    [1, 1, 2, 3, 1, 3, 1, 4],
                    [2, 1, 1, 1, 3, 3, 1, 4],
                    [1, 2, 1, 1, 3, 3, 1, 4],
                    [1, 2, 1, 1, 3, 1, 3, 4],
                    [1, 1, 1, 1, 2, 3, 3, 4],
                    [3, 1, 1, 2, 3, 1, 1, 4],
                    [1, 1, 1, 2, 1, 3, 3, 4],
                    [1, 1, 1, 3, 1, 2, 3, 4]
                    ]
    ],
    'clusters': [  # Clusters is the number of onsets per cluster
        [3, 1, 2, 2],
        [1, 4, 1, 2],
        [2, 3, 1, 2],
        [2, 3, 2, 1],
        [5, 1, 1, 1],
        [1, 3, 1, 3],
        [4, 2, 1, 1],
        [4, 2, 1, 1]
    ],
    'accents': [  # Accented onsets of the first 3 examples
        [0, 2, 4, 8, 12],
        [0, 2, 5, 8, 12],
        [1, 3, 5, 8, 12]
    ],
    'best_clocks': [  # ((phase, period), counterevidence)
        ((0, 4), 0),
        ((0, 4), 1),
        ((0, 4), 2),
        ((0, 4), 3),
        ((0, 4), 4),
        ((0, 4), 5),
        ((1, 4), 6),
        ((0, 4), 8)
    ]
}


fitch2007 = {
    'examples': [onsets for _, _, onsets
                 in sorted(fitch_atime_dataset(3), key=lambda x: x[0])],
    'clusters': [
        [1, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 3, 1, 1, 3, 1, 1, 2],
        [3, 5, 5, 2],
        [1, 3, 1, 1, 3, 1, 1, 3, 1],
        [1, 4, 1, 4, 1, 4],
        [1, 2, 2, 1, 2, 2, 1, 2, 2],
        [1, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
        [1, 1, 1, 3, 1, 1, 3, 1, 1, 2],
        [1, 1, 1, 2, 1, 1, 2, 1, 1, 1]
    ]
}


@pytest.mark.parametrize('examples, cluster_onsets', [
    (povel1985['examples'], povel1985['clusters']),
    (fitch2007['examples'], fitch2007['clusters'])
])
def test_cluster_info(examples, cluster_onsets):
    assert len(povel1985['examples']) == len(povel1985['clusters'])

    for ex, c in zip(povel1985['examples'], povel1985['clusters']):
        assert np.sum(c) == len(ex)


@pytest.mark.parametrize('ibi', [150, 200, 250])
@pytest.mark.parametrize('dataset', [povel1985, fitch2007])
def test_clustering_function_povel(ibi, dataset):
    for ex, expected_clusters in zip(dataset['examples'], dataset['clusters']):
        onsets = np.array(ex) * ibi
        cs = pe.cluster_onsets(onsets)
        assert expected_clusters == cs


def test_accents():
    for onsets, expected_accents in zip(povel1985['examples'],
                                        povel1985['accents']):
        accents = pe.accented_onsets(onsets)
        assert accents == expected_accents


def test_best_hypothesis():
    for ex, (ex_best_clock, ex_cv) in zip(povel1985['examples'],
                                          povel1985['best_clocks']):
        best_clock, cv = pe.best_clock(ex, 1, 16)
        assert cv == ex_cv
        assert best_clock == ex_best_clock
