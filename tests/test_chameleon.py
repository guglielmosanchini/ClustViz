from clustviz.chameleon.graphtools import (
    knn_graph,
    pre_part_graph,
    get_cluster,
    connecting_edges,
)
from clustviz.chameleon.chameleon import (
    merge_best,
    cluster,
    rebuild_labels,
    len_edges,
)

from sklearn.datasets import make_blobs

import pandas as pd
import numpy as np

import pytest
import sys


def test_knn_graph():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2]])
    k = 2

    graph = knn_graph(df, k, symmetrical=False, verbose=False)

    condition0 = list(graph.edges) == [
        (0, 4),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 4),
        (3, 4),
    ]
    condition1 = list(graph.nodes) == [0, 1, 2, 3, 4]

    assert condition0 & condition1


def test_knn_graph_sym():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2]])
    k = 2

    graph = knn_graph(df, k, symmetrical=True, verbose=False)

    condition0 = list(graph.edges) == [(0, 4), (0, 3), (1, 2), (3, 4)]
    condition1 = list(graph.nodes) == [0, 1, 2, 3, 4]

    assert condition0 & condition1


def test_pre_part_graph():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2], [5, 5], [7, 2]])
    k = 3

    pregraph = knn_graph(df, k, symmetrical=False, verbose=False)
    _ = pre_part_graph(pregraph, 2, df, verbose=False)
    print(df["cluster"].values)
    assert (df["cluster"].values == np.array([0, 1, 1, 0, 0, 1, 1])).all()


@pytest.mark.skipif("win" in sys.platform, reason="provides ci==3, not 5")
def test_merge_best():
    df = pd.DataFrame(make_blobs(60, random_state=42)[0])
    knn = 6

    pregraph = knn_graph(df, knn, symmetrical=False, verbose=False)
    graph = pre_part_graph(pregraph, 10, df, verbose=False)
    df, max_score, ci = merge_best(
        graph=graph, df=df, alpha=2, k=3, verbose=False, verbose2=False
    )

    condition0 = round(max_score) == 1
    condition1 = ci == 5

    assert condition0 & condition1


@pytest.mark.skipif(
    sys.platform == "linux", reason="provides slightly different result"
)
def test_cluster():
    df = pd.DataFrame(make_blobs(50, random_state=42)[0])

    res, dendr_height = cluster(
        df,
        3,
        knn=4,
        m=10,
        alpha=2.0,
        plot=False,
        verbose0=False,
        verbose1=False,
        verbose2=False,
    )

    condition0 = sorted(list(res["cluster"].values)) == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3,
                                                         3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                         4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6,
                                                         6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9]

    condition1 = round(dendr_height[9], 1) == 0.8

    assert condition0 & condition1


def test_rebuild_labels():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2], [5, 5], [7, 2]])

    pregraph = knn_graph(df, 3, symmetrical=False, verbose=False)
    _ = pre_part_graph(pregraph, 10, df, verbose=False)

    df_bis = rebuild_labels(df)

    assert sorted(list(df_bis["cluster"])) == [1, 1, 1, 2, 2, 2, 2]


def test_get_cluster():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2], [5, 5], [7, 2]])
    k = 3

    pregraph = knn_graph(df, 3, symmetrical=False, verbose=False)
    graph = pre_part_graph(pregraph, k, df, verbose=False)

    condition0 = get_cluster(graph, 0) == [0, 3, 4]
    condition1 = get_cluster(graph, k - 1) == [1, 2, 5, 6]

    assert condition0 & condition1


def test_len_edges():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2], [5, 5], [7, 2]])

    pregraph = knn_graph(df, 3, symmetrical=False, verbose=False)
    graph = pre_part_graph(pregraph, 10, df, verbose=False)

    condition0 = len_edges(graph, [0, 3, 4]) == 3
    condition1 = len_edges(graph, [1, 2, 5, 6]) == 6

    assert condition0 & condition1


def test_connecting_edges():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2], [5, 5], [7, 2]])

    pregraph = knn_graph(df, 3, symmetrical=False, verbose=False)
    graph = pre_part_graph(pregraph, 10, df, verbose=False)

    assert connecting_edges(([0, 3, 4], [1, 2, 5, 6]), graph) == [
        (0, 5),
        (3, 5),
        (4, 5),
    ]
