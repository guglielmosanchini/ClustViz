from clustviz.chameleon.graphtools import (
    knn_graph,
    pre_part_graph,
    plot2d_graph,
    plot2d_data,
)
from clustviz.chameleon.chameleon2 import cluster2, conn_comp
from clustviz.chameleon.chameleon import merge_best

import pandas as pd
from sklearn.datasets import make_blobs


def test_cluster2():
    df = pd.DataFrame(
        [[1, 1], [2, 2], [2, 1], [0, 0], [1, 2], [1, 3], [10, 10], [11, 11]]
    )

    res = cluster2(df, plot=False)
    condition0 = sorted(list(res[0]["cluster"])) == [1, 1, 1, 1, 1, 1, 2, 2]
    condition1 = res[1] == {}

    assert condition0 & condition1


def test_conn_comp():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2], [2, 1], [5, 5], [7, 6]])

    pregraph = knn_graph(df, 3)
    graph = pre_part_graph(pregraph, 10, df)

    res = conn_comp(graph)
    condition0 = sorted(res[0]) == [0, 3, 4, 5]
    condition1 = sorted(res[1]) == [1, 2, 6, 7]

    assert condition0 & condition1


def test_plot2d_graph():
    df = pd.DataFrame([[1, 1], [6, 5], [6, 6], [0, 0], [1, 2], [2, 1], [5, 5], [7, 6]])

    pregraph = knn_graph(df, 3)
    graph = pre_part_graph(pregraph, 10, df)
    plot2d_graph(graph)


def test_plot2d_data():
    df = pd.DataFrame(make_blobs(60, random_state=42)[0])
    knn = 6

    pregraph = knn_graph(df, knn, verbose=False)
    graph = pre_part_graph(pregraph, 10, df, verbose=False)
    df, max_score, ci = merge_best(
        graph=graph, df=df, a=2, k=3, verbose=False, verbose2=False
    )

    plot2d_data(df)
