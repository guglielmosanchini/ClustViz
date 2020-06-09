
from algorithms.chameleon.graphtools import knn_graph, pre_part_graph
from algorithms.chameleon.chameleon2 import cluster2, conn_comp
import pandas as pd

def test_cluster2():
    df = pd.DataFrame([[1, 1], [2, 2], [2, 1], [0, 0], [1, 2], [1, 3], [10, 10], [11, 11]])

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