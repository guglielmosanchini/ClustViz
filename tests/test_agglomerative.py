from clustviz.agglomerative import (
    update_mat,
    dist_mat_gen,
    dist_mat,
    compute_ward_ij,
    sl_dist,
    avg_dist,
    cl_dist,
    agg_clust,
    point_plot_mod
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test_dist_mat_gen():
    df_for_dist_mat_gen = pd.DataFrame([[0, 0, 1], [0, 2, 0]])

    assert dist_mat_gen(df_for_dist_mat_gen).equals(
        pd.DataFrame([[np.inf, 2], [2, np.inf]])
    )


def test_update_mat_single():
    df_for_update_mat = pd.DataFrame(
        [[np.inf, 1, 4], [1, np.inf, 2], [4, 2, np.inf]],
        columns=["a", "b", "c"],
        index=["a", "b", "c"],
    )

    temp_values = update_mat(df_for_update_mat, 1, 0, "single").values

    assert (temp_values == [[np.inf, 2], [2, np.inf]]).all()


def test_update_mat_average():
    df_for_update_mat = pd.DataFrame(
        [[np.inf, 1, 4], [1, np.inf, 2], [4, 2, np.inf]],
        columns=["a", "b", "c"],
        index=["a", "b", "c"],
    )

    temp_values = update_mat(df_for_update_mat, 1, 0, "average").values
    print(temp_values)

    assert (temp_values == [[np.inf, 3], [3, np.inf]]).all()


def test_update_mat_complete():
    df_for_update_mat = pd.DataFrame(
        [[np.inf, 1, 4], [1, np.inf, 2], [4, 2, np.inf]],
        columns=["a", "b", "c"],
        index=["a", "b", "c"],
    )

    temp_values = update_mat(df_for_update_mat, 1, 0, "complete").values

    assert (temp_values == [[np.inf, 4], [4, np.inf]]).all()


def test_dist_mat_single():
    df_for_dist_mat = pd.DataFrame([[0, 0, 1], [0, 2, 0]])

    assert dist_mat(df_for_dist_mat, "single").equals(
        pd.DataFrame([[np.inf, 2], [np.inf, np.inf]])
    )


def test_dist_mat_avg():
    df_for_dist_mat = pd.DataFrame([[0, 0, 1], [0, 2, 0]])

    assert dist_mat(df_for_dist_mat, "average").equals(
        pd.DataFrame([[np.inf, 2], [np.inf, np.inf]])
    )


def test_dist_mat_complete():
    df_for_dist_mat = pd.DataFrame([[0, 0, 1], [0, 2, 0]])

    assert dist_mat(df_for_dist_mat, "complete").equals(
        pd.DataFrame([[np.inf, 2], [np.inf, np.inf]])
    )


def test_compute_ward_ij():
    X = [[1, 2], [3, 2], [0, 0], [1, 1]]

    b = pd.DataFrame(X, index=["0", "1", "2", "3"], columns=["0x", "0y"])

    assert compute_ward_ij(X, b) == (("0", "3"), 0.5, 0.5)


def test_sl_dist():
    first_cluster = [np.array([3, 1]), np.array([1, 7]), np.array([2, 1])]
    second_cluster = [np.array([1, 1]), np.array([3, 6]), np.array([1, 3])]

    assert sl_dist(first_cluster, second_cluster) == 1


def test_avg_dist():
    first_cluster = [np.array([1, 1]), np.array([2, 1])]
    second_cluster = [np.array([0, 1]), np.array([4, 1])]

    assert avg_dist(first_cluster, second_cluster) == 2


def test_cl_dist():
    first_cluster = [np.array([1, 1]), np.array([2, 1])]
    second_cluster = [np.array([0, 1]), np.array([4, 1])]

    assert cl_dist(first_cluster, second_cluster) == 3


def test_agg_clust_ward():
    X = np.array([[1, 2], [3, 2], [0, 0], [1, 1]])

    agg_clust(X, linkage="ward", plotting=False)


def test_agg_clust_single():
    X = np.array([[1, 2], [3, 2], [0, 0], [1, 1]])

    agg_clust(X, linkage="single", plotting=False)


def test_plot_fn(monkeypatch):
    X = np.array([[1, 2], [3, 2], [0, 0]])
    a = pd.DataFrame(
        [[0.0, 0.0, np.nan, np.nan, np.nan, np.nan], [1.0, 2.0, 3, 2, np.nan, np.nan]],
        index=["2", "(0)-(1)"],
        columns=["0x", "0y", "1x", "1y", "2x", "2y"],
    )

    monkeypatch.setattr(plt, "show", lambda: None)
    point_plot_mod(X, a, 2.57)
