from algorithms.agglomerative import (
    update_mat,
    dist_mat_gen,
    dist_mat,
    compute_ward_ij,
    sl_dist,
    avg_dist,
)

import pandas as pd
import numpy as np


def test_dist_mat_gen():
    df_for_dist_mat_gen = pd.DataFrame([[0, 0, 1], [0, 2, 0]])

    assert dist_mat_gen(df_for_dist_mat_gen).equals(
        pd.DataFrame([[np.inf, 2], [2, np.inf]])
    )


def test_update_mat():
    df_for_update_mat = pd.DataFrame(
        [[np.inf, 1, 4], [1, np.inf, 2], [4, 2, np.inf]],
        columns=["a", "b", "c"],
        index=["a", "b", "c"],
    )

    temp_values = update_mat(df_for_update_mat, 1, 0, "single").values

    assert (temp_values == [[np.inf, 2], [2, np.inf]]).all()


def test_dist_mat():
    df_for_dist_mat = pd.DataFrame([[0, 0, 1], [0, 2, 0]])

    assert dist_mat(df_for_dist_mat, "single").equals(
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
