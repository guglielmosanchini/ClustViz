from code.algorithms.cure import dist_clust_cure, update_mat_cure, sel_rep_fast, sel_rep, Chernoff_Bounds, dist_mat_gen_cure, \
    cure_sample_part, cure
from code.algorithms.agglomerative import dist_mat_gen

import numpy as np
import pandas as pd


def test_dist_clust_cure():
    first_cluster = [np.array([3, 1]), np.array([1, 7]), np.array([2, 1])]
    second_cluster = [np.array([1, 1]), np.array([3, 6]), np.array([1, 3])]

    assert dist_clust_cure(first_cluster, second_cluster) == 1


def test_update_mat_cure():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2]])

    b = pd.DataFrame(X, index=["0", "1", "2"], columns=["0x", "0y"])

    dist_mat = dist_mat_gen(b)

    rep = {str(i): [X[i]] for i in range(len(X))}
    rep.update({'(0)-(1)': [np.array([0, 0.05]), np.array([0, 0.95])]})

    res = update_mat_cure(dist_mat, "0", "1", rep, "(0)-(1)")

    expected_df = pd.DataFrame([[np.inf, 1.05], [1.05, np.inf]], index=["2", "(0)-(1)"], columns=["2", "(0)-(1)"])

    assert res.equals(expected_df)


def test_sel_rep():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2]])

    rep = {str(i): [X[i]] for i in range(len(X))}
    rep.update({'(0)-(1)': [np.array([0, 0.05]), np.array([0, 0.95])]})

    clusters = {str(i): np.array(X[i]) for i in range(len(X))}
    w = [clusters["0"], clusters["1"]]
    del clusters["0"]
    del clusters["1"]
    clusters["(0)-(1)"] = w

    res = sel_rep(clusters, "(0)-(1)", 1, 0.5)

    assert (res[0] == np.array([0., 0.25])).all()


def test_sel_rep_fast():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2]])

    rep = {str(i): [X[i]] for i in range(len(X))}
    rep.update({'(0)-(1)': [np.array([0, 0.05]), np.array([0, 0.95])]})

    clusters = {str(i): np.array(X[i]) for i in range(len(X))}
    w = [clusters["0"], clusters["1"]]
    del clusters["0"]
    del clusters["1"]
    clusters["(0)-(1)"] = w

    res = sel_rep_fast(rep["0"] + rep["1"], clusters, "(0)-(1)", 1, 0.5)

    assert (res[0] == np.array([0., 0.25])).all()


def test_Chernoff_Bounds():
    u_min = 1
    f = 0.01
    N = 100
    d = 0.9
    k = 3

    assert round(Chernoff_Bounds(u_min, f, N, d, k), 2) == 23.03


def test_dist_mat_gen_cure():
    expected_df = pd.DataFrame([[np.inf, 0, 1], [0, np.inf, 2], [1, 2, np.inf]],
                               index=["0", "1", "2"], columns=["0", "1", "2"])

    res_df = dist_mat_gen_cure({"0": np.array([1, 2]), "1": np.array([2, 3]), "2": np.array([0, 0])})

    assert expected_df.equals(res_df)


def test_cure_sample_part_1():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 2],
                  [1, 3]])

    res = cure_sample_part(X, 2, plotting=False)

    exp = {'(1)-(4)': [np.array([0, 1]), np.array([1, 3])], '(2)-(0)': [np.array([0, 2]), np.array([0, 0])]}
    l1 = list(res[0].values())
    l1 = [item for sublist in l1 for item in sublist]
    l2 = list(exp.values())
    l2 = [item for sublist in l2 for item in sublist]

    assert all([np.allclose(x, y) for x, y in zip(l1, l2)])


def test_cure_sample_part_2():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 2],
                  [1, 3]])

    res = cure_sample_part(X, 2, plotting=False)
    exp2 = {'(1)-(4)': [np.array([0.15, 1.3]), np.array([0.85, 2.7])],
            '(2)-(0)': [np.array([0., 1.7]), np.array([0., 0.3])]}
    l1 = list(res[1].values())
    l1 = [item for sublist in l1 for item in sublist]
    l2 = list(exp2.values())
    l2 = [item for sublist in l2 for item in sublist]

    assert all([np.allclose(x, y) for x, y in zip(l1, l2)])


def test_cure_sample_part_3():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 2],
                  [1, 3]])

    res = cure_sample_part(X, 2, plotting=False)

    exp_df = pd.DataFrame([[0.0, 1.0, 1, 3], [0.0, 2.0, 0, 0]],
                          index=["(1)-(4)", "(2)-(0)"], columns=["0x", "0y", "1x", "1y"])
    exp_df = exp_df.astype({"1x": "object", "1y": "object"})

    assert res[2].equals(exp_df)


def test_cure_1():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 2],
                  [1, 3]])

    res = cure(X, 2, plotting=False)

    exp = {'(0)-(1)': [np.array([0, 0]), np.array([0, 1])],
           '(4)-((2)-(3))': [np.array([0, 2]), np.array([1, 2]), np.array([1, 3])]}
    l1 = list(res[0].values())
    l1 = [item for sublist in l1 for item in sublist]
    l2 = list(exp.values())
    l2 = [item for sublist in l2 for item in sublist]

    assert all([np.allclose(x, y) for x, y in zip(l1, l2)])


def test_cure_2():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 2],
                  [1, 3]])

    res = cure(X, 2, plotting=False)

    exp2 = {'(0)-(1)': [np.array([0., 0.05]), np.array([0., 0.95])],
            '(4)-((2)-(3))': [np.array([0.96666667, 2.93333333]), np.array([0.11166667, 2.03333333]),
                              np.array([0.92166667, 2.03333333])]}
    l1 = list(res[1].values())
    l1 = [item for sublist in l1 for item in sublist]
    l2 = list(exp2.values())
    l2 = [item for sublist in l2 for item in sublist]

    assert all([np.allclose(x, y) for x, y in zip(l1, l2)])


def test_cure_3():
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 2],
                  [1, 3]])

    res = cure(X, 2, plotting=False)

    exp_df = pd.DataFrame([[0.0, 0.0, 0, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                           [1.0, 3.0, 0, 2, 1, 2, np.nan, np.nan, np.nan, np.nan]],
                          index=["(0)-(1)", "(4)-((2)-(3))"],
                          columns=["0x", "0y", "1x", "1y", "2x", "2y", "3x", "3y", "4x", "4y"])
    exp_df = exp_df.astype({"1x": "object", "1y": "object", "2x": "object", "2y": "object", "3x": "object",
                            "3y": "object", "4x": "object", "4y": "object"})

    assert res[2].equals(exp_df)
