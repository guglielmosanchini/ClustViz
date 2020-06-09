from algorithms.optics import (
    scan_neigh1,
    scan_neigh2,
    minPTSdist,
    OPTICS,
    ExtractDBSCANclust,
)

import numpy as np


def test_scan_neigh1():
    data = {
        0: np.array([1, 1]),
        1: np.array([2, 1]),
        2: np.array([1, 4]),
        3: np.array([1, 1.5]),
        4: np.array([2, 1.5]),
    }

    res_dict = scan_neigh1(data, np.array([2, 1]), 0.5)

    assert list(res_dict.keys()) == ["4"]


def test_scan_neigh2():
    data = {
        0: np.array([1, 1]),
        1: np.array([2, 1]),
        2: np.array([1, 4]),
        3: np.array([1, 1.5]),
        4: np.array([2, 1.5]),
    }

    assert scan_neigh2(data, np.array([2, 1]), 0.5) == [("4", 0.5)]


def test_minPTSdist():
    data = {
        0: np.array([1, 1]),
        1: np.array([2, 1]),
        2: np.array([1, 4]),
        3: np.array([1, 1.5]),
        4: np.array([2, 1.5]),
    }

    assert minPTSdist(data, 0, 8, 0.5) == np.inf


def test_ExtractDBSCANclust():
    D = np.array([[0, 0], [0, 1], [1, 0], [3, 3], [3, 4], [3, 3.5]])

    res = OPTICS(D, 1.5, 2, plot=False, plot_reach=False)

    assert sorted(ExtractDBSCANclust(res[0], res[1], 1).values()) == [
        0,
        0,
        0,
        1,
        1,
        1,
    ]


def test_OPTICS():
    D = np.array(
        [
            [0, 0],
            [0, 1],
            [3, 3],
            [0, 2],
            [1, 1],
            [3, 5],
            [3, 4],
            [1, 2],
            [1, 3],
        ]
    )

    expected0 = {
        "5": np.inf,
        "6": 1.0,
        "2": 1.0,
        "8": 2.0,
        "7": 1.0,
        "3": 1.0,
        "4": 1.0,
        "1": 1.0,
        "0": 1.0,
    }
    expected1 = {
        "5": 1.0,
        "6": 1.0,
        "2": 1.0,
        "8": 1.0,
        "7": 1.0,
        "3": 1.0,
        "4": 1.0,
        "1": 1.0,
        "0": 1.0,
    }

    res = OPTICS(D, 4, 2, plot=False, plot_reach=False)

    condition0 = sorted(list(expected0.values())) == sorted(
        list(res[0].values())
    )
    condition1 = sorted(list(expected1.values())) == sorted(
        list(res[1].values())
    )

    assert condition1 & condition0
