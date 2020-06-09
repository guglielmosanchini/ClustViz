from code.algorithms.clarans import compute_cost_clarans

import pandas as pd


def test_compute_cost_clarans():

    X = pd.DataFrame([[1, 1], [6, 5], [6, 6]])

    assert compute_cost_clarans(X, [0, 2]) == (1.0, {0: [0], 2: [1, 2]})