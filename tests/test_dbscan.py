from code.algorithms.dbscan import scan_neigh1_mod, DBSCAN
import numpy as np

def test_scan_neigh1_mod():
    data = [np.array([0, 1]), np.array([0, 2]), np.array([0, 4]), np.array([0, 5])]
    point = "1"
    eps = 1
    X_dict = dict(zip([str(i) for i in range(len(data))], data))
    scan_neigh1_mod(X_dict, X_dict[point], eps)
    res = list(scan_neigh1_mod(X_dict, X_dict[point], eps).values())
    exp = list({'0': np.array([0, 1]), '1': np.array([0, 2])}.values())

    assert all([np.allclose(x, y) for x, y in zip(res, exp)])

def test_DBSCAN():
    data = [np.array([0, 1]), np.array([0, 2]), np.array([0, 4]), np.array([0, 5])]
    eps = 1
    minPTS = 1

    assert DBSCAN(data, eps, minPTS) == {'0': 0, '1': 0, '2': 1, '3': 1}