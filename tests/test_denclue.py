from algorithms.denclue.denclue import gauss_dens, grad_gauss_dens, square_wave_dens, FindRect, pop_cubes, \
    highly_pop_cubes, check_connection, connect_cubes, near_with_cube, near_without_cube, density_attractor, \
    assign_cluster, DENCLUE

import numpy as np
from sklearn.datasets import make_blobs


def test_gauss_dens():
    D = [[1, 0], [0, 0], [2, 0]]
    res = gauss_dens([0, 0], D, 1, dist="euclidean")
    assert round(res, 2) == 1.74


def test_grad_gauss_dens():
    D = [[1, 0], [0, 2]]
    res = grad_gauss_dens([0, 0], D, 1, dist="euclidean")
    first_component = round(res[0], 2) == 0.61
    second_component = round(res[1], 2) == 0.27

    assert first_component & second_component


def test_square_wave_dens():
    D = [[1, 0], [0, 0], [2, 0]]
    res = square_wave_dens([0, 0], D, 1, dist="euclidean")
    assert res == 2


def test_pop_cubes():
    D = np.array([[0, 0], [0, 1], [0, 2]])

    res = pop_cubes(D, 1)

    condition0 = res[0] == {(0, 0): [2, [0, 1], [[0, 0], [0, 1]]], (0, 1): [1, [0, 2], [[0, 2]]]}
    condition1 = res[1] == {(0, 0): (-0.05, -0.05, 1.95, 1.95), (0, 1): (-0.05, 1.95, 1.95, 3.95)}

    assert condition0 & condition1


def test_FindRect():
    coord_dict = {(0, 0): (-0.05, -0.05, 1.95, 1.95), (0, 1): (-0.05, 1.95, 1.95, 3.95)}
    point = [0, 2]
    assert FindRect(point, coord_dict) == (0, 1)


def test_highly_pop_cubes():
    z = {(0, 0): [3, [1, 2], [[0, 0], [0, 1], [1, 1]]], (0, 1): [1, [0, 2], [[0, 2]]]}
    assert highly_pop_cubes(z, 2) == {(0, 0): [3, [1, 2], [[0, 0], [0, 1], [1, 1]]]}


def test_check_connection():
    hpc = {(0, 0): [2, [0, 1], [[0, 0], [0, 1]]], (0, 1): [1, [0, 2], [[0, 2]]]}

    assert check_connection(hpc[(0, 0)], hpc[0, 1], 1)


def test_connect_cubes():
    z = {(0, 0): [4, [2.1, 3.0], [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.1, 1.0]]],
         (0, 1): [3, [2.0, 7.0], [[0.0, 2.0], [1.0, 2.0], [1.0, 3.0]]],
         (2, 2): [2, [8.0, 9.0], [[4.0, 5.0], [4.0, 4.0]]], (3, 2): [2, [13.0, 8.0], [[6.0, 4.0], [7.0, 4.0]]]}

    hpc = {(0, 0): [4, [2.1, 3.0], [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.1, 1.0]]]}

    new_cubes = connect_cubes(hpc, z, s=1)

    assert new_cubes == {(0, 0): [4, [2.1, 3.0], [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.1, 1.0]]],
                         (0, 1): [3, [2.0, 7.0], [[0.0, 2.0], [1.0, 2.0], [1.0, 3.0]]]}


def test_near_with_cube():
    z = {(0, 0): [4, [2.1, 3.0], [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.1, 1.0]]],
         (0, 1): [3, [2.0, 7.0], [[0.0, 2.0], [1.0, 2.0], [1.0, 3.0]]],
         (2, 2): [2, [8.0, 9.0], [[4.0, 5.0], [4.0, 4.0]]], (3, 2): [2, [13.0, 8.0], [[6.0, 4.0], [7.0, 4.0]]]}

    res = near_with_cube(np.array([0.2, 0.2]), z[(0, 0)], z, 1)

    assert res == [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.1, 1.0], [0.0, 2.0], [1.0, 2.0], [1.0, 3.0]]


def test_near_without_cube():
    z = {(0, 0): [4, [2.1, 3.0], [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.1, 1.0]]],
         (0, 1): [3, [2.0, 7.0], [[0.0, 2.0], [1.0, 2.0], [1.0, 3.0]]],
         (2, 2): [2, [8.0, 9.0], [[4.0, 5.0], [4.0, 4.0]]], (3, 2): [2, [13.0, 8.0], [[6.0, 4.0], [7.0, 4.0]]]}
    d = {(0, 0): (-0.05, -0.05, 1.95, 1.95), (1, 0): (1.95, -0.05, 3.95, 1.95),
         (2, 0): (3.95, -0.05, 5.95, 1.95), (3, 0): (5.95, -0.05, 7.95, 1.95),
         (0, 1): (-0.05, 1.95, 1.95, 3.95), (1, 1): (1.95, 1.95, 3.95, 3.95),
         (2, 1): (3.95, 1.95, 5.95, 3.95), (3, 1): (5.95, 1.95, 7.95, 3.95),
         (0, 2): (-0.05, 3.95, 1.95, 5.95), (1, 2): (1.95, 3.95, 3.95, 5.95),
         (2, 2): (3.95, 3.95, 5.95, 5.95), (3, 2): (5.95, 3.95, 7.95, 5.95)}

    res = near_without_cube(np.array([0.2, 0.2]), d, z, 1)

    assert res == [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.1, 1.0], [0.0, 2.0], [1.0, 2.0], [1.0, 3.0]]


def test_density_attractor():
    D = np.array([[0, 0], [0, 1],
                  [0, 2], [1, 1],
                  [1.1, 1], [1, 2],
                  [1, 3], [4, 5],
                  [4, 4], [6, 4], [7, 4]
                  ])
    z, d = pop_cubes(data=D, s=1)

    r, o = density_attractor(data=D, x=[0.0, 0.0], coord_dict=d, tot_cubes=z,
                             s=1, xi=2, delta=0.02, max_iter=600, dist="euclidean")

    condition0 = (o == [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    condition1 = (round(r[0][0], 2) == 0.62)
    condition2 = (round(r[0][1], 2) == 1.34)

    assert condition0 & condition1 & condition2

def test_assign_cluster():
    D = np.array([[0, 0], [0, 1],
                  [0, 2], [1, 1],
                  [1.1, 1], [1, 2],
                  [1, 3], [4, 5],
                  [4, 4], [6, 4], [7, 4]
                  ])

    clust_dict, proc = assign_cluster(data=D, others=[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                                      attractor=(np.array([0.62154195, 1.34237831]), True),
                                      clust_dict={}, processed=[])

    condition0 = (round(clust_dict[0][0], 2) == 0.62)
    condition1 = (round(clust_dict[0][1], 2) == 1.34)
    condition0bis = (round(clust_dict[1][0], 2) == 0.62)
    condition1bis = (round(clust_dict[1][1], 2) == 1.34)
    condition2 = proc == [0, 1]

    assert condition0 & condition1 & condition2 & condition0bis & condition1bis


def test_DENCLUE():

    D = make_blobs(15, random_state=42)[0]

    assert DENCLUE(D, s=1.5, plotting=False, xi=2) == [1, 1, -1, 0, -1, 0, -1, 1, 0, 1, -1, 0, -1, 0, 1]
