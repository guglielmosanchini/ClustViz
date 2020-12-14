from typing import Tuple, Dict, Union, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import ceil
# hidden import for 3d plot, dont delete it
from mpl_toolkits import mplot3d
from tqdm.auto import tqdm
from itertools import groupby
from collections import OrderedDict, Counter

from clustviz.utils import euclidean_distance, flatten_list, COLOR_DICT, FONTSIZE_NORMAL, SIZE_NORMAL, annotate_points

CubeInfo = Dict[str, Union[int, List[float], List[list]]]
"""
Info of a cube (rectangle), it is made up of the number of points contained in the cube,
the linear sum of the x and y coordinated of the points in the cube, and of the coordinates
of the points themselves. For example
``{'num_points': 2, 'linear_sum': np.array([3, 4]), 'points_coords': np.array([[1, 1], [2, 3])}``.
"""
Cubes = Dict[Tuple[int, int], CubeInfo]
"""
Cubes (rectangles), having as keys the ``(row, column)`` tuple, and as values ``CubeInfo``.
"""
CubesCoords = Dict[Tuple[int, int], Tuple[float, float, float, float]]
"""
Coordinates of the cubes (rectangles), having as keys the (row, column) tuple, and as values
the minimum x, the minimum y, the maximum x and maximum y of the cube.
For example
``{ (0, 0): (-0.05, -0.05, 1.95, 1.95), (1, 0): (1.95, -0.05, 3.95, 1.95), ... }``.
"""


def gaussian_influence(x: np.ndarray, y: np.ndarray, s: float, dist: str = "euclidean") -> float:
    """
    Return the value of the Gaussian influence function in ``(x,y)`` with standard deviation ``s``.

    :param x: first point.
    :param y: second point.
    :param s: standard deviation of the Gaussian.
    :param dist: distance to use in the Gaussian.
    :return: value for the Gaussian in ``(x,y)`` with standard deviation ``s``.
    """

    if dist == "euclidean":
        return np.exp(
            -(np.power(euclidean_distance(x, y), 2) / (2 * (s ** 2)))
        )


def gaussian_density(x: np.ndarray, D: np.ndarray, s: float, dist: str = "euclidean") -> float:
    """
    Compute the Gaussian density of a point with respect to a dataset.

    :param x: point whose density is to be computed.
    :param D: dataset.
    :param s: standard deviation of the Gaussian.
    :param dist: distance to use in the Gaussian.
    :return: Gaussian density at point ``x`` with respect to dataset ``D``, using a Gaussian function with
             distance dist and standard deviation ``s``.
    """

    res = 0
    for i in range(len(D)):
        res += gaussian_influence(x, D[i], s, dist)

    return res


def gradient_gaussian_density(x: np.ndarray, D: np.ndarray, s: float, dist: str = "euclidean") -> np.ndarray:
    """
    Compute the gradient of the Gaussian density function, used to find the density-attractors, at a point.

    :param x: point.
    :param D: dataset.
    :param s: standard deviation of the Gaussian.
    :param dist: distance to use in the Gaussian.
    :return: gradient of the Gaussian density at point ``x`` with respect to dataset ``D``, using a Gaussian function with
             distance ``dist`` and standard deviation ``s``.
    """
    res = np.zeros(2)
    for i in range(len(D)):
        res += gaussian_influence(x, D[i], s, dist) * (
                np.array(D[i]) - np.array(x)
        )
    return res


def square_wave_influence(x: np.ndarray, y: np.ndarray, s: float, dist: str = "euclidean") -> int:
    """
    Compute the square-wave influence function in ``(x,y)`` with standard deviation ``s``.

    :param x: first point.
    :param y: second point.
    :param s: cut-off.
    :param dist: distance to use.
    :return: ``if dist(x, y) <= s, return 1, else 0.``
    """
    if dist == "euclidean":
        if euclidean_distance(x, y) <= s:
            return 1
        else:
            return 0


def square_wave_density(x: np.ndarray, D: np.ndarray, s: float, dist: str = "euclidean") -> int:
    """
    Compute the square-wave density of a point with respect to a dataset.

    :param x: point whose density is to be computed.
    :param D: dataset.
    :param s: cut-off.
    :param dist: distance to use.
    :return: square-wave density at point ``x`` with respect to dataset ``D``, using a square-wave function with
             distance ``dist`` and cut-off ``s``.
    """

    res = 0
    for i in range(len(D)):
        res += square_wave_influence(x, D[i], s, dist)
    return res


def square_wave_gradient(x: np.ndarray, D: np.ndarray, s: float, dist: str = "euclidean") -> np.ndarray:
    """
    Compute the gradient of the square-wave density function of a point with respect to a dataset.

    :param x: point whose density is to be computed.
    :param D: dataset.
    :param s: cut-off.
    :param dist: distance to use.
    :return: gradient of the square-wave density function at point ``x`` with respect to dataset ``D``, using a
             square-wave function with distance ``dist`` and cut-off ``s``.
    """

    res = np.zeros(2)
    for i in range(len(D)):
        res += square_wave_influence(x, D[i], s, dist) * (
                np.array(D[i]) - np.array(x)
        )
    return res


def FindPoint(x1: float, y1: float, x2: float, y2: float, x: float, y: float) -> bool:
    """
    Check if the point ``(x,y)`` is inside the rectangle determined by ``x1, y1, x2, y2``.

    :param x1: minimum ``x`` coordinate of the rectangle vertices.
    :param y1: minimum ``y`` coordinate of the rectangle vertices.
    :param x2: maximum ``x`` coordinate of the rectangle vertices.
    :param y2: maximum ``y`` coordinate of the rectangle vertices.
    :param x: ``x`` coordinate of the point to be examined.
    :param y: ``y`` coordinate of the point to be examined.
    :return: ``True`` if the point ``(x, y)`` lies inside the rectangle, ``False`` otherwise.
    """
    if (x1 < x < x2) and (y1 < y < y2):
        return True
    else:
        return False


def FindRect(point: np.ndarray, coord_dict: CubesCoords) -> Optional[Tuple[int, int]]:
    """
    Find the key of the cube (rectangle) containing the point (if any).

    :param point: point whose cube (rectangle) is to be found.
    :param coord_dict: dictionary of the rectangles' coordinates.
    :return: key of the cube, e.g. ``(1, 3)``, containing the point; if it does not exist, return ``None``.
    """
    for k, v in coord_dict.items():
        if FindPoint(v[0], v[1], v[2], v[3], point[0], point[1]) is True:
            return k
    return None


def form_populated_cubes(a: float, b: float, c: float, d: float, data: np.ndarray) -> CubeInfo:
    """
    For the given input cube (rectangle), compute how many points of the input dataset lie in it, store their
    coordinates and compute the sum of their ``x`` and ``y`` coordinates.

    :param a: minimum ``x`` coordinate of the rectangle.
    :param b: minimum ``y`` coordinate of the rectangle.
    :param c: maximum ``x`` coordinate of the rectangle.
    :param d: maximum ``y`` coordinate of the rectangle.
    :param data: input dataset.
    :return: dictionary of number of points lying in the cube, the linear sum of their ``x`` and ``y`` coordinates,
             their coordinates.
    """

    num_points = 0
    linear_sum = np.zeros(2)
    points_coords = []
    cube_info_keys = ["num_points", "linear_sum", "points_coords"]

    # for each point of the dataset
    for el in data:
        # if the point lies in the cube
        if FindPoint(a, b, c, d, el[0], el[1]) is True:
            num_points += 1
            linear_sum[0] += el[0]
            linear_sum[1] += el[1]
            points_coords.append([el[0], el[1]])

    cube_info_values = [num_points, linear_sum, np.array(points_coords)]
    cube_info = dict(zip(cube_info_keys, cube_info_values))

    return cube_info


def plot_min_bound_rect(data: np.ndarray) -> tuple:
    """
    Plot the minimal bounding rectangle of the input dataset.

    :param data: input dataset.
    """

    fig, ax = plt.subplots(figsize=(22, 15))
    # draw a scatter plot of the whole dataset
    ax.scatter(data[:, 0], data[:, 1], s=100, edgecolor="black")

    # find the minimum x and y coordinates of the dataset, and the rectangle width
    rect_min = data.min(axis=0)
    rect_diff = data.max(axis=0) - rect_min

    x0 = rect_min[0] - 0.05
    y0 = rect_min[1] - 0.05

    # minimal bounding rectangle
    ax.add_patch(
        Rectangle(
            (x0, y0),
            rect_diff[0] + 0.1,
            rect_diff[1] + 0.1,
            fill=None,
            color="r",
            alpha=1,
            linewidth=3,
        )
    )
    # plt.show()
    return fig, ax


def pop_cubes(data: np.ndarray, s: float) -> Tuple[Cubes, CubesCoords]:
    """
    Find the populated cubes (rectangles containing at least one point).

    :param data: input dataset.
    :param s: ``sigma``, determines the influence of a point in its neighborhood.
    :return: the ``(x,y)`` coordinates of the populated cubes, with how many points it contains, the coordinates of
             its center of mass, and the coordinates of the points belonging to it; the coordinates of the cube
             (rectangle) itself.
    """

    populated_cubes = {}
    corresp_key_coord = {}

    rect_min = data.min(axis=0)
    rect_diff = data.max(axis=0) - rect_min
    x0 = rect_min[0] - 0.05
    y0 = rect_min[1] - 0.05
    # find the number of rows and columns of the rectangles grid; each of the rectangles is 2*s wide
    num_width = int(ceil((rect_diff[0] + 0.1) / (2 * s)))
    num_height = int(ceil((rect_diff[1] + 0.1) / (2 * s)))

    # loop through each cube
    for h in range(num_height):
        for w in range(num_width):

            a = x0 + w * (2 * s)  # minimum x coordinate of rectangle
            b = y0 + h * (2 * s)  # minimum y coordinate of rectangle
            c = a + 2 * s  # maximum x coordinate of rectangle
            d = b + 2 * s  # minimum y coordinate of rectangle

            # info about the cube
            cl = form_populated_cubes(a, b, c, d, data)

            # coordinates of the cube
            corresp_key_coord[(w, h)] = (a, b, c, d)

            # if the cube contains at least one point, add it to the populated cubes list
            if cl["num_points"] > 0:
                populated_cubes[(w, h)] = cl

    return populated_cubes, corresp_key_coord


def plot_grid_rect(data: np.ndarray, s: float, cube_kind: str = "populated") -> None:
    """
    Plot the cubes, with colors highlighting populated and highly populated cubes.

    :param data: input dataset.
    :param s: ``sigma``, determines the influence of a point in its neighborhood.
    :param cube_kind: option to consider populated cubes of highly populated cubes.
    """

    if cube_kind not in ['populated', 'highly_populated']:
        raise ValueError("cube_kind parameter must be one of: 'populated', 'highly_populated'.")

    # find info and coordinates of populated cubes
    cl, ckc = pop_cubes(data, s)

    cl_copy = cl.copy()

    # compute centers of mass of each populated cube
    coms = np.array([center_of_mass(list(cl.values())[i]) for i in range(len(cl))])

    if cube_kind == "highly_populated":
        # find highly populated cubes
        cl = highly_pop_cubes(cl, xi_c=3)
        # compute centers of mass of each highly populated cube
        coms_hpc = np.array([center_of_mass(list(cl.values())[i]) for i in range(len(cl))])

    # draw the minimal bounding rectangle
    fig, ax = plot_min_bound_rect(data)

    # draw the centers of mass
    ax.scatter(
        coms[:, 0],
        coms[:, 1],
        s=100,
        color="red",
        edgecolor="black",
    )

    # draw circles around highly populated cubes
    if cube_kind == "highly_populated":
        for i in range(len(coms_hpc)):
            ax.add_artist(
                plt.Circle(
                    (coms_hpc[i, 0], coms_hpc[i, 1]),
                    4 * s,
                    color="red",
                    fill=False,
                    linewidth=2,
                    alpha=0.6,
                )
            )

    # find the cubes connected to (highly) populated cubes
    connected_cubes = find_connected_cubes(cl, cl_copy, s)
    # find the final cubes ((highly) populated cubes + cubes connected to them)
    tot_cubes = {**cl, **connected_cubes}

    new_clusts = {i: tot_cubes[i] for i in tot_cubes.keys() if i not in cl.keys()}

    # plot connected cubes in yellow
    for key in list(new_clusts.keys()):
        (a, b, c, d) = ckc[key]
        ax.add_patch(
            Rectangle(
                (a, b),
                2 * s,
                2 * s,
                fill=True,
                color="yellow",
                alpha=0.3,
                linewidth=3,
            )
        )

    # draw the rectangles grid
    for key in list(ckc.keys()):

        (a, b, c, d) = ckc[key]

        # color the cube only if it is (highly) populated
        if key in list(cl.keys()):
            color_or_not = True if cl[key]["num_points"] > 0 else False
        else:
            color_or_not = False

        ax.add_patch(
            Rectangle(
                (a, b),
                2 * s,
                2 * s,
                fill=color_or_not,
                color="g",
                alpha=0.3,
                linewidth=3,
            )
        )
    plt.show()


def check_border_points_rectangles(data: np.ndarray, populated_cubes: Cubes) -> None:
    """
    Check if any of the points lie on the borders of the cubes, by checking if the sum of the number of points
    contained in each populated cube is equal to the total number of points of the dataset.

    :param data: input dataset.
    :param populated_cubes: populated cubes.
    """
    count = 0
    for i in range(len(populated_cubes)):
        count += list(populated_cubes.values())[i]["num_points"]

    if count == len(data):
        print("No points lie on the borders of rectangles")
    else:
        diff = len(data) - count
        print("{0} point(s) lie(s) on the border of rectangles".format(diff))


def highly_pop_cubes(pop_cub: Cubes, xi_c: float) -> Cubes:
    """
    Find highly populated cubes, i.e. cubes containing at least ``xi_c`` points.

    :param pop_cub: populated cubes.
    :param xi_c: ``xi_c = xi/2d``, where ``xi`` determines whether a density attractor is significant.
    :return: highly populated cubes.
    """
    highly_populated_cubes = {}

    # loop through the populated cubes
    for key in list(pop_cub.keys()):
        # if the number of points contained is greater than xi_c, add the cube to the highly populated cubes
        if pop_cub[key]['num_points'] >= xi_c:
            highly_populated_cubes[key] = pop_cub[key]

    return highly_populated_cubes


def center_of_mass(cube: CubeInfo):
    """compute the center of mass of a cube (rectangle)"""
    return np.array(cube['linear_sum']) / cube['num_points']


def check_connection(cube1: CubeInfo, cube2: CubeInfo, s: float, dist: str = "euclidean") -> bool:
    """
    Check if two cubes are connected (the distance between their centers of mass is not greater than ``4*s``).

    :param cube1: first cube.
    :param cube2: second cube.
    :param s: ``sigma``, determines the influence of a point in its neighborhood.
    :param dist: distance to use.
    :return: ``True`` if the cubes are connected,``False`` otherwise.
    """
    c1 = center_of_mass(cube1)
    c2 = center_of_mass(cube2)
    if dist == "euclidean":
        d = euclidean_distance(c1, c2)
    if d <= 4 * s:
        return True
    else:
        return False


def find_connected_cubes(hp_cubes: Cubes, cubes: Cubes, s: float, dist: str = "euclidean") -> Cubes:
    """
    Return connected cubes, i.e. cubes whose centers of mass' distances with a highly populated cube's center of mass
    is less than ``4*s``.

    :param hp_cubes: highly populated cubes.
    :param cubes: cubes.
    :param s: sigma, determines the influence of a point in its neighborhood.
    :param dist: distance to use.
    :return: new dictionary of cubes.
    """
    i = 0
    connected_cubes_keys = []

    # if a cube is connected to a highly populated cube, store its key
    for k1, c1 in list(hp_cubes.items()):
        for k2, c2 in list(cubes.items()):
            if k1 != k2:
                if check_connection(c1, c2, s, dist) is True:
                    connected_cubes_keys.append(k2)
        i += 1

    # remove duplicates
    keys_fin = set(connected_cubes_keys)

    connected_cubes = {key: cubes[key] for key in keys_fin}

    return connected_cubes


def near_with_cube(x: np.ndarray, cube_x: CubeInfo, tot_cubes: Cubes, s: float) -> np.ndarray:
    """
    Find points of cubes that are connected with cube_x and whose center of mass' distance from ``x``
    is less or equal to ``4*s``. The point itself is included.

    :param x: examined point.
    :param cube_x: cube which ``x`` belongs to.
    :param tot_cubes: the final cubes (highly populated + connected).
    :param s: sigma, determines the influence of a point in its neighborhood.
    :return: list of points belonging to cubes connected to ``cube_x`` and whose center of mass' distance from ``x``
             is less or equal to ``4*s``.
    """

    near_list = []

    # loop through the total cubes
    for cube in tot_cubes.values():

        d = euclidean_distance(x, center_of_mass(cube))

        # if the conditions are met, append all the points belonging to the cube to near_list
        if (d <= 4 * s) and (check_connection(cube_x, cube, s)):
            near_list.append(cube["points_coords"])

    near_list = flatten_list(near_list)

    return np.array(near_list)


def near_without_cube(x: np.ndarray, coord_dict: CubesCoords, tot_cubes: Cubes, s: float) -> np.ndarray:
    """
    Find the cube that ``x`` belongs to, and then find the points of cubes that are connected with it and whose center
    of mass' distance from ``x`` is less or equal to ``4*s``. The point itself is included.

    :param x: examined point.
    :param coord_dict: dictionary of the rectangles' coordinates.
    :param tot_cubes: the final cubes (highly populated + connected).
    :param s: sigma, determines the influence of a point in its neighborhood.
    :return: list of points belonging to cubes connected to ``cube_x`` and whose center of mass' distance from ``x``
             is less or equal to ``4*s``.
    """

    k = FindRect(x, coord_dict)

    if k is None:
        return np.array([])
    try:
        cube_x = tot_cubes[k]
    except KeyError:
        return np.array([])

    near_list = near_with_cube(x=x, cube_x=cube_x, tot_cubes=tot_cubes, s=s)

    return near_list


def plot_3d_or_contour(data: np.ndarray, s: float, three: bool = False, scatter: bool = False, prec: int = 3) -> None:
    """
    Plot the density function for the input dataset, either in 3D or 2D, using a contour
    plot.

    :param data: input dataset.
    :param s: ``sigma``, determines the influence of a point in its neighborhood.
    :param three: if ``True``, execute 3D plot and do not plot 2D countour plot.
    :param scatter: if ``True``, and if three is ``False``, draw a scatter plot on top
                    of the countour plot.
    :param prec: precision used to compute density function.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x_data = [data[:, 0].min(), data[:, 0].max()]
    y_data = [data[:, 1].min(), data[:, 1].max()]
    mixed_data = [min(x_data[0], y_data[0]), max(x_data[1], y_data[1])]

    xx = np.outer(
        np.linspace(mixed_data[0] - 1, mixed_data[1] + 1, prec * 10),
        np.ones(prec * 10),
    )
    yy = xx.copy().T  # transpose
    z = np.empty((prec * 10, prec * 10))
    for i, a in tqdm(enumerate(range(prec * 10))):
        for j, b in enumerate(range(prec * 10)):
            z[i, j] = gaussian_density(x=np.array([xx[i][a], yy[i][b]]), D=data, s=s)

    if three is True:
        ax = plt.axes(projection="3d")
        ax.plot_surface(xx, yy, z, cmap="winter", edgecolor="none")
        plt.show()
    else:
        CS = ax.contour(xx, yy, z, cmap="winter", edgecolor="none")
        ax.clabel(CS, inline=1, fontsize=FONTSIZE_NORMAL)

        if (scatter is True) and (three is False):
            plt.scatter(
                np.array(data)[:, 0],
                np.array(data)[:, 1],
                s=300,
                edgecolor="black",
                color="yellow",
                alpha=0.6,
            )

        plt.show()


def assign_cluster(data: np.ndarray, others: np.ndarray, attractor: Optional[Tuple[np.ndarray, bool]],
                   clust_dict: Dict[int, np.ndarray], processed: List[int]) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """
    Assign a density attractor to (a) point(s) or mark it/them as outlier(s).

    :param data: input dataset.
    :param others: coordinates of the point(s) whose clusters have to be assigned.
    :param attractor: coordinates of the point and flag to indicate if it is an outlier, i.e. if the density attractor
                      is significant.
    :param clust_dict: dictionary of points with the coordinates of their density attractor.
    :param processed: points that have been processed.
    :return: dictionary of clusters, i.e. points with their density attractor, and list of processed points.
    """
    if others is None or attractor is None:
        print("None")

        return clust_dict, processed

    # loop through the point(s) whose density attractor has been found
    for point in others:

        # take the index of the point
        point_index = np.nonzero(data == point)[0][0]

        if point_index not in processed:
            processed.append(point_index)

        # if point belongs to a cluster
        if attractor[1] is True:
            clust_dict[point_index] = attractor[0]
        # if point is an outlier
        else:
            clust_dict[point_index] = np.array([-1])

    return clust_dict, processed


def plot_infl(data: np.ndarray, s: float, xi: float) -> None:
    """
    Plot points of the dataset, showing which of them could be density attractors.

    :param data: input dataset.
    :param s: ``sigma``, determines the influence of a point in its neighborhood.
    :param xi: ``xi``, determines whether a density attractor is significant.
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.set_title("Significance of possible density attractors")

    z = []
    for point in data:
        z.append(gaussian_density(x=point, D=data, s=s))

    x_plot = list(range(len(data)))

    X_over = [x_plot[j] for j in range(len(data)) if z[j] >= xi]
    Y_over = [z[j] for j in range(len(data)) if z[j] >= xi]

    X_under = [x_plot[j] for j in range(len(data)) if z[j] < xi]
    Y_under = [z[j] for j in range(len(data)) if z[j] < xi]

    ax.scatter(
        X_over,
        Y_over,
        s=300,
        color="green",
        edgecolor="black",
        alpha=0.7,
        label="possibly significant",
    )

    ax.scatter(
        X_under,
        Y_under,
        s=300,
        color="yellow",
        edgecolor="black",
        alpha=0.7,
        label="not significant",
    )

    ax.axhline(xi, color="red", linewidth=2, label="xi")

    ax.set_ylabel("influence")

    # add indexes to points in plot
    for i, txt in enumerate(range(len(data))):
        ax.annotate(
            txt, (i, z[i]), fontsize=FONTSIZE_NORMAL, size=SIZE_NORMAL, ha="center", va="center"
        )

    ax.axis('off')
    ax.legend()
    plt.show()


def plot_3d_both(data: np.ndarray, s: float, xi: Optional[float] = None, prec: int = 3) -> None:
    """
    Show a 3D plot of the density function, with a horizontal plane cutting it at height xi, above which
    points can be considered signicant density attractors. Below this, a scatter plot and a countour plot
    show the actual points of the dataset, colored by significance of their 'density-attractiveness'.

    :param data: input dataset.
    :param s: ``sigma``, determines the influence of a point in its neighborhood.
    :param xi: ``xi``, determines whether a density attractor is significant.
    :param prec: precision used to compute density function.
    """
    from matplotlib import cm

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    x_data = [data[:, 0].min(), data[:, 0].max()]
    y_data = [data[:, 1].min(), data[:, 1].max()]
    mixed_data = [min(x_data[0], y_data[0]), max(x_data[1], y_data[1])]

    xx = np.outer(
        np.linspace(mixed_data[0] - 1, mixed_data[1] + 1, prec * 10),
        np.ones(prec * 10),
    )
    yy = xx.copy().T  # transpose
    z = np.empty((prec * 10, prec * 10))
    z_xi = np.empty((prec * 10, prec * 10))

    for i, a in tqdm(enumerate(range(prec * 10))):

        for j, b in enumerate(range(prec * 10)):

            z[i, j] = gaussian_density(x=np.array([xx[i][a], yy[i][b]]), D=data, s=s)
            if xi is not None:
                if z[i, j] >= xi:
                    z_xi[i, j] = z[i, j]
                else:
                    z_xi[i, j] = xi

    # to set colors according to xi value, red if greater, yellow if smaller
    if xi is not None:
        xi_data = []
        for a, b in zip(data[:, 0], data[:, 1]):
            to_be_eval = gaussian_density(x=np.array([a, b]), D=data, s=s)
            if to_be_eval >= xi:
                xi_data.append("red")
            else:
                xi_data.append("yellow")

    offset = -15

    if xi is not None:
        plane = ax.plot_surface(xx, yy, z_xi, cmap=cm.ocean, alpha=0.9)

    surf = ax.plot_surface(xx, yy, z, alpha=0.8, cmap=cm.ocean)

    cset = ax.contourf(xx, yy, z, zdir="z", offset=offset, cmap=cm.ocean)

    if xi is not None:
        color_plot = xi_data
    else:
        color_plot = "red"

    ax.scatter(
        data[:, 0],
        data[:, 1],
        offset,
        s=30,
        edgecolor="black",
        color=color_plot,
        alpha=0.6,
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_xlabel("X")

    ax.set_ylabel("Y")

    ax.set_zlabel("Z")
    ax.set_zlim(offset, np.max(z))
    # ax.set_title('3D surface with 2D contour plot projections')

    plt.show()


def density_attractor(data: np.ndarray, x: np.ndarray, coord_dict: CubesCoords, tot_cubes: Cubes,
                      s: float, xi: float, delta: float = 0.05, max_iter: int = 100,
                      dist: str = "euclidean") -> Union[Tuple[Tuple[float, bool], np.ndarray],
                                                        Tuple[None, None]]:
    """
    Find the density attractor for point x with a hill-climbing procedure. To speed up computations, during the
    procedure, store all the points y such that dist(x, y) <= s/2: they will belong to the same cluster as x.

    :param data: input dataset.
    :param x: point whose density attractors is to be found.
    :param coord_dict: dictionary of the rectangles' coordinates.
    :param tot_cubes: the final cubes (highly populated + connected).
    :param s: ``sigma``, determines the influence of a point in its neighborhood.
    :param xi: ``xi``, determines whether a density attractor is significant.
    :param delta: ``delta`` of gradient descent.
    :param max_iter: maximum number of iteration for finding a density attractor.
    :param dist: distance to use.
    :return: the coordinates of the density attractor, a flag to indicate its significance, and the list of the
             coordinates of the point(s) attracted by that density attractor.
    """

    x_i = []
    it_number = 0

    other_points = []

    while it_number < max_iter:

        # only in the first iteration
        if it_number == 0:
            x_i.append(x)
            it_number += 1

            for point in data:
                # if the condition is met, the point is going to have the same density attractor
                if euclidean_distance(point, x) <= s / 2:
                    other_points.append(list(point))
            continue

        old = x_i[-1]
        near_old = near_without_cube(
            x=old, coord_dict=coord_dict, tot_cubes=tot_cubes, s=s
        )

        grad = gradient_gaussian_density(x=old, D=near_old, s=s, dist=dist)

        new_x = old + delta * (grad / np.linalg.norm(grad))

        for point in data:
            # if the condition is met, the point is going to have the same density attractor
            if euclidean_distance(point, new_x) <= s / 2:
                other_points.append(list(point))

        near_new = near_without_cube(
            x=new_x, coord_dict=coord_dict, tot_cubes=tot_cubes, s=s
        )

        dens_new = gaussian_density(x=new_x, D=near_new, s=s, dist=dist)
        dens_old = gaussian_density(x=old, D=near_old, s=s, dist=dist)

        x_i.append(new_x)

        it_number += 1

        if dens_new < dens_old:
            attractor = old

            # xi determines if a density attractor is significant or not
            if dens_old >= xi:
                res = (attractor, True)
            else:
                res = (attractor, False)

            # probably these two last steps are useless
            other_points.sort()
            other_points = list(k for k, _ in groupby(other_points))

            return res, np.array(other_points)

    print(f"Max iteration number {max_iter} reached!")
    return None, None


def plot_clust_dict(data: np.ndarray, coord_df: pd.DataFrame) -> None:
    """
    Draw a scatter plot of the dataset, highlighting the clusters, the outliers and the density attractors, marked
    with a cross.

    :param data: input dataset.
    :param coord_df: dataframe of points with cluster labels and coordinates of density attractors.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    col = [
        COLOR_DICT[coord_df['label'][i] % len(COLOR_DICT)]
        if coord_df['label'][i] != -1
        else "red"
        for i in range(len(coord_df))
    ]

    # plot the points, colored according to the cluster they belong to (red if outliers)
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=300,
        edgecolor="black",
        color=col,
        alpha=0.8,
    )

    df_dens_attr = coord_df.groupby("label").mean()

    # draw density attractors, marked with a red X
    for i in range(df_dens_attr.iloc[-1].name + 1):
        ax.scatter(
            df_dens_attr.loc[i, "x"],
            df_dens_attr.loc[i, "y"],
            color="red",
            marker="X",
            s=300,
            edgecolor="black",
        )

    # add indexes to points in plot
    annotate_points(annotations=range(len(data)), points=data, ax=ax)

    plt.show()


def extract_cluster_labels(data: np.ndarray, cld: Dict[int, np.ndarray], tol: float = 2) -> pd.DataFrame:
    """
    Extract the labels from the dictionary of points with the coordinates of their density attractors.

    :param data: input dataset.
    :param cld: dictionary of points with the coordinates of their density attractors.
    :param tol: tolerance to merge points with the same density attractors.
    :return: dataframe of points with cluster labels and coordinates of density attractors.
    """

    def similarity(x: np.ndarray, y: np.ndarray, _tol: float) -> bool:
        """check if two vectors are equal, with a given tolerance"""
        if (abs(x[0] - y[0]) <= _tol) and (abs(x[1] - y[1]) <= _tol):
            return True
        else:
            return False

    cld_sorted = OrderedDict(sorted(cld.items(), key=lambda t: t[0]))
    val = list(cld_sorted.values())
    l_mod = [np.round(val[i], 1) for i in range(len(val))]

    lr = {i: l_mod[i] for i in range(len(l_mod)) if len(l_mod[i]) == 2}
    da_list = list(range(len(data)))
    for i, el in enumerate(l_mod):
        if len(el) == 1:
            da_list[i] = -1
        else:
            for k, coord in lr.items():
                if similarity(coord, el, tol) is True:
                    da_list[i] = da_list[k]

    keys = list(Counter(da_list).keys())
    keys.sort()
    if -1 in keys:
        range_labels = len(keys) - 1
        labels = [-1] + list(range(range_labels))
    else:
        range_labels = len(keys)
        labels = list(range(range_labels))

    label_dict = dict(zip(keys, labels))
    fin_labels = [label_dict[i] for i in da_list]

    df = pd.DataFrame(l_mod)
    if len(Counter(fin_labels)) == 1:
        df["added"] = [np.nan] * len(df)
    df.columns = ["x", "y"]
    df["label"] = fin_labels

    return df


def DENCLUE(data: np.ndarray, s: float, xi: float = 3, xi_c: float = 3, tol: float = 2, dist: str = "euclidean",
            prec: int = 20, plotting: bool = True) -> list:
    """
    Execute the DENCLUE algorithm, whose basic idea is to model the overall point density analytically as the sum of
    influence functions of the data points. Clusters can then be identified by determining density-attractors.

    :param data: input dataset.
    :param s: ``sigma``, determines the influence of a point in its neighborhood.
    :param xi: ``xi``, determines whether a density attractor is significant.
    :param xi_c: ``xi/2d``, where ``d=2`` is the dimension of input data.
    :param tol: tolerance for determining if two density attractors coincide.
    :param dist: distance to use.
    :param prec: precision used to compute density function.
    :param plotting: if ``True``, show plots.
    :return: list of cluster labels.
    """
    clust_dict = {}
    processed = []

    z, d = pop_cubes(data=data, s=s)
    print("Number of populated cubes: ", len(z))
    check_border_points_rectangles(data, z)
    hpc = highly_pop_cubes(z, xi_c=xi_c)
    print("Number of highly populated cubes: ", len(hpc))
    connected_cubes = find_connected_cubes(hpc, z, s=s)
    new_cubes = {**hpc, **connected_cubes}

    if plotting is True:
        plot_grid_rect(data, s=s, cube_kind="populated")
        plot_grid_rect(data, s=s, cube_kind="highly_populated")

        plot_3d_or_contour(data, s=s, three=False, scatter=True, prec=prec)

        plot_3d_both(data, s=s, xi=xi, prec=prec)

    if len(new_cubes) != 0:

        temp_points = [cube["points_coords"] for cube in list(new_cubes.values())]
        points_to_process = flatten_list(temp_points)

    else:
        points_to_process = []

    initial_noise = []
    for elem in data:
        if len((np.nonzero(points_to_process == elem))[0]) == 0:
            initial_noise.append(elem)

    for num, point in tqdm(enumerate(points_to_process)):
        delta = 0.02
        r, o = None, None

        while r is None:
            r, o = density_attractor(data=data, x=point, coord_dict=d, tot_cubes=new_cubes, s=s,
                                     xi=xi, delta=delta, max_iter=600, dist=dist)
            delta = delta * 2

        clust_dict, proc = assign_cluster(data=data, others=o, attractor=r,
                                          clust_dict=clust_dict, processed=processed)

    for point in initial_noise:
        point_index = np.nonzero(data == point)[0][0]
        clust_dict[point_index] = [-1]

    try:
        coord_df = extract_cluster_labels(data, clust_dict, tol)
    except Exception as e:
        print(
            "There was an error when extracting clusters. Increase the number of points or try with a less"
            " pathological case: look at the other plots to have an idea of why it failed."
        )
        print(e)

    if plotting is True:
        plot_clust_dict(data, coord_df)

    return list(coord_df["label"].values)
