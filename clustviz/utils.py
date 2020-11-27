import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import ConvexHull

# OLD_COLOR_DICT = {
#     0: "seagreen",
#     1: "lightcoral",
#     2: "yellow",
#     3: "grey",
#     4: "pink",
#     5: "navy",
#     6: "orange",
#     7: "purple",
#     8: "salmon",
#     9: "olive",
#     10: "brown",
#     11: "tan",
#     12: "plum",
#     13: "red",
#     14: "lightblue",
#     15: "khaki",
#     16: "gainsboro",
#     17: "peachpuff",
# }

FONTSIZE_NORMAL = 10
FONTSIZE_BIGGER = 12

SIZE_NORMAL = 10
SIZE_BIGGER = 12

COLOR_DICT = {
    0: "seagreen",
    1: "lightcoral",
    2: "yellow",
    3: "grey",
    4: "pink",
    5: "turquoise",
    6: "orange",
    7: "purple",
    8: "yellowgreen",
    9: "olive",
    10: "brown",
    11: "tan",
    12: "plum",
    13: "rosybrown",
    14: "lightblue",
    15: "khaki",
    16: "gainsboro",
    17: "peachpuff",
    18: "lime",
    19: "peru",
    20: "dodgerblue",
    21: "teal",
    22: "royalblue",
    23: "tomato",
    24: "bisque",
    25: "palegreen",
}

DBSCAN_COLOR_DICT = {

    -1: "red",
    0: "lightblue",
    1: "lightcoral",
    2: "yellow",
    3: "grey",
    4: "pink",
    5: "navy",
    6: "orange",
    7: "purple",
    8: "salmon",
    9: "olive",
    10: "brown",
    11: "tan",
    12: "lime",
}

CURE_REPS_COLORS = [
    "red",
    "crimson",
    "indianred",
    "lightcoral",
    "salmon",
    "darksalmon",
    "firebrick",
]


def encircle(x, y, ax, **kwargs):
    """plot a line-boundary around a cluster (at least 3 points are required)"""
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kwargs)
    ax.add_patch(poly)


def convert_colors(dict_colors, alpha=0.5):
    """modify the transparency of each color of a dictionary of colors to the desired alpha"""
    new_dict_colors = {}

    for i, col in enumerate(dict_colors.values()):
        new_dict_colors[i] = tuple(list(colors.to_rgb(col)) + [alpha])

    return new_dict_colors


def euclidean_distance(a, b):
    """Returns Euclidean distance of two arrays"""
    return np.linalg.norm(np.array(a) - np.array(b))


def dist1(x, y):
    """Original euclidean distance"""
    return np.sqrt(np.sum((x - y) ** 2))


def dist2(data, x, y):
    """ Euclidean distance which takes keys of a dictionary (X_dict) as inputs """
    return np.sqrt(np.sum((data[x] - data[y]) ** 2))


def chernoffBounds(u_min, f, N, d, k):
    """
    u_min: size of the smallest cluster u.
    f: percentage of cluster points (0 <= f <= 1).
    N: total size.
    s: sample size.
    d: 0 <= d <= 1
    the probability that the sample contains less than f*|u| points of cluster u is less than d.

    If one uses as |u| the minimum cluster size we are interested in, the result is
    the minimum sample size that guarantees that for k clusters
    the probability of selecting fewer than f*|u| points from any one of the clusters u is less than k*d.

    """

    l = np.log(1 / d)
    res = (
            f * N + N / u_min * l + N / u_min * np.sqrt(l ** 2 + 2 * f * u_min * l)
    )
    print(
        "If the sample size is {0}, the probability of selecting fewer "
        "than {1} points from".format(math.ceil(res), round(f * u_min))
        + " any one of the clusters is less than {0}".format(k * d)
    )

    return res
