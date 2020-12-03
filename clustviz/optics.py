import random
from typing import Dict, Optional, Iterable, List, Any, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

from clustviz.utils import dist1, dist2, DBSCAN_COLOR_DICT, annotate_points


def point_plot(X: np.ndarray, X_dict: Dict[str, np.ndarray], o: str,
               eps: float, processed: Optional[Iterable] = None, col: str = "yellow") -> None:
    """
    Plots a scatter plot of points, where the point (x,y) is light black and
    surrounded by a red circle of radius eps, where processed point are plotted
    in col (yellow by default) and without edgecolor, whereas still-to-process points are green
    with black edgecolor.

    :param X: input array.
    :param X_dict: input dictionary version of X.
    :param o: point that is currently inspected.
    :param eps: radius of the circle to plot around the point (x,y).
    :param processed: already processed points, to plot in col
    :param col: color to use for processed points, yellow by default.
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    # plot every point in color lime
    ax.scatter(X[:, 0], X[:, 1], s=300, color="lime", edgecolor="black")

    # plot clustered points according to appropriate colors
    if processed is not None:
        for i in processed:
            ax.scatter(X_dict[i][0], X_dict[i][1], color=col, s=300)

    # plot last added point in black and surround it with a red circle
    plt.scatter(
        x=X_dict[o][0], y=X_dict[o][1], s=400, color="black", alpha=0.4
    )

    circle1 = plt.Circle(
        (X_dict[o][0], X_dict[o][1]),
        eps,
        color="r",
        fill=False,
        linewidth=3,
        alpha=0.7,
    )
    ax.add_artist(circle1)

    # add indexes to points in plot
    annotate_points(annotations=range(len(X)), points=X, ax=ax)

    plt.show()


def scan_neigh1(data: Dict[str, np.ndarray], point: np.ndarray, eps: float) -> Dict[str, np.ndarray]:
    """
    Neighborhood search for a point of a given dataset-dictionary (data)
    with a fixed eps.

    :param data: input dictionary.
    :param point: point whose neighborhood is to be examined.
    :param eps: radius of search.
    :return: dictionary of neighborhood points.
    """

    neigh = {}

    for i, element in enumerate(data.values()):

        d = dist1(element, point)

        if (d <= eps) and (d != 0):
            neigh.update({str(i): element})

    return neigh


def scan_neigh2(data: Dict[str, np.ndarray], point: np.ndarray, eps: float) -> List[str]:
    """
    Variation of scan_neigh1 that returns only the keys of the input dictionary
    with the euclidean distances <= eps from the point.

    :param data: input dictionary.
    :param point: point whose neighborhood is to be examined.
    :param eps: radius of search.
    :return: keys of dictionary of neighborhood points, ordered by distance.
    """

    distances = {}

    for i, element in enumerate(data.values()):

        d = dist1(element, point)

        if (d <= eps) and (d != 0):
            distances.update({str(i): d})

    d_sorted = sorted(distances.items(), key=lambda x: x[1], reverse=False)

    return d_sorted


def minPTSdist(data: Dict[str, np.ndarray], o: str, minPTS: int, eps: float) -> Union[float, Any]:
    """
    Returns the minPTS-distance of a point if it is a core point,
    else it returns np.inf

    :param data: input dictionary.
    :param o: key of point of interest.
    :param minPTS: minimum number of neighbors for a point to be considered a core point.
    :param eps: radius of a point within which to search for minPTS points
    :return: minPTS-distance of data[o] or np.inf
    """

    S = scan_neigh2(data, data[o], eps)

    if len(S) >= minPTS - 1:

        return S[minPTS - 2][1]

    else:

        return np.inf


def reach_dist(data: Dict[str, np.ndarray], x: str, y: str, minPTS: int, eps: float) -> Union[float, Any]:
    """
    Reachability distance
    (even if it is not a distance because it isn't symmetrical)

    :param data: input dictionary.
    :param x: first point.
    :param y: second point.
    :param minPTS: minimum number of neighbors for a point to be considered a core point.
    :param eps: radius of a point within which to search for minPTS points.
    :return: reachability distance of x and y

    """

    return max(dist2(data, x, y), minPTSdist(data, y, minPTS, eps))


def reach_plot(data: Dict[str, np.ndarray], ClustDist: Dict[str, float], eps: float) -> None:
    """
    Plots the reachability plot, along with a horizontal line denoting eps,
    from the ClustDist produced by OPTICS

    :param data: input dictionary.
    :param ClustDist: output of OPTICS function, dictionary of the form point_index:reach_dist.
    :param eps: radius of a point within which to search for minPTS points.

    """

    height_factor_noise = 1.15
    epsilon_stringcode = "\u03B5"

    plot_dic = {}

    # create dictionary for reachability plot, keys will be the bar labels and the value will be the height
    # if the value is infinity, the height will be eps*1.15 by default
    for key, value in ClustDist.items():

        if np.isinf(value) == True:
            plot_dic[key] = eps * height_factor_noise
        else:
            plot_dic[key] = ClustDist[key]

    missing_keys = list(set(data.keys()) - set(ClustDist.keys()))

    tick_list = list(ClustDist.keys()) + [" "] * len(missing_keys)

    # add the necessary zeroes for points that are still to be processed
    for m_k in missing_keys:
        plot_dic[m_k] = 0

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # bar plot of reachability
    ax.bar(plot_dic.keys(), plot_dic.values(), color="lightblue")

    ax.set_xticks(tick_list)

    # plot horizontal line for eps
    ax.axhline(eps, color="red", linewidth=3)

    ax1 = ax.twinx()
    ax1.set_ylim(ax.get_ylim())
    ax1.set_yticks([eps])
    ax1.set_yticklabels([epsilon_stringcode])

    plt.show()


def OPTICS(X: np.ndarray, eps: float, minPTS: int, plot: bool = True,
           plot_reach: bool = False) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Executes the OPTICS algorithm. Similar to DBSCAN, but uses a priority queue.

    :param X: input array
    :param eps: radius of a point within which to search for minPTS points.
    :param minPTS: minimum number of neighbors for a point to be considered a core point.
    :param plot: if True, the scatter plot of the function point_plot is displayed at each step.
    :param plot_reach: if True, the reachability plot is displayed at each step.
    :return: ClustDist, a dictionary of the form point_index:reach_dist, and
             CoreDist, a dictionary of the form point_index:core_dist
    """

    ClustDist = {}
    CoreDist = {}
    Seed = {}
    processed = []

    # create dictionary
    X_dict = dict(zip([str(i) for i in range(len(X))], X))

    # until all points have been processed
    while len(processed) != len(X):

        # if queue is empty take a random point
        if len(Seed) == 0:

            unprocessed = list(set(list(X_dict.keys())) - set(processed))

            (o, r) = (random.choice(unprocessed), np.inf)

        # else take the minimum and delete it from the queue
        else:

            (o, r) = (min(Seed, key=Seed.get), Seed[min(Seed, key=Seed.get)])

            del Seed[o]

        # scan the neighborhood of the point
        N = scan_neigh1(X_dict, X_dict[o], eps)
        # update the cluster dictionary and the core distance dictionary
        ClustDist.update({o: r})

        CoreDist.update({o: minPTSdist(X_dict, o, minPTS, eps)})

        if plot is True:

            point_plot(X, X_dict, o, eps, processed)

            if plot_reach is True:
                reach_plot(X_dict, ClustDist, eps)

        # mark o as processed
        processed.append(o)

        # if the point is core
        if len(N) >= minPTS - 1:
            # for each unprocessed point in the neighborhood
            for n in N:

                if n in processed:

                    continue

                else:
                    # compute its reach_dist from o
                    p = reach_dist(X_dict, n, o, minPTS, eps)
                    # if it is in Seed, update its reach_dist if it is lower
                    if n in Seed:

                        if p < Seed[n]:
                            Seed[n] = p
                    # else, insert it into the Seed
                    else:

                        Seed.update({n: p})

    return ClustDist, CoreDist


def ExtractDBSCANclust(ClustDist: Dict[str, float], CoreDist: Dict[str, float], eps_db: float) -> Dict[str, int]:
    """
    Extracts cluster in a DBSCAN fashion; one can use any eps_db <= eps of OPTICS

    :param ClustDist: ClustDist of OPTICS, a dictionary of the form point_index:reach_dist
    :param CoreDist: CoreDist of OPTICS, a dictionary of the form point_index:core_dist
    :param eps_db: the eps to choose for DBSCAN
    :return: dictionary of clusters, of the form point_index:cluster_label

    """

    Clust_Dict = {}

    noise = -1

    clust_id = -1

    for key, value in ClustDist.items():
        # new cluster
        if value > eps_db:
            # if it is a core point
            if CoreDist[key] <= eps_db:

                clust_id += 1

                Clust_Dict[key] = clust_id
            # else, mark it as noise
            else:

                Clust_Dict[key] = noise
        # same cluster
        else:

            Clust_Dict[key] = clust_id

    return Clust_Dict


def plot_clust(X: np.ndarray, ClustDist: Dict[str, float], CoreDist: Dict[str, float],
               eps: float, eps_db: float) -> None:
    """
    Plot a scatter plot on the left, where points are colored according to the cluster they belong to,
    and a reachability plot on the right, where colors correspond to the clusters, and the two horizontal
    lines represent eps and eps_db

    :param X: input array
    :param ClustDist: ClustDist of OPTICS, a dictionary of the form point_index:reach_dist
    :param CoreDist: CoreDist of OPTICS, a dictionary of the form point_index:core_dist
    :param eps: the eps used to run OPTICS
    :param eps_db: the eps to choose for DBSCAN

    """

    if eps_db > eps:
        raise ValueError(f"eps_db: {eps_db} must be less than eps: {eps}")

    X_dict = dict(zip([str(i) for i in range(len(X))], X))
    # extract the cluster dictionary using DBSCAN
    cl = ExtractDBSCANclust(ClustDist, CoreDist, eps_db)

    new_dict = {
        key: (val1, cl[key])
        for key, val1 in zip(list(X_dict.keys()), list(X_dict.values()))
    }

    new_dict = OrderedDict((k, new_dict[k]) for k in list(ClustDist.keys()))

    df = pd.DataFrame(
        dict(
            x=[i[0][0] for i in list(new_dict.values())],
            y=[i[0][1] for i in list(new_dict.values())],
            label=[i[1] for i in list(new_dict.values())],
        ),
        index=list(new_dict.keys()),
    )

    # first plot: scatter plot of points colored according to the cluster they belong to
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    grouped = df.groupby("label")
    for key, group in grouped:
        group.plot(
            ax=ax1,
            kind="scatter",
            x="x",
            y="y",
            label=key,
            color=DBSCAN_COLOR_DICT[key % len(DBSCAN_COLOR_DICT) if key != -1 else -1],
            s=300,
            edgecolor="black",
        )

    ax1.set_xlabel("")
    ax1.set_ylabel("")

    annotate_points(annotations=range(len(X)), points=X, ax=ax1)

    # second plot: reachability plot, with colors corresponding to clusters
    height_factor_noise = 1.15
    epsilon_stringcode = "\u03B5"
    plot_dic = {}

    for key, value in ClustDist.items():

        if np.isinf(value) == True:

            plot_dic[key] = eps * height_factor_noise

        else:

            plot_dic[key] = ClustDist[key]

    ax2.bar(
        plot_dic.keys(),
        plot_dic.values(),
        color=[DBSCAN_COLOR_DICT[i % len(DBSCAN_COLOR_DICT)] if i != -1 else DBSCAN_COLOR_DICT[-1] for i in df.label],
    )

    ax2.axhline(eps, color="black", linewidth=3)

    ax2.axhline(eps_db, color="black", linewidth=3)

    ax3 = ax2.twinx()
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_yticks([eps, eps_db])
    ax3.set_yticklabels([epsilon_stringcode, epsilon_stringcode + "'"])

    plt.show()
