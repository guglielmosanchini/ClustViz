import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from algorithms.optics import dist1
import random


def scan_neigh1_mod(data, point, eps):
    """
    Neighborhood search for a point of a given dataset-dictionary (data)
    with a fixed eps; it returns also the point itself, differently from
    scan_neigh1 of OPTICS.

    :param data: input dictionary.
    :param point: point whose neighborhood is to be examined.
    :param eps: radius of search.
    :return: dictionary of neighborhood points.
    """

    neigh = {}

    for i, element in enumerate(data.values()):

        d = dist1(element, point)

        if d <= eps:
            neigh.update({str(i): element})

    return neigh


def point_plot_mod(X, X_dict, point, eps, ClustDict):
    """
    Plots a scatter plot of points, where the point (x,y) is light black and
    surrounded by a red circle of radius eps, where already processed point are plotted
    according to ClustDict and without edgecolor, whereas still-to-process points are green
    with black edgecolor.

    :param X: input array.
    :param X_dict: input dictionary version of X.
    :param point: coordinates of the point that is currently inspected.
    :param eps: radius of the circle to plot around the point (x,y).
    :param ClustDict: dictionary of the form point_index:cluster_label, built by DBSCAN
    """

    colors = {
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

    fig, ax = plt.subplots(figsize=(14, 6))

    # plot scatter points in color lime
    plt.scatter(X[:, 0], X[:, 1], s=300, color="lime", edgecolor="black")

    # plot colors according to clusters
    for i in ClustDict:
        plt.scatter(
            X_dict[i][0], X_dict[i][1], color=colors[ClustDict[i] % 12], s=300
        )

    # plot the last added point bigger and black, with a red circle surrounding it
    plt.scatter(
        x=X_dict[point][0], y=X_dict[point][1], s=400, color="black", alpha=0.4
    )

    circle1 = plt.Circle(
        (X_dict[point][0], X_dict[point][1]),
        eps,
        color="r",
        fill=False,
        linewidth=3,
        alpha=0.7,
    )
    ax.add_artist(circle1)

    # add indexes to points in the scatterplot
    for i, txt in enumerate([i for i in range(len(X))]):
        ax.annotate(
            txt,
            (X[:, 0][i], X[:, 1][i]),
            fontsize=10,
            size=10,
            ha="center",
            va="center",
        )

    plt.show()


def plot_clust_DB(X, ClustDict, eps, circle_class=None, noise_circle=True):
    """
    Scatter plot of the data points, colored according to the cluster they belong to; circle_class Plots
    circles around some or all points, with a radius of eps; if Noise_circle is True, circle are also plotted
    around noise points.

    :param X: input array.
    :param ClustDict: dictionary of the form point_index:cluster_label, built by DBSCAN.
    :param eps: radius of the circles to plot around the points.
    :param circle_class: if == "true", plots circles around every non-noise point, else plots circles
                         only around points belonging to certain clusters, e.g. circle_class = [1,2] will
                         plot circles around points belonging to clusters 1 and 2.
    :param noise_circle: if True, plots circles around noise points

    """
    # create dictionary of X
    X_dict = dict(zip([str(i) for i in range(len(X))], X))

    # create new dictionary of X, adding the cluster label
    new_dict = {
        key: (val1, ClustDict[key])
        for key, val1 in zip(list(X_dict.keys()), list(X_dict.values()))
    }

    new_dict = OrderedDict((k, new_dict[k]) for k in list(ClustDict.keys()))

    df = pd.DataFrame(
        dict(
            x=[i[0][0] for i in list(new_dict.values())],
            y=[i[0][1] for i in list(new_dict.values())],
            label=[i[1] for i in list(new_dict.values())],
        ),
        index=new_dict.keys(),
    )

    colors = {
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

    fig, ax1 = plt.subplots(1, 1, figsize=(18, 6))

    lista_lab = list(df.label.value_counts().index)

    # plot points colored according to the cluster they belong to
    for lab in lista_lab:
        df_sub = df[df.label == lab]
        plt.scatter(
            df_sub.x,
            df_sub.y,
            color=colors[lab % 12],
            s=300,
            edgecolor="black",
        )

    # plot circles around noise, colored according to the cluster they belong to
    if noise_circle is True:

        df_noise = df[df.label == -1]

        for i in range(len(df_noise)):
            ax1.add_artist(
                plt.Circle(
                    (df_noise["x"].iloc[i], df_noise["y"].iloc[i]),
                    eps,
                    color="r",
                    fill=False,
                    linewidth=3,
                    alpha=0.7,
                )
            )

    # plot circles around points, colored according to the cluster they belong to
    if circle_class is not None:
        # around every points or only around specified clusters
        if circle_class != "true":
            lista_lab = circle_class

        for lab in lista_lab:

            if lab != -1:

                df_temp = df[df.label == lab]

                for i in range(len(df_temp)):
                    ax1.add_artist(
                        plt.Circle(
                            (df_temp["x"].iloc[i], df_temp["y"].iloc[i]),
                            eps,
                            color=colors[lab],
                            fill=False,
                            linewidth=3,
                            alpha=0.7,
                        )
                    )

    ax1.set_xlabel("")
    ax1.set_ylabel("")

    # plot labels of points
    for i, txt in enumerate([i for i in range(len(X))]):
        ax1.annotate(
            txt,
            (X[:, 0][i], X[:, 1][i]),
            fontsize=10,
            size=10,
            ha="center",
            va="center",
        )

    plt.show()


def DBSCAN(data, eps, minPTS, plotting=False, print_details=False):
    """
    DBSCAN algorithm.

    :param data: input array.
    :param eps: radius of a point within which to search for minPTS points.
    :param minPTS: minimum number of neighbors for a point to be considered a core point.
    :param plotting: if True, executes point_plot_mod, plotting every time a points is
                     added to a clusters
    :param print_details: if True, prints the length of the "external" NearestNeighborhood
                          and of the "internal" one (in the while loop).
    :return ClustDict: dictionary of the form point_index:cluster_label.

    """

    # initialize dictionary of clusters
    ClustDict = {}

    clust_id = -1

    X_dict = dict(zip([str(i) for i in range(len(data))], data))

    processed = []

    # for every point in the dataset
    for point in X_dict:

        # if it hasnt been visited
        if point not in processed:
            # mark it as visited
            processed.append(point)
            # scan its neighborhood
            N = scan_neigh1_mod(X_dict, X_dict[point], eps)

            if print_details is True:
                print("len(N): ", len(N))
            # if there are less than minPTS in its neighborhood, classify it as noise
            if len(N) < minPTS:

                ClustDict.update({point: -1})

                if plotting is True:
                    point_plot_mod(data, X_dict, point, eps, ClustDict)
            # else if it is a Core point
            else:
                # increase current id of cluster
                clust_id += 1
                # put it in the cluster dictionary
                ClustDict.update({point: clust_id})

                if plotting is True:
                    point_plot_mod(data, X_dict, point, eps, ClustDict)
                # add it to the temporary processed list
                processed_list = [point]
                # remove it from the neighborhood N
                del N[point]
                # until the neighborhood is empty
                while len(N) > 0:

                    if print_details is True:
                        print("len(N) in while loop: ", len(N))
                    # take a random point in neighborhood
                    n = random.choice(list(N.keys()))
                    # but the point must not be in processed_list aka already visited
                    while n in processed_list:
                        n = random.choice(list(N.keys()))
                    # put it in processed_list
                    processed_list.append(n)
                    # remove it from the neighborhood
                    del N[n]
                    # if it hasnt been visited
                    if n not in processed:
                        # mark it as visited
                        processed.append(n)
                        # scan its neighborhood
                        N_2 = scan_neigh1_mod(X_dict, X_dict[n], eps)

                        if print_details is True:
                            print("len N2: ", len(N_2))
                        # if it is a core point
                        if len(N_2) >= minPTS:
                            # add each element of its neighborhood to the neighborhood N
                            for element in N_2:

                                if element not in processed_list:
                                    N.update({element: X_dict[element]})

                    # if n has not been inserted into cluster dictionary or if it has previously been
                    # classified as noise, update the cluster dictionary
                    if (n not in ClustDict) or (ClustDict[n] == -1):
                        ClustDict.update({n: clust_id})

                    if plotting is True:
                        point_plot_mod(data, X_dict, n, eps, ClustDict)

    return ClustDict
