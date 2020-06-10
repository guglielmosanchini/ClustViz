import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.optics import dist1
from algorithms.agglomerative import dist_mat_gen
from matplotlib.patches import Rectangle
from collections import Counter, OrderedDict
from copy import deepcopy
import random
import math
from GUI_classes.utils_gui import encircle, convert_colors


def point_plot_mod2(
    X,
    a,
    reps,
    level_txt,
    level2_txt=None,
    par_index=None,
    u=None,
    u_cl=None,
    initial_ind=None,
    last_reps=None,
    not_sampled=None,
    not_sampled_ind=None,
    n_rep_fin=None,
):
    """
    Scatter-plot of input data points, colored according to the cluster they belong to.
    A rectangle with red borders is displayed around the last merged cluster; representative points
    of last merged cluster are also plotted in red, along with the center of mass, plotted as a
    red cross. The current number of clusters and current distance are also displayed in the right
    upper corner.
    In the last phase of CURE algorithm variation for large datasets, arrows are
    displayed from every not sampled point to its closest representative point; moreover, representative
    points are surrounded by small circles, to make them more visible. Representative points of different
    clusters are plotted in different nuances of red.

    :param X: input data array.
    :param a: input dataframe built by CURE algorithm, listing the cluster and the x and y
              coordinates of each point.
    :param reps: list of the coordinates of representative points.
    :param level_txt: distance at which current merging occurs displayed in the upper right corner.
    :param level2_txt: incremental distance (not used).
    :param par_index: partial index to take the shuffling of indexes into account.
    :param u: first cluster to be merged.
    :param u_cl: second cluster to be merged.
    :param initial_ind: initial partial index.
    :param last_reps: dictionary of last representative points.
    :param not_sampled: coordinates of points that have not been initially sampled, in the large dataset version.
    :param not_sampled_ind: indexes of not_sampled point_indices.
    :param n_rep_fin: number of representatives to use for each cluster in the final assignment phase in the large
                      dataset version.
    :return list_keys_diz: if par_index is not None, returns the new indexes of par_index.

    """

    # diz is used to take the shuffling of data into account, e.g. if the first row doesn'#
    # correspond to point 0: this is useful for the large dataset version of CURE, where data points
    # are randomly sampled, but the initial indices are kept to be plotted.
    if par_index is not None:
        diz = dict(zip(par_index, [i for i in range(len(par_index))]))

    fig, ax = plt.subplots(figsize=(14, 6))

    # points that still need to be processed are plotted in lime color
    plt.scatter(X[:, 0], X[:, 1], s=300, color="lime", edgecolor="black")

    # drops the totally null columns, so that the number of columns goes to 2*(cardinality of biggest cluster)
    a = a.dropna(1, how="all")

    colors = {
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
    }
    color_dict_rect = convert_colors(colors, alpha=0.3)

    # to speed things up, this splits all points inside the clusters' names, and start gives the starting index
    # that shows where clusters with more than 1 element start (because they are always appended to a)
    len_ind = [len(i.split("-")) for i in list(a.index)]
    start = np.min([i for i in range(len(len_ind)) if len_ind[i] > 1])

    # for each cluster, take the single points composing it and plot them in the appropriate color, if
    # necessary taking the labels of par_index into account
    for ind, i in enumerate(range(start, len(a))):
        point = a.iloc[i].name.replace("(", "").replace(")", "").split("-")
        if par_index is not None:
            X_clust = [X[diz[point[j]], 0] for j in range(len(point))]
            Y_clust = [X[diz[point[j]], 1] for j in range(len(point))]

            ax.scatter(X_clust, Y_clust, s=350, color=colors[ind % 18])
        else:
            point = [int(i) for i in point]
            X_clust = [X[point[j], 0] for j in range(len(point))]
            Y_clust = [X[point[j], 1] for j in range(len(point))]

            ax.scatter(X_clust, Y_clust, s=350, color=colors[ind % 18])

    # last merged cluster, so the last element of matrix a
    point = a.iloc[-1].name.replace("(", "").replace(")", "").split("-")
    # finding the new center of mass the newly merged cluster
    if par_index is not None:
        point = [diz[point[i]] for i in range(len(point))]
        com = X[point].mean(axis=0)
    else:
        point = [int(i) for i in point]
        com = X[point].mean(axis=0)

    # plotting the center of mass, marked with an X
    plt.scatter(
        com[0], com[1], s=400, color="r", marker="X", edgecolor="black"
    )

    # plotting representative points in red
    x_reps = [i[0] for i in reps]
    y_reps = [i[1] for i in reps]
    plt.scatter(x_reps, y_reps, s=360, color="r", edgecolor="black")

    # finding the right measures for the rectangle
    rect_min = X[point].min(axis=0)
    rect_diff = X[point].max(axis=0) - rect_min

    xmin, xmax, ymin, ymax = plt.axis()
    xwidth = xmax - xmin
    ywidth = ymax - ymin

    # adding the rectangle, using two rectangles one above the other to use different colors
    # for the border and for the inside
    if len(point) <= 2:

        ax.add_patch(
            Rectangle(
                (rect_min[0] - xwidth * 0.02, rect_min[1] - ywidth * 0.04),
                rect_diff[0] + xwidth * 0.04,
                rect_diff[1] + ywidth * 0.08,
                fill=True,
                color=color_dict_rect[ind % 18],
                linewidth=3,
                ec="red",
            )
        )
    else:
        encircle(
            X_clust,
            Y_clust,
            ax=ax,
            color=color_dict_rect[ind % 18],
            linewidth=3,
            ec="red",
        )

    # adding labels to points in the plot

    if initial_ind is not None:
        for i, txt in enumerate(initial_ind):
            ax.annotate(
                txt,
                (X[:, 0][i], X[:, 1][i]),
                fontsize=10,
                size=10,
                ha="center",
                va="center",
            )
    else:
        for i, txt in enumerate([i for i in range(len(X))]):
            ax.annotate(
                txt,
                (X[:, 0][i], X[:, 1][i]),
                fontsize=10,
                size=10,
                ha="center",
                va="center",
            )

    # adding the annotations
    ax.annotate(
        "min_dist: " + str(round(level_txt, 5)),
        (xmax * 0.75, ymax * 0.9),
        fontsize=12,
        size=12,
    )

    if level2_txt is not None:
        ax.annotate(
            "dist_incr: " + str(round(level2_txt, 5)),
            (xmax * 0.75, ymax * 0.8),
            fontsize=12,
            size=12,
        )

    ax.annotate(
        "n° clust: " + str(len(a)),
        (xmax * 0.75, ymax * 0.7),
        fontsize=12,
        size=12,
    )

    plt.show()

    # everything down from here refers to the last phase of the large dataset version, the assignment phase
    if last_reps is not None:

        fig, ax = plt.subplots(figsize=(14, 6))

        # plot all the points in color lime
        plt.scatter(X[:, 0], X[:, 1], s=300, color="lime", edgecolor="black")

        # find the centers of mass of the clusters using the matrix a to find which points belong to
        # which cluster
        coms = []
        for ind, i in enumerate(range(0, len(a))):
            point = a.iloc[i].name.replace("(", "").replace(")", "").split("-")
            for j in range(len(point)):
                plt.scatter(
                    X[diz[point[j]], 0],
                    X[diz[point[j]], 1],
                    s=350,
                    color=colors[ind % 18],
                )
            point = [diz[point[i]] for i in range(len(point))]
            coms.append(X[point].mean(axis=0))

        # variations of red to plot the representative points of the various clusters
        colors_reps = [
            "red",
            "crimson",
            "indianred",
            "lightcoral",
            "salmon",
            "darksalmon",
            "firebrick",
        ]

        # flattening the last_reps values
        flat_reps = [
            item for sublist in list(last_reps.values()) for item in sublist
        ]

        # plotting the representatives, surrounded by small circles, and the centers of mass, marked with X
        for i in range(len(last_reps)):
            len_rep = len(list(last_reps.values())[i])

            x = [
                list(last_reps.values())[i][j][0]
                for j in range(min(n_rep_fin, len_rep))
            ]
            y = [
                list(last_reps.values())[i][j][1]
                for j in range(min(n_rep_fin, len_rep))
            ]

            plt.scatter(
                x, y, s=400, color=colors_reps[i % 7], edgecolor="black"
            )
            plt.scatter(
                coms[i][0],
                coms[i][1],
                s=400,
                color=colors_reps[i % 7],
                marker="X",
                edgecolor="black",
            )

            for num in range(min(n_rep_fin, len_rep)):
                plt.gcf().gca().add_artist(
                    plt.Circle(
                        (x[num], y[num]),
                        xwidth * 0.03,
                        color=colors_reps[i % 7],
                        fill=False,
                        linewidth=3,
                        alpha=0.7,
                    )
                )

            plt.scatter(
                not_sampled[:, 0],
                not_sampled[:, 1],
                s=400,
                color="lime",
                edgecolor="black",
            )

        # find the closest representative for not sampled points, and draw an arrow connecting the points
        # to its closest representative
        for ind in range(len(not_sampled)):
            dist_int = []
            for el in flat_reps:
                dist_int.append(dist1(not_sampled[ind], el))
            ind_min = np.argmin(dist_int)

            plt.arrow(
                not_sampled[ind][0],
                not_sampled[ind][1],
                flat_reps[ind_min][0] - not_sampled[ind][0],
                flat_reps[ind_min][1] - not_sampled[ind][1],
                length_includes_head=True,
                head_width=0.03,
                head_length=0.05,
            )

        # plotting the indexes for each point
        for i, txt in enumerate(initial_ind):
            ax.annotate(
                txt,
                (X[:, 0][i], X[:, 1][i]),
                fontsize=10,
                size=10,
                ha="center",
                va="center",
            )

        if not_sampled_ind is not None:
            for i, txt in enumerate(not_sampled_ind):
                ax.annotate(
                    txt,
                    (not_sampled[:, 0][i], not_sampled[:, 1][i]),
                    fontsize=10,
                    size=10,
                    ha="center",
                    va="center",
                )

        plt.show()

    # if par_index is not None, diz is updated with the last merged cluster and its keys are returned
    if par_index is not None:
        diz["(" + u + ")" + "-" + "(" + u_cl + ")"] = len(diz)
        list_keys_diz = list(diz.keys())

        return list_keys_diz


def dist_clust_cure(rep_u, rep_v):
    """
    Compute the distance of two clusters based on the minimum distance found between the
    representatives of one cluster and the ones of the other.

    :param rep_u: list of representatives of the first cluster
    :param rep_v: list of representatives of the second cluster
    :return: distance between two clusters
    """
    rep_u = np.array(rep_u)
    rep_v = np.array(rep_v)
    distances = []
    for i in rep_u:
        for j in rep_v:
            distances.append(dist1(i, j))
    return np.min(distances)


def update_mat_cure(mat, i, j, rep_new, name):
    """
    Update distance matrix of CURE, by computing the new distances from the new representatives.

    :param mat: input dataframe built by CURE algorithm, listing the cluster and the x and y
                coordinates of each point.
    :param i: row index of cluster to be merged.
    :param j: column index of cluster to be merged.
    :param rep_new: dictionary of new representatives.
    :param name: string of the form "(" + u + ")" + "-" + "(" + u_cl + ")", containing the new
                 name of the newly merged cluster.
    :return mat: updated matrix with new distances
    """
    # taking the 2 rows to be updated
    a1 = mat.loc[i]
    b1 = mat.loc[j]

    key_lists = list(rep_new.keys())

    # update all distances from the new cluster with new representatives
    vec = []
    for i in range(len(mat)):
        vec.append(dist_clust_cure(rep_new[name], rep_new[key_lists[i]]))

    # adding new row
    mat.loc["(" + a1.name + ")" + "-" + "(" + b1.name + ")", :] = vec
    # adding new column
    mat["(" + a1.name + ")" + "-" + "(" + b1.name + ")"] = vec + [np.inf]

    # dropping the old row and the old column
    mat = mat.drop([a1.name, b1.name], 0)
    mat = mat.drop([a1.name, b1.name], 1)

    return mat


def sel_rep(clusters, name, c, alpha):
    """
    Select c representatives of the clusters: first one is the farthest from the centroid,
    the others c-1 are the farthest from the already selected representatives. It doesn't use
    the old representatives, so it is slower than sel_rep_fast.

    :param clusters: dictionary of clusters.
    :param name: name of the cluster we want to select representatives from.
    :param c: number of representatives we want to extract.
    :param alpha: 0<=float<=1, it determines how much the representative points are moved
                 toward the centroid: 0 means they aren't modified, 1 means that all points
                 collapse to the centroid.
    :return others: list of representative points.
    """

    # if the cluster has c points or less, just take all of them as representatives and shrink them
    # according to the parameter alpha
    if len(clusters[name]) <= c:

        others = clusters[name]
        com = np.mean(clusters[name], axis=0)

        for i in range(len(others)):
            others[i] = others[i] + alpha * (com - others[i])

        return others

    # if the cluster has more than c points, use the procedure described in the documentation to pick
    # the representative points
    else:

        others = []  # the representatives
        indexes = (
            []
        )  # their indexes, to avoid picking one point multiple times

        points = clusters[name]
        com = np.mean(points, axis=0)

        # compute distances from the centroid
        distances_com = {i: dist1(points[i], com) for i in range(len(points))}
        index = max(distances_com, key=distances_com.get)

        indexes.append(index)
        others.append(
            np.array(points[index])
        )  # first point is the farthest from the centroid

        # selecting the other c-1 points
        for step in range(min(c - 1, len(points) - 1)):
            # here we store the distances of the current point from the alredy selected representatives
            partial_distances = {str(i): [] for i in range(len(points))}
            for i in range(len(points)):
                if i not in indexes:
                    for k in range(len(others)):
                        partial_distances[str(i)].append(
                            [dist1(points[i], np.array(others[k]))]
                        )
            partial_distances = dict(
                (k, [np.sum(v)]) for k, v in partial_distances.items()
            )
            index2 = max(partial_distances, key=partial_distances.get)
            indexes.append(int(index2))
            others.append(
                points[int(index2)]
            )  # other points are the farthest from the already selected representatives

        # perform the shrinking according to the parameter alpha
        for i in range(len(others)):
            others[i] = others[i] + alpha * (com - others[i])

        return others


def sel_rep_fast(prec_reps, clusters, name, c, alpha):
    """
    Select c representatives of the clusters from the previously computed representatives,
    so it is faster than sel_rep.

    :param prec_reps: list of previously computed representatives.
    :param clusters: dictionary of clusters.
    :param name: name of the cluster we want to select representatives from.
    :param c: number of representatives we want to extract.
    :param alpha: 0<=float<=1, it determines how much the representative points are moved
                 toward the centroid: 0 means they aren't modified, 1 means that all points
                 collapse to the centroid.
    :return others: list of representative points.
    """

    com = np.mean(clusters[name], axis=0)

    # if the cluster has c points or less, just take all of them as representatives and shrink them
    # according to the parameter alpha
    if len(prec_reps) <= c:

        others = prec_reps
        for i in range(len(others)):
            others[i] = others[i] + alpha * (com - others[i])

        return others

    # if the cluster has more than c points, use the procedure described in the documentation to pick
    # the representative points
    else:

        others = []  # the representatives
        indexes = (
            []
        )  # their indexes, to avoid picking one point multiple times

        points = prec_reps  # use old representatives

        distances_com = {i: dist1(points[i], com) for i in range(len(points))}
        index = max(distances_com, key=distances_com.get)

        indexes.append(index)
        others.append(np.array(points[index]))  # first point

        # selecting the other c-1 points
        for step in range(min(c - 1, len(points) - 1)):
            # here we store the distances of the current point from the alredy selected representatives
            partial_distances = {str(i): [] for i in range(len(points))}
            for i in range(len(points)):
                if i not in indexes:
                    for k in range(len(others)):
                        partial_distances[str(i)].append(
                            [dist1(points[i], np.array(others[k]))]
                        )
            partial_distances = dict(
                (k, [np.sum(v)]) for k, v in partial_distances.items()
            )
            index2 = max(partial_distances, key=partial_distances.get)
            indexes.append(int(index2))
            others.append(
                points[int(index2)]
            )  # other points are the farthest from the already selected representatives

        # perform the shrinking according to the parameter alpha
        for i in range(len(others)):
            others[i] = others[i] + alpha * (com - others[i])

        return others


def cure(
    X,
    k,
    c=3,
    alpha=0.1,
    plotting=True,
    preprocessed_data=None,
    partial_index=None,
    n_rep_finalclust=None,
    not_sampled=None,
    not_sampled_ind=None,
):
    """
    CURE algorithm: hierarchical agglomerative clustering using representatives.

    :param X: input data array.
    :param k: desired number of clusters.
    :param c: number of representatives for each cluster.
    :param alpha: parameter that regulates the shrinking of representative points toward the centroid.
    :param plotting: if True, plots all intermediate steps.

    #the following parameter are used for the large dataset variation of CURE

    :param preprocessed_data: if not None, must be of the form (clusters,representatives,matrix_a,X_dist1),
                              which is used to perform a warm start.
    :param partial_index: if not None, is is used as index of the matrix_a, of cluster points and of
                          representatives.
    :param n_rep_finalclust: the final representative points used to classify the not_sampled points.
    :param not_sampled: points not sampled in the initial phase.
    :param not_sampled_ind: indexes of not_sampled points.
    :return (clusters, rep, a): returns the clusters dictionary, the dictionary of representatives,
                                the matrix a


    """

    # starting from raw data
    if preprocessed_data is None:
        # building a dataframe storing the x and y coordinates of input data points
        l = [[i, i] for i in range(len(X))]
        flat_list = [item for sublist in l for item in sublist]
        col = [
            str(el) + "x" if i % 2 == 0 else str(el) + "y"
            for i, el in enumerate(flat_list)
        ]

        # using the original indexes if necessary
        if partial_index is not None:
            a = pd.DataFrame(index=partial_index, columns=col)
        else:
            a = pd.DataFrame(
                index=[str(i) for i in range(len(X))], columns=col
            )

        # adding the real coordinates
        a["0x"] = X.T[0]
        a["0y"] = X.T[1]

        b = a.dropna(axis=1, how="all")

        # initial clusters
        if partial_index is not None:
            clusters = dict(zip(partial_index, X))
        else:
            clusters = {str(i): np.array(X[i]) for i in range(len(X))}

        # build Xdist
        X_dist1 = dist_mat_gen(b)

        # initialize representatives
        if partial_index is not None:
            rep = {partial_index[i]: [X[int(i)]] for i in range(len(X))}
        else:
            rep = {str(i): [X[i]] for i in range(len(X))}

        # just as placeholder for while loop
        heap = [1] * len(X_dist1)

        # store minimum distances between clusters for each iteration
        levels = []

    # use precomputed data
    else:

        clusters = preprocessed_data[0]
        rep = preprocessed_data[1]
        a = preprocessed_data[2]
        X_dist1 = preprocessed_data[3]
        heap = [1] * len(X_dist1)
        levels = []

    # store original index
    if partial_index is not None:
        initial_index = deepcopy(partial_index)

    # while the desired number of clusters has not been reached
    while len(heap) > k:

        # find minimum value of heap queue, which stores clusters according to the distance from
        # their closest cluster

        list_argmin = list(X_dist1.apply(lambda x: np.argmin(x)).values)
        list_min = list(X_dist1.min(axis=0).values)
        heap = dict(zip(list(X_dist1.index), list_min))
        heap = dict(OrderedDict(sorted(heap.items(), key=lambda kv: kv[1])))
        closest = dict(zip(list(X_dist1.index), list_argmin))

        # get minimum keys and delete them from heap and closest dictionaries
        u = min(heap, key=heap.get)
        levels.append(heap[u])
        del heap[u]
        # u_cl = str(closest[u])
        u_cl = X_dist1.columns[closest[u]]
        del closest[u]

        # form the new cluster
        if (np.array(clusters[u]).shape == (2,)) and (
            np.array(clusters[u_cl]).shape == (2,)
        ):
            w = [clusters[u], clusters[u_cl]]
        elif (np.array(clusters[u]).shape != (2,)) and (
            np.array(clusters[u_cl]).shape == (2,)
        ):
            clusters[u].append(clusters[u_cl])
            w = clusters[u]
        elif (np.array(clusters[u]).shape == (2,)) and (
            np.array(clusters[u_cl]).shape != (2,)
        ):
            clusters[u_cl].append(clusters[u])
            w = clusters[u_cl]
        else:
            w = clusters[u] + clusters[u_cl]

        # delete old cluster
        del clusters[u]
        del clusters[u_cl]

        # set new name
        name = "(" + u + ")" + "-" + "(" + u_cl + ")"
        clusters[name] = w

        # update representatives
        rep[name] = sel_rep_fast(rep[u] + rep[u_cl], clusters, name, c, alpha)

        # update distance matrix
        X_dist1 = update_mat_cure(X_dist1, u, u_cl, rep, name)

        # delete old representatives
        del rep[u]
        del rep[u_cl]

        dim1 = int(a.loc[u].notna().sum())
        # update the matrix a with the new cluster
        a.loc["(" + u + ")" + "-" + "(" + u_cl + ")", :] = a.loc[u].fillna(
            0
        ) + a.loc[u_cl].shift(dim1, fill_value=0)
        a = a.drop(u, 0)
        a = a.drop(u_cl, 0)

        if plotting is True:

            # in the large dataset version of CURE
            if partial_index is not None:

                # only in last step of large dataset version of CURE
                if (
                    (len(heap) == k)
                    and (not_sampled is not None)
                    and (not_sampled_ind is not None)
                ):

                    # take random representative points from the final representatives
                    final_reps = {
                        list(rep.keys())[i]: random.sample(
                            list(rep.values())[i],
                            min(n_rep_finalclust, len(list(rep.values())[i])),
                        )
                        for i in range(len(rep))
                    }

                    partial_index = point_plot_mod2(
                        X=X,
                        a=a,
                        reps=rep[name],
                        level_txt=levels[-1],
                        par_index=partial_index,
                        u=u,
                        u_cl=u_cl,
                        initial_ind=initial_index,
                        last_reps=final_reps,
                        not_sampled=not_sampled,
                        not_sampled_ind=not_sampled_ind,
                        n_rep_fin=n_rep_finalclust,
                    )

                # in the intermediate steps of the large dataset version
                else:
                    partial_index = point_plot_mod2(
                        X=X,
                        a=a,
                        reps=rep[name],
                        level_txt=levels[-1],
                        par_index=partial_index,
                        u=u,
                        u_cl=u_cl,
                        initial_ind=initial_index,
                    )
            else:
                point_plot_mod2(X, a, rep[name], levels[-1])

    return clusters, rep, a


def plot_results_cure(clust):
    """
    Scatter plot of data points, colored according to the cluster they belong to, after performing
    CURE algorithm.

    :param clust: output of CURE algorithm, dictionary of the form cluster_labels+point_indices: coords of points
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    cl_list = []
    for num_clust in range(len(clust)):
        cl_list.append(np.array(clust[list(clust.keys())[num_clust]]))
        try:
            plt.scatter(cl_list[-1][:, 0], cl_list[-1][:, 1], s=300)
        except:
            plt.scatter(cl_list[-1][0], cl_list[-1][1], s=300)
    plt.show()


def Chernoff_Bounds(u_min, f, N, d, k):
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


def dist_mat_gen_cure(dictionary):
    """
    Build distance matrix for CURE algorithm, using the dictionary of representatives.

    :param dictionary: dictionary of representative points, the only ones used to compute distances
                       between clusters.
    :return D: distance matrix as dataframe

    """
    D = pd.DataFrame()
    ind = list(dictionary.keys())
    k = 0
    for i in ind:
        for j in ind[k:]:
            if i != j:

                a = dictionary[i]
                b = dictionary[j]

                D.loc[i, j] = dist_clust_cure(a, b)
                D.loc[j, i] = D.loc[i, j]
            else:

                D.loc[i, j] = np.inf

        k += 1

    D = D.fillna(np.inf)

    return D


def cure_sample_part(
    X,
    k,
    c=3,
    alpha=0.3,
    u_min=None,
    f=0.3,
    d=0.02,
    p=None,
    q=None,
    n_rep_finalclust=None,
    plotting=True,
):
    """
    CURE algorithm variation for large datasets.
    Partition the sample space into p partitions, each of size len(X)/p, then partially cluster each
    partition until the final number of clusters in each partition reduces to n/(pq). Then run a second
    clustering pass on the n/q partial clusters for all the partitions.

    :param X: input data array.
    :param k: desired number of clusters.
    :param c: number of representatives for each cluster.
    :param alpha: parameter that regulates the shrinking of representative points toward the centroid.
    :param u_min: size of the smallest cluster u.
    :param f: percentage of cluster points (0 <= f <= 1) we would like to have in the sample.
    :param d: (0 <= d <= 1) the probability that the sample contains less than f*|u| points of cluster u is less than d.
    :param p: the number of partitions.
    :param q: the number >1 such that each partition reduces to n/(pq) clusters.
    :param n_rep_finalclust: number of representatives to use in the final assignment phase.
    :param plotting: if True, plots all intermediate steps.
    :return (clusters, rep, mat_a): returns the clusters dictionary, the dictionary of representatives,
                                the matrix a
    """

    # choose the parameters suggested by the paper if the user doesnt provide input parameters
    if u_min is None:
        u_min = round(len(X) / k)

    if n_rep_finalclust is None:
        n_rep_finalclust = c

    l = [[i, i] for i in range(len(X))]
    flat_list = [item for sublist in l for item in sublist]
    col = [
        str(el) + "x" if i % 2 == 0 else str(el) + "y"
        for i, el in enumerate(flat_list)
    ]
    a = pd.DataFrame(index=[str(i) for i in range(len(X))], columns=col)
    a["0x"] = X.T[0]
    a["0y"] = X.T[1]
    b = a.dropna(axis=1, how="all")

    # this is done to ensure that the algorithm starts even when input params are bad
    while True:
        try:
            print("new f: ", f)
            print("new d: ", d)
            n = math.ceil(
                Chernoff_Bounds(u_min=u_min, f=f, N=len(X), k=k, d=d)
            )
            b_sampled = b.sample(n, random_state=42)
            break
        except:
            if f >= 0.19:
                f = f - 0.1
            else:
                d = d * 2

    b_notsampled = b.loc[
        [str(i) for i in range(len(b)) if str(i) not in b_sampled.index], :
    ]

    # find the best p and q according to the paper
    if (p is None) and (q is None):

        def g(x):
            res = (x[1] - 1) / (x[0] * x[1]) + 1 / (x[1] ** 2)
            return res

        results = {}
        for i in range(2, 15):
            for j in range(2, 15):
                results[(i, j)] = g([i, j])
        p, q = max(results, key=results.get)
        print("p: ", p)
        print("q: ", q)

    if (n / (p * q)) < 2 * k:
        print("n/pq is less than 2k, results could be wrong")

    # form the partitions
    lin_sp = np.linspace(0, n, p + 1, dtype="int")
    # lin_sp
    b_partitions = []
    for num_p in range(p):
        try:
            b_partitions.append(
                b_sampled.iloc[lin_sp[num_p] : lin_sp[num_p + 1]]
            )
        except:
            b_partitions.append(b_sampled.iloc[lin_sp[num_p] :])

    k_prov = round(n / (p * q))

    # perform clustering on each partition separately
    partial_clust1 = []
    partial_rep1 = []
    partial_a1 = []

    for i in range(p):
        print("\n")
        print(i)
        clusters, rep, mat_a = cure(
            b_partitions[i].values,
            k=k_prov,
            c=c,
            alpha=alpha,
            plotting=plotting,
            partial_index=b_partitions[i].index,
        )
        partial_clust1.append(clusters)
        partial_rep1.append(rep)
        partial_a1.append(mat_a)

    # merging all data into single components
    # clusters
    clust_tot = {}
    for d in partial_clust1:
        clust_tot.update(d)
    # representatives
    rep_tot = {}
    for d in partial_rep1:
        rep_tot.update(d)
    # mat a
    diz = {i: len(b_partitions[i]) for i in range(p)}
    num_freq = Counter(diz.values()).most_common(1)[0][0]
    bad_ind = [
        list(diz.keys())[i] for i in range(len(diz)) if diz[i] != num_freq
    ]

    for ind in bad_ind:
        partial_a1[ind]["{0}x".format(diz[ind])] = [np.nan] * k_prov
        partial_a1[ind]["{0}y".format(diz[ind])] = [np.nan] * k_prov

    for i in range(len(partial_a1) - 1):
        if i == 0:
            a_tot = partial_a1[i].append(partial_a1[i + 1])
        else:
            a_tot = a_tot.append(partial_a1[i + 1])
    # mat Xdist
    X_dist_tot = dist_mat_gen_cure(rep_tot)

    # final_clustering
    prep_data = [clust_tot, rep_tot, a_tot, X_dist_tot]
    clusters, rep, mat_a = cure(
        b_sampled.values,
        k=k,
        c=c,
        alpha=alpha,
        preprocessed_data=prep_data,
        partial_index=b_sampled.index,
        n_rep_finalclust=n_rep_finalclust,
        not_sampled=b_notsampled.values,
        plotting=plotting,
        not_sampled_ind=b_notsampled.index,
    )

    return clusters, rep, mat_a


def demo_parameters():
    """Four plots showing the effects on the sample size of various parameters"""

    plt.figure(figsize=(12, 10))
    plt.suptitle("Effects on sample size from different parameters")

    ax0 = plt.subplot(2, 2, 1)
    # plt.plot(d, k*res)
    u_size = 6000
    f = 0.20
    N = 20000
    k = 4
    d = np.linspace(0.0000001, 1, 100)
    ax0.set_title("u_min: {0}, f:{1}, k:{2}".format(u_size, f, k))
    res = k * (
        f * N
        + N / u_size * np.log(1 / d)
        + N
        / u_size
        * np.sqrt(np.log(1 / d) ** 2 + 2 * f * u_size * np.log(1 / d))
    )
    plt.axhline(N, color="r")
    plt.plot(d, res)
    plt.xlabel("d")
    plt.ylabel("sample size")

    ax1 = plt.subplot(2, 2, 2)

    u_size = 3000
    f = 0.2
    N = 20000
    d = 0.1
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ax1.set_title("u_min: {0}, f:{1}, d:{2}".format(u_size, f, d))
    res = [
        k[i]
        * (
            f * N
            + N / u_size * np.log(1 / d)
            + N
            / u_size
            * np.sqrt(np.log(1 / d) ** 2 + 2 * f * u_size * np.log(1 / d))
        )
        for i in range(len(k))
    ]
    plt.axhline(N, color="r")
    plt.plot(k, res)
    plt.xlabel("k")

    ax2 = plt.subplot(2, 2, 3)

    u_size = 5000
    f = np.linspace(0.00001, 1, 100)
    N = 20000
    d = 0.1
    k = 4
    ax2.set_title("u_min: {0}, d:{1}, k:{2}".format(u_size, d, k))
    res = k * (
        f * N
        + N / u_size * np.log(1 / d)
        + N
        / u_size
        * np.sqrt(np.log(1 / d) ** 2 + 2 * f * u_size * np.log(1 / d))
    )
    plt.axhline(N, color="r")
    plt.plot(f, res)
    plt.xlabel("f")
    plt.ylabel("sample size")

    ax3 = plt.subplot(2, 2, 4)

    u_size = np.linspace(200, 10000, 30)
    f = 0.2
    N = 20000
    d = 0.1
    k = 4
    ax3.set_title("f: {0}, d:{1}, k:{2}".format(f, d, k))
    res = k * (
        f * N
        + N / u_size * np.log(1 / d)
        + N
        / u_size
        * np.sqrt(np.log(1 / d) ** 2 + 2 * f * u_size * np.log(1 / d))
    )
    plt.axhline(N, color="r")
    plt.plot(u_size, res)
    plt.xlabel("u_min")

    plt.show()
