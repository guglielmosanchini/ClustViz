import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.optics import dist1
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from matplotlib import colors


def encircle(x, y, ax, **kw):
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)
    ax.add_patch(poly)


def convert_colors(dict_colors, alpha=0.5):
    new_dict_colors = {}

    for i, col in enumerate(dict_colors.values()):
        new_dict_colors[i] = tuple(list(colors.to_rgb(col)) + [alpha])

    return new_dict_colors


def update_mat(mat, i, j, linkage):
    """
    Updates the input distance matrix in the position (i,j), according to the provided
    linkage method.

    :param mat: input matrix as dataframe.
    :param i: row index.
    :param j: column indexes.
    :param linkage: linkage method; can be single, complete, average or ward.
    :return mat: updated matrix as dataframe.

    """

    a1 = mat.iloc[i]
    b1 = mat.iloc[j]

    if linkage == "single":

        vec = [np.min([p, q]) for p, q in zip(a1.values, b1.values)]
        vec[i] = np.inf
        vec[j] = np.inf

    elif linkage == "complete":

        vec = [np.max([p, q]) for p, q in zip(a1.values, b1.values)]

    elif linkage == "average":

        l_a1 = len(a1.name.replace("(", "").replace(")", "").split("-"))
        l_b1 = len(b1.name.replace("(", "").replace(")", "").split("-"))
        vec = [
            (l_a1 * a1[k] + l_b1 * b1[k]) / (l_a1 + l_b1)
            for k in range(len(a1))
        ]

    mat.loc["(" + a1.name + ")" + "-" + "(" + b1.name + ")", :] = vec
    mat["(" + a1.name + ")" + "-" + "(" + b1.name + ")"] = vec + [np.inf]

    mat = mat.drop([a1.name, b1.name], 0)
    mat = mat.drop([a1.name, b1.name], 1)

    return mat


def point_plot_mod(X, a, level_txt, level2_txt=None):
    """
    Scatter plot of data points, colored according to the cluster they belong to. The most recently
    merged cluster is enclosed in a rectangle of the same color as its points, with red borders.
    In the top right corner, the total distance is shown, along with the current number of clusters.
    When using Ward linkage, also the increment in distance is shown.

    :param X: input data as array.
    :param a: distance matrix built by agg_clust/agg_clust_mod.
    :param level_txt: dist_tot displayed.
    :param level2_txt: dist_incr displayed.
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    plt.scatter(X[:, 0], X[:, 1], s=300, color="lime", edgecolor="black")

    a = a.dropna(1, how="all")

    color_dict = {
        0: "seagreen",
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
        12: "plum",
        13: "red",
        14: "lightblue",
        15: "khaki",
        16: "gainsboro",
        17: "peachpuff",
    }

    color_dict_rect = convert_colors(color_dict, alpha=0.3)

    len_ind = [len(i.split("-")) for i in list(a.index)]
    start = np.min([i for i in range(len(len_ind)) if len_ind[i] > 1])

    for ind, i in enumerate(range(start, len(a))):
        point = a.iloc[i].name.replace("(", "").replace(")", "").split("-")
        point = [int(i) for i in point]

        X_clust = [X[point[j], 0] for j in range(len(point))]
        Y_clust = [X[point[j], 1] for j in range(len(point))]

        plt.scatter(X_clust, Y_clust, s=350, color=color_dict[ind % 17])

    point = a.iloc[-1].name.replace("(", "").replace(")", "").split("-")
    point = [int(i) for i in point]
    rect_min = X[point].min(axis=0)
    rect_diff = X[point].max(axis=0) - rect_min

    xmin, xmax, ymin, ymax = plt.axis()
    xwidth = xmax - xmin
    ywidth = ymax - ymin

    if len(X_clust) <= 2:

        ax.add_patch(
            Rectangle(
                (rect_min[0] - xwidth * 0.02, rect_min[1] - ywidth * 0.04),
                rect_diff[0] + xwidth * 0.04,
                rect_diff[1] + ywidth * 0.08,
                fill=True,
                color=color_dict_rect[ind % 17],
                linewidth=3,
                ec="red",
            )
        )
    else:
        encircle(
            X_clust,
            Y_clust,
            ax=ax,
            color=color_dict_rect[ind % 17],
            linewidth=3,
            ec="red",
        )

    for i, txt in enumerate([i for i in range(len(X))]):
        ax.annotate(
            txt,
            (X[:, 0][i], X[:, 1][i]),
            fontsize=10,
            size=10,
            ha="center",
            va="center",
        )

    ax.annotate(
        "dist_tot: " + str(round(level_txt, 5)),
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


def dist_mat(df, linkage):
    """
    Takes as input the dataframe created by agg_clust/agg_clust_mod and outputs
    the distance matrix; it is actually an upper triangular matrix, the symmetrical
    values are replaced with  np.inf.

    :param df: input dataframe, with first column corresponding to x-coordinates and
               second column corresponding to y-coordinates of data points.
    :param linkage: linkage method; can be single, complete, average or ward.
    :return D: distance matrix.

    """

    even_num = [i for i in range(2, len(df) + 1) if i % 2 == 0]
    D = pd.DataFrame()
    ind = list(df.index)
    k = 0
    for i in ind:
        for j in ind[k:]:
            if i != j:

                a = df.loc[i].values
                b = df.loc[j].values
                z1 = [i for i in even_num if i <= len(a)]
                z2 = [i for i in even_num if i <= len(b)]
                a = [a[: z1[0]]] + [
                    a[z1[i]: z1[i + 1]] for i in range(len(z1) - 1)
                ]
                b = [b[: z2[0]]] + [
                    b[z2[i]: z2[i + 1]] for i in range(len(z2) - 1)
                ]

                if linkage == "single":
                    D.loc[i, j] = sl_dist(a, b)
                elif linkage == "complete":
                    D.loc[i, j] = cl_dist(a, b)
                elif linkage == "average":
                    D.loc[i, j] = avg_dist(a, b)
            else:

                D.loc[i, j] = np.inf

        k += 1

    D = D.fillna(np.inf)

    return D


# DEPRECATED
def dist_mat_full(df, linkage):
    """Variation of dist_mat, outputs the full distance matrix instead of an upper triangular one"""

    even_num = [i for i in range(2, len(df) + 1) if i % 2 == 0]
    D = pd.DataFrame()
    ind = list(df.index)
    k = 0
    for i in ind:
        for j in ind[k:]:
            if i != j:

                a = df.loc[i].values
                b = df.loc[j].values
                z1 = [i for i in even_num if i <= len(a)]
                z2 = [i for i in even_num if i <= len(b)]
                a = [a[: z1[0]]] + [
                    a[z1[i]: z1[i + 1]] for i in range(len(z1) - 1)
                ]
                b = [b[: z2[0]]] + [
                    b[z2[i]: z2[i + 1]] for i in range(len(z2) - 1)
                ]

                if linkage == "single":
                    D.loc[i, j] = sl_dist(a, b)
                    D.loc[j, i] = sl_dist(a, b)
                elif linkage == "complete":
                    D.loc[i, j] = cl_dist(a, b)
                    D.loc[j, i] = cl_dist(a, b)
                elif linkage == "average":
                    D.loc[i, j] = avg_dist(a, b)
                    D.loc[j, i] = avg_dist(a, b)
            else:

                D.loc[i, j] = np.inf

        k += 1

    D = D.fillna(np.inf)

    return D


def dist_mat_gen(df):
    """Variation of dist_mat, uses only single_linkage method"""

    even_num = [i for i in range(2, len(df) + 1) if i % 2 == 0]
    D = pd.DataFrame()
    ind = list(df.index)
    k = 0
    for i in ind:
        for j in ind[k:]:
            if i != j:

                a = df.loc[i].values
                b = df.loc[j].values
                z1 = [i for i in even_num if i <= len(a)]
                z2 = [i for i in even_num if i <= len(b)]
                a = [a[: z1[0]]] + [
                    a[z1[i]: z1[i + 1]] for i in range(len(z1) - 1)
                ]
                b = [b[: z2[0]]] + [
                    b[z2[i]: z2[i + 1]] for i in range(len(z2) - 1)
                ]

                D.loc[i, j] = sl_dist(a, b)
                D.loc[j, i] = sl_dist(a, b)
            else:

                D.loc[i, j] = np.inf

        k += 1

    D = D.fillna(np.inf)

    return D


def compute_var(X, df):
    """
    Compute total intra-cluster variance of the cluster configuration inferred from df.

    :param X: input data as array.
    :param df: input dataframe built by agg_clust/agg_clust_mod, listing the cluster and the x and y
                coordinates of each point.
    :return: centroids dataframe with their coordinates and the single variances of the corresponding
             clusters, and the total intra-cluster variance.
    """

    cleaned_index = [
        i.replace("(", "").replace(")", "").split("-") for i in df.index
    ]
    cent_x_tot = []
    for li in cleaned_index:
        cent_x = []
        for el in li:
            cent_x.append(X[int(el)][0])
        cent_x_tot.append(np.mean(cent_x))
    cent_y_tot = []
    for li in cleaned_index:
        cent_y = []
        for el in li:
            cent_y.append(X[int(el)][1])
        cent_y_tot.append(np.mean(cent_y))

    centroids = pd.DataFrame(index=df.index)
    centroids["cx"] = cent_x_tot
    centroids["cy"] = cent_y_tot

    var_int = compute_var_sing(df, centroids)

    centroids["var"] = var_int

    return centroids, centroids["var"].sum()


def compute_var_sing(df, centroids):
    """
    Compute every internal variance in clusters; clusters are found in df,
    whereas centroids are saved in centroids.

    :param df:  input dataframe built by agg_clust/agg_clust_mod, listing the cluster and the x and y
                coordinates of each point.
    :param centroids: dataframe of the centroids of clusters, with their x and y coordinates.
    :return var_int: list of intra-cluster variances.

    """
    even_num = [i for i in range(2, len(df) + 1) if i % 2 == 0]
    var_int = []
    for i in list(df.index):
        az = df.loc[i].values
        z1 = [i for i in even_num if i <= len(az)]
        az = [az[: z1[0]]] + [
            az[z1[i]: z1[i + 1]] for i in range(len(z1) - 1)
        ]
        az = [az[i] for i in range(len(az)) if np.isinf(az[i]).sum() != 2]

        internal_dist = []
        for el in az:
            distance = (dist1(el, centroids.loc[i, ["cx", "cy"]].values)) ** 2
            internal_dist.append(distance)
        var_int.append(np.sum(internal_dist))

    return var_int


def compute_ward_ij(data, df):
    """
    Compute difference in total within-cluster variance, with squared euclidean
    distance, and finds the best cluster according to Ward criterion.

    :param data: input data array.
    :param df:  input dataframe built by agg_clust/agg_clust_mod, listing the cluster and the x and y
                coordinates of each point.
    :return: (i,j) indices of best cluster (the one for which the increase in intra-cluster variance is minimum)
             new_summ: new total intra-cluster variance
             par_var: increment in total intra-cluster variance, i.e. minimum increase in total intra-cluster variance
    """

    even_num = [i for i in range(2, len(data) + 1) if i % 2 == 0]

    (centroids, summ) = compute_var(data, df)
    variances = {}
    k = 0
    ind = list(df.index)

    partial_var = {}

    for i in ind:
        for j in ind[k:]:
            if i != j:
                az = df.loc[i].values
                bz = df.loc[j].values
                z1 = [i for i in even_num if i <= len(az)]
                z2 = [i for i in even_num if i <= len(bz)]
                az = [az[: z1[0]]] + [
                    az[z1[i]: z1[i + 1]] for i in range(len(z1) - 1)
                ]
                bz = [bz[: z2[0]]] + [
                    bz[z2[i]: z2[i + 1]] for i in range(len(z2) - 1)
                ]
                d = az + bz
                valid = [
                    d[i] for i in range(len(d)) if np.isinf(d[i]).sum() != 2
                ]
                # print(valid)
                centroid = np.mean(valid, axis=0)
                var_int_par = []
                for el in valid:
                    var_int_par.append(dist1(el, centroid) ** 2)
                var_intz = np.sum(var_int_par)
                partial_var[(i, j)] = (
                        var_intz
                        - centroids.loc[i]["var"]
                        - centroids.loc[j]["var"]
                )

                var_new = summ + partial_var[(i, j)]
                variances[(i, j)] = var_new
        k += 1

    (i, j) = min(variances, key=variances.get)
    new_summ = np.min(list(variances.values()))
    par_var = partial_var[(i, j)]
    if new_summ == summ:
        print("wrong")

    return (i, j), new_summ, par_var


def sl_dist(a, b):
    """Distance for single_linkage method, i.e. min[dist(x,y)] for x in a & y in b """
    distances = []
    for i in a:
        for j in b:
            distances.append(dist1(i, j))
    distances = [i for i in distances if np.isnan(i) == False]
    return np.min(distances)


def cl_dist(a, b):
    """Distance for complete_linkage method, i.e. max[dist(x,y)] for x in a & y in b """
    distances = []
    for i in a:
        for j in b:
            distances.append(dist1(i, j))
    distances = [
        i
        for i in distances
        if (np.isnan(i) == False) and (np.isinf(i) == False)
    ]
    return np.max(distances)


def avg_dist(a, b):
    """Distance for average_linkage method, i.e. mean[dist(x,y)] for x in a & y in b """
    distances = []
    for i in a:
        for j in b:
            distances.append(dist1(i, j))
    distances = [
        i
        for i in distances
        if (np.isnan(i) == False) and (np.isinf(i) == False)
    ]
    return np.mean(distances)


# DEPRECATED, agg_clust_mod is faster
def agg_clust(X, linkage):
    """
    Perform hierarchical agglomerative clustering with the provided linkage method, plotting every step
    of cluster aggregation.

    :param X: input data array
    :param linkage: linkage method; can be single, complete, average or ward.
    """
    levels = []
    levels2 = []
    ind_list = []

    l = [[i, i] for i in range(len(X))]
    flat_list = [item for sublist in l for item in sublist]
    col = [
        str(el) + "x" if i % 2 == 0 else str(el) + "y"
        for i, el in enumerate(flat_list)
    ]

    a = pd.DataFrame(index=[str(i) for i in range(len(X))], columns=col)

    a["0x"] = X.T[0]
    a["0y"] = X.T[1]

    while len(a) > 1:

        b = a.dropna(axis=1, how="all")

        if (linkage == "single") or (linkage == "average"):

            b = b.fillna(np.inf)

        elif linkage == "complete":

            b = b.fillna(np.NINF)

        elif linkage == "ward":

            b = b.fillna(np.inf)

            if len(a) == len(X):
                var_sum = 0
                levels.append(var_sum)
                levels2.append(var_sum)

        else:
            print("input metric is not valid")
            return

        if linkage != "ward":
            X_dist1 = dist_mat(b, linkage)
            # find indexes of minimum
            (i, j) = np.unravel_index(
                np.array(X_dist1).argmin(), np.array(X_dist1).shape
            )
            levels.append(np.min(np.array(X_dist1)))
            ind_list.append((i, j))
            new_clust = a.iloc[[i, j], :]

        elif linkage == "ward":
            ((i, j), var_sum, par_var) = compute_ward_ij(X, b)

            levels.append(var_sum)
            levels2.append(par_var)
            ind_list.append((i, j))
            new_clust = a.loc[[i, j], :]

        a = a.drop([new_clust.iloc[0].name], 0)
        a = a.drop([new_clust.iloc[1].name], 0)

        dim1 = int(new_clust.iloc[0].notna().sum())

        a.loc[
        "("
        + new_clust.iloc[0].name
        + ")"
        + "-"
        + "("
        + new_clust.iloc[1].name
        + ")",
        :,
        ] = new_clust.iloc[0].fillna(0) + new_clust.iloc[1].shift(
            dim1, fill_value=0
        )

        if linkage != "ward":
            point_plot_mod(X, a, levels[-1])
        else:
            point_plot_mod(X, a, levels[-2], levels2[-1])


def agg_clust_mod(X, linkage):
    """
    Perform hierarchical agglomerative clustering with the provided linkage method, plotting every step
    of cluster aggregation.

    :param X: input data array
    :param linkage: linkage method; can be single, complete, average or ward.
    """

    levels = []
    levels2 = []
    ind_list = []

    # build matrix a, used to store points of clusters with their coordinates
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

    # initial distance matrix
    X_dist1 = dist_mat_gen(b)
    var_sum = 0
    levels.append(var_sum)
    levels2.append(var_sum)

    # until the desired number of clusters is reached
    while len(a) > 1:

        if linkage == "ward":
            # find indexes corresponding to the minimum increase in total intra-cluster variance
            b = a.dropna(axis=1, how="all")
            b = b.fillna(np.inf)
            ((i, j), var_sum, par_var) = compute_ward_ij(X, b)

            levels.append(var_sum)
            levels2.append(par_var)
            ind_list.append((i, j))
            new_clust = a.loc[[i, j], :]

        else:
            # find indexes corresponding to the minimum distance
            (i, j) = np.unravel_index(
                np.array(X_dist1).argmin(), np.array(X_dist1).shape
            )
            levels.append(np.min(np.array(X_dist1)))
            ind_list.append((i, j))
            new_clust = a.iloc[[i, j], :]

            # update distance matrix
            X_dist1 = update_mat(X_dist1, i, j, linkage)

        a = a.drop([new_clust.iloc[0].name], 0)
        a = a.drop([new_clust.iloc[1].name], 0)

        dim1 = int(new_clust.iloc[0].notna().sum())

        new_cluster_name = (
                "("
                + new_clust.iloc[0].name
                + ")"
                + "-"
                + "("
                + new_clust.iloc[1].name
                + ")"
        )

        a.loc[new_cluster_name, :] = new_clust.iloc[0].fillna(
            0
        ) + new_clust.iloc[1].shift(dim1, fill_value=0)

        if linkage != "ward":
            point_plot_mod(X, a, levels[-1])
        else:
            point_plot_mod(X, a, levels[-2], levels2[-1])
