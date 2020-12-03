import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from typing import Tuple, Iterable

from clustviz.utils import convert_colors, encircle, dist1, flatten_list, cluster_points, \
    COLOR_DICT, FONTSIZE_BIGGER, annotate_points


def update_mat(mat: pd.DataFrame, i: int, j: int, linkage: str) -> pd.DataFrame:
    """
    Update the input distance matrix in the position (i, j), according to the provided
    linkage method.

    :param mat: distance dataframe.
    :param i: row index.
    :param j: column index.
    :param linkage: linkage method; can be single, complete, average.
    :return: updated distance dataframe.

    """

    x = mat.iloc[i]
    y = mat.iloc[j]

    if linkage == "single":

        new_distances = [np.min([p, q]) for p, q in zip(x.values, y.values)]
        new_distances[i] = np.inf
        new_distances[j] = np.inf

    elif linkage == "complete":

        new_distances = [np.max([p, q]) for p, q in zip(x.values, y.values)]

    elif linkage == "average":

        x_length = len(cluster_points(x.name))
        y_length = len(cluster_points(y.name))
        new_distances = [
            (x_length * x[k] + y_length * y[k]) / (x_length + y_length)
            for k in range(len(x))
        ]
    else:
        raise ValueError(f'Input linkage parameter {linkage} is invalid. '
                         'Possible linkage parameters: "single", "complete", "average"')

    # create row and column of the new cluster
    new_cluster_name = "(" + x.name + ")-(" + y.name + ")"
    mat.loc[new_cluster_name, :] = new_distances
    mat[new_cluster_name] = new_distances + [np.inf]

    # drop row and column referring to the old cluster
    mat = mat.drop([x.name, y.name], 0)
    mat = mat.drop([x.name, y.name], 1)

    return mat


def point_plot_mod(X: np.ndarray, distance_matrix: pd.DataFrame, level_txt: float, level2_txt: float = None) -> None:
    """
    Scatter plot of data points, colored according to the cluster they belong to. The most recently
    merged cluster is enclosed in a rectangle of the same color as its points, with red borders.
    In the top right corner, the total distance is shown, along with the current number of clusters.
    When using Ward linkage, also the increment in distance is shown.

    :param X: input data as array.
    :param distance_matrix: distance matrix built by agg_clust.
    :param level_txt: dist_tot displayed.
    :param level2_txt: dist_incr displayed.
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.scatter(X[:, 0], X[:, 1], s=300, color="lime", edgecolor="black", zorder=3)

    distance_matrix = distance_matrix.dropna(1, how="all")

    color_dict_rect = convert_colors(COLOR_DICT, alpha=0.3)

    len_ind = [len(i.split("-")) for i in list(distance_matrix.index)]
    start = np.min([i for i in range(len(len_ind)) if len_ind[i] > 1])

    for ind, i in enumerate(range(start, len(distance_matrix))):
        points = cluster_points(distance_matrix.iloc[i].name)
        points = [int(i) for i in points]

        X_clust = [X[points[j], 0] for j in range(len(points))]
        Y_clust = [X[points[j], 1] for j in range(len(points))]

        ax.scatter(X_clust, Y_clust, s=350, color=COLOR_DICT[ind % len(COLOR_DICT)], zorder=3)

    points = cluster_points(distance_matrix.iloc[-1].name)
    points = [int(i) for i in points]
    rect_min = X[points].min(axis=0)
    rect_diff = X[points].max(axis=0) - rect_min

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # xmin, xmax, ymin, ymax = plt.axis()
    xwidth = xmax - xmin
    ywidth = ymax - ymin

    if len(X_clust) <= 2:

        ax.add_patch(
            Rectangle(
                (rect_min[0] - xwidth * 0.02, rect_min[1] - ywidth * 0.04),
                rect_diff[0] + xwidth * 0.04,
                rect_diff[1] + ywidth * 0.08,
                fill=True,
                color=color_dict_rect[ind % len(COLOR_DICT)],
                linewidth=3,
                ec="red",
                zorder=2
            )
        )
    else:
        encircle(
            X_clust,
            Y_clust,
            ax=ax,
            color=color_dict_rect[ind % len(COLOR_DICT)],
            linewidth=3,
            ec="red",
            zorder=2
        )

    annotate_points(annotations=range(len(X)), points=X, ax=ax)

    num_clust = "n° clust: " + str(len(distance_matrix))
    dist_tot = "dist_tot: " + str(round(level_txt, 5))
    dist_incr = " --- dist_incr: " + str(round(level2_txt, 5)) if level2_txt is not None else ""

    title = num_clust + " --- " + dist_tot + dist_incr

    ax.set_title(title, fontsize=FONTSIZE_BIGGER)

    plt.show()


def dist_mat(df: pd.DataFrame, linkage: str) -> pd.DataFrame:
    """
    take as input the dataframe created by agg_clust and output the distance matrix;
    it is actually an upper triangular matrix, the symmetrical values are replaced with np.inf.

    :param df: input dataframe, with the first column corresponding to x-coordinates and
               the second column corresponding to y-coordinates of data points.
    :param linkage: linkage method; can be single, complete, average.
    :return: distance matrix.

    """

    even_num = [i for i in range(2, len(df) + 1) if i % 2 == 0]
    distance_matrix = pd.DataFrame()
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
                    distance_matrix.loc[i, j] = sl_dist(a, b)
                elif linkage == "complete":
                    distance_matrix.loc[i, j] = cl_dist(a, b)
                elif linkage == "average":
                    distance_matrix.loc[i, j] = avg_dist(a, b)
                else:
                    raise ValueError(f'Input linkage parameter {linkage} is invalid. '
                                     'Possible linkage parameters: "single", "complete", "average"')
            else:

                distance_matrix.loc[i, j] = np.inf

        k += 1

    distance_matrix = distance_matrix.fillna(np.inf)

    return distance_matrix


def dist_mat_gen(df: pd.DataFrame) -> pd.DataFrame:
    """Variation of dist_mat, uses only single_linkage method"""

    even_num = [i for i in range(2, len(df) + 1) if i % 2 == 0]
    distance_matrix = pd.DataFrame()
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

                distance_matrix.loc[i, j] = sl_dist(a, b)
                distance_matrix.loc[j, i] = sl_dist(a, b)
            else:

                distance_matrix.loc[i, j] = np.inf

        k += 1

    distance_matrix = distance_matrix.fillna(np.inf)

    return distance_matrix


def compute_var(X: np.ndarray, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Compute total intra-cluster variance of the cluster configuration inferred from df.

    :param X: input data as array.
    :param df: input dataframe built by agg_clust, listing the cluster and the x and y
                coordinates of each point.
    :return: centroids dataframe with their coordinates and the single variances of the corresponding
             clusters, and the total intra-cluster variance.
    """

    cleaned_index = [cluster_points(i) for i in df.index]

    cent_x_tot = []
    cent_y_tot = []

    for li in cleaned_index:

        cent_x = []
        cent_y = []
        for el in li:
            cent_x.append(X[int(el)][0])
            cent_y.append(X[int(el)][1])
        cent_x_tot.append(np.mean(cent_x))
        cent_y_tot.append(np.mean(cent_y))

    centroids = pd.DataFrame(index=df.index)
    centroids["cx"] = cent_x_tot
    centroids["cy"] = cent_y_tot

    var_int = compute_var_sing(df, centroids)

    centroids["var"] = var_int

    return centroids, centroids["var"].sum()


def compute_var_sing(df: pd.DataFrame, centroids: pd.DataFrame) -> list:
    """
    Compute every internal variance in clusters; clusters are found in df,
    whereas centroids are saved in centroids.

    :param df:  input dataframe built by agg_clust, listing the cluster and the x and y
                coordinates of each point.
    :param centroids: dataframe of the centroids of clusters, with their x and y coordinates.
    :return: list of intra-cluster variances.

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


def compute_ward_ij(data: np.ndarray, df: pd.DataFrame) -> Tuple[Tuple, float, float]:
    """
    Compute difference in total within-cluster variance, with squared euclidean
    distance, and finds the best cluster according to Ward criterion.

    :param data: input data array.
    :param df:  input dataframe built by agg_clust, listing the cluster and the x and y
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

    return (i, j), new_summ, par_var


def sl_dist(a: Iterable, b: Iterable) -> float:
    """Distance for single_linkage method, i.e. min[dist(x,y)] for x in a & y in b """
    distances = []
    for i in a:
        for j in b:
            distances.append(dist1(i, j))
    distances = [i for i in distances if np.isnan(i) == False]
    return np.min(distances)


def cl_dist(a: Iterable, b: Iterable) -> float:
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


def avg_dist(a: Iterable, b: Iterable) -> float:
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


def agg_clust(X: np.ndarray, linkage: str, plotting: bool = True) -> None:
    """
    Perform hierarchical agglomerative clustering with the provided linkage method, plotting every step
    of cluster aggregation.

    :param X: input data array
    :param linkage: linkage method; can be single, complete, average or ward.
    :param plotting: if True, execute plots.
    """

    levels = []
    levels2 = []
    ind_list = []

    # build matrix df, used to store points of clusters with their coordinates
    double_index = [[i, i] for i in range(len(X))]
    flat_list = flatten_list(double_index)
    col = [
        str(el) + "x" if i % 2 == 0 else str(el) + "y"
        for i, el in enumerate(flat_list)
    ]

    df = pd.DataFrame(index=[str(i) for i in range(len(X))], columns=col)

    df["0x"] = X.T[0]
    df["0y"] = X.T[1]

    df_nonan = df.dropna(axis=1, how="all")

    # initial distance matrix
    distance_matrix = dist_mat_gen(df_nonan)
    var_sum = 0
    levels.append(var_sum)
    levels2.append(var_sum)

    # until the desired number of clusters is reached
    while len(df) > 1:

        if linkage == "ward":
            # find indexes corresponding to the minimum increase in total intra-cluster variance
            df_nonan = df.dropna(axis=1, how="all")
            df_nonan = df_nonan.fillna(np.inf)
            ((i, j), var_sum, par_var) = compute_ward_ij(X, df_nonan)

            levels.append(var_sum)
            levels2.append(par_var)
            ind_list.append((i, j))
            new_clust = df.loc[[i, j], :]

        else:
            # find indexes corresponding to the minimum distance
            (i, j) = np.unravel_index(
                np.array(distance_matrix).argmin(), np.array(distance_matrix).shape
            )
            levels.append(np.min(np.array(distance_matrix)))
            ind_list.append((i, j))
            new_clust = df.iloc[[i, j], :]

            # update distance matrix
            distance_matrix = update_mat(distance_matrix, i, j, linkage)

        df = df.drop([new_clust.iloc[0].name], 0)
        df = df.drop([new_clust.iloc[1].name], 0)

        dim1 = int(new_clust.iloc[0].notna().sum())

        new_cluster_name = (
                "("
                + new_clust.iloc[0].name
                + ")-("
                + new_clust.iloc[1].name
                + ")"
        )

        df.loc[new_cluster_name, :] = new_clust.iloc[0].fillna(
            0
        ) + new_clust.iloc[1].shift(dim1, fill_value=0)

        if plotting is True:

            if linkage != "ward":
                point_plot_mod(X, df, levels[-1])
            else:
                point_plot_mod(X, df, levels[-2], levels2[-1])
