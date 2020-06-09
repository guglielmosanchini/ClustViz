import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import ceil
from mpl_toolkits import mplot3d
from tqdm.auto import tqdm
from itertools import groupby
from collections import OrderedDict, Counter


def euclidean_distance(a, b):
    """Returns Euclidean distance of two arrays"""
    return np.linalg.norm(np.array(a) - np.array(b))


def gauss_infl_function(x, y, s, dist="euclidean"):
    if dist == "euclidean":
        return np.exp(
            -(np.power(euclidean_distance(x, y), 2) / (2 * (s ** 2)))
        )


def gauss_dens(x, D, s, dist="euclidean"):
    N = len(D)
    res = 0
    for i in range(N):
        res += gauss_infl_function(x, D[i], s, dist)

    return res


def grad_gauss_dens(x, D, s, dist="euclidean"):
    N = len(D)
    res = 0
    for i in range(N):
        res += gauss_infl_function(x, D[i], s, dist) * (
            np.array(D[i]) - np.array(x)
        )
    return res


def square_wave_infl(x, y, s, dist="euclidean"):
    if dist == "euclidean":
        if euclidean_distance(x, y) <= s:
            return 1
        else:
            return 0


def square_wave_dens(x, D, s, dist="euclidean"):
    N = len(D)
    res = 0
    for i in range(N):
        res += square_wave_infl(x, D[i], s, dist)
    return res


def square_wave_grad(x, D, s, dist="euclidean"):
    N = len(D)
    res = 0
    for i in range(N):
        res += square_wave_infl(x, D[i], s, dist) * (
            np.array(D[i]) - np.array(x)
        )
    return res


def FindPoint(x1, y1, x2, y2, x, y):
    if (x1 < x < x2) and (y1 < y < y2):
        return True
    else:
        return False


def FindRect(point, coord_dict):
    for k, v in coord_dict.items():
        if FindPoint(v[0], v[1], v[2], v[3], point[0], point[1]) is True:
            return k
    return None


def form_populated_cubes(a, b, c, d, data):
    N_c = 0
    lin_sum = [0, 0]
    points = []
    for el in data:
        if FindPoint(a, b, c, d, el[0], el[1]) is True:
            N_c += 1
            lin_sum[0] += el[0]
            lin_sum[1] += el[1]
            points.append([el[0], el[1]])

    cluster = [N_c, lin_sum, points]

    return cluster


def min_bound_rect(data):
    plt.figure(figsize=(22, 15))
    plt.scatter(data[:, 0], data[:, 1], s=100, edgecolor="black")

    rect_min = data.min(axis=0)
    rect_diff = data.max(axis=0) - rect_min

    x0 = rect_min[0] - 0.05
    y0 = rect_min[1] - 0.05

    # minimal bounding rectangle
    plt.gcf().gca().add_patch(
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


def pop_cubes(data, s):
    populated_cubes = {}

    corresp_key_coord = {}

    rect_min = data.min(axis=0)
    rect_diff = data.max(axis=0) - rect_min
    x0 = rect_min[0] - 0.05
    y0 = rect_min[1] - 0.05
    num_width = int(ceil((rect_diff[0] + 0.1) / (2 * s)))
    num_height = int(ceil((rect_diff[1] + 0.1) / (2 * s)))

    for h in range(num_height):
        for w in range(num_width):
            a = x0 + w * (2 * s)
            b = y0 + h * (2 * s)
            c = a + 2 * s
            d = b + 2 * s

            cl = form_populated_cubes(a, b, c, d, data)

            corresp_key_coord[(w, h)] = (a, b, c, d)

            if cl[0] > 0:
                populated_cubes[(w, h)] = cl

    return populated_cubes, corresp_key_coord


def plot_grid_rect(data, s, cube_kind="populated", color_grids=True):
    cl, ckc = pop_cubes(data, s)

    cl_copy = cl.copy()

    coms = [center_of_mass(list(cl.values())[i]) for i in range(len(cl))]

    if cube_kind == "highly_populated":
        cl = highly_pop_cubes(cl, xi_c=3)
        coms_hpc = [
            center_of_mass(list(cl.values())[i]) for i in range(len(cl))
        ]

    min_bound_rect(data)

    plt.scatter(
        np.array(coms)[:, 0],
        np.array(coms)[:, 1],
        s=100,
        color="red",
        edgecolor="black",
    )

    if cube_kind == "highly_populated":
        for i in range(len(coms_hpc)):
            plt.gcf().gca().add_artist(
                plt.Circle(
                    (np.array(coms_hpc)[i, 0], np.array(coms_hpc)[i, 1]),
                    4 * s,
                    color="red",
                    fill=False,
                    linewidth=2,
                    alpha=0.6,
                )
            )

    tot_cubes = connect_cubes(cl, cl_copy, s)

    new_clusts = {
        i: tot_cubes[i]
        for i in list(tot_cubes.keys())
        if i not in list(cl.keys())
    }

    for key in list(new_clusts.keys()):
        (a, b, c, d) = ckc[key]
        plt.gcf().gca().add_patch(
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

    for key in list(ckc.keys()):

        (a, b, c, d) = ckc[key]

        if color_grids is True:
            if key in list(cl.keys()):
                color_or_not = True if cl[key][0] > 0 else False
            else:
                color_or_not = False
        else:
            color_or_not = False

        plt.gcf().gca().add_patch(
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


def check_border_points_rectangles(data, pop_clust):
    count = 0
    for i in range(len(pop_clust)):
        count += list(pop_clust.values())[i][0]

    if count == len(data):
        print("No points lie on the borders of rectangles")
    else:
        diff = len(data) - count
        print("{0} point(s) lie(s) on the border of rectangles".format(diff))


def highly_pop_cubes(pop_cub, xi_c):
    highly_pop_cubes = {}

    for key in list(pop_cub.keys()):
        if pop_cub[key][0] >= xi_c:
            highly_pop_cubes[key] = pop_cub[key]

    return highly_pop_cubes


def center_of_mass(cube):
    return np.array(cube[1]) / cube[0]


def check_connection(cube1, cube2, s, dist="euclidean"):
    c1 = center_of_mass(cube1)
    c2 = center_of_mass(cube2)
    if dist == "euclidean":
        d = euclidean_distance(c1, c2)
    if d <= 4 * s:
        return True
    else:
        return False


def connect_cubes(hp_cubes, cubes, s, dist="euclidean"):
    i = 0
    rel_keys = []

    for k1, c1 in list(hp_cubes.items()):
        for k2, c2 in list(cubes.items()):

            if k1 != k2:

                if check_connection(c1, c2, s, dist) is True:
                    rel_keys.append([k1, k2])
        i += 1

    keys_fin = [item for sublist in rel_keys for item in sublist]

    new_cubes = {i: cubes[i] for i in keys_fin}

    return new_cubes


def near_with_cube(x, cube_x, tot_cubes, s):  # includes the point itself

    near_list = []

    for cube in list(tot_cubes.values()):

        d = euclidean_distance(x, center_of_mass(cube))

        if (d <= 4 * s) and (check_connection(cube_x, cube, s)):
            near_list.append(cube[2])

    near_list = [item for sublist in near_list for item in sublist]

    return near_list


def near_without_cube(
    x, coord_dict, tot_cubes, s
):  # includes the point itself

    k = FindRect(x, coord_dict)

    if k is None:
        return []
    try:
        cube_x = tot_cubes[k]
    except:
        return []

    near_list = []

    for cube in list(tot_cubes.values()):

        d = euclidean_distance(x, center_of_mass(cube))

        if (d <= 4 * s) and (check_connection(cube_x, cube, s)):
            near_list.append(cube[2])

    near_list = [item for sublist in near_list for item in sublist]

    return near_list


# %matplotlib notebook
# %matplotlib inline


def plot_3d_or_contour(data, s, three=False, scatter=False, prec=3):
    fig, ax = plt.subplots(figsize=(14, 6))

    x_data = [np.array(data)[:, 0].min(), np.array(data)[:, 0].max()]
    y_data = [np.array(data)[:, 1].min(), np.array(data)[:, 1].max()]
    mixed_data = [min(x_data[0], y_data[0]), max(x_data[1], y_data[1])]

    xx = np.outer(
        np.linspace(mixed_data[0] - 1, mixed_data[1] + 1, prec * 10),
        np.ones(prec * 10),
    )
    yy = xx.copy().T  # transpose
    z = np.empty((prec * 10, prec * 10))
    for i, a in tqdm(enumerate(range(prec * 10))):
        for j, b in enumerate(range(prec * 10)):
            z[i, j] = gauss_dens(x=np.array([xx[i][a], yy[i][b]]), D=data, s=s)

    if three is True:
        ax = plt.axes(projection="3d")
        ax.plot_surface(xx, yy, z, cmap="winter", edgecolor="none")
        plt.show()
    else:
        CS = ax.contour(xx, yy, z, cmap="winter", edgecolor="none")
        ax.clabel(CS, inline=1, fontsize=10)

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


def assign_cluster(data, others, attractor, clust_dict, processed):
    if others is None or attractor is None:
        print("None")

        return clust_dict, processed

    for point in others:

        point_index = np.nonzero(data == point)[0][0]

        if point_index in processed:
            continue
        else:
            processed.append(point_index)

        if attractor[1] is True:
            clust_dict[point_index] = attractor[0]
        else:
            clust_dict[point_index] = [-1]

    return clust_dict, processed


def plot_infl(data, s, xi):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.set_title("Significance of possible density attractors")

    z = []
    for a, b in zip(np.array(data)[:, 0], np.array(data)[:, 1]):
        z.append(gauss_dens(x=np.array([a, b]), D=data, s=s))

    x_plot = [i for i in range(len(data))]

    X_over = [x_plot[j] for j in range(len(data)) if z[j] >= xi]
    Y_over = [z[j] for j in range(len(data)) if z[j] >= xi]

    X_under = [x_plot[j] for j in range(len(data)) if z[j] < xi]
    Y_under = [z[j] for j in range(len(data)) if z[j] < xi]

    plt.scatter(
        X_over,
        Y_over,
        s=300,
        color="green",
        edgecolor="black",
        alpha=0.7,
        label="possibly significant",
    )

    plt.scatter(
        X_under,
        Y_under,
        s=300,
        color="yellow",
        edgecolor="black",
        alpha=0.7,
        label="not significant",
    )

    plt.axhline(xi, color="red", linewidth=2, label="xi")

    ax.set_ylabel("influence")

    # add indexes to points in plot
    for i, txt in enumerate(range(len(data))):
        ax.annotate(
            txt, (i, z[i]), fontsize=10, size=10, ha="center", va="center"
        )

    ax.legend()
    plt.show()


def plot_3d_both(data, s, xi=None, prec=3):
    from matplotlib import cm

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    x_data = [np.array(data)[:, 0].min(), np.array(data)[:, 0].max()]
    y_data = [np.array(data)[:, 1].min(), np.array(data)[:, 1].max()]
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

            z[i, j] = gauss_dens(x=np.array([xx[i][a], yy[i][b]]), D=data, s=s)
            if xi is not None:
                if z[i, j] >= xi:
                    z_xi[i, j] = z[i, j]
                else:
                    z_xi[i, j] = xi

    # to set colors according to xi value, red if greater, yellow if smaller
    if xi is not None:
        xi_data = []
        for a, b in zip(np.array(data)[:, 0], np.array(data)[:, 1]):
            to_be_eval = gauss_dens(x=np.array([a, b]), D=data, s=s)
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
        np.array(data)[:, 0],
        np.array(data)[:, 1],
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


def density_attractor(
    data,
    x,
    coord_dict,
    tot_cubes,
    s,
    xi,
    delta=0.05,
    max_iter=100,
    dist="euclidean",
):
    x_i = []
    it_number = 0

    other_points = []

    while it_number < max_iter:

        if it_number == 0:
            x_i.append(x)
            it_number += 1

            for point in data:
                if euclidean_distance(point, x) <= s / 2:
                    other_points.append(list(point))
            continue

        old = x_i[-1]
        near_old = near_without_cube(
            x=old, coord_dict=coord_dict, tot_cubes=tot_cubes, s=s
        )

        grad = grad_gauss_dens(x=old, D=near_old, s=s, dist=dist)

        new_x = old + delta * (grad / np.linalg.norm(grad))

        for point in data:
            if euclidean_distance(point, new_x) <= s / 2:
                other_points.append(list(point))

        near_new = near_without_cube(
            x=new_x, coord_dict=coord_dict, tot_cubes=tot_cubes, s=s
        )

        dens_new = gauss_dens(x=new_x, D=near_new, s=s, dist=dist)
        dens_old = gauss_dens(x=old, D=near_old, s=s, dist=dist)

        x_i.append(new_x)

        it_number += 1

        if dens_new < dens_old:
            attractor = old
            # print("iterations: ", it_number)
            if dens_old >= xi:
                res = (attractor, True)
            else:
                res = (attractor, False)

            other_points.sort()
            other_points = list(k for k, _ in groupby(other_points))

            return res, other_points

    print("Max iteration number reached!")
    return None, None


def plot_clust_dict(data, lab_dict):
    fig, ax = plt.subplots(figsize=(14, 6))

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
        18: "lime",
        19: "peru",
        20: "dodgerblue",
        21: "teal",
        22: "royalblue",
        23: "tomato",
        24: "bisque",
        25: "palegreen",
    }

    col = [
        colors[lab_dict.label[i] % len(colors)]
        if lab_dict.label[i] != -1
        else "red"
        for i in range(len(lab_dict))
    ]

    plt.scatter(
        np.array(data)[:, 0],
        np.array(data)[:, 1],
        s=300,
        edgecolor="black",
        color=col,
        alpha=0.8,
    )

    df_dens_attr = lab_dict.groupby("label").mean()

    for i in range(df_dens_attr.iloc[-1].name + 1):
        plt.scatter(
            df_dens_attr.loc[i]["x"],
            df_dens_attr.loc[i]["y"],
            color="red",
            marker="X",
            s=300,
            edgecolor="black",
        )

    # add indexes to points in plot
    for i, txt in enumerate(range(len(data))):
        ax.annotate(
            txt,
            (np.array(data)[i, 0], np.array(data)[i, 1]),
            fontsize=10,
            size=10,
            ha="center",
            va="center",
        )

    plt.show()


def extract_cluster_labels(data, cld, tol=2):
    def similarity(x, y, tol=0.1):
        if (abs(x[0] - y[0]) <= tol) and (abs(x[1] - y[1]) <= tol):
            return True
        else:
            return False

    cld_sorted = OrderedDict(sorted(cld.items(), key=lambda t: t[0]))
    l = list(cld_sorted.values())
    l_mod = [np.round(l[i], 1) for i in range(len(l))]

    lr = {i: l_mod[i] for i in range(len(l_mod)) if len(l_mod[i]) == 2}
    da_list = [i for i in range(len(data))]
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
        labels = [-1] + [i for i in range(range_labels)]
    else:
        range_labels = len(keys)
        labels = [i for i in range(range_labels)]

    label_dict = dict(zip(keys, labels))
    fin_labels = [label_dict[i] for i in da_list]

    df = pd.DataFrame(l_mod)
    if len(Counter(fin_labels)) == 1:
        df["added"] = [np.nan] * len(df)
    df.columns = ["x", "y"]
    df["label"] = fin_labels
    # df.groupby("label").mean()

    return fin_labels, df


def DENCLUE(
    data, s, xi=3, xi_c=3, tol=2, dist="euclidean", prec=20, plotting=True
):
    clust_dict = {}
    processed = []

    z, d = pop_cubes(data=data, s=s)
    print("Number of populated cubes: ", len(z))
    check_border_points_rectangles(data, z)
    hpc = highly_pop_cubes(z, xi_c=xi_c)
    print("Number of highly populated cubes: ", len(hpc))
    new_cubes = connect_cubes(hpc, z, s=s)

    if plotting is True:
        plot_grid_rect(data, s=s, cube_kind="populated")
        plot_grid_rect(data, s=s, cube_kind="highly_populated")

        plot_3d_or_contour(data, s=s, three=False, scatter=True, prec=prec)

        plot_3d_both(data, s=s, xi=xi, prec=prec)

    if len(new_cubes) != 0:
        points_to_process = [
            item
            for sublist in np.array(list(new_cubes.values()))[:, 2]
            for item in sublist
        ]
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
            r, o = density_attractor(
                data=data,
                x=point,
                coord_dict=d,
                tot_cubes=new_cubes,
                s=s,
                xi=xi,
                delta=delta,
                max_iter=600,
                dist=dist,
            )
            delta = delta * 2

        clust_dict, proc = assign_cluster(
            data=data,
            others=o,
            attractor=r,
            clust_dict=clust_dict,
            processed=processed,
        )

    for point in initial_noise:
        point_index = np.nonzero(data == point)[0][0]
        clust_dict[point_index] = [-1]

    try:
        lab, coord_df = extract_cluster_labels(data, clust_dict, tol)
    except:
        print(
            "There was an error when extracting clusters. Increase number of points or try with a less"
            " pathological case: look at the other plots to have an idea of why it failed."
        )

    if plotting is True:
        plot_clust_dict(data, coord_df)

    return lab
