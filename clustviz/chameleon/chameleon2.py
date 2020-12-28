import numpy as np
import pandas as pd
import itertools
import networkx as nx
from collections import OrderedDict, deque
from tqdm import tqdm
from typing import List, Union, Tuple, Generator, Dict

from clustviz.chameleon.graphtools import connecting_edges, get_weights, plot2d_data, plot2d_graph, knn_graph, \
    pre_part_graph
from clustviz.chameleon.chameleon import (
    internal_closeness,
    get_cluster,
    len_edges,
    rebuild_labels,
)

NxGraph = nx.Graph


def w_int_closeness(graph: NxGraph, cluster: List[int]) -> float:
    """
    Compute the internal closeness of the input cluster weighted by the number of its internal edges.

    :param graph: kNN graph.
    :param cluster: cluster represented by a list of nodes belonging to it.
    :return: weighted internal closeness.
    """
    return internal_closeness(graph, cluster) / len_edges(graph, cluster)


def relative_closeness2(graph: NxGraph, cluster_i: List[int], cluster_j: List[int], m_fact: float) -> float:
    """
    Compute the relative closeness of the two input clusters.

    :param graph: kNN graph.
    :param cluster_i: first cluster.
    :param cluster_j: second cluster.
    :param m_fact: multiplicative factor for clusters composed of a single node.
    :return: relative closeness of the two input clusters.
    """
    edges = connecting_edges((cluster_i, cluster_j), graph)

    if not edges:

        return 0.0
    else:

        S_bar = np.mean(get_weights(graph, edges))

    sCi, sCj = (
        internal_closeness(graph, cluster_i),
        internal_closeness(graph, cluster_j),
    )

    ratio = S_bar / (sCi + sCj)

    if (len_edges(graph, cluster_i) == 0) or (
        len_edges(graph, cluster_j) == 0
    ):

        return m_fact * ratio

    else:

        return (
            len_edges(graph, cluster_i) + len_edges(graph, cluster_j)
        ) * ratio


def relative_interconnectivity2(graph: NxGraph, cluster_i: List[int], cluster_j: List[int],
                                beta: float) -> float:
    """
    Compute the relative interconnectivity of the two input clusters.

    :param graph: kNN graph.
    :param cluster_i: first cluster.
    :param cluster_j: second cluster.
    :param beta: exponent of the rho factor; the larger, the less encouraged the merging of clusters connected
                 by a large number of edges relative to the number of edges inside the cluster.
    :return: relative interconnectivity of the two input clusters.
    """
    if (len_edges(graph, cluster_i) == 0) or (len_edges(graph, cluster_j) == 0):
        return 1.0
    else:
        edges = connecting_edges((cluster_i, cluster_j), graph)
        denom = min(len_edges(graph, cluster_i), len_edges(graph, cluster_j))

    return (len(edges) / denom) * np.power(rho(graph, cluster_i, cluster_j), beta)


def rho(graph: NxGraph, cluster_i: List[int], cluster_j: List[int]) -> float:
    """
    Compute the rho factor, which discourages the algorithm from merging clusters with different densities.

    :param graph: kNN graph.
    :param cluster_i: first cluster.
    :param cluster_j: second cluster.
    :return: rho factor.
    """
    s_Ci, s_Cj = (
        w_int_closeness(graph, cluster_i),
        w_int_closeness(graph, cluster_j),
    )

    return min(s_Ci, s_Cj) / max(s_Ci, s_Cj)


def merge_score2(graph: NxGraph, ci: List[int], cj: List[int], alpha: float, beta: float, m_fact: float) -> float:
    """
    Compute the score associated with the merging of the two clusters.

    :param graph: kNN graph.
    :param ci: first cluster.
    :param cj: second cluster.
    :param alpha: exponent of relative closeness; the larger, the more important relative closeness is than
                  relative interconnectivity.
    :param beta: exponent of the rho factor; the larger, the less encouraged the merging of clusters connected
                 by a large number of edges relative to the number of edges inside the cluster.
    :param m_fact: multiplicative factor for clusters composed of a single node.
    :return: merging score
    """
    ri = relative_interconnectivity2(graph, ci, cj, beta)
    rc_pot = np.power(relative_closeness2(graph, ci, cj, m_fact), alpha)
    if (ri != 0) and (rc_pot != 0):
        return ri * rc_pot
    else:
        return ri + rc_pot


def merge_best2(graph: NxGraph, df: pd.DataFrame, alpha: float, beta: float, m_fact: float,
                k: int, verbose: bool = False, verbose2: bool = True) -> Union[Tuple[pd.DataFrame, float, int], bool]:
    """
    Find the two clusters with the highest score and merge them.

    :param graph: kNN graph.
    :param df: input dataframe.
    :param alpha: exponent of relative closeness; the larger, the more important relative closeness is than
                  relative interconnectivity.
    :param beta: exponent of the rho factor; the larger, the less encouraged the merging of clusters connected
                 by a large number of edges relative to the number of edges inside the cluster.
    :param m_fact: multiplicative factor for clusters composed of a single node.
    :param k: desired number of clusters.
    :param verbose: if True, print additional infos.
    :param verbose2: if True, print labels of merging clusters and their score.
    :return: input dataframe with clustering label column, maximum merging score and newly merged cluster label.
    """
    clusters = np.unique(df["cluster"])
    max_score = 0
    ci, cj = -1, -1
    if len(clusters) <= k:
        return False

    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            if verbose:
                print(f"Checking c{i} c{j}.")
            gi = get_cluster(graph, i)
            gj = get_cluster(graph, j)
            edges = connecting_edges((gi, gj), graph)
            if not edges:
                continue
            ms = merge_score2(graph, gi, gj, alpha, beta, m_fact)
            if verbose:
                print(f"Merge score: {ms}.")
            if ms > max_score:
                if verbose:
                    print(f"Better than: {max_score}.")
                max_score = ms
                ci, cj = i, j

    if max_score > 0:
        if verbose2:
            print(f"Merging c{ci} and c{cj}.")
            print(f"score: {max_score}.")

        df.loc[df["cluster"] == cj, "cluster"] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.nodes[p]["cluster"] == cj:
                graph.nodes[p]["cluster"] = ci
    else:
        print("No Merging.")
        print(f"score: {max_score}.")
        print("Early stopping.")
        print("Increase k of kNN to perform each merging step.")

    return df, max_score, ci


def cluster2(df: pd.DataFrame, k: int = None, knn: int = None, m: int = 30, alpha: float = 2.0, beta: float = 1,
             m_fact: float = 1e3, verbose: bool = False, verbose1: bool = True, verbose2: bool = True,
             plot: bool = True, auto_extract: bool = False) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """

    :param df: input dataframe.
    :param k: desired number of clusters.
    :param knn: parameter k of K-nearest_neighbors.
    :param m: number of clusters to reach in the initial clustering phase.
    :param alpha: exponent of relative closeness; the larger, the more important relative closeness is than
                  relative interconnectivity.
    :param beta: exponent of the rho factor; the larger, the less encouraged the merging of clusters connected
                 by a large number of edges relative to the number of edges inside the cluster.
    :param m_fact: multiplicative factor for clusters composed of a single node.
    :param verbose:
    :param verbose1:
    :param verbose2:
    :param plot: if True, show plots.
    :param auto_extract:
    :return:
    """
    if knn is None:
        knn = int(round(2 * np.log(len(df))))

    if k is None:
        k = 1
    if verbose:
        print(f"Building symmetrical kNN graph (k = {knn})...")
    graph_knn = knn_graph(df=df, k=knn, symmetrical=True, verbose=verbose1)

    if plot:
        plot2d_graph(graph_knn, print_clust=False)

    graph_pp = pre_part_graph(graph_knn, m, df, verbose1, plotting=plot)
    if verbose:
        print("flood fill...")

    graph_ff, increased_m = flood_fill(graph_pp, graph_knn, df)

    m = increased_m
    if verbose:
        print(f"new m: {m}")

    if plot:
        plot2d_graph(graph_ff, print_clust=False)

    merging_similarities = {}
    iterm = (
        tqdm(enumerate(range(m - k)), total=m - k)
        if verbose1
        else enumerate(range(m - k))
    )

    for i, _ in iterm:

        df, ms, ci = merge_best2(
            graph_ff, df, alpha, beta, m_fact, k, False, verbose2
        )

        if ms == 0:
            break

        merging_similarities[m - (i + 1)] = ms

        if plot:
            plot2d_data(df, ci)
    if verbose:
        print(f"merging_similarities: {merging_similarities}")
    res = rebuild_labels(df)

    if auto_extract is True:
        extract_optimal_n_clust(merging_similarities, m)

    return res, merging_similarities


def connected_components(connected_points: dict) -> Generator:
    """
    Find connected components from a dictionary of connected nodes.

    :param connected_points: (symmetrically) connected points.
    :return: connected components.
    """
    seen = set()

    for root in list(connected_points.keys()):
        if root not in seen:
            seen.add(root)
            component = []
            queue = deque([root])

            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in connected_points[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            yield component


def prepro_edge(graph: nx.Graph) -> OrderedDict:
    """
    Build a dictionary having points as keys and all the points that are symmetrically
    connected through edges to the key point as values, i.e. 0: [5, 7] means that there are edges 0->5 and 0->7, but
    also 5->0 and 7->0.

    :param graph: kNN graph.
    :return: dictionary of symmetrically connected points.
    """
    z = np.array((graph.edges()))
    g = pd.DataFrame(z, columns=["a", "b"])
    g_bis = pd.concat([g["b"], g["a"]], axis=1, keys=["a", "b"])
    g = g.append(g_bis, ignore_index=True)
    g["b"] = g["b"].astype("str")
    g1 = g.groupby("a")["b"].apply(lambda x: ",".join(x))
    g1 = g1.apply(lambda x: x.split(","))
    for k in list(g1.index):
        g1[k] = [int(i) for i in g1[k]]
    g1 = dict(g1)
    for i in range(len(graph)):
        if i not in list(g1.keys()):
            g1[i] = []
    g1 = OrderedDict(sorted(g1.items(), key=lambda t: t[0]))
    return g1


# def conn_comp(graph: nx.Graph) -> List[list]:
#     """
#     Find the connected components of the input graph, e.g. [[0,2], [1,3]], with numbers corresponding to nodes.
#
#     :param graph: kNN graph.
#     :return: list of connected componenents, each one identified by its nodes.
#     """
#     sym_connected_points = prepro_edge(graph)
#
#     return list(connected_components(sym_connected_points))


def flood_fill(preprocessed_graph: NxGraph, knn_graph: NxGraph, df: pd.DataFrame) -> Tuple[NxGraph, int]:
    """
    Find clusters composed by more than one connected component and divide them accordingly. Adjust
    the parameter m, which indicates the number of clusters to reach in the initial phase.

    :param preprocessed_graph: clustered kNN graph.
    :param knn_graph: kNN graph.
    :param df: input dataframe.
    :return: preprocessed graph with updated cluster labels, new m parameter.
    """
    len_0_clusters = 0
    cl_dict = {
        list(preprocessed_graph.nodes)[i]: preprocessed_graph.nodes[i]["cluster"]
        for i in range(len(preprocessed_graph))
    }
    new_cl_ind = max(cl_dict.values()) + 1
    dic_edge = prepro_edge(knn_graph)

    # print(cl_dict)
    # print("******"*10)
    # print(dic_edge)

    for num in range(max(cl_dict.values()) + 1):
        points = [k for k, v in cl_dict.items() if v == num]
        restr_dict = {p: dic_edge[p] for p in points}
        r_dict = {}

        for k in restr_dict.keys():
            r_dict[k] = [i for i in restr_dict[k] if i in points]

        cc_list = list(connected_components(r_dict))
        print("cluster_label: {0}, #_connected_components: {1}".format(num, len(cc_list)))
        if len(cc_list) == 1:
            continue
        elif len(cc_list) == 0:
            len_0_clusters += 1
        else:
            # skip the first
            for component in cc_list[1:]:
                print(f"new index for the component: {new_cl_ind}")
                for el in component:
                    cl_dict[el] = new_cl_ind
                new_cl_ind += 1

    df["cluster"] = list(cl_dict.values())

    for i in range(len(preprocessed_graph)):
        preprocessed_graph.nodes[i]["cluster"] = cl_dict[i]

    increased_m = max(cl_dict.values()) + 1 - len_0_clusters

    return preprocessed_graph, increased_m


def dendrogram_height(merging_similarities: Dict[int, float], m: int) -> Dict[int, float]:
    """
    Find dendrogram height, defined with a recursive sum of the reciprocal of the merging scores.

    :param merging_similarities: merging scores of the algorithm.
    :param m: initial number of clusters.
    :return: dendrogram height.
    """
    dh = {m - 1: (1 / merging_similarities[m - 1])}

    for i in list(merging_similarities.keys())[:-1]:
        dh[i - 1] = dh[i] + 1 / merging_similarities[i - 1]

    return dh


def find_bigger_jump(dh: Dict[int, float], jump: float) -> float:
    """
    Find a bigger jump in the dendrogram levels.

    :param dh: dendrogram height.
    :param jump: threshold to exceed.
    :return: best level where to cut off th dendrogram if found, else 0.
    """
    lower = list(dh.values())[int(len(dh) / 2) + 1]
    for i in list(range(int(len(dh) / 2), len(dh))):
        upper = list(dh.values())[i]
        if upper - lower > jump:
            return lower + (upper - lower) / 2
        lower = upper
    return 0


def first_jump_cutoff(dh: Dict[int, float], mult: float, eta: float, m: int) -> float:
    """
    Find the first large gap between tree level, which heuristically is the best level where clusters should be
    divided.

    :param dh: dendrogram height.
    :param mult: additional factor.
    :param eta: decrease coefficient.
    :param m: initial number of clusters.
    :return: best level where to cut off dendrogram.
    """
    half = int(round(len(dh) / 2))
    l = reversed(range(m - 1 - half, m - 1))
    half_dict = {j: dh[j] for j in l}
    avg = np.mean(list(half_dict.values()))
    res = 0

    while mult > 0:
        res = find_bigger_jump(dh, mult * avg)
        if res != 0:
            return res
        else:
            mult /= eta


def find_nearest_height(dh: Dict[int, float], value: float) -> int:
    """
    Find nearest height to cutoff value.

    :param dh: dendrogram height.
    :param value: first_jump cutoff value.
    :return: nearest dendrogram height to cutoff value.
    """
    idx = np.searchsorted(list(dh.values()), value, side="left")
    el = list(dh.values())[idx]
    key_list = [k for (k, v) in dh.items() if v == el]
    return key_list[0]


def extract_optimal_n_clust(merging_similarities: Dict[int, float], m: int, f: float = 1000, eta: float = 2) -> None:
    """
    Extract the optimal number of clusters using the dendrogram.

    :param merging_similarities: merging scores of the algorithm.
    :param m: initial number of clusters.
    :param f: threshold parameter to determine if jump is large enough.
    :param eta: decrease coefficient.
    """
    dh = dendrogram_height(merging_similarities, m)

    if len(dh) <= 3:
        print("Insufficient merging steps to perform auto_extract; decrease k and/or increase m.")
        return

    fjc = first_jump_cutoff(dh, f, eta, m)

    opt_n_clust = find_nearest_height(dh, fjc)

    print(f"Optimal number of clusters: {opt_n_clust}")
