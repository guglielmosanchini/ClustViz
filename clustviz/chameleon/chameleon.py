import numpy as np
import pandas as pd
import itertools
from collections import Counter, OrderedDict
from tqdm import tqdm
import networkx as nx
from typing import List, Tuple, Union

from clustviz.chameleon.graphtools import bisection_weights, connecting_edges, get_weights, get_cluster, knn_graph, \
    plot2d_graph, pre_part_graph, plot2d_data

NxGraph = nx.Graph


def len_edges(graph: NxGraph, cluster: List[int]) -> int:
    """
    Compute the number of edges that interconnect the nodes of the input cluster.

    :param graph: kNN graph.
    :param cluster: cluster represented by a list of nodes belonging to it.
    :return: number of edges interconnecting the nodes of the input graph.
    """
    cluster_graph = graph.subgraph(cluster)
    edges = cluster_graph.edges()
    return len(edges)


def internal_interconnectivity(graph, cluster: List[int]) -> float:
    """
    Compute the weighted sum of edges that partition the graph into two roughly equal parts.

    :param graph: kNN graph.
    :param cluster: cluster represented by a list of nodes belonging to it.
    :return: sum of the bisection weights.
    """
    return np.sum(bisection_weights(graph, cluster))


def relative_interconnectivity(graph: NxGraph, cluster_i: List[int], cluster_j: List[int]) -> float:
    """
    Compute the relative interconnectivity of two clusters of a graph, measured on the connecting edges.

    :param graph: kNN graph.
    :param cluster_i: first cluster.
    :param cluster_j: second cluster.
    :return: relative interconnectivity of the two input clusters.
    """
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        return 0.0
    # EC: sum of the weights of connecting edges of clusters i and j
    EC = np.sum(get_weights(graph, edges))
    ECci, ECcj = (
        internal_interconnectivity(graph, cluster_i),
        internal_interconnectivity(graph, cluster_j),
    )

    if ECci + ECcj != 0:
        rel_int = EC / ((ECci + ECcj) / 2.0)
    else:
        rel_int = np.inf

    return rel_int


def internal_closeness(graph: NxGraph, cluster: List[int]) -> float:
    """
    Compute the internal closeness of the input cluster, i.e. the sum of weights of the edges fully contained in it.

    :param graph: kNN graph.
    :param cluster: cluster represented by a list of nodes belonging to it.
    :return: internal closeness of the input cluster.
    """
    cluster_graph = graph.subgraph(cluster)
    edges = cluster_graph.edges()
    weights = get_weights(cluster_graph, edges)
    return np.sum(weights)


def relative_closeness(graph: NxGraph, cluster_i: List[int], cluster_j: List[int]) -> float:
    """
    Compute the relative closeness of two clusters of a graph, measured on the connecting edges.

    :param graph: kNN graph.
    :param cluster_i: first cluster.
    :param cluster_j: second cluster.
    :return: relative closeness of the two input clusters.
    """
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(get_weights(graph, edges))
    # originally by Moonpuck
    # Ci, Cj = internal_closeness(graph, cluster_i), internal_closeness(graph, cluster_j)
    # original paper
    Ci, Cj = len(cluster_i), len(cluster_j)
    # paper of chameleon2
    # Ci,Cj = len_edges(graph, cluster_i), len_edges(graph, cluster_j)
    bis_weight_i = bisection_weights(graph, cluster_i)
    bis_weight_j = bisection_weights(graph, cluster_j)

    if len(bis_weight_i) == 0 or len(bis_weight_j) == 0:
        return np.nan
    else:
        SECci = np.mean(bis_weight_i)
        SECcj = np.mean(bis_weight_j)
        return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


def merge_score(graph: NxGraph, cluster_i: List[int], cluster_j: List[int], alpha: float) -> float:
    """
    Compute the score associated with the merging of the two clusters.

    :param graph: kNN graph.
    :param cluster_i: first cluster.
    :param cluster_j: second cluster.
    :param alpha: exponent of relative closeness; the larger, the more important relative closeness is than
                  relative interconnectivity.
    :return: merging score.
    """
    ri = relative_interconnectivity(graph, cluster_i, cluster_j)
    rc_pot = np.power(relative_closeness(graph, cluster_i, cluster_j), alpha)

    if (ri != 0) and (rc_pot != 0):
        return ri * rc_pot
    else:
        return ri + rc_pot


def merge_best(graph, df, alpha, k, verbose=False, verbose2=True) -> Union[Tuple[pd.DataFrame, float, int], bool]:
    """
    Find the two clusters with the highest score and merge them.

    :param graph: kNN graph.
    :param df: input dataframe.
    :param alpha: exponent of relative closeness; the larger, the more important relative closeness is than
                  relative interconnectivity.
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
                print(f"Checking c_{i} c_{j}")
            gi = get_cluster(graph, i)
            gj = get_cluster(graph, j)
            edges = connecting_edges((gi, gj), graph)
            if not edges:
                continue
            ms = merge_score(graph, gi, gj, alpha)
            if verbose:
                print(f"Merge score: {ms}")
            if ms > max_score:
                if verbose:
                    print(f"Better than: {max_score}")
                max_score = ms
                ci, cj = i, j

    if max_score > 0:
        if verbose2:
            print(f"Merging c_{ci} and c_{cj}")
            print(f"score: {max_score}")

        df.loc[df["cluster"] == cj, "cluster"] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.nodes[p]["cluster"] == cj:
                graph.nodes[p]["cluster"] = ci
    else:
        if verbose:
            print("No Merging")
            print(f"score: {max_score}")
            print("Early stopping")

    return df, max_score, ci


def cluster(df: pd.DataFrame, k: int, knn: int = 10, m: int = 30, alpha: float = 2.0, verbose0: bool = False,
            verbose1: bool = False, verbose2: bool = True, plot: bool = True) -> Tuple[pd.DataFrame, OrderedDict]:
    """
    Chameleon clustering: build the K-NN graph, partition it into m clusters

    :param df: input dataframe.
    :param k: desired number of clusters.
    :param knn: parameter k of K-nearest_neighbors.
    :param m: number of clusters to reach in the initial clustering phase.
    :param alpha: exponent of relative closeness; the larger, the more important relative closeness is than
                  relative interconnectivity.
    :param verbose0: if True, print general infos.
    :param verbose1: if True, print infos about the prepartitioning phase.
    :param verbose2: if True, print labels of merging clusters and their scores in the merging phase.
    :param plot: if True, show plots.
    :return: dataframe with cluster labels and dictionary of merging scores (similarities).
    """
    if k is None:
        k = 1

    if verbose0:
        print(f"Building kNN graph (k = {knn})...")

    graph = knn_graph(df=df, k=knn, symmetrical=False, verbose=verbose1)

    if plot is True:
        plot2d_graph(graph, print_clust=False)

    graph = pre_part_graph(graph, m, df, verbose1, plotting=plot)

    # to account for cases where initial_clust is too big or k is already reached before the merging phase
    cl_dict = OrderedDict({
        list(graph.nodes)[i]: graph.nodes[i]["cluster"]
        for i in range(len(graph))
    })
    m = len(Counter(cl_dict.values()))

    if verbose0:
        print(f"actual init_clust: {m}")

    merging_similarities = OrderedDict({})
    iterm = (tqdm(enumerate(range(m - k)), total=m - k) if verbose1 else enumerate(range(m - k)))

    for i, _ in iterm:

        df, ms, ci = merge_best(graph, df, alpha, k, False, verbose2)

        if ms == 0:
            break

        merging_similarities[m - (i + 1)] = ms

        if plot:
            plot2d_data(df, ci)

    res = rebuild_labels(df)

    return res, merging_similarities


def rebuild_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the clustering labels of the input dataframe, i.e. bring them into the range starting from 1.
    For example, if they range from 6 to 10, bring them to the range 1 to 5.

    :param df: dataframe obtained from the merging phase of the algorithm.
    :return: dataframe with cleaned labels.
    """
    cleaned_df = df.copy()
    clusters = df["cluster"].unique()  # list(pd.DataFrame(df["cluster"].value_counts()).index)
    c = 1
    for i in clusters:
        cleaned_df.loc[df["cluster"] == i, "cluster"] = c
        c = c + 1
    return cleaned_df
