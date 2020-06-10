import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from algorithms.chameleon.visualization import *
from collections import OrderedDict

import metis


def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def knn_graph(df, k, verbose=False):
    """return the weighted K-NearestNeighbor graph of input df"""
    points = [p[1:] for p in df.itertuples()]
    g = nx.Graph()
    for i in range(0, len(points)):
        g.add_node(i)
    iterpoints = (
        tqdm(enumerate(points), total=len(points))
        if verbose
        else enumerate(points)
    )
    for i, p in iterpoints:
        distances = list(map(lambda x: euclidean_distance(p, x), points))
        closests = np.argsort(distances)[
            1 : k + 1
        ]  # second through kth closest
        for c in closests:
            g.add_edge(
                i,
                c,
                weight=1.0 / distances[c],
                similarity=int(1.0 / distances[c] * 1e4),
            )
        g.node[i]["pos"] = p
    g.graph["edge_weight_attr"] = "similarity"
    return g


def knn_graph_sym(df, k, verbose=False):
    """return the weighted symmetrical K-NearestNeighbor graph of input df"""
    points = [p[1:] for p in df.itertuples()]
    g = nx.Graph()
    for i in range(0, len(points)):
        g.add_node(i)
    iterpoints = (
        tqdm(enumerate(points), total=len(points))
        if verbose
        else enumerate(points)
    )
    for i, p in iterpoints:
        distances = list(map(lambda x: euclidean_distance(p, x), points))
        closests = np.argsort(distances)[
            1 : k + 1
        ]  # second through kth closest
        for c in closests:
            distances2 = list(
                map(lambda x: euclidean_distance(points[c], x), points)
            )
            closests2 = np.argsort(distances2)[1 : k + 1]
            if i in closests2:
                g.add_edge(
                    i,
                    c,
                    weight=1.0 / distances[c],
                    similarity=int(1.0 / distances[c] * 1e4),
                )
        g.node[i]["pos"] = p
    g.graph["edge_weight_attr"] = "similarity"
    return g


def part_graph(graph, k, df=None):
    """return the input graph with the clustering obtained through mincut-bisection"""
    edgecuts, parts = metis.part_graph(graph, 2, objtype="cut", ufactor=250, seed=42)
    # print(edgecuts)
    for i, p in enumerate(graph.nodes()):
        graph.node[p]["cluster"] = parts[i]
    if df is not None:
        df["cluster"] = nx.get_node_attributes(graph, "cluster").values()
    return graph


def pre_part_graph(graph, k, df=None, verbose=True, plotting=False):
    """return the graph after partitioning it into k clusters"""
    if verbose:
        print("Begin clustering...")
    clusters = 0
    for i, p in enumerate(graph.nodes()):
        graph.node[p]["cluster"] = 0
    cnts = OrderedDict({0: len(graph.nodes())})
    while clusters < k - 1:
        maxc = -1
        maxcnt = 0
        # find key and value of biggest cluster
        for key, val in cnts.items():
            if val > maxcnt:
                maxc = key
                maxcnt = val
        # take the nodes of the biggest cluster
        s_nodes = [n for n in graph.node if graph.node[n]["cluster"] == maxc]
        s_graph = graph.subgraph(s_nodes)
        # bisect the biggest cluster such that the edge-cut is minimized
        edgecuts, parts = metis.part_graph(
            s_graph, 2, objtype="cut", ufactor=250, seed=42
        )
        new_part_cnt = 0
        # adjust cluster labels according to the new bisection
        new_biggest_clust_label = pd.Series(parts).value_counts().idxmax()
        for i, p in enumerate(s_graph.nodes()):
            if parts[i] == new_biggest_clust_label:
                graph.node[p]["cluster"] = clusters + 1
                new_part_cnt += 1
        if plotting is True:
            plot2d_graph(graph)
        cnts[maxc] = cnts[maxc] - new_part_cnt
        cnts[clusters + 1] = new_part_cnt
        clusters += 1

    # edgecuts, parts = metis.part_graph(graph, k)
    # add clustering details to df
    if df is not None:
        df["cluster"] = nx.get_node_attributes(graph, "cluster").values()
    return graph


def get_cluster(graph, clusters):
    """return the list of nodes belonging to specific cluster(s)"""
    nodes = [n for n in graph.node if graph.node[n]["cluster"] in clusters]
    return nodes


def connecting_edges(partitions, graph):
    """
    return only the edges that connect nodes of the first cluster with nodes of the second cluster, in the form of
    a list of tuples [(0, 5), (3, 5)] (e.g. the only connecting-edges are the ones connecting node_0 to node_5 and
    node_3 to node_5
    :param partitions: tuple with two clusters.
    :param graph: NetworkX graph.

    """
    cut_set = []
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph:
                if b in graph[a]:
                    cut_set.append((a, b))
    return cut_set


def min_cut_bisector(graph):
    """return the edges that connect nodes belonging to the two new partitions obtained through min-cut bisection"""
    graph = graph.copy()
    graph = part_graph(graph, 2)
    partitions = get_cluster(graph, [0]), get_cluster(graph, [1])
    return connecting_edges(partitions, graph)


def get_weights(graph, edges):
    return [graph[edge[0]][edge[1]]["weight"] for edge in edges]


def bisection_weights(graph, cluster):
    """return the weights of the edges that 'roughly' bisect the cluster"""
    cluster = graph.subgraph(cluster)
    edges = min_cut_bisector(cluster)
    weights = get_weights(cluster, edges)
    return weights
