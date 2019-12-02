import itertools
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm.auto import tqdm
from algorithms.chameleon.graphtools import *


def internal_interconnectivity(graph, cluster):
    return np.sum(bisection_weights(graph, cluster))


def relative_interconnectivity(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    EC = np.sum(get_weights(graph, edges))
    ECci, ECcj = internal_interconnectivity(graph, cluster_i), internal_interconnectivity(graph, cluster_j)
    return EC / ((ECci + ECcj) / 2.0)


def internal_closeness(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = get_weights(cluster, edges)
    return np.sum(weights)

def w_int_closeness(graph,cluster):
    return internal_closeness(graph,cluster) / len(cluster.edges)



def relative_closeness(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(get_weights(graph, edges))
    #Ci, Cj = internal_closeness(graph, cluster_i), internal_closeness(graph, cluster_j)
    Ci, Cj = len(cluster_i), len(cluster_j)
    #Ci,Cj = len(cluster_i.edges), len(cluster_j.edges)
    SECci, SECcj = np.mean(bisection_weights(graph, cluster_i)), np.mean(
        bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


def merge_score(g, ci, cj, a):
    return relative_interconnectivity(g, ci, cj) * np.power(relative_closeness(g, ci, cj), a)


def merge_best(graph, df, a, k, verbose=False, verbose2=True):
    clusters = np.unique(df['cluster'])
    max_score = 0
    ci, cj = -1, -1
    if len(clusters) <= k:
        return False

    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            if verbose:
                print("Checking c%d c%d" % (i, j))
            gi = get_cluster(graph, [i])
            gj = get_cluster(graph, [j])
            edges = connecting_edges(
                (gi, gj), graph)
            if not edges:
                continue
            ms = merge_score(graph, gi, gj, a)
            if verbose:
                print("Merge score: %f" % (ms))
            if ms > max_score:
                if verbose:
                    print("Better than: %f" % (max_score))
                max_score = ms
                ci, cj = i, j

    if max_score > 0:
        if verbose2:

            print("Merging c%d and c%d" % (ci, cj))
            print("score: ", max_score)

        df.loc[df['cluster'] == cj, 'cluster'] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.node[p]['cluster'] == cj:
                graph.node[p]['cluster'] = ci
    else:
        print("No Merging")
        print("score: ", max_score)
    return (df, max_score > 0, ci)


def cluster(df, k, knn=10, m=30, alpha=2.0, verbose=True, verbose2=True, plot=True):
    print("Building kNN graph (k = %d)..." % (knn))
    graph = knn_graph(df, knn, verbose)

    plot2d_graph(graph, print_clust=False)

    graph = pre_part_graph(graph, m, df, verbose, plotting=plot)

    #plot2d_graph(graph)

    iterm = tqdm(enumerate(range(m - k)), total=m-k) if verbose else enumerate(range(m-k))
    for i in iterm:
        df, m, ci = merge_best(graph, df, alpha, k, False, verbose2)
        if plot:
            plot2d_data(df, ci)
    res = rebuild_labels(df)
    return res

def cluster2(df, k, knn=10, m=30, alpha=2.0, verbose=True, verbose2=True, plot=True):
    print("Building kNN graph (k = %d)..." % (knn))
    graph_knn = knn_graph_sym(df, knn, verbose)

    plot2d_graph(graph_knn, print_clust=False)

    graph_pp = pre_part_graph(graph_knn, m, df, verbose, plotting=plot)

    #plot2d_graph(graph_pp)

    print("flood fill...")

    graph_ff = flood_fill(graph_pp, graph_knn, df)

    plot2d_graph(graph_ff, print_clust=False)

    iterm = tqdm(enumerate(range(m - k)), total=m-k) if verbose else enumerate(range(m-k))
    for i in iterm:
        df, m, ci = merge_best(graph_ff, df, alpha, k, False, verbose2)
        if plot:
            plot2d_data(df, ci)
    res = rebuild_labels(df)
    return res

def rebuild_labels(df):
    ans = df.copy()
    clusters = list(pd.DataFrame(df['cluster'].value_counts()).index)
    c = 1
    for i in clusters:
        ans.loc[df['cluster'] == i, 'cluster'] = c
        c = c + 1
    return ans


def connected_components(graph):
    from collections import deque
    seen = set()

    #for root in range(len(graph)):
    for root in list(graph.keys()):
        if root not in seen:
            seen.add(root)
            component = []
            queue = deque([root])

            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            yield component

def prepro_edge(knn_gr):
    z = np.array((knn_gr.edges()))
    g = pd.DataFrame(z, columns=["a", "b"])
    g_bis = pd.concat([g['b'], g['a']], axis=1, keys=['a', 'b'])
    g = g.append(g_bis, ignore_index=True)
    g["b"] = g["b"].astype("str")
    g1 = g.groupby("a")["b"].apply(lambda x: ",".join(x))
    g1 = g1.apply(lambda x: x.split(","))
    for k in list(g1.index):
        g1[k] = [int(i) for i in g1[k]]
    g1 = dict(g1)
    for i in range(len(knn_gr)):
        if i not in list(g1.keys()):
            g1[i] = []
    g1 = OrderedDict(sorted(g1.items(), key=lambda t: t[0]))
    return g1

def conn_comp(df, knn_gr):

    from algorithms.chameleon.graphtools import knn_graph, pre_part_graph

    g1 = prepro_edge(df, knn_gr)

    return list(connected_components(g1))

def flood_fill(graph, knn_gr, df):

    cl_dict = {list(graph.node)[i] : graph.node[i]["cluster"] for i in range(len(graph))}
    new_cl_ind = max(cl_dict.values()) + 1
    dic_edge = prepro_edge(knn_gr)

    for num in range(max(cl_dict.values())+1):
        points = [i for i in list(cl_dict.keys()) if list(cl_dict.values())[i]==num]
        restr_dict = {list(dic_edge.keys())[i] : dic_edge[i] for i in points}
        r_dict = {}

        for i in list(restr_dict.keys()):
            r_dict[i] = [i for i in restr_dict[i] if i in points]

        cc_list = list(connected_components(r_dict))
        print("num_cluster: {0}, len: {1}".format(num,len(cc_list)))
        if len(cc_list) == 1:
            continue
        else:
            #skip the first
            for component in cc_list[1:]:
                print("comp e ind: ", new_cl_ind)
                for el in component:
                    cl_dict[el] = new_cl_ind
                new_cl_ind += 1

    df["cluster"]= list(cl_dict.values())

    for i in range(len(graph)):
        graph.node[i]["cluster"] = cl_dict[i]

    return graph
