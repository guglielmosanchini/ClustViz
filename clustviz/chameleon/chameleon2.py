import itertools
from clustviz.chameleon.graphtools import *
from clustviz.chameleon.chameleon import (
    internal_closeness,
    get_cluster,
    len_edges,
    rebuild_labels,
)


def w_int_closeness(graph, cluster):
    return internal_closeness(graph, cluster) / len_edges(graph, cluster)


def relative_closeness2(graph, cluster_i, cluster_j, m_fact):
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


def relative_interconnectivity2(graph, cluster_i, cluster_j, b):
    if (len_edges(graph, cluster_i) == 0) or (
        len_edges(graph, cluster_j) == 0
    ):

        return 1.0

    else:

        edges = connecting_edges((cluster_i, cluster_j), graph)
        denom = min(len_edges(graph, cluster_i), len_edges(graph, cluster_j))

    return (len(edges) / denom) * np.power(rho(graph, cluster_i, cluster_j), b)


def rho(graph, cluster_i, cluster_j):
    s_Ci, s_Cj = (
        w_int_closeness(graph, cluster_i),
        w_int_closeness(graph, cluster_j),
    )

    return min(s_Ci, s_Cj) / max(s_Ci, s_Cj)


def merge_score2(g, ci, cj, a, b, m_fact):
    ri = relative_interconnectivity2(g, ci, cj, b)
    rc_pot = np.power(relative_closeness2(g, ci, cj, m_fact), a)
    if (ri != 0) and (rc_pot != 0):
        return ri * rc_pot
    else:
        return ri + rc_pot


def merge_best2(graph, df, a, b, m_fact, k, verbose=False, verbose2=True):
    clusters = np.unique(df["cluster"])
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
            edges = connecting_edges((gi, gj), graph)
            if not edges:
                continue
            ms = merge_score2(graph, gi, gj, a, b, m_fact)
            if verbose:
                print("Merge score: %f" % ms)
            if ms > max_score:
                if verbose:
                    print("Better than: %f" % max_score)
                max_score = ms
                ci, cj = i, j

    if max_score > 0:
        if verbose2:
            print("Merging c%d and c%d" % (ci, cj))
            print("score: ", max_score)

        df.loc[df["cluster"] == cj, "cluster"] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.nodes[p]["cluster"] == cj:
                graph.nodes[p]["cluster"] = ci
    else:
        print("No Merging")
        print("score: ", max_score)
        print("early stopping")
        print("increase k of k-NN if you want to perform each merging step")

    return df, max_score, ci


def cluster2(
    df,
    k=None,
    knn=None,
    m=30,
    alpha=2.0,
    beta=1,
    m_fact=1e3,
    verbose=False,
    verbose1=True,
    verbose2=True,
    plot=True,
    auto_extract=False,
):
    if knn is None:
        knn = int(round(2 * np.log(len(df))))

    if k is None:
        k = 1
    if verbose:
        print("Building kNN graph (k = %d)..." % knn)
    graph_knn = knn_graph_sym(df, knn, verbose1)

    if plot:
        plot2d_graph(graph_knn, print_clust=False)

    graph_pp = pre_part_graph(graph_knn, m, df, verbose1, plotting=plot)
    if verbose:
        print("flood fill...")

    graph_ff, increased_m = flood_fill(graph_pp, graph_knn, df)

    m = increased_m
    if verbose:
        print("new m: ", m)

    if plot:
        plot2d_graph(graph_ff, print_clust=False)

    dendr_height = {}
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

        dendr_height[m - (i + 1)] = ms

        if plot:
            plot2d_data(df, ci)
    if verbose:
        print("dendr_height", dendr_height)
    res = rebuild_labels(df)

    if auto_extract is True:
        extract_optimal_n_clust(dendr_height, m)

    return res, dendr_height


def connected_components(graph):
    """return generator of connected component of a graph"""
    from collections import deque

    seen = set()

    # for root in range(len(graph)):
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
    g_bis = pd.concat([g["b"], g["a"]], axis=1, keys=["a", "b"])
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


def conn_comp(knn_gr):
    """return list of lists of connected component, e.g. [[0,2], [1,3]], with numbers corresponding to nodes"""

    g1 = prepro_edge(knn_gr)

    return list(connected_components(g1))


def flood_fill(graph, knn_gr, df):
    len_0_clusters = 0
    cl_dict = {
        list(graph.nodes)[i]: graph.nodes[i]["cluster"]
        for i in range(len(graph))
    }
    new_cl_ind = max(cl_dict.values()) + 1
    dic_edge = prepro_edge(knn_gr)

    for num in range(max(cl_dict.values()) + 1):
        points = [
            i for i in list(cl_dict.keys()) if list(cl_dict.values())[i] == num
        ]
        restr_dict = {list(dic_edge.keys())[i]: dic_edge[i] for i in points}
        r_dict = {}

        for i in list(restr_dict.keys()):
            r_dict[i] = [i for i in restr_dict[i] if i in points]

        cc_list = list(connected_components(r_dict))
        print("num_cluster: {0}, len: {1}".format(num, len(cc_list)))
        if len(cc_list) == 1:
            continue
        elif len(cc_list) == 0:
            len_0_clusters += 1
        else:
            # skip the first
            for component in cc_list[1:]:
                print("new index for the component: ", new_cl_ind)
                for el in component:
                    cl_dict[el] = new_cl_ind
                new_cl_ind += 1

    df["cluster"] = list(cl_dict.values())

    for i in range(len(graph)):
        graph.nodes[i]["cluster"] = cl_dict[i]

    increased_m = max(cl_dict.values()) + 1 - len_0_clusters

    return graph, increased_m


def tree_height(h, m):
    sim = {m - 1: (1 / h[m - 1])}
    for i in list(h.keys())[:-1]:
        sim[i - 1] = sim[i] + 1 / h[i - 1]
    return sim


def find_bigger_jump(th, jump):
    lower = list(th.values())[int(len(th) / 2) + 1]
    for i in list(range(int(len(th) / 2), len(th))):
        upper = list(th.values())[i]
        if upper - lower > jump:
            return lower + (upper - lower) / 2
        lower = upper
    return 0


def first_jump_cutoff(th, mult, factor, m):
    half = int(round(len(th) / 2))
    l = reversed(range(m - 1 - half, m - 1))
    half_dict = {j: th[j] for j in l}
    avg = np.mean(list(half_dict.values()))
    res = 0

    while mult > 0:
        res = find_bigger_jump(th, mult * avg)
        if res != 0:
            return res
        else:
            mult = mult / factor


def find_nearest_height(th, value):
    idx = np.searchsorted(list(th.values()), value, side="left")
    el = list(th.values())[idx]
    key_list = [k for (k, v) in th.items() if v == el]
    return key_list[0]


def extract_optimal_n_clust(h, m, f=1000, eta=2):
    th = tree_height(h, m)

    if len(th) <= 3:
        print(
            "insufficient merging steps to perform auto_extract; decrease k and/or increase m"
        )

        return

    fjc = first_jump_cutoff(th, f, eta, m)

    opt_n_clust = find_nearest_height(th, fjc)

    print("Optimal number of clusters: ", opt_n_clust)
