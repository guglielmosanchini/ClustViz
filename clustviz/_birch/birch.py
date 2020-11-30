import numpy as np
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from pyclustering.container.cftree import measurement_type
from pyclustering.cluster.birch import birch as birch_pyclustering
from clustviz._birch.cftree import cftree
from clustviz.utils import COLOR_DICT, FONTSIZE_NORMAL, SIZE_NORMAL


class birch(birch_pyclustering):

    def __init__(self, data, number_clusters, branching_factor=50, max_node_entries=200, diameter=0.5,
                 type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
                 entry_size_limit=500,
                 diameter_multiplier=1.5,
                 ccore=True):

        super().__init__(data, number_clusters, branching_factor, max_node_entries, diameter, type_measurement,
                         entry_size_limit, diameter_multiplier, ccore)

        # otherwise it would refer to pyclustering original cftree
        self.__tree = cftree(branching_factor, max_node_entries, diameter, type_measurement)

    def return_tree(self):
        """return the tree built by the algorithm"""
        return self.__tree

    def process(self, plotting=False):
        """!
        @brief Performs cluster analysis in line with rules of BIRCH algorithm.

        @return (birch) Returns itself (BIRCH instance).

        @see get_clusters()

        """

        self.__insert_data(plotting=plotting)
        print("extracting features")
        self.__extract_features()

        print("features: ", self.__features)

        cf_data = [feature.get_centroid() for feature in self.__features]

        algorithm = agglomerative(cf_data, self.__number_clusters, type_link.SINGLE_LINK).process()
        self.__cf_clusters = algorithm.get_clusters()

        cf_labels = cluster_encoder(type_encoding.CLUSTER_INDEX_LIST_SEPARATION, self.__cf_clusters, cf_data). \
            set_encoding(type_encoding.CLUSTER_INDEX_LABELING).get_clusters()

        self.__clusters = [[] for _ in range(len(self.__cf_clusters))]
        for index_point in range(len(self.__pointer_data)):
            index_cf_entry = np.argmin(np.sum(np.square(
                np.subtract(cf_data, self.__pointer_data[index_point])), axis=1))
            index_cluster = cf_labels[index_cf_entry]
            self.__clusters[index_cluster].append(index_point)

        return self

    def __insert_data(self, plotting=False):
        """!
        @brief Inserts input data to the tree.

        @remark If number of maximum number of entries is exceeded than diameter is increased and tree is rebuilt.

        """

        for index_point in range(0, len(self.__pointer_data)):
            if (index_point != 0) and (plotting is True):
                plot_tree_fin(self.__tree)
                plot_birch_leaves(self.__tree, data=self.__pointer_data)

            print("\n\n")
            print("index: {}".format(index_point))
            point = self.__pointer_data[index_point]
            print("point {}".format(point))
            self.__tree.insert_point(point)

            if self.__tree.amount_entries > self.__entry_size_limit:
                print("rebuilding tree")
                self.__tree = self.__rebuild_tree(index_point)

    def __rebuild_tree(self, index_point):
        """!
        @brief Rebuilt tree in case of maxumum number of entries is exceeded.

        @param[in] index_point (uint): Index of point that is used as end point of re-building.

        @return (cftree) Rebuilt tree with encoded points till specified point from input data space.

        """

        rebuild_result = False
        increased_diameter = self.__tree.threshold * self.__diameter_multiplier

        tree = None

        while rebuild_result is False:
            # increase diameter and rebuild tree
            if increased_diameter == 0.0:
                increased_diameter = 1.0

            # build tree with update parameters
            tree = cftree(self.__tree.branch_factor, self.__tree.max_entries, increased_diameter,
                          self.__tree.type_measurement)

            for index_point in range(0, index_point + 1):
                point = self.__pointer_data[index_point]
                tree.insert_point(point)

                if tree.amount_entries > self.__entry_size_limit:
                    increased_diameter *= self.__diameter_multiplier
                    continue

            # Re-build is successful.
            rebuild_result = True

        return tree


def plot_tree_fin(tree, info=False):
    """
    Plot the final CFtree built by BIRCH. Leaves are colored, and every node displays the
    total number of elements in its child nodes.

    :param tree: tree built during BIRCH algorithm execution.
    :param info: if True, tree height, number of nodes, leaves and entries are printed.
    """

    import graphviz

    height = tree.height

    if info is True:
        print("Tree height is {0}".format(height))
        print("Number of nodes: {0}".format(tree.amount_nodes))
        print("Number of leaves: {0}".format(len(tree.leafes)))
        print("Number of entries: {0}".format(tree.amount_entries))

    node_limit = 2704
    if tree.amount_nodes > node_limit:
        print("Too many nodes, limit is {0}".format(node_limit))

        return

    def feat_create(level_nodes):
        """
        Auxiliary function that returns for each node level the features, the number
        of points and the successors
        """
        features = []
        features_num = []
        succ_num = []
        for el in level_nodes:
            f = el.feature
            features.append(f)
            features_num.append(f.number_points)
            try:
                succ_num.append(len(el.successors))
            except:
                pass

        return features, features_num, succ_num

    # collecting data for each tree level except bottom
    feat = []
    feat_num = []
    succ_num = []
    for lev in range(height):
        (f1, f2, s1) = feat_create(tree.get_level_nodes(lev))
        feat.append(f1)
        feat_num.append(f2)
        succ_num.append(s1)

    # collect data of leaves
    single_entries = []
    for z in tree.get_level_nodes(height - 1):
        sing_ent_prov = []
        for single_entry in z.entries:
            sing_ent_prov.append(single_entry.number_points)
        single_entries.append(sing_ent_prov)

    # creating names for nodes
    prov = (
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c "
        "d e f g h i j k l m n o p q r s t u v w x y z".split(" ")
    )
    lett = []
    for i in range(len(prov)):
        for j in range(len(prov)):
            lett.append(prov[i] + prov[j])

    # creating the tree
    dot = graphviz.Digraph(comment="Clustering")
    # root
    dot.node(lett[0], str(feat_num[0][0]))

    # all other levels
    placeholder = 0

    for level in range(1, height + 1):
        # all levels between root and leaves
        if level != height:
            for q in range(1, len(feat_num[level]) + 1):
                dot.node(lett[placeholder + q], str(feat_num[level][q - 1]))
            placeholder += q
        # leaves with colors
        else:
            for q in range(1, len(single_entries) + 1):
                dot.node(
                    lett[placeholder + q],
                    str(single_entries[q - 1]),
                    color=COLOR_DICT[(q - 1) % len(COLOR_DICT)],
                    style="filled",
                )

    # adding edges between nodes
    a = 0
    b = 0
    # for all nodes except leaves
    for level in range(0, height):
        for num_succs in succ_num[level]:
            for el in range(num_succs):
                dot.edge(lett[a], lett[b + el + 1])
            a += 1
            b += el + 1
    # for leaves
    for i in range(len(single_entries)):
        dot.edge(lett[a], lett[b + i + 1])
        a += 1

    graph = graphviz.Source(dot)  # .view()
    # show tree
    display(graph)


def plot_birch_leaves(tree, data):
    """
    Scatter plot of data point, with colors according to the leaf the belong to. Points in the same entry in a leaf
    are represented by a cross, with the number of points over it.

    :param tree: tree built during BIRCH algorithm execution.
    :param data: input data as array of list of list

    """

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(14, 6))

    # plot every point in white
    plt.scatter(
        np.array(data)[:, 0],
        np.array(data)[:, 1],
        s=300,
        color="white",
        edgecolor="black",
    )

    # for every leaf
    for i, el in enumerate(tree.get_level_nodes(tree.height - 1)):
        # for every entry in the leaf
        for entry in el.entries:
            # if it is a single point, plot it with its color
            if entry.number_points == 1:
                plt.scatter(
                    entry.linear_sum[0],
                    entry.linear_sum[1],
                    color=COLOR_DICT[i % len(COLOR_DICT)],
                    s=300,
                    edgecolor="black",
                )
            # else, plot the entry centroid as a cross and leave the points white
            else:
                plt.scatter(
                    entry.get_centroid()[0],
                    entry.get_centroid()[1],
                    color=COLOR_DICT[i % len(COLOR_DICT)],
                    marker="X",
                    s=200,
                )
                plt.annotate(
                    entry.number_points,
                    (entry.get_centroid()[0], entry.get_centroid()[1]),
                    fontsize=18,
                )

    # plot indexes of points in plot
    for i, txt in enumerate(range(len(data))):
        ax.annotate(
            txt,
            (np.array(data)[:, 0][i], np.array(data)[:, 1][i]),
            fontsize=FONTSIZE_NORMAL,
            size=SIZE_NORMAL,
            ha="center",
            va="center",
        )

    plt.show()

