"""!

@brief Cluster analysis algorithm: BIRCH
@details Implementation based on paper @cite article::birch::1.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2019
@copyright GNU Public License

@cond GNU_PUBLIC_LICENSE
    PyClustering is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyClustering is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond

"""

from pyclustering.utils import linear_sum, square_sum

from pyclustering.cluster.encoder import type_encoding

from algorithms.birch.cftree import cftree, measurement_type

from pyclustering.container.cftree import cfentry


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

    if tree.amount_nodes > 2704:
        print("Too many nodes, limit is 2704")

        return

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
                    color=colors[(q - 1) % 17],
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
                    color=colors[i % 18],
                    s=300,
                    edgecolor="black",
                )
            # else, plot the entry centroid as a cross and leave the points white
            else:
                plt.scatter(
                    entry.get_centroid()[0],
                    entry.get_centroid()[1],
                    color=colors[i % 18],
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
            fontsize=10,
            size=10,
            ha="center",
            va="center",
        )

    plt.show()


class birch:
    """!
    @brief Class represents clustering algorithm BIRCH.

    Example how to extract clusters from 'OldFaithful' sample using BIRCH algorithm:
    @code
        from pyclustering.cluster.birch import birch, measurement_type
        from pyclustering.cluster import cluster_visualizer
        from pyclustering.utils import read_sample
        from pyclustering.samples.definitions import FAMOUS_SAMPLES

        # Sample for cluster analysis (represented by list)
        sample = read_sample(FAMOUS_SAMPLES.SAMPLE_OLD_FAITHFUL)

        # Create BIRCH algorithm
        birch_instance = birch(sample, 2)

        # Cluster analysis
        birch_instance.process()

        # Obtain results of clustering
        clusters = birch_instance.get_clusters()

        # Visualize allocated clusters
        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, sample)
        visualizer.show()
    @endcode

    """

    def __init__(
        self,
        data,
        number_clusters,
        branching_factor=5,
        max_node_entries=5,
        initial_diameter=0.1,
        type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
        entry_size_limit=200,
        diameter_multiplier=1.5,
        ccore=True,
    ):
        """!
        @brief Constructor of clustering algorithm BIRCH.

        @param[in] data (list): Input data presented as list of points (objects), where each point
        should be represented by list or tuple.
        @param[in] number_clusters (uint): Number of clusters that should be allocated.
        @param[in] branching_factor (uint): Maximum number of successor that might be contained
        by each non-leaf node in CF-Tree.
        @param[in] max_node_entries (uint): Maximum number of entries that might be contained
        by each leaf node in CF-Tree.
        @param[in] initial_diameter (double): Initial diameter that used for CF-Tree construction,
        it can be increase if entry_size_limit is exceeded.
        @param[in] type_measurement (measurement_type): Type measurement used for calculation distance metrics.
        @param[in] entry_size_limit (uint): Maximum number of entries that can be stored in CF-Tree,
        if it is exceeded during creation then diameter is increased and CF-Tree is rebuilt.
        @param[in] diameter_multiplier (double): Multiplier that is used for increasing diameter
         when entry_size_limit is exceeded.
        @param[in] ccore (bool): If True than CCORE (C++ part of the library) will be used for solving the problem.

        @remark Despite eight arguments only the first two are mandatory, others can be ommitted.
        In this case default values are used for instance creation.

        """

        self.__pointer_data = data
        self.__number_clusters = number_clusters

        self.__measurement_type = type_measurement
        self.__entry_size_limit = entry_size_limit
        self.__diameter_multiplier = diameter_multiplier
        self.__ccore = ccore

        self.__verify_arguments()

        self.__features = None
        self.__tree = cftree(
            branching_factor,
            max_node_entries,
            initial_diameter,
            type_measurement,
        )

        self.__clusters = []
        self.__noise = []

    def process(self, plotting=False):
        """!
        @brief Performs cluster analysis in line with rules of BIRCH algorithm.

        @return (birch) Returns itself (BIRCH instance).

        @see get_clusters()

        """

        self.__insert_data(plotting=plotting)
        self.__extract_features()

        # in line with specification modify hierarchical algorithm should be used for further clustering
        current_number_clusters = len(self.__features)

        while current_number_clusters > self.__number_clusters:
            indexes = self.__find_nearest_cluster_features()

            self.__features[indexes[0]] += self.__features[indexes[1]]
            self.__features.pop(indexes[1])

            current_number_clusters = len(self.__features)

        # decode data
        self.__decode_data()
        return self

    def return_tree(self):
        """modified from the original version by me on 24.11.19"""
        return self.__tree

    def get_clusters(self):
        """!
        @brief Returns list of allocated clusters, each cluster contains indexes of objects in list of data.

        @remark Allocated noise can be returned only after data processing (use method process() before).
        Otherwise empty list is returned.

        @return (list) List of allocated clusters.

        @see process()
        @see get_noise()

        """

        return self.__clusters

    def get_cluster_encoding(self):
        """!
        @brief Returns clustering result representation type that indicate how clusters are encoded.

        @return (type_encoding) Clustering result representation.

        @see get_clusters()

        """

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION

    def __verify_arguments(self):
        """!
        @brief Verify input parameters for the algorithm and throw exception in case of incorrectness.

        """
        if len(self.__pointer_data) == 0:
            raise ValueError(
                "Input data is empty (size: '%d')." % len(self.__pointer_data)
            )

        if self.__number_clusters <= 0:
            raise ValueError(
                "Amount of cluster (current value: '%d') for allocation should be greater than 0."
                % self.__number_clusters
            )

        if self.__entry_size_limit <= 0:
            raise ValueError(
                "Limit entry size (current value: '%d') should be greater than 0."
                % self.__entry_size_limit
            )

    def __extract_features(self):
        """!
        @brief Extracts features from CF-tree cluster.

        """
        print("extracting features")
        self.__features = []

        if len(self.__tree.leafes) == 1:
            # parameters are too general, copy all entries
            for entry in self.__tree.leafes[0].entries:
                self.__features.append(entry)

        else:
            # copy all leaf clustering features
            for node in self.__tree.leafes:
                self.__features.append(node.feature)

    def __decode_data(self):
        """!
        @brief Decodes data from CF-tree features.

        """

        self.__clusters = [[] for _ in range(self.__number_clusters)]
        self.__noise = []

        for index_point in range(0, len(self.__pointer_data)):
            (_, cluster_index) = self.__get_nearest_feature(
                self.__pointer_data[index_point], self.__features
            )

            self.__clusters[cluster_index].append(index_point)

    def __insert_data(self, plotting=False):
        """!
        @brief Inserts input data to the tree.

        @remark If number of maximum number of entries is exceeded than diameter is increased and tree is rebuilt.

        """

        for index_point in range(0, len(self.__pointer_data)):
            if (index_point != 0) and (plotting is True):
                plot_tree_fin(self.__tree)
                plot_birch_leaves(self.__tree, data=self.__pointer_data)

            print("\n")
            print("\n")
            print("index: ", index_point)
            point = self.__pointer_data[index_point]
            print("point ", point)
            self.__tree.insert_cluster([point])

            if self.__tree.amount_entries > self.__entry_size_limit:
                print("rebuilding tree")
                self.__tree = self.__rebuild_tree(index_point)

        # self.__tree.show_feature_distribution(self.__pointer_data);

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
            tree = cftree(
                self.__tree.branch_factor,
                self.__tree.max_entries,
                increased_diameter,
                self.__tree.type_measurement,
            )

            for index_point in range(0, index_point + 1):
                point = self.__pointer_data[index_point]
                tree.insert_cluster([point])

                if tree.amount_entries > self.__entry_size_limit:
                    increased_diameter *= self.__diameter_multiplier
                    continue

            # Re-build is successful.
            rebuild_result = True

        return tree

    def __find_nearest_cluster_features(self):
        """!
        @brief Find pair of nearest CF entries.

        @return (list) List of two nearest entries that are represented by list [index_point1, index_point2].

        """

        minimum_distance = float("Inf")
        index1 = 0
        index2 = 0

        print("\n")
        for index_candidate1 in range(0, len(self.__features)):
            feature1 = self.__features[index_candidate1]
            for index_candidate2 in range(
                index_candidate1 + 1, len(self.__features)
            ):
                feature2 = self.__features[index_candidate2]

                distance = feature1.get_distance(
                    feature2, self.__measurement_type
                )
                if distance < minimum_distance:
                    minimum_distance = distance

                    index1 = index_candidate1
                    index2 = index_candidate2

        print("nearest features are: ", index1, index2)
        return [index1, index2]

    def __get_nearest_feature(self, point, feature_collection):
        """!
        @brief Find nearest entry for specified point.

        @param[in] point (list): Pointer to point from input dataset.
        @param[in] feature_collection (list): Feature collection that is used for
        obtaining nearest feature for the specified point.

        @return (double, uint) Tuple of distance to nearest entry to the specified point and index of that entry.

        """

        minimum_distance = float("Inf")
        index_nearest_feature = -1

        for index_entry in range(0, len(feature_collection)):
            point_entry = cfentry(1, linear_sum([point]), square_sum([point]))

            distance = feature_collection[index_entry].get_distance(
                point_entry, self.__measurement_type
            )
            if distance < minimum_distance:
                minimum_distance = distance
                index_nearest_feature = index_entry

        return minimum_distance, index_nearest_feature
