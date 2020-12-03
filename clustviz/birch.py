import numpy as np
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from pyclustering.cluster.birch import birch as birch_pyclustering
from pyclustering.cluster import cluster_visualizer

from pyclustering.container.cftree import leaf_node, non_leaf_node, cfnode_type, measurement_type
from pyclustering.container.cftree import cftree as cftree_pyclustering

from clustviz.utils import COLOR_DICT, annotate_points


class cftree(cftree_pyclustering):

    def insert(self, entry):
        """
        Insert clustering feature to the tree.

        :param entry: clustering feature that should be inserted.

        """
        print("insert entry")
        if self.__root is None:
            print("first time")
            node = leaf_node(entry, None, [entry])

            self.__root = node
            self.__leafes.append(node)

            # Update statistics
            self.__amount_entries += 1
            self.__amount_nodes += 1
            self.__height += 1  # root has successor now
        else:
            print("recursive insert")
            child_node_updation = self.__recursive_insert(entry, self.__root)
            if child_node_updation is True:
                print("try merge_nearest_successors")
                # Splitting has been finished, check for possibility to merge (at least we have already two children).
                if self.__merge_nearest_successors(self.__root) is True:
                    self.__amount_nodes -= 1

    def __recursive_insert(self, entry, search_node) -> bool:
        """
        Recursive insert of the entry to the tree.
        It performs all required procedures during insertion such as splitting, merging.

        :param entry: clustering feature.
        :param search_node: node from that insertion should be started.

        :return: True if number of nodes at the below level is changed, otherwise False.

        """

        # Non-leaf node
        if search_node.type == cfnode_type.CFNODE_NONLEAF:
            print("insert for non-leaf")
            return self.__insert_for_noneleaf_node(entry, search_node)

        # Leaf is reached
        else:
            print("insert for leaf")
            return self.__insert_for_leaf_node(entry, search_node)

    def __insert_for_leaf_node(self, entry, search_node):
        """
        Recursive insert entry from leaf node to the tree.

        :param entry: Clustering feature.
        :param: search_node (cfnode): None-leaf node from that insertion should be started.

        @return (bool) True if number of nodes at the below level is changed, otherwise False.

        """

        node_amount_updation = False

        # Try to absorb by the entity
        index_nearest_entry = search_node.get_nearest_index_entry(
            entry, self.__type_measurement
        )
        nearest_entry = search_node.entries[index_nearest_entry]
        merged_entry = nearest_entry + entry
        print("index_nearest_entry", index_nearest_entry)
        print("nearest entry", nearest_entry)

        print("diam:", merged_entry.get_diameter())
        # Otherwise try to add new entry
        if merged_entry.get_diameter() > self.__threshold:
            print("diam greater than threshold")
            # If it's not exceeded append entity and update feature of the leaf node.
            search_node.insert_entry(entry)

            # Otherwise current node should be splitted

            if len(search_node.entries) > self.__max_entries:
                print("node has to split")
                self.__split_procedure(search_node)
                node_amount_updation = True

            # Update statistics
            self.__amount_entries += 1

        else:
            print("diam ok")
            search_node.entries[index_nearest_entry] = merged_entry
            search_node.feature += entry

        return node_amount_updation

    def __insert_for_noneleaf_node(self, entry, search_node):
        """!
        @brief Recursive insert entry from none-leaf node to the tree.

        @param[in] entry (cfentry): Clustering feature.
        @param[in] search_node (cfnode): None-leaf node from that insertion should be started.

        @return (bool) True if number of nodes at the below level is changed, otherwise False.

        """

        node_amount_updation = False

        min_key = lambda child_node: child_node.get_distance(
            search_node, self.__type_measurement
        )
        nearest_child_node = min(search_node.successors, key=min_key)
        print("nearestchildnode: ", nearest_child_node)
        print("recursive insert in !!!insert_for_nonleaf!!!")
        child_node_updation = self.__recursive_insert(
            entry, nearest_child_node
        )

        # Update clustering feature of none-leaf node.
        search_node.feature += entry

        # Check branch factor, probably some leaf has been splitted and threshold has been exceeded.
        if len(search_node.successors) > self.__branch_factor:
            print("over branch_factor ")

            # Check if it's aleady root then new root should be created (height is increased in this case).
            if search_node is self.__root:
                print("height increases")
                self.__root = non_leaf_node(
                    search_node.feature, None, [search_node]
                )
                search_node.parent = self.__root

                # Update statistics
                self.__amount_nodes += 1
                self.__height += 1

            print("split non-leaf node")
            [new_node1, new_node2] = self.__split_nonleaf_node(search_node)

            # Update parent list of successors
            parent = search_node.parent
            parent.successors.remove(search_node)
            parent.successors.append(new_node1)
            parent.successors.append(new_node2)

            # Update statistics
            self.__amount_nodes += 1
            node_amount_updation = True

        elif child_node_updation is True:
            # Splitting has been finished, check for possibility to merge (at least we have already two children).
            if self.__merge_nearest_successors(search_node) is True:
                self.__amount_nodes -= 1

        return node_amount_updation

    def __merge_nearest_successors(self, node):
        """!
        @brief Find nearest sucessors and merge them.

        @param[in] node (non_leaf_node): Node whose two nearest successors should be merged.

        @return (bool): True if merging has been successfully performed, otherwise False.

        """

        merging_result = False

        if node.successors[0].type == cfnode_type.CFNODE_NONLEAF:
            [
                nearest_child_node1,
                nearest_child_node2,
            ] = node.get_nearest_successors(self.__type_measurement)

            if (
                    len(nearest_child_node1.successors)
                    + len(nearest_child_node2.successors)
                    <= self.__branch_factor
            ):
                node.successors.remove(nearest_child_node2)
                if nearest_child_node2.type == cfnode_type.CFNODE_LEAF:
                    self.__leafes.remove(nearest_child_node2)

                nearest_child_node1.merge(nearest_child_node2)

                merging_result = True

        if merging_result is True:
            print("merging successful")
        else:
            print("merging not successful")
        return merging_result

    def __split_leaf_node(self, node):
        """!
        @brief Performs splitting of the specified leaf node.

        @param[in] node (leaf_node): Leaf node that should be splitted.

        @return (list) New pair of leaf nodes [leaf_node1, leaf_node2].

        @warning Splitted node is transformed to non_leaf.

        """
        print("split leaf")
        # search farthest pair of entries
        [farthest_entity1, farthest_entity2] = node.get_farthest_entries(
            self.__type_measurement
        )
        print("farthest1 ", farthest_entity1)
        print("farthest2 ", farthest_entity2)

        # create new nodes
        new_node1 = leaf_node(
            farthest_entity1, node.parent, [farthest_entity1]
        )
        new_node2 = leaf_node(
            farthest_entity2, node.parent, [farthest_entity2]
        )

        # re-insert other entries
        for entity in node.entries:
            if (entity is not farthest_entity1) and (
                    entity is not farthest_entity2
            ):
                distance1 = new_node1.feature.get_distance(
                    entity, self.__type_measurement
                )
                distance2 = new_node2.feature.get_distance(
                    entity, self.__type_measurement
                )

                if distance1 < distance2:
                    new_node1.insert_entry(entity)
                else:
                    new_node2.insert_entry(entity)

        print("new_node1 ", new_node1)
        print("new_node2 ", new_node2)

        return [new_node1, new_node2]

    def show_feature_distribution(self, data=None):
        """!
         @brief Shows feature distribution.
         @details Only features in 1D, 2D, 3D space can be visualized.

         @param[in] data (list): List of points that will be used for visualization,
         if it not specified than feature will be displayed only.

         """
        visualizer = cluster_visualizer()

        print("amount of nodes: ", self.__amount_nodes)

        if data is not None:
            visualizer.append_cluster(data, marker="x")

        for level in range(0, self.height):
            level_nodes = self.get_level_nodes(level)

            centers = [node.feature.get_centroid() for node in level_nodes]
            visualizer.append_cluster(
                centers, None, markersize=(self.height - level + 1) * 5
            )

        visualizer.show()


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
    annotate_points(annotations=range(len(data)), points=np.array(data), ax=ax)

    plt.show()
