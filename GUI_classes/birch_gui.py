from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from pyclustering.cluster import cluster_visualizer

from GUI_classes.utils_gui import choose_dataset, pause_execution

from algorithms.birch.cftree import cfnode_type, leaf_node, non_leaf_node
from pyclustering.cluster.encoder import type_encoding
from pyclustering.container.cftree import cfentry, measurement_type
from pyclustering.utils import linear_sum, square_sum

import numpy as np
import graphviz
from shutil import rmtree

from GUI_classes.generic_gui import StartingGui

import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ["PATH"] += os.pathsep + '/usr/local/bin'

class BIRCH_class(StartingGui):
    def __init__(self):
        super(BIRCH_class, self).__init__(name="BIRCH", twinx=False, first_plot=True, second_plot=False,
                                          function=self.start_BIRCH, extract=False, stretch_plot=False)

        # self.canvas_graphviz = FigureCanvas(Figure(figsize=(12, 5)))
        # self.ax_gv = self.canvas_up.figure.subplots()
        # self.ax_gv.set_xticks([], [])
        # self.ax_gv.set_yticks([], [])

        self.label_graphviz = QLabel(self)
        self.label_graphviz.setFixedSize(1100, 250)
        self.gridlayout.addWidget(self.label_graphviz, 1, 1)

    def start_BIRCH(self):

        self.ax1.cla()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))
        QCoreApplication.processEvents()

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_clust = int(self.line_edit_n_clust.text())
        self.branching_factor = int(self.line_edit_branching_factor.text())
        self.initial_diameter = float(self.line_edit_initial_diameter.text())
        self.max_node_entries = int(self.line_edit_max_node_entries.text())
        self.n_points = int(self.line_edit_np.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        self.button_run.setEnabled(False)
        self.checkbox_saveimg.setEnabled(False)
        self.button_delete_pics.setEnabled(False)
        self.slider.setEnabled(False)

        if self.first_run_occurred is True:
            self.ind_run += 1
            self.ind_extr_fig = 0
            if self.save_plots is True:
                self.checkBoxChangedAction(self.checkbox_saveimg.checkState())
        else:
            if Qt.Checked == self.checkbox_saveimg.checkState():
                self.first_run_occurred = True
                self.checkBoxChangedAction(self.checkbox_saveimg.checkState())

        self.checkbox_gif.setEnabled(False)

        birch_instance = birch_gui(self.X.tolist(), self.n_clust, initial_diameter=self.initial_diameter,
                                   max_node_entries=self.max_node_entries, branching_factor=self.branching_factor,
                                   log=self.log, ax=self.ax1, canvas=self.canvas_up, save_fig=self.save_plots,
                                   ind_run=self.ind_run, delay=self.delay, label_graphviz=self.label_graphviz)

        birch_instance.process(plotting=True)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)


def plot_tree_fin_gui(tree, log, ind_run, ind_fig, label_graphviz, save_plots=False, info=True):
    """
    Plot the final CFtree built by BIRCH. Leaves are colored, and every node displays the
    total number of elements in its child nodes.

    :param tree: tree built during BIRCH algorithm execution.
    :param info: if True, tree height, number of nodes, leaves and entries are printed.
    """

    height = tree.height

    if info is True:
        log.appendPlainText("Tree height is {0}".format(height))
        log.appendPlainText("Number of nodes: {0}".format(tree.amount_nodes))
        log.appendPlainText("Number of leaves: {0}".format(len(tree.leafes)))
        log.appendPlainText("Number of entries: {0}".format(tree.amount_entries))

    if tree.amount_nodes > 2704:
        log.appendPlainText("Too many nodes, limit is 2704")

        return

    colors = {0: "seagreen", 1: 'forestgreen', 2: 'yellow', 3: 'grey', 4: 'pink', 5: 'turquoise',
              6: 'orange', 7: 'purple', 8: 'yellowgreen', 9: 'red', 10: 'cyan',
              11: 'tan', 12: 'plum', 13: 'rosybrown', 14: 'lightblue', 15: "khaki",
              16: "gainsboro", 17: "peachpuff"}

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
    prov = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c " \
           "d e f g h i j k l m n o p q r s t u v w x y z".split(" ")
    lett = []
    for i in range(len(prov)):
        for j in range(len(prov)):
            lett.append(prov[i] + prov[j])

    # creating the tree
    dot = graphviz.Digraph(comment='Clustering', format="png")
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
                dot.node(lett[placeholder + q], str(single_entries[q - 1]),
                         color=colors[(q - 1) % 17], style="filled")

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

    # graph = graphviz.Source(dot)
    # graph.view()
    graph_path = dot.render(filename='./Images/BIRCH_{:02}/graph_{:02}'.format(ind_run, ind_fig))

    pixmap = QPixmap(graph_path)
    label_graphviz.setScaledContents(True)
    label_graphviz.setPixmap(pixmap)

    folder = './Images/BIRCH_{:02}'.format(ind_run)
    if save_plots is False:
        rmtree(folder)

    QCoreApplication.processEvents()


def plot_birch_leaves_gui(tree, data, ax, canvas, ind_run, ind_fig, name="BIRCH", save_plots=False):
    """
    Scatter plot of data point, with colors according to the leaf the belong to. Points in the same entry in a leaf
    are represented by a cross, with the number of points over it.

    :param tree: tree built during BIRCH algorithm execution.
    :param data: input data as array of list of list

    """
    ax.clear()
    if ind_fig is not None:
        ax.set_title("{} run number {}".format(name, ind_fig + 1))
    else:
        ax.set_title("{} final clustering".format(name))

    colors = {0: "seagreen", 1: 'forestgreen', 2: 'yellow', 3: 'grey', 4: 'pink', 5: 'turquoise',
              6: 'orange', 7: 'purple', 8: 'yellowgreen', 9: 'red', 10: 'cyan',
              11: 'tan', 12: 'plum', 13: 'rosybrown', 14: 'lightblue', 15: "khaki",
              16: "gainsboro", 17: "peachpuff"}

    # plot every point in white with white edgecolor (invisible)
    ax.scatter(np.array(data)[:, 0], np.array(data)[:, 1], s=300, color="white", edgecolor="white")

    # for every leaf
    for i, el in enumerate(tree.get_level_nodes(tree.height - 1)):
        # for every entry in the leaf
        for entry in el.entries:
            # if it is a single point, plot it with its color
            if entry.number_points == 1:
                ax.scatter(entry.linear_sum[0], entry.linear_sum[1], color=colors[i % 18], s=300, edgecolor="black")
            # else, plot the entry centroid as a cross and leave the points white
            else:
                ax.scatter(entry.get_centroid()[0], entry.get_centroid()[1], color=colors[i % 18], marker="X", s=200)
                ax.annotate(entry.number_points, (entry.get_centroid()[0], entry.get_centroid()[1]), fontsize=18)

    # plot indexes of points in plot
    for i, txt in enumerate(range(len(data))):
        ax.annotate(txt, (np.array(data)[:, 0][i], np.array(data)[:, 1][i]),
                    fontsize=10, size=10, ha='center', va='center')

    canvas.draw()

    if save_plots is True:
        if ind_fig is not None:
            canvas.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(name, ind_run, ind_fig))
        else:
            canvas.figure.savefig('./Images/{}_{:02}/fig_fin.png'.format(name, ind_run))

    QCoreApplication.processEvents()


class birch_gui:

    def __init__(self, data, number_clusters, branching_factor, max_node_entries, initial_diameter,
                 log, ax, canvas, save_fig, ind_run, delay, label_graphviz,
                 type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
                 entry_size_limit=200, diameter_multiplier=1.5, ccore=True):
        """!
        @brief Constructor of clustering algorithm BIRCH.

        @param[in] data (list): Input data presented as list of points (objects), where each point should be
        represented by list or tuple.
        @param[in] number_clusters (uint): Number of clusters that should be allocated.
        @param[in] branching_factor (uint): Maximum number of successor that might be contained by each non-leaf
        node in CF-Tree.
        @param[in] max_node_entries (uint): Maximum number of entries that might be contained by each leaf node
        in CF-Tree.
        @param[in] initial_diameter (double): Initial diameter that used for CF-Tree construction, it can be
        increase if entry_size_limit is exceeded.
        @param[in] type_measurement (measurement_type): Type measurement used for calculation distance metrics.
        @param[in] entry_size_limit (uint): Maximum number of entries that can be stored in CF-Tree, if it
        is exceeded during creation then diameter is increased and CF-Tree is rebuilt.
        @param[in] diameter_multiplier (double): Multiplier that is used for increasing diameter when
        entry_size_limit is exceeded.
        @param[in] ccore (bool): If True than CCORE (C++ part of the library) will be used for solving the problem.

        @remark Despite eight arguments only the first two are mandatory, others can be omitted.
        In this case default values are used for instance creation.

        """
        self.log = log
        self.ax = ax
        self.canvas = canvas
        self.save_fig = save_fig
        self.ind_run = ind_run
        self.delay = delay
        self.label_graphviz = label_graphviz

        self.__pointer_data = data
        self.__number_clusters = number_clusters

        self.__measurement_type = type_measurement
        self.__entry_size_limit = entry_size_limit
        self.__diameter_multiplier = diameter_multiplier
        self.__ccore = ccore

        self.__verify_arguments()

        self.__features = None
        self.__tree = cftree_gui(branch_factor=branching_factor, max_entries=max_node_entries,
                                 threshold=initial_diameter,
                                 log=self.log, type_measurement=type_measurement)

        self.__clusters = []
        self.__noise = []

    def process(self, plotting=False):
        """!
        @brief Performs cluster analysis in line with rules of BIRCH algorithm.

        @return (birch) Returns itself (BIRCH instance).

        @see get_clusters()

        """
        self.index_for_saving_plot = 0
        self.__insert_data(plotting=plotting)

        plot_tree_fin_gui(tree=self.__tree, log=self.log, ind_run=self.ind_run,
                          ind_fig=self.index_for_saving_plot, label_graphviz=self.label_graphviz,
                          save_plots=self.save_fig)
        plot_birch_leaves_gui(tree=self.__tree, data=self.__pointer_data, ax=self.ax, canvas=self.canvas,
                              ind_run=self.ind_run, ind_fig=self.index_for_saving_plot,
                              save_plots=self.save_fig)

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
            raise ValueError("Input data is empty (size: '%d')." % len(self.__pointer_data))

        if self.__number_clusters <= 0:
            raise ValueError("Amount of cluster (current value: '%d') for allocation should be greater than 0." %
                             self.__number_clusters)

        if self.__entry_size_limit <= 0:
            raise ValueError("Limit entry size (current value: '%d') should be greater than 0." %
                             self.__entry_size_limit)

    def __extract_features(self):
        """!
        @brief Extracts features from CF-tree cluster.

        """
        self.log.appendPlainText("extracting features")
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
            (_, cluster_index) = self.__get_nearest_feature(self.__pointer_data[index_point], self.__features)

            self.__clusters[cluster_index].append(index_point)

    def __insert_data(self, plotting=False):
        """!
        @brief Inserts input data to the tree.

        @remark If number of maximum number of entries is exceeded than diameter is increased and tree is rebuilt.

        """

        for index_point in range(0, len(self.__pointer_data)):
            if (index_point != 0) and (plotting is True):
                if self.delay != 0:
                    pause_execution(self.delay)
                plot_tree_fin_gui(tree=self.__tree, log=self.log, ind_run=self.ind_run,
                                  ind_fig=self.index_for_saving_plot, label_graphviz=self.label_graphviz,
                                  save_plots=self.save_fig)
                plot_birch_leaves_gui(tree=self.__tree, data=self.__pointer_data, ax=self.ax, canvas=self.canvas,
                                      ind_run=self.ind_run, ind_fig=self.index_for_saving_plot,
                                      save_plots=self.save_fig)
                self.index_for_saving_plot += 1

            self.log.appendPlainText("")
            self.log.appendPlainText("index: {}".format(index_point))
            point = self.__pointer_data[index_point]
            self.log.appendPlainText("point [{}, {}]".format(round(point[0], 2), round(point[1], 2)))
            self.__tree.insert_cluster([point])

            if self.__tree.amount_entries > self.__entry_size_limit:
                self.log.appendPlainText("rebuilding tree")
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
            tree = cftree_gui(branch_factor=self.__tree.branch_factor, max_entries=self.__tree.max_entries,
                              log=self.log, threshold=increased_diameter,
                              type_measurement=self.__tree.type_measurement)

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

        self.log.appendPlainText("")
        for index_candidate1 in range(0, len(self.__features)):
            feature1 = self.__features[index_candidate1]
            for index_candidate2 in range(index_candidate1 + 1, len(self.__features)):
                feature2 = self.__features[index_candidate2]

                distance = feature1.get_distance(feature2, self.__measurement_type)
                if distance < minimum_distance:
                    minimum_distance = distance

                    index1 = index_candidate1
                    index2 = index_candidate2

        self.log.appendPlainText("nearest features: {}, {}".format(index1, index2))
        return [index1, index2]

    def __get_nearest_feature(self, point, feature_collection):
        """!
        @brief Find nearest entry for specified point.

        @param[in] point (list): Pointer to point from input dataset.
        @param[in] feature_collection (list): Feature collection that is used for obtaining nearest feature for the specified point.

        @return (double, uint) Tuple of distance to nearest entry to the specified point and index of that entry.

        """

        minimum_distance = float("Inf")
        index_nearest_feature = -1

        for index_entry in range(0, len(feature_collection)):
            point_entry = cfentry(1, linear_sum([point]), square_sum([point]))

            distance = feature_collection[index_entry].get_distance(point_entry, self.__measurement_type)
            if distance < minimum_distance:
                minimum_distance = distance
                index_nearest_feature = index_entry

        return minimum_distance, index_nearest_feature


class cftree_gui:
    """!
    @brief CF-Tree representation.
    @details A CF-tree is a height-balanced tree with two parameters: branching factor and threshold.

    """

    @property
    def root(self):
        """!
        @return (cfnode) Root of the tree.

        """
        return self.__root

    @property
    def leafes(self):
        """!
        @return (list) List of all non-leaf nodes in the tree.

        """
        return self.__leafes

    @property
    def amount_nodes(self):
        """!
        @return (unit) Number of nodes (leaf and non-leaf) in the tree.

        """
        return self.__amount_nodes

    @property
    def amount_entries(self):
        """!
        @return (uint) Number of entries in the tree.

        """
        return self.__amount_entries

    @property
    def height(self):
        """!
        @return (uint) Height of the tree.

        """
        return self.__height

    @property
    def branch_factor(self):
        """!
        @return (uint) Branching factor of the tree.
        @details Branching factor defines maximum number of successors in each non-leaf node.

        """
        return self.__branch_factor

    @property
    def threshold(self):
        """!
        @return (double) Threshold of the tree that represents maximum diameter of sub-clusters that is formed
        by leaf node entries.

        """
        return self.__threshold

    @property
    def max_entries(self):
        """!
        @return (uint) Maximum number of entries in each leaf node.

        """
        return self.__max_entries

    @property
    def type_measurement(self):
        """!
        @return (measurement_type) Type that is used for measuring.

        """
        return self.__type_measurement

    def __init__(self, branch_factor, max_entries, threshold, log,
                 type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE):
        """!
        @brief Create CF-tree.

        @param[in] branch_factor (uint): Maximum number of children for non-leaf nodes.
        @param[in] max_entries (uint): Maximum number of entries for leaf nodes.
        @param[in] threshold (double): Maximum diameter of feature clustering for each leaf node.
        @param[in] type_measurement (measurement_type): Measurement type that is used for calculation distance metrics.

        """

        self.__root = None
        self.log = log

        self.__branch_factor = branch_factor  # maximum number of children
        if self.__branch_factor < 2:
            self.__branch_factor = 2

        self.__threshold = threshold  # maximum diameter of sub-clusters stored at the leaf nodes
        self.__max_entries = max_entries

        self.__leafes = []

        self.__type_measurement = type_measurement

        # statistics
        self.__amount_nodes = 0  # root, despite it can be None.
        self.__amount_entries = 0
        self.__height = 0  # tree size with root.

    def get_level_nodes(self, level):
        """!
        @brief Traverses CF-tree to obtain nodes at the specified level.

        @param[in] level (uint): CF-tree level from that nodes should be returned.

        @return (list) List of CF-nodes that are located on the specified level of the CF-tree.

        """

        level_nodes = []
        if level < self.__height:
            level_nodes = self.__recursive_get_level_nodes(level, self.__root)

        return level_nodes

    def __recursive_get_level_nodes(self, level, node):
        """!
        @brief Traverses CF-tree to obtain nodes at the specified level recursively.

        @param[in] level (uint): Current CF-tree level.
        @param[in] node (cfnode): CF-node from that traversing is performed.

        @return (list) List of CF-nodes that are located on the specified level of the CF-tree.

        """

        level_nodes = []
        if level is 0:
            level_nodes.append(node)

        else:
            for sucessor in node.successors:
                level_nodes += self.__recursive_get_level_nodes(level - 1, sucessor)

        return level_nodes

    def insert_cluster(self, cluster):
        """!
        @brief Insert cluster that is represented as list of points where each point is represented by list of coordinates.
        @details Clustering feature is created for that cluster and inserted to the tree.

        @param[in] cluster (list): Cluster that is represented by list of points that should be inserted to the tree.

        """
        self.log.appendPlainText("insert cluster")
        entry = cfentry(len(cluster), linear_sum(cluster), square_sum(cluster))
        self.insert(entry)

    def insert(self, entry):
        """!
        @brief Insert clustering feature to the tree.

        @param[in] entry (cfentry): Clustering feature that should be inserted.

        """
        self.log.appendPlainText("insert entry")
        if self.__root is None:
            self.log.appendPlainText("first time")
            node = leaf_node(entry, None, [entry], None)

            self.__root = node
            self.__leafes.append(node)

            # Update statistics
            self.__amount_entries += 1
            self.__amount_nodes += 1
            self.__height += 1  # root has successor now
        else:
            self.log.appendPlainText("recursive insert")
            child_node_updation = self.__recursive_insert(entry, self.__root)
            if child_node_updation is True:
                self.log.appendPlainText("try merge_nearest_successors")
                # Splitting has been finished, check for possibility to merge (at least we have already two children).
                if self.__merge_nearest_successors(self.__root) is True:
                    self.__amount_nodes -= 1

    def find_nearest_leaf(self, entry, search_node=None):
        """!
        @brief Search nearest leaf to the specified clustering feature.

        @param[in] entry (cfentry): Clustering feature.
        @param[in] search_node (cfnode): Node from that searching should be started, if None then search process will be started for the root.

        @return (leaf_node) Nearest node to the specified clustering feature.

        """

        if search_node is None:
            search_node = self.__root

        nearest_node = search_node

        if search_node.type == cfnode_type.CFNODE_NONLEAF:
            min_key = lambda child_node: child_node.feature.get_distance(entry, self.__type_measurement)
            nearest_child_node = min(search_node.successors, key=min_key)

            nearest_node = self.find_nearest_leaf(entry, nearest_child_node)

        return nearest_node

    def __recursive_insert(self, entry, search_node):
        """!
        @brief Recursive insert of the entry to the tree.
        @details It performs all required procedures during insertion such as splitting, merging.

        @param[in] entry (cfentry): Clustering feature.
        @param[in] search_node (cfnode): Node from that insertion should be started.

        @return (bool) True if number of nodes at the below level is changed, otherwise False.

        """

        # None-leaf node
        if search_node.type == cfnode_type.CFNODE_NONLEAF:
            self.log.appendPlainText("insert for non-leaf")
            return self.__insert_for_noneleaf_node(entry, search_node)

        # Leaf is reached
        else:
            self.log.appendPlainText("insert for leaf")
            return self.__insert_for_leaf_node(entry, search_node)

    def __insert_for_leaf_node(self, entry, search_node):
        """!
        @brief Recursive insert entry from leaf node to the tree.

        @param[in] entry (cfentry): Clustering feature.
        @param[in] search_node (cfnode): None-leaf node from that insertion should be started.

        @return (bool) True if number of nodes at the below level is changed, otherwise False.

        """

        node_amount_updation = False

        # Try to absorb by the entity
        index_nearest_entry = search_node.get_nearest_index_entry(entry, self.__type_measurement)
        self.log.appendPlainText("index_nearest_entry: {}".format(index_nearest_entry))
        # self.log.appendPlainText("nearest entry: {}".format(search_node.entries[index_nearest_entry]))
        merged_entry = search_node.entries[index_nearest_entry] + entry

        self.log.appendPlainText("diameter: {}".format(merged_entry.get_diameter()))
        # Otherwise try to add new entry
        if merged_entry.get_diameter() > self.__threshold:
            self.log.appendPlainText("diameter greater than threshold")
            # If it's not exceeded append entity and update feature of the leaf node.
            search_node.insert_entry(entry)

            # Otherwise current node should be splitted

            if len(search_node.entries) > self.__max_entries:
                self.log.appendPlainText("node has to split")
                self.__split_procedure(search_node)
                node_amount_updation = True

            # Update statistics
            self.__amount_entries += 1

        else:
            self.log.appendPlainText("diameter ok")
            search_node.entries[index_nearest_entry] = merged_entry
            search_node.feature += entry

        return node_amount_updation

    def __insert_for_noneleaf_node(self, entry, search_node):
        """!
        @brief Recursive insert entry from non-leaf node to the tree.

        @param[in] entry (cfentry): Clustering feature.
        @param[in] search_node (cfnode): Non-leaf node from that insertion should be started.

        @return (bool) True if number of nodes at the below level is changed, otherwise False.

        """

        node_amount_updation = False

        min_key = lambda child_node: child_node.get_distance(search_node, self.__type_measurement)
        nearest_child_node = min(search_node.successors, key=min_key)
        # self.log.appendPlainText("nearestchildnode: {}".format(nearest_child_node))
        # self.log.appendPlainText("recursive insert entry from non-leaf node to the tree")
        child_node_updation = self.__recursive_insert(entry, nearest_child_node)

        # Update clustering feature of none-leaf node.
        search_node.feature += entry

        # Check branch factor, probably some leaf has been splitted and threshold has been exceeded.
        if len(search_node.successors) > self.__branch_factor:
            self.log.appendPlainText("over branch_factor ")

            # Check if it's aleady root then new root should be created (height is increased in this case).
            if search_node is self.__root:
                self.log.appendPlainText("height increases")
                self.__root = non_leaf_node(search_node.feature, None, [search_node], None)
                search_node.parent = self.__root

                # Update statistics
                self.__amount_nodes += 1
                self.__height += 1

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
            [nearest_child_node1, nearest_child_node2] = node.get_nearest_successors(self.__type_measurement)

            if len(nearest_child_node1.successors) + len(nearest_child_node2.successors) <= self.__branch_factor:
                node.successors.remove(nearest_child_node2)
                if nearest_child_node2.type == cfnode_type.CFNODE_LEAF:
                    self.__leafes.remove(nearest_child_node2)

                nearest_child_node1.merge(nearest_child_node2)

                merging_result = True

        if merging_result is True:
            self.log.appendPlainText("merging successful")
        else:
            self.log.appendPlainText("merging not successful")
        return merging_result

    def __split_procedure(self, split_node):
        """!
        @brief Starts node splitting procedure in the CF-tree from the specify node.

        @param[in] split_node (cfnode): CF-tree node that should be splitted.

        """
        if split_node is self.__root:
            self.__root = non_leaf_node(split_node.feature, None, [split_node], None)
            split_node.parent = self.__root

            # Update statistics
            self.__amount_nodes += 1
            self.__height += 1

        [new_node1, new_node2] = self.__split_leaf_node(split_node)

        self.__leafes.remove(split_node)
        self.__leafes.append(new_node1)
        self.__leafes.append(new_node2)

        # Update parent list of successors
        parent = split_node.parent
        parent.successors.remove(split_node)
        parent.successors.append(new_node1)
        parent.successors.append(new_node2)

        # Update statistics
        self.__amount_nodes += 1

    def __split_nonleaf_node(self, node):
        """!
        @brief Performs splitting of the specified non-leaf node.

        @param[in] node (non_leaf_node): Non-leaf node that should be splitted.

        @return (list) New pair of non-leaf nodes [non_leaf_node1, non_leaf_node2].

        """
        self.log.appendPlainText("split non-leaf node")
        [farthest_node1, farthest_node2] = node.get_farthest_successors(self.__type_measurement)

        # create new non-leaf nodes
        new_node1 = non_leaf_node(farthest_node1.feature, node.parent, [farthest_node1], None)
        new_node2 = non_leaf_node(farthest_node2.feature, node.parent, [farthest_node2], None)

        farthest_node1.parent = new_node1
        farthest_node2.parent = new_node2

        # re-insert other successors
        for successor in node.successors:
            if (successor is not farthest_node1) and (successor is not farthest_node2):
                distance1 = new_node1.get_distance(successor, self.__type_measurement)
                distance2 = new_node2.get_distance(successor, self.__type_measurement)

                if distance1 < distance2:
                    new_node1.insert_successor(successor)
                else:
                    new_node2.insert_successor(successor)

        return [new_node1, new_node2]

    def __split_leaf_node(self, node):
        """!
        @brief Performs splitting of the specified leaf node.

        @param[in] node (leaf_node): Leaf node that should be splitted.

        @return (list) New pair of leaf nodes [leaf_node1, leaf_node2].

        @warning Splitted node is transformed to non_leaf.

        """
        self.log.appendPlainText("split leaf node")
        # search farthest pair of entries
        [farthest_entity1, farthest_entity2] = node.get_farthest_entries(self.__type_measurement)
        # self.log.appendPlainText("farthest1: {}".format(farthest_entity1))
        # self.log.appendPlainText("farthest2: {}".format(farthest_entity2))

        # create new nodes
        new_node1 = leaf_node(farthest_entity1, node.parent, [farthest_entity1], None)
        new_node2 = leaf_node(farthest_entity2, node.parent, [farthest_entity2], None)

        # re-insert other entries
        for entity in node.entries:
            if (entity is not farthest_entity1) and (entity is not farthest_entity2):
                distance1 = new_node1.feature.get_distance(entity, self.__type_measurement)
                distance2 = new_node2.feature.get_distance(entity, self.__type_measurement)

                if distance1 < distance2:
                    new_node1.insert_entry(entity)
                else:
                    new_node2.insert_entry(entity)

        # self.log.appendPlainText("new_node1: {}".format(new_node1))
        # self.log.appendPlainText("new_node2: {}".format(new_node2))

        return [new_node1, new_node2]

    def show_feature_distribution(self, data=None):
        """!
         @brief Shows feature distribution.
         @details Only features in 1D, 2D, 3D space can be visualized.

         @param[in] data (list): List of points that will be used for visualization, if it not specified than feature will be displayed only.

         """
        visualizer = cluster_visualizer()

        self.log.appendPlainText("amount of nodes: {}".format(self.__amount_nodes))

        if data is not None:
            visualizer.append_cluster(data, marker='x')

        for level in range(0, self.height):
            level_nodes = self.get_level_nodes(level)

            centers = [node.feature.get_centroid() for node in level_nodes]
            visualizer.append_cluster(centers, None, markersize=(self.height - level + 1) * 5)

        visualizer.show()
