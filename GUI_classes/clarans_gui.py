from PyQt5.QtCore import Qt, QCoreApplication
import random

from pyclustering.cluster.encoder import type_encoding
from pyclustering.utils import euclidean_distance_square

from GUI_classes.utils_gui import choose_dataset, pause_execution
from GUI_classes.pam_gui import KMedoids_gui


from GUI_classes.generic_gui import StartingGui, GraphWindow


class CLARANS_class(StartingGui):
    def __init__(self):
        super(CLARANS_class, self).__init__(name="CLARANS", twinx=False, first_plot=True, second_plot=False,
                                            function=self.start_CLARANS, extract=False, stretch_plot=False)

        self.example_index = 0
        self.button_examples_graph.clicked.connect(lambda: self.openGraphWindow(self.example_index))

    def start_CLARANS(self):
        self.ax1.cla()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_medoids = int(self.line_edit_n_medoids.text())
        self.numlocal_clarans = int(self.line_edit_numlocal_clarans.text())
        self.maxneighbors_clarans = int(self.line_edit_maxneighbors_clarans.text())
        self.n_points = int(self.line_edit_np.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        self.button_run.setEnabled(False)
        self.checkbox_saveimg.setEnabled(False)
        self.button_delete_pics.setEnabled(False)
        self.button_examples_graph.setEnabled(False)
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

        clarans_gui(data=self.X, number_clusters=self.n_medoids, numlocal=self.numlocal_clarans,
                    maxneighbor=self.maxneighbors_clarans, log=self.log, ax=self.ax1,
                    canvas=self.canvas_up, save_fig=self.save_plots, ind_run=self.ind_run,
                    delay=self.delay).process(plotting=True)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.button_examples_graph.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)

    def openGraphWindow(self, ind):
        self.w = GraphWindow(example_index=ind)

        if ind != 9:
            self.example_index += 1
        else:
            self.example_index = 0
        self.w.show()


class clarans_gui:
    """!
    @brief Class represents clustering algorithm CLARANS (a method for clustering objects for spatial data mining).

    """

    def __init__(self, data, number_clusters, numlocal, maxneighbor, log, ax, canvas, save_fig, ind_run, delay):
        """!
        @brief Constructor of clustering algorithm CLARANS.
        @details The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each
        search of a local minima.

        @param[in] data (list): Input data that is presented as list of points (objects), each point should be
        represented by list or tuple.
        @param[in] number_clusters (uint): Amount of clusters that should be allocated.
        @param[in] numlocal (uint): The number of local minima obtained (amount of iterations for solving the problem).
        @param[in] maxneighbor (uint): The maximum number of neighbors examined.

        """
        self.log = log
        self.ax = ax
        self.canvas = canvas
        self.save_fig = save_fig
        self.ind_run = ind_run
        self.delay = delay

        self.__pointer_data = data
        self.__numlocal = numlocal
        self.__maxneighbor = maxneighbor
        self.__number_clusters = number_clusters

        self.__clusters = []
        self.__current = []
        self.__belong = []

        self.__optimal_medoids = []
        self.__optimal_estimation = float('inf')

        self.__verify_arguments()

        self.PAM = KMedoids_gui(n_cluster=self.__number_clusters, log=self.log, ax=self.ax, canvas=self.canvas,
                                save_fig=self.save_fig, ind_run=self.ind_run, delay=self.delay)

    def __verify_arguments(self):
        """!
        @brief Verify input parameters for the algorithm and throw exception in case of incorrectness.

        """
        if len(self.__pointer_data) == 0:
            raise ValueError("Input data is empty (size: '%d')." % len(self.__pointer_data))

        if self.__number_clusters <= 0:
            raise ValueError("Amount of cluster (current value: '%d') for allocation should be greater than 0." %
                             self.__number_clusters)

        if self.__numlocal < 0:
            raise ValueError("Local minima (current value: '%d') should be greater or equal to 0." % self.__numlocal)

        if self.__maxneighbor < 0:
            raise ValueError("Maximum number of neighbors (current value: '%d') should be greater or "
                             "equal to 0." % self.__maxneighbor)

    def process(self, plotting=False):
        """!
        @brief Performs cluster analysis in line with rules of CLARANS algorithm.

        @return (clarans) Returns itself (CLARANS instance).

        @see get_clusters()
        @see get_medoids()

        """

        random.seed()
        index_for_saving_plots = 0

        # loop for a numlocal number of times
        for _ in range(0, self.__numlocal):

            self.log.appendPlainText("")
            self.log.appendPlainText("numlocal (iteration): {}".format(_ + 1))
            # set (current) random medoids
            self.__current = random.sample(range(0, len(self.__pointer_data)), self.__number_clusters)

            # update clusters in line with random allocated medoids
            self.__update_clusters(self.__current)

            # optimize configuration
            self.__optimize_configuration()

            # obtain cost of current cluster configuration and compare it with the best obtained
            estimation = self.__calculate_estimation()
            if estimation < self.__optimal_estimation:
                self.log.appendPlainText("Better configuration found with "
                                         "medoids: {0} and cost: {1}".format(self.__current[:], estimation))
                self.__optimal_medoids = self.__current[:]
                self.__optimal_estimation = estimation

                if plotting is True:
                    self.__update_clusters(self.__optimal_medoids)
                    if self.delay != 0:
                        pause_execution(self.delay)
                    self.PAM.plot_pam_gui(data=self.__pointer_data, name="CLARANS",
                                          cl=dict(zip(self.__optimal_medoids, self.__clusters)),
                                          ax=self.ax, canvas=self.canvas, ind_run=self.ind_run,
                                          ind_fig=index_for_saving_plots, save_plots=self.save_fig
                                          )

            else:
                self.log.appendPlainText("Configuration found does not improve current "
                                         "best one because its cost is {0}".format(estimation))
                if plotting is True:
                    self.__update_clusters(self.__current[:])
                    if self.delay != 0:
                        pause_execution(self.delay)
                    self.PAM.plot_pam_gui(data=self.__pointer_data, cl=dict(zip(self.__current[:], self.__clusters)),
                                          ax=self.ax, canvas=self.canvas, ind_run=self.ind_run, name="CLARANS",
                                          ind_fig=index_for_saving_plots, save_plots=self.save_fig
                                          )

            index_for_saving_plots += 1
        self.__update_clusters(self.__optimal_medoids)

        if plotting is True:
            self.log.appendPlainText("")
            self.log.appendPlainText("FINAL RESULT")
            if self.delay != 0:
                pause_execution(self.delay)
            self.PAM.plot_pam_gui(data=self.__pointer_data, cl=dict(zip(self.__optimal_medoids, self.__clusters)),
                                  ax=self.ax, canvas=self.canvas, ind_run=self.ind_run, name="CLARANS",
                                  ind_fig=None, save_plots=self.save_fig
                                  )

        return self

    def get_clusters(self):
        """!
        @brief Returns allocated clusters by the algorithm.

        @remark Allocated clusters can be returned only after data processing (use method process()), otherwise empty list is returned.

        @return (list) List of allocated clusters, each cluster contains indexes of objects in list of data.

        @see process()
        @see get_medoids()

        """

        return self.__clusters

    def get_medoids(self):
        """!
        @brief Returns list of medoids of allocated clusters.

        @see process()
        @see get_clusters()

        """

        return self.__optimal_medoids

    def get_cluster_encoding(self):
        """!
        @brief Returns clustering result representation type that indicate how clusters are encoded.

        @return (type_encoding) Clustering result representation.

        @see get_clusters()

        """

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION

    def __update_clusters(self, medoids):
        """!
        @brief Forms cluster in line with specified medoids by calculation distance from each point to medoids.

        """

        self.__belong = [0] * len(self.__pointer_data)
        self.__clusters = [[] for i in range(len(medoids))]
        for index_point in range(len(self.__pointer_data)):
            index_optim = -1
            dist_optim = 0.0

            for index in range(len(medoids)):
                dist = euclidean_distance_square(self.__pointer_data[index_point], self.__pointer_data[medoids[index]])

                if (dist < dist_optim) or (index is 0):
                    index_optim = index
                    dist_optim = dist

            self.__clusters[index_optim].append(index_point)
            self.__belong[index_point] = index_optim

        # If cluster is not able to capture object it should be removed
        self.__clusters = [cluster for cluster in self.__clusters if len(cluster) > 0]

    def __optimize_configuration(self):
        """!
        @brief Finds quasi-optimal medoids and updates in line with them clusters in line with algorithm's rules.

        """
        index_neighbor = 0
        counter = 0
        while index_neighbor < self.__maxneighbor:
            # get random current medoid that is to be replaced
            current_medoid_index = self.__current[random.randint(0, self.__number_clusters - 1)]
            current_medoid_cluster_index = self.__belong[current_medoid_index]

            # get new candidate to be medoid
            candidate_medoid_index = random.randint(0, len(self.__pointer_data) - 1)

            while candidate_medoid_index in self.__current:
                candidate_medoid_index = random.randint(0, len(self.__pointer_data) - 1)

            candidate_cost = 0.0
            for point_index in range(0, len(self.__pointer_data)):
                if point_index not in self.__current:
                    # get non-medoid point and its medoid
                    point_cluster_index = self.__belong[point_index]
                    point_medoid_index = self.__current[point_cluster_index]

                    # get other medoid that is nearest to the point (except current and candidate)
                    other_medoid_index = self.__find_another_nearest_medoid(point_index, current_medoid_index)
                    other_medoid_cluster_index = self.__belong[other_medoid_index]

                    # for optimization calculate all required distances
                    # from the point to current medoid
                    distance_current = euclidean_distance_square(self.__pointer_data[point_index],
                                                                 self.__pointer_data[current_medoid_index])

                    # from the point to candidate median
                    distance_candidate = euclidean_distance_square(self.__pointer_data[point_index],
                                                                   self.__pointer_data[candidate_medoid_index])

                    # from the point to nearest (own) medoid
                    distance_nearest = float('inf')
                    if ((point_medoid_index != candidate_medoid_index) and (
                            point_medoid_index != current_medoid_cluster_index)):
                        distance_nearest = euclidean_distance_square(self.__pointer_data[point_index],
                                                                     self.__pointer_data[point_medoid_index])

                    # apply rules for cost calculation
                    if point_cluster_index == current_medoid_cluster_index:
                        # case 1:
                        if distance_candidate >= distance_nearest:
                            candidate_cost += distance_nearest - distance_current

                        # case 2:
                        else:
                            candidate_cost += distance_candidate - distance_current

                    elif point_cluster_index == other_medoid_cluster_index:
                        # case 3 ('nearest medoid' is the representative object of that cluster and object is
                        # more similar to 'nearest' than to 'candidate'):
                        if distance_candidate > distance_nearest:
                            pass;

                        # case 4:
                        else:
                            candidate_cost += distance_candidate - distance_nearest

            if candidate_cost < 0:
                counter += 1
                # set candidate that has won
                self.__current[current_medoid_cluster_index] = candidate_medoid_index

                # recalculate clusters
                self.__update_clusters(self.__current)

                # reset iterations and starts investigation from the begining
                index_neighbor = 0

            else:

                index_neighbor += 1

        self.log.appendPlainText("Medoid set changed {0} times".format(counter))

    def __find_another_nearest_medoid(self, point_index, current_medoid_index):
        """!
        @brief Finds the another nearest medoid for the specified point that is different from the specified medoid.

        @param[in] point_index: index of point in dataspace for that searching of medoid in current list of medoids is perfomed.
        @param[in] current_medoid_index: index of medoid that shouldn't be considered as a nearest.

        @return (uint) index of the another nearest medoid for the point.

        """
        other_medoid_index = -1
        other_distance_nearest = float('inf')
        for index_medoid in self.__current:
            if index_medoid != current_medoid_index:
                other_distance_candidate = euclidean_distance_square(self.__pointer_data[point_index],
                                                                     self.__pointer_data[current_medoid_index])

                if other_distance_candidate < other_distance_nearest:
                    other_distance_nearest = other_distance_candidate
                    other_medoid_index = index_medoid

        return other_medoid_index

    def __calculate_estimation(self):
        """!
        @brief Calculates estimation (cost) of the current clusters. The lower the estimation,
               the more optimally configuration of clusters.

        @return (double) estimation of current clusters.

        """
        estimation = 0.0
        for index_cluster in range(0, len(self.__clusters)):
            cluster = self.__clusters[index_cluster]
            index_medoid = self.__current[index_cluster]
            for index_point in cluster:
                estimation += euclidean_distance_square(self.__pointer_data[index_point],
                                                        self.__pointer_data[index_medoid])

        return estimation



