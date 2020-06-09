# SachinKalsi GITHUB
# https://github.com/SachinKalsi/kmedoids/blob/master/KMedoids.py
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import random
import pprint


class KMedoids:
    def __init__(
        self,
        n_cluster=2,
        max_iter=10,
        tol=0.1,
        start_prob=0.8,
        end_prob=0.99,
        random_state=42,
    ):
        """ Kmedoids constructor called """
        if (
            start_prob < 0
            or start_prob >= 1
            or end_prob < 0
            or end_prob >= 1
            or start_prob > end_prob
        ):
            raise ValueError("Invalid input")
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tol = tol
        self.start_prob = start_prob
        self.end_prob = end_prob

        self.medoids = []  # empty medois
        self.clusters = {}  # empty clusters
        self.tol_reached = float("inf")
        self.current_distance = 0

        self.__data = None
        self.__is_csr = None
        self.__rows = 0
        self.__columns = 0
        self.cluster_distances = {}

        self.__random_state = random_state

    def fit(self, data):
        print("fitting")
        self.__data = data
        self.__set_data_type()
        self.__start_algo()
        return self

    def __start_algo(self):
        print("starting algo")
        self.__initialize_medoids()  # choosing initial medoids
        # computing clusters and cluster_distances
        self.clusters, self.cluster_distances = self.__calculate_clusters(
            self.medoids
        )
        # print cluster and cluster_distances
        print("clusters: ")
        pprint.PrettyPrinter(indent=2, compact=True).pprint(self.clusters)
        print("clusters_distances: ", self.cluster_distances)
        plot_pam(self.__data, self.clusters)

        self.__update_clusters()

    def __update_clusters(self):
        for i in range(
            self.max_iter
        ):  # to stop if convergence isn't reached whithin max_iter iterations
            print("\n")
            print("iteration n°: ", i)
            # compute distance obtained by swapping medoids in the clusters
            cluster_dist_with_new_medoids = (
                self.__swap_and_recalculate_clusters()
            )
            # if the new sum of cluster_distances is smaller than the old one
            if (
                self.__is_new_cluster_dist_small(cluster_dist_with_new_medoids)
                is True
            ):
                print("new is smaller")
                # compute clusters and cluster_distance with new medoids
                (
                    self.clusters,
                    self.cluster_distances,
                ) = self.__calculate_clusters(self.medoids)
                print("clusters: ")
                pprint.PrettyPrinter(indent=2, compact=True).pprint(
                    self.clusters
                )
                plot_pam(self.__data, self.clusters)
                # print("clusters_distances: ", self.cluster_distances)
            else:
                # if the sum of cluster_distances doesn't improve, terminate the algorithm
                print("termination")
                break

    def __is_new_cluster_dist_small(self, cluster_dist_with_new_medoids):
        """returns True if the new sum of cluster_distances is smaller than the previous one, and updates the
        medoids, else returns False """
        # compute the existing sum of cluster_distances
        existance_dist = self.calculate_distance_of_clusters()
        print("present dist: ", existance_dist)
        # computes the new sum of cluster_distances
        new_dist = self.calculate_distance_of_clusters(
            cluster_dist_with_new_medoids
        )
        print("new dist: ", new_dist)

        # if it is better, substitute the old medoids with the new ones and return True, else return False
        if (
            existance_dist > new_dist
            and (existance_dist - new_dist) > self.tol
        ):
            self.medoids = cluster_dist_with_new_medoids.keys()
            return True

        return False

    def calculate_distance_of_clusters(self, cluster_dist=None):
        """if no argument is provided, just sum the distances of the existing cluster_distances, else sum the distances
        of the input cluster_distances """
        if cluster_dist is None:
            cluster_dist = self.cluster_distances
        dist = 0
        for medoid in cluster_dist.keys():
            dist += cluster_dist[medoid]
        return dist

    def __swap_and_recalculate_clusters(self):
        # http://www.math.le.ac.uk/people/ag153/homepage/KmeansKmedoids/Kmeans_Kmedoids.html
        """returns dictionary of new cluster_distances obtained by swapping medoids in each cluster"""
        print("swap and recompute")
        cluster_dist = {}
        for medoid in self.medoids:  # for each medoid
            is_shortest_medoid_found = False
            for data_index in self.clusters[
                medoid
            ]:  # for each point in the current medoid's cluster
                if data_index != medoid:  # exclude the medoid itself
                    # create a list of the elements of the cluster
                    cluster_list = list(self.clusters[medoid])
                    # make the current point the temporary medoid
                    cluster_list[
                        self.clusters[medoid].index(data_index)
                    ] = medoid
                    # compute new cluster distance obtained by swapping the medoid
                    new_distance = self.calculate_inter_cluster_distance(
                        data_index, cluster_list
                    )
                    # if this new distance is smaller than the previous one
                    if new_distance < self.cluster_distances[medoid]:
                        print("new better medoid: ", data_index)
                        cluster_dist[data_index] = new_distance
                        is_shortest_medoid_found = True
                        break  # exit for loop for this medoid, since a better one has been found
            # if no better medoid has been found, keep the current one
            if is_shortest_medoid_found is False:
                print("no better medoids found, keep: ", medoid)
                cluster_dist[medoid] = self.cluster_distances[medoid]
        print("cluster_dist: ", cluster_dist)
        return cluster_dist

    def calculate_inter_cluster_distance(self, medoid, cluster_list):
        """computes the average distance of points in a cluster from their medoid"""
        distance = 0
        for data_index in cluster_list:
            distance += self.__get_distance(medoid, data_index)
        return distance / len(cluster_list)

    def __calculate_clusters(self, medoids):
        """returns the clusters and the relative distances (average distance of each element of the cluster from the
        medoid) """
        clusters = (
            {}
        )  # it will be of the form {medoid0: [elements of cluster0], medoid1: [elements of cluster1], ...}
        cluster_distances = {}
        # initialize empty clusters and cluster_distances
        for medoid in medoids:
            clusters[medoid] = []
            cluster_distances[medoid] = 0

        for row in range(self.__rows):  # for every row of input data
            # compute nearest medoid and relative distance from row
            (
                nearest_medoid,
                nearest_distance,
            ) = self.__get_shortest_distance_to_medoid(row, medoids)
            # add this distance to the distances relative to the nearest_medoid cluster
            cluster_distances[nearest_medoid] += nearest_distance
            # add the row to the nearest_medoid cluster
            clusters[nearest_medoid].append(row)

        # divide each cluster_distance for the number of element in its corresponding cluster, to obtain the average
        # distance
        for medoid in medoids:
            cluster_distances[medoid] /= len(clusters[medoid])
        return clusters, cluster_distances

    def __get_shortest_distance_to_medoid(self, row_index, medoids):
        """returns closest medoid and relative distance from the input row (point)"""
        min_distance = float("inf")
        current_medoid = None

        for medoid in medoids:
            current_distance = self.__get_distance(
                medoid, row_index
            )  # compute distance from input row to medoid
            if (
                current_distance < min_distance
            ):  # if it happens to be shorter than all previously computed distances
                min_distance = current_distance  # save it as min_distance
                current_medoid = (
                    medoid  # choose this medoid as the closest one
                )
        return current_medoid, min_distance

    def __initialize_medoids(self):
        """Kmeans++ initialisation"""
        print("initializing medoids with kmeans++")
        random.seed(self.__random_state)

        self.medoids.append(
            random.randint(0, self.__rows - 1)
        )  # choosing a random row from data

        while (
            len(self.medoids) != self.n_cluster
        ):  # until the number of medoids reaches the number of clusters
            self.medoids.append(
                self.__find_distant_medoid()
            )  # choose as next medoid the most distant from the previously chosen ones

    def __find_distant_medoid(self):
        """returns a row corresponding to a point which is considerably distant from its closest medoid"""
        distances = []
        indices = []
        for row in range(self.__rows):  # for every row in data
            indices.append(row)
            distances.append(
                self.__get_shortest_distance_to_medoid(row, self.medoids)[1]
            )  # shortest distance from row to its closest medoid
        distances_index = np.argsort(
            distances
        )  # the sorted indices of the distances
        choosen_dist = self.__select_distant_medoid(
            distances_index
        )  # the index corresponding to the distance chosen
        return indices[
            choosen_dist
        ]  # row corresponding to the chosen distance

    def __select_distant_medoid(self, distances_index):
        """returns a random index of the distances_index between start and end"""
        start_index = round(
            self.start_prob * len(distances_index)
        )  # normally 0.8*len(dist)
        end_index = round(
            self.end_prob * (len(distances_index) - 1)
        )  # normally 0.99*len(dist)
        # returns a random index corresponding to a row which is distant from its closest medoid, but not necessarily
        # the row with the maximum distance from its medoid
        return distances_index[random.randint(start_index, end_index)]

    def __get_distance(self, x1, x2):
        """computes euclidean distance, with an initial transformation based on input data"""
        a = (
            self.__data[x1].toarray()
            if self.__is_csr is True
            else np.array(self.__data[x1])
        )
        b = (
            self.__data[x2].toarray()
            if self.__is_csr is True
            else np.array(self.__data[x2])
        )
        return np.linalg.norm(a - b)

    def __set_data_type(self):
        """to check whether the given input is of type list or csr """
        print("setting data type")
        if isinstance(self.__data, csr_matrix):
            self.__is_csr = True
            self.__rows = self.__data.shape[0]
            self.__columns = self.__data.shape[1]
        elif isinstance(self.__data, list):
            self.__is_csr = False
            self.__rows = len(self.__data)
            self.__columns = len(self.__data[0])
        else:
            raise ValueError("Invalid input")


def plot_pam(data, cl, equal_axis_scale=False):
    """
    Scatterplot of data points, with colors according to cluster labels.
    Centers of mass of the clusters are marked with an X.

    :param data: input data sample as dataframe.
    :param cl: cluster dictionary.
    :param equal_axis_scale: if True, axis are plotted with the same scaling.

    """

    fig, ax = plt.subplots(figsize=(14, 6))

    # all points are plotted in white
    plt.scatter(
        np.array(data)[:, 0],
        np.array(data)[:, 1],
        s=300,
        color="white",
        edgecolor="black",
    )

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

    # plot the points with colors according to the cluster they belong to
    for i, el in enumerate(list(cl.values())):
        plt.scatter(
            np.array(data)[el, 0],
            np.array(data)[el, 1],
            s=300,
            color=colors[i % 18],
            edgecolor="black",
        )

    # plot centers of mass, marked with an X
    for i, el in enumerate(list(cl.keys())):
        plt.scatter(
            np.array(data)[el, 0],
            np.array(data)[el, 1],
            s=500,
            color="red",
            marker="X",
            edgecolor="black",
        )

    # add indexes to points in plot
    for i, txt in enumerate([i for i in range(len(data))]):
        ax.annotate(
            txt,
            (np.array(data)[:, 0][i], np.array(data)[:, 1][i]),
            fontsize=10,
            size=10,
            ha="center",
            va="center",
        )

    if equal_axis_scale is True:
        ax.set_aspect("equal", adjustable="box")

    plt.show()
