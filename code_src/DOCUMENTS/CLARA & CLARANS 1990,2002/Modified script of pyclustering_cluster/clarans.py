"""!

@brief Cluster analysis algorithm: CLARANS.
@details Implementation based on paper @cite article::clarans::1.

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


import random

from pyclustering.cluster.encoder import type_encoding

from pyclustering.utils import euclidean_distance_square


def plot_pam(data, cl, equal_axis_scale = False):
    import matplotlib.pyplot as plt
    import numpy as np

    fig,ax = plt.subplots(figsize=(14,6))
    plt.scatter(np.array(data)[:,0], np.array(data)[:,1], s=300, color="white", edgecolor="black")
    colors = { 0:"seagreen", 1:'beige', 2:'yellow', 3:'grey',
                   4:'pink', 5:'turquoise', 6:'orange', 7:'purple', 8:'yellowgreen', 9:'olive', 10:'brown',
                   11:'tan', 12: 'plum', 13:'rosybrown', 14:'lightblue', 15:"khaki", 16:"gainsboro", 17:"peachpuff"}

    for i,el in enumerate(list(cl.values())):
        plt.scatter(np.array(data)[el,0], np.array(data)[el,1], s=300, color=colors[i%17], edgecolor="black")

    for i,el in enumerate(list(cl.keys())):
        plt.scatter(np.array(data)[el,0], np.array(data)[el,1], s=500, color="red", marker="X", edgecolor="black")

    xmin, xmax, ymin, ymax = plt.axis()
    xwidth = xmax - xmin
    ywidth = ymax - ymin

    xw1 = xwidth*0.01
    yw1 = ywidth*0.01

    xw2 = xwidth*0.005
    yw2 = ywidth*0.01

    xw3 = xwidth*0.01
    yw3 = ywidth*0.01

    for i, txt in enumerate(range(len(data))):
        if len(str(txt))==2:
            ax.annotate(txt, (np.array(data)[:,0][i]-xw1, np.array(data)[:,1][i]-yw1), fontsize=12, size=12)
        elif len(str(txt))==1:
            ax.annotate(txt, (np.array(data)[:,0][i]-xw2, np.array(data)[:,1][i]-yw2), fontsize=12, size=12)
        else:
            ax.annotate(txt, (np.array(data)[:,0][i]-xw3, np.array(data)[:,1][i]-yw3), fontsize=9, size=9)

    if equal_axis_scale == True:
        plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


class clarans:
    """!
    @brief Class represents clustering algorithm CLARANS (a method for clustering objects for spatial data mining).

    """

    def __init__(self, data, number_clusters, numlocal, maxneighbor):
        """!
        @brief Constructor of clustering algorithm CLARANS.
        @details The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each search of a local minima.

        @param[in] data (list): Input data that is presented as list of points (objects), each point should be represented by list or tuple.
        @param[in] number_clusters (uint): Amount of clusters that should be allocated.
        @param[in] numlocal (uint): The number of local minima obtained (amount of iterations for solving the problem).
        @param[in] maxneighbor (uint): The maximum number of neighbors examined.

        """

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

        # loop for a numlocal number of times
        for _ in range(0, self.__numlocal):

            print("numlocal: ", _)
            # set (current) random medoids
            self.__current = random.sample(range(0, len(self.__pointer_data)), self.__number_clusters)

            # update clusters in line with random allocated medoids
            self.__update_clusters(self.__current)

            # optimize configuration
            self.__optimize_configuration()

            # obtain cost of current cluster configuration and compare it with the best obtained
            estimation = self.__calculate_estimation()
            if estimation < self.__optimal_estimation:
                print("Better configuration found with medoids: {0} and cost: {1}".format(self.__current[:], estimation))
                self.__optimal_medoids = self.__current[:]
                self.__optimal_estimation = estimation

                if plotting == True:
                    self.__update_clusters(self.__optimal_medoids)
                    plot_pam(self.__pointer_data,
                            dict(zip(self.__optimal_medoids,self.__clusters)))

            else:
                print("Configuration found does not improve current best one because its cost is {0}".format(estimation))
                if plotting == True:
                    self.__update_clusters(self.__optimal_medoids)
                    plot_pam(self.__pointer_data,
                            dict(zip(self.__optimal_medoids,self.__clusters)))

        self.__update_clusters(self.__optimal_medoids)
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
        while (index_neighbor < self.__maxneighbor):
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
                    distance_current = euclidean_distance_square(self.__pointer_data[point_index], self.__pointer_data[current_medoid_index])

                    # from the point to candidate median
                    distance_candidate = euclidean_distance_square(self.__pointer_data[point_index], self.__pointer_data[candidate_medoid_index])

                    # from the point to nearest (own) medoid
                    distance_nearest = float('inf')
                    if ( (point_medoid_index != candidate_medoid_index) and (point_medoid_index != current_medoid_cluster_index) ):
                        distance_nearest = euclidean_distance_square(self.__pointer_data[point_index], self.__pointer_data[point_medoid_index])

                    # apply rules for cost calculation
                    if (point_cluster_index == current_medoid_cluster_index):
                        # case 1:
                        if (distance_candidate >= distance_nearest):
                            candidate_cost += distance_nearest - distance_current

                        # case 2:
                        else:
                            candidate_cost += distance_candidate - distance_current

                    elif (point_cluster_index == other_medoid_cluster_index):
                        # case 3 ('nearest medoid' is the representative object of that cluster and object is more similar to 'nearest' than to 'candidate'):
                        if (distance_candidate > distance_nearest):
                            pass;

                        # case 4:
                        else:
                            candidate_cost += distance_candidate - distance_nearest

            if (candidate_cost < 0):
                counter+=1
                # set candidate that has won
                self.__current[current_medoid_cluster_index] = candidate_medoid_index

                # recalculate clusters
                self.__update_clusters(self.__current)

                # reset iterations and starts investigation from the begining
                index_neighbor = 0

            else:

                index_neighbor += 1

        print("Medoid set changed {0} times".format(counter))


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
            if (index_medoid != current_medoid_index):
                other_distance_candidate = euclidean_distance_square(self.__pointer_data[point_index], self.__pointer_data[current_medoid_index])

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
                estimation += euclidean_distance_square(self.__pointer_data[index_point], self.__pointer_data[index_medoid])

        return estimation
