import random
from typing import Tuple, Dict, Any

import scipy
import itertools
import graphviz
import numpy as np
import pandas as pd
from clustviz.pam import plot_pam

from pyclustering.utils import euclidean_distance_square
from pyclustering.cluster.clarans import clarans as clarans_pyclustering


class clarans(clarans_pyclustering):

    def process(self, plotting: bool = False):
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
            self.__current = random.sample(
                range(0, len(self.__pointer_data)), self.__number_clusters
            )

            # update clusters in line with random allocated medoids
            self.__update_clusters(self.__current)

            # optimize configuration
            self.__optimize_configuration()

            # obtain cost of current cluster configuration and compare it with the best obtained
            estimation = self.__calculate_estimation()
            if estimation < self.__optimal_estimation:
                print(
                    "Better configuration found with medoids: {0} and cost: {1}".format(
                        self.__current[:], estimation
                    )
                )
                self.__optimal_medoids = self.__current[:]
                self.__optimal_estimation = estimation

                if plotting is True:
                    self.__update_clusters(self.__optimal_medoids)
                    plot_pam(
                        self.__pointer_data,
                        dict(zip(self.__optimal_medoids, self.__clusters)),
                    )

            else:
                print(
                    "Configuration found does not improve current best one because its cost is {0}".format(
                        estimation
                    )
                )
                if plotting is True:
                    self.__update_clusters(self.__current[:])
                    plot_pam(
                        self.__pointer_data,
                        dict(zip(self.__current[:], self.__clusters)),
                    )

        self.__update_clusters(self.__optimal_medoids)

        if plotting is True:
            print("FINAL RESULT:")
            plot_pam(
                self.__pointer_data,
                dict(zip(self.__optimal_medoids, self.__clusters)),
            )

        return self

    def __optimize_configuration(self):
        """!
        @brief Finds quasi-optimal medoids and updates in line with them clusters in line with algorithm's rules.

        """
        index_neighbor = 0
        counter = 0
        while index_neighbor < self.__maxneighbor:
            # get random current medoid that is to be replaced
            current_medoid_index = self.__current[
                random.randint(0, self.__number_clusters - 1)
            ]
            current_medoid_cluster_index = self.__belong[current_medoid_index]

            # get new candidate to be medoid
            candidate_medoid_index = random.randint(
                0, len(self.__pointer_data) - 1
            )

            while candidate_medoid_index in self.__current:
                candidate_medoid_index = random.randint(
                    0, len(self.__pointer_data) - 1
                )

            candidate_cost = 0.0
            for point_index in range(0, len(self.__pointer_data)):
                if point_index not in self.__current:
                    # get non-medoid point and its medoid
                    point_cluster_index = self.__belong[point_index]
                    point_medoid_index = self.__current[point_cluster_index]

                    # get other medoid that is nearest to the point (except current and candidate)
                    other_medoid_index = self.__find_another_nearest_medoid(
                        point_index, current_medoid_index
                    )
                    other_medoid_cluster_index = self.__belong[
                        other_medoid_index
                    ]

                    # for optimization calculate all required distances
                    # from the point to current medoid
                    distance_current = euclidean_distance_square(
                        self.__pointer_data[point_index],
                        self.__pointer_data[current_medoid_index],
                    )

                    # from the point to candidate median
                    distance_candidate = euclidean_distance_square(
                        self.__pointer_data[point_index],
                        self.__pointer_data[candidate_medoid_index],
                    )

                    # from the point to nearest (own) medoid
                    distance_nearest = float("inf")
                    if (point_medoid_index != candidate_medoid_index) and (
                            point_medoid_index != current_medoid_cluster_index
                    ):
                        distance_nearest = euclidean_distance_square(
                            self.__pointer_data[point_index],
                            self.__pointer_data[point_medoid_index],
                        )

                    # apply rules for cost calculation
                    if point_cluster_index == current_medoid_cluster_index:
                        # case 1:
                        if distance_candidate >= distance_nearest:
                            candidate_cost += (
                                    distance_nearest - distance_current
                            )

                        # case 2:
                        else:
                            candidate_cost += (
                                    distance_candidate - distance_current
                            )

                    elif point_cluster_index == other_medoid_cluster_index:
                        # case 3 ('nearest medoid' is the representative object of that cluster and object is more
                        # similar to 'nearest' than to 'candidate'):
                        if distance_candidate > distance_nearest:
                            pass

                        # case 4:
                        else:
                            candidate_cost += (
                                    distance_candidate - distance_nearest
                            )

            if candidate_cost < 0:
                counter += 1
                # set candidate that has won
                self.__current[
                    current_medoid_cluster_index
                ] = candidate_medoid_index

                # recalculate clusters
                self.__update_clusters(self.__current)

                # reset iterations and starts investigation from the begining
                index_neighbor = 0

            else:

                index_neighbor += 1

        print("Medoid set changed {0} times".format(counter))


def compute_cost_clarans(data: pd.DataFrame, _cur_choice: list) -> Tuple[float, Dict[Any, list]]:
    """
    A function to compute the configuration cost. (modified from that of CLARA)

    :param data: The input dataframe.
    :param _cur_choice: The current set of medoid choices.
    :return: The total configuration cost, the medoids.
    """
    total_cost = 0.0
    medoids = {}
    for idx in _cur_choice:
        medoids[idx] = []

    for i in list(data.index):
        choice = -1
        min_cost = np.inf
        for m in medoids:
            # fast_euclidean from CLARA
            tmp = np.linalg.norm(data.loc[m] - data.loc[i])
            if tmp < min_cost:
                choice = m
                min_cost = tmp

        medoids[choice].append(i)
        total_cost += min_cost
    # print("total_cost: ", total_cost)
    return total_cost, medoids


def plot_tree_clarans(data: pd.DataFrame, k: int) -> None:
    """
    plot G_{k,n} as in the paper of CLARANS; only to use with small input data.

    :param data: input DataFrame.
    :param k: number of points in each combination (possible set of medoids).
    """

    n = len(data)
    num_points = int(scipy.special.binom(n, k))
    num_neigh = k * (n - k)

    if (num_points > 50) or (num_neigh > 10):
        print(
            "Either graph nodes are more than 50 or neighbors are more than 10, the graph would be too big"
        )
        return

    # all possibile combinations of k elements from input data
    name_nodes = list(itertools.combinations(list(data.index), k))

    dot = graphviz.Digraph(comment="Clustering")

    # draw nodes, also adding the configuration cost
    for i in range(num_points):
        tot_cost, meds = compute_cost_clarans(data, list(name_nodes[i]))
        tc = round(tot_cost, 3)

        dot.node(str(name_nodes[i]), str(name_nodes[i]) + ": " + str(tc))

    # only connect nodes if they have k-1 common elements
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                if (
                        len(set(list(name_nodes[i])) & set(list(name_nodes[j])))
                        == k - 1
                ):
                    dot.edge(str(name_nodes[i]), str(name_nodes[j]))

    graph = graphviz.Source(dot)  # .view()
    display(graph)
