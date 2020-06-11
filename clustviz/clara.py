# akalino GITHUB
# https://github.com/akalino/Clustering/blob/master/clara.py

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ClaraClustering(object):
    """The clara clustering algorithm.
    Basically an iterative guessing version of k-medoids that makes things a lot faster
    for bigger data sets.
    """

    def __init__(self, max_iter=100000):
        """Class initialization.
        :param max_iter: The default number of max iterations
        """
        self.max_iter = max_iter
        self.dist_cache = dict()

    def clara(self, _df, _k, _fn):
        """The main clara clustering iterative algorithm.
        :param _df: Input dataframe.
        :param _k: Number of medoids.
        :param _fn: The distance function to use.
        :return: The minimized cost, the best medoid choices and the final configuration.
        """
        size = len(_df)
        if size > 100000:
            niter = 1000
            runs = 1
        else:
            niter = self.max_iter
            runs = 5

        # initialize min_avg_cost to infinity
        min_avg_cost = np.inf
        best_choices = []
        best_results = {}

        for j in range(runs):  # usually 5 times
            print("\n")
            print("run number: ", j)
            # take 40+_k*2 random indexes from input data
            sampling_idx = random.sample([i for i in range(size)], 40 + _k * 2)
            # take the corresponding rows from input dataframe _df
            prov_dic = {i: sampling_idx[i] for i in range(40 + _k * 2)}
            print(prov_dic)
            sampling_data = []
            for idx in sampling_idx:
                sampling_data.append(_df.iloc[idx])

            # create the sample dataframe
            sampled_df = pd.DataFrame(sampling_data, index=sampling_idx)

            # return total cost, medoids and clusters of sampled_df
            pre_cost, pre_choice, pre_medoids = self.k_medoids(
                sampled_df, _k, _fn, niter
            )
            plot_pam_mod(sampled_df, pre_medoids, _df)
            print("RESULTS OF K-MEDOIDS")
            print("pre_cost: ", pre_cost)
            print("pre_choice: ", pre_choice)
            print("pre_medoids: ", pre_medoids)

            # compute average cost and clusters of whole input dataframe
            tmp_avg_cost, tmp_medoids = self.average_cost(_df, _fn, pre_choice)

            print("RESULTS OF WHOLE DATASET EVALUATION")
            print("tmp_avg_cost: ", tmp_avg_cost)
            print("tmp_medoids: ", tmp_medoids)
            # if the new cost is lower
            if tmp_avg_cost < min_avg_cost:
                print(
                    "new_cost is lower, from {0} to {1}".format(
                        round(min_avg_cost, 4), round(tmp_avg_cost, 4)
                    )
                )
                min_avg_cost = tmp_avg_cost
                best_choices = list(pre_choice)
                best_results = dict(tmp_medoids)

            elif tmp_avg_cost == min_avg_cost:
                print("new_cost is equal")
            else:
                print("new_cost is higher")

        print("\n")
        print("FINAL RESULT:")
        plot_pam_mod(_df, best_results, _df)

        return min_avg_cost, best_choices, best_results

    def k_medoids(self, _df, _k, _fn, _niter):
        """The original k-medoids algorithm.
        :param _df: Input data frame.
        :param _k: Number of medoids.
        :param _fn: The distance function to use.
        :param _niter: The number of iterations.
        :return: Cost of configuration, the medoids (list) and the clusters (dictionary).

        Pseudo-code for the k-medoids algorithm.
        1. Sample k of the n data points as the medoids.
        2. Associate each data point to the closest medoid.
        3. While the cost of the data point space configuration is decreasing.
            1. For each medoid m and each non-medoid point o:
                1. Swap m and o, recompute cost.
                2. If global cost increased, swap back.
        """

        # Do some smarter setting of initial cost configuration
        pc1, medoids_sample = self.cheat_at_sampling(_df, _k, _fn, 17)
        print("initial medoids sample: ", medoids_sample)
        prior_cost, medoids = self.compute_cost(_df, _fn, medoids_sample)

        current_cost = prior_cost
        print("current_cost: ", current_cost)
        iter_count = 0
        best_choices = []
        best_results = {}

        # print('Running with {m} iterations'.format(m=_niter))
        while iter_count < _niter:
            for m in medoids:
                clust_iter = 0
                for item in medoids[m]:
                    if item != m:
                        idx = medoids_sample.index(m)
                        swap_temp = medoids_sample[idx]
                        medoids_sample[idx] = item
                        tmp_cost, tmp_medoids = self.compute_cost(
                            _df, _fn, medoids_sample, True
                        )

                        if (tmp_cost < current_cost) & (clust_iter < 1):
                            best_choices = list(medoids_sample)

                            best_results = dict(tmp_medoids)
                            current_cost = tmp_cost
                            clust_iter += 1
                        else:

                            best_choices = best_choices
                            best_results = best_results
                            current_cost = current_cost
                        medoids_sample[idx] = swap_temp

            iter_count += 1
            if best_choices == medoids_sample:
                print("Best configuration found! best_choices: ", best_choices)
                break

            if current_cost <= prior_cost:
                if current_cost < prior_cost:
                    print(
                        "Better configuration found! curr_cost:{0}, prior_cost:{1}".format(
                            round(current_cost, 2), round(prior_cost, 2)
                        )
                    )
                else:
                    print("Equal cost")
                prior_cost = current_cost
                medoids = best_results
                medoids_sample = best_choices

            print("new_medoids: ", best_choices)

        return current_cost, best_choices, best_results

    def compute_cost(self, _df, _fn, _cur_choice, cache_on=False):
        """A function to compute the configuration cost.
        :param _df: The input dataframe.
        :param _fn: The distance function.
        :param _cur_choice: The current set of medoid choices.
        :param cache_on: Binary flag to turn caching.
        :return: The total configuration cost, the medoids.
        """
        total_cost = 0.0
        medoids = {}
        for idx in _cur_choice:
            medoids[idx] = []

        for i in list(_df.index):
            choice = -1
            min_cost = np.inf
            for m in medoids:
                if cache_on:
                    tmp = self.dist_cache.get((m, i), None)

                if not cache_on or tmp is None:
                    if _fn == "manhattan":
                        tmp = self.manhattan_distance(_df.loc[m], _df.loc[i])
                    elif _fn == "cosine":
                        tmp = self.cosine_distance(_df.loc[m], _df.loc[i])
                    elif _fn == "euclidean":
                        tmp = self.euclidean_distance(_df.loc[m], _df.loc[i])
                    elif _fn == "fast_euclidean":
                        tmp = self.fast_euclidean(_df.loc[m], _df.loc[i])
                    else:
                        print(
                            "You need to input a valid distance function (manhattan, cosine, euclidean or "
                            "fast_euclidean)."
                        )

                if cache_on:
                    self.dist_cache[(m, i)] = tmp

                if tmp < min_cost:
                    choice = m
                    min_cost = tmp

            medoids[choice].append(i)
            total_cost += min_cost
        # print("total_cost: ", total_cost)
        return total_cost, medoids

    def average_cost(self, _df, _fn, _cur_choice):
        """A function to compute the average cost.
        :param _df: The input data frame.
        :param _fn: The distance function.
        :param _cur_choice: The current medoid candidates.
        :return: The average cost, the new medoids.
        """
        _tc, _m = self.compute_cost(_df, _fn, _cur_choice)
        avg_cost = _tc / len(_m)
        return avg_cost, _m

    def cheat_at_sampling(self, _df, _k, _fn, _nsamp):
        """A function to cheat at sampling for speed ups.
        :param _df: The input dataframe.
        :param _k: The number of medoids.
        :param _fn: The distance function.
        :param _nsamp: The number of samples.
        :return: The best score, the medoids.
        """
        # this function tries _nsamp different configurations of initial medoids and chooses the one with the lowest
        # cost
        print("cheating at sampling")
        score_holder = []
        medoid_holder = []
        for _ in range(_nsamp):  # 17 by default
            # take _k random points as medoids_sample
            medoids_sample = random.sample(list(_df.index), _k)
            # compute cost and medoids with this medoids_sample
            prior_cost, medoids = self.compute_cost(
                _df, _fn, medoids_sample, True
            )
            # store the cost and medoids
            score_holder.append(prior_cost)
            medoid_holder.append(medoids)

        # take the minimum cost and the corresponding medoids
        idx = score_holder.index(min(score_holder))
        ms = medoid_holder[idx].keys()
        return score_holder[idx], list(ms)

    def euclidean_distance(self, v1, v2):
        """Slow function for euclidean distance.
        :param v1: The first vector.
        :param v2: The second vector.
        :return: The euclidean distance between v1 and v2.
        """
        dist = 0
        for a1, a2 in zip(v1, v2):
            dist += abs(a1 - a2) ** 2
        return dist

    def fast_euclidean(self, v1, v2):
        """Faster function for euclidean distance.
        :param v1: The first vector.
        :param v2: The second vector.
        :return: The euclidean distance between v1 and v2.
        """
        return np.linalg.norm(v1 - v2)

    def manhattan_distance(self, v1, v2):
        """Function for manhattan distance.
        :param v1: The first vector.
        :param v2: The second vector.
        :return: The manhattan distance between v1 and v2.
        """
        dist = 0
        for a1, a2 in zip(v1, v2):
            dist += abs(a1 - a2)
        return dist

    def cosine_distance(self, v1, v2):
        """Function for cosine distance.
        :param v1: The first vector.
        :param v2: The second vector.
        :return: The cosine distance between v1 and v2.
        """
        xx, yy, xy = 0, 0, 0
        for a1, a2 in zip(v1, v2):
            xx += a1 * a1
            yy += a2 * a2
            xy += a1 * a2
        return float(xy) / np.sqrt(xx * yy)


def plot_pam_mod(data, cl, full, equal_axis_scale=False):
    """
    Scatterplot of data points, with colors according to cluster labels. Only sampled
    points are plotted, the others are only displayed with their indexes; moreover,
    centers of mass of the clusters are marked with an X.

    :param data: input data sample as dataframe.
    :param cl: cluster dictionary.
    :param full: full input dataframe.
    :param equal_axis_scale: if True, axis are plotted with the same scaling.

    """

    fig, ax = plt.subplots(figsize=(14, 6))

    # just as a habit, it actually doesnt plot anything because points are white with white edgecolor
    plt.scatter(
        full.iloc[:, 0],
        full.iloc[:, 1],
        s=300,
        color="white",
        edgecolor="white",
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

    # plot the sampled point, with colors according to the cluster they belong to
    for i, el in enumerate(list(cl.values())):
        plt.scatter(
            data.loc[el, 0],
            data.loc[el, 1],
            s=300,
            color=colors[i % 18],
            edgecolor="black",
        )

    # plot centers of mass, marked with an X
    for i, el in enumerate(list(cl.keys())):
        plt.scatter(
            data.loc[el, 0],
            data.loc[el, 1],
            s=500,
            color="red",
            marker="X",
            edgecolor="black",
        )

    # plot indexes of points in plot
    for i, txt in enumerate([i for i in range(len(full))]):
        ax.annotate(
            txt,
            (full.iloc[i, 0], full.iloc[i, 1]),
            fontsize=10,
            size=10,
            ha="center",
            va="center",
        )

    if equal_axis_scale is True:
        ax.set_aspect("equal", adjustable="box")

    plt.show()
