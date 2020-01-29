from PyQt5.QtCore import QCoreApplication, Qt

import numpy as np
import pandas as pd

from algorithms.optics import dist1
from algorithms.agglomerative import dist_mat_gen
from matplotlib.patches import Rectangle
from collections import Counter
from copy import deepcopy
from math import ceil

from collections import OrderedDict

import random
import matplotlib.pyplot as plt

from GUI_classes.utils_gui import choose_dataset, pause_execution, encircle, convert_colors

from GUI_classes.generic_gui import StartingGui

from algorithms.cure import dist_mat_gen_cure, update_mat_cure, sel_rep_fast


class LARGE_CURE_class(StartingGui):
    def __init__(self):
        super(LARGE_CURE_class, self).__init__(name="LARGE CURE", twinx=False, first_plot=False, second_plot=False,
                                               function=self.start_LARGE_CURE, extract=False, stretch_plot=False)
        self.first_run_occurred_mod = False

        self.canvas_fin = FigureCanvas(Figure(figsize=(12, 5)))
        self.ax_fin = self.canvas_fin.figure.subplots()
        self.ax_fin.set_xticks([], [])
        self.ax_fin.set_yticks([], [])
        self.ax_fin.set_title("LARGE CURE final step")

    def start_LARGE_CURE(self):

        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_clust = int(self.line_edit_n_clust.text())
        self.n_repr = int(self.line_edit_n_repr.text())
        self.alpha_cure = float(self.line_edit_alpha_cure.text())
        self.p_cure = int(self.line_edit_p_cure.text())
        self.q_cure = int(self.line_edit_q_cure.text())
        self.n_points = int(self.line_edit_np.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        self.SetWindows(number=self.p_cure, first_run_boolean=self.first_run_occurred_mod)

        # self.button_extract.setEnabled(False)
        self.button_run.setEnabled(False)
        self.checkbox_saveimg.setEnabled(False)
        self.button_delete_pics.setEnabled(False)

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

        self.cure_sample_part(delay=self.delay)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        # self.button_extract.setEnabled(True)
        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)

        self.first_run_occurred_mod = True

    def point_plot_mod2_gui(self, data, a, reps, ax, canvas, level_txt, level2_txt=None,
                            par_index=None, u=None, u_cl=None, initial_ind=None, last_reps=None,
                            not_sampled=None, not_sampled_ind=None, n_rep_fin=None, save_plots=False,
                            ind_fig=None):
        """
        Scatter-plot of input data points, colored according to the cluster they belong to.
        A rectangle with red borders is displayed around the last merged cluster; representative points
        of last merged cluster are also plotted in red, along with the center of mass, plotted as a
        red cross. The current number of clusters and current distance are also displayed in the right
        upper corner.
        In the last phase of CURE algorithm variation for large datasets, arrows are
        displayed from every not sampled point to its closest representative point; moreover, representative
        points are surrounded by small circles, to make them more visible. Representative points of different
        clusters are plotted in different nuances of red.

        :param a: input dataframe built by CURE algorithm, listing the cluster and the x and y
                  coordinates of each point.
        :param reps: list of the coordinates of representative points.
        :param level_txt: distance at which current merging occurs displayed in the upper right corner.
        :param level2_txt: incremental distance (not used).
        :param par_index: partial index to take the shuffling of indexes into account.
        :param u: first cluster to be merged.
        :param u_cl: second cluster to be merged.
        :param initial_ind: initial partial index.
        :param last_reps: dictionary of last representative points.
        :param not_sampled: coordinates of points that have not been initially sampled, in the large dataset version.
        :param not_sampled_ind: indexes of not_sampled point_indices.
        :param n_rep_fin: number of representatives to use for each cluster in the final assignment phase in the large
                          dataset version.
        :return list_keys_diz: if par_index is not None, returns the new indexes of par_index.

        """
        ax.cla()
        ax.set_title("{} procedure".format(self.name))
        # diz is used to take the shuffling of data into account, e.g. if the first row doesn'#
        # correspond to point 0: this is useful for the large dataset version of CURE, where data points
        # are randomly sampled, but the initial indices are kept to be plotted.
        if par_index is not None:
            diz = dict(zip(par_index, [i for i in range(len(par_index))]))

        # points that still need to be processed are plotted in lime color
        ax.scatter(data[:, 0], data[:, 1], s=300, color="lime", edgecolor="black")

        # drops the totally null columns, so that the number of columns goes to 2*(cardinality of biggest cluster)
        a = a.dropna(1, how="all")

        colors = {0: "seagreen", 1: 'beige', 2: 'yellow', 3: 'grey',
                  4: 'pink', 5: 'turquoise', 6: 'orange', 7: 'purple', 8: 'yellowgreen', 9: 'olive', 10: 'brown',
                  11: 'tan', 12: 'plum', 13: 'rosybrown', 14: 'lightblue', 15: "khaki", 16: "gainsboro",
                  17: "peachpuff"}

        color_dict_rect = convert_colors(colors, alpha=0.3)

        # to speed things up, this splits all points inside the clusters' names, and start gives the starting index
        # that shows where clusters with more than 1 element start (because they are always appended to a)
        len_ind = [len(i.split("-")) for i in list(a.index)]
        start = np.min([i for i in range(len(len_ind)) if len_ind[i] > 1])

        # for each cluster, take the single points composing it and plot them in the appropriate color, if
        # necessary taking the labels of par_index into account
        for ind, i in enumerate(range(start, len(a))):
            point = a.iloc[i].name.replace("(", "").replace(")", "").split("-")
            if par_index is not None:
                X_clust = [data[diz[point[j]], 0] for j in range(len(point))]
                Y_clust = [data[diz[point[j]], 1] for j in range(len(point))]

                ax.scatter(X_clust, Y_clust, s=350, color=colors[ind % 18])
            else:
                point = [int(i) for i in point]
                X_clust = [data[point[j], 0] for j in range(len(point))]
                Y_clust = [data[point[j], 1] for j in range(len(point))]

                ax.scatter(X_clust, Y_clust, s=350, color=colors[ind % 18])

        # last merged cluster, so the last element of matrix a
        point = a.iloc[-1].name.replace("(", "").replace(")", "").split("-")
        # finding the new center of mass the newly merged cluster
        if par_index is not None:
            point = [diz[point[i]] for i in range(len(point))]
            com = data[point].mean(axis=0)
        else:
            point = [int(i) for i in point]
            com = data[point].mean(axis=0)

        # plotting the center of mass, marked with an X
        ax.scatter(com[0], com[1], s=400, color="r", marker="X", edgecolor="black")

        # plotting representative points in red
        x_reps = [i[0] for i in reps]
        y_reps = [i[1] for i in reps]
        ax.scatter(x_reps, y_reps, s=360, color="r", edgecolor="black")

        # finding the right measures for the rectangle
        rect_min = data[point].min(axis=0)
        rect_diff = data[point].max(axis=0) - rect_min

        xwidth = ax.axis()[1] - ax.axis()[0]
        ywidth = ax.axis()[3] - ax.axis()[2]

        # adding the rectangle, using two rectangles one above the other to use different colors
        # for the border and for the inside
        if len(point) <= 5:

            ax.add_patch(Rectangle((rect_min[0] - xwidth * 0.02, rect_min[1] - ywidth * 0.04),
                                   rect_diff[0] + xwidth * 0.04, rect_diff[1] + ywidth * 0.08, fill=True,
                                   color=color_dict_rect[ind % 18], linewidth=3,
                                   ec="red"))
        else:
            encircle(X_clust, Y_clust, ax=ax, color=color_dict_rect[ind % 18], linewidth=3, ec="red", zorder=0)

        # adding labels to points in the plot

        if initial_ind is not None:
            for i, txt in enumerate(initial_ind):
                ax.annotate(txt, (data[:, 0][i], data[:, 1][i]), fontsize=10, size=10, ha='center',
                            va='center')
        else:
            for i, txt in enumerate([i for i in range(len(data))]):
                ax.annotate(txt, (data[:, 0][i], data[:, 1][i]), fontsize=10, size=10, ha='center',
                            va='center')

        # adding the annotations
        self.log.appendPlainText("")
        self.log.appendPlainText("min_dist: " + str(round(level_txt, 5)))

        if level2_txt is not None:
            self.log.appendPlainText("dist_incr: " + str(round(level2_txt, 5)))

        self.log.appendPlainText("n° clust: " + str(len(a)))

        canvas.draw()

        if save_plots is True:
            canvas.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

        # everything down from here refers to the last phase of the large dataset version, the assignment phase
        if last_reps is not None:

            # plot all the points in color lime
            ax.scatter(data[:, 0], data[:, 1], s=300, color="lime", edgecolor="black")

            # find the centers of mass of the clusters using the matrix a to find which points belong to
            # which cluster
            coms = []
            for ind, i in enumerate(range(0, len(a))):
                point = a.iloc[i].name.replace("(", "").replace(")", "").split("-")
                for j in range(len(point)):
                    ax.scatter(data[diz[point[j]], 0], data[diz[point[j]], 1], s=350, color=colors[ind % 18])
                point = [diz[point[i]] for i in range(len(point))]
                coms.append(data[point].mean(axis=0))

            # variations of red to plot the representative points of the various clusters
            colors_reps = ["red", "crimson", "indianred", "lightcoral", "salmon", "darksalmon", "firebrick"]

            # flattening the last_reps values
            flat_reps = [item for sublist in list(last_reps.values()) for item in sublist]

            # plotting the representatives, surrounded by small circles, and the centers of mass, marked with X
            for i in range(len(last_reps)):
                len_rep = len(list(last_reps.values())[i])

                x = [list(last_reps.values())[i][j][0] for j in range(min(n_rep_fin, len_rep))]
                y = [list(last_reps.values())[i][j][1] for j in range(min(n_rep_fin, len_rep))]

                ax.scatter(x, y, s=400, color=colors_reps[i], edgecolor="black", zorder=10)
                ax.scatter(coms[i][0], coms[i][1], s=400, color=colors_reps[i], marker="X", edgecolor="black")

                for num in range(min(n_rep_fin, len_rep)):
                    ax.add_artist(plt.Circle((x[num], y[num]), xwidth * 0.03,
                                             color=colors_reps[i], fill=False, linewidth=3, alpha=0.7))

                ax.scatter(not_sampled[:, 0], not_sampled[:, 1], s=400, color="lime", edgecolor="black")

            # find the closest representative for not sampled points, and draw an arrow connecting the points
            # to its closest representative
            for ind in range(len(not_sampled)):
                dist_int = []
                for el in flat_reps:
                    dist_int.append(dist1(not_sampled[ind], el))
                ind_min = np.argmin(dist_int)

                ax.arrow(not_sampled[ind][0], not_sampled[ind][1],
                         flat_reps[ind_min][0] - not_sampled[ind][0], flat_reps[ind_min][1] - not_sampled[ind][1],
                         length_includes_head=True, head_width=0.03, head_length=0.05)

            # plotting the indexes for each point
            for i, txt in enumerate(initial_ind):
                ax.annotate(txt, (data[:, 0][i], data[:, 1][i]), fontsize=10, size=10, ha='center',
                            va='center')

            if not_sampled_ind is not None:
                for i, txt in enumerate(not_sampled_ind):
                    ax.annotate(txt, (not_sampled[:, 0][i], not_sampled[:, 1][i]), fontsize=10, size=10,
                                ha='center', va='center')

            canvas.draw()

            if save_plots is True:
                canvas.figure.savefig(
                    './Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

            QCoreApplication.processEvents()

        # if par_index is not None, diz is updated with the last merged cluster and its keys are returned
        if par_index is not None:
            diz["(" + u + ")" + "-" + "(" + u_cl + ")"] = len(diz)
            list_keys_diz = list(diz.keys())

            return list_keys_diz

    def cure_gui(self, data, k, ax, canvas, plotting=True, preprocessed_data=None,
                 partial_index=None, n_rep_finalclust=None, not_sampled=None, not_sampled_ind=None, delay=0):
        """
        CURE algorithm: hierarchical agglomerative clustering using representatives.
        :param data: input data.
        :param plotting: if True, plots all intermediate steps.
        :param k: the desired number of clusters.

        #the following parameter are used for the large dataset variation of CURE

        :param preprocessed_data: if not None, must be of the form (clusters,representatives,matrix_a,X_dist1),
                                  which is used to perform a warm start.
        :param partial_index: if not None, is is used as index of the matrix_a, of cluster points and of
                              representatives.
        :param n_rep_finalclust: the final representative points used to classify the not_sampled points.
        :param not_sampled: points not sampled in the initial phase.
        :param not_sampled_ind: indexes of not_sampled points.
        :return (clusters, rep, a): returns the clusters dictionary, the dictionary of representatives,
                                    the matrix a


        """
        ax.cla()
        ax.set_title("{} procedure".format(self.name))

        index_for_saving_plots = 0
        # starting from raw data
        if preprocessed_data is None:
            # building a dataframe storing the x and y coordinates of input data points
            l = [[i, i] for i in range(len(data))]
            flat_list = [item for sublist in l for item in sublist]
            col = [str(el) + "x" if i % 2 == 0 else str(el) + "y" for i, el in enumerate(flat_list)]

            # using the original indexes if necessary
            if partial_index is not None:
                a = pd.DataFrame(index=partial_index, columns=col)
            else:
                a = pd.DataFrame(index=[str(i) for i in range(len(data))], columns=col)

            # adding the real coordinates
            a["0x"] = data.T[0]
            a["0y"] = data.T[1]

            b = a.dropna(axis=1, how="all")

            # initial clusters
            if partial_index is not None:
                clusters = dict(zip(partial_index, data))
            else:
                clusters = {str(i): np.array(data[i]) for i in range(len(data))}

            # build Xdist
            X_dist1 = dist_mat_gen(b)

            # initialize representatives
            if partial_index is not None:
                rep = {partial_index[i]: [data[int(i)]] for i in range(len(data))}
            else:
                rep = {str(i): [data[i]] for i in range(len(data))}

            # just as placeholder for while loop
            heap = [1] * len(X_dist1)

            # store minimum distances between clusters for each iteration
            levels = []

        # use precomputed data
        else:

            clusters = preprocessed_data[0]
            rep = preprocessed_data[1]
            a = preprocessed_data[2]
            X_dist1 = preprocessed_data[3]
            heap = [1] * len(X_dist1)
            levels = []

        # store original index
        if partial_index is not None:
            initial_index = deepcopy(partial_index)

        # while the desired number of clusters has not been reached
        while len(heap) > k:

            # find minimum value of heap queu, which stores clusters according to the distance from
            # their closest cluster
            list_argmin = list(X_dist1.apply(lambda x: np.argmin(x)).values)
            list_min = list(X_dist1.min(axis=0).values)
            heap = dict(zip(list(X_dist1.index), list_min))
            heap = dict(OrderedDict(sorted(heap.items(), key=lambda kv: kv[1])))
            closest = dict(zip(list(X_dist1.index), list_argmin))

            # get minimum keys and delete them from heap and closest dictionaries
            u = min(heap, key=heap.get)
            levels.append(heap[u])
            del heap[u]
            u_cl = closest[u]
            del closest[u]

            # form the new cluster
            if (np.array(clusters[u]).shape == (2,)) and (np.array(clusters[u_cl]).shape == (2,)):
                w = [clusters[u], clusters[u_cl]]
            elif (np.array(clusters[u]).shape != (2,)) and (np.array(clusters[u_cl]).shape == (2,)):
                clusters[u].append(clusters[u_cl])
                w = clusters[u]
            elif (np.array(clusters[u]).shape == (2,)) and (np.array(clusters[u_cl]).shape != (2,)):
                clusters[u_cl].append(clusters[u])
                w = clusters[u_cl]
            else:
                w = clusters[u] + clusters[u_cl]

            # delete old cluster
            del clusters[u]
            del clusters[u_cl]

            # set new name
            name = "(" + u + ")" + "-" + "(" + u_cl + ")"
            clusters[name] = w

            # update representatives
            rep[name] = sel_rep_fast(rep[u] + rep[u_cl], clusters, name, self.n_repr, self.alpha_cure)

            # update distance matrix
            X_dist1 = update_mat_cure(X_dist1, u, u_cl, rep, name)

            # delete old representatives
            del rep[u]
            del rep[u_cl]

            if plotting == True:
                if delay != 0:
                    pause_execution(self.delay)

                dim1 = int(a.loc[u].notna().sum())
                # update the matrix a with the new cluster
                a.loc["(" + u + ")" + "-" + "(" + u_cl + ")", :] = a.loc[u].fillna(0) + a.loc[u_cl].shift(dim1,
                                                                                                          fill_value=0)
                a = a.drop(u, 0)
                a = a.drop(u_cl, 0)

                # in the large dataset version of CURE
                if partial_index is not None:

                    # only in last step of large dataset version of CURE
                    if (len(heap) == k) and (not_sampled is not None) and (not_sampled_ind is not None):

                        # take random representative points from the final representatives
                        final_reps = {list(rep.keys())[i]: random.sample(list(rep.values())[i],
                                                                         min(n_rep_finalclust,
                                                                             len(list(rep.values())[i])))
                                      for i in range(len(rep))}

                        partial_index = self.point_plot_mod2_gui(data=data, a=a, reps=rep[name], ax=ax, canvas=canvas,
                                                                 level_txt=levels[-1], par_index=partial_index,
                                                                 u=u, u_cl=u_cl, initial_ind=initial_index,
                                                                 last_reps=final_reps, not_sampled=not_sampled,
                                                                 not_sampled_ind=not_sampled_ind,
                                                                 n_rep_fin=n_rep_finalclust, save_plots=self.save_plots,
                                                                 ind_fig=index_for_saving_plots)

                    # in the intermediate steps of the large dataset version
                    else:
                        partial_index = self.point_plot_mod2_gui(data=data, a=a, reps=rep[name], ax=ax, canvas=canvas,
                                                                 level_txt=levels[-1], par_index=partial_index,
                                                                 u=u, u_cl=u_cl, initial_ind=initial_index,
                                                                 save_plots=self.save_plots,
                                                                 ind_fig=index_for_saving_plots)
                else:
                    self.point_plot_mod2_gui(a=a, reps=rep[name], ax=ax, canvas=canvas, level_txt=levels[-1],
                                             save_plots=self.save_plots,
                                             ind_fig=index_for_saving_plots)

            index_for_saving_plots += 1

        return clusters, rep, a

    def cure_sample_part(self, u_min=None, f=0.3, d=0.02, n_rep_finalclust=None, delay=0):
        """
        CURE algorithm variation for large datasets.
        Partition the sample space into p partitions, each of size len(X)/p, then partially cluster each
        partition until the final number of clusters in each partition reduces to n/(pq). Then run a second
        clustering pass on the n/q partial clusters for all the partitions.

        :param u_min: size of the smallest cluster u.
        :param f: percentage of cluster points (0 <= f <= 1) we would like to have in the sample.
        :param d: (0 <= d <= 1) the probability that the sample contains less than f*|u| points of cluster u is less than d.
        :param n_rep_finalclust: number of representatives to use in the final assignment phase.
        :return (clusters, rep, mat_a): returns the clusters dictionary, the dictionary of representatives,
                                    the matrix a
        """

        # choose the parameters suggested by the paper if the user doesnt provide input parameters
        if u_min is None:
            u_min = round(len(self.X) / self.n_clust)

        if n_rep_finalclust is None:
            n_rep_finalclust = self.n_repr

        l = [[i, i] for i in range(len(self.X))]
        flat_list = [item for sublist in l for item in sublist]
        col = [str(el) + "x" if i % 2 == 0 else str(el) + "y" for i, el in enumerate(flat_list)]
        a = pd.DataFrame(index=[str(i) for i in range(len(self.X))], columns=col)
        a["0x"] = self.X.T[0]
        a["0y"] = self.X.T[1]
        b = a.dropna(axis=1, how="all")

        # this is done to ensure that the algorithm starts even when input params are bad
        while True:
            try:
                self.log.appendPlainText("")
                self.log.appendPlainText("new f: {}".format(round(f, 4)))
                self.log.appendPlainText("new d: {}".format(round(d, 4)))
                n = ceil(self.Chernoff_Bounds_gui(u_min=u_min, f=f, N=len(self.X), k=self.n_clust, d=d))
                b_sampled = b.sample(n, random_state=42)
                break
            except:
                if f >= 0.19:
                    f = f - 0.1
                else:
                    d = d * 2

        b_notsampled = b.loc[[str(i) for i in range(len(b)) if str(i) not in b_sampled.index], :]

        # find the best p and q according to the paper
        if (self.p_cure is None) and (self.q_cure is None):

            def g(x):
                res = (x[1] - 1) / (x[0] * x[1]) + 1 / (x[1] ** 2)
                return res

            results = {}
            for i in range(2, 15):
                for j in range(2, 15):
                    results[(i, j)] = g([i, j])
            self.p_cure, self.q_cure = max(results, key=results.get)
            self.log.appendPlainText("p was automatically set to: {}".format(self.p_cure))
            self.log.appendPlainText("q was automatically set to: {}".format(self.q_cure))

        if (n / (self.p_cure * self.q_cure)) < 2 * self.n_clust:
            self.log.appendPlainText("")
            self.log.appendPlainText("CAUTION")
            self.log.appendPlainText("n/pq is less than 2k, results could be wrong")

        # form the partitions
        z = round(n / self.p_cure)
        lin_sp = np.linspace(0, n, self.p_cure + 1, dtype="int")
        # lin_sp
        b_partitions = []
        for num_p in range(self.p_cure):
            try:
                b_partitions.append(b_sampled.iloc[lin_sp[num_p]:lin_sp[num_p + 1]])
            except:
                b_partitions.append(b_sampled.iloc[lin_sp[num_p]:])

        k_prov = round(n / (self.p_cure * self.q_cure))

        # perform clustering on each partition separately
        partial_clust1 = []
        partial_rep1 = []
        partial_a1 = []

        dict_axes = {2: [self.ax1, self.ax2], 3: [self.ax1, self.ax2, self.ax3],
                     4: [self.ax1, self.ax2, self.ax3, self.ax4]}

        dict_canvas = {2: [self.canvas_up, self.canvas_2], 3: [self.canvas_up, self.canvas_2, self.canvas_3],
                       4: [self.canvas_up, self.canvas_2, self.canvas_3, self.canvas_4]}

        p_dict_list = dict_axes[int(self.p_cure)]
        p_dict_canvas = dict_canvas[int(self.p_cure)]

        for i in range(self.p_cure):
            self.log.appendPlainText("")
            self.log.appendPlainText("partition number: {}".format(i + 1))
            clusters, rep, mat_a = self.cure_gui(data=b_partitions[i].values, k=k_prov, ax=p_dict_list[i],
                                                 canvas=p_dict_canvas[i],
                                                 partial_index=b_partitions[i].index, delay=delay)
            partial_clust1.append(clusters)
            partial_rep1.append(rep)
            partial_a1.append(mat_a)

        # merging all data into single components
        # clusters
        clust_tot = {}
        for d in partial_clust1:
            clust_tot.update(d)
        # representatives
        rep_tot = {}
        for d in partial_rep1:
            rep_tot.update(d)
        # mat a
        diz = {i: len(b_partitions[i]) for i in range(self.p_cure)}
        num_freq = Counter(diz.values()).most_common(1)[0][0]
        bad_ind = [list(diz.keys())[i] for i in range(len(diz)) if diz[i] != num_freq]

        for ind in bad_ind:
            partial_a1[ind]["{0}x".format(diz[ind])] = [np.nan] * k_prov
            partial_a1[ind]["{0}y".format(diz[ind])] = [np.nan] * k_prov

        for i in range(len(partial_a1) - 1):
            if i == 0:
                a_tot = partial_a1[i].append(partial_a1[i + 1])
            else:
                a_tot = a_tot.append(partial_a1[i + 1])
        # mat Xdist
        X_dist_tot = dist_mat_gen_cure(rep_tot)

        # final_clustering
        prep_data = [clust_tot, rep_tot, a_tot, X_dist_tot]

        self.openFinalStepWindow(ax=self.ax_fin, canvas=self.canvas_fin)

        clusters, rep, mat_a = self.cure_gui(b_sampled.values, k=self.n_clust, ax=self.ax_fin,
                                             canvas=self.canvas_fin,
                                             preprocessed_data=prep_data,
                                             partial_index=b_sampled.index, n_rep_finalclust=n_rep_finalclust,
                                             not_sampled=b_notsampled.values,
                                             not_sampled_ind=b_notsampled.index, delay=delay)

        return clusters, rep, mat_a

    def Chernoff_Bounds_gui(self, u_min, f, N, d, k):
        """
        u_min: size of the smallest cluster u.
        f: percentage of cluster points (0 <= f <= 1).
        N: total size.
        s: sample size.
        d: 0 <= d <= 1
        the probability that the sample contains less than f*|u| points of cluster u is less than d.

        If one uses as |u| the minimum cluster size we are interested in, the result is
        the minimum sample size that guarantees that for k clusters
        the probability of selecting fewer than f*|u| points from any one of the clusters u is less than k*d.

        """

        l = np.log(1 / d)
        res = f * N + N / u_min * l + N / u_min * np.sqrt(l ** 2 + 2 * f * u_min * l)
        self.log.appendPlainText("")

        msg = ("If the sample size is {}, the probability of selecting fewer than {} points from"
               " any one of the clusters is less than {}".format(ceil(res), round(f * u_min), k * d))

        self.log.appendPlainText(msg)

        return res

    def openFinalStepWindow(self, ax, canvas):
        self.w = FinalStepWindow(ax=ax, canvas=canvas)
        self.w.show()
        # self.hide()


from PyQt5.QtWidgets import QLabel, QWidget, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


class FinalStepWindow(QMainWindow):
    def __init__(self, ax, canvas):
        super().__init__()
        self.setWindowTitle("prova prova")

        canvas.draw()

        # if save_plots is True:
        #     canvas.figure.savefig(
        #         './Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

