from PyQt5.QtCore import QCoreApplication, Qt

import numpy as np
import pandas as pd

from algorithms.optics import dist1
from algorithms.agglomerative import dist_mat_gen
from matplotlib.patches import Rectangle
from copy import deepcopy

from collections import OrderedDict

import random
import matplotlib.pyplot as plt

from GUI_classes.utils_gui import choose_dataset, pause_execution, encircle, convert_colors

from GUI_classes.generic_gui import StartingGui

from algorithms.cure import update_mat_cure, sel_rep_fast


class CURE_class(StartingGui):
    def __init__(self):
        super(CURE_class, self).__init__(name="CURE", twinx=False, first_plot=True, second_plot=False,
                                         function=self.start_CURE, extract=False, stretch_plot=True)

    def start_CURE(self):

        self.ax1.cla()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_clust = int(self.line_edit_n_clust.text())
        self.n_repr = int(self.line_edit_n_repr.text())
        self.alpha_cure = float(self.line_edit_alpha_cure.text())
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

        self.cure_gui(data=self.X, k=self.n_clust, delay=self.delay)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        # self.button_extract.setEnabled(True)
        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)

    def point_plot_mod2_gui(self, a, reps, level_txt, level2_txt=None,
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
        self.ax1.cla()
        self.ax1.set_title("{} procedure".format(self.name))
        # diz is used to take the shuffling of data into account, e.g. if the first row doesn'#
        # correspond to point 0: this is useful for the large dataset version of CURE, where data points
        # are randomly sampled, but the initial indices are kept to be plotted.
        if par_index is not None:
            diz = dict(zip(par_index, [i for i in range(len(par_index))]))

        # points that still need to be processed are plotted in lime color
        self.ax1.scatter(self.X[:, 0], self.X[:, 1], s=300, color="lime", edgecolor="black")

        # drops the totally null columns, so that the number of columns goes to 2*(cardinality of biggest cluster)
        a = a.dropna(1, how="all")

        colors = {0: "seagreen", 1: 'lightcoral', 2: 'yellow', 3: 'grey',
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
                X_clust = [self.X[diz[point[j]], 0] for j in range(len(point))]
                Y_clust = [self.X[diz[point[j]], 1] for j in range(len(point))]

                self.ax1.scatter(X_clust, Y_clust, s=350, color=colors[ind % 18])
            else:
                point = [int(i) for i in point]
                X_clust = [self.X[point[j], 0] for j in range(len(point))]
                Y_clust = [self.X[point[j], 1] for j in range(len(point))]

                self.ax1.scatter(X_clust, Y_clust, s=350, color=colors[ind % 18])

        # last merged cluster, so the last element of matrix a
        point = a.iloc[-1].name.replace("(", "").replace(")", "").split("-")
        # finding the new center of mass the newly merged cluster
        if par_index is not None:
            point = [diz[point[i]] for i in range(len(point))]
            com = self.X[point].mean(axis=0)
        else:
            point = [int(i) for i in point]
            com = self.X[point].mean(axis=0)

        # plotting the center of mass, marked with an X
        self.ax1.scatter(com[0], com[1], s=400, color="r", marker="X", edgecolor="black")

        # plotting representative points in red
        x_reps = [i[0] for i in reps]
        y_reps = [i[1] for i in reps]
        self.ax1.scatter(x_reps, y_reps, s=360, color="r", edgecolor="black")

        # finding the right measures for the rectangle
        rect_min = self.X[point].min(axis=0)
        rect_diff = self.X[point].max(axis=0) - rect_min

        xwidth = self.ax1.axis()[1] - self.ax1.axis()[0]
        ywidth = self.ax1.axis()[3] - self.ax1.axis()[2]

        # adding the rectangle, using two rectangles one above the other to use different colors
        # for the border and for the inside
        if len(point) <= 5:

            self.ax1.add_patch(Rectangle((rect_min[0] - xwidth * 0.02, rect_min[1] - ywidth * 0.04),
                                         rect_diff[0] + xwidth * 0.04, rect_diff[1] + ywidth * 0.08, fill=True,
                                         color=color_dict_rect[ind % 18], linewidth=3,
                                         ec="red"))
        else:
            encircle(X_clust, Y_clust, ax=self.ax1, color=color_dict_rect[ind % 18], linewidth=3, ec="red", zorder=0)

        # adding labels to points in the plot

        if initial_ind is not None:
            for i, txt in enumerate(initial_ind):
                self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center',
                                  va='center')
        else:
            for i, txt in enumerate([i for i in range(len(self.X))]):
                self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center',
                                  va='center')

        # adding the annotations
        self.log.appendPlainText("")
        self.log.appendPlainText("min_dist: " + str(round(level_txt, 5)))

        if level2_txt is not None:
            self.log.appendPlainText("dist_incr: " + str(round(level2_txt, 5)))

        self.log.appendPlainText("n° clust: " + str(len(a)))

        self.canvas_up.draw()

        if save_plots is True:
            self.canvas_up.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

        # everything down from here refers to the last phase of the large dataset version, the assignment phase
        if last_reps is not None:

            # plot all the points in color lime
            self.ax1.scatter(self.X[:, 0], self.X[:, 1], s=300, color="lime", edgecolor="black")

            # find the centers of mass of the clusters using the matrix a to find which points belong to
            # which cluster
            coms = []
            for ind, i in enumerate(range(0, len(a))):
                point = a.iloc[i].name.replace("(", "").replace(")", "").split("-")
                for j in range(len(point)):
                    self.ax1.scatter(self.X[diz[point[j]], 0], self.X[diz[point[j]], 1], s=350, color=colors[ind % 18])
                point = [diz[point[i]] for i in range(len(point))]
                coms.append(self.X[point].mean(axis=0))

            # variations of red to plot the representative points of the various clusters
            colors_reps = ["red", "crimson", "indianred", "lightcoral", "salmon", "darksalmon", "firebrick"]

            # flattening the last_reps values
            flat_reps = [item for sublist in list(last_reps.values()) for item in sublist]

            # plotting the representatives, surrounded by small circles, and the centers of mass, marked with X
            for i in range(len(last_reps)):
                len_rep = len(list(last_reps.values())[i])

                x = [list(last_reps.values())[i][j][0] for j in range(min(n_rep_fin, len_rep))]
                y = [list(last_reps.values())[i][j][1] for j in range(min(n_rep_fin, len_rep))]

                self.ax1.scatter(x, y, s=400, color=colors_reps[i], edgecolor="black")
                self.ax1.scatter(coms[i][0], coms[i][1], s=400, color=colors_reps[i], marker="X", edgecolor="black")

                for num in range(min(n_rep_fin, len_rep)):
                    self.ax1.add_artist(plt.Circle((x[num], y[num]), xwidth * 0.03,
                                                   color=colors_reps[i], fill=False, linewidth=3, alpha=0.7))

                self.ax1.scatter(not_sampled[:, 0], not_sampled[:, 1], s=400, color="lime", edgecolor="black")

            # find the closest representative for not sampled points, and draw an arrow connecting the points
            # to its closest representative
            for ind in range(len(not_sampled)):
                dist_int = []
                for el in flat_reps:
                    dist_int.append(dist1(not_sampled[ind], el))
                ind_min = np.argmin(dist_int)

                self.ax1.arrow(not_sampled[ind][0], not_sampled[ind][1],
                               flat_reps[ind_min][0] - not_sampled[ind][0], flat_reps[ind_min][1] - not_sampled[ind][1],
                               length_includes_head=True, head_width=0.03, head_length=0.05)

            # plotting the indexes for each point
            for i, txt in enumerate(initial_ind):
                self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center',
                                  va='center')

            if not_sampled_ind is not None:
                for i, txt in enumerate(not_sampled_ind):
                    self.ax1.annotate(txt, (not_sampled[:, 0][i], not_sampled[:, 1][i]), fontsize=10, size=10,
                                      ha='center', va='center')

            self.canvas_up.draw()

            if save_plots is True:
                self.canvas_up.figure.savefig(
                    './Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

            QCoreApplication.processEvents()

        # if par_index is not None, diz is updated with the last merged cluster and its keys are returned
        if par_index is not None:
            diz["(" + u + ")" + "-" + "(" + u_cl + ")"] = len(diz)
            list_keys_diz = list(diz.keys())

            return list_keys_diz

    def cure_gui(self, data, k, plotting=True, preprocessed_data=None,
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
        self.ax1.cla()
        self.ax1.set_title("{} procedure".format(self.name))

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

                        partial_index = self.point_plot_mod2_gui(a=a, reps=rep[name],
                                                                 level_txt=levels[-1], par_index=partial_index,
                                                                 u=u, u_cl=u_cl, initial_ind=initial_index,
                                                                 last_reps=final_reps, not_sampled=not_sampled,
                                                                 not_sampled_ind=not_sampled_ind,
                                                                 n_rep_fin=n_rep_finalclust, save_plots=self.save_plots,
                                                                 ind_fig=index_for_saving_plots)

                    # in the intermediate steps of the large dataset version
                    else:
                        partial_index = self.point_plot_mod2_gui(a=a, reps=rep[name],
                                                                 level_txt=levels[-1], par_index=partial_index,
                                                                 u=u, u_cl=u_cl, initial_ind=initial_index,
                                                                 save_plots=self.save_plots,
                                                                 ind_fig=index_for_saving_plots)
                else:
                    self.point_plot_mod2_gui(a=a, reps=rep[name], level_txt=levels[-1], save_plots=self.save_plots,
                                             ind_fig=index_for_saving_plots)

                index_for_saving_plots += 1

        return clusters, rep, a

