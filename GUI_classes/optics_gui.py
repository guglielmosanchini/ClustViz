from PyQt5.QtCore import QCoreApplication, Qt

import numpy as np
import pandas as pd

from collections import OrderedDict

from algorithms.optics import scan_neigh1, reach_dist, minPTSdist, ExtractDBSCANclust

import random
import matplotlib.pyplot as plt

from GUI_classes.utils_gui import choose_dataset, pause_execution

from GUI_classes.generic_gui import StartingGui


class OPTICS_class(StartingGui):
    def __init__(self):
        super(OPTICS_class, self).__init__(name="OPTICS", twinx=True, first_plot=True, second_plot=True,
                                           function=self.start_OPTICS, extract=True)

    def start_OPTICS(self):

        self.ax.cla()
        self.ax1.cla()
        self.ax_t.cla()
        self.ax1_t.cla()
        self.ax1_t.set_yticks([], [])

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.eps = float(self.line_edit_eps.text())
        self.mp = int(self.line_edit_mp.text())
        self.eps_extr = float(self.line_edit_eps_extr.text())
        self.n_points = int(self.line_edit_np.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        self.button_extract.setEnabled(False)
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

        self.OPTICS_gui(plot=True, plot_reach=True, delay=self.delay)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_extract.setEnabled(True)
        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)

    def start_EXTRACT_OPTICS(self):

        self.ax.cla()
        self.ax1.cla()
        self.ax_t.cla()
        self.ax1_t.cla()
        self.ax1_t.set_yticks([], [])

        self.verify_input_parameters(extract=True)

        if self.param_check is False:
            return

        self.eps_extr = float(self.line_edit_eps_extr.text())
        self.plot_clust_gui(save_plots=self.save_plots)
        if (Qt.Checked == self.checkbox_saveimg.checkState()):
            self.ind_extr_fig += 1
            self.first_run_occurred = True
        self.clear_seed_log(final=True)

    def point_plot_gui(self, X_dict, coords, neigh, processed=None, col='yellow', save_plots=False, ind_fig=None):
        """
        Plots a scatter plot of points, where the point (x,y) is light black and
        surrounded by a red circle of radius eps, where processed point are plotted
        in col (yellow by default) and without edgecolor, whereas still-to-process points are green
        with black edgecolor.

        :param X_dict: input dictionary version of X.
        :param coords: coordinates of the point that is currently inspected.
        :param neigh: neighborhood of the point as dictionary.
        :param processed: already processed points, to plot in col
        :param col: color to use for processed points, yellow by default.
        :param ind_fig: index of the figure that is saved.
        :param save_plots: if True, the produced image is saved.

        """

        # fig, ax = plt.subplots(figsize=(14, 6))
        self.ax1.cla()
        self.ax1.set_title("{} procedure".format(self.name))

        # plot every point in color lime
        self.ax1.scatter(self.X[:, 0], self.X[:, 1], s=300, color="lime", edgecolor="black", label="unprocessed")

        # plot clustered points according to appropriate colors
        if processed is not None:
            X_not_proc = [X_dict[i][0] for i in processed]
            Y_not_proc = [X_dict[i][1] for i in processed]
            self.ax1.scatter(X_not_proc, Y_not_proc, s=300, color=col, label="processed")

        # plot points in neighboorhood in red, if neigh is not empty
        if len(neigh) != 0:
            neigh_array = np.array(list(neigh.values()))
            self.ax1.scatter(neigh_array[:, 0], neigh_array[:, 1], s=300, color="red", label="neighbors")

        # plot last added point in black and surround it with a red circle
        self.ax1.scatter(x=coords[0], y=coords[1], s=400, color="black", alpha=0.4)

        circle1 = plt.Circle((coords[0], coords[1]), self.eps, color='r', fill=False, linewidth=3, alpha=0.7)
        self.ax1.add_artist(circle1)

        for i, txt in enumerate([i for i in range(len(self.X))]):
            self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center', va='center')

        # self.ax1.set_aspect('equal')
        self.ax1.legend(fontsize=8)
        self.canvas_up.draw()

        if save_plots is True:
            self.canvas_up.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

    def reach_plot_gui(self, data, save_plots=False, ind_fig=None):
        """
        Plots the reachability plot, along with a horizontal line denoting eps,
        from the ClustDist produced by OPTICS.

        :param data: input dictionary.
        :param ind_fig: index of the figure that is saved.
        :param save_plots: if True, the produced image is saved.
        """

        plot_dic = {}

        # create dictionary for reachability plot, keys will be the bar labels and the value will be the height
        # if the value is infinity, the height will be eps*1.15 by default
        for key, value in self.ClustDist.items():

            if np.isinf(value) == True:

                plot_dic[key] = self.eps * 1.15

            else:

                plot_dic[key] = self.ClustDist[key]

        missing_keys = list(set(data.keys()) - set(self.ClustDist.keys()))

        tick_list = list(self.ClustDist.keys()) + [' '] * (len(missing_keys))

        # add the necessary zeroes for points that are still to be processed
        for m_k in missing_keys:
            plot_dic[m_k] = 0

        # fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        self.ax.cla()

        self.ax.set_title("Reachability Plot")
        self.ax.set_ylabel("reachability distance")

        self.ax.bar(plot_dic.keys(), plot_dic.values())

        self.ax.set_xticklabels(tick_list, rotation=90, fontsize=8)

        # plot horizontal line for eps
        self.ax.axhline(self.eps, color="red", linewidth=3)

        self.ax_t.set_ylim(self.ax.get_ylim())
        self.ax_t.set_yticks([self.eps])
        self.ax_t.set_yticklabels(["\u03B5"])

        self.canvas_down.draw()

        if save_plots is True:
            self.canvas_down.figure.savefig('./Images/{}_{:02}/reachplot_{:02}.png'.format(self.name,
                                                                                           self.ind_run, ind_fig))

        QCoreApplication.processEvents()

    def plot_clust_gui(self, save_plots=False):
        """
        Plot a scatter plot on the left, where points are colored according to the cluster they belong to,
        and a reachability plot on the right, where colors correspond to the clusters, and the two horizontal
        lines represent eps and eps_db.
        """

        self.ax1.set_title("OPTICS Cluster Plot")

        self.ax.set_title("OPTICS Reachability Plot")
        self.ax.set_ylabel("reachability distance")

        X_dict = dict(zip([str(i) for i in range(len(self.X))], self.X))

        # extract the cluster dictionary using DBSCAN
        cl = ExtractDBSCANclust(self.ClustDist, self.CoreDist, self.eps_extr)

        new_dict = {key: (val1, cl[key]) for key, val1 in zip(list(X_dict.keys()), list(X_dict.values()))}

        new_dict = OrderedDict((k, new_dict[k]) for k in list(self.ClustDist.keys()))

        df = pd.DataFrame(dict(x=[i[0][0] for i in list(new_dict.values())],
                               y=[i[0][1] for i in list(new_dict.values())],
                               label=[i[1] for i in list(new_dict.values())]), index=new_dict.keys())

        colors = {-1: 'red', 0: 'lightblue', 1: 'beige', 2: 'yellow', 3: 'grey',
                  4: 'pink', 5: 'navy', 6: 'orange', 7: 'purple', 8: 'salmon', 9: 'olive', 10: 'brown',
                  11: 'tan', 12: 'lime'}

        # first plot: scatter plot of points colored according to the cluster they belong to
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=self.ax1, kind='scatter', x='x', y='y', label=key,
                       color=colors[key % 13 if key != -1 else -1],
                       s=300,
                       edgecolor="black")

        self.ax1.set_xlabel("")
        self.ax1.set_ylabel("")

        for i, txt in enumerate([i for i in range(len(self.X))]):
            self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center', va='center')

        # second plot: reachability plot, with colors corresponding to clusters
        plot_dic = {}

        for key, value in self.ClustDist.items():

            if np.isinf(value) == True:

                plot_dic[key] = self.eps * 1.15

            else:

                plot_dic[key] = self.ClustDist[key]

        tick_list = list(self.ClustDist.keys())

        self.ax.bar(plot_dic.keys(), plot_dic.values(),
                    color=[colors[i % 13] if i != -1 else "red" for i in df.label])

        self.ax.axhline(self.eps, color="black", linewidth=3)

        self.ax.axhline(self.eps_extr, color="black", linewidth=3)

        self.ax_t.set_ylim(self.ax.get_ylim())
        self.ax_t.set_yticks([self.eps, self.eps_extr])
        self.ax_t.set_yticklabels(["\u03B5", "\u03B5" + "\'"])
        self.ax.set_xticklabels(tick_list, rotation=90, fontsize=8)

        self.canvas_up.draw()
        self.canvas_down.draw()

        if save_plots is True:
            self.canvas_up.figure.savefig('./Images/{}_{:02}/fig_fin_{:02}.png'.format(self.name, self.ind_run,
                                                                                       self.ind_extr_fig))
            self.canvas_down.figure.savefig('./Images/{}_{:02}/reach_plot_fin_{:02}.png'.format(self.name,
                                                                                                self.ind_run,
                                                                                                self.ind_extr_fig))

        QCoreApplication.processEvents()

    def clear_seed_log(self, Seed=None, point=None, final=False):
        """ Take care of the log, updating it with information about the current point beign examined,
        the current seed queue and, after the execution of OPTICS, it displays the core distance of every point.

        :param Seed: current seed queue created by OPTICS.
        :param point: current point being examined by OPTICS.
        :param final: if True, plot core distances produced by OPTICS.

        """

        # during execution, add the current point being examined to the log, along with all other points
        # in the seed queue, listed with their reachability distances
        if final is False:

            self.log.clear()
            self.log.appendPlainText("SEED QUEUE")
            self.log.appendPlainText("")
            self.log.appendPlainText("current point: " + str(point))
            self.log.appendPlainText("")

            if len(Seed) != 0:
                rounded_values = [round(i, 3) for i in list(Seed.values())]
                rounded_dict = {k: v for k, v in zip(Seed.keys(), rounded_values)}
                self.log.appendPlainText("queue: ")
                self.log.appendPlainText("")
                for k, v in rounded_dict.items():
                    self.log.appendPlainText(str(k) + ": " + str(v))
            else:
                self.log.appendPlainText("empty queue")

        # at the end of the algorithm, plot the dictionary of core distances, ordered by value
        else:
            self.log.clear()
            self.log.appendPlainText("CORE DISTANCES")
            self.log.appendPlainText("")
            rounded_values = [round(i, 3) for i in list(self.CoreDist.values())]
            rounded_dict = {k: v for k, v in zip(self.CoreDist.keys(), rounded_values)}
            rounded_dict_sorted = {k: v for k, v in sorted(rounded_dict.items(), key=lambda item: item[1])}
            for k, v in rounded_dict_sorted.items():
                self.log.appendPlainText(str(k) + ": " + str(v))

    def OPTICS_gui(self, plot=True, plot_reach=False, delay=0):
        """
        Executes the OPTICS algorithm. Similar to DBSCAN, but uses a priority queue.

        :param plot: if True, the scatter plot of the function point_plot is displayed at each step.
        :param plot_reach: if True, the reachability plot is displayed at each step.
        :param delay: seconds for which to delay the algorithm, so that the images displayes in the GUI
                      show at a slower pace.
        :return (ClustDist, CoreDist): ClustDist, a dictionary of the form point_index:reach_dist, and
                 CoreDist, a dictionary of the form point_index:core_dist
        """

        self.ClustDist = {}
        self.CoreDist = {}
        Seed = {}
        processed = []
        index_for_saving_plots = 0

        # create dictionary
        X_dict = dict(zip([str(i) for i in range(len(self.X))], self.X))

        # until all points have been processed
        while len(processed) != len(self.X):

            # if queue is empty take a random point
            if len(Seed) == 0:

                unprocessed = list(set(list(X_dict.keys())) - set(processed))

                (o, r) = (random.choice(unprocessed), np.inf)

                self.clear_seed_log(Seed, o)

            # else take the minimum and delete it from the queue
            else:

                (o, r) = (min(Seed, key=Seed.get), Seed[min(Seed, key=Seed.get)])

                self.clear_seed_log(Seed, o)

                del Seed[o]

                self.clear_seed_log(Seed, o)

            # scan the neighborhood of the point
            N = scan_neigh1(X_dict, X_dict[o], self.eps)

            # update the cluster dictionary and the core distance dictionary
            self.ClustDist.update({o: r})

            self.CoreDist.update({o: minPTSdist(X_dict, o, self.mp, self.eps)})

            if delay != 0:
                pause_execution(delay)

            if plot == True:
                self.point_plot_gui(X_dict, X_dict[o], N, processed, save_plots=self.save_plots,
                                    ind_fig=index_for_saving_plots)

                if plot_reach == True:
                    self.reach_plot_gui(X_dict, save_plots=self.save_plots, ind_fig=index_for_saving_plots)
                    index_for_saving_plots += 1

            # mark o as processed
            processed.append(o)

            # if the point is core
            if len(N) >= self.mp - 1:
                # for each unprocessed point in the neighborhood
                for n in N:

                    if n in processed:

                        continue

                    else:
                        # compute its reach_dist from o
                        p = reach_dist(X_dict, n, o, self.mp, self.eps)

                        # if it is in Seed, update its reach_dist if it is lower
                        if n in Seed:

                            if p < Seed[n]:
                                Seed[n] = p

                                self.clear_seed_log(Seed, o)
                        # else, insert it into the Seed
                        else:

                            Seed.update({n: p})

                            self.clear_seed_log(Seed, o)

        self.start_EXTRACT_OPTICS()
