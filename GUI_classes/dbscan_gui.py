from PyQt5.QtCore import QCoreApplication, QRect, Qt
import pandas as pd
from collections import OrderedDict

import random

from algorithms.dbscan import scan_neigh1_mod

import matplotlib.pyplot as plt

from GUI_classes.utils_gui import choose_dataset, pause_execution

from GUI_classes.generic_gui import StartingGui


class DBSCAN_class(StartingGui):
    def __init__(self):
        super(DBSCAN_class, self).__init__(name="DBSCAN", twinx=False, first_plot=True, second_plot=False,
                                           function=self.start_DBSCAN, stretch_plot=True)

    def start_DBSCAN(self):

        self.ax1.cla()

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.eps = float(self.line_edit_eps.text())
        self.mp = int(self.line_edit_mp.text())
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

        self.DBSCAN_gui(plotting=True, print_details=True, delay=self.delay)

        self.ax1.cla()
        self.plot_clust_DB_gui(save_plots=self.save_plots)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)

    def update_log(self, point=None, msg=None, initial=False, noise=False, change_current=False, change_subcurrent=False):
        """ Take care of the log, updating it with information about the current point being examined,

        :param point: current point being examined by DBSCAN.
        :param msg: message to be displayed.
        :param initial: what to do at the start of the algorithm.
        :param change_subcurrent: True means that the current subpoint examined has changed.
        :param change_current: True means that the current point examined has changed.
        :param noise: True means that the current point has been classified as noise.

        """
        if initial is True:
            self.log.clear()
            self.log.appendPlainText("{} LOG".format(self.name))
            self.log.appendPlainText("")

        else:
            if change_current is True:
                self.log.appendPlainText("")
                self.log.appendPlainText("current point: " + str(point))

            if change_subcurrent is True:
                self.log.appendPlainText("")
                self.log.appendPlainText("  current subpoint: " + str(point))

            if noise is True:
                self.log.appendPlainText("")
                self.log.appendPlainText("  noise")

            self.log.appendPlainText("")
            self.log.appendPlainText(msg)

    def point_plot_mod_gui(self, X_dict, point, save_plots=False, ind_fig=None):
        """
        Plots a scatter plot of points, where the point (x,y) is light black and
        surrounded by a red circle of radius eps, where already processed point are plotted
        according to ClustDict and without edgecolor, whereas still-to-process points are green
        with black edgecolor.

        :param X_dict: input dictionary version of self.X.
        :param point: coordinates of the point that is currently inspected.
        :param ind_fig: index of the current plot.
        :param save_plots: if True, saves the plot.

        """

        colors = {-1: 'red', 0: 'lightblue', 1: 'beige', 2: 'yellow', 3: 'grey',
                  4: 'pink', 5: 'navy', 6: 'orange', 7: 'purple', 8: 'salmon', 9: 'olive', 10: 'brown',
                  11: 'tan', 12: 'lime'}

        self.ax1.cla()
        self.ax1.set_title("DBSCAN procedure")

        # plot scatter points in color lime
        self.ax1.scatter(self.X[:, 0], self.X[:, 1], s=300, color="lime", edgecolor="black")

        # plot colors according to clusters
        for i in self.ClustDict:
            self.ax1.scatter(X_dict[i][0], X_dict[i][1],
                             color=colors[self.ClustDict[i] % 12] if self.ClustDict[i] != -1 else colors[-1], s=300)

        # plot the last added point bigger and black, with a red circle surrounding it
        self.ax1.scatter(x=X_dict[point][0], y=X_dict[point][1], s=400, color="black", alpha=0.4)

        circle1 = plt.Circle((X_dict[point][0], X_dict[point][1]), self.eps, color='r',
                             fill=False, linewidth=3, alpha=0.7)
        self.ax1.add_artist(circle1)

        for i, txt in enumerate([i for i in range(len(self.X))]):
            self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center', va='center')

        # self.ax1.set_aspect('equal')
        # self.ax1.legend(fontsize=8)

        self.canvas_up.draw()
        if save_plots is True:
            self.canvas_up.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

    def plot_clust_DB_gui(self, save_plots=False, circle_class=None, noise_circle=False):
        """
        Scatter plot of the data points, colored according to the cluster they belong to; circle_class Plots
        circles around some or all points, with a radius of eps; if Noise_circle is True, circle are also plotted
        around noise points.


        :param circle_class: if True, plots circles around every non-noise point, else plots circles
                             only around points belonging to certain clusters, e.g. circle_class = [1,2] will
                             plot circles around points belonging to clusters 1 and 2.
        :param noise_circle: if True, plots circles around noise points
        :param save_plots: if True, saves the plot.

        """
        # create dictionary of X
        X_dict = dict(zip([str(i) for i in range(len(self.X))], self.X))

        # create new dictionary of X, adding the cluster label
        new_dict = {key: (val1, self.ClustDict[key]) for key, val1 in zip(list(X_dict.keys()), list(X_dict.values()))}

        new_dict = OrderedDict((k, new_dict[k]) for k in list(self.ClustDict.keys()))

        df = pd.DataFrame(dict(x=[i[0][0] for i in list(new_dict.values())],
                               y=[i[0][1] for i in list(new_dict.values())],
                               label=[i[1] for i in list(new_dict.values())]), index=new_dict.keys())

        colors = {-1: 'red', 0: 'lightblue', 1: 'beige', 2: 'yellow', 3: 'grey',
                  4: 'pink', 5: 'navy', 6: 'orange', 7: 'purple', 8: 'salmon', 9: 'olive', 10: 'brown',
                  11: 'tan', 12: 'lime'}

        lista_lab = list(df.label.value_counts().index)

        # plot points colored according to the cluster they belong to
        for lab in lista_lab:
            df_sub = df[df.label == lab]
            self.ax1.scatter(df_sub.x, df_sub.y, color=colors[lab % 12] if lab != -1 else colors[-1],
                            s=300, edgecolor="black", label=lab)

        # plot circles around noise, colored according to the cluster they belong to
        if noise_circle == True:

            df_noise = df[df.label == -1]

            for i in range(len(df_noise)):
                self.ax1.add_artist(plt.Circle((df_noise["x"].iloc[i],
                                               df_noise["y"].iloc[i]), self.eps, color='r', fill=False, linewidth=3,
                                              alpha=0.7))

        # plot circles around points, colored according to the cluster they belong to
        if circle_class is not None:
            # around every points or only around specified clusters
            if circle_class != "true":
                lista_lab = circle_class

            for lab in lista_lab:

                if lab != -1:

                    df_temp = df[df.label == lab]

                    for i in range(len(df_temp)):
                        self.ax1.add_artist(plt.Circle((df_temp["x"].iloc[i], df_temp["y"].iloc[i]),
                                                      self.eps, color=colors[lab], fill=False, linewidth=3, alpha=0.7))

        self.ax1.set_title("DBSCAN Cluster Plot")
        self.ax1.set_xlabel("")
        self.ax1.set_ylabel("")

        for i, txt in enumerate([i for i in range(len(self.X))]):
            self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center', va='center')

        # self.ax1.set_aspect('equal')
        self.ax1.legend()
        self.canvas_up.draw()

        if save_plots is True:
            self.canvas_up.figure.savefig('./Images/{}_{:02}/fig_fin_{:02}.png'.format(self.name, self.ind_run,
                                                                                        self.ind_extr_fig))

        QCoreApplication.processEvents()

    def DBSCAN_gui(self, plotting=True, print_details=True, delay=0):
        """
        DBSCAN algorithm.

        :param plotting: if True, executes point_plot_mod, plotting every time a points is
                         added to a clusters
        :param print_details: if True, prints the length of the "external" NearestNeighborhood
                              and of the "internal" one (in the while loop).
        :param delay: seconds for which to delay the algorithm, so that the images displayes in the GUI
                      show at a slower pace.
        :return ClustDict: dictionary of the form point_index:cluster_label.

        """
        self.update_log(initial=True)
        index_for_saving_plots = 0

        # initialize dictionary of clusters
        self.ClustDict = {}

        clust_id = -1

        X_dict = dict(zip([str(i) for i in range(len(self.X))], self.X))

        processed = []

        processed_list = []

        # for every point in the dataset
        for point in X_dict:

            # if it hasnt been visited
            if point not in processed:
                # mark it as visited
                processed.append(point)
                # scan its neighborhood
                N = scan_neigh1_mod(X_dict, X_dict[point], self.eps)

                if print_details == True:
                    self.update_log(point, "  initial len(N): " + str(len(N)), change_current=True)
                    # print("len(N): ", len(N))
                # if there are less than minPTS in its neighborhood, classify it as noise
                if len(N) < self.mp:

                    self.ClustDict.update({point: -1})

                    if plotting == True:
                        if delay != 0:
                            pause_execution(delay)

                        self.point_plot_mod_gui(X_dict, point, save_plots=self.save_plots,
                                                ind_fig=index_for_saving_plots)
                        index_for_saving_plots += 1
                        self.update_log(noise=True)
                # else if it is a Core point
                else:
                    # increase current id of cluster
                    clust_id += 1
                    # put it in the cluster dictionary
                    self.ClustDict.update({point: clust_id})

                    if plotting == True:
                        if delay != 0:
                            pause_execution(delay)

                        self.point_plot_mod_gui(X_dict, point, save_plots=self.save_plots,
                                                ind_fig=index_for_saving_plots)
                        index_for_saving_plots += 1
                    # add it to the temporary processed list
                    processed_list = [point]
                    # remove it from the neighborhood N
                    del N[point]
                    # until the neighborhood is empty
                    while len(N) > 0:

                        # take a random point in neighborhood
                        n = random.choice(list(N.keys()))

                        if print_details == True:
                            self.update_log(n, "     updated len(N): " + str(len(N)), change_subcurrent=True)
                            # print("len(N) in while loop: ", len(N))

                        # but the point must not be in processed_list aka already visited
                        while n in processed_list:
                            n = random.choice(list(N.keys()))
                        # put it in processed_list
                        processed_list.append(n)
                        # remove it from the neighborhood
                        del N[n]
                        # if it hasnt been visited
                        if n not in processed:
                            # mark it as visited
                            processed.append(n)
                            # scan its neighborhood
                            N_2 = scan_neigh1_mod(X_dict, X_dict[n], self.eps)

                            if print_details == True:
                                self.update_log(point, "     len(N_sub): " + str(len(N_2)))
                                # print("len(N2): ", len(N_2))
                            # if it is a core point
                            if len(N_2) >= self.mp:
                                # add each element of its neighborhood to the neighborhood N
                                for element in N_2:

                                    if element not in processed_list:
                                        N.update({element: X_dict[element]})

                        # if n has not been inserted into cluster dictionary or if it has previously been
                        # classified as noise, update the cluster dictionary
                        if (n not in self.ClustDict) or (self.ClustDict[n] == -1):
                            self.ClustDict.update({n: clust_id})

                        if plotting == True:
                            if delay != 0:
                                pause_execution(delay)
                            self.point_plot_mod_gui(X_dict, n, save_plots=self.save_plots,
                                                    ind_fig=index_for_saving_plots)
                            index_for_saving_plots += 1
