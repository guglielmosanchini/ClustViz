from PyQt5.QtCore import QCoreApplication, Qt

import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd

from algorithms.agglomerative import dist_mat_gen, compute_ward_ij, update_mat

from GUI_classes.utils_gui import choose_dataset, pause_execution, encircle, convert_colors

from GUI_classes.generic_gui import StartingGui


class AGGLOMERATIVE_class(StartingGui):
    def __init__(self):
        super(AGGLOMERATIVE_class, self).__init__(name="AGGLOMERATIVE", twinx=False, first_plot=True, second_plot=False,
                                                  function=self.start_AGGL, extract=False, stretch_plot=True)

    def start_AGGL(self):

        self.ax1.cla()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_clust = int(self.line_edit_n_clust.text())
        self.n_points = int(self.line_edit_np.text())
        self.linkage = self.combobox_linkage.currentText()

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

        self.agg_clust_mod_gui(delay=self.delay)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)

    def point_plot_mod_gui(self, a, level_txt, level2_txt=None, save_plots=False, ind_fig=None):
        """
        Scatter plot of data points, colored according to the cluster they belong to. The most recently
        merged cluster is enclosed in a rectangle of the same color as its points, with red borders.
        In the top right corner, the total distance is shown, along with the current number of clusters.
        When using Ward linkage, also the increment in distance is shown.

        :param a: distance matrix built by agg_clust/agg_clust_mod.
        :param level_txt: dist_tot displayed.
        :param level2_txt: dist_incr displayed.
        :param save_plots: if True, the produced image is saved.
        :param ind_fig: index of the figure that is saved.
        """
        self.ax1.clear()
        self.ax1.set_title("{} procedure".format(self.name))
        self.ax1.scatter(self.X[:, 0], self.X[:, 1], s=300, color="lime", edgecolor="black")

        a = a.dropna(1, how="all")

        colors = {0: "seagreen", 1: 'beige', 2: 'yellow', 3: 'grey',
                  4: 'pink', 5: 'navy', 6: 'orange', 7: 'purple', 8: 'salmon', 9: 'olive', 10: 'brown',
                  11: 'tan', 12: 'plum', 13: 'red', 14: 'lightblue', 15: "khaki", 16: "gainsboro", 17: "peachpuff"}

        color_dict_rect = convert_colors(colors, alpha=0.3)

        len_ind = [len(i.split("-")) for i in list(a.index)]
        start = np.min([i for i in range(len(len_ind)) if len_ind[i] > 1])

        for ind, i in enumerate(range(start, len(a))):
            point = a.iloc[i].name.replace("(", "").replace(")", "").split("-")
            point = [int(i) for i in point]

            X_clust = [self.X[point[j], 0] for j in range(len(point))]
            Y_clust = [self.X[point[j], 1] for j in range(len(point))]

            self.ax1.scatter(X_clust, Y_clust, s=350, color=colors[ind % 17])

        point = a.iloc[-1].name.replace("(", "").replace(")", "").split("-")
        point = [int(i) for i in point]
        rect_min = self.X[point].min(axis=0)
        rect_diff = self.X[point].max(axis=0) - rect_min

        xwidth = self.ax1.axis()[1] - self.ax1.axis()[0]
        ywidth = self.ax1.axis()[3] - self.ax1.axis()[2]

        if len(X_clust) <= 5:

            self.ax1.add_patch(Rectangle((rect_min[0] - xwidth * 0.02, rect_min[1] - ywidth * 0.04),
                                   rect_diff[0] + xwidth * 0.04, rect_diff[1] + ywidth * 0.08, fill=True,
                                   color=color_dict_rect[ind % 17], linewidth=3,
                                   ec="red"))
        else:
            encircle(X_clust, Y_clust, ax=self.ax1, color=color_dict_rect[ind % 17], linewidth=3, ec="red", zorder=0)

        for i, txt in enumerate([i for i in range(len(self.X))]):
            self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center', va='center')

        self.log.appendPlainText("")
        self.log.appendPlainText("n° clust: " + str(len(a)))
        self.log.appendPlainText("dist_tot: " + str(round(level_txt, 5)))
        if level2_txt is not None:
            self.log.appendPlainText("dist_incr: " + str(round(level2_txt, 5)))

        self.canvas_up.draw()

        if save_plots is True:
            self.canvas_up.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

    def agg_clust_mod_gui(self, delay=0):
        """
        Perform hierarchical agglomerative clustering with the provided linkage method, plotting every step
        of cluster aggregation.

        :param delay: seconds for which to delay the algorithm, so that the images displayes in the GUI
                      show at a slower pace.

        """

        levels = []
        levels2 = []
        ind_list = []
        index_for_saving_plots = 0

        # build matrix a, used to store points of clusters with their coordinates
        l = [[i, i] for i in range(len(self.X))]
        flat_list = [item for sublist in l for item in sublist]
        col = [str(el) + "x" if i % 2 == 0 else str(el) + "y" for i, el in enumerate(flat_list)]

        a = pd.DataFrame(index=[str(i) for i in range(len(self.X))], columns=col)

        a["0x"] = self.X.T[0]
        a["0y"] = self.X.T[1]

        b = a.dropna(axis=1, how="all")

        # initial distance matrix
        X_dist1 = dist_mat_gen(b)
        var_sum = 0
        levels.append(var_sum)
        levels2.append(var_sum)

        # until the desired number of clusters is reached
        while len(a) > self.n_clust:

            if self.linkage == "ward":
                # find indexes corresponding to the minimum increase in total intra-cluster variance
                b = a.dropna(axis=1, how="all")
                b = b.fillna(np.inf)
                ((i, j), var_sum, par_var) = compute_ward_ij(self.X, b)

                levels.append(var_sum)
                levels2.append(par_var)
                ind_list.append((i, j))
                new_clust = a.loc[[i, j], :]

            else:
                # find indexes corresponding to the minimum distance
                (i, j) = np.unravel_index(np.array(X_dist1).argmin(), np.array(X_dist1).shape)
                levels.append(np.min(np.array(X_dist1)))
                ind_list.append((i, j))
                new_clust = a.iloc[[i, j], :]

                # update distance matrix
                X_dist1 = update_mat(X_dist1, i, j, self.linkage)

            a = a.drop([new_clust.iloc[0].name], 0)
            a = a.drop([new_clust.iloc[1].name], 0)

            dim1 = int(new_clust.iloc[0].notna().sum())

            new_cluster_name = "(" + new_clust.iloc[0].name + ")" + "-" + "(" + new_clust.iloc[1].name + ")"

            a.loc[new_cluster_name, :] = new_clust.iloc[0].fillna(0) + new_clust.iloc[1].shift(dim1, fill_value=0)

            if delay != 0:
                pause_execution(self.delay)

            if self.linkage != "ward":
                self.point_plot_mod_gui(a, levels[-1], save_plots=self.save_plots, ind_fig=index_for_saving_plots)
            else:
                self.point_plot_mod_gui(a, levels[-2], levels2[-1], save_plots=self.save_plots,
                                        ind_fig=index_for_saving_plots)
            index_for_saving_plots += 1
