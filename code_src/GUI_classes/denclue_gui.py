from PyQt5.QtCore import QCoreApplication, Qt

import numpy as np
from matplotlib.patches import Rectangle, Circle

from GUI_classes.utils_gui import choose_dataset

from GUI_classes.generic_gui import StartingGui, FinalStepWindow

from algorithms.denclue.denclue import pop_cubes, check_border_points_rectangles, highly_pop_cubes, connect_cubes, \
    density_attractor, assign_cluster, extract_cluster_labels, center_of_mass, gauss_dens


class DENCLUE_class(StartingGui):
    def __init__(self):
        super(DENCLUE_class, self).__init__(name="DENCLUE", twinx=False, first_plot=False, second_plot=False,
                                            function=self.start_DENCLUE, extract=False, stretch_plot=False)
        self.label_slider.hide()

        self.first_run_occurred_mod = False

        self.dict_checkbox_names = {0: "highly_pop", 1: "contour", 2: "3dplot", 3: "clusters"}

        self.plot_list = [False, False, False, False]

        # self.checkbox_pop_cubes.stateChanged.connect(lambda: self.number_of_plots_gui(0))
        self.checkbox_highly_pop_cubes.stateChanged.connect(lambda: self.number_of_plots_gui(0))
        self.checkbox_contour.stateChanged.connect(lambda: self.number_of_plots_gui(1))
        self.checkbox_3dplot.stateChanged.connect(lambda: self.number_of_plots_gui(2))
        self.checkbox_clusters.stateChanged.connect(lambda: self.number_of_plots_gui(3))

        # influence plot
        self.button_infl_denclue.clicked.connect(lambda: self.plot_infl_gui(ax=self.ax_infl,
                                                                            canvas=self.canvas_infl))

    def number_of_plots_gui(self, number):
        # if number == 0:
        #     if self.checkbox_pop_cubes.isChecked():
        #         self.plot_list[number] = True
        #     else:
        #         self.plot_list[number] = False
        if number == 0:
            if self.checkbox_highly_pop_cubes.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False
        if number == 1:
            if self.checkbox_contour.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False
        if number == 2:
            if self.checkbox_3dplot.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False
        if number == 3:
            if self.checkbox_clusters.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False

    def start_DENCLUE(self):

        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))
        self.log.appendPlainText("")
        QCoreApplication.processEvents()

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.sigma_denclue = float(self.line_edit_sigma_denclue.text())
        self.xi_denclue = float(self.line_edit_xi_denclue.text())
        self.xi_c_denclue = float(self.line_edit_xi_c_denclue.text())
        self.tol_denclue = float(self.line_edit_tol_denclue.text())
        self.prec_denclue = int(self.line_edit_prec_denclue.text())
        self.n_points = int(self.line_edit_np.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        self.SetWindowsDENCLUE(pic_list=self.plot_list,
                               first_run_boolean=self.first_run_occurred_mod)

        self.button_run.setEnabled(False)
        self.checkbox_saveimg.setEnabled(False)
        self.button_delete_pics.setEnabled(False)

        QCoreApplication.processEvents()

        if self.first_run_occurred is True:
            self.ind_run += 1
            self.ind_extr_fig = 0
            if self.save_plots is True:
                self.checkBoxChangedAction(self.checkbox_saveimg.checkState())
        else:
            if Qt.Checked == self.checkbox_saveimg.checkState():
                self.first_run_occurred = True
                self.checkBoxChangedAction(self.checkbox_saveimg.checkState())

        if np.array(self.plot_list).sum() != 0:
            self.DENCLUE_gui(data=self.X, s=self.sigma_denclue, xi=self.xi_denclue, xi_c=self.xi_c_denclue,
                             tol=self.tol_denclue, prec=self.prec_denclue, save_plots=self.save_plots)
        else:
            self.display_empty_message()

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        self.button_delete_pics.setEnabled(True)

        self.first_run_occurred_mod = True

    def display_empty_message(self):

        self.log.appendPlainText("You did not select anything to plot")
        QCoreApplication.processEvents()

    def DENCLUE_gui(self, data, s, xi, xi_c, tol, prec, save_plots, dist="euclidean"):

        clust_dict = {}
        processed = []

        z, d = pop_cubes(data=data, s=s)
        self.log.appendPlainText("Number of populated cubes: {}".format(len(z)))
        check_border_points_rectangles(data, z)
        hpc = highly_pop_cubes(z, xi_c=xi_c)
        self.log.appendPlainText("Number of highly populated cubes: {}".format(len(hpc)))
        self.log.appendPlainText("")
        new_cubes = connect_cubes(hpc, z, s=s)

        if self.plot_list[0] == True:
            self.plot_grid_rect_gui(data, s=s, cube_kind="highly_populated", ax=self.axes_list[0],
                                    canvas=self.canvas_list[0], save_plots=save_plots, ind_fig=0)
        if self.plot_list[1] == True:
            self.plot_3d_or_contour_gui(data, s=s, three=False, scatter=True, prec=prec, ax=self.axes_list[1],
                                        canvas=self.canvas_list[1], save_plots=save_plots, ind_fig=1)
        if self.plot_list[2] == True:
            self.plot_3d_both_gui(data, s=s, xi=xi, prec=prec, ax=self.axes_list[2], canvas=self.canvas_list[2],
                                  save_plots=save_plots, ind_fig=2)

        if self.plot_list[3] == False:
            return

        if len(new_cubes) != 0:
            points_to_process = [item for sublist in np.array(list(new_cubes.values()))[:, 2] for item in sublist]
        else:
            points_to_process = []

        initial_noise = []
        for elem in data:
            if len((np.nonzero(points_to_process == elem))[0]) == 0:
                initial_noise.append(elem)

        for num, point in enumerate(points_to_process):

            if num == int(len(points_to_process) / 4):
                self.log.appendPlainText("hill-climb progress: 25%")
                QCoreApplication.processEvents()
            if num == int(len(points_to_process) / 2):
                self.log.appendPlainText("hill-climb progress: 50%")
                QCoreApplication.processEvents()
            if num == int((len(points_to_process) / 4) * 3):
                self.log.appendPlainText("hill-climb progress: 75%")
                QCoreApplication.processEvents()

            delta = 0.02
            r, o = None, None

            while r is None:
                r, o = density_attractor(data=data, x=point, coord_dict=d, tot_cubes=new_cubes,
                                         s=s, xi=xi, delta=delta, max_iter=600, dist=dist)
                delta = delta * 2

            clust_dict, proc = assign_cluster(data=data, others=o, attractor=r,
                                              clust_dict=clust_dict, processed=processed)

        self.log.appendPlainText("hill-climb progress: 100%")
        QCoreApplication.processEvents()

        for point in initial_noise:
            point_index = np.nonzero(data == point)[0][0]
            clust_dict[point_index] = [-1]

        try:
            lab, coord_df = extract_cluster_labels(data, clust_dict, tol)
        except:
            self.log.appendPlainText("")
            self.log.appendPlainText("There was an error when extracting clusters. Increase number "
                                     "of points or try with a less "
                                     "pathological case: see the other plots to have an idea of why it failed.")
            return

        if self.plot_list[3] == True:
            self.plot_clust_dict_gui(data, coord_df, ax=self.axes_list[3], canvas=self.canvas_list[3],
                                     save_plots=save_plots, ind_fig=3)

        return lab

    def plot_grid_rect_gui(self, data, s, ax, canvas, save_plots, ind_fig,
                           cube_kind="populated", color_grids=True):

        ax.clear()
        ax.set_title("Highly populated cubes")

        cl, ckc = pop_cubes(data, s)

        cl_copy = cl.copy()

        coms = [center_of_mass(list(cl.values())[i]) for i in range(len(cl))]
        coms_hpc = []

        if cube_kind == "highly_populated":
            cl = highly_pop_cubes(cl, xi_c=3)
            coms_hpc = [center_of_mass(list(cl.values())[i]) for i in range(len(cl))]

        ax.scatter(data[:, 0], data[:, 1], s=100, edgecolor="black")

        rect_min = data.min(axis=0)
        rect_diff = data.max(axis=0) - rect_min

        x0 = rect_min[0] - .05
        y0 = rect_min[1] - .05

        # minimal bounding rectangle
        ax.add_patch(Rectangle((x0, y0), rect_diff[0] + .1, rect_diff[1] + .1, fill=None,
                               color='r', alpha=1, linewidth=3
                               ))

        ax.scatter(np.array(coms)[:, 0], np.array(coms)[:, 1], s=100, marker="X", color="red", edgecolor="black",
                   label="centers of mass")

        if cube_kind == "highly_populated":
            for i in range(len(coms_hpc)):
                ax.add_artist(Circle((np.array(coms_hpc)[i, 0], np.array(coms_hpc)[i, 1]),
                                     4 * s, color="red", fill=False, linewidth=2, alpha=0.6))

        tot_cubes = connect_cubes(cl, cl_copy, s)

        new_clusts = {i: tot_cubes[i] for i in list(tot_cubes.keys()) if i not in list(cl.keys())}

        for key in list(new_clusts.keys()):
            (a, b, c, d) = ckc[key]
            ax.add_patch(Rectangle((a, b), 2 * s, 2 * s, fill=True,
                                   color='yellow', alpha=0.3, linewidth=3
                                   ))

        for key in list(ckc.keys()):

            (a, b, c, d) = ckc[key]

            if color_grids is True:
                if key in list(cl.keys()):
                    color_or_not = True if cl[key][0] > 0 else False
                else:
                    color_or_not = False
            else:
                color_or_not = False

            ax.add_patch(Rectangle((a, b), 2 * s, 2 * s, fill=color_or_not,
                                   color='g', alpha=0.3, linewidth=3
                                   ))
        ax.legend(fontsize=8)
        canvas.draw()

        if save_plots is True:
            canvas.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

    def plot_3d_both_gui(self, data, s, ax, canvas, save_plots, ind_fig, xi=None, prec=3):

        ax.clear()
        from matplotlib import cm

        x_data = [np.array(data)[:, 0].min(), np.array(data)[:, 0].max()]
        y_data = [np.array(data)[:, 1].min(), np.array(data)[:, 1].max()]
        mixed_data = [min(x_data[0], y_data[0]), max(x_data[1], y_data[1])]

        xx = np.outer(np.linspace(mixed_data[0] - 1, mixed_data[1] + 1, prec * 10), np.ones(prec * 10))
        yy = xx.copy().T  # transpose
        z = np.empty((prec * 10, prec * 10))
        z_xi = np.empty((prec * 10, prec * 10))

        for i, a in enumerate(range(prec * 10)):

            if i == int((prec * 10) / 4):
                self.log.appendPlainText("3dplot progress: 25%")
                QCoreApplication.processEvents()
            if i == (prec * 10) / 2:
                self.log.appendPlainText("3dplot progress: 50%")
                QCoreApplication.processEvents()
            if i == int(((prec * 10) / 4) * 3):
                self.log.appendPlainText("3dplot progress: 75%")
                QCoreApplication.processEvents()

            for j, b in enumerate(range(prec * 10)):

                z[i, j] = gauss_dens(x=np.array([xx[i][a], yy[i][b]]), D=data, s=s)
                if xi is not None:
                    if z[i, j] >= xi:
                        z_xi[i, j] = z[i, j]
                    else:
                        z_xi[i, j] = xi

        # to set colors according to xi value, red if greater, yellow if smaller
        if xi is not None:
            xi_data = []
            for a, b in zip(np.array(data)[:, 0], np.array(data)[:, 1]):
                to_be_eval = gauss_dens(x=np.array([a, b]), D=data, s=s)
                if to_be_eval >= xi:
                    xi_data.append("red")
                else:
                    xi_data.append("yellow")

        offset = -15

        if xi is not None:
            plane = ax.plot_surface(xx, yy, z_xi, cmap=cm.ocean, alpha=0.9)

        surf = ax.plot_surface(xx, yy, z, alpha=0.8, cmap=cm.ocean)

        cset = ax.contourf(xx, yy, z, zdir='z', offset=offset, cmap=cm.ocean)

        if xi is not None:
            color_plot = xi_data
        else:
            color_plot = "red"

        ax.scatter(np.array(data)[:, 0], np.array(data)[:, 1], offset, s=30,
                   edgecolor="black", color=color_plot, alpha=0.6)

        canvas.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim(offset, np.max(z))
        ax.set_title('3D surface with 2D contour plot projections')

        cond_1 = np.sum(self.plot_list) == 4
        cond_2 = (np.sum(self.plot_list) == 3) and (self.plot_list[2] == True)

        if cond_1 or cond_2:
            self.openFinalStepWindow_2(canvas=canvas)

        self.log.appendPlainText("3dplot progress: 100%")
        self.log.appendPlainText("")
        canvas.draw()

        if save_plots is True:
            canvas.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

    def plot_3d_or_contour_gui(self, data, s, ax, canvas, save_plots, ind_fig, three=False, scatter=False, prec=3):

        ax.clear()
        ax.set_title("Scatterplot with Countour plot")

        x_data = [np.array(data)[:, 0].min(), np.array(data)[:, 0].max()]
        y_data = [np.array(data)[:, 1].min(), np.array(data)[:, 1].max()]
        mixed_data = [min(x_data[0], y_data[0]), max(x_data[1], y_data[1])]

        xx = np.outer(np.linspace(mixed_data[0] - 1, mixed_data[1] + 1, prec * 10), np.ones(prec * 10))
        yy = xx.copy().T  # transpose
        z = np.empty((prec * 10, prec * 10))
        for i, a in enumerate(range(prec * 10)):

            if i == int((prec * 10) / 4):
                self.log.appendPlainText("contour progress: 25%")
                QCoreApplication.processEvents()
            if i == (prec * 10) / 2:
                self.log.appendPlainText("contour progress: 50%")
                QCoreApplication.processEvents()
            if i == int(((prec * 10) / 4) * 3):
                self.log.appendPlainText("contour progress: 75%")
                QCoreApplication.processEvents()

            for j, b in enumerate(range(prec * 10)):
                z[i, j] = gauss_dens(x=np.array([xx[i][a], yy[i][b]]), D=data, s=s)

        if three is True:
            pass
            # ax = plt.axes(projection="3d")
            # ax.plot_surface(xx, yy, z, cmap='winter', edgecolor='none')
            # plt.show()
        else:
            CS = ax.contour(xx, yy, z, cmap='winter')
            ax.clabel(CS, inline=1, fontsize=10)

            if (scatter is True) and (three is False):
                ax.scatter(np.array(data)[:, 0], np.array(data)[:, 1], s=300, edgecolor="black", color="yellow",
                           alpha=0.6)

            self.log.appendPlainText("contour progress: 100%")
            self.log.appendPlainText("")
            QCoreApplication.processEvents()

            canvas.draw()

            if save_plots is True:
                canvas.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

            QCoreApplication.processEvents()

    def plot_infl_gui(self, ax, canvas):

        ax.clear()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))
        self.log.appendPlainText("")

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_points = int(self.line_edit_np.text())
        self.sigma_denclue = float(self.line_edit_sigma_denclue.text())
        self.xi_denclue = float(self.line_edit_xi_denclue.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        data = self.X
        s = self.sigma_denclue
        xi = self.xi_denclue

        ax.set_title("Significance of possible density attractors")

        z = []
        for a, b in zip(np.array(data)[:, 0], np.array(data)[:, 1]):
            z.append(gauss_dens(x=np.array([a, b]), D=data, s=s))

        x_plot = [i for i in range(len(data))]

        X_over = [x_plot[j] for j in range(len(data)) if z[j] >= xi]
        Y_over = [z[j] for j in range(len(data)) if z[j] >= xi]

        X_under = [x_plot[j] for j in range(len(data)) if z[j] < xi]
        Y_under = [z[j] for j in range(len(data)) if z[j] < xi]

        ax.scatter(X_over, Y_over, s=300, color="green", edgecolor="black",
                   alpha=0.7, label="possibly significant")

        ax.scatter(X_under, Y_under, s=300, color="yellow", edgecolor="black",
                   alpha=0.7, label="not significant")

        ax.axhline(xi, color="red", linewidth=2, label="xi")

        ax.set_ylabel("influence")

        # add indexes to points in plot
        for i, txt in enumerate(range(len(data))):
            ax.annotate(txt, (i, z[i]), fontsize=10, size=10, ha='center', va='center')

        ax.legend()
        self.openFinalStepWindow_4(canvas=canvas)

        canvas.draw()

        QCoreApplication.processEvents()

    def plot_clust_dict_gui(self, data, lab_dict, ax, canvas, save_plots, ind_fig):

        ax.clear()
        ax.set_title("DENCLUE clusters")

        colors = {0: "seagreen", 1: 'lightcoral', 2: 'yellow', 3: 'grey', 4: 'pink', 5: 'turquoise',
                  6: 'orange', 7: 'purple', 8: 'yellowgreen', 9: 'olive', 10: 'brown',
                  11: 'tan', 12: 'plum', 13: 'rosybrown', 14: 'lightblue', 15: "khaki",
                  16: "gainsboro", 17: "peachpuff", 18: "lime", 19: "peru",
                  20: "dodgerblue", 21: "teal", 22: "royalblue", 23: "tomato",
                  24: "bisque", 25: "palegreen"}

        col = [colors[lab_dict.label[i] % len(colors)] if lab_dict.label[i] != -1 else "red" for i in
               range(len(lab_dict))]

        ax.scatter(np.array(data)[:, 0], np.array(data)[:, 1], s=300, edgecolor="black", color=col, alpha=0.8)

        df_dens_attr = lab_dict.groupby("label").mean()

        x_dens_attr = [df_dens_attr.loc[i]["x"] for i in range(df_dens_attr.iloc[-1].name + 1)]
        y_dens_attr = [df_dens_attr.loc[i]["y"] for i in range(df_dens_attr.iloc[-1].name + 1)]

        ax.scatter(x_dens_attr, y_dens_attr, color="red", marker="X", s=300,
                   edgecolor="black", label="density attractors")

        # add indexes to points in plot
        for i, txt in enumerate(range(len(data))):
            ax.annotate(txt, (np.array(data)[i, 0], np.array(data)[i, 1]), fontsize=10, size=10, ha='center',
                        va='center')

        cond_1 = np.sum(self.plot_list) == 4
        cond_2 = (np.sum(self.plot_list) == 3) and (self.plot_list[2] == False)

        if cond_1 or cond_2:
            self.openFinalStepWindow_3(canvas=canvas)

        ax.legend(fontsize=8)
        canvas.draw()

        if save_plots is True:
            canvas.figure.savefig('./Images/{}_{:02}/fig_{:02}.png'.format(self.name, self.ind_run, ind_fig))

        QCoreApplication.processEvents()

    def openFinalStepWindow_2(self, canvas):
        self.w2 = FinalStepWindow(canvas=canvas, window_title="3D PLOT")
        self.w2.show()

    def openFinalStepWindow_3(self, canvas):
        self.w3 = FinalStepWindow(canvas=canvas, window_title="Clustering")
        self.w3.show()

    def openFinalStepWindow_4(self, canvas):
        self.w4 = FinalStepWindow(canvas=canvas, window_title="Influence function")
        self.w4.show()
