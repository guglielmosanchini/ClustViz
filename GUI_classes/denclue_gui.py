from PyQt5.QtCore import QCoreApplication, Qt

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Circle

from GUI_classes.utils_gui import choose_dataset

from GUI_classes.generic_gui import StartingGui

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from algorithms.denclue.denclue import pop_cubes, check_border_points_rectangles, highly_pop_cubes, connect_cubes, \
    density_attractor, assign_cluster, extract_cluster_labels, center_of_mass, gauss_dens


class DENCLUE_class(StartingGui):
    def __init__(self):
        super(DENCLUE_class, self).__init__(name="DENCLUE", twinx=False, first_plot=False, second_plot=False,
                                            function=self.start_DENCLUE, extract=False, stretch_plot=False)
        self.label_slider.hide()

        self.canvas_fin = FigureCanvas(Figure(figsize=(12, 5)))
        self.ax_fin = self.canvas_fin.figure.subplots()
        self.ax_fin.set_xticks([], [])
        self.ax_fin.set_yticks([], [])
        self.ax_fin.set_title("LARGE CURE final step")

        self.first_run_occurred_mod = False

        self.dict_checkbox_names = {0: "pop", 1: "highly_pop", 2: "contour", 3: "3dplot", 4: "clusters"}

        self.plot_list = [False, False, False, False, False]

        self.checkbox_pop_cubes.stateChanged.connect(lambda: self.number_of_plots_gui(0))
        self.checkbox_highly_pop_cubes.stateChanged.connect(lambda: self.number_of_plots_gui(1))
        self.checkbox_contour.stateChanged.connect(lambda: self.number_of_plots_gui(2))
        self.checkbox_3dplot.stateChanged.connect(lambda: self.number_of_plots_gui(3))
        self.checkbox_clusters.stateChanged.connect(lambda: self.number_of_plots_gui(4))

    def number_of_plots_gui(self, number):
        if number == 0:
            if self.checkbox_pop_cubes.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False
        if number == 1:
            if self.checkbox_highly_pop_cubes.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False
        if number == 2:
            if self.checkbox_contour.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False
        if number == 3:
            if self.checkbox_3dplot.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False
        if number == 4:
            if self.checkbox_clusters.isChecked():
                self.plot_list[number] = True
            else:
                self.plot_list[number] = False

    def start_DENCLUE(self):

        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))

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

        self.DENCLUE_gui(data=self.X, s=self.sigma_denclue, xi=self.xi_denclue, xi_c=self.xi_c_denclue,
                         tol=self.tol_denclue, prec=self.prec_denclue)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        # self.button_extract.setEnabled(True)
        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)

        self.first_run_occurred_mod = True

    def DENCLUE_gui(self, data, s, xi=3, xi_c=3, tol=2, dist="euclidean", prec=20, plotting=True, ax, canvas,
                    save_plots, ind_fig, ind_fig_bis):
        clust_dict = {}
        processed = []

        z, d = pop_cubes(data=data, s=s)
        self.log.appendPlainText("Number of populated cubes: {}".format(len(z)))
        check_border_points_rectangles(data, z)
        hpc = highly_pop_cubes(z, xi_c=xi_c)
        self.log.appendPlainText("Number of highly populated cubes: {}".format(len(hpc)))
        new_cubes = connect_cubes(hpc, z, s=s)

        if plotting is True:
            self.plot_grid_rect_gui(data, s=s, cube_kind="populated")
            self.plot_grid_rect_gui(data, s=s, cube_kind="highly_populated")

            self.plot_3d_or_contour_gui(data, s=s, three=False, scatter=True, prec=prec)

            self.plot_3d_both_gui(data, s=s, xi=xi, prec=prec)

        points_to_process = [item for sublist in np.array(list(new_cubes.values()))[:, 2] for item in sublist]

        initial_noise = []
        for elem in data:
            if len((np.nonzero(points_to_process == elem))[0]) == 0:
                initial_noise.append(elem)

        for num, point in enumerate(points_to_process):
            delta = 0.02
            r, o = None, None

            while r is None:
                r, o = density_attractor(data=data, x=point, coord_dict=d, tot_cubes=new_cubes,
                                         s=s, xi=xi, delta=delta, max_iter=600, dist=dist)
                delta = delta * 2

            clust_dict, proc = assign_cluster(data=data, others=o, attractor=r,
                                              clust_dict=clust_dict, processed=processed)

        for point in initial_noise:
            point_index = np.nonzero(data == point)[0][0]
            clust_dict[point_index] = [-1]

        lab, coord_df = extract_cluster_labels(data, clust_dict, tol)

        if plotting is True:
            plot_clust_dict(data, coord_df)

        return lab

    def plot_grid_rect_gui(self, data, s,  ax, canvas, save_plots, ind_fig, ind_fig_bis,
                           cube_kind="populated", color_grids=True):
        cl, ckc = pop_cubes(data, s)

        cl_copy = cl.copy()

        coms = [center_of_mass(list(cl.values())[i]) for i in range(len(cl))]

        if cube_kind == "highly_populated":
            cl = highly_pop_cubes(cl, xi_c=3)
            coms_hpc = [center_of_mass(list(cl.values())[i]) for i in range(len(cl))]

        self.min_bound_rect_gui(data)

        ax.scatter(np.array(coms)[:, 0], np.array(coms)[:, 1], s=100, color="red", edgecolor="black")

        if cube_kind == "highly_populated":
            for i in range(len(coms_hpc)):
                ax.add_artist(Circle((np.array(coms_hpc)[i, 0], np.array(coms_hpc)[i, 1]),
                                                      4 * s, color="red", fill=False, linewidth=2, alpha=0.6))

        tot_cubes = connect_cubes(cl, cl_copy, s)

        new_clusts = {i: tot_cubes[i] for i in list(tot_cubes.keys()) if i not in list(cl.keys())}

        for key in list(new_clusts.keys()):
            (a, b, c, d) = ckc[key]
            ax.add_patch(Rectangle((a, b),2 * s, 2 * s, fill=True,
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

        canvas.draw()

    def min_bound_rect_gui(self, data, ax, canvas):

        ax.scatter(data[:, 0], data[:, 1], s=100, edgecolor="black")

        rect_min = data.min(axis=0)
        rect_diff = data.max(axis=0) - rect_min

        x0 = rect_min[0] - .05
        y0 = rect_min[1] - .05

        # minimal bounding rectangle
        ax.add_patch(Rectangle((x0, y0), rect_diff[0] + .1, rect_diff[1] + .1, fill=None,
                                color='r', alpha=1, linewidth=3
                                ))

    def plot_3d_both_gui(self, data, s, ax, canvas, save_plots, ind_fig, ind_fig_bis, xi=None, prec=3):

        from matplotlib import cm
        ##################################################
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

        x_data = [np.array(data)[:, 0].min(), np.array(data)[:, 0].max()]
        y_data = [np.array(data)[:, 1].min(), np.array(data)[:, 1].max()]
        mixed_data = [min(x_data[0], y_data[0]), max(x_data[1], y_data[1])]

        xx = np.outer(np.linspace(mixed_data[0] - 1, mixed_data[1] + 1, prec * 10), np.ones(prec * 10))
        yy = xx.copy().T  # transpose
        z = np.empty((prec * 10, prec * 10))
        z_xi = np.empty((prec * 10, prec * 10))

        for i, a in enumerate(range(prec * 10)):

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

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        ax.set_xlabel('X')

        ax.set_ylabel('Y')

        ax.set_zlabel('Z')
        ax.set_zlim(offset, np.max(z))
        # ax.set_title('3D surface with 2D contour plot projections')

        canvas.draw()