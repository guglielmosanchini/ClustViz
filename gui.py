from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QApplication, QComboBox, QGridLayout, QGroupBox, \
    QLineEdit, QPlainTextEdit
from PyQt5.QtCore import QCoreApplication, QRect
from PyQt5.QtGui import QDoubleValidator, QIntValidator
import numpy as np
import pandas as pd
from collections import OrderedDict
import sys
# import qdarkstyle
import random

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from algorithms.optics import scan_neigh1, reach_dist, minPTSdist, ExtractDBSCANclust

from sklearn.datasets.samples_generator import make_blobs, make_moons, make_circles

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QT5Agg')

# TODO: button images
# TODO: update tooltips
# TODO: adjust button positions
# TODO: play/pause button
# TODO: comment everything


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        ''' ========== WINDOW ====================================================================================== '''
        self.setGeometry(100, 100, 1290, 850)

        self.status = "running"

        # upper plot
        self.canvas_up = FigureCanvas(Figure(figsize=(12, 5)))
        self.ax1 = self.canvas_up.figure.subplots()
        self.ax1_t = self.ax1.twinx()
        self.ax1_t.set_xticks([], [])
        self.ax1_t.set_yticks([], [])
        self.ax1.set_xticks([], [])
        self.ax1.set_yticks([], [])
        self.ax1.set_title("OPTICS procedure")

        # lower plot
        self.canvas_down = FigureCanvas(Figure(figsize=(12, 5)))
        self.ax = self.canvas_down.figure.subplots()
        self.ax_t = self.ax.twinx()
        self.ax_t.set_xticks([], [])
        self.ax_t.set_yticks([], [])
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])
        self.ax.set_title("OPTICS Reachability Plot")
        self.ax.set_ylabel("reachability distance")

        # box containing everything
        self.groupbox = QGroupBox(self)
        self.groupbox.setGeometry(QRect(30, 10, 1200, 720))

        # parameters initialization
        self.eps = 2
        self.mp = 3
        self.eps_extr = 1
        self.n_points = 50
        self.X = make_blobs(n_samples=self.n_points, centers=4, n_features=2, cluster_std=1.8, random_state=42)[0]
        self.ClustDist = {}
        self.CoreDist = {}
        self.param_check = True

        # grid where the two pictures, the log and the button box are inserted (row,column)
        gridlayout = QGridLayout(self.groupbox)
        gridlayout.addWidget(self.canvas_up, 1, 1)
        gridlayout.addWidget(self.canvas_down, 2, 1)

        # upper part

        self.label_alg = QLabel(self)
        self.label_alg.setText("Choose a clustering algorithm: ")
        self.label_alg.setToolTip("hello mona")
        gridlayout.addWidget(self.label_alg, 0, 0)

        # buttons algorithms
        self.groupbox_alg = QGroupBox("Algorithms")
        gridlayout.addWidget(self.groupbox_alg, 0, 1)

        gridlayout_alg = QGridLayout(self.groupbox_alg)

        self.button_alg1 = QPushButton("OPTICS", self)
        self.button_alg1.clicked.connect(lambda: None)
        self.button_alg2 = QPushButton("DBSCAN", self)
        self.button_alg2.clicked.connect(lambda: None)
        self.button_alg3 = QPushButton("AGGLOMERATIVE", self)
        self.button_alg3.clicked.connect(lambda: None)
        self.button_alg4 = QPushButton("DENCLUE", self)
        self.button_alg4.clicked.connect(lambda: None)
        self.button_alg5 = QPushButton("CURE", self)
        self.button_alg5.clicked.connect(lambda: None)
        self.button_alg6 = QPushButton("PAM", self)
        self.button_alg6.clicked.connect(lambda: None)
        self.button_alg7 = QPushButton("CLARA", self)
        self.button_alg7.clicked.connect(lambda: None)
        self.button_alg8 = QPushButton("CLARANS", self)
        self.button_alg8.clicked.connect(lambda: None)
        self.button_alg9 = QPushButton("BIRCH", self)
        self.button_alg9.clicked.connect(lambda: None)
        self.button_alg10 = QPushButton("CHAMELEON", self)
        self.button_alg10.clicked.connect(lambda: None)

        gridlayout_alg.addWidget(self.button_alg1, 0, 0)
        gridlayout_alg.addWidget(self.button_alg2, 0, 1)
        gridlayout_alg.addWidget(self.button_alg3, 0, 2)
        gridlayout_alg.addWidget(self.button_alg4, 0, 3)
        gridlayout_alg.addWidget(self.button_alg5, 0, 4)
        gridlayout_alg.addWidget(self.button_alg6, 1, 0)
        gridlayout_alg.addWidget(self.button_alg7, 1, 1)
        gridlayout_alg.addWidget(self.button_alg8, 1, 2)
        gridlayout_alg.addWidget(self.button_alg9, 1, 3)
        gridlayout_alg.addWidget(self.button_alg10, 1, 4)

        # START BUTTON
        self.button_run = QPushButton("START", self)
        self.button_run.clicked.connect(lambda: self.start_OPTICS())

        # EXTRACT BUTTON
        self.button_extract = QPushButton("EXTRACT", self)
        self.button_extract.clicked.connect(lambda: self.start_EXTRACT_OPTICS())
        self.button_extract.setEnabled(False)

        # n_points LABEL
        label_np = QLabel(self)
        label_np.setText("n_points:")
        label_np.setToolTip("ciao huhuh")

        self.line_edit_np = QLineEdit(self)
        self.line_edit_np.resize(30, 40)
        self.line_edit_np.setText(str(self.n_points))

        self.n_points_validator = QIntValidator(5, 200, self)
        self.line_edit_np.setValidator(self.n_points_validator)

        # eps LABEL
        label_eps = QLabel(self)
        label_eps.setText("eps (\u03B5):")
        label_eps.setToolTip("ciao prova")

        self.line_edit_eps = QLineEdit(self)
        self.line_edit_eps.resize(30, 40)
        self.line_edit_eps.setText(str(self.eps))

        self.eps_validator = QDoubleValidator(0, 1000, 4, self)
        self.line_edit_eps.setValidator(self.eps_validator)

        # minPTS LABEL
        label_mp = QLabel(self)
        label_mp.setText("minPTS:")
        label_mp.setToolTip("ciao genny")

        self.line_edit_mp = QLineEdit(self)
        self.line_edit_mp.setGeometry(360, 10, 30, 40)
        self.line_edit_mp.setText(str(self.mp))

        self.mp_validator = QIntValidator(1, 200, self)
        self.line_edit_mp.setValidator(self.mp_validator)

        # eps_extr LABEL
        label_eps_extr = QLabel(self)
        label_eps_extr.setText("eps_extr (\u03B5\'):")
        label_eps_extr.setToolTip("ciao bufu")

        self.line_edit_eps_extr = QLineEdit(self)
        self.line_edit_eps_extr.setGeometry(670, 10, 50, 40)
        self.line_edit_eps_extr.setText(str(self.eps_extr))

        self.eps_extr_validator = QDoubleValidator(0, 1000, 4, self)
        self.line_edit_eps_extr.setValidator(self.eps_extr_validator)

        # dataset LABEL
        label_ds = QLabel(self)
        label_ds.setText("dataset:")
        label_mp.setToolTip("ciao gino")

        # COMBOBOX of datasets
        self.combobox = QComboBox(self)
        self.combobox.setGeometry(750, 10, 120, 30)
        self.combobox.addItem("blobs")
        self.combobox.addItem("moons")
        self.combobox.addItem("scatter")
        self.combobox.addItem("circle")

        # LOG
        self.log = QPlainTextEdit("SEED QUEUE")
        self.log.setGeometry(900, 60, 350, 400)
        self.log.setStyleSheet(
            """QPlainTextEdit {background-color: #FFF;
                               color: #000000;
                               font-family: Courier;}""")

        gridlayout.addWidget(self.log, 2, 0)

        # buttons GROUPBOX
        self.groupbox_buttons = QGroupBox("Parameters")
        self.groupbox_buttons.setGeometry(15,30, 450,200)

        gridlayout.addWidget(self.groupbox_buttons, 1, 0)


        gridlayout_but = QGridLayout(self.groupbox_buttons)
        gridlayout_but.addWidget(label_ds, 0, 0)
        gridlayout_but.addWidget(self.combobox, 0, 1)

        gridlayout_but.addWidget(label_np, 1, 0)
        gridlayout_but.addWidget(self.line_edit_np, 1, 1)
        gridlayout_but.addWidget(label_eps, 2, 0)
        gridlayout_but.addWidget(self.line_edit_eps, 2, 1)
        gridlayout_but.addWidget(label_mp, 3, 0)
        gridlayout_but.addWidget(self.line_edit_mp, 3, 1)
        gridlayout_but.addWidget(label_eps_extr, 4, 0)
        gridlayout_but.addWidget(self.line_edit_eps_extr, 4, 1)

        gridlayout_but.addWidget(self.button_run, 5, 0)
        gridlayout_but.addWidget(self.button_extract, 6, 0)


        self.statusBar().showMessage('Message in statusbar.')

        self.show()

    def show_error_message(self, check, msg):
        if check[0] != 2:

            if self.param_check is True:
                self.log.clear()

            self.param_check = False
            self.log.appendPlainText("ERROR")
            self.log.appendPlainText("")
            self.log.appendPlainText(msg)
            self.log.appendPlainText("")



    def verify_input_parameters(self, extract=False):

        self.param_check = True

        check_eps_extr = self.eps_extr_validator.validate(self.line_edit_eps_extr.text(), 0)
        self.show_error_message(check_eps_extr,
                                "The parameter eps_extr must lie between {0} and {1}, and can have a maximum of {2} decimal places.".format(
                                    0, 1000, 4))
        if extract is False:

            check_n_points = self.n_points_validator.validate(self.line_edit_np.text(), 0)
            check_eps = self.eps_validator.validate(self.line_edit_eps.text(), 0)
            check_mp = self.mp_validator.validate(self.line_edit_mp.text(), 0)


            self.show_error_message(check_n_points, "The parameter n_points must be an integer and lie between {0} and {1}.".format(5, 200))
            self.show_error_message(check_eps, "The parameter eps must lie between {0} and {1}, and can have a maximum of {2} decimal places.".format(0, 1000, 4))
            self.show_error_message(check_mp, "The parameter minPTS must be an integer and lie between {0} and {1}.".format(1, 200))



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

        chosen_dataset = self.combobox.currentText()

        if chosen_dataset == "blobs":
            self.X= make_blobs(n_samples=self.n_points, centers=4, n_features=2, cluster_std=1.5, random_state=42)[0]
        elif chosen_dataset == "moons":
            self.X= make_moons(n_samples=self.n_points, noise=0.05, random_state=42)[0]
        elif chosen_dataset == "scatter":
            self.X = make_blobs(n_samples=self.n_points, cluster_std=[2, 2, 2], random_state=42)[0]
        elif chosen_dataset == "circle":
            self.X = make_circles(n_samples=self.n_points, noise=0, random_state=42)[0]

        self.button_extract.setEnabled(False)
        self.button_run.setEnabled(False)
        self.OPTICS_gui(plot=True, plot_reach=True)
        self.button_extract.setEnabled(True)
        self.button_run.setEnabled(True)

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
        self.plot_clust_gui()
        self.clear_seed_log(final=True)


    def status_change(self):
        if self.status == "running":
            self.status = "pause"
        else:
            self.status = "running"

    def point_plot_gui(self, X_dict, coords, neigh, processed=None, col='yellow'):
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
        """

        # fig, ax = plt.subplots(figsize=(14, 6))
        self.ax1.cla()
        self.ax1.set_title("OPTICS procedure")

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
            self.ax1.scatter(neigh_array[:,0], neigh_array[:,1], s=300, color="red", label="neighbors")

        # plot last added point in black and surround it with a red circle
        self.ax1.scatter(x=coords[0], y=coords[1], s=400, color="black", alpha=0.4)

        circle1 = plt.Circle((coords[0], coords[1]), self.eps, color='r', fill=False, linewidth=3, alpha=0.7)
        self.ax1.add_artist(circle1)

        for i, txt in enumerate([i for i in range(len(self.X))]):
            self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center', va='center')

        # self.ax1.set_aspect('equal')
        self.ax1.legend(fontsize=8)
        self.canvas_up.draw()
        QCoreApplication.processEvents()

    def reach_plot_gui(self, data):
        """
        Plots the reachability plot, along with a horizontal line denoting eps,
        from the ClustDist produced by OPTICS.

        :param data: input dictionary.
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

        # self.ax.set_xticks(tick_list)
        self.ax.set_xticklabels(tick_list, rotation=90, fontsize=8)

        # plot horizontal line for eps
        self.ax.axhline(self.eps, color="red", linewidth=3)

        self.ax_t.set_ylim(self.ax.get_ylim())
        self.ax_t.set_yticks([self.eps])
        self.ax_t.set_yticklabels(["\u03B5"])

        self.canvas_down.draw()
        QCoreApplication.processEvents()

    def plot_clust_gui(self):
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




    def OPTICS_gui(self, plot=True, plot_reach=False):
        """
        Executes the OPTICS algorithm. Similar to DBSCAN, but uses a priority queue.

        :param plot: if True, the scatter plot of the function point_plot is displayed at each step.
        :param plot_reach: if True, the reachability plot is displayed at each step.
        :return (ClustDist, CoreDist): ClustDist, a dictionary of the form point_index:reach_dist, and
                 CoreDist, a dictionary of the form point_index:core_dist
        """

        self.ClustDist = {}
        self.CoreDist = {}
        Seed = {}
        processed = []

        # create dictionary
        X_dict = dict(zip([str(i) for i in range(len(self.X))], self.X))

        # until all points have been processed
        while len(processed) != len(self.X):

            if self.status == "running":

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

                if plot == True:

                    self.point_plot_gui(X_dict, X_dict[o], N, processed)

                    if plot_reach == True:
                        self.reach_plot_gui(X_dict)

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


if __name__ == '__main__':
    # create the application and the main window
    app = QApplication(sys.argv)
    # pg.setConfigOption('background', 'w')
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = Window()
    win.update()
    # setup stylesheet
    # run
    sys.exit(app.exec_())
