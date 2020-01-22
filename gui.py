from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QApplication, QComboBox, QGridLayout, QGroupBox, \
    QLineEdit, QPlainTextEdit
from PyQt5.QtCore import QTimer, QCoreApplication, QRect
from PyQt5.QtChart import QChart, QChartView, QValueAxis, QBarCategoryAxis, QBarSet, QBarSeries, \
    QHorizontalPercentBarSeries
import pyqtgraph as pg
import numpy as np
import pandas as pd
from collections import OrderedDict
import time
import sys
import qdarkstyle
import random

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from algorithms.optics import scan_neigh1, reach_dist, minPTSdist, ExtractDBSCANclust

from sklearn.datasets.samples_generator import make_blobs, make_moons, make_circles

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QT5Agg')


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        ''' ========== WINDOW ====================================================================================== '''
        self.setGeometry(150, 150, 1290, 800)

        self.status = "running"

        self.canvas1 = FigureCanvas(Figure(figsize=(12, 5)))
        self.ax1 = self.canvas1.figure.subplots()
        self.ax1_t = self.ax1.twinx()
        self.ax1_t.set_xticks([], [])
        self.ax1_t.set_yticks([], [])
        self.ax1.set_xticks([], [])
        self.ax1.set_yticks([], [])
        self.ax1.set_title("OPTICS procedure")
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("y")

        self.canvas = FigureCanvas(Figure(figsize=(12, 5)))
        self.ax = self.canvas.figure.subplots()
        self.ax_t = self.ax.twinx()
        self.ax_t.set_xticks([], [])
        self.ax_t.set_yticks([], [])
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])
        self.ax.set_title("Reachability Plot")
        self.ax.set_xlabel("point")
        self.ax.set_ylabel("reachability distance")

        self.groupbox = QGroupBox(self)
        self.groupbox.setGeometry(QRect(30, 50, 850, 700))

        self.eps = 2
        self.mp = 3
        self.e_extr = 1
        self.X, self.y = make_blobs(n_samples=20, centers=4, n_features=3, cluster_std=1.8, random_state=42)
        self.ClustDist = {}
        self.CoreDist = {}

        gridlayout = QGridLayout(self.groupbox)
        gridlayout.addWidget(self.canvas1, 0, 0)
        gridlayout.addWidget(self.canvas, 1, 0)

        button = QPushButton("START", self)
        button.setGeometry(30, 10, 100, 30)
        button.clicked.connect(lambda: self.aux())

        button2 = QPushButton("PAUSE", self)
        button2.setGeometry(150, 10, 100, 30)
        button2.clicked.connect(lambda: self.status_change())

        button3 = QPushButton("EXTRACT", self)
        button3.setGeometry(260, 10, 100, 30)
        button3.clicked.connect(lambda: self.aux2())

        label_eps = QLabel(self)
        label_eps.setText("eps:")
        label_eps.setGeometry(390, 10, 50, 30)
        label_eps.setToolTip("ciao prova")

        self.line_edit_eps = QLineEdit(self)
        self.line_edit_eps.setGeometry(420, 10, 50, 30)
        self.line_edit_eps.setText(str(self.eps))

        label_mp = QLabel(self)
        label_mp.setText("minPTS:")
        label_mp.setGeometry(500, 10, 70, 30)
        label_mp.setToolTip("ciao genny")

        self.line_edit_mp = QLineEdit(self)
        self.line_edit_mp.setGeometry(590, 10, 50, 30)
        self.line_edit_mp.setText(str(self.mp))

        label_eps_extr = QLabel(self)
        label_eps_extr.setText("eps extr:")
        label_eps_extr.setGeometry(620, 10, 50, 30)
        label_eps_extr.setToolTip("ciao bufu")

        self.line_edit_eps_extr = QLineEdit(self)
        self.line_edit_eps_extr.setGeometry(670, 10, 50, 30)
        self.line_edit_eps_extr.setText(str(self.e_extr))

        self.combobox = QComboBox(self)
        self.combobox.setGeometry(750, 10, 100, 30)
        self.combobox.addItem("blobs")
        self.combobox.addItem("moons")
        self.combobox.addItem("scatter")

        self.log = QPlainTextEdit(self)
        self.log.setGeometry(900, 60, 350, 400)
        self.log.appendPlainText("current point: ")
        self.log.setStyleSheet(
            """QPlainTextEdit {background-color: #FFF;
                               color: #000000;
                               text-decoration: underline;
                               font-family: Courier;}""")

        self.statusBar().showMessage('Message in statusbar.')

        self.show()

    def aux(self):

        self.ax.cla()
        self.ax1.cla()
        self.ax_t.cla()
        self.ax1_t.cla()
        self.ax1_t.set_yticks([], [])

        self.eps = float(self.line_edit_eps.text())
        self.mp = int(self.line_edit_mp.text())
        self.e_extr = float(self.line_edit_eps_extr.text())

        chosen_dataset = self.combobox.currentText()

        if chosen_dataset == "blobs":
            self.X, self.y = make_blobs(n_samples=40, centers=4, n_features=3, cluster_std=1.8, random_state=42)
        elif chosen_dataset == "moons":
            self.X, self.y = make_moons(n_samples=80, noise=0.05, random_state=42)
        elif chosen_dataset == "scatter":
            self.X = make_blobs(n_samples=120, cluster_std=[3.5, 3.5, 3.5], random_state=42)[0]

        self.OPTICS_gui(plot=True, plot_reach=True)

    def aux2(self):

        self.ax.cla()
        self.ax1.cla()
        self.ax_t.cla()
        self.ax1_t.cla()
        self.ax1_t.set_yticks([], [])

        self.e_extr = float(self.line_edit_eps_extr.text())
        self.plot_clust_gui()

    def status_change(self):
        if self.status == "running":
            self.status = "pause"
        else:
            self.status = "running"

    def point_plot_gui(self, X_dict, x, y, processed=None, col='yellow'):
        """
        Plots a scatter plot of points, where the point (x,y) is light black and
        surrounded by a red circle of radius eps, where processed point are plotted
        in col (yellow by default) and without edgecolor, whereas still-to-process points are green
        with black edgecolor.

        :param X_dict: input dictionary version of X.
        :param x: x-coordinate of the point that is currently inspected.
        :param y: y-coordinate of the point that is currently inspected.
        :param processed: already processed points, to plot in col
        :param col: color to use for processed points, yellow by default.
        """

        # fig, ax = plt.subplots(figsize=(14, 6))
        self.ax1.cla()
        self.ax1.set_title("OPTICS procedure")

        # plot every point in color lime
        self.ax1.scatter(self.X[:, 0], self.X[:, 1], s=300, color="lime", edgecolor="black")

        # plot clustered points according to appropriate colors
        if processed is not None:
            for i in processed:
                self.ax1.scatter(X_dict[i][0], X_dict[i][1], s=300, color=col)

        # plot last added point in black and surround it with a red circle
        self.ax1.scatter(x=x, y=y, s=400, color="black", alpha=0.4)

        circle1 = plt.Circle((x, y), self.eps, color='r', fill=False, linewidth=3, alpha=0.7)
        self.ax1.add_artist(circle1)

        for i, txt in enumerate([i for i in range(len(self.X))]):
            self.ax1.annotate(txt, (self.X[:, 0][i], self.X[:, 1][i]), fontsize=10, size=10, ha='center', va='center')

        # self.ax1.set_aspect('equal')
        self.canvas1.draw()

    def Reach_plot_gui(self, data):
        """
        Plots the reachability plot, along with a horizontal line denoting eps,
        from the ClustDist produced by OPTICS

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

        self.canvas.draw()

    def plot_clust_gui(self):
        """
        Plot a scatter plot on the left, where points are colored according to the cluster they belong to,
        and a reachability plot on the right, where colors correspond to the clusters, and the two horizontal
        lines represent eps and eps_db
        """

        self.ax1.set_title("OPTICS procedure")

        self.ax.set_title("Reachability Plot")
        self.ax.set_ylabel("reachability distance")

        X_dict = dict(zip([str(i) for i in range(len(self.X))], self.X))

        # extract the cluster dictionary using DBSCAN
        cl = ExtractDBSCANclust(self.ClustDist, self.CoreDist, self.e_extr)

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

        self.ax.bar(plot_dic.keys(), plot_dic.values(),
                    color=[colors[i % 13] if i != -1 else "red" for i in df.label])

        self.ax.axhline(self.eps, color="black", linewidth=3)

        self.ax.axhline(self.e_extr, color="black", linewidth=3)


        self.ax_t.set_ylim(self.ax.get_ylim())
        self.ax_t.set_yticks([self.eps, self.e_extr])
        self.ax_t.set_yticklabels(["\u03B5", "\u03B5" + "\'"])

        self.canvas1.draw()
        self.canvas.draw()
        QCoreApplication.processEvents()

    def clear_seed_log(self, Seed, point):
        self.log.clear()
        self.log.appendPlainText("current point: " + str(point))
        self.log.appendPlainText("")

        if len(Seed) !=0:
            rounded_values = [round(i, 3) for i in list(Seed.values())]
            rounded_dict = {k: v for k, v in zip(Seed.keys(), rounded_values)}
            self.log.appendPlainText("neighbors: ")
            self.log.appendPlainText("")
            for k, v in rounded_dict.items():
                self.log.appendPlainText(str(k) + ": " + str(v))
        else:
            self.log.appendPlainText("no neighbors")



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

                    self.point_plot_gui(X_dict, X_dict[o][0], X_dict[o][1], processed)
                    QCoreApplication.processEvents()

                    if plot_reach == True:
                        self.Reach_plot_gui(X_dict)
                        QCoreApplication.processEvents()

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

        self.ax.cla()
        self.ax1.cla()
        self.ax_t.cla()
        self.ax1_t.cla()
        self.ax1_t.set_yticks([], [])
        self.plot_clust_gui()
        # return ClustDist, CoreDist


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
