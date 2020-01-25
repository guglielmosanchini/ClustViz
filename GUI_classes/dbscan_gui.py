from PyQt5.QtWidgets import QPushButton, QLabel, QComboBox, QGridLayout, QGroupBox, \
    QLineEdit, QPlainTextEdit, QWidget
from PyQt5.QtCore import QCoreApplication, QRect
from PyQt5.QtGui import QDoubleValidator, QIntValidator
import pandas as pd
from collections import OrderedDict
# import qdarkstyle
import random

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from algorithms.dbscan import scan_neigh1_mod

from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt

from GUI_classes.utils_gui import choose_dataset, LabeledSlider, pause_execution


class DBSCAN_class(QWidget):
    def __init__(self):
        super(DBSCAN_class, self).__init__()

        self.setWindowTitle("DBSCAN")
        self.setGeometry(100, 100, 1290, 850)

        # upper plot
        self.canvas_up = FigureCanvas(Figure(figsize=(12, 5)))
        self.ax1 = self.canvas_up.figure.subplots()
        self.ax1.set_xticks([], [])
        self.ax1.set_yticks([], [])
        self.ax1.set_title("DBSCAN procedure")

        # box containing everything
        self.groupbox = QGroupBox(self)
        self.groupbox.setGeometry(QRect(30, 10, 1200, 720))

        # parameters initialization
        self.eps = 2
        self.mp = 3
        self.n_points = 50
        self.X = make_blobs(n_samples=self.n_points, centers=4, n_features=2, cluster_std=1.8, random_state=42)[0]
        self.ClustDict = {}
        self.delay = 0
        self.param_check = True

        # grid where the two pictures, the log and the button box are inserted (row,column)
        gridlayout = QGridLayout(self.groupbox)
        gridlayout.addWidget(self.canvas_up, 0, 1)

        # START BUTTON
        self.button_run = QPushButton("START", self)
        self.button_run.clicked.connect(lambda: self.start_DBSCAN())
        self.button_run.setToolTip("Perform clustering.")

        # SLIDER
        label_slider = QLabel(self)
        label_slider.setText("delay:")
        label_slider.setToolTip("Delay each step of the algorithm for the desidered number of seconds.")

        self.slider = LabeledSlider(minimum=0, maximum=3, interval=1, single_step=0.5)
        self.slider.sl.valueChanged.connect(self.changedValue)
        self.slider.setFixedHeight(50)

        # n_points LABEL
        label_np = QLabel(self)
        label_np.setText("n_points:")
        label_np.setToolTip("Number of points of the dataset. It can lie between 5 and 200.")

        self.line_edit_np = QLineEdit(self)
        self.line_edit_np.resize(30, 40)
        self.line_edit_np.setText(str(self.n_points))

        self.n_points_validator = QIntValidator(5, 200, self)
        self.line_edit_np.setValidator(self.n_points_validator)

        # eps LABEL
        label_eps = QLabel(self)
        label_eps.setText("eps (\u03B5):")
        label_eps.setToolTip("""The maximum distance between two samples for one to be considered as in the neighborhood 
                                of the other. """)

        self.line_edit_eps = QLineEdit(self)
        self.line_edit_eps.resize(30, 40)
        self.line_edit_eps.setText(str(self.eps))

        self.eps_validator = QDoubleValidator(0, 1000, 4, self)
        self.line_edit_eps.setValidator(self.eps_validator)

        # minPTS LABEL
        label_mp = QLabel(self)
        label_mp.setText("minPTS:")
        label_mp.setToolTip("The number of samples in a neighborhood for a point to be considered as a core point.")

        self.line_edit_mp = QLineEdit(self)
        self.line_edit_mp.setGeometry(360, 10, 30, 40)
        self.line_edit_mp.setText(str(self.mp))

        self.mp_validator = QIntValidator(1, 200, self)
        self.line_edit_mp.setValidator(self.mp_validator)

        # dataset LABEL
        label_ds = QLabel(self)
        label_ds.setText("dataset:")
        label_ds.setToolTip("Choose among four sklearn generated datasets to perform clustering.")

        # COMBOBOX of datasets
        self.combobox = QComboBox(self)
        self.combobox.setGeometry(750, 10, 120, 30)
        self.combobox.addItem("blobs")
        self.combobox.addItem("moons")
        self.combobox.addItem("scatter")
        self.combobox.addItem("circle")

        # LOG
        self.log = QPlainTextEdit("DBSCAN LOG")
        self.log.setGeometry(900, 60, 350, 400)
        self.log.setStyleSheet(
            """QPlainTextEdit {background-color: #FFF;
                               color: #000000;
                               font-family: Courier;}""")
        self.log.setFixedHeight(335)
        gridlayout.addWidget(self.log, 1, 1)

        # buttons GROUPBOX
        self.groupbox_buttons = QGroupBox("Parameters")
        self.groupbox_buttons.setFixedSize(200,350)

        gridlayout.addWidget(self.groupbox_buttons, 0, 0)

        gridlayout_but = QGridLayout(self.groupbox_buttons)
        gridlayout_but.addWidget(label_ds, 0, 0)
        gridlayout_but.addWidget(self.combobox, 0, 1)

        gridlayout_but.addWidget(label_np, 1, 0)
        gridlayout_but.addWidget(self.line_edit_np, 1, 1)
        gridlayout_but.addWidget(label_eps, 2, 0)
        gridlayout_but.addWidget(self.line_edit_eps, 2, 1)
        gridlayout_but.addWidget(label_mp, 3, 0)
        gridlayout_but.addWidget(self.line_edit_mp, 3, 1)

        gridlayout_but.addWidget(self.button_run, 4, 0)
        gridlayout_but.addWidget(label_slider, 5, 0)
        gridlayout_but.addWidget(self.slider, 5, 1)


        self.show()

    def changedValue(self):
        size = self.slider.sl.value()
        self.delay = size

    def show_error_message(self, check, msg):
        if check[0] != 2:

            if self.param_check is True:
                self.log.clear()

            self.param_check = False
            self.log.appendPlainText("ERROR")
            self.log.appendPlainText("")
            self.log.appendPlainText(msg)
            self.log.appendPlainText("")

    def verify_input_parameters(self):

        self.param_check = True

        check_n_points = self.n_points_validator.validate(self.line_edit_np.text(), 0)
        check_eps = self.eps_validator.validate(self.line_edit_eps.text(), 0)
        check_mp = self.mp_validator.validate(self.line_edit_mp.text(), 0)

        self.show_error_message(check_n_points,
                                "The parameter n_points must be an integer and lie between {0} and {1}.".format(5,
                                                                                                                200))
        self.show_error_message(check_eps,
                                "The parameter eps must lie between {0} and {1}, and can have a maximum of {2} decimal places.".format(
                                    0, 1000, 4))
        self.show_error_message(check_mp,
                                "The parameter minPTS must be an integer and lie between {0} and {1}.".format(1,
                                                                                                              200))

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
        self.DBSCAN_gui(plotting=True, print_details=True, delay=self.delay)

        self.ax1.cla()
        self.plot_clust_DB_gui()

        self.button_run.setEnabled(True)

    def update_log(self, point=None, msg=None, initial=False, noise=False, change_current=False, change_subcurrent=False):
        """ Take care of the log, updating it with information about the current point being examined,

        :param point: current point being examined by DBSCAN.
        :param msg: message to be displayed.
        :param initial: what to do at the start of the algorithm.

        """
        if initial is True:
            self.log.clear()
            self.log.appendPlainText("DBSCAN LOG")
            self.log.appendPlainText("")

        else:
            if change_current is True:
                self.log.appendPlainText("")
                self.log.appendPlainText("current point: " + str(point))

            if change_subcurrent is True:
                self.log.appendPlainText("")
                self.log.appendPlainText("\tcurrent subpoint: " + str(point))

            if noise is True:
                self.log.appendPlainText("")
                self.log.appendPlainText("\tnoise")

            self.log.appendPlainText("")
            self.log.appendPlainText(msg)

    def point_plot_mod_gui(self, X_dict, point):
        """
        Plots a scatter plot of points, where the point (x,y) is light black and
        surrounded by a red circle of radius eps, where already processed point are plotted
        according to ClustDict and without edgecolor, whereas still-to-process points are green
        with black edgecolor.

        :param X_dict: input dictionary version of self.X.
        :param point: coordinates of the point that is currently inspected.

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
        QCoreApplication.processEvents()

    def plot_clust_DB_gui(self, circle_class=None, noise_circle=False):
        """
        Scatter plot of the data points, colored according to the cluster they belong to; circle_class Plots
        circles around some or all points, with a radius of eps; if Noise_circle is True, circle are also plotted
        around noise points.

        :param circle_class: if True, plots circles around every non-noise point, else plots circles
                             only around points belonging to certain clusters, e.g. circle_class = [1,2] will
                             plot circles around points belonging to clusters 1 and 2.
        :param noise_circle: if True, plots circles around noise points

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
                    self.update_log(point, "\tinitial len(N): " + str(len(N)), change_current=True)
                    # print("len(N): ", len(N))
                # if there are less than minPTS in its neighborhood, classify it as noise
                if len(N) < self.mp:

                    self.ClustDict.update({point: -1})

                    if plotting == True:
                        if delay != 0:
                            pause_execution(self.delay)

                        self.point_plot_mod_gui(X_dict, point)
                        self.update_log(noise=True)
                # else if it is a Core point
                else:
                    # increase current id of cluster
                    clust_id += 1
                    # put it in the cluster dictionary
                    self.ClustDict.update({point: clust_id})

                    if plotting == True:
                        if delay != 0:
                            pause_execution(self.delay)

                        self.point_plot_mod_gui(X_dict, point)
                    # add it to the temporary processed list
                    processed_list = [point]
                    # remove it from the neighborhood N
                    del N[point]
                    # until the neighborhood is empty
                    while len(N) > 0:

                        # take a random point in neighborhood
                        n = random.choice(list(N.keys()))

                        if print_details == True:
                            self.update_log(n, "\t\tupdated len(N): " + str(len(N)), change_subcurrent=True)
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
                                self.update_log(point, "\t\tlen(N_sub): " + str(len(N_2)))
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
                                pause_execution(self.delay)
                            self.point_plot_mod_gui(X_dict, n)
