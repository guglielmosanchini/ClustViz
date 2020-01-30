from PyQt5.QtCore import Qt, QCoreApplication
import random

from pyclustering.cluster.encoder import type_encoding
from pyclustering.utils import euclidean_distance_square

from GUI_classes.utils_gui import choose_dataset, pause_execution
from GUI_classes.pam_gui import KMedoids_gui

from GUI_classes.generic_gui import StartingGui, GraphWindow


class CHAMELEON_class(StartingGui):
    def __init__(self):
        super(CHAMELEON_class, self).__init__(name="CHAMELEON", twinx=False, first_plot=True, second_plot=False,
                                              function=self.start_CHAMELEON, extract=False, stretch_plot=False)

        self.example_index = 0
        self.button_examples_graph.clicked.connect(lambda: self.openGraphWindow(self.example_index))

    def start_CHAMELEON(self):
        self.ax1.cla()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_points = int(self.line_edit_np.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        self.button_run.setEnabled(False)
        self.checkbox_saveimg.setEnabled(False)
        self.button_delete_pics.setEnabled(False)
        self.button_examples_graph.setEnabled(False)
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

        # clarans_gui(data=self.X, number_clusters=self.n_medoids, numlocal=self.numlocal_clarans,
        #             maxneighbor=self.maxneighbors_clarans, log=self.log, ax=self.ax1,
        #             canvas=self.canvas_up, save_fig=self.save_plots, ind_run=self.ind_run,
        #             delay=self.delay).process(plotting=True)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.button_examples_graph.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)
