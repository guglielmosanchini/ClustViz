from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QGridLayout, QGroupBox, QTabWidget, QWidget, \
    QVBoxLayout, QTabBar

from PyQt5.QtGui import QFont
# import qdarkstyle
import sys

from GUI_classes.optics_gui import OPTICS_class
from GUI_classes.dbscan_gui import DBSCAN_class
from GUI_classes.agglomerative_gui import AGGLOMERATIVE_class
from GUI_classes.cure_gui import CURE_class
from GUI_classes.cure_large_gui import LARGE_CURE_class
from GUI_classes.clara_gui import CLARA_class
from GUI_classes.clarans_gui import CLARANS_class
from GUI_classes.birch_gui import BIRCH_class
from GUI_classes.pam_gui import PAM_class
from GUI_classes.denclue_gui import DENCLUE_class
from GUI_classes.chameleon_gui import CHAMELEON_class
from GUI_classes.chameleon2_gui import CHAMELEON2_class

# TODO: button images
# TODO: play/pause button
# TODO: comment everything

import os
os.chdir("./code/")

# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook


class Main_Window(QWidget):
    def __init__(self, parent):
        super(Main_Window, self).__init__(parent)

        # main layout of Initial Tab
        self.layout = QVBoxLayout(self)
        # initialization of all tabs
        self.OPTICS_tab = None
        self.DBSCAN_tab = None
        self.AGGLOMERATIVE_tab = None
        self.CURE_tab = None
        self.LARGE_CURE_tab = None
        self.CLARA_tab = None
        self.CLARANS_tab = None
        self.DENCLUE_tab = None
        self.CHAMELEON_tab = None
        self.CHAMELEON2_tab = None
        self.PAM_tab = None
        self.BIRCH_tab = None
        # current index and current dictionary of open tabs
        self.current_index = 0
        self.open_tab_dict = {}
        # fonts
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(12)

        font_title = QFont()
        font_title.setFamily("Arial")
        font_title.setPointSize(14)
        # tab widget initializer
        self.tabs = QTabWidget()
        self.tabs.setFont(font)
        # allow tab to be closed
        self.tabs.setTabsClosable(True)
        # set what to do when pressing x on a tab
        self.tabs.tabCloseRequested.connect(self.closeTab)
        # initialize Initial Tab
        self.initial_tab = QWidget()
        # layout of central box
        gridlayout = QGridLayout(self.initial_tab)

        self.initial_tab.groupbox_alg = QGroupBox("CHOOSE A CLUSTERING ALGORITHM: ")
        # self.initial_tab.groupbox_alg.setFont(font_title)
        self.initial_tab.groupbox_alg.setFixedSize(600, 400)
        gridlayout.addWidget(self.initial_tab.groupbox_alg)

        gridlayout_alg = QGridLayout(self.initial_tab.groupbox_alg)

        # buttons and functions dictionaries
        self.button_dictionary = {1: "OPTICS", 2: "DBSCAN", 3: "AGGLOMERATIVE", 4: "DENCLUE",
                                  5: "CURE", 6: "LARGE CURE", 7: "PAM", 8: "CLARA", 9: "CLARANS", 10: "BIRCH",
                                  11: "CHAMELEON", 12: "CHAMELEON2"}
        self.swapped_button_dictionary = dict([(value, key) for key, value in self.button_dictionary.items()])

        self.functions = {1: self.open_OPTICS, 2: self.open_DBSCAN, 3: self.open_AGGLOMERATIVE, 4: self.open_DENCLUE,
                          5: self.open_CURE, 6: self.open_LARGE_CURE, 7: self.open_PAM, 8: self.open_CLARA,
                          9: self.open_CLARANS, 10: self.open_BIRCH, 11: self.open_CHAMELEON, 12: self.open_CHAMELEON2}

        # buttons
        self.initial_tab.buttons = []
        k = 0
        h = 0
        for i, (key, value) in enumerate(self.button_dictionary.items()):
            self.initial_tab.buttons.append(QPushButton(self.button_dictionary[key], self))
            self.initial_tab.buttons[-1].clicked.connect(self.functions[key])
            gridlayout_alg.addWidget(self.initial_tab.buttons[-1], h, k)
            if k != 3:
                k += 1
            else:
                h += 1
                k = 0

        # adding the Initial Tab and making it unclosable
        self.tabs.addTab(self.initial_tab, "Initial Tab")
        self.tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)
        # add tha tab widget to the main layout
        self.layout.addWidget(self.tabs)

        self.show()

    def change_button_status(self, number):
        if self.initial_tab.buttons[number].isEnabled():
            self.initial_tab.buttons[number].setEnabled(False)
        else:
            self.initial_tab.buttons[number].setEnabled(True)

    def open_OPTICS(self):
        """Open OPTICS_tab and disable its button in the Initial Tab"""
        self.OPTICS_tab = OPTICS_class()
        self.tabs.addTab(self.OPTICS_tab, "OPTICS")
        self.current_index += 1
        index = self.swapped_button_dictionary["OPTICS"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_DBSCAN(self):
        """Open DBSCAN_tab and disable its button in the Initial Tab"""
        self.DBSCAN_tab = DBSCAN_class()
        self.tabs.addTab(self.DBSCAN_tab, "DBSCAN")
        self.current_index += 1
        index = self.swapped_button_dictionary["DBSCAN"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_CURE(self):
        """Open CURE_tab and disable its button in the Initial Tab"""
        self.CURE_tab = CURE_class()
        self.tabs.addTab(self.CURE_tab, "CURE")
        self.current_index += 1
        index = self.swapped_button_dictionary["CURE"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_LARGE_CURE(self):
        """Open LARGE_CURE_tab and disable its button in the Initial Tab"""
        self.LARGE_CURE_tab = LARGE_CURE_class()
        self.tabs.addTab(self.LARGE_CURE_tab, "LARGE CURE")
        self.current_index += 1
        index = self.swapped_button_dictionary["LARGE CURE"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_AGGLOMERATIVE(self):
        """Open AGGLOMERATIVE_tab and disable its button in the Initial Tab"""
        self.AGGLOMERATIVE_tab = AGGLOMERATIVE_class()
        self.tabs.addTab(self.AGGLOMERATIVE_tab, "AGGLOMERATIVE")
        self.current_index += 1
        index = self.swapped_button_dictionary["AGGLOMERATIVE"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_CLARA(self):
        """Open CLARA_tab and disable its button in the Initial Tab"""
        self.CLARA_tab = CLARA_class()
        self.tabs.addTab(self.CLARA_tab, "CLARA")
        self.current_index += 1
        index = self.swapped_button_dictionary["CLARA"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_CLARANS(self):
        """Open CLARANS_tab and disable its button in the Initial Tab"""
        self.CLARANS_tab = CLARANS_class()
        self.tabs.addTab(self.CLARANS_tab, "CLARANS")
        self.current_index += 1
        index = self.swapped_button_dictionary["CLARANS"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_PAM(self):
        """Open PAM_tab and disable its button in the Initial Tab"""
        self.PAM_tab = PAM_class()
        self.tabs.addTab(self.PAM_tab, "PAM")
        self.current_index += 1
        index = self.swapped_button_dictionary["PAM"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_CHAMELEON(self):
        """Open CHAMELEON_tab and disable its button in the Initial Tab"""
        self.CHAMELEON_tab = CHAMELEON_class()
        self.tabs.addTab(self.CHAMELEON_tab, "CHAMELEON")
        self.current_index += 1
        index = self.swapped_button_dictionary["CHAMELEON"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_CHAMELEON2(self):
        """Open CHAMELEON2_tab and disable its button in the Initial Tab"""
        self.CHAMELEON2_tab = CHAMELEON2_class()
        self.tabs.addTab(self.CHAMELEON2_tab, "CHAMELEON2")
        self.current_index += 1
        index = self.swapped_button_dictionary["CHAMELEON2"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_BIRCH(self):
        """Open BIRCH_tab and disable its button in the Initial Tab"""
        self.BIRCH_tab = BIRCH_class()
        self.tabs.addTab(self.BIRCH_tab, "BIRCH")
        self.current_index += 1
        index = self.swapped_button_dictionary["BIRCH"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def open_DENCLUE(self):
        """Open DENCLUE_tab and disable its button in the Initial Tab"""
        self.DENCLUE_tab = DENCLUE_class()
        self.tabs.addTab(self.DENCLUE_tab, "DENCLUE")
        self.current_index += 1
        index = self.swapped_button_dictionary["DENCLUE"]
        self.open_tab_dict.update({self.current_index: self.button_dictionary[index]})
        self.tabs.setCurrentIndex(self.current_index)
        self.change_button_status(index - 1)

    def closeTab(self, currentIndex):
        """Close the selected tab and reenable its button in the Initial Tab"""
        currentQWidget = self.tabs.widget(currentIndex)
        currentQWidget.deleteLater()
        self.tabs.removeTab(currentIndex)
        self.current_index -= 1
        temp_index = self.open_tab_dict[currentIndex]
        self.initial_tab.buttons[self.swapped_button_dictionary[temp_index] - 1].setEnabled(True)

        # the following serves the purpose of readjusting self.open_tab_dict
        del self.open_tab_dict[currentIndex]

        if currentIndex == len(self.open_tab_dict) + 1:
            return

        if len(self.open_tab_dict) != 0:
            for el in range(currentIndex, len(self.open_tab_dict) + 1):
                self.open_tab_dict.update({el: self.open_tab_dict[el + 1]})

            del self.open_tab_dict[len(self.open_tab_dict)]


class main(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clustering Algorithms Visualization")
        self.setGeometry(50, 50, 1290, 850)

        self.table_widget = Main_Window(self)
        self.setCentralWidget(self.table_widget)

        self.statusBar().showMessage('https://github.com/guglielmosanchini/Clustering')

        self.show()


if __name__ == '__main__':

    # QApplication.setStyle('Fusion')
    app = QApplication(sys.argv)
    # pg.setConfigOption('background', 'w')
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = main()
    win.update()

    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")
