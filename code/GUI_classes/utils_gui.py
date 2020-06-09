from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QStyle, QStyleOptionSlider
from PyQt5.QtCore import QRect, QPoint, Qt
from sklearn.datasets import make_blobs, make_moons, make_circles
import time
from matplotlib.pyplot import Polygon
from scipy.spatial import ConvexHull
from matplotlib import colors
import numpy as np



def choose_dataset(chosen_dataset, n_points):
    X = None

    if chosen_dataset == "blobs":
        X = make_blobs(n_samples=n_points, centers=4, n_features=2, cluster_std=1.5, random_state=42)[0]
    elif chosen_dataset == "moons":
        X = make_moons(n_samples=n_points, noise=0.05, random_state=42)[0]
    elif chosen_dataset == "scatter":
        X = make_blobs(n_samples=n_points, cluster_std=[2.5, 2.5, 2.5], random_state=42)[0]
    elif chosen_dataset == "circle":
        X = make_circles(n_samples=n_points, noise=0, random_state=42)[0]

    return X


def pause_execution(seconds):
    time.sleep(seconds)


def encircle(x, y, ax, **kw):
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = Polygon(p[hull.vertices, :], **kw)
    ax.add_patch(poly)


def convert_colors(dict_colors, alpha=0.5):
    new_dict_colors = {}

    for i, col in enumerate(dict_colors.values()):
        new_dict_colors[i] = tuple(list(colors.to_rgb(col)) + [alpha])

    return new_dict_colors


class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, minimum, maximum, interval=1, single_step=1, orientation=Qt.Horizontal,
                 labels=None, parent=None):
        super(LabeledSlider, self).__init__(parent=parent)

        levels = range(minimum, maximum + 1, interval)
        if labels is not None:
            if not isinstance(labels, (tuple, list)):
                raise Exception("<labels> is a list or tuple.")
            if len(labels) != len(levels):
                raise Exception("Size of <labels> doesn't match levels.")
            self.levels = list(zip(levels, labels))
        else:
            self.levels = list(zip(levels, map(str, levels)))

        if orientation == Qt.Horizontal:
            self.layout = QtWidgets.QVBoxLayout(self)
        elif orientation == Qt.Vertical:
            self.layout = QtWidgets.QHBoxLayout(self)
        else:
            raise Exception("<orientation> wrong.")

        # gives some space to print labels
        self.left_margin = 0
        self.top_margin = 0
        self.right_margin = 0
        self.bottom_margin = 0

        self.layout.setContentsMargins(self.left_margin, self.top_margin,
                                       self.right_margin, self.bottom_margin)

        self.sl = QtWidgets.QSlider(orientation, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        self.sl.setSingleStep(single_step)
        if orientation == Qt.Horizontal:
            self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
            self.sl.setMinimumWidth(10)  # just to make it easier to read
        else:
            self.sl.setTickPosition(QtWidgets.QSlider.TicksLeft)
            self.sl.setMinimumHeight(300)  # just to make it easier to read
        self.sl.setTickInterval(interval)

        self.layout.addWidget(self.sl)

    def paintEvent(self, e):

        super(LabeledSlider, self).paintEvent(e)

        style = self.sl.style()
        painter = QPainter(self)
        st_slider = QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation = self.sl.orientation()

        length = style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
        available = style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

        for v, v_str in self.levels:

            # get the size of the label
            rect = painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            if self.sl.orientation() == Qt.Horizontal:
                # I assume the offset is half the length of slider, therefore
                # + length//2
                x_loc = QStyle.sliderPositionFromValue(self.sl.minimum(),
                                                       self.sl.maximum(), v, available) + length // 2

                # left bound of the text = center - half of text width + L_margin
                left = x_loc - rect.width() // 2 + self.left_margin
                bottom = self.rect().bottom()

                # enlarge margins if clipping
                # if v == self.sl.minimum():
                #     if left <= 0:
                #         self.left_margin = rect.width() // 2 - x_loc
                #     if self.bottom_margin <= rect.height():
                #         self.bottom_margin = rect.height()
                #
                #     self.layout.setContentsMargins(self.left_margin,
                #                                    self.top_margin, self.right_margin,
                #                                    self.bottom_margin)
                #
                # if v == self.sl.maximum() and rect.width() // 2 >= self.right_margin:
                #     self.right_margin = rect.width() // 2
                #     self.layout.setContentsMargins(self.left_margin,
                #                                    self.top_margin, self.right_margin,
                #                                    self.bottom_margin)

            else:
                y_loc = QStyle.sliderPositionFromValue(self.sl.minimum(),
                                                       self.sl.maximum(), v, available, upsideDown=True)

                bottom = y_loc + length // 2 + rect.height() // 2 + self.top_margin - 3
                # there is a 3 px offset that I can't attribute to any metric

                left = self.left_margin - rect.width()
                if left <= 0:
                    self.left_margin = rect.width() + 2
                    self.layout.setContentsMargins(self.left_margin,
                                                   self.top_margin, self.right_margin,
                                                   self.bottom_margin)

            pos = QPoint(left, bottom)
            painter.drawText(pos, v_str)

        return
