from PyQt5.QtCore import Qt, QCoreApplication

import itertools
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
from collections import Counter

from algorithms.chameleon.graphtools import knn_graph, pre_part_graph, get_cluster, connecting_edges
from algorithms.chameleon.visualization import plot2d_data
from algorithms.chameleon.chameleon import rebuild_labels, merge_score

from GUI_classes.utils_gui import choose_dataset, pause_execution

from GUI_classes.generic_gui import StartingGui, GraphWindow

# TODO: fix everything on mac and try on windows

class CHAMELEON_class(StartingGui):
    def __init__(self):
        super(CHAMELEON_class, self).__init__(name="CHAMELEON", twinx=False, first_plot=True, second_plot=False,
                                              function=self.start_CHAMELEON, extract=False, stretch_plot=False)

    def start_CHAMELEON(self):
        self.ax1.cla()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_clust = int(self.line_edit_n_clust.text())
        self.knn_cham = int(self.line_edit_knn_cham.text())
        self.init_clust_cham = int(self.line_edit_init_clust_cham.text())
        self.alpha_cham = float(self.line_edit_alpha_cham.text())
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

        res, h = self.cluster_gui(pd.DataFrame(self.X), k=self.n_clust, knn=self.knn_cham, m=self.init_clust_cham,
                                  alpha=self.alpha_cham, plot=True)

        #plot2d_data(res)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.button_examples_graph.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)

    def cluster_gui(self, df, k, knn=10, m=30, alpha=2.0, verbose=True, verbose2=True, plot=True):
        if k is None:
            k = 1

        self.log.appendPlainText("Building kNN graph (k = {})...".format(knn))
        graph = knn_graph(df, knn, verbose)

        self.plot2d_graph_gui(graph=graph, )

        graph = pre_part_graph(graph, m, df, verbose, plotting=plot)

        dendr_height = {}
        iterm = tqdm(enumerate(range(m - k)), total=m - k) if verbose else enumerate(range(m - k))

        for i, _ in iterm:

            df, ms, ci = self.merge_best(graph, df, alpha, k, verbose=False, verbose2=verbose2)

            if ms == 0:
                break

            dendr_height[m - (i + 1)] = ms

            if plot:
                plot2d_data(df, ci)

        res = rebuild_labels(df)

        return res, dendr_height

    def merge_best(self, graph, df, a, k, verbose=True, verbose2=True):
        clusters = np.unique(df['cluster'])
        max_score = 0
        ci, cj = -1, -1
        if len(clusters) <= k:
            return False

        for combination in itertools.combinations(clusters, 2):
            i, j = combination
            if i != j:
                if verbose:
                    self.log.appendPlainText("Checking c{} and c{}".format(i, j))
                gi = get_cluster(graph, [i])
                gj = get_cluster(graph, [j])
                edges = connecting_edges((gi, gj), graph)
                if not edges:
                    continue
                ms = merge_score(graph, gi, gj, a)
                if verbose:
                    self.log.appendPlainText("Merge score: {}".format(round(ms, 4)))
                if ms > max_score:
                    if verbose:
                        self.log.appendPlainText("Better than: {}".format(round(max_score, 4)))
                    max_score = ms
                    ci, cj = i, j

        if max_score > 0:

            if verbose2:
                self.log.appendPlainText("Merging c{} and c{}".format(ci, cj))
                self.log.appendPlainText("score: {}".format(max_score))

            df.loc[df['cluster'] == cj, 'cluster'] = ci

            for i, p in enumerate(graph.nodes()):
                if graph.node[p]['cluster'] == cj:
                    graph.node[p]['cluster'] = ci
        else:
            self.log.appendPlainText("No Merging")
            self.log.appendPlainText("score: {}".format(round(max_score, 4)))
            self.log.appendPlainText("early stopping")

        return df, max_score, ci

    def plot2d_graph_gui(self, graph, print_clust=True):
        pos = nx.get_node_attributes(graph, 'pos')
        colors = {0: "seagreen", 1: 'beige', 2: 'yellow', 3: 'grey', 4: 'pink', 5: 'turquoise',
                  6: 'orange', 7: 'purple', 8: 'yellowgreen', 9: 'olive', 10: 'brown',
                  11: 'tan', 12: 'plum', 13: 'rosybrown', 14: 'lightblue', 15: "khaki",
                  16: "gainsboro", 17: "peachpuff", 18: "lime", 19: "peru",
                  20: "dodgerblue", 21: "teal", 22: "royalblue", 23: "tomato",
                  24: "bisque", 25: "palegreen"}

        el = nx.get_node_attributes(graph, 'cluster').values()
        cmc = Counter(el).most_common()
        c = [colors[i % len(colors)] for i in el]

        if print_clust is True:
            self.log.appendPlainText("clusters: {}".format(cmc))

        if len(el) != 0:  # is set
            # print(pos)
            nx.draw(graph, pos, node_color=c, node_size=60, edgecolors="black", ax=self.ax1)
        else:
            nx.draw(graph, pos, node_size=60, edgecolors="black", ax=self.ax1)

